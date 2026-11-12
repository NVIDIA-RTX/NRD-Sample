// © 2024 NVIDIA Corporation

#define SHARC_UPDATE 1

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Input
NRI_RESOURCE( Texture2D<float4>, gIn_PrevGradient, t, 0, SET_OTHER );

// Outputs
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_CurrGradient, u, 0, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Gradient, u, 1, SET_OTHER );

// Compile-time constants
#define CURR 0
#define PREV 1
#define FULL 2

float4 Trace( uint2 pixelPos, compiletime int mode )
{
    // Sample position
    /*
    // Camera jitter works well for 1 path/pixel in SHARC update, but can leave small coverage gaps in rare cases.
    // Use independent random jitter to improve coverage, even when camera jitter is disabled.
    float2 jitter = mode == PREV ? ( gJitterPrev * gRectSizePrev ) : ( gJitter * gRectSize );
    */

    float2 jitter = Rng::Hash::GetFloat2( ) - 0.5;
    float2 sampleUv = ( pixelPos + 0.5 + jitter ) * gInvSharcRenderSize;

    // Primary ray
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );

    float3 Xoffset, ray;
    if( mode == PREV )
    {
        Xoffset = Geometry::AffineTransform( gViewToWorldPrev, Xv );
        ray = gOrthoMode == 0.0 ? normalize( Geometry::RotateVector( gViewToWorldPrev, Xv ) ) : -gViewDirection.xyz; // TODO: add "gViewDirectionPrev"
    }
    else
    {
        Xoffset = Geometry::AffineTransform( gViewToWorld, Xv );
        ray = gOrthoMode == 0.0 ? normalize( Geometry::RotateVector( gViewToWorld, Xv ) ) : -gViewDirection.xyz;
    }

    // Jump through delta events // TODO: bad for history confidence
    GeometryProps geometryProps;

    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;
    float2 mip = GetConeAngleFromAngularRadius( 0.0, gTanPixelAngularRadius * SHARC_DOWNSCALE );
    uint DELTA_BOUNCES_NUM = mode == FULL ? PT_GLASS_DIRECT_BOUNCES_NUM : 1;

    [loop]
    for( uint bounce = 1; bounce <= DELTA_BOUNCES_NUM; bounce++ )
    {
        uint flags = bounce == DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;

        geometryProps = CastRay( Xoffset, ray, 0.0, INF, mip, gWorldTlas, flags, 0 );
        MaterialProps materialProps = GetMaterialProps( geometryProps );

        bool isGlass = geometryProps.Has( FLAG_TRANSPARENT );
        bool isDelta = IsDelta( materialProps );

        if( !( isGlass || isDelta ) || geometryProps.IsMiss( ) )
            break;

        // True mirrors always reflect, glass samples reflection or refraction
        bool isReflection = true;
        if( isGlass )
        {
            float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
            float F = BRDF::FresnelTerm_Dielectric( eta, NoV );
            float rnd = Rng::Hash::GetFloat( );

            // Since we don't know the energy of the reflection, both reflection and refraction paths need to get coverage
            bool isTotalInternalReflection = eta * eta * ( 1.0 - NoV * NoV ) >= 1.0;
            isReflection = isTotalInternalReflection || rnd < clamp( F, 0.25, 0.75 );
        }

        eta = GetDeltaEventRay( geometryProps, isReflection, eta, Xoffset, ray );
    }

    // Miss?
    if( geometryProps.IsMiss( ) )
        return float4( 0.0, 0.0, 0.0, FP16_MAX );

    // SHARC state
    SharcParameters sharcParams = GetSharcParameters( );

    SharcState sharcState;
    SharcInit( sharcState );

    // Primary
    MaterialProps materialProps = GetMaterialProps( geometryProps );

    float3 materialDemodulation0 = GetMaterialFactor( geometryProps, materialProps );
    float pathThroughput = 1.0; // materials maybe demodulated via "1.0 / Color::Luminance( materialDemodulation0 )", but it will require adjustments in other places...

    float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW );

    float4 gradientData; // IMPORTANT: direct emission must be excluded for history confidence calculations
    gradientData.x = Color::Luminance( L ) * pathThroughput; // direct sun lighting is not noisy, but nice to have here to invoke more history rejections if the sun is moving
    gradientData.yz = Packing::EncodeUnitVector( Geometry::RotateVector( gWorldToViewPrev, geometryProps.N ) );
    gradientData.w = Geometry::AffineTransform( gWorldToViewPrev, geometryProps.X ).z * FP16_VIEWZ_SCALE;

    if( mode != PREV )
    {
        SharcHitData sharcHitData = GetSharcHitData( geometryProps, materialProps, materialDemodulation0 );

        SharcSetThroughput( sharcState, 1.0 );
        SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, 1.0 ); // 0 bounce => no cache resampling => always returns "true"
    }

    // Secondary
    uint BOUNCES_NUM = mode == FULL ? SHARC_PROPAGATION_DEPTH : 1;
    uint bounceNum = BOUNCES_NUM;

    while( bounceNum ) // why not "for"? because DXC produced invalid code for "for" after introduction the compile-time constant "mode"
    {
        //=============================================================================================================================================================
        // Origin point
        //=============================================================================================================================================================

        bool isStaticAtOrigin = false;
        {
            // Estimate diffuse probability
            float diffuseProbability = GetClampedDiffuseProbability( geometryProps, materialProps );

            // Diffuse or specular?
            bool isDiffuse = Rng::Hash::GetFloat( ) < diffuseProbability;
            float3 throughput = 1.0 / ( isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability ) );

            // Importance sampling
            uint sampleMaxNum = GetImportanceSampleNum( materialProps.roughness, isDiffuse, bounceNum == BOUNCES_NUM );

            float3 ray = GenerateRayAndUpdateThroughput( geometryProps, materialProps, throughput, sampleMaxNum, isDiffuse, 0, 0, 0 );

            pathThroughput *= Color::Luminance( throughput );

            if( mode != PREV )
                SharcSetThroughput( sharcState, throughput );

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            isStaticAtOrigin = geometryProps.Has( FLAG_STATIC );

            // TODO: account for glass transmittance and absorption ( extinction ) during cache propagation when their impact is significant
            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
            geometryProps = CastRay( GetXoffset( geometryProps.X, geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );

            materialProps = GetMaterialProps( geometryProps );
        }

        // Update
        if( geometryProps.IsMiss( ) )
        {
            if( mode != PREV )
                SharcUpdateMiss( sharcParams, sharcState, materialProps.Lemi );

            bounceNum = 1; // aka "break"
        }
        else
        {
            // Lighting at hit
            float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW ); // no emission here

            { // Lighting for gradients
                float l = Color::Luminance( L + materialProps.Lemi ); // lighting + self emission

                // Lighting is not clean enough to be used for detecting changes in shadows. Use "hitT" instead to estimate changes
                // in shadows casted by dynamic objects. Convert "shadow" to "lighting" to avoid processing two signals separately.
                bool isDynamicAtHit = !geometryProps.Has( FLAG_STATIC );
                float ao = Math::Sqrt01( geometryProps.hitT / gHitDistSettings.x );
                float indirectShadowing = 1.0 - ao; // any estimation can be used here
                indirectShadowing *= float( isStaticAtOrigin && isDynamicAtHit ); // this is important
                l += indirectShadowing * 25.0 / gExposure; // make calculations exposure-independent

                gradientData.x += l * pathThroughput;
            }

            if( mode != PREV )
            {
                SharcHitData sharcHitData = GetSharcHitData( geometryProps, materialProps );

                // This introduces discrepancies with "PREV" if output radiance includes more than 1 bounce lighting
                bool continueTracing = SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, Rng::Hash::GetFloat( ) );
                if( !continueTracing )
                    bounceNum = 1; // aka "break"
            }
        }

        bounceNum--;
    }

    // Apply exposure to better fit into FP16
    gradientData.x *= gExposure;

    // Clamp to ~white point ( 1 is OK too )
    gradientData.x = min( gradientData.x, 2.0 );

    return gradientData;
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadID )
{
    // Current gradient data
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float16_t4 gradientCurr = ( float16_t4 )Trace( pixelPos, CURR );
    gOut_CurrGradient[ pixelPos ] = gradientCurr; // will be "gIn_PrevGradient" in the next frame

    // ( Optional ) "Reach" potentially new areas visible through glass
    Trace( pixelPos, FULL );

    // Previous gradient data
    // It's important to use FP16 for calculations to avoid imprecision issues, because "stored" data comes from an FP16 texture
    Rng::Hash::Initialize( pixelPos, gFrameIndex - 1 );

    float16_t4 prevGradient = ( float16_t4 )Trace( pixelPos, PREV ); // no SHARC update
    float16_t4 prevGradientStored = ( float16_t4 )gIn_PrevGradient[ pixelPos ];

    // Irradiance gradient: it includes materials, i.e can be normalized to the blurred final HDR image
    float16_t gradient = abs( prevGradient.x - prevGradientStored.x );

    // Apply simple occlusion factor to eliminate false-positives left by dynamic objects ( 1st order disocclusions are handled by NRD itself )
    float zOcclusion = abs( prevGradient.w - prevGradientStored.w ) / max( prevGradient.w,  prevGradientStored.w );
    gradient *= ( float16_t )Math::SmoothStep( 0.25, 0.05, zOcclusion );

    // Which "small g-buffer" to use "stored" or "traced"? "Stored" represents the real state of the previous frame
    gOut_Gradient[ pixelPos ] = float16_t4( gradient, prevGradientStored.yzw );
}
