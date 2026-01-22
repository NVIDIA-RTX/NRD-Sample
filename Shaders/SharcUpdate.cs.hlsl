// © 2024 NVIDIA Corporation

#define SHARC_UPDATE 1

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

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
    float2 jitter = mode == PREV ? ( gJitterPrev * gRectSizePrev ) : ( gJitter * gRectSize ); // it works well for 1 path per pixel, better use "Rng::Hash::GetFloat2( )" in other cases
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
    uint DELTA_BOUNCES_NUM = mode == FULL ? PT_DELTA_BOUNCES_NUM : 1;

    [loop]
    for( uint bounce = 1; bounce <= DELTA_BOUNCES_NUM; bounce++ )
    {
        uint flags = bounce == DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;

        geometryProps = CastRay( Xoffset, ray, 0.0, INF, mip, gWorldTlas, flags, 0 );
        MaterialProps materialProps = GetMaterialProps( geometryProps );

        bool isGlass = geometryProps.Has( FLAG_TRANSPARENT );
        bool isDelta = IsDelta( materialProps ); // TODO: verify corner cases

        if( !( isGlass || isDelta ) || geometryProps.IsMiss( ) )
            break;

        // Reflection or refraction?
        float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
        float F = BRDF::FresnelTerm_Dielectric( eta, NoV );
        float rnd = Rng::Hash::GetFloat( );
        bool isReflection = isDelta ? true : rnd < F;

        eta = GetDeltaEventRay( geometryProps, isReflection, eta, Xoffset, ray );
    }

    // Miss?
    if( geometryProps.IsMiss( ) )
        return float4( 0.0, 0.0, 0.0, FP16_MAX );

    // SHARC state
    HashGridParameters hashGridParams;
    hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
    hashGridParams.sceneScale = SHARC_SCENE_SCALE;
    hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

    HashMapData hashMapData;
    hashMapData.capacity = SHARC_CAPACITY;
    hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

    SharcParameters sharcParams;
    sharcParams.gridParameters = hashGridParams;
    sharcParams.hashMapData = hashMapData;
    sharcParams.radianceScale = SHARC_RADIANCE_SCALE;
    sharcParams.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;
    sharcParams.accumulationBuffer = gInOut_SharcAccumulated;
    sharcParams.resolvedBuffer = gInOut_SharcResolved;

    SharcState sharcState;
    SharcInit( sharcState );

    // Primary
    MaterialProps materialProps = GetMaterialProps( geometryProps );

    float3 materialDemodulation0 = GetMaterialDemodulation( geometryProps, materialProps );
    float pathThroughput = 1.0; // materials maybe demodulated via "1.0 / Color::Luminance( materialDemodulation0 )", but it will require adjustments in other places...

    float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW );

    float4 gradientData; // IMPORTANT: direct emission must be excluded for history confidence calculations, sun direct lighting is not needed too because it doesn't go through NRD...
    gradientData.x = 0; // use "Color::Luminance( L ) * pathThroughput" for primary sun lighting
    gradientData.yz = Packing::EncodeUnitVector( Geometry::RotateVector( gWorldToViewPrev, geometryProps.N ) );
    gradientData.w = Geometry::AffineTransform( gWorldToViewPrev, geometryProps.X ).z * FP16_VIEWZ_SCALE;

    if( mode != PREV )
    {
        SharcHitData sharcHitData;
        sharcHitData.positionWorld = GetGlobalPos( geometryProps.X );
        sharcHitData.materialDemodulation = materialDemodulation0;
        sharcHitData.normalWorld = geometryProps.N;
        sharcHitData.emissive = materialProps.Lemi;

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

        float3 throughput = 1.0;
        {
            // Estimate diffuse probability
            float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );
            diffuseProbability = float( diffuseProbability != 0.0 ) * clamp( diffuseProbability, 0.25, 0.75 );

            // Diffuse or specular?
            bool isDiffuse = Rng::Hash::GetFloat( ) < diffuseProbability;
            throughput /= isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability );

            // Importance sampling
            uint sampleMaxNum = 0;
            if( bounceNum == BOUNCES_NUM && gDisableShadowsAndEnableImportanceSampling )
                sampleMaxNum = PT_IMPORTANCE_SAMPLES_NUM * ( isDiffuse ? 1.0 : GetSpecMagicCurve( materialProps.roughness ) );
            sampleMaxNum = max( sampleMaxNum, 1 );

            float3 ray = GenerateRayAndUpdateThroughput( geometryProps, materialProps, throughput, sampleMaxNum, isDiffuse, Rng::Hash::GetFloat2( ), 0 );

            pathThroughput *= Color::Luminance( throughput );

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
            geometryProps = CastRay( GetXoffset( geometryProps.X, geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, 0 );
            materialProps = GetMaterialProps( geometryProps );
        }

        // Update
        if( mode != PREV )
            SharcSetThroughput( sharcState, throughput );

        if( geometryProps.IsMiss( ) )
        {
            if( mode != PREV )
                SharcUpdateMiss( sharcParams, sharcState, materialProps.Lemi );

            bounceNum = 1; // aka "break"
        }
        else
        {
            float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW );

            gradientData.x += Color::Luminance( L + materialProps.Lemi ) * pathThroughput;

            if( mode != PREV )
            {
                SharcHitData sharcHitData;
                sharcHitData.positionWorld = GetGlobalPos( geometryProps.X );
                sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
                sharcHitData.normalWorld = geometryProps.N;
                sharcHitData.emissive = materialProps.Lemi;

                // This introduces discrepancies with "PREV" if output radiance includes more than 1 bounce lighting
                bool continueTracing = SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, Rng::Hash::GetFloat( ) );
                if( !continueTracing )
                    bounceNum = 1; // aka "break"
            }
        }

        bounceNum--;
    }

    return gradientData;
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Current gradient data
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float4 gradientCurr = Trace( pixelPos, CURR );
    gOut_CurrGradient[ pixelPos ] = gradientCurr; // will be "gIn_PrevGradient" in the next frame

    // ( Optional ) "Reach" potentially new areas visible through glass
    Trace( pixelPos, FULL );

    // Previous gradient data
    Rng::Hash::Initialize( pixelPos, gFrameIndex - 1 );

    float4 prevGradient = Trace( pixelPos, PREV ); // no SHARC update
    float4 prevGradientStored = gIn_PrevGradient[ pixelPos ];

    // Irradiance gradient: it includes materials, i.e can be normalized to the blurred final HDR image
    float gradient = abs( prevGradient.x - prevGradientStored.x ); // hitT gradient? no...

    // Apply simple occlusion factor to eliminate false-positives left by dynamic objects ( 1st order disocclusions are handled by NRD itself )
    float zOcclusion = abs( prevGradient.w - prevGradientStored.w ) / max( prevGradient.w,  prevGradientStored.w );
    gradient *= Math::SmoothStep( 0.25, 0.05, zOcclusion );

    // Which "small g-buffer" to use "stored" or "traced"? "Stored" represents the real state of the previous frame
    gOut_Gradient[ pixelPos ] = float4( gradient, prevGradientStored.yzw );
}
