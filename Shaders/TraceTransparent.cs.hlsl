// © 2022 NVIDIA Corporation

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_ComposedDiff, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedSpec_ViewZ, t, 1, SET_OTHER );

// Outputs
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_Composed, u, 0, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gInOut_Mv, u, 1, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 2, SET_OTHER );

//========================================================================================
// TRACE TRANSPARENT
//========================================================================================

struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Pixel position
    uint2 pixelPos;

    // Initial event is reflection
    bool isReflection;
};

// TODO: think about adding a specialized delta-event denoiser in NRD:
//  Inputs:
//      - Lsum ( delta events gathered across the path )
//      - reflections or transmissions prevail?
//  Principle:
//      - add missing component (reflection or transmission) from neighboring pixels
float4 TraceTransparent( TraceTransparentDesc desc, out GlassMotion glassMotion )
{
    GeometryProps geometryProps = desc.geometryProps;

    TracePayload payload = ( TracePayload )0;
    payload.eta = BRDF::IOR::Air / BRDF::IOR::Glass;
    payload.throughput = 1.0;
    glassMotion = ( GlassMotion )0;

    float coneAngle = GetConeAngleFromRoughness( geometryProps.mip, 0.0 ).y;
    float seed = Sequence::Bayer4x4( desc.pixelPos, gFrameIndex );
    bool isReflection = TraceGlass( geometryProps, payload, desc.isReflection, coneAngle, seed, true, glassMotion );

    // Normalize the net turn by a flat reflection; forward transmission favors opaque motion
    if( !desc.isReflection )
    {
        float NoV = abs( dot( desc.geometryProps.N, desc.geometryProps.V ) );
        glassMotion.reflectionWeight *= saturate( 0.5 * ( 1.0 - dot( desc.geometryProps.V, geometryProps.V ) ) / max( NoV * NoV, 1e-6 ) );
    }

    MaterialProps materialProps = GetMaterialProps( geometryProps );

    float4 Lcached = float4( materialProps.Lemi, 0.0 );
    if( !geometryProps.IsMiss( ) )
    {
        // L1 cache - screen-space reprojection
        float3 prevLdiff, prevLspec;
        float reprojectionWeight = ReprojectIrradiance( false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec );
        Lcached = float4( prevLdiff + prevLspec, reprojectionWeight );

        // L2 cache - SHARC
        bool isSharcAllowed = Rng::Hash::GetFloat( ) > Lcached.w;
        isSharcAllowed &= gSHARC && NRD_MODE < OCCLUSION;

        if( isSharcAllowed )
        {
            SharcParameters sharcParams = GetSharcParameters( );
            float voxelSize = GetSharcVoxelSize( geometryProps.X, sharcParams.hashGridParameters );
            float jitterAmount = USE_SHARC_DITHERING * float( USE_SHARC_DEBUG == 0 );
            float3 jitter = GetSharcJitter( geometryProps.N, voxelSize, jitterAmount );

            float3 sharcRadiance = 0;
            bool isFound = TryGetSharcRadiance( sharcParams, geometryProps, materialProps, jitter, sharcRadiance );
            if( isFound )
                Lcached = float4( sharcRadiance, 1.0 );
        }

        // Cache miss - compute lighting, if not found in caches
        if( Rng::Hash::GetFloat( ) > Lcached.w )
        {
            float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW ) + materialProps.Lemi;
            Lcached.xyz = max( Lcached.xyz, L );
        }
    }

    // Radiance and virtual distance
    return float4( Lcached.xyz * payload.throughput, payload.accumulatedHitDist );
}

float2 GetGlassMotion( GlassMotion glassMotion, float virtualDistance, GeometryProps geometryProps, float2 opaqueMotion )
{
    float2 motion = opaqueMotion;

    if( glassMotion.reflectionWeight > 0.0 )
    {
        float3 Xvirtual = geometryProps.X - geometryProps.V * virtualDistance;
        float2 virtualMotion = GetMotion( Xvirtual, Xvirtual ).xy;

        motion = lerp( motion, virtualMotion, glassMotion.reflectionWeight );
    }

    if( glassMotion.surfaceWeight > 0.0 )
    {
        // Reproject the weighted glass-layer distance
        float surfaceDistance = glassMotion.weightedSurfaceDistance / max( glassMotion.surfaceWeight, 1e-6 );
        float3 Xsurface = geometryProps.X - geometryProps.V * surfaceDistance;
        float3 XsurfacePrev = geometryProps.Xprev - geometryProps.V * surfaceDistance;
        float2 surfaceMotion = GetMotion( Xsurface, XsurfacePrev ).xy;

        motion = lerp( motion, surfaceMotion, glassMotion.surfaceWeight );
    }

    return motion;
}

// Keep the smaller parallel-edge contrast in each direction to reject isolated path noise
float GetQuadDetail( float luminance )
{
    float dx = luminance - QuadReadAcrossX( luminance );
    float dy = luminance - QuadReadAcrossY( luminance );
    float2 gradient = abs( float2( dx, dy ) );

    gradient = min( gradient, QuadReadAcrossDiagonal( gradient ) );

    return gradient.x + gradient.y;
}

//========================================================================================
// MAIN
//========================================================================================

[numthreads( 16, 16, 1 )]
void main( int2 pixelPos : SV_DispatchThreadID )
{
    if( any( pixelPos >= int2( gRectSize ) ) )
        return;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Trace to the first layer of glass...
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    float4 motion = gInOut_Mv[ pixelPos ];
    float viewZAndTaaMask = motion.w;
    float viewZ = abs( viewZAndTaaMask ) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float viewZafterPsr = abs( gIn_ComposedSpec_ViewZ[ pixelPos ].w ) / FP16_VIEWZ_SCALE;

    float eps = viewZ * 0.001;
    bool hasPsr = gPSR && PT_PSR_BOUNCES_NUM != 0 && !( abs( viewZafterPsr - viewZ ) < eps );
    uint flags = hasPsr ? GEOMETRY_ALL : FLAG_TRANSPARENT;

    float rayDirectionZ = Geometry::RotateVector( gWorldToView, cameraRayDirection ).z;
    float tmax = ( viewZ + eps ) / abs( rayDirectionZ ); // DoF friendly

    GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmax, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, flags, 0 );

    float3 Lsum = 1.0;

    TraceTransparentDesc desc = ( TraceTransparentDesc )0;
    desc.geometryProps = geometryPropsT;
    desc.pixelPos = pixelPos;

    // ...through PSRs
    [branch]
    if( !geometryPropsT.Has( FLAG_TRANSPARENT ) && hasPsr && gOnScreen == SHOW_FINAL )
    {
        float3x3 mirrorMatrix = Geometry::GetMirrorMatrix( 0 );
        float3 X0 = geometryPropsT.X;
        float3 V0 = geometryPropsT.V;
        float accumulatedHitDist = 0.0;
        float accumulatedCurvature = 0.0;

        [loop]
        for( uint bounce = 0; bounce < PT_PSR_BOUNCES_NUM && !geometryPropsT.IsMiss( ); bounce++ )
        {
            MaterialProps materialProps = GetMaterialProps( geometryPropsT );
            if( !IsDelta( materialProps ) )
                break;

            accumulatedCurvature += materialProps.curvature;
            mirrorMatrix = mul( Geometry::GetMirrorMatrix( materialProps.N ), mirrorMatrix );

            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );
            float NoV = abs( dot( materialProps.N, geometryPropsT.V ) );
            Lsum *= BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );

            float3 ray = reflect( -geometryPropsT.V, materialProps.N );
            float2 mipAndCone = GetConeAngleFromRoughness( geometryPropsT.mip, materialProps.roughness );
            geometryPropsT = CastRay( GetXoffset( geometryPropsT.X, geometryPropsT.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, PT_RAY_FLAGS );
            accumulatedHitDist += ApplyThinLensEquation( geometryPropsT.hitT, accumulatedCurvature );

            if( geometryPropsT.Has( FLAG_TRANSPARENT ) && gOnScreen == SHOW_FINAL )
            {
                // Trace from the physical hit
                desc.geometryProps = geometryPropsT;

                // Patch the working hit for reprojection
                float3 displacement = Geometry::RotateVectorInverse( mirrorMatrix, geometryPropsT.Xprev - geometryPropsT.X );
                geometryPropsT.X = X0 - V0 * accumulatedHitDist;
                geometryPropsT.Xprev = geometryPropsT.X + displacement;
                geometryPropsT.V = Geometry::RotateVectorInverse( mirrorMatrix, geometryPropsT.V );
                geometryPropsT.N = Geometry::RotateVectorInverse( mirrorMatrix, geometryPropsT.N );

                break;
            }
        }
    }

    float glassViewZ = 1e6;
    float3 reflectionMv = 0;
    float3 transmissionMv = 0;

    if( geometryPropsT.Has( FLAG_TRANSPARENT ) && gOnScreen == SHOW_FINAL )
    {
        // Patch guides for RR
        [branch]
        if( gRR )
            gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( geometryPropsT.N, 0.0, 0 );

        // Split primary reflection and transmission to reduce noise
        desc.isReflection = true;
        GlassMotion reflectionMotion;
        float4 reflection = TraceTransparent( desc, reflectionMotion );
        reflectionMv.z = Color::Luminance( reflection.xyz );
        reflectionMv.xy = GetGlassMotion( reflectionMotion, reflection.w, geometryPropsT, motion.xy );
        reflectionMv.xy *= reflectionMv.z;

        desc.isReflection = false;
        GlassMotion transmissionMotion;
        float4 transmission = TraceTransparent( desc, transmissionMotion );
        transmissionMv.z = Color::Luminance( transmission.xyz );
        transmissionMv.xy = GetGlassMotion( transmissionMotion, transmission.w, geometryPropsT, motion.xy );
        transmissionMv.xy *= transmissionMv.z;

        Lsum *= reflection.xyz + transmission.xyz;

        // TAA needs the nearest visible glass or PSR viewZ, with the negative mask preserved
        glassViewZ = abs( Geometry::AffineTransform( gWorldToView, geometryPropsT.X ).z );
        motion.w = -min( abs( viewZAndTaaMask ), glassViewZ * FP16_VIEWZ_SCALE );
    }
    else
    {
        // Composed without glass
        float3 diff = gIn_ComposedDiff[ pixelPos ];
        float3 spec = gIn_ComposedSpec_ViewZ[ pixelPos ].xyz;

        Lsum = diff + spec;
    }

    // Average compatible motion in 2x2 quad
    [branch]
    if( WaveActiveAnyTrue( geometryPropsT.Has( FLAG_TRANSPARENT ) ) && all( ( pixelPos | 1 ) < int2( gRectSize ) ) )
    {
        float pixelSize2X = gUnproject * lerp( glassViewZ, 1.0, abs( gOrthoMode ) ) * 2.0; // allow extra viewZ variation for curved or obliquely viewed glass
        float3 reflectionSum = 0;
        float3 transmissionSum = 0;
        float sampleCount = 0;

        [unroll]
        for( uint lane = 0; lane < 4; lane++ )
        {
            float donorViewZ = QuadReadLaneAt( glassViewZ, lane );
            float3 reflectionDonor = QuadReadLaneAt( reflectionMv, lane );
            float3 transmissionDonor = QuadReadLaneAt( transmissionMv, lane );

            if( abs( donorViewZ - glassViewZ ) < pixelSize2X )
            {
                reflectionSum += reflectionDonor;
                transmissionSum += transmissionDonor;
                sampleCount += 1.0;
            }
        }

        float2 detail;
        detail.x = GetQuadDetail( reflectionMv.z );
        detail.y = GetQuadDetail( transmissionMv.z );

        if( geometryPropsT.Has( FLAG_TRANSPARENT ) && sampleCount > 1.0 )
        {
            reflectionMv = reflectionSum;
            transmissionMv = transmissionSum;

            // Favor reflected features over smooth transmission with full quad support
            if( sampleCount == 4.0 && reflectionMv.z != 0.0 && transmissionMv.z != 0.0 )
            {
                float totalEnergy = reflectionMv.z + transmissionMv.z;
                float detailWeight = saturate( detail.x / max( detail.y, 1e-6 ) );
                float reflectionScale = max( reflectionMv.z, detailWeight * totalEnergy ) / reflectionMv.z;
                float transmissionScale = min( transmissionMv.z, ( 1.0 - detailWeight ) * totalEnergy ) / transmissionMv.z;

                reflectionMv *= reflectionScale;
                transmissionMv *= transmissionScale;
            }
        }
    }

    float totalEnergy = reflectionMv.z + transmissionMv.z;
    if( totalEnergy != 0.0 )
    {
        // Prefer initial reflection slightly ( +6.25% ) when both motion guides contribute
        float reflectionFraction = reflectionMv.z / totalEnergy;
        float transmissionFraction = transmissionMv.z / totalEnergy;
        reflectionMv *= 1.0 + 0.25 * transmissionFraction;
        transmissionMv *= 1.0 - 0.25 * reflectionFraction;

        // Final mix
        motion.xy = ( reflectionMv.xy + transmissionMv.xy ) / totalEnergy;
    }

    // Opaque denoising has consumed dZ; pass the glass mask to TAA and SR/RR instead.
    motion.z = ( geometryPropsT.Has( FLAG_TRANSPARENT ) && gOnScreen == SHOW_FINAL ) ? 1.0 : 0.0;
    gInOut_Mv[ pixelPos ] = motion;

    // Output
    gOut_Composed[ pixelPos ] = Lsum * gExposure; // apply exposure
}
