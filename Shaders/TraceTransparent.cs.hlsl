// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

#define SHARC_QUERY 1
#include "SharcCommon.h"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_ComposedDiff, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedSpec_ViewZ, t, 1, 1 );

// Outputs
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_Composed, u, 0, 1 );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gInOut_Mv, u, 1, 1 );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 2, 1 );

//========================================================================================
// TRACE TRANSPARENT
//========================================================================================

struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Pixel position
    uint2 pixelPos;

    // Is reflection or refraction in first segment?
    bool isReflection;
};

// TODO: think about adding a specialized delta-event denoiser in NRD:
//  Inputs:
//      - Lsum ( delta events gathered across the path )
//      - reflections or refractions prevail?
//  Principle:
//      - add missing component (reflection or refraction) from neighboring pixels
float3 TraceTransparent( TraceTransparentDesc desc )
{
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;

    GeometryProps geometryProps = desc.geometryProps;
    float pathThroughput = 1.0;
    bool isReflection = desc.isReflection;
    float bayer = Sequence::Bayer4x4( desc.pixelPos, gFrameIndex );

    [loop]
    for( uint bounce = 1; bounce <= PT_DELTA_BOUNCES_NUM; bounce++ )
    {
        // Reflection or refraction?
        float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
        float F = BRDF::FresnelTerm_Dielectric( eta, NoV );

        if( bounce == 1 )
            pathThroughput *= isReflection ? F : 1.0 - F;
        else
        {
            float rnd = frac( bayer + Sequence::Halton( bounce, 3 ) ); // "Halton( bounce, 2 )" works worse than others

            // Bonus dithering to break banding // TODO: needed for "transparent machines"
            /*
            float ditherMask = Sequence::Bayer4x4( desc.pixelPos, bounce ); // or
            //float ditherMask = Rng::Hash::GetFloat( ); // or
            rnd = saturate( rnd + ( ditherMask - 0.5 ) * 0.04 );
            */

            [flatten]
            if( gDenoiserType == DENOISER_REFERENCE || gRR )
                rnd = Rng::Hash::GetFloat( );
            else
                F = clamp( F, PT_GLASS_MIN_F, 1.0 - PT_GLASS_MIN_F ); // TODO: needed?

            isReflection = rnd < F; // TODO: if "F" is clamped, "pathThroughput" should be adjusted too
        }

        // Trace
        uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;

        float3 Xoffset, ray;
        eta = GetDeltaEventRay( geometryProps, isReflection, eta, Xoffset, ray );

        geometryProps = CastRay( Xoffset, ray, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, 0.0 ), gWorldTlas, flags, 0 );

        // TODO: ideally each "medium" should have "eta" and "extinction" parameters in "TraceTransparentDesc" and "TraceOpaqueDesc"
        bool isAir = eta < 1.0;
        float extinction = isAir ? 0.0 : 1.0; // TODO: tint color? // TODO: 0.01 for "transparent machines"
        if( !geometryProps.IsSky() ) // TODO: fix for non-convex geometry
            pathThroughput *= exp( -extinction * geometryProps.hitT );

        // Is opaque hit found?
        if( !geometryProps.Has( FLAG_TRANSPARENT ) ) // TODO: stop if pathThroughput is low
            break;
    }

    // This is always non-transparent
    MaterialProps materialProps = GetMaterialProps( geometryProps );

    float4 Lcached = float4( materialProps.Lemi, 0.0 );
    if( !geometryProps.IsSky( ) )
    {
        // L1 cache - reproject previous frame, carefully treating specular
        float3 prevLdiff, prevLspec;
        float reprojectionWeight = ReprojectIrradiance( false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec );
        Lcached = float4( prevLdiff + prevLspec, reprojectionWeight );

        // L2 cache - SHARC
        HashGridParameters hashGridParams;
        hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
        hashGridParams.sceneScale = SHARC_SCENE_SCALE;
        hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

        float3 Xglobal = GetGlobalPos( geometryProps.X );
        uint level = HashGridGetLevel( Xglobal, hashGridParams );
        float voxelSize = HashGridGetVoxelSize( level, hashGridParams );
        float smc = GetSpecMagicCurve( materialProps.roughness );

        float3x3 mBasis = Geometry::GetBasis( geometryProps.N );
        float2 rndScaled = ( Rng::Hash::GetFloat2( ) - 0.5 ) * voxelSize * USE_SHARC_DITHERING;
        Xglobal += mBasis[ 0 ] * rndScaled.x + mBasis[ 1 ] * rndScaled.y;

        SharcHitData sharcHitData;
        sharcHitData.positionWorld = Xglobal;
        sharcHitData.normalWorld = geometryProps.N;
        sharcHitData.emissive = materialProps.Lemi;

        HashMapData hashMapData;
        hashMapData.capacity = SHARC_CAPACITY;
        hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

        SharcParameters sharcParams;
        sharcParams.gridParameters = hashGridParams;
        sharcParams.hashMapData = hashMapData;
        sharcParams.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;
        sharcParams.voxelDataBuffer = gInOut_SharcVoxelDataBuffer;
        sharcParams.voxelDataBufferPrev = gInOut_SharcVoxelDataBufferPrev;

        bool isSharcAllowed = gSHARC && NRD_MODE < OCCLUSION; // trivial
        isSharcAllowed &= Rng::Hash::GetFloat( ) > Lcached.w; // probabilistically estimate the need
        isSharcAllowed &= geometryProps.hitT > voxelSize; // voxel angular size is acceptable // TODO: can be skipped to get flat ambient in some cases

        float3 sharcRadiance;
        if (isSharcAllowed && SharcGetCachedRadiance( sharcParams, sharcHitData, sharcRadiance, false))
            Lcached = float4( sharcRadiance, 1.0 );

        // Cache miss - compute lighting, if not found in caches
        if( Rng::Hash::GetFloat( ) > Lcached.w )
        {
            float3 L = GetShadowedLighting( geometryProps, materialProps );
            Lcached.xyz = max( Lcached.xyz, L );
        }
    }

    // Output
    return Lcached.xyz * pathThroughput;
}

//========================================================================================
// MAIN
//========================================================================================

[numthreads( 16, 16, 1 )]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Composed without glass
    float3 diff = gIn_ComposedDiff[ pixelPos ];
    float3 spec = gIn_ComposedSpec_ViewZ[ pixelPos ].xyz;
    float3 Lsum = diff + spec * float( gOnScreen == SHOW_FINAL );

    // Primary ray for transparent geometry only
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    float viewZAndTaaMask = gInOut_Mv[ pixelPos ].w;
    float viewZ = Math::Sign( gNearZ ) * abs( viewZAndTaaMask ) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );
    float tmin0 = gOrthoMode == 0 ? length( Xv ) : abs( Xv.z );

    GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, FLAG_TRANSPARENT, 0 );

    // Trace delta events
    if( !geometryPropsT.IsSky( ) && geometryPropsT.hitT < tmin0 && gOnScreen == SHOW_FINAL )
    {
        // Append "glass" mask to "hair" mask
        viewZAndTaaMask = viewZAndTaaMask < 0.0 ? viewZAndTaaMask : -viewZAndTaaMask;

        // Patch motion vectors replacing MV for the background with MV for the closest glass layer.
        // IMPORTANT: surface-based motion can be used only if the object is curved.
        // TODO: let's use the simplest heuristic for now, but better switch to some "smart" interpolation between
        // MVs for the primary opaque surface hit and the primary glass surface hit.
        float3 mvT = GetMotion( geometryPropsT.X, geometryPropsT.Xprev );
        gInOut_Mv[ pixelPos ] = float4( mvT, viewZAndTaaMask );

        // Patch guides for RR
        [branch]
        if( gRR )
            gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( geometryPropsT.N, 0.0, 0 );

        // Trace transparent stuff
        TraceTransparentDesc desc = ( TraceTransparentDesc )0;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;

        // IMPORTANT: use 1 reflection path and 1 refraction path at the primary glass hit to significantly reduce noise
        // TODO: use probabilistic split at the primary glass hit when denoising becomes available
        desc.isReflection = true;
        float3 reflection = TraceTransparent( desc );
        Lsum = reflection;

        desc.isReflection = false;
        float3 refraction = TraceTransparent( desc );
        Lsum += refraction;
    }

    // Apply exposure
    Lsum = ApplyExposure( Lsum );

    // Output
    gOut_Composed[ pixelPos ] = Lsum;
}
