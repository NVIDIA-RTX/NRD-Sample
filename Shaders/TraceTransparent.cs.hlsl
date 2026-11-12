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
    GeometryProps geometryProps = desc.geometryProps;

    TracePayload payload = ( TracePayload )0;
    payload.eta = BRDF::IOR::Air / BRDF::IOR::Glass;
    payload.throughput = 1.0;

    float coneAngle = GetConeAngleFromRoughness( geometryProps.mip, 0.0 ).y;
    float bayer = Sequence::Bayer4x4( desc.pixelPos, gFrameIndex );
    bool isReflection = TraceGlass( geometryProps, payload, desc.isReflection, coneAngle, bayer, true );
    MaterialProps materialProps = GetMaterialProps( geometryProps );

    float4 Lcached = float4( materialProps.Lemi, 0.0 );
    if( !geometryProps.IsMiss( ) )
    {
        // L1 cache - screen-space reprojection
        float3 prevLdiff, prevLspec;
        float reprojectionWeight = ReprojectIrradiance( false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec );
        Lcached = float4( prevLdiff + prevLspec, reprojectionWeight );

        // L2 cache - SHARC
        bool isSharcAllowed = Rng::Hash::GetFloat( ) > Lcached.w; // is needed?

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

    // Output
    return Lcached.xyz * payload.throughput;
}

//========================================================================================
// MAIN
//========================================================================================

[numthreads( 16, 16, 1 )]
void main( int2 pixelPos : SV_DispatchThreadID )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Primary ray for glass and, when PSR is active, the first mirror
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    float viewZAndTaaMask = gInOut_Mv[ pixelPos ].w;
    float viewZ = Math::Sign( gNearZ ) * abs( viewZAndTaaMask ) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float viewZafterPsr = abs( gIn_ComposedSpec_ViewZ[ pixelPos ].w ) / FP16_VIEWZ_SCALE;
    float eps = max( abs( viewZ ) * 1e-3, 1e-6 ); // tolerate FP16 depth rounding
    bool hasPsr = PT_PSR_BOUNCES_NUM != 0 && !( abs( viewZafterPsr - abs( viewZ ) ) < eps );
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );
    float tmin0 = gOrthoMode == 0 ? length( Xv ) : abs( Xv.z );

    // PSR needs the first opaque mirror too; reuse this hit instead of tracing the primary ray again
    uint flags = hasPsr ? GEOMETRY_ALL : FLAG_TRANSPARENT;
    float tmax = hasPsr ? INF : tmin0;
    GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmax, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, flags, 0 );

    bool hasGlass = geometryPropsT.Has( FLAG_TRANSPARENT );
    float3 psrThroughput = 1.0;
    float3x3 mirrorMatrix = Geometry::GetMirrorMatrix( 0 );
    float3 Xvirtual = geometryPropsT.X;

    // Search reflected segments only for pixels whose opaque surface was replaced by PSR
    [branch]
    if( !hasGlass && hasPsr )
    {
        GeometryProps geometryProps = geometryPropsT;
        float3 X0 = geometryProps.X;
        float3 V0 = geometryProps.V;
        float accumulatedHitDist = 0.0;
        float accumulatedCurvature = 0.0;

        [loop]
        for( uint bounce = 0; bounce < PT_PSR_BOUNCES_NUM && !geometryProps.IsMiss( ); bounce++ )
        {
            MaterialProps materialProps = GetMaterialProps( geometryProps );
            if( !IsDelta( materialProps ) )
                break;

            accumulatedCurvature += materialProps.curvature;
            mirrorMatrix = mul( Geometry::GetMirrorMatrix( materialProps.N ), mirrorMatrix );

            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );
            float NoV = abs( dot( materialProps.N, geometryProps.V ) );
            psrThroughput *= BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );

            float3 ray = reflect( -geometryProps.V, materialProps.N );
            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
            geometryProps = CastRay( GetXoffset( geometryProps.X, geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, PT_RAY_FLAGS );
            accumulatedHitDist += ApplyThinLensEquation( geometryProps.hitT, accumulatedCurvature );

            if( geometryProps.Has( FLAG_TRANSPARENT ) )
            {
                geometryPropsT = geometryProps;
                Xvirtual = X0 - V0 * accumulatedHitDist;
                hasGlass = true;
                break;
            }
        }
    }

    float3 Lsum = 0;
    if( hasGlass )
    {
        // Append "glass" mask to "hair" mask
        viewZAndTaaMask = -abs( viewZAndTaaMask );

        // Patch motion vectors replacing MV for the background with MV for the closest glass layer.
        // IMPORTANT: surface-based motion can be used only if the object is curved.
        // TODO: let's use the simplest heuristic for now, but better switch to some "smart" interpolation between
        // MVs for the primary opaque surface hit and the primary glass surface hit.
        float3 XvirtualPrev = Xvirtual + Geometry::RotateVectorInverse( mirrorMatrix, geometryPropsT.Xprev - geometryPropsT.X );
        float3 mvT = GetMotion( Xvirtual, XvirtualPrev );
        gInOut_Mv[ pixelPos ] = float4( mvT, viewZAndTaaMask );

        // Patch guides for RR
        [branch]
        if( gRR )
            gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( Geometry::RotateVectorInverse( mirrorMatrix, geometryPropsT.N ), 0.0, 0 );

        // Trace transparent stuff
        TraceTransparentDesc desc = ( TraceTransparentDesc )0;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;

        // IMPORTANT: use 1 reflection path and 1 refraction path at the primary glass hit to significantly reduce noise
        desc.isReflection = true;
        float3 reflection = TraceTransparent( desc );
        Lsum = reflection;

        desc.isReflection = false;
        float3 refraction = TraceTransparent( desc );
        Lsum += refraction;

        Lsum *= psrThroughput;
    }
    else
    {
        // Composed without glass
        float3 diff = gIn_ComposedDiff[ pixelPos ];
        float3 spec = gIn_ComposedSpec_ViewZ[ pixelPos ].xyz;

        Lsum = diff + spec;
    }

    // Output
    gOut_Composed[ pixelPos ] = Lsum * gExposure; // apply exposure
}
