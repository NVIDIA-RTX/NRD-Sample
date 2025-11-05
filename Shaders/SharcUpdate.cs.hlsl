// © 2024 NVIDIA Corporation

#define SHARC_UPDATE 1

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

void Trace( GeometryProps geometryProps )
{
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

    MaterialProps materialProps = GetMaterialProps( geometryProps );

    // Update SHARC cache ( this is always a hit )
    {
        SharcHitData sharcHitData;
        sharcHitData.positionWorld = GetGlobalPos( geometryProps.X );
        sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
        sharcHitData.normalWorld = geometryProps.N;
        sharcHitData.emissive = materialProps.Lemi;

        SharcSetThroughput( sharcState, 1.0 );

        float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW );
        if( !SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, 1.0 ) )
            return;
    }

    // Secondary rays
    [loop]
    for( uint bounce = 1; bounce <= SHARC_PROPAGATION_DEPTH; bounce++ )
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
            if( bounce == 1 && gDisableShadowsAndEnableImportanceSampling )
                sampleMaxNum = PT_IMPORTANCE_SAMPLES_NUM * ( isDiffuse ? 1.0 : GetSpecMagicCurve( materialProps.roughness ) );
            sampleMaxNum = max( sampleMaxNum, 1 );

            float2 rnd2 = Rng::Hash::GetFloat2( );
            float3 ray = GenerateRayAndUpdateThroughput( geometryProps, materialProps, throughput, sampleMaxNum, isDiffuse, rnd2, 0 );

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
            geometryProps = CastRay( geometryProps.GetXoffset( geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, 0 );
            materialProps = GetMaterialProps( geometryProps );
        }

        { // Update SHARC cache
            SharcSetThroughput( sharcState, throughput );

            if( geometryProps.IsMiss( ) )
            {
                SharcUpdateMiss( sharcParams, sharcState, materialProps.Lemi );
                break;
            }
            else
            {
                SharcHitData sharcHitData;
                sharcHitData.positionWorld = GetGlobalPos( geometryProps.X );
                sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
                sharcHitData.normalWorld = geometryProps.N;
                sharcHitData.emissive = materialProps.Lemi;

                float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW );
                if( !SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, Rng::Hash::GetFloat( ) ) )
                    break;
            }
        }
    }
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Sample position
    float2 sampleUv = ( pixelPos + 0.5 + gJitter * gRectSize ) * SHARC_DOWNSCALE * gInvRectSize;

    // Primary ray
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );

    float3 Xoffset = Geometry::AffineTransform( gViewToWorld, Xv );
    float3 ray = gOrthoMode == 0.0 ? normalize( Geometry::RotateVector( gViewToWorld, Xv ) ) : -gViewDirection.xyz;

    // Skip delta events
    GeometryProps geometryProps;
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;
    float2 mip = GetConeAngleFromAngularRadius( 0.0, gTanPixelAngularRadius * SHARC_DOWNSCALE );

    [loop]
    for( uint bounce = 1; bounce <= PT_DELTA_BOUNCES_NUM; bounce++ )
    {
        uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;

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

    // Opaque path
    if( !geometryProps.IsMiss( ) )
        Trace( geometryProps ); // TODO: looping this for 4-8 iterations helps to improve cache quality, but it's expensive
}
