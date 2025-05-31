// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

#define SHARC_UPDATE 1
#include "SharcCommon.h"

float3 GetAmbientBRDF( GeometryProps geometryProps, MaterialProps materialProps, bool approximate = false )
{
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float3 Fenv = Rf0;
    if( !approximate )
    {
        float NoV = abs( dot( materialProps.N, geometryProps.V ) );
        Fenv = BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );
    }

    Fenv *= GetSpecMagicCurve( materialProps.roughness );

    float3 ambBRDF = albedo * ( 1.0 - Fenv ) + Fenv;
    ambBRDF *= float( !geometryProps.IsSky( ) );

    return ambBRDF;
}

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
    sharcParams.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;
    sharcParams.voxelDataBuffer = gInOut_SharcVoxelDataBuffer;
    sharcParams.voxelDataBufferPrev = gInOut_SharcVoxelDataBufferPrev;

    SharcState sharcState;
    SharcInit( sharcState );

    MaterialProps materialProps = GetMaterialProps( geometryProps, USE_SHARC_V_DEPENDENT == 0 );

    // Update SHARC cache ( this is always a hit )
    {
        SharcHitData sharcHitData;
        sharcHitData.positionWorld = GetGlobalPos( geometryProps.X ) + ( Rng::Hash::GetFloat4( ).xyz - 0.5 ) * SHARC_POS_DITHER;
        sharcHitData.normalWorld = normalize( geometryProps.N + ( Rng::Hash::GetFloat4( ).xyz - 0.5 ) * SHARC_NORMAL_DITHER );
        sharcHitData.emissive = materialProps.Lemi;

        SharcSetThroughput( sharcState, 1.0 );

        float3 L = GetShadowedLighting( geometryProps, materialProps, SKIP_EMISSIVE );
        if( !SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, 1.0 ) )
            return;
    }

    // Secondary rays
    [loop]
    for( uint bounce = 1; bounce <= SHARC_PROPOGATION_DEPTH; bounce++ )
    {
        //=============================================================================================================================================================
        // Origin point
        //=============================================================================================================================================================

        float3 throughput = 1.0;
        {
            // Estimate diffuse probability
            #if( USE_SHARC_V_DEPENDENT == 1 )
                float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );
            #else
                float diffuseProbability = 1.0;
            #endif

            // Diffuse or specular?
            bool isDiffuse = Rng::Hash::GetFloat( ) < diffuseProbability;
            throughput /= abs( float( !isDiffuse ) - diffuseProbability );

            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

            // Choose a ray
            float3x3 mLocalBasis = Geometry::GetBasis( materialProps.N );

            float3 Vlocal = Geometry::RotateVector( mLocalBasis, geometryProps.V );
            float3 ray = 0;
            uint samplesNum = 0;

            // If IS is enabled, generate up to PT_IMPORTANCE_SAMPLES_NUM rays depending on roughness
            // If IS is disabled, there is no need to generate up to PT_IMPORTANCE_SAMPLES_NUM rays for specular because VNDF v3 doesn't produce rays pointing inside the surface
            uint maxSamplesNum = 0;
            if( bounce == 1 && gDisableShadowsAndEnableImportanceSampling ) // TODO: use IS in each bounce?
                maxSamplesNum = PT_IMPORTANCE_SAMPLES_NUM * ( isDiffuse ? 1.0 : materialProps.roughness );
            maxSamplesNum = max( maxSamplesNum, 1 );

            for( uint sampleIndex = 0; sampleIndex < maxSamplesNum; sampleIndex++ )
            {
                float2 rnd = Rng::Hash::GetFloat2( );

                // Generate a ray in local space
                float3 r;
                {
                    if( isDiffuse )
                        r = ImportanceSampling::Cosine::GetRay( rnd );
                    else
                    {
                        float3 Hlocal = ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, PT_SPEC_LOBE_ENERGY );
                        r = reflect( -Vlocal, Hlocal );
                    }
                }

                // Transform to world space
                r = Geometry::RotateVectorInverse( mLocalBasis, r );

                // Importance sampling for direct lighting
                // TODO: move direct lighting tracing into a separate pass:
                // - currently AO and SO get replaced with useless distances to closest lights if IS is on
                // - better separate direct and indirect lighting denoising

                //   1. If IS enabled, check the ray in LightBVH
                bool isMiss = false;
                if( gDisableShadowsAndEnableImportanceSampling && maxSamplesNum != 1 )
                    isMiss = CastVisibilityRay_AnyHit( geometryProps.GetXoffset( geometryProps.N ), r, 0.0, INF, mipAndCone, gLightTlas, FLAG_NON_TRANSPARENT, 0 );

                //   2. Count rays hitting emissive surfaces
                if( !isMiss )
                    samplesNum++;

                //   3. Save either the first ray or the current ray hitting an emissive
                if( !isMiss || sampleIndex == 0 )
                    ray = r;
            }

            // Adjust throughput by percentage of rays hitting any emissive surface
            // IMPORTANT: do not modify throughput if there is no a hit, it's needed to cast a non-IS ray and get correct AO / SO at least
            if( samplesNum != 0 )
                throughput *= float( samplesNum ) / float( maxSamplesNum );

            // ( Optional ) Helpful insignificant fixes
            #if( USE_SHARC_V_DEPENDENT == 1 )
                float a = dot( geometryProps.N, ray );
                if( a < 0.0 )
                {
                    if( isDiffuse )
                    {
                        // Terminate diffuse paths pointing inside the surface
                        throughput = 0.0;
                    }
                    else
                    {
                        // Patch ray direction to avoid self-intersections: https://arxiv.org/pdf/1705.01263.pdf ( Appendix 3 )
                        float b = dot( geometryProps.N, materialProps.N );
                        ray = normalize( ray + materialProps.N * Math::Sqrt01( 1.0 - a * a ) / b );
                    }
                }
            #endif

            // Update path throughput
            #if( USE_SHARC_V_DEPENDENT == 1 )
                float3 albedo, Rf0;
                BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                float3 H = normalize( geometryProps.V + ray );
                float VoH = abs( dot( geometryProps.V, H ) );
                float NoL = saturate( dot( materialProps.N, ray ) );

                if( isDiffuse )
                {
                    float NoV = abs( dot( materialProps.N, geometryProps.V ) );
                    throughput *= saturate( albedo * Math::Pi( 1.0 ) * BRDF::DiffuseTerm_Burley( materialProps.roughness, NoL, NoV, VoH ) );
                }
                else
                {
                    float3 F = BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    throughput *= F;

                    // See paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf
                    throughput *= BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );
                }
            #else
               throughput = GetAmbientBRDF( geometryProps, materialProps );
            #endif

            // Translucency
            if( USE_TRANSLUCENCY && geometryProps.Has( FLAG_LEAF ) && isDiffuse )
            {
                if( Rng::Hash::GetFloat( ) < LEAF_TRANSLUCENCY )
                {
                    ray = -ray;
                    geometryProps.X -= LEAF_THICKNESS * geometryProps.N;
                    throughput /= LEAF_TRANSLUCENCY;
                }
                else
                    throughput /= 1.0 - LEAF_TRANSLUCENCY;
            }

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            geometryProps = CastRay( geometryProps.GetXoffset( geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, 0 );
            materialProps = GetMaterialProps( geometryProps, USE_SHARC_V_DEPENDENT == 0 );
        }

        { // Update SHARC cache
            SharcSetThroughput( sharcState, throughput );

            if( geometryProps.IsSky( ) )
            {
                SharcUpdateMiss( sharcParams, sharcState, materialProps.Lemi );
                break;
            }
            else
            {
                SharcHitData sharcHitData;
                sharcHitData.positionWorld = GetGlobalPos( geometryProps.X ) + ( Rng::Hash::GetFloat4( ).xyz - 0.5 ) * SHARC_POS_DITHER;
                sharcHitData.normalWorld = normalize( geometryProps.N + ( Rng::Hash::GetFloat4( ).xyz - 0.5 ) * SHARC_NORMAL_DITHER );
                sharcHitData.emissive = materialProps.Lemi;

                float3 L = GetShadowedLighting( geometryProps, materialProps, SKIP_EMISSIVE );
                if( !SharcUpdateHit( sharcParams, sharcState, sharcHitData, L, Rng::Hash::GetFloat( ) ) )
                    break;
            }
        }
    }
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    /*
    TODO: modify SHARC to support:
    - material de-modulation
    - 2 levels of detail: fine and coarse ( large voxels )
    - firefly suppression
    - anti-lag
    - dynamic "sceneScale"
    - auto "sceneScale" adjustment to guarantee desired number of samples in voxels on average
    */

    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Sample position
    float2 sampleUv = ( pixelPos + 0.5 + gJitter * gRectSize ) * SHARC_DOWNSCALE * gInvRectSize;

    // Primary ray
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );

    float3 Xoffset = Geometry::AffineTransform( gViewToWorld, Xv );
    float3 ray = gOrthoMode == 0.0 ? normalize( Geometry::RotateVector( gViewToWorld, Xv ) ) : -gViewDirection.xyz;

    // Force some portion of rays to be absolutely random to keep cache alive behind the camera
    if( Rng::Hash::GetFloat( ) < 0.2 )
        ray = normalize( Rng::Hash::GetFloat4( ).xyz - 0.5 );

    // Skip delta events
    GeometryProps geometryProps;
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;
    float2 mip = GetConeAngleFromAngularRadius( 0.0, gTanPixelAngularRadius * SHARC_DOWNSCALE );

    [loop]
    for( uint bounce = 1; bounce <= PT_DELTA_BOUNCES_NUM; bounce++ ) // TODO: stop if pathThroughput is low?
    {
        uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;

        geometryProps = CastRay( Xoffset, ray, 0.0, INF, mip, gWorldTlas, flags, 0 );
        MaterialProps materialProps = GetMaterialProps( geometryProps, USE_SHARC_V_DEPENDENT == 0 );

        bool isDelta = geometryProps.Has( FLAG_TRANSPARENT );

        if( !isDelta || geometryProps.IsSky() )
            break;

        // Reflection or refraction?
        float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
        float F = BRDF::FresnelTerm_Dielectric( eta, NoV );
        float rnd = Rng::Hash::GetFloat( );
        bool isReflection = rnd < F;

        eta = GetDeltaEventRay( geometryProps, isReflection, eta, Xoffset, ray );
    }

    // Opaque path
    if( !geometryProps.IsSky( ) )
        Trace( geometryProps ); // TODO: looping this for 4-8 iterations helps to improve cache quality, but it's expensive
}
