// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_PrevComposedDiff, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 1, SET_OTHER );
NRI_RESOURCE( Texture2D<uint3>, gIn_ScramblingRanking, t, 2, SET_OTHER );
NRI_RESOURCE( Texture2D<uint4>, gIn_Sobol, t, 3, SET_OTHER );

// Outputs
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Mv, u, 0, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 1, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 2, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_BaseColor_Metalness, u, 3, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectLighting, u, 4, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectEmission, u, 5, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_PsrThroughput, u, 6, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float2>, gOut_ShadowData, u, 7, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Shadow_Translucency, u, 8, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 9, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 10, SET_OTHER );

#if( NRD_MODE == SH )
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffSh, u, 11, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecSh, u, 12, SET_OTHER );
#endif

float2 GetBlueNoise( uint2 pixelPos, uint seed = 0 )
{
    // https://eheitzresearch.wordpress.com/772-2/
    // https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html

    // Sample index
    uint sampleIndex = ( gFrameIndex + seed ) & ( BLUE_NOISE_TEMPORAL_DIM - 1 );

    // The algorithm
    uint3 A = gIn_ScramblingRanking[ pixelPos & ( BLUE_NOISE_SPATIAL_DIM - 1 ) ];
    uint rankedSampleIndex = sampleIndex ^ A.z;
    uint4 B = gIn_Sobol[ uint2( rankedSampleIndex & 255, 0 ) ];
    float4 blue = ( float4( B ^ A.xyxy ) + 0.5 ) * ( 1.0 / 256.0 );

    // ( Optional ) Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = Sequence::Bayer4x4ui( pixelPos, gFrameIndex );
    float2 dither = ( float2( d & 3, d >> 2 ) + 0.5 ) * ( 1.0 / 4.0 );
    blue += ( dither.xyxy - 0.5 ) * ( 1.0 / 256.0 );

    // Don't use blue noise in these cases
    [flatten]
    if( gDenoiserType == DENOISER_REFERENCE || gRR )
        blue.xy = Rng::Hash::GetFloat2( );

    return saturate( blue.xy );
}

float4 GetRadianceFromPreviousFrame( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos )
{
    // Reproject previous frame
    float3 prevLdiff, prevLspec;
    float prevFrameWeight = ReprojectIrradiance( true, false, gIn_PrevComposedDiff, gIn_PrevComposedSpec_PrevViewZ, geometryProps, pixelPos, prevLdiff, prevLspec );

    // Estimate how strong lighting at hit depends on the view direction
    float diffuseProbabilityBiased = EstimateDiffuseProbability( geometryProps, materialProps, true );
    float3 prevLsum = prevLdiff + prevLspec * diffuseProbabilityBiased;

    float diffuseLikeMotion = lerp( diffuseProbabilityBiased, 1.0, Math::Sqrt01( materialProps.curvature ) ); // TODO: review
    prevFrameWeight *= diffuseLikeMotion;

    float a = Color::Luminance( prevLdiff );
    float b = Color::Luminance( prevLspec );
    prevFrameWeight *= lerp( diffuseProbabilityBiased, 1.0, ( a + NRD_EPS ) / ( a + b + NRD_EPS ) );

    // Avoid really bad reprojection
    return float4( prevLsum * saturate( prevFrameWeight / 0.001 ), prevFrameWeight );
}

float GetMaterialID( GeometryProps geometryProps, MaterialProps materialProps )
{
    bool isHair = geometryProps.Has( FLAG_HAIR );
    bool isMetal = materialProps.metalness > 0.5;

    return isHair ? MATERIAL_ID_HAIR : ( isMetal ? MATERIAL_ID_METAL : MATERIAL_ID_DEFAULT );
}

//========================================================================================
// TRACE OPAQUE
//========================================================================================

/*
The function has not been designed to trace primary hits. But still can be used to trace
direct and indirect lighting.

Prerequisites:
    Rng::Hash::Initialize( )

Derivation:
    Lsum = L0 + BRDF0 * ( L1 + BRDF1 * ( L2 + BRDF2 * ( L3 +  ... ) ) )

    Lsum = L0 +
        L1 * BRDF0 +
        L2 * BRDF0 * BRDF1 +
        L3 * BRDF0 * BRDF1 * BRDF2 +
        ...
*/

struct TraceOpaqueResult
{
    float3 diffRadiance;
    float diffHitDist;

    float3 specRadiance;
    float specHitDist;

#if( NRD_MODE == SH )
    float3 diffDirection;
    float3 specDirection;
#endif
};

TraceOpaqueResult TraceOpaque( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos, float3x3 mirrorMatrix, float4 Lpsr )
{
    TraceOpaqueResult result = ( TraceOpaqueResult )0;
    result.specHitDist = NRD_FrontEnd_SpecHitDistAveraging_Begin( );

    float viewZ0 = Geometry::AffineTransform( gWorldToView, geometryProps.X ).z;
    float roughness0 = materialProps.roughness;

    // Material de-modulation ( convert irradiance into radiance )
    float3 diffFactor0, specFactor0;
    {
        float3 albedo, Rf0;
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

        NRD_MaterialFactors( materialProps.N, geometryProps.V, albedo, Rf0, materialProps.roughness, diffFactor0, specFactor0 );

        // We can combine radiance ( for everything ) and irradiance ( for hair ) in denoising if material ID test is enabled
        if( geometryProps.Has( FLAG_HAIR ) && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        {
            diffFactor0 = 1.0;
            specFactor0 = 1.0;
        }
    }

    if( USE_SHARC_DEBUG != 0 )
    {
        HashGridParameters hashGridParams;
        hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
        hashGridParams.sceneScale = SHARC_SCENE_SCALE;
        hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

        SharcHitData sharcHitData;
        sharcHitData.positionWorld = GetGlobalPos( geometryProps.X );
        sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
        sharcHitData.normalWorld = geometryProps.N;
        sharcHitData.emissive = materialProps.Lemi;

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

    #if( USE_SHARC_DEBUG == 2 )
        result.diffRadiance = HashGridDebugColoredHash( sharcHitData.positionWorld, hashGridParams );
    #else
        bool isValid = SharcGetCachedRadiance( sharcParams, sharcHitData, result.diffRadiance, true );

        // Highlight invalid cells
        // result.diffRadiance = isValid ?  result.diffRadiance : float3( 1.0, 0.0, 0.0 );
    #endif

        result.diffRadiance /= diffFactor0;

        return result;
    }

    float accumulatedHitDist = 0;
    float accumulatedDiffuseLikeMotion = 0;
    float accumulatedCurvature = 0;

    float3 Lsum = Lpsr.xyz;
    float3 pathThroughput = 1.0 - Lpsr.w;
    bool isDiffusePath = false;

    [loop]
    for( uint bounce = 1; bounce <= gBounceNum && !geometryProps.IsSky( ); bounce++ )
    {
        //=============================================================================================================================================================
        // Origin point
        //=============================================================================================================================================================

        bool isDiffuse = false;
        float lobeTanHalfAngleAtOrigin = 0.0;
        {
            // Diffuse probability
            float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );

            float rnd = Rng::Hash::GetFloat( );
            if( bounce == 1 && !gRR )
            {
                // Clamp probability to a sane range to guarantee a sample in 3x3 area ( see NRD docs )
                diffuseProbability = float( diffuseProbability != 0.0 ) * clamp( diffuseProbability, 0.25, 0.75 );
                rnd = Sequence::Bayer4x4( pixelPos, gFrameIndex ) + rnd / 16.0;
            }

            // Diffuse or specular?
            isDiffuse = rnd < diffuseProbability; // TODO: if "diffuseProbability" is clamped, "pathThroughput" should be adjusted too
            pathThroughput /= isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability );

            // Special case for primary surface ( 1st bounce starts here )
            uint importanceSampleMaxNum = 0;
            if( bounce == 1 )
            {
                isDiffusePath = isDiffuse;
                if( gDisableShadowsAndEnableImportanceSampling )
                    importanceSampleMaxNum = PT_IMPORTANCE_SAMPLES_NUM * ( isDiffuse ? 1.0 : GetSpecMagicCurve( materialProps.roughness ) );
            }
            importanceSampleMaxNum = max( importanceSampleMaxNum, 1 );

            // Choose a ray
            float3 ray = 0;
            {

                float3x3 mLocalBasis = geometryProps.Has( FLAG_HAIR ) ? Hair_GetBasis( materialProps.N, materialProps.T ) : Geometry::GetBasis( materialProps.N );
                float3 Vlocal = Geometry::RotateVector( mLocalBasis, geometryProps.V );

                // Importance sampling
                float3 rayLocal = 0;
                uint samplesNum = 0;

                for( uint sampleIndex = 0; sampleIndex < importanceSampleMaxNum; sampleIndex++ )
                {
                    float2 rnd = Rng::Hash::GetFloat2( ); // TODO: blue noise?

                    // Generate a ray in local space
                    float3 candidateRayLocal;
                #if( RTXCR_INTEGRATION == 1 )
                    if( geometryProps.Has( FLAG_HAIR ) )
                    {
                        float2 rand[2] = { Rng::Hash::GetFloat2( ), Rng::Hash::GetFloat2( ) };
                        float h = 2.0 * Rng::Hash::GetFloat( ) - 1.0;
                        float lobeRandom = Rng::Hash::GetFloat( );

                        float3 specular = 0.0;
                        float3 diffuse = 0.0;
                        float pdf = 0.0;

                        RTXCR_HairInteractionSurface hairSurface = Hair_GetSurface( Vlocal );
                        RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial( );
                        RTXCR_SampleFarFieldBcsdf( hairSurface, hairMaterial, Vlocal, h, lobeRandom, rand, candidateRayLocal, specular, diffuse, pdf );
                    }
                    else
                #endif
                    if( isDiffuse )
                        candidateRayLocal = ImportanceSampling::Cosine::GetRay( rnd );
                    else
                    {
                        float3 Hlocal = ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, PT_SPEC_LOBE_ENERGY );
                        candidateRayLocal = reflect( -Vlocal, Hlocal );
                    }

                    // If IS enabled, check the ray in LightBVH
                    bool isMiss = false;
                    if( gDisableShadowsAndEnableImportanceSampling && importanceSampleMaxNum != 1 )
                    {
                        float3 candidateRay = Geometry::RotateVectorInverse( mLocalBasis, candidateRayLocal );
                        float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
                        isMiss = CastVisibilityRay_AnyHit( geometryProps.GetXoffset( geometryProps.N ), candidateRay, 0.0, INF, mipAndCone, gLightTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
                    }

                    // Count rays hitting emissive surfaces
                    if( !isMiss )
                        samplesNum++;

                    // Save either the first ray or the current ray hitting an emissive
                    if( !isMiss || sampleIndex == 0 )
                        rayLocal = candidateRayLocal;
                }

                // Adjust throughput by percentage of rays hitting any emissive surface
                // IMPORTANT: do not modify throughput if there is no a hit, it's needed to cast a non-IS ray and get correct AO / SO at least
                if( samplesNum != 0 )
                    pathThroughput *= float( samplesNum ) / float( importanceSampleMaxNum );

                // Update path throughput
                float3 albedo, Rf0;
                BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                float3 Nlocal = float3( 0, 0, 1 );
                float3 Hlocal = normalize( Vlocal + rayLocal );

                float NoL = saturate( dot( Nlocal, rayLocal ) );
                float VoH = abs( dot( Vlocal, Hlocal ) );
                float NoV = abs( dot( Nlocal, Vlocal ) );

            #if( RTXCR_INTEGRATION == 1 )
                if( geometryProps.Has( FLAG_HAIR ) )
                {
                    float3 specular = 0.0;
                    float3 diffuse = 0.0;
                    float pdf = 0.0;

                    RTXCR_HairInteractionSurface hairGeometry = Hair_GetSurface( Vlocal );
                    RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial( );
                    RTXCR_HairFarFieldBcsdfEval( hairGeometry, hairMaterial, rayLocal, Vlocal, specular, diffuse, pdf );

                    pathThroughput *= pdf > 0.0 ? ( specular + diffuse ) / pdf : 0.0;
                }
                else
            #endif
                if( isDiffuse )
                    pathThroughput *= saturate( albedo * Math::Pi( 1.0 ) * BRDF::DiffuseTerm_Burley( materialProps.roughness, NoL, NoV, VoH ) );
                else
                {
                    float3 F = BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    pathThroughput *= F;

                    // See paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf
                    pathThroughput *= BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );
                }

                // Translucency
                if( USE_TRANSLUCENCY && geometryProps.Has( FLAG_LEAF ) && isDiffuse )
                {
                    if( Rng::Hash::GetFloat( ) < LEAF_TRANSLUCENCY )
                    {
                        rayLocal = -rayLocal;
                        geometryProps.X -= LEAF_THICKNESS * geometryProps.N;
                        pathThroughput /= LEAF_TRANSLUCENCY;
                    }
                    else
                        pathThroughput /= 1.0 - LEAF_TRANSLUCENCY;
                }

                // Transform to world space
                ray = Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                // ( Optional ) Helpful insignificant fixes
            #if( RTXCR_INTEGRATION == 1 )
                NoL = geometryProps.Has( FLAG_HAIR | FLAG_SKIN ) ? 0.0 : NoL;
            #endif
                if( NoL < 0.0 )
                {
                    if( isDiffuse )
                    {
                        // Terminate diffuse paths pointing inside the surface
                        pathThroughput = 0.0;
                    }
                    else
                    {
                        // Patch ray direction and shading normal to avoid self-intersections: https://arxiv.org/pdf/1705.01263.pdf ( Appendix 3 )
                        float b = abs( dot( geometryProps.N, materialProps.N ) ) * 0.99;

                        ray = normalize( ray + geometryProps.N * abs( NoL ) * Math::PositiveRcp( b ) );
                        materialProps.N = normalize( geometryProps.V + ray );
                    }
                }
            }

            // ( Optional ) Save sampling direction for the 1st bounce
        #if( NRD_MODE == SH )
            if( bounce == 1 )
            {
                float3 psrRay = Geometry::RotateVectorInverse( mirrorMatrix, ray );

                if( isDiffuse )
                    result.diffDirection += psrRay;
                else
                    result.specDirection += psrRay;
            }
        #endif

            // Abort if expected contribution of the current bounce is low
            /*
            GOOD PRACTICE:
            - terminate path if "pathThroughput" is smaller than some threshold
            - approximate ambient at the end of the path
            - re-use data from the previous frame
            */

            if( PT_THROUGHPUT_THRESHOLD != 0.0 && Color::Luminance( pathThroughput ) < PT_THROUGHPUT_THRESHOLD )
                break;

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            float roughnessTemp = isDiffuse ? 1.0 : materialProps.roughness;
            lobeTanHalfAngleAtOrigin = roughnessTemp * roughnessTemp / ( 1.0 + roughnessTemp * roughnessTemp );

            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
            geometryProps = CastRay( geometryProps.GetXoffset( geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
            materialProps = GetMaterialProps( geometryProps ); // TODO: try to read metrials only if L1- and L2- lighting caches failed
        }

        //=============================================================================================================================================================
        // Hit point
        //=============================================================================================================================================================

        {
            //=============================================================================================================================================================
            // Lighting
            //=============================================================================================================================================================

            float4 Lcached = float4( materialProps.Lemi, 0.0 );
            if( !geometryProps.IsSky( ) )
            {
                // L1 cache - reproject previous frame, carefully treating specular
                Lcached = GetRadianceFromPreviousFrame( geometryProps, materialProps, pixelPos );

                // L2 cache - SHARC
                HashGridParameters hashGridParams;
                hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
                hashGridParams.sceneScale = SHARC_SCENE_SCALE;
                hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
                hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

                float3 Xglobal = GetGlobalPos( geometryProps.X );
                uint level = HashGridGetLevel( Xglobal, hashGridParams );
                float voxelSize = HashGridGetVoxelSize( level, hashGridParams );

                float footprint = geometryProps.hitT * lobeTanHalfAngleAtOrigin * 2.0;
                float footprintNorm = saturate( footprint / voxelSize );

                float2 rndScaled = ImportanceSampling::Cosine::GetRay( Rng::Hash::GetFloat2( ) ).xy;
                rndScaled *= 1.0 - footprintNorm; // reduce dithering if cone is already wide
                rndScaled *= voxelSize;
                rndScaled *= USE_SHARC_DITHERING;

                float3x3 mBasis = Geometry::GetBasis( geometryProps.N );
                Xglobal += mBasis[ 0 ] * rndScaled.x + mBasis[ 1 ] * rndScaled.y;

                SharcHitData sharcHitData;
                sharcHitData.positionWorld = Xglobal;
                sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
                sharcHitData.normalWorld = geometryProps.N;
                sharcHitData.emissive = materialProps.Lemi;

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

                bool isSharcAllowed = !geometryProps.Has( FLAG_HAIR ); // ignore if the hit is hair // TODO: if hair don't allow if hitT is too short
                isSharcAllowed &= Rng::Hash::GetFloat( ) > Lcached.w; // is needed?
                isSharcAllowed &= Rng::Hash::GetFloat( ) < ( bounce == gBounceNum ? 1.0 : footprintNorm ); // is voxel size acceptable?

                float3 sharcRadiance;
                if( isSharcAllowed && SharcGetCachedRadiance( sharcParams, sharcHitData, sharcRadiance, false ) )
                    Lcached = float4( sharcRadiance, 1.0 );

                // Cache miss - compute lighting, if not found in caches
                if( Rng::Hash::GetFloat( ) > Lcached.w )
                {
                    float3 L = GetLighting( geometryProps, materialProps, LIGHTING | SHADOW ) + materialProps.Lemi;
                    Lcached.xyz = bounce < gBounceNum ? L : max( Lcached.xyz, L );
                }
            }

            //=============================================================================================================================================================
            // Other
            //=============================================================================================================================================================

            // Accumulate lighting
            float3 L = Lcached.xyz * pathThroughput;
            Lsum += L;

            // ( Biased ) Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information
            pathThroughput *= 1.0 - Lcached.w;

            // Accumulate path length for NRD ( see "README/NOISY INPUTS" )
            float a = Color::Luminance( L );
            float b = Color::Luminance( Lsum ); // already includes L
            float importance = a / ( b + 1e-6 );

            importance *= 1.0 - Color::Luminance( materialProps.Lemi ) / ( a + 1e-6 );

            float diffuseLikeMotion = EstimateDiffuseProbability( geometryProps, materialProps, true );
            diffuseLikeMotion = isDiffuse ? 1.0 : diffuseLikeMotion;

            accumulatedHitDist += ApplyThinLensEquation( geometryProps.hitT, accumulatedCurvature ) * Math::SmoothStep( 0.2, 0.0, accumulatedDiffuseLikeMotion );
            accumulatedDiffuseLikeMotion += 1.0 - importance * ( 1.0 - diffuseLikeMotion );
            accumulatedCurvature += materialProps.curvature; // yes, after hit
        }
    }

    // Normalize hit distances for REBLUR and REFERENCE ( needed only for AO ) before averaging
    float normHitDist = accumulatedHitDist;
    if( gDenoiserType != DENOISER_RELAX )
        normHitDist = REBLUR_FrontEnd_GetNormHitDist( accumulatedHitDist, viewZ0, gHitDistParams, isDiffusePath ? 1.0 : roughness0 );

    // Accumulate diffuse and specular separately for denoising
    if( !USE_SANITIZATION || NRD_IsValidRadiance( Lsum ) )
    {
        if( isDiffusePath )
        {
            result.diffRadiance += Lsum;
            result.diffHitDist += normHitDist;
        }
        else
        {
            result.specRadiance += Lsum;
            NRD_FrontEnd_SpecHitDistAveraging_Add( result.specHitDist, normHitDist );
        }
    }

    // Material de-modulation ( convert irradiance into radiance )
    result.diffRadiance /= diffFactor0;
    result.specRadiance /= specFactor0;

    NRD_FrontEnd_SpecHitDistAveraging_End( result.specHitDist );

    return result;
}

//========================================================================================
// MAIN
//========================================================================================

void WriteResult( uint2 outPixelPos, float4 diff, float4 spec, float4 diffSh, float4 specSh )
{
    gOut_Diff[ outPixelPos ] = diff;
    gOut_Spec[ outPixelPos ] = spec;

#if( NRD_MODE == SH )
    gOut_DiffSh[ outPixelPos ] = diffSh;
    gOut_SpecSh[ outPixelPos ] = specSh;
#endif
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Pixel and sample UV
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
    {
    #if( USE_DRS_STRESS_TEST == 1 )
        WriteResult( pixelPos, GARBAGE, GARBAGE, GARBAGE, GARBAGE );
    #endif

        return;
    }

    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    //================================================================================================================================================================================
    // Primary ray
    //================================================================================================================================================================================

    float3 cameraRayOrigin = 0;
    float3 cameraRayDirection = 0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    GeometryProps geometryProps0 = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, INF, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, FLAG_NON_TRANSPARENT, 0 );
    MaterialProps materialProps0 = GetMaterialProps( geometryProps0 );

    //================================================================================================================================================================================
    // Primary surface replacement ( aka jump through mirrors )
    //================================================================================================================================================================================

    float3 psrThroughput = 1.0;
    float3x3 mirrorMatrix = Geometry::GetMirrorMatrix( 0 ); // identity
    float accumulatedHitDist = 0.0;
    float accumulatedCurvature = 0.0;
    uint bounceNum = PT_PSR_BOUNCES_NUM;

    float3 X0 = geometryProps0.X;
    float3 V0 = geometryProps0.V;
    float viewZ0 = Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;

    bool isTaa5x5 = geometryProps0.Has( FLAG_HAIR | FLAG_SKIN ) || geometryProps0.IsSky( ); // switched TAA to "higher quality & slower response" mode
    float viewZAndTaaMask0 = abs( viewZ0 ) * FP16_VIEWZ_SCALE * ( isTaa5x5 ? -1.0 : 1.0 );

    [loop]
    while( bounceNum && !geometryProps0.IsSky( ) && IsDelta( materialProps0 ) )
    {
        { // Origin point
            // Accumulate curvature
            accumulatedCurvature += materialProps0.curvature; // yes, before hit

            // Accumulate mirror matrix
            mirrorMatrix = mul( Geometry::GetMirrorMatrix( materialProps0.N ), mirrorMatrix );

            // Choose a ray
            float3 ray = reflect( -geometryProps0.V, materialProps0.N );

            // Update throughput
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps0.baseColor, materialProps0.metalness, albedo, Rf0 );

            float NoV = abs( dot( materialProps0.N, geometryProps0.V ) );
            float3 Fenv = BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps0.roughness );

            psrThroughput *= Fenv;

            // Trace to the next hit
            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps0.mip, materialProps0.roughness );
            geometryProps0 = CastRay( geometryProps0.GetXoffset( geometryProps0.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
            materialProps0 = GetMaterialProps( geometryProps0 );
        }

        { // Hit point
            // Accumulate hit distance representing virtual point position ( see "README/NOISY INPUTS" )
            accumulatedHitDist += ApplyThinLensEquation( geometryProps0.hitT, accumulatedCurvature ) ; // TODO: take updated from NRD
        }

        bounceNum--;
    }

    //================================================================================================================================================================================
    // G-buffer ( guides )
    //================================================================================================================================================================================

    // Motion
    float3 Xvirtual = X0 - V0 * accumulatedHitDist;
    float3 XvirtualPrev = Xvirtual + geometryProps0.Xprev - geometryProps0.X;
    float3 motion = GetMotion( Xvirtual, XvirtualPrev );

    gOut_Mv[ pixelPos ] = float4( motion, viewZAndTaaMask0 ); // IMPORTANT: keep viewZ before PSR ( needed for glass )

    // ViewZ
    float viewZ = Geometry::AffineTransform( gWorldToView, Xvirtual ).z;
    viewZ = geometryProps0.IsSky( ) ? Math::Sign( viewZ ) * INF : viewZ;

    gOut_ViewZ[ pixelPos ] = viewZ;

    // Emission
    gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi * psrThroughput;

    // Early out
    if( geometryProps0.IsSky( ) )
    {
    #if( USE_INF_STRESS_TEST == 1 )
        WriteResult( pixelPos, GARBAGE, GARBAGE, GARBAGE, GARBAGE );
    #endif

        return;
    }

    // Normal, roughness and material ID
    float3 N = Geometry::RotateVectorInverse( mirrorMatrix, materialProps0.N );
#if( RTXCR_INTEGRATION == 1 )
    if( geometryProps0.Has( FLAG_HAIR ) )
    {
        // Generate a better guide for hair
        float3 B = cross( geometryProps0.V, geometryProps0.T.xyz );
        float3 n = normalize( cross( geometryProps0.T.xyz, B ) );

        float pixelSize = gUnproject * lerp( abs( viewZ ), 1.0, abs( gOrthoMode ) );
        float f = NRD_GetNormalizedStrandThickness( STRAND_THICKNESS, pixelSize );
        f = lerp( 0.0, 0.25, f );

        N = normalize( lerp( n, N, f ) );
    }
#endif

    float materialID = GetMaterialID( geometryProps0, materialProps0 );
#if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
    materialID = frac( geometryProps0.X ).x < 0.05 ? MATERIAL_ID_HAIR : materialID;
#endif

    gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( N, materialProps0.roughness, materialID );

    // Base color and metalness
    gOut_BaseColor_Metalness[ pixelPos ] = float4( Color::ToSrgb( materialProps0.baseColor ), materialProps0.metalness );

    // Direct lighting
    float3 Xshadow;
    float3 Ldirect = GetLighting( geometryProps0, materialProps0, LIGHTING | SSS, Xshadow );

    gOut_DirectLighting[ pixelPos ] = Ldirect; // "psrThroughput" applied in "Composition"
    gOut_PsrThroughput[ pixelPos ] = psrThroughput;

    // Lighting at PSR hit, if found
    float4 Lpsr = 0;
    if( !geometryProps0.IsSky( ) && bounceNum != PT_PSR_BOUNCES_NUM )
    {
        // L1 cache - reproject previous frame, carefully treating specular
        Lpsr = GetRadianceFromPreviousFrame( geometryProps0, materialProps0, pixelPos );

        // Subtract direct lighting, process it separately
        float3 L = Ldirect * GetLighting( geometryProps0, materialProps0, SHADOW ) + materialProps0.Lemi;
        Lpsr.xyz = max( Lpsr.xyz - L, 0.0 );

        // This is important!
        Lpsr.xyz *= Lpsr.w;
    }

    //================================================================================================================================================================================
    // Secondary rays
    //================================================================================================================================================================================

    TraceOpaqueResult result = TraceOpaque( geometryProps0, materialProps0, pixelPos, mirrorMatrix, Lpsr );

#if( USE_MOVING_EMISSION_FIX == 1 )
    // Or emissives ( not having lighting in diffuse and specular ) can use a different material ID
    result.diffRadiance += materialProps0.Lemi / Math::Pi( 2.0 );
    result.specRadiance += materialProps0.Lemi / Math::Pi( 2.0 );
#endif

#if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
    if( frac( geometryProps0.X ).x < 0.05 )
        result.diffRadiance = float3( 0, 10, 0 ) * Color::Luminance( result.diffRadiance );
#endif

#if( USE_SIMULATED_FIREFLY_TEST == 1 )
    const float maxFireflyEnergyScaleFactor = 10000.0;
    result.diffRadiance /= lerp( 1.0 / maxFireflyEnergyScaleFactor, 1.0, Rng::Hash::GetFloat( ) );
#endif

    //================================================================================================================================================================================
    // Sun shadow
    //================================================================================================================================================================================

    geometryProps0.X = Xshadow;

    float2 rnd = GetBlueNoise( pixelPos );
    rnd = ImportanceSampling::Cosine::GetRay( rnd ).xy;
    rnd *= gTanSunAngularRadius;

    float3 sunDirection = normalize( gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz );
    float3 Xoffset = geometryProps0.GetXoffset( sunDirection, PT_SHADOW_RAY_OFFSET );
    float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps0.mip, gTanSunAngularRadius );

    float shadowTranslucency = ( Color::Luminance( Ldirect ) != 0.0 && !gDisableShadowsAndEnableImportanceSampling ) ? 1.0 : 0.0;
    float shadowHitDist = 0.0;

    while( shadowTranslucency > 0.01 )
    {
        GeometryProps geometryPropsShadow = CastRay( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, 0 );

        // Update hit dist
        shadowHitDist += geometryPropsShadow.hitT;

        // Terminate on miss ( before updating translucency! )
        if( geometryPropsShadow.IsSky( ) )
            break;

        // ( Biased ) Cheap approximation of shadows through glass
        float NoV = abs( dot( geometryPropsShadow.N, sunDirection ) );
        shadowTranslucency *= lerp( geometryPropsShadow.Has( FLAG_TRANSPARENT ) ? 0.9 : 0.0, 0.0, Math::Pow01( 1.0 - NoV, 2.5 ) );

        // Go to the next hit
        Xoffset += sunDirection * ( geometryPropsShadow.hitT + 0.001 );
    }

    float penumbra = SIGMA_FrontEnd_PackPenumbra( shadowHitDist, gTanSunAngularRadius );
    float4 translucency = SIGMA_FrontEnd_PackTranslucency( shadowHitDist, shadowTranslucency );

    gOut_ShadowData[ pixelPos ] = penumbra;
    gOut_Shadow_Translucency[ pixelPos ] = translucency;

    //================================================================================================================================================================================
    // Output
    //================================================================================================================================================================================

    float4 outDiff = 0.0;
    float4 outSpec = 0.0;
    float4 outDiffSh = 0.0;
    float4 outSpecSh = 0.0;

    if( gDenoiserType == DENOISER_RELAX )
    {
    #if( NRD_MODE == SH )
        outDiff = RELAX_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #else
        outDiff = RELAX_FrontEnd_PackRadianceAndHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackRadianceAndHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
    }
    else
    {
    #if( NRD_MODE == SH )
        outDiff = REBLUR_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #else
        outDiff = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
    }

    WriteResult( pixelPos, outDiff, outSpec, outDiffSh, outSpecSh );
}
