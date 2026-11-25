// © 2022 NVIDIA Corporation

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_PrevComposedDiff, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 1, SET_OTHER );

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

float4 GetRadianceFromPreviousFrame( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos )
{
    float3 prevLdiff, prevLspec;
    float prevFrameWeight = ReprojectIrradiance( true, false, gIn_PrevComposedDiff, gIn_PrevComposedSpec_PrevViewZ, geometryProps, pixelPos, prevLdiff, prevLspec );

    // Estimate how strong lighting at hit depends on the view direction
    float normCurvature = saturate( sqrt( abs( materialProps.curvature ) ) / 2.5 );
    float specConfidence = GetSpecMagicCurve( materialProps.roughness );
    specConfidence = lerp( specConfidence, 1.0, normCurvature );

    float diffLum = Color::Luminance( prevLdiff );
    float specLum = Color::Luminance( prevLspec );
    float specWeight = specLum / ( diffLum + specLum + NRD_EPS );

    prevFrameWeight *= lerp( 1.0, specConfidence, specWeight );

    float3 prevLsum = prevLdiff + prevLspec * specConfidence;

    // Avoid really bad reprojection
    prevLsum *= saturate( prevFrameWeight / 0.05 );

    return float4( prevLsum, prevFrameWeight );
}

float GetMaterialID( GeometryProps geometryProps, MaterialProps materialProps )
{
    bool isHair = geometryProps.Has( FLAG_HAIR );
    bool isMetal = materialProps.metalness > 0.5;

    return isHair ? MATERIAL_ID_HAIR : ( isMetal ? MATERIAL_ID_METAL : MATERIAL_ID_DEFAULT );
}

float3 GetSharcDebugColor( GeometryProps geometryProps, MaterialProps materialProps )
{
    SharcParameters sharcParams = GetSharcParameters( );
    SharcHitData sharcHitData = GetSharcHitData( geometryProps, materialProps );

    float3 color = 0;
    #if( USE_SHARC_DEBUG == 2 )
        color = HashGridDebugColoredHash( sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParams.hashGridParameters );
    #else
        bool isValid = SharcGetCachedRadiance( sharcParams, sharcHitData, color, true );

        // Highlight invalid cells
        color = isValid ?  color : float3( 1.0, 0.0, 0.0 );
    #endif

    return color;
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
    float3 radiance;
    float accumulatedHitDist;

#if( NRD_MODE == SH )
    float3 direction;
#endif
};

TraceOpaqueResult TraceOpaque( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos, float3x3 mirrorMatrix, float4 Lpsr, out bool isDiffusePath )
{
    TraceOpaqueResult result = ( TraceOpaqueResult )0;
    isDiffusePath = false;

    TracePayload payload = ( TracePayload )0;
    payload.radiance = Lpsr.xyz;
    payload.throughput = 1.0 - Lpsr.w;
    payload.eta = BRDF::IOR::Air / BRDF::IOR::Glass;

    [loop]
    for( uint bounce = 1; bounce <= gBounceNum && !geometryProps.IsMiss( ); bounce++ )
    {
        //=============================================================================================================================================================
        // Origin point
        //=============================================================================================================================================================

        bool isDiffuse = false;
        float roughnessAtOrigin = 0.0;
        {
            // Diffuse probability
            float diffuseProbability = GetClampedDiffuseProbability( geometryProps, materialProps );

            // Diffuse or specular?
            float rnd = Rng::Hash::GetFloat( );
            if( bounce == 1 && !gRR )
            {
                // Guarantee a sample in 3x3 area ( for the 1st bounce, see NRD docs )
                float bayer = Sequence::Bayer4x4( pixelPos, gFrameIndex );
                float jitter = Sequence::Weyl1D( rsqrt( 7.0 ), gFrameIndex ); // screen-uniform

                // Fix harmonic interference of "bayer" and "blue noise" ( i.e. decorrelate ), which are both "pow-of-2" structures ( it doesn't break white noise )
                rnd = frac( bayer + jitter );
            }

            isDiffuse = rnd < diffuseProbability;
            payload.throughput /= isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability );

            // Importance sampling
            uint sampleMaxNum = GetImportanceSampleNum( materialProps.roughness, isDiffuse, bounce == 1 );

            float3 ray = GenerateRayAndUpdateThroughput( geometryProps, materialProps, payload.throughput, sampleMaxNum, isDiffuse, pixelPos, bounce, GR_HAIR | GR_ALLOW_BN  );

            // Special case for primary surface ( 1st bounce starts here )
            if( bounce == 1 )
            {
                isDiffusePath = isDiffuse;

                // ( Optional ) Save sampling direction for the 1st bounce
                #if( NRD_MODE == SH )
                    float3 psrRay = Geometry::RotateVectorInverse( mirrorMatrix, ray );

                    result.direction = psrRay;
                #endif
            }

            // Abort tracing if the current bounce contribution is low
            if( PT_THROUGHPUT_THRESHOLD != 0.0 && Color::Luminance( payload.throughput ) < PT_THROUGHPUT_THRESHOLD )
                break;

            //=========================================================================================================================================================
            // Trace to the next hit
            //=========================================================================================================================================================

            roughnessAtOrigin = isDiffuse ? 1.0 : materialProps.roughness;

            bool traceGlass = USE_GLASS_IN_OPAQUE_TRACING && ( bounce == 1 || payload.eta > 1.0 );
            uint flags = traceGlass ? GEOMETRY_ALL : FLAG_NON_TRANSPARENT;

            float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, roughnessAtOrigin );
            geometryProps = CastRay( GetXoffset( geometryProps.X, geometryProps.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, flags, PT_RAY_FLAGS );

            // Trace through glass ( events do not consume the opaque bounce budget )
            if( traceGlass )
            {
                GlassMotion unused = ( GlassMotion )0;
                TraceGlass( geometryProps, payload, false, mipAndCone.y, 0.0, false, unused );
            }

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
            if( !geometryProps.IsMiss( ) )
            {
                // L1 cache - reproject previous frame, carefully treating specular
                Lcached = GetRadianceFromPreviousFrame( geometryProps, materialProps, pixelPos );

                // L2 cache - SHARC
                SharcParameters sharcParams = GetSharcParameters( );
                float voxelSize = GetSharcVoxelSize( geometryProps.X, sharcParams.hashGridParameters );
                float lobeTanHalfAngleAtOrigin = roughnessAtOrigin * roughnessAtOrigin / ( 1.0 + roughnessAtOrigin * roughnessAtOrigin );
                float footprint = geometryProps.hitT * lobeTanHalfAngleAtOrigin * 2.0;
                float footprintNorm = saturate( footprint / voxelSize );

                bool isSharcAllowed = !geometryProps.Has( FLAG_HAIR ); // ignore if the hit is hair // TODO: if hair don't allow if hitT is too short
                isSharcAllowed &= Rng::Hash::GetFloat( ) > Lcached.w; // is needed?
                isSharcAllowed &= Rng::Hash::GetFloat( ) < ( bounce == gBounceNum ? 1.0 : footprintNorm ); // is voxel size acceptable?
                isSharcAllowed &= !USE_WHITE_FURNACE;

                if( isSharcAllowed )
                {
                    float jitterAmount = USE_SHARC_DITHERING * ( 1.0 - footprintNorm ); // reduce dithering if cone is already wide
                    float3 jitter = GetSharcJitter( geometryProps.N, voxelSize, jitterAmount );

                    float3 sharcRadiance = 0;
                    bool isFound = TryGetSharcRadiance( sharcParams, geometryProps, materialProps, jitter, sharcRadiance );

                    if( isFound )
                    {
                        // Apply occlusion estimation for the last bounce
                    #if( USE_AO_FOR_LAST_BOUNCE == 1 )
                        if( bounce == gBounceNum )
                            sharcRadiance *= Math::Sqrt01( geometryProps.hitT / voxelSize );
                    #endif

                        Lcached = float4( sharcRadiance, 1.0 );
                    }
                }

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
            float3 L = Lcached.xyz * payload.throughput;
            payload.radiance += L;

            // ( Biased ) Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information
            payload.throughput *= 1.0 - Lcached.w;

            // Accumulate path length for NRD ( see "README/NOISY INPUTS" )
            payload.accumulatedHitDist += ApplyThinLensEquation( geometryProps.hitT, payload.accumulatedCurvature );

            // Wide lobes and low-value tails act as additional curvature
            float importance = saturate( Color::Luminance( L ) / ( Color::Luminance( payload.radiance ) + 1e-6 ) );
            float tailWeight = importance * ( 1.0 - roughnessAtOrigin * roughnessAtOrigin );
            payload.accumulatedCurvature += 0.5 * ( 1.0 - tailWeight ) / max( geometryProps.hitT * tailWeight, 1e-6 );
            payload.accumulatedCurvature += materialProps.curvature; // yes, after hit: exclude primary curvature, affect subsequent segments
        }
    }

    result.radiance = payload.radiance;
    result.accumulatedHitDist = payload.accumulatedHitDist;

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
void main( uint2 pixelPos : SV_DispatchThreadID )
{
    //================================================================================================================================================================================
    // Primary ray
    //================================================================================================================================================================================

    GeometryProps geometryProps0;
    MaterialProps materialProps0;
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

        float3 cameraRayOrigin = 0;
        float3 cameraRayDirection = 0;
        GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

        geometryProps0 = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, INF, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, FLAG_NON_TRANSPARENT, 0 );
        materialProps0 = GetMaterialProps( geometryProps0 );
    }

    //================================================================================================================================================================================
    // Primary surface replacement ( aka jump through mirrors )
    //================================================================================================================================================================================

    float3x3 mirrorMatrix = Geometry::GetMirrorMatrix( 0 ); // identity
    float3 psrThroughput = 1.0;
    float3 Xvirtual;
    float viewZAndTaaMask0;
    bool hasPsr;
    {
        float accumulatedHitDist = 0.0;
        float accumulatedCurvature = 0.0;
        uint bounceNum = PT_PSR_BOUNCES_NUM;

        float3 X0 = geometryProps0.X;
        float3 V0 = geometryProps0.V;
        float viewZ0 = Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;

        bool isTaa5x5 = geometryProps0.Has( FLAG_HAIR | FLAG_SKIN ) || geometryProps0.IsMiss( ); // switched TAA to "higher quality & slower response" mode
        viewZAndTaaMask0 = abs( viewZ0 ) * FP16_VIEWZ_SCALE * ( isTaa5x5 ? -1.0 : 1.0 );

        [loop]
        while( bounceNum && !geometryProps0.IsMiss( ) && IsDelta( materialProps0 ) )
        {
            { // Origin point
                // Accumulate curvature and mirror matrix
                accumulatedCurvature += materialProps0.curvature; // yes, before hit: include the replaced primary mirror
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
                geometryProps0 = CastRay( GetXoffset( geometryProps0.X, geometryProps0.N ), ray, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
                materialProps0 = GetMaterialProps( geometryProps0 );
            }

            { // Hit point
                // Accumulate hit distance representing virtual point position ( see "README/NOISY INPUTS" )
                accumulatedHitDist += ApplyThinLensEquation( geometryProps0.hitT, accumulatedCurvature ); // TODO: take updated from NRD
            }

            bounceNum--;
        }

        Xvirtual = X0 - V0 * accumulatedHitDist;
        hasPsr = bounceNum != PT_PSR_BOUNCES_NUM;
    }

    //================================================================================================================================================================================
    // G-buffer ( guides )
    //================================================================================================================================================================================

    float4 Lpsr = 0;
    float3 Xshadow;
    float3 Ldirect;
    {
        // Motion
        float3 XvirtualPrev = Xvirtual + geometryProps0.Xprev - geometryProps0.X;
        float3 motion = GetMotion( Xvirtual, XvirtualPrev );

        gOut_Mv[ pixelPos ] = float4( motion, viewZAndTaaMask0 ); // IMPORTANT: keep viewZ before PSR ( needed for glass )

        // ViewZ
        float viewZ = Geometry::AffineTransform( gWorldToView, Xvirtual ).z;
        viewZ = geometryProps0.IsMiss( ) ? Math::Sign( viewZ ) * INF : viewZ;

        gOut_ViewZ[ pixelPos ] = viewZ;

        // Emission
        gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi * psrThroughput;

        // Early out
        if( geometryProps0.IsMiss( ) )
        {
        #if( USE_INF_STRESS_TEST == 1 )
            WriteResult( pixelPos, GARBAGE, GARBAGE, GARBAGE, GARBAGE );
        #endif

            return;
        }

        // PSR throughput
        gOut_PsrThroughput[ pixelPos ] = psrThroughput;

        // Normal, roughness and material ID
        float3 N = Geometry::RotateVectorInverse( mirrorMatrix, materialProps0.N );
    #if( RTXCR_INTEGRATION == 1 )
        if( geometryProps0.Has( FLAG_HAIR ) )
        {
            // Generate a better guide for hair
            float3 B = cross( geometryProps0.V, geometryProps0.T.xyz );
            float3 n = normalize( cross( geometryProps0.T.xyz, B ) );

            float pixelSize = gUnproject * lerp( abs( viewZ ), 1.0, abs( gOrthoMode ) );
            float f = NRD_GetNormalizedStrandThickness( STRAND_THICKNESS / gUnitToMetersMultiplier, pixelSize );
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
        Ldirect = GetLighting( geometryProps0, materialProps0, LIGHTING | SSS, Xshadow );

        gOut_DirectLighting[ pixelPos ] = Ldirect; // "psrThroughput" applied in "Composition"

        // Lighting at PSR hit, if found
        if( hasPsr )
        {
            float3 L = Ldirect * GetLighting( geometryProps0, materialProps0, SHADOW ) + materialProps0.Lemi;

            // L1 cache - reproject previous frame, carefully treating specular
            Lpsr = GetRadianceFromPreviousFrame( geometryProps0, materialProps0, pixelPos );

            // Subtract direct lighting, process it separately
            Lpsr.xyz = max( Lpsr.xyz - L, 0.0 );

            // This is important!
            Lpsr.xyz *= Lpsr.w;
        }
    }

    //================================================================================================================================================================================
    // Sun shadow
    //================================================================================================================================================================================

    {
        float2 rnd = Rng::Hash::GetFloat2( );
        if( USE_BLUE_NOISE_FOR_SHADOWS )
            rnd = GetBlueNoise( pixelPos, gIn_ScramblingRanking4, 4, gFrameIndex );

        float3 sunDirection = SampleSunDirection( rnd );
        float3 Xoffset = GetXoffset( Xshadow, sunDirection, PT_SHADOW_RAY_OFFSET );
        float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps0.mip, gTanSunAngularRadius );

        float shadowTranslucency = ( Color::Luminance( Ldirect ) != 0.0 && !gDisableShadowsAndEnableImportanceSampling ) ? 1.0 : 0.0;
        float shadowHitDist = 0.0;

        while( shadowTranslucency > 0.01 )
        {
            GeometryProps geometryPropsShadow = CastRay( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, 0 );

            // Update hit dist
            shadowHitDist += geometryPropsShadow.hitT;

            // Terminate on miss ( before updating translucency! )
            if( geometryPropsShadow.IsMiss( ) )
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
    }

    //================================================================================================================================================================================
    // Secondary rays
    //================================================================================================================================================================================

    {
        // Material de-modulation factors
        float3 diffFactor0;
        float3 specFactor0;
        {
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps0.baseColor, materialProps0.metalness, albedo, Rf0 );

            GetMaterialFactors( materialProps0.N, geometryProps0.V, albedo, Rf0, materialProps0.roughness, geometryProps0.Has( FLAG_HAIR ), diffFactor0, specFactor0 );
        }

        bool isDiffusePath = true;
        TraceOpaqueResult result = ( TraceOpaqueResult )0;
    #if( USE_SHARC_DEBUG != 0 )
        result.radiance = GetSharcDebugColor( geometryProps0, materialProps0 );
    #else
        float hitDistViewZ = Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;
        result = TraceOpaque( geometryProps0, materialProps0, pixelPos, mirrorMatrix, Lpsr, isDiffusePath );

        // Normalize hit distance for REBLUR using the actual PSR surface, not the virtual guide
        if( gDenoiserType != DENOISER_RELAX )
            result.accumulatedHitDist = REBLUR_FrontEnd_GetNormHitDist( result.accumulatedHitDist, hitDistViewZ, gHitDistSettings.xyz, isDiffusePath ? 1.0 : materialProps0.roughness );

        if( USE_SANITIZATION && !NRD_IsValidRadiance( result.radiance ) )
        {
            result.radiance = 0;
            result.accumulatedHitDist = 0;
        }
    #endif

        // Material de-modulation ( remove material factors for NRD )
        float3 demodulatedRadiance = result.radiance / ( isDiffusePath ? diffFactor0 : specFactor0 );

    #if( USE_MOVING_EMISSION_FIX == 1 )
        // Or emissives ( not having lighting in diffuse and specular ) can use a different material ID
        demodulatedRadiance += materialProps0.Lemi / Math::Pi( 2.0 );
    #endif

    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( isDiffusePath && frac( geometryProps0.X ).x < 0.05 )
            demodulatedRadiance = float3( 0, 10, 0 ) * Color::Luminance( demodulatedRadiance );
    #endif

    #if( USE_SIMULATED_FIREFLY_TEST == 1 )
        const float maxFireflyEnergyScaleFactor = 10000.0;
        float fireflyScale = lerp( 1.0 / maxFireflyEnergyScaleFactor, 1.0, Rng::Hash::GetFloat( ) );
        if( isDiffusePath )
            demodulatedRadiance /= fireflyScale;
    #endif

        float4 outRadiance = 0.0;
        float4 outSh = 0.0;

        if( gDenoiserType == DENOISER_RELAX )
        {
        #if( NRD_MODE == SH )
            outRadiance = RELAX_FrontEnd_PackSh( demodulatedRadiance, result.accumulatedHitDist, result.direction, outSh, USE_SANITIZATION );
        #else
            outRadiance = RELAX_FrontEnd_PackRadianceAndHitDist( demodulatedRadiance, result.accumulatedHitDist, USE_SANITIZATION );
        #endif
        }
        else
        {
        #if( NRD_MODE == SH )
            outRadiance = REBLUR_FrontEnd_PackSh( demodulatedRadiance, result.accumulatedHitDist, result.direction, outSh, USE_SANITIZATION );
        #else
            outRadiance = REBLUR_FrontEnd_PackRadianceAndNormHitDist( demodulatedRadiance, result.accumulatedHitDist, USE_SANITIZATION );
        #endif
        }

        // Only the lobe selected at the primary / PSR surface has a sample
        float4 outDiff = isDiffusePath ? outRadiance : 0.0;
        float4 outSpec = isDiffusePath ? 0.0 : outRadiance;
        float4 outDiffSh = isDiffusePath ? outSh : 0.0;
        float4 outSpecSh = isDiffusePath ? 0.0 : outSh;

        WriteResult( pixelPos, outDiff, outSpec, outDiffSh, outSpecSh );
    }
}
