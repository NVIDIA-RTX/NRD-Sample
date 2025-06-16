// © 2022 NVIDIA Corporation

//=============================================================================================
// SETTINGS
//=============================================================================================

// NORMAL - common (non specialized) denoisers
// SH - SH (spherical harmonics or spherical gaussian) denoisers
#define NRD_MODE                            NORMAL // NORMAL, SH (NRD sample recompile required)
#define SIGMA_TRANSLUCENT                   1

// Default = 1
#define USE_IMPORTANCE_SAMPLING             1
#define USE_SHARC_DITHERING                 1 // must be in [0; 1] range
#define USE_TRANSLUCENCY                    1 // translucent foliage
#define USE_MOVING_EMISSION_FIX             1 // fixes a dark tail, left by an animated emissive object

// Default = 0
#define USE_SANITIZATION                    0 // NRD sample is NAN/INF free
#define USE_SIMULATED_MATERIAL_ID_TEST      0 // "material ID" debugging
#define USE_SIMULATED_FIREFLY_TEST          0 // "anti-firefly" debugging
#define USE_DRS_STRESS_TEST                 0 // NRD must not touch GARBAGE data outside of DRS rectangle
#define USE_INF_STRESS_TEST                 0 // NRD must not touch GARBAGE data outside of denoising range
#define USE_ANOTHER_COBALT                  0 // another cobalt variant
#define USE_PUDDLES                         0 // add puddles
#define USE_RANDOMIZED_ROUGHNESS            0 // randomize roughness ( a common case in games )
#define USE_STOCHASTIC_SAMPLING             0 // needed?
#define USE_LOAD                            0 // Load vs SampleLevel
#define USE_SHARC_DEBUG                     0 // 1 - show cache, 2 - show grid (NRD sample recompile required)
#define USE_TAA_DEBUG                       0 // 1 - show weight

//=============================================================================================
// CONSTANTS
//=============================================================================================

// NRD variant
#define NORMAL                              0
#define SH                                  1 // NORMAL + SH (SG) resolve

// Denoiser
#define DENOISER_REBLUR                     0
#define DENOISER_RELAX                      1
#define DENOISER_REFERENCE                  2

// Predefined material override
#define MATERIAL_GYPSUM                     1
#define MATERIAL_COBALT                     2

// Material ID
#define MATERIAL_ID_DEFAULT                 0.0
#define MATERIAL_ID_METAL                   1.0
#define MATERIAL_ID_HAIR                    2.0
#define MATERIAL_ID_SELF_REFLECTION         3.0

// Mip mode
#define MIP_VISIBILITY                      0 // for visibility: emission, shadow and alpha mask
#define MIP_LESS_SHARP                      1 // for normal
#define MIP_SHARP                           2 // for albedo and roughness

// Register spaces ( sets )
#define SET_GLOBAL                          0
#define SET_OTHER                           1
#define SET_RAY_TRACING                     2
#define SET_SHARC                           3
#define SET_PUSH                            4

// Path tracing
#define PT_THROUGHPUT_THRESHOLD             0.001
#define PT_IMPORTANCE_SAMPLES_NUM           16
#define PT_SPEC_LOBE_ENERGY                 0.95 // trimmed to 95%
#define PT_SHADOW_RAY_OFFSET                1.0 // pixels
#define PT_BOUNCE_RAY_OFFSET                0.25 // pixels
#define PT_GLASS_RAY_OFFSET                 0.05 // pixels
#define PT_MAX_FIREFLY_RELATIVE_INTENSITY   20.0 // no more than 20x energy increase in case of probabilistic sampling
#define PT_EVIL_TWIN_LOBE_TOLERANCE         0.005 // normalized %
#define PT_GLASS_MIN_F                      0.05 // adds a bit of stability and bias
#define PT_DELTA_BOUNCES_NUM                8
#define PT_PSR_BOUNCES_NUM                  2
#define PT_RAY_FLAGS                        0

// Spatial HAsh-based Radiance Cache ( SHARC )
#define SHARC_CAPACITY                      ( 1 << 22 )
#define SHARC_SCENE_SCALE                   45.0
#define SHARC_DOWNSCALE                     5
#define SHARC_ANTI_FIREFLY                  true
#define SHARC_STALE_FRAME_NUM_MIN           32 // new version uses 8 by default, old value offers more stability in voxels with low number of samples ( critical for glass )
#define SHARC_SEPARATE_EMISSIVE             1

// Blue noise
#define BLUE_NOISE_SPATIAL_DIM              128 // see StaticTexture::ScramblingRanking
#define BLUE_NOISE_TEMPORAL_DIM             4 // good values: 4-8 for shadows, 8-16 for occlusion, 8-32 for lighting

// Other
#define FP16_MAX                            65504.0
#define INF                                 1e5
#define LINEAR_BLOCK_SIZE                   256
#define FP16_VIEWZ_SCALE                    0.125 // TODO: tuned for meters, needs to be scaled down for cm and mm
#define MAX_MIP_LEVEL                       11.0
#define LEAF_TRANSLUCENCY                   0.25
#define LEAF_THICKNESS                      0.001 // TODO: viewZ dependent?
#define STRAND_THICKNESS                    80e-6f
#define TAA_HISTORY_SHARPNESS               0.66 // sharper ( was 0.5 )
#define TAA_SIGMA_SCALE                     2.0 // allow nano ghosting ( was 1.0 ) // TODO: can negatively affect moving shadows
#define GARBAGE                             sqrt( -1.0 ) // sqrt( -1.0 ) or -log( 0.0 ) or 32768.0

// Instance flags
#define FLAG_FIRST_BIT                      25 // this + number of flags must be <= 32
#define NON_FLAG_MASK                       ( ( 1 << FLAG_FIRST_BIT ) - 1 )

#define FLAG_NON_TRANSPARENT                0x01 // geometry flag: non-transparent
#define FLAG_TRANSPARENT                    0x02 // geometry flag: transparent
#define FLAG_FORCED_EMISSION                0x04 // animated emissive cube
#define FLAG_STATIC                         0x08 // no velocity
#define FLAG_HAIR                           0x20 // hair
#define FLAG_LEAF                           0x40 // leaf

#define GEOMETRY_ALL                        ( FLAG_NON_TRANSPARENT | FLAG_TRANSPARENT )

//===============================================================
// STRUCTS
//===============================================================
// IMPORTANT: sizeof( float3 ) == 16 in C++ code!

struct PrimitiveData
{
    float16_t2 uv0;
    float16_t2 uv1;
    float16_t2 uv2;
    float16_t2 n0;

    float16_t2 n1;
    float16_t2 n2;
    float16_t2 t0;
    float16_t2 t1;

    float16_t2 t2;
    float16_t2 bitangentSign_unused;
    float worldArea;
    float uvArea;
};

struct InstanceData
{
    // For static: mObjectToWorld
    // For rigid dynamic: mWorldToWorldPrev
    // For deformable dynamic: mObjectToWorldPrev
    float4 mOverloadedMatrix0;
    float4 mOverloadedMatrix1;
    float4 mOverloadedMatrix2;

    float4 baseColorAndMetalnessScale;
    float4 emissionAndRoughnessScale;

    uint32_t textureOffsetAndFlags;
    uint32_t primitiveOffset;
    uint32_t unused;

    // TODO: handling object scale embedded into the transformation matrix (assuming uniform scale)
    // TODO: sign represents triangle winding
    float scale;
};

//===============================================================
// RESOURCES
//===============================================================

#include "NRI.hlsl"

NRI_RESOURCE( cbuffer, GlobalConstants, b, 0, SET_GLOBAL )
{
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float4 gSunBasisX;
    float4 gSunBasisY;
    float4 gSunDirection;
    float4 gCameraGlobalPos;
    float4 gCameraGlobalPosPrev;
    float4 gViewDirection;
    float4 gHairBaseColor;
    float2 gHairBetas;
    float2 gWindowSize; // represents DPI handling ( >= gOutputSize )
    float2 gOutputSize; // represents native resolution ( >= gRenderSize )
    float2 gRenderSize; // up to native resolution ( >= gRectSize )
    float2 gRectSize; // dynamic resolution scaling
    float2 gInvWindowSize;
    float2 gInvOutputSize;
    float2 gInvRenderSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gEmissionIntensity;
    float gNearZ;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gPrevFrameConfidence;
    float gUnproject;
    float gAperture;
    float gFocalDistance;
    float gFocalLength;
    float gTAA;
    float gHdrScale;
    float gExposure;
    float gMipBias;
    float gOrthoMode;
    uint32_t gSharcMaxAccumulatedFrameNum;
    uint32_t gDenoiserType;
    uint32_t gDisableShadowsAndEnableImportanceSampling; // TODO: remove - modify GetSunIntensity to return 0 if sun is below horizon
    uint32_t gFrameIndex;
    uint32_t gForcedMaterial;
    uint32_t gUseNormalMap;
    uint32_t gBounceNum;
    uint32_t gResolve;
    uint32_t gValidation;
    uint32_t gSR;
    uint32_t gRR;
    uint32_t gIsSrgb;
};

#if( !defined( __cplusplus ) )

#include "ml.hlsli"
#include "NRD.hlsli"

NRI_RESOURCE( SamplerState, gLinearMipmapLinearSampler, s, 0, SET_GLOBAL );
NRI_RESOURCE( SamplerState, gLinearMipmapNearestSampler, s, 1, SET_GLOBAL );
NRI_RESOURCE( SamplerState, gNearestMipmapNearestSampler, s, 2, SET_GLOBAL );

#define gLinearSampler gLinearMipmapLinearSampler
#define gNearestSampler gNearestMipmapNearestSampler

//=============================================================================================
// MISC
//=============================================================================================

// For SHARC
float3 GetGlobalPos( float3 X )
{
    return gCameraGlobalPos.xyz * gCameraGlobalPos.w + X;
}

// Taken out from NRD
float GetSpecMagicCurve( float roughness )
{
    float f = 1.0 - exp2( -200.0 * roughness * roughness );
    f *= Math::Pow01( roughness, 0.25 );

    return f;
}

float ApplyThinLensEquation( float hitDist, float curvature )
{
    return hitDist / ( 2.0 * curvature * hitDist + 1.0 );
}

float3 GetMotion( float3 X, float3 Xprev )
{
    float3 motion = Xprev - X;

    float viewZ = Geometry::AffineTransform( gWorldToView, X ).z;
    float2 sampleUv = Geometry::GetScreenUv( gWorldToClip, X );

    float viewZprev = Geometry::AffineTransform( gWorldToViewPrev, Xprev ).z;
    float2 sampleUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );

    // IMPORTANT: scaling to "pixel" unit significantly improves utilization of FP16
    motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;

    // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues caused by FP16 rounding negative effects
    motion.z = viewZprev - viewZ;

    return motion;
}

float3 ApplyExposure( float3 Lsum )
{
    Lsum *= gExposure;

    return Lsum;
}

float3 ApplyTonemap( float3 Lsum )
{
    Lsum = gHdrScale * Color::HdrToLinear_Uncharted( Lsum );

    return Lsum;
}

float4 BicubicFilterNoCorners( Texture2D<float4> tex, SamplerState samp, float2 samplePos, float2 invResourceSize, float sharpness )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5;
    float2 f = saturate( samplePos - centerPos );
    float2 f2 = f * f;
    float2 f3 = f * f2;
    float2 w0 = -sharpness * f3 + 2.0 * sharpness * f2 - sharpness * f;
    float2 w1 = ( 2.0 - sharpness ) * f3 - ( 3.0 - sharpness ) * f2 + 1.0;
    float2 w2 = -( 2.0 - sharpness ) * f3 + ( 3.0 - 2.0 * sharpness ) * f2 + sharpness * f;
    float2 w3 = sharpness * f3 - sharpness * f2;
    float2 wl2 = w1 + w2;
    float2 tc2 = invResourceSize * ( centerPos + w2 * Math::PositiveRcp( wl2 ) );
    float2 tc0 = invResourceSize * ( centerPos - 1.0 );
    float2 tc3 = invResourceSize * ( centerPos + 2.0 );

    float w = wl2.x * w0.y;
    float4 color = tex.SampleLevel( samp, float2( tc2.x, tc0.y ), 0 ) * w;
    float sum = w;

    w = w0.x  * wl2.y;
    color += tex.SampleLevel( samp, float2( tc0.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * wl2.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc2.y ), 0 ) * w;
    sum += w;

    w = w3.x * wl2.y;
    color += tex.SampleLevel( samp, float2( tc3.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * w3.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc3.y ), 0 ) * w;
    sum += w;

    color *= Math::PositiveRcp( sum );

    return color;
}

void GetCameraRay( out float3 origin, out float3 direction, float2 sampleUv )
{
    // https://www.slideshare.net/TiagoAlexSousa/graphics-gems-from-cryengine-3-siggraph-2013 ( slides 23+ )

    // Pinhole ray
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );
    direction = normalize( Xv );

    // Distorted ray
    float2 rnd = Rng::Hash::GetFloat2( );
    rnd = ImportanceSampling::Cosine::GetRay( rnd ).xy;
    Xv.xy += rnd * gAperture;

    float3 Fv = direction * gFocalDistance; // z-plane
    #if 0
        Fv /= dot( vForward, direction ); // radius
    #endif

    origin = Geometry::AffineTransform( gViewToWorld, Xv );
    direction = gOrthoMode == 0.0 ? normalize( Geometry::RotateVector( gViewToWorld, Fv - Xv ) ) : -gViewDirection.xyz;
}

float GetCircleOfConfusion( float distance ) // diameter
{
    float F = gFocalLength; // focal lenght ( deducted from FOV )
    float A = gAperture; // aperture diameter
    float P = gFocalDistance; // focal distance

    return gOrthoMode == 0.0 ? abs( A * ( F * ( P - distance ) ) / ( distance * ( P - F ) ) ) : A;
}

//=============================================================================================
// VERY SIMPLE SKY MODEL
//=============================================================================================

#define SKY_INTENSITY 1.0
#define SUN_INTENSITY 10.0

float3 GetSunIntensity( float3 v )
{
    float b = dot( v, gSunDirection.xyz );
    float d = length( v - gSunDirection.xyz * b );

    float glow = saturate( 1.015 - d );
    glow *= b * 0.5 + 0.5;
    glow *= 0.6;

    float a = Math::Sqrt01( 1.0 - b * b ) / b;
    float sun = 1.0 - Math::SmoothStep( gTanSunAngularRadius * 0.9, gTanSunAngularRadius * 1.66 + 0.01, a );
    sun *= float( b > 0.0 );
    sun *= 1.0 - Math::Pow01( 1.0 - v.z, 4.85 );
    sun *= Math::SmoothStep( 0.0, 0.1, gSunDirection.z );
    sun += glow;

    float3 sunColor = lerp( float3( 1.0, 0.6, 0.3 ), float3( 1.0, 0.9, 0.7 ), Math::Sqrt01( gSunDirection.z ) );
    sunColor *= sun;

    sunColor *= Math::SmoothStep( -0.01, 0.05, gSunDirection.z );

    return Color::FromGamma( sunColor ) * SUN_INTENSITY;
}

float3 GetSkyIntensity( float3 v )
{
    float atmosphere = sqrt( 1.0 - saturate( v.z ) );

    float scatter = pow( saturate( gSunDirection.z ), 1.0 / 15.0 );
    scatter = 1.0 - clamp( scatter, 0.8, 1.0 );

    float3 scatterColor = lerp( float3( 1.0, 1.0, 1.0 ), float3( 1.0, 0.3, 0.0 ) * 1.5, scatter );
    float3 skyColor = lerp( float3( 0.2, 0.4, 0.8 ), float3( scatterColor ), atmosphere / 1.3 );
    skyColor *= saturate( 1.0 + gSunDirection.z );

    float ground = 0.5 + 0.5 * Math::SmoothStep( -1.0, 0.0, v.z );
    skyColor *= ground;

    return Color::FromGamma( skyColor ) * SKY_INTENSITY + GetSunIntensity( v );
}

#endif
