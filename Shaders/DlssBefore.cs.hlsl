// © 2022 NVIDIA Corporation

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 1, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_Spec, t, 2, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_Mv, t, 3, SET_OTHER );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float>, gInOut_ViewZ, u, 0, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_DiffAlbedo, u, 1, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_SpecAlbedo, u, 2, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float>, gOut_SpecHitDistance, u, 3, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 4, SET_OTHER );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Mv, u, 5, SET_OTHER );

#define BORDER      2
#define GROUP_X     16
#define GROUP_Y     16
#define BUFFER_X    ( GROUP_X + BORDER * 2 )
#define BUFFER_Y    ( GROUP_Y + BORDER * 2 )

groupshared float4 s_Mv[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );

    s_Mv[ sharedPos.y ][ sharedPos.x ] = gIn_Mv[ globalPos ];
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadID, uint2 threadPos : SV_GroupThreadID, uint threadIndex : SV_GroupIndex )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvRenderSize;

    PRELOAD_INTO_SMEM;

    if( any( pixelPos >= uint2( gRectSize ) ) )
        return;

    float4 motion = s_Mv[ threadPos.y + BORDER ][ threadPos.x + BORDER ];
    float2 glassMotionSum = 0;
    float glassMotionWeightSum = 0;

    [unroll]
    for( int dy = -BORDER; dy <= BORDER; dy++ )
    {
        [unroll]
        for( int dx = -BORDER; dx <= BORDER; dx++ )
        {
            float4 sampleMotion = s_Mv[ threadPos.y + BORDER + dy ][ threadPos.x + BORDER + dx ];
            float r2 = Math::LengthSquared( float2( dx, dy ) );
            float w = exp( -TAA_BLUR_FALLOFF * r2 );

            // Smooth noisy glass MV
            w *= sampleMotion.z;
            glassMotionSum += sampleMotion.xy * w;
            glassMotionWeightSum += w;
        }
    }

    if( glassMotionWeightSum > 0.0 )
        motion.xy = glassMotionSum / glassMotionWeightSum;

    gOut_Mv[ pixelPos ] = motion;

    float viewZ = gInOut_ViewZ[ pixelPos ];
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZ, gOrthoMode );

    // SR specific
    if( gSR )
    {
        float4 clipPos = Geometry::ProjectiveTransform( gViewToClip, Xv );

        gInOut_ViewZ[ pixelPos ] = clipPos.z / clipPos.w; // SR doesn't support linear viewZ
    }

    // RR specific
    if( gRR )
    {
        float normMaterialID;
        float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ], normMaterialID );
        float3 N = normalAndRoughness.xyz;
        float roughness = normalAndRoughness.w;

        float3 albedo, Rf0;
        float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0 );

        float3 V = gOrthoMode == 0 ? Geometry::RotateVector( gViewToWorld, normalize( Xv ) ) : gViewDirection.xyz;

        float NoV = abs( dot( N, V ) );
        float3 Fenv = BRDF::EnvironmentTerm_Rtg( Rf0, NoV, roughness );

        float scaleHitDistance = gDenoiserType == DENOISER_RELAX ? 1.0 : _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings.xyz, roughness );
        float specHitDistance = gIn_Spec[ pixelPos ].w * scaleHitDistance;

        bool isSky = abs( viewZ ) == INF;

        gOut_DiffAlbedo[ pixelPos ] = isSky ? 0.0 : albedo * ( 1.0 - Fenv ); // NGX doesn't support sRGB, manual linearization needed
        gOut_SpecAlbedo[ pixelPos ] = isSky ? 0.0 : Fenv;
        gOut_SpecHitDistance[ pixelPos ] = isSky ? 0.0 : specHitDistance;
        gOut_Normal_Roughness[ pixelPos ] = float4( N, roughness ); // NGX supports only this encoding
    }
}