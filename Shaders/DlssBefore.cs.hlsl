// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 1, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_Spec, t, 2, SET_OTHER );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float>, gInOut_ViewZ, u, 0, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_DiffAlbedo, u, 1, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_SpecAlbedo, u, 2, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float>, gOut_SpecHitDistance, u, 3, SET_OTHER );
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 4, SET_OTHER );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvRenderSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

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

        float3 X = Geometry::AffineTransform( gViewToWorld, Xv );
        float3 V = gOrthoMode == 0 ? Geometry::RotateVector( gViewToWorld, normalize( Xv ) ) : gViewDirection.xyz;

        float NoV = abs( dot( N, V ) );
        float3 Fenv = BRDF::EnvironmentTerm_Rtg( Rf0, NoV, roughness );

        float scaleHitDistance = gDenoiserType == DENOISER_RELAX ? 1.0 : _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistParams, roughness );
        float specHitDistance = gIn_Spec[ pixelPos ].w * scaleHitDistance;

        bool isSky = abs( viewZ ) == INF;

        gOut_DiffAlbedo[ pixelPos ] = isSky ? 0.0 : albedo * ( 1.0 - Fenv ); // NGX doesn't support sRGB, manual linearization needed
        gOut_SpecAlbedo[ pixelPos ] = isSky ? 0.0 : Fenv;
        gOut_SpecHitDistance[ pixelPos ] = isSky ? 0.0 : specHitDistance;
        gOut_Normal_Roughness[ pixelPos ] = float4( N, roughness ); // NGX supports only this encoding
    }
}