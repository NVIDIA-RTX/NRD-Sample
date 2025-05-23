// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_PostAA, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PreAA, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Validation, t, 2, 1 );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_Final, u, 0, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvWindowSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Upsampling
    float3 upsampled = BicubicFilterNoCorners( gIn_PostAA, gLinearSampler, pixelUv * gOutputSize, gInvOutputSize, 0.66 ).xyz;

    // Noisy input
    float3 input = gIn_PreAA.SampleLevel( gNearestSampler, pixelUv * gRectSize * gInvRenderSize, 0 ).xyz;

    input = ApplyTonemap( input );
    if( gIsSrgb )
        input = Color::ToSrgb( saturate( input ) );

    // Split screen - noisy input / denoised output
    float3 result = pixelUv.x < gSeparator ? input : upsampled;

    // Dithering
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float rnd = Rng::Hash::GetFloat( );
    result += ( rnd - 0.5 ) / ( gIsSrgb ? 256.0 : 1024.0 );

    // Split screen - vertical line
    float verticalLine = saturate( 1.0 - abs( pixelUv.x - gSeparator ) * gWindowSize.x / 3.5 );
    verticalLine = saturate( verticalLine / 0.5 );
    verticalLine *= float( gSeparator != 0.0 );

    const float3 nvColor = float3( 118.0, 185.0, 0.0 ) / 255.0;
    result = lerp( result, nvColor * verticalLine, verticalLine );

    // Validation layer
    if( gValidation )
    {
        float4 validation = gIn_Validation.SampleLevel( gNearestSampler, pixelUv, 0 );
        result = lerp( result, validation.xyz, validation.w );
    }

    // Debug
    #if( USE_TAA_DEBUG == 1 )
        result = gIn_PostAA[ pixelPos ].w;
    #endif

    // Output
    gOut_Final[ pixelPos ] = result;
}