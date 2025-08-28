// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float3>, gOut_Image, u, 0, SET_OTHER );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvOutputSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    float3 color = gOut_Image[ pixelPos ];

    color = ApplyTonemap( color );
    if( gIsSrgb )
        color = Color::ToSrgb( saturate( color ) );

    // Output
    gOut_Image[ pixelPos ] = color;
}
