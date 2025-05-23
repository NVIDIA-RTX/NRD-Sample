// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Mv, t, 0, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Composed, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_History, t, 2, 1 );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Result, u, 0, 1 );

#define BORDER          2
#define GROUP_X         16
#define GROUP_Y         16
#define BUFFER_X        ( GROUP_X + BORDER * 2 )
#define BUFFER_Y        ( GROUP_Y + BORDER * 2 )

#define PRELOAD_INTO_SMEM \
    int2 groupBase = pixelPos - threadPos - BORDER; \
    uint stageNum = ( BUFFER_X * BUFFER_Y + GROUP_X * GROUP_Y - 1 ) / ( GROUP_X * GROUP_Y ); \
    [unroll] \
    for( uint stage = 0; stage < stageNum; stage++ ) \
    { \
        uint virtualIndex = threadIndex + stage * GROUP_X * GROUP_Y; \
        uint2 newId = uint2( virtualIndex % BUFFER_X, virtualIndex / BUFFER_X ); \
        if( stage == 0 || virtualIndex < BUFFER_X * BUFFER_Y ) \
            Preload( newId, groupBase + newId ); \
    } \
    GroupMemoryBarrierWithGroupSync( )

groupshared float3 s_Color[ BUFFER_Y ][ BUFFER_X ];
groupshared float3 s_Mv[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );

    s_Color[ sharedPos.y ][ sharedPos.x ] = ApplyTonemap( gIn_Composed[ globalPos ] );
    s_Mv[ sharedPos.y ][ sharedPos.x ] = gIn_Mv[ globalPos ].xyw; // dZ is not needed
}

// TODO: move to ml?
// https://www.cs.rit.edu/~ncs/color/t_convert.html
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
float3 XyzToLab( float3 x )
{
    x /= float3( 95.0489, 100.0, 108.8840 );
    x = lerp( 7.787 * x + 16.0 / 116.0, pow( x, 0.333333 ), x > 0.008856 );

    float l = lerp( 903.3 * x.y, 116.0 * pow( x.y, 0.333333 ) - 16.0, x.y > 0.008856 );
    float a = 500.0 * ( x.x - x.y );
    float b = 200.0 * ( x.y - x.z );

    return float3( l, a, b );
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( int2 threadPos : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    PRELOAD_INTO_SMEM;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Neighborhood
    float sum = 0;
    float3 m1 = 0;
    float3 m2 = 0;
    float3 input = 0;

    float3 centerMv = s_Mv[ threadPos.y + BORDER ][threadPos.x + BORDER ];
    float minViewZ = abs( centerMv.z );
    int2 offseti = int2( BORDER, BORDER );

    bool want5x5 = centerMv.z < 0.0; // 5x5 is needed for hair ( super thin ) and glass ( noisy ), also it's safe to use it for sky to get better edges

    [unroll]
    for( int dy = 0; dy <= BORDER * 2; dy++ )
    {
        [unroll]
        for( int dx = 0; dx <= BORDER * 2; dx++ )
        {
            if( !want5x5 && ( dx == 0 || dx == BORDER * 2 || dy == 0 || dy == BORDER * 2 ) )
                continue;

            int2 offset = int2( dx, dy );
            int2 smemPos = threadPos + offset;

            float3 c = s_Color[ smemPos.y ][ smemPos.x ];
            float3 mv = s_Mv[ smemPos.y ][ smemPos.x ].xyz;
            float viewZ = abs( mv.z );

            if( dx == BORDER && dy == BORDER )
                input = c;
            else if( viewZ < minViewZ )
            {
                minViewZ = viewZ;
                offseti = offset;
            }

            float r2 = Math::LengthSquared( offset / BORDER - 1.0 );
            float w = exp( -r2 );

            m1 += c * w;
            m2 += c * c * w;
            sum += w;
        }
    }

    m1 /= sum;
    m2 /= sum;

    float3 sigma = sqrt( abs( m2 - m1 * m1 ) ) * TAA_SIGMA_SCALE;

    // Previous pixel position
    float3 mv = s_Mv[ threadPos.y + offseti.y ][ threadPos.x + offseti.x ].xyz * float3( gInvRectSize.xy, 1.0 );
    float2 pixelUvPrev = pixelUv + mv.xy;

    // History
    float2 pixelPosPrev = saturate( pixelUvPrev ) * gRectSizePrev;
    float4 history = BicubicFilterNoCorners( gIn_History, gLinearSampler, pixelPosPrev, gInvRenderSize, TAA_HISTORY_SHARPNESS );

    history.xyz = max( history.xyz, 0.0 ); // yes, not "saturate"

    // Remove transfer
    if( gIsSrgb )
        history.xyz = Color::FromSrgb( history.xyz );

    // Update mix rate
    float mixRate = saturate( history.w );
    mixRate /= 1.0 + mixRate;

    // Disocclusion #1
    bool isInScreen = all( saturate( pixelUvPrev ) == pixelUvPrev );
    mixRate = !isInScreen ? 1.0 : mixRate;

    // Disocclusion #2
    float3 clampedHistory = Color::ClampAabb( m1, sigma, history.xyz );
    #if 0 // good enough?
        float diff = length( clampedHistory - history.xyz );
        diff = Math::Pow01( diff, 1.2 );
    #else
        float3 a = XyzToLab( Color::RgbToXyz( clampedHistory ) );
        float3 b = XyzToLab( Color::RgbToXyz( history.xyz ) );

        const float JND = 2.3; // just noticable difference
        float diff = length( a - b ) / ( JND * 3.0 );
    #endif
    mixRate += diff;

    // Clamp mix rate
    mixRate = saturate( mixRate );

    // TODO: anti-flickering, compatible with "mixRate"?

    // Final mix
    float3 result = lerp( clampedHistory, input, max( mixRate, gTAA ) );

    // Apply transfer
    if( gIsSrgb )
        result = Color::ToSrgb( result );

    // Output
    gOut_Result[ pixelPos ] = float4( result, mixRate );
}
