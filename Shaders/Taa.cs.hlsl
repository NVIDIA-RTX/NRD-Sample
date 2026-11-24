// © 2022 NVIDIA Corporation

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Mv, t, 0, SET_OTHER );
NRI_RESOURCE( Texture2D<float3>, gIn_Composed, t, 1, SET_OTHER );
NRI_RESOURCE( Texture2D<float4>, gIn_History, t, 2, SET_OTHER );

NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Result, u, 0, SET_OTHER );

#define BORDER          2
#define GROUP_X         16
#define GROUP_Y         16
#define BUFFER_X        ( GROUP_X + BORDER * 2 )
#define BUFFER_Y        ( GROUP_Y + BORDER * 2 )

static const float TAA_WEIGHTS[ 3 ] = { 1.0, exp( -TAA_BLUR_FALLOFF ), exp( -4.0 * TAA_BLUR_FALLOFF ) };

groupshared float3 s_Color[ BUFFER_Y ][ BUFFER_X ];
groupshared float4 s_Mv[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );

    s_Color[ sharedPos.y ][ sharedPos.x ] = ApplyTonemap( gIn_Composed[ globalPos ] );
    s_Mv[ sharedPos.y ][ sharedPos.x ] = gIn_Mv[ globalPos ]; // XY motion, glass mask, viewZ with "want 5x5 mask"
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

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( int2 threadPos : SV_GroupThreadID, int2 pixelPos : SV_DispatchThreadID, uint threadIndex : SV_GroupIndex )
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
    float2 glassMotionSum = 0;
    float glassMotionWeightSum = 0;

    float4 centerMv = s_Mv[ threadPos.y + BORDER ][threadPos.x + BORDER ];
    float minViewZ = abs( centerMv.w );
    int2 offseti = int2( BORDER, BORDER );
    bool want5x5 = centerMv.w < 0.0; // 5x5 is needed for hair ( super thin ) and glass ( noisy ), also it's safe to use it for sky to get better edges

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
            float4 mv = s_Mv[ smemPos.y ][ smemPos.x ];
            float viewZ = abs( mv.w );

            // Keep candidate MV dilation at 3x3 even when color statistics use 5x5
            if( dx == BORDER && dy == BORDER )
                input = c;
            else if( viewZ < minViewZ && all( abs( offset - BORDER ) <= 1 ) )
            {
                minViewZ = viewZ;
                offseti = offset;
            }

            float w = TAA_WEIGHTS[ abs( dx - BORDER ) ] * TAA_WEIGHTS[ abs( dy - BORDER ) ];

            // Color moments
            m1 += c * w;
            m2 += c * c * w;
            sum += w;

            // Smooth noisy glass MV
            w *= mv.z;
            glassMotionSum += mv.xy * w;
            glassMotionWeightSum += w;
        }
    }

    m1 /= sum;
    m2 /= sum;

    float3 sigma = sqrt( abs( m2 - m1 * m1 ) ) * TAA_SIGMA_SCALE;

    // Previous pixel position
    float2 mv = s_Mv[ threadPos.y + offseti.y ][ threadPos.x + offseti.x ].xy;
    if( glassMotionWeightSum > 0.0 )
        mv = glassMotionSum / glassMotionWeightSum;

    float2 pixelUvPrev = pixelUv + mv * gInvRectSize;

    // History
    float2 pixelPosPrev = saturate( pixelUvPrev ) * gRectSizePrev;
    float4 history = BicubicFilterNoCorners( gIn_History, gLinearClamp, pixelPosPrev, gInvRenderSize, TAA_HISTORY_SHARPNESS );

    history.xyz = max( history.xyz, 0.0 ); // yes, not "saturate"

    // Update mix rate
    float mixRate = saturate( history.w );
    mixRate /= 1.0 + mixRate;

    // Disocclusion #1
    bool isInScreen = all( saturate( pixelUvPrev ) == pixelUvPrev );
    mixRate = !isInScreen ? 1.0 : mixRate;

    // Disocclusion #2
    float3 clampedHistory = Color::ClampAabb( m1, sigma, history.xyz );
    float3 delta = Color::RgbToYCoCg( history.xyz - clampedHistory ) / gHdrScale;
    float error = length( delta * float3( 1.0, 0.5, 0.5 ) );
    float response = saturate( ( error - 0.01 ) * 4.0 );
    mixRate = max( mixRate, response );

    // TODO: anti-flickering, compatible with "mixRate"?

    // Final mix
    float3 result = lerp( clampedHistory, input, max( mixRate, gTAA ) );

    // Output
    gOut_Result[ pixelPos ] = float4( result, mixRate );
}
