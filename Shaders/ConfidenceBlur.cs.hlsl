// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"

struct PushConstants
{
    int step;
};

NRI_ROOT_CONSTANTS( PushConstants, g_PushConstants, 1, SET_ROOT );

// Inputs
NRI_RESOURCE( Texture2D<float4>, gIn_Gradient, t, 0, SET_OTHER );

// Outputs
NRI_FORMAT("unknown") NRI_RESOURCE( RWTexture2D<float4>, gOut_Gradient, u, 0, SET_OTHER );

float2 GetGeometryWeightParams( float3 Nv, float3 Xv )
{
    const float planeDistSensitivity = 0.02;

    float frustumSize = gRectSize.x * gUnproject * lerp( abs( Xv.z ), 1.0, abs( gOrthoMode ) );
    float norm = planeDistSensitivity * frustumSize;
    float a = 1.0 / norm;
    float b = dot( Nv, Xv ) * a;

    return float2( a, -b );
}

#define ComputeNonExponentialWeight( x, px, py ) \
    Math::SmoothStep( 1.0, 0.0, abs( ( x ) * px + py ) )

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( pixelPos + 0.5 ) * gInvSharcRenderSize;

    float4 data0 = gIn_Gradient[ pixelPos ];
    float z0 = data0.w / FP16_VIEWZ_SCALE;
    bool isLastPass = g_PushConstants.step == 5;

    if( abs( z0 ) > INF )
    {
        gOut_Gradient[ pixelPos ] = float4( isLastPass, data0.yzw );
        return;
    }

    float3 Xv0 = Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, z0, gOrthoMode );
    float3 Nv0 = Packing::DecodeUnitVector( data0.yz );
    float2 geometryWeightParams = GetGeometryWeightParams( Nv0, Xv0 );
    float gradient = data0.x;
    float sum = 1.0;

    [unroll]
    for( int i = -2; i <= 2; i++ )
    {
        [unroll]
        for( int j = -2; j <= 2; j++ )
        {
            if( i == 0 && j == 0 )
                continue;

            int2 pos = pixelPos + int2( i, j ) * g_PushConstants.step;
            float2 uv = ( pos + 0.5 ) * gInvSharcRenderSize;

            float4 data = gIn_Gradient.SampleLevel( gNearestClamp, uv, 0 );

            // Gaussian weight
            float d = length( int2( i, j ) ) / 2.0;
            float w = exp( -2.0 * d * d );

            // Plane distance weight
            float z = data.w / FP16_VIEWZ_SCALE;
            float3 Xv = Geometry::ReconstructViewPosition( uv, gCameraFrustum, z, gOrthoMode );
            float NoX = dot( Nv0, Xv );
            w *= ComputeNonExponentialWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );

            // Normal weight
            float3 Nv = Packing::DecodeUnitVector( data.yz );
            float NoN = saturate( dot( Nv0, Nv ) );
            w *= NoN * NoN;

            // Accumulate
            gradient += data.x * w;
            sum += w;
        }
    }

    gradient /= sum;

    if( isLastPass )
    {
        // Last pass converts "gradient" to "history confidence"
        // ( Optional ) apply remapping, dithering...
        gradient = Color::HdrToLinear_Uncharted( gradient * gExposure ).x; // or normalize to the blurred final image or SHARC cache
        gradient = 1.0 - Color::ToSrgb( gradient ).x;
    }

    gOut_Gradient[ pixelPos ] = float4( gradient, data0.yzw );
}
