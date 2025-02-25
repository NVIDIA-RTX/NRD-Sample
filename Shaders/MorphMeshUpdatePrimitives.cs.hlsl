// © 2022 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( StructuredBuffer<uint>, gIn_MorphMeshIndices, t, 0, SET_MORPH );
NRI_RESOURCE( StructuredBuffer<float16_t4>, gIn_MorphedPositions, t, 1, SET_MORPH );
NRI_RESOURCE( StructuredBuffer<MorphedAttributes>, gIn_MorphedAttributes, t, 2, SET_MORPH );

// Outputs
NRI_RESOURCE( RWStructuredBuffer<PrimitiveData>, gInOut_PrimitiveData, u, 0, SET_MORPH );
NRI_RESOURCE( RWStructuredBuffer<MorphedPrimitivePrevPositions>, gOut_PrimitivePrevPositions, u, 1, SET_MORPH );

float ComputeWorldArea( float3 p0, float3 p1, float3 p2 )
{
    float3 edge20 = p2 - p0;
    float3 edge10 = p1 - p0;

    return max( length( cross( edge20, edge10 ) ), 1e-9f );
}

float ComputeCurvature( float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, float worldArea )
{
    float curvature10 = abs( dot( n1 - n0, p1 - p0 ) ) / Math::LengthSquared( p1 - p0 );
    float curvature21 = abs( dot( n2 - n1, p2 - p1 ) ) / Math::LengthSquared( p2 - p1 );
    float curvature02 = abs( dot( n0 - n2, p0 - p2 ) ) / Math::LengthSquared( p0 - p2 );

    float curvature = max( max( curvature10, curvature21 ), curvature02 );

    // Stage 2
    float invTriArea = 1.0f / ( worldArea * 0.5f );
    curvature10 = Math::Sqrt( Math::LengthSquared( n1 - n0 ) * invTriArea );
    curvature21 = Math::Sqrt( Math::LengthSquared( n2 - n1 ) * invTriArea );
    curvature02 = Math::Sqrt( Math::LengthSquared( n0 - n2 ) * invTriArea );

    curvature = max( curvature, curvature10 );
    curvature = max( curvature, curvature21 );
    curvature = max( curvature, curvature02 );

    return curvature;
}

float ComputeWorldToUvUnits( float2 uv0, float2 uv1, float2 uv2, float worldArea )
{
    float3 uvEdge20 = float3( uv2 - uv0, 0.0f );
    float3 uvEdge10 = float3( uv1 - uv0, 0.0f );
    float uvArea = length( cross( uvEdge20, uvEdge10 ) );

    return uvArea == 0 ? 1.0f : Math::Sqrt( uvArea / worldArea );
}

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint primitiveIndex : SV_DispatchThreadId )
{
    if( primitiveIndex >= gNumPrimitives )
        return;

    uint i0 = gIn_MorphMeshIndices[ gIndexOffset + primitiveIndex * 3 + 0 ];
    uint i1 = gIn_MorphMeshIndices[ gIndexOffset + primitiveIndex * 3 + 1 ];
    uint i2 = gIn_MorphMeshIndices[ gIndexOffset + primitiveIndex * 3 + 2 ];

    MorphedAttributes a0 = gIn_MorphedAttributes[ gAttributesOffset + i0 ];
    MorphedAttributes a1 = gIn_MorphedAttributes[ gAttributesOffset + i1 ];
    MorphedAttributes a2 = gIn_MorphedAttributes[ gAttributesOffset + i2 ];

    PrimitiveData result = gInOut_PrimitiveData[ gPrimitiveOffset + primitiveIndex ];

    // TODO: not needed for hair because in any case the curvature defined ONLY by hair thickness will be found (not very useful)
    #if 0
        float3 p0 = gIn_MorphedPositions[ gPositionFrameOffsets.x + i0 ].xyz;
        float3 p1 = gIn_MorphedPositions[ gPositionFrameOffsets.x + i1 ].xyz;
        float3 p2 = gIn_MorphedPositions[ gPositionFrameOffsets.x + i2 ].xyz;

        float2 uv0 = ( float2 )result.uv0;
        float2 uv1 = ( float2 )result.uv1;
        float2 uv2 = ( float2 )result.uv2;

        float3 n0 = Packing::DecodeUnitVector( ( float2 )result.n0, true, true );
        float3 n1 = Packing::DecodeUnitVector( ( float2 )result.n1, true, true );
        float3 n2 = Packing::DecodeUnitVector( ( float2 )result.n2, true, true );

        float worldArea = ComputeWorldArea( p0, p1, p2 );
        result.worldToUvUnits = ComputeWorldToUvUnits( uv0, uv1, uv2, worldArea );

        float16_t curvature = ( float16_t )ComputeCurvature( p0, p1, p2, n0, n1, n2, worldArea );
        result.curvature0_curvature1 = curvature.xx;
        result.curvature2_bitangentSign.x = curvature;
    #endif

    result.n0 = a0.N;
    result.n1 = a1.N;
    result.n2 = a2.N;

    result.t0 = a0.T;
    result.t1 = a1.T;
    result.t2 = a2.T;

    gInOut_PrimitiveData[ gPrimitiveOffset + primitiveIndex ] = result;

    uint index = gMorphedPrimitiveOffset + primitiveIndex;
    gOut_PrimitivePrevPositions[ index ].pos0 = gIn_MorphedPositions[ gPositionFrameOffsets.y + i0 ];
    gOut_PrimitivePrevPositions[ index ].pos1 = gIn_MorphedPositions[ gPositionFrameOffsets.y + i1 ];
    gOut_PrimitivePrevPositions[ index ].pos2 = gIn_MorphedPositions[ gPositionFrameOffsets.y + i2 ];
}
