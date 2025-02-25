// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    gInOut_SharcVoxelDataBuffer[ threadIndex ] = 0;
}
