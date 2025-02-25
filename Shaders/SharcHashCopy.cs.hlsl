// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

#include "SharcCommon.h"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    HashMapData hashMapData;
    hashMapData.capacity = SHARC_CAPACITY;
    hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

    SharcCopyHashEntry( threadIndex, hashMapData, gInOut_SharcHashCopyOffsetBuffer );
}
