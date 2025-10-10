// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    HashGridParameters hashGridParams;
    hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
    hashGridParams.sceneScale = SHARC_SCENE_SCALE;
    hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

    HashMapData hashMapData;
    hashMapData.capacity = SHARC_CAPACITY;
    hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

    SharcParameters sharcParams;
    sharcParams.gridParameters = hashGridParams;
    sharcParams.hashMapData = hashMapData;
    sharcParams.radianceScale = SHARC_RADIANCE_SCALE;
    sharcParams.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;
    sharcParams.accumulationBuffer = gInOut_SharcAccumulated;
    sharcParams.resolvedBuffer = gInOut_SharcResolved;

    SharcResolveParameters sharcResolveParameters;
    sharcResolveParameters.cameraPositionPrev = gCameraGlobalPosPrev.xyz;
    sharcResolveParameters.accumulationFrameNum = gSharcMaxAccumulatedFrameNum;
    sharcResolveParameters.staleFrameNumMax = SHARC_STALE_FRAME_NUM_MIN;
    sharcResolveParameters.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;

    SharcResolveEntry( threadIndex, sharcParams, sharcResolveParameters );
}
