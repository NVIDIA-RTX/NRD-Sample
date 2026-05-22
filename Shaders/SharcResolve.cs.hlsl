// © 2024 NVIDIA Corporation

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    HashGridParameters hashGridParameters;
    hashGridParameters.cameraPosition = gCameraGlobalPos.xyz;
    hashGridParameters.sceneScale = SHARC_SCENE_SCALE;
    hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    HashGridData hashGridData;
    hashGridData.capacity = SHARC_CAPACITY;
    hashGridData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

    SharcParameters sharcParams;
    sharcParams.hashGridParameters = hashGridParameters;
    sharcParams.hashGridData = hashGridData;
    sharcParams.radianceScale = SHARC_RADIANCE_SCALE;
    sharcParams.accumulationBuffer = gInOut_SharcAccumulated;
    sharcParams.resolvedBuffer = gInOut_SharcResolved;

    SharcResolveParameters sharcResolveParameters;
    sharcResolveParameters.cameraPositionPrev = gCameraGlobalPosPrev.xyz;
    sharcResolveParameters.accumulationFrameNum = gMaxAccumulatedFrameNum;
    sharcResolveParameters.responsiveFrameNum = SHARC_RESPONSIVE_FRAME_NUM;
    sharcResolveParameters.staleFrameNumMax = SHARC_STALE_FRAME_NUM_MIN;
    sharcResolveParameters.frameIndex = gFrameIndex;

    SharcResolveEntry( threadIndex, sharcParams, sharcResolveParameters );
}
