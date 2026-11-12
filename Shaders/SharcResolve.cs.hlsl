// © 2024 NVIDIA Corporation

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    SharcParameters sharcParams = GetSharcParameters( );

    SharcResolveParameters sharcResolveParameters;
    sharcResolveParameters.cameraPositionPrev = gCameraGlobalPosPrev.xyz;
    sharcResolveParameters.accumulationFrameNum = gMaxAccumulatedFrameNum;
    sharcResolveParameters.responsiveFrameNum = SHARC_RESPONSIVE_FRAME_NUM;
    sharcResolveParameters.staleFrameNumMax = SHARC_STALE_FRAME_NUM_MIN;
    sharcResolveParameters.frameIndex = gFrameIndex;

    SharcResolveEntry( threadIndex, sharcParams, sharcResolveParameters );
}
