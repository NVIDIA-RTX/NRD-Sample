// © 2022 NVIDIA Corporation

#include "NRIFramework.h"

// NRD and NRI-based integration
#include "NRD.h"
#include "NRDIntegration.hpp"

#ifdef _WIN32
#    undef APIENTRY
#    include <windows.h>
#endif

//=================================================================================
// Settings
//=================================================================================

// NRD mode and other shared settings are here
#include "../Shaders/Include/Shared.hlsli"

constexpr uint32_t MAX_ANIMATED_INSTANCE_NUM = 512;
constexpr auto BLAS_RIGID_MESH_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE | nri::AccelerationStructureBits::ALLOW_COMPACTION;
constexpr auto BLAS_DEFORMABLE_MESH_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_BUILD | nri::AccelerationStructureBits::ALLOW_UPDATE;
constexpr auto TLAS_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE;
constexpr float ACCUMULATION_TIME = 0.5f;      // seconds
constexpr float NEAR_Z = 0.001f;               // m
constexpr float GLASS_THICKNESS = 0.002f;      // m
constexpr float CAMERA_BACKWARD_OFFSET = 0.0f; // m, 3rd person camera offset
constexpr float NIS_SHARPNESS = 0.2f;
constexpr bool CAMERA_RELATIVE = true;
constexpr bool ALLOW_BLAS_MERGING = true;
constexpr bool ALLOW_HDR = false;                   // use "WIN + ALT + B" to switch HDR mode
constexpr bool USE_LOW_PRECISION_FP_FORMATS = true; // saves a bit of memory and performance
constexpr bool USE_DLSS_TNN = false;                // replace CNN (legacy) with TNN (better)
constexpr nri::UpscalerType upscalerType = nri::UpscalerType::DLSR;
constexpr bool NRD_ALLOW_DESCRIPTOR_CACHING = true;
constexpr bool NRD_PROMOTE_FLOAT16_TO_32 = false;
constexpr bool NRD_DEMOTE_FLOAT32_TO_16 = false;
constexpr bool NRD_RESTORE_INITIAL_STATE = false;
constexpr int32_t MAX_HISTORY_FRAME_NUM = (int32_t)std::min(60u, std::min(nrd::REBLUR_MAX_HISTORY_FRAME_NUM, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
constexpr uint32_t TEXTURES_PER_MATERIAL = 4;
constexpr uint32_t MAX_TEXTURE_TRANSITIONS_NUM = 32;
constexpr uint32_t DYNAMIC_CONSTANT_BUFFER_SIZE = 1024 * 1024; // 1MB
constexpr uint32_t MAX_ANIMATION_HISTORY_FRAME_NUM = 2;

#if (SIGMA_TRANSLUCENT == 1)
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW_TRANSLUCENCY
#else
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW
#endif

//=================================================================================
// Important tests, sensitive to regressions or just testing base functionality
//=================================================================================

const std::vector<uint32_t> interior_checkMeTests = {{1, 3, 6, 8, 9, 10, 12, 13, 14, 23, 27, 28, 29, 31, 32, 35, 43, 44, 47, 53,
    59, 60, 62, 67, 75, 76, 79, 81, 95, 96, 107, 109, 111, 110, 114, 120, 124,
    126, 127, 132, 133, 134, 139, 140, 142, 145, 148, 150, 155, 156, 157, 160,
    161, 162, 164, 168, 169, 171, 172, 173, 174}};

//=================================================================================
// Tests, where IQ improvement would be "nice to have"
//=================================================================================

const std::vector<uint32_t> REBLUR_interior_improveMeTests = {{108, 110, 153, 174, 191, 192}};

const std::vector<uint32_t> RELAX_interior_improveMeTests = {{114, 144, 148, 156, 159}};

const std::vector<uint32_t> DLRR_interior_improveMeTests = {{
    1, 6, 159,   // snappy specular tracking
    4, 181,      // boily reaction to importance sampling
    62, 98, 112, // diffuse missing details and ghosting
    185, 186,    // missing material details (low confidence reprojection)
    220,         // patterns
    221,         // ortho
    222,         // diffuse darkening
}};

// TODO: add tests for SIGMA, active when "Shadow" visualization is on

//=================================================================================

// UI
#define UI_YELLOW ImVec4(1.0f, 0.9f, 0.0f, 1.0f)
#define UI_GREEN ImVec4(0.5f, 0.9f, 0.0f, 1.0f)
#define UI_RED ImVec4(1.0f, 0.1f, 0.0f, 1.0f)
#define UI_HEADER ImVec4(0.7f, 1.0f, 0.7f, 1.0f)
#define UI_HEADER_BACKGROUND ImVec4(0.7f * 0.3f, 1.0f * 0.3f, 0.7f * 0.3f, 1.0f)
#define UI_DEFAULT ImGui::GetStyleColorVec4(ImGuiCol_Text)

enum MvType : int32_t {
    MV_2D,
    MV_25D,
};

enum class AccelerationStructure : uint32_t {
    TLAS_World,
    TLAS_Emissive,

    BLAS_MergedOpaque,
    BLAS_MergedTransparent,
    BLAS_MergedEmissive,
    BLAS_Other
};

enum class Buffer : uint32_t {
    // DEVICE (read only)
    InstanceData,
    MorphMeshIndices,
    MorphMeshVertices,

    // DEVICE
    MorphedPositions,
    MorphedAttributes,
    MorphedPrimitivePrevPositions,
    PrimitiveData,
    SharcHashEntries,
    SharcHashCopyOffset,
    SharcVoxelDataPing,
    SharcVoxelDataPong,

    // DEVICE (scratch)
    WorldScratch,
    LightScratch,
    MorphMeshScratch,
};

enum class Texture : uint32_t {
    ViewZ,
    Mv,
    Normal_Roughness,
    PsrThroughput,
    BaseColor_Metalness,
    DirectLighting,
    DirectEmission,
    Shadow,
    Diff,
    Spec,
    Unfiltered_Penumbra,
    Unfiltered_Diff,
    Unfiltered_Spec,
    Unfiltered_Translucency,
    Validation,
    Composed,

    // History
    ComposedDiff,
    ComposedSpec_ViewZ,
    TaaHistory,
    TaaHistoryPrev,

    // RR guides
    RRGuide_DiffAlbedo,
    RRGuide_SpecAlbedo,
    RRGuide_SpecHitDistance,
    RRGuide_Normal_Roughness, // only RGBA16f encoding is supported

    // Output resolution
    DlssOutput,
    PreFinal,

    // Window resolution
    Final,

    // SH
#if (NRD_MODE == SH)
    Unfiltered_DiffSh,
    Unfiltered_SpecSh,
    DiffSh,
    SpecSh,
#endif

    // Read-only
    MaterialTextures,
};

enum class Pipeline : uint32_t {
    MorphMeshUpdateVertices,
    MorphMeshUpdatePrimitives,
    SharcUpdate,
    SharcResolve,
    SharcHashCopy,
    TraceOpaque,
    Composition,
    TraceTransparent,
    Taa,
    Final,
    DlssBefore,
    DlssAfter,
};

enum class Descriptor : uint32_t {
    World_AccelerationStructure,
    Light_AccelerationStructure,

    LinearMipmapLinear_Sampler,
    LinearMipmapNearest_Sampler,
    NearestMipmapNearest_Sampler,

    Global_ConstantBuffer,
    MorphTargetPose_ConstantBuffer,
    MorphTargetUpdatePrimitives_ConstantBuffer,

    InstanceData_Buffer,
    MorphMeshIndices_Buffer,
    MorphMeshVertices_Buffer,

    MorphedPositions_Buffer,
    MorphedPositions_StorageBuffer,
    MorphedAttributes_Buffer,
    MorphedAttributes_StorageBuffer,
    MorphedPrimitivePrevData_Buffer,
    MorphedPrimitivePrevData_StorageBuffer,
    PrimitiveData_Buffer,
    PrimitiveData_StorageBuffer,

    SharcHashEntries_StorageBuffer,
    SharcHashCopyOffset_StorageBuffer,
    SharcVoxelDataPing_StorageBuffer,
    SharcVoxelDataPong_StorageBuffer,

    ViewZ_Texture,
    ViewZ_StorageTexture,
    Mv_Texture,
    Mv_StorageTexture,
    Normal_Roughness_Texture,
    Normal_Roughness_StorageTexture,
    PsrThroughput_Texture,
    PsrThroughput_StorageTexture,
    BaseColor_Metalness_Texture,
    BaseColor_Metalness_StorageTexture,
    DirectLighting_Texture,
    DirectLighting_StorageTexture,
    DirectEmission_Texture,
    DirectEmission_StorageTexture,
    Shadow_Texture,
    Shadow_StorageTexture,
    Diff_Texture,
    Diff_StorageTexture,
    Spec_Texture,
    Spec_StorageTexture,
    Unfiltered_Penumbra_Texture,
    Unfiltered_Penumbra_StorageTexture,
    Unfiltered_Diff_Texture,
    Unfiltered_Diff_StorageTexture,
    Unfiltered_Spec_Texture,
    Unfiltered_Spec_StorageTexture,
    Unfiltered_Translucency_Texture,
    Unfiltered_Translucency_StorageTexture,
    Validation_Texture,
    Validation_StorageTexture,
    Composed_Texture,
    Composed_StorageTexture,

    // History
    ComposedDiff_Texture,
    ComposedDiff_StorageTexture,
    ComposedSpec_ViewZ_Texture,
    ComposedSpec_ViewZ_StorageTexture,
    TaaHistory_Texture,
    TaaHistory_StorageTexture,
    TaaHistoryPrev_Texture,
    TaaHistoryPrev_StorageTexture,

    // RR guides
    RRGuide_DiffAlbedo_Texture,
    RRGuide_DiffAlbedo_StorageTexture,
    RRGuide_SpecAlbedo_Texture,
    RRGuide_SpecAlbedo_StorageTexture,
    RRGuide_SpecHitDistance_Texture,
    RRGuide_SpecHitDistance_StorageTexture,
    RRGuide_Normal_Roughness_Texture,
    RRGuide_Normal_Roughness_StorageTexture,

    // Output resolution
    DlssOutput_Texture,
    DlssOutput_StorageTexture,
    PreFinal_Texture,
    PreFinal_StorageTexture,

    // Window resolution
    Final_Texture,
    Final_StorageTexture,

    // SH
#if (NRD_MODE == SH)
    Unfiltered_DiffSh_Texture,
    Unfiltered_DiffSh_StorageTexture,
    Unfiltered_SpecSh_Texture,
    Unfiltered_SpecSh_StorageTexture,
    DiffSh_Texture,
    DiffSh_StorageTexture,
    SpecSh_Texture,
    SpecSh_StorageTexture,
#endif

    // Read-only
    MaterialTextures,
};

enum class DescriptorSet : uint32_t {
    Global0,
    TraceOpaque1,
    Composition1,
    TraceTransparent1,
    Taa1a,
    Taa1b,
    Final1,
    DlssBefore1,
    DlssAfter1,
    RayTracing2,
    MorphTargetPose3,
    MorphTargetUpdatePrimitives3,
    SharcPing4,
    SharcPong4,

    MAX_NUM
};

// NRD sample doesn't use several instances of the same denoiser in one NRD instance (like REBLUR_DIFFUSE x 3),
// thus we can use fields of "nrd::Denoiser" enum as unique identifiers
#define NRD_ID(x) nrd::Identifier(nrd::Denoiser::x)

struct NRIInterface
    : public nri::CoreInterface,
      public nri::HelperInterface,
      public nri::RayTracingInterface,
      public nri::ResourceAllocatorInterface,
      public nri::StreamerInterface,
      public nri::SwapChainInterface,
      public nri::UpscalerInterface {};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

struct Settings {
    double motionStartTime = 0.0;

    float maxFps = 60.0f;
    float camFov = 90.0f;
    float sunAzimuth = -147.0f;
    float sunElevation = 45.0f;
    float sunAngularDiameter = 0.533f;
    float exposure = 80.0f;
    float roughnessOverride = 0.0f;
    float metalnessOverride = 0.0f;
    float emissionIntensity = 1.0f;
    float debug = 0.0f;
    float meterToUnitsMultiplier = 1.0f;
    float emulateMotionSpeed = 1.0f;
    float animatedObjectScale = 1.0f;
    float separator = 0.0f;
    float animationProgress = 0.0f;
    float animationSpeed = 0.0f;
    float hitDistScale = 3.0f;
    float unused1 = 0.0f;
    float resolutionScale = 1.0f;
    float sharpness = 0.15f;

    int32_t maxAccumulatedFrameNum = 31;
    int32_t maxFastAccumulatedFrameNum = 7;
    int32_t onScreen = 0;
    int32_t forcedMaterial = 0;
    int32_t animatedObjectNum = 5;
    int32_t activeAnimation = 0;
    int32_t motionMode = 0;
    int32_t denoiser = DENOISER_REBLUR;
    int32_t rpp = 1;
    int32_t bounceNum = 1;
    int32_t tracingMode = RESOLUTION_HALF;
    int32_t mvType = MV_25D;

    bool cameraJitter = true;
    bool limitFps = false;
    bool SHARC = true;
    bool PSR = false;
    bool indirectDiffuse = true;
    bool indirectSpecular = true;
    bool normalMap = true;
    bool TAA = true;
    bool animatedObjects = false;
    bool animateScene = false;
    bool animateSun = false;
    bool nineBrothers = false;
    bool blink = false;
    bool pauseAnimation = true;
    bool emission = false;
    bool linearMotion = true;
    bool emissiveObjects = false;
    bool importanceSampling = true;
    bool specularLobeTrimming = true;
    bool ortho = false;
    bool adaptiveAccumulation = true;
    bool usePrevFrame = true;
    bool windowAlignment = true;
    bool boost = false;
    bool SR = false;
    bool RR = false;
};

struct DescriptorDesc {
    const char* debugName;
    void* resource;
    nri::Format format;
    nri::TextureUsageBits textureUsage;
    nri::BufferUsageBits bufferUsage;
    bool isArray;
};

struct TextureState {
    Texture texture;
    nri::AccessLayoutStage after;
};

struct AnimatedInstance {
    float3 basePosition;
    float3 rotationAxis;
    float3 elipseAxis;
    float durationSec = 5.0f;
    float progressedSec = 0.0f;
    uint32_t instanceID = 0;
    bool reverseRotation = true;
    bool reverseDirection = true;

    float4x4 Animate(float elapsedSeconds, float scale, float3& position) {
        float angle = progressedSec / durationSec;
        angle = Pi(angle * 2.0f - 1.0f);

        float3 localPosition;
        localPosition.x = cos(reverseDirection ? -angle : angle);
        localPosition.y = sin(reverseDirection ? -angle : angle);
        localPosition.z = localPosition.y;

        position = basePosition + localPosition * elipseAxis * scale;

        float4x4 transform;
        transform.SetupByRotation(reverseRotation ? -angle : angle, rotationAxis);
        transform.AddScale(scale);

        progressedSec = fmod(progressedSec + elapsedSeconds, durationSec);

        return transform;
    }
};

class Sample : public SampleBase {
public:
    inline Sample() {
        m_QueuedFrames.resize(GetQueuedFrameNum());
    }

    ~Sample();

    inline float GetDenoisingRange() const {
        return 4.0f * m_Scene.aabb.GetRadius();
    }

    inline bool IsDlssEnabled() const {
        return m_Settings.SR || m_Settings.RR;
    }

    inline nri::Texture*& Get(Texture index) {
        return m_Textures[(uint32_t)index];
    }

    inline nri::TextureBarrierDesc& GetState(Texture index) {
        return m_TextureStates[(uint32_t)index];
    }

    inline nri::Buffer*& Get(Buffer index) {
        return m_Buffers[(uint32_t)index];
    }

    inline nri::Pipeline*& Get(Pipeline index) {
        return m_Pipelines[(uint32_t)index];
    }

    inline nri::Descriptor*& Get(Descriptor index) {
        return m_Descriptors[(uint32_t)index];
    }

    inline nri::DescriptorSet*& Get(DescriptorSet index) {
        return m_DescriptorSets[(uint32_t)index];
    }

    inline nri::AccelerationStructure*& Get(AccelerationStructure index) {
        return m_AccelerationStructures[(uint32_t)index];
    }

    inline void InitCmdLine(cmdline::parser& cmdLine) override {
        cmdLine.add<int32_t>("dlssQuality", 'd', "DLSS quality: [-1: 4]", false, -1, cmdline::range(-1, 4));
        cmdLine.add("debugNRD", 0, "enable NRD validation");
    }

    inline void ReadCmdLine(cmdline::parser& cmdLine) override {
        m_DlssQuality = cmdLine.get<int32_t>("dlssQuality");
        m_DebugNRD = cmdLine.exist("debugNRD");
    }

    inline nrd::RelaxSettings GetDefaultRelaxSettings() const {
        nrd::RelaxSettings defaults = {};
        // Helps to mitigate fireflies emphasized by DLSS

#if (NRD_MODE < OCCLUSION)
        // defaults.enableAntiFirefly = m_DlssQuality != -1 && IsDlssEnabled(); // TODO: currently doesn't help in this case, but makes the image darker
#endif

        return defaults;
    }

    inline nrd::ReblurSettings GetDefaultReblurSettings() const {
        nrd::ReblurSettings defaults = {};

#if (NRD_MODE < OCCLUSION)
        // Helps to mitigate fireflies emphasized by DLSS
        defaults.enableAntiFirefly = m_DlssQuality != -1 && IsDlssEnabled();
#else
        // Occlusion signal is cleaner by the definition
        defaults.historyFixFrameNum = 2;

        // TODO: experimental, but works well so far
        defaults.minBlurRadius = 5.0f;
        defaults.lobeAngleFraction = 0.5f;
#endif

        return defaults;
    }

    inline float3 GetSunDirection() const {
        float3 sunDirection;
        sunDirection.x = cos(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.y = sin(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.z = sin(radians(m_Settings.sunElevation));

        return sunDirection;
    }

    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    void LoadScene();
    void AddInnerGlassSurfaces();
    void GenerateAnimatedCubes();
    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipelineLayoutAndDescriptorPool();
    void CreatePipelines();
    void CreateAccelerationStructures();
    void CreateSamplers();
    void CreateResources(nri::Format swapChainFormat);
    void CreateDescriptorSets();
    void CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, nri::Dim_t width, nri::Dim_t height, nri::Mip_t mipNum, nri::Dim_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state);
    void CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage);
    void UploadStaticData();
    void UpdateConstantBuffer(uint32_t frameIndex, float resetHistoryFactor);
    void RestoreBindings(nri::CommandBuffer& commandBuffer, bool isEven, bool needRayTracing = true);
    void GatherInstanceData();
    uint16_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM>& transitions);

private:
    // NRD
    nrd::Integration m_NRD = {};
    nrd::RelaxSettings m_RelaxSettings = {};
    nrd::ReblurSettings m_ReblurSettings = {};
    nrd::SigmaSettings m_SigmaSettings = {};
    nrd::ReferenceSettings m_ReferenceSettings = {};

    // NRI
    NRIInterface NRI = {};
    utils::Scene m_Scene;
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::Upscaler* m_NIS = nullptr;
    nri::Upscaler* m_DLSR = nullptr;
    nri::Upscaler* m_DLRR = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureBarrierDesc> m_TextureStates;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_AccelerationStructures;
    std::vector<BackBuffer> m_SwapChainBuffers;

    // Data
    std::vector<InstanceData> m_InstanceData;
    std::vector<nri::TopLevelInstance> m_WorldTlasData;
    std::vector<nri::TopLevelInstance> m_LightTlasData;
    std::vector<AnimatedInstance> m_AnimatedInstances;
    std::array<float, 256> m_FrameTimes = {};
    Settings m_Settings = {};
    Settings m_SettingsPrev = {};
    Settings m_SettingsDefault = {};
    const std::vector<uint32_t>* m_checkMeTests = nullptr;
    const std::vector<uint32_t>* m_improveMeTests = nullptr;
    float4 m_HairBaseColor = float4(0.510f, 0.395f, 0.218f, 1.0f);
    float3 m_PrevLocalPos = {};
    float2 m_HairBetas = float2(0.25f, 0.6f);
    uint2 m_RenderResolution = {};
    uint64_t m_MorphMeshScratchSize = 0;
    nri::BufferOffset m_WorldTlasDataLocation = {};
    nri::BufferOffset m_LightTlasDataLocation = {};
    uint32_t m_GlobalConstantBufferOffset = 0;
    uint32_t m_OpaqueObjectsNum = 0;
    uint32_t m_TransparentObjectsNum = 0;
    uint32_t m_EmissiveObjectsNum = 0;
    uint32_t m_ProxyInstancesNum = 0;
    uint32_t m_LastSelectedTest = uint32_t(-1);
    uint32_t m_TestNum = uint32_t(-1);
    int32_t m_DlssQuality = int32_t(-1);
    float m_SigmaTemporalStabilizationStrength = 1.0f;
    float m_UiWidth = 0.0f;
    float m_MinResolutionScale = 0.5f;
    float m_DofAperture = 0.0f;
    float m_DofFocalDistance = 1.0f;
    float m_SdrScale = 1.0f;
    bool m_ShowUi = true;
    bool m_ForceHistoryReset = false;
    bool m_Resolve = true;
    bool m_DebugNRD = false;
    bool m_ShowValidationOverlay = false;
    bool m_PositiveZ = true;
    bool m_ReversedZ = false;
    bool m_IsSrgb = false;
    bool m_GlassObjects = false;
    bool m_IsReloadShadersSucceeded = true;
};

Sample::~Sample() {
    if (!m_Device)
        return;

    NRI.WaitForIdle(*m_GraphicsQueue);

    NRI.DestroyUpscaler(*m_NIS);
    NRI.DestroyUpscaler(*m_DLSR);
    NRI.DestroyUpscaler(*m_DLRR);

    m_NRD.Destroy();

    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI.DestroyCommandBuffer(*queuedFrame.commandBuffer);
        NRI.DestroyCommandAllocator(*queuedFrame.commandAllocator);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);

    for (uint32_t i = 0; i < m_Textures.size(); i++)
        NRI.DestroyTexture(*m_Textures[i]);

    for (uint32_t i = 0; i < m_Buffers.size(); i++)
        NRI.DestroyBuffer(*m_Buffers[i]);

    for (uint32_t i = 0; i < m_Descriptors.size(); i++)
        NRI.DestroyDescriptor(*m_Descriptors[i]);

    for (uint32_t i = 0; i < m_Pipelines.size(); i++)
        NRI.DestroyPipeline(*m_Pipelines[i]);

    for (uint32_t i = 0; i < m_AccelerationStructures.size(); i++)
        NRI.DestroyAccelerationStructure(*m_AccelerationStructures[i]);

    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);
    NRI.DestroyStreamer(*m_Streamer);

    DestroyUI();

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
    Rng::Hash::Initialize(m_RngState, 106937, 69);

    nri::AdapterDesc bestAdapterDesc = {};
    uint32_t adapterDescsNum = 1;
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(&bestAdapterDesc, adapterDescsNum));

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &bestAdapterDesc;
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::ResourceAllocatorInterface), (nri::ResourceAllocatorInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::UpscalerInterface), (nri::UpscalerInterface*)&NRI));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Create streamer
        nri::StreamerDesc streamerDesc = {};
        streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        streamerDesc.constantBufferSize = DYNAMIC_CONSTANT_BUFFER_SIZE;
        streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        streamerDesc.dynamicBufferUsageBits = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT;
        streamerDesc.queuedFrameNum = GetQueuedFrameNum();
        NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));
    }

    { // Create upscaler: NIS
        nri::UpscalerDesc upscalerDesc = {};
        upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
        upscalerDesc.type = nri::UpscalerType::NIS;
        NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_NIS));
    }

    // Create upscalers: DLSR and DLRR
    m_RenderResolution = GetOutputResolution();

    if (m_DlssQuality != -1) {
        nri::UpscalerBits upscalerFlags = nri::UpscalerBits::DEPTH_INFINITE;
        upscalerFlags |= NRD_MODE < OCCLUSION ? nri::UpscalerBits::HDR : nri::UpscalerBits::NONE;
        upscalerFlags |= m_ReversedZ ? nri::UpscalerBits::DEPTH_INVERTED : nri::UpscalerBits::NONE;

        nri::UpscalerMode mode = nri::UpscalerMode::NATIVE;
        if (m_DlssQuality == 0)
            mode = nri::UpscalerMode::ULTRA_PERFORMANCE;
        else if (m_DlssQuality == 1)
            mode = nri::UpscalerMode::PERFORMANCE;
        else if (m_DlssQuality == 2)
            mode = nri::UpscalerMode::BALANCED;
        else if (m_DlssQuality == 3)
            mode = nri::UpscalerMode::QUALITY;

        if (NRI.IsUpscalerSupported(*m_Device, nri::UpscalerType::DLSR)) {
            nri::VideoMemoryInfo videoMemoryInfo1 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

            nri::UpscalerDesc upscalerDesc = {};
            upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
            upscalerDesc.type = upscalerType;
            upscalerDesc.mode = mode;
            upscalerDesc.flags = upscalerFlags;
            upscalerDesc.preset = USE_DLSS_TNN ? 10 : 0;
            NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_DLSR));

            nri::UpscalerProps upscalerProps = {};
            NRI.GetUpscalerProps(*m_DLSR, upscalerProps);

            float sx = float(upscalerProps.renderResolutionMin.w) / float(upscalerProps.renderResolution.w);
            float sy = float(upscalerProps.renderResolutionMin.h) / float(upscalerProps.renderResolution.h);

            m_RenderResolution = {upscalerProps.renderResolution.w, upscalerProps.renderResolution.h};
            m_MinResolutionScale = sy > sx ? sy : sx;

            nri::VideoMemoryInfo videoMemoryInfo2 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

            printf("Render resolution (%u, %u)\n", m_RenderResolution.x, m_RenderResolution.y);
            printf("DLSS-SR: allocated %.2f Mb\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));

            m_Settings.SR = true;
        }

        if (NRI.IsUpscalerSupported(*m_Device, nri::UpscalerType::DLRR)) {
            nri::VideoMemoryInfo videoMemoryInfo1 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

            nri::UpscalerDesc upscalerDesc = {};
            upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
            upscalerDesc.type = nri::UpscalerType::DLRR;
            upscalerDesc.mode = mode;
            upscalerDesc.flags = upscalerFlags;
            NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_DLRR));

            nri::VideoMemoryInfo videoMemoryInfo2 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

            printf("DLSS-RR: allocated %.2f Mb\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));

            m_Settings.RR = true;
        }
    }

    // Initialize NRD: REBLUR, RELAX and SIGMA in one instance
    {
        const nrd::DenoiserDesc denoisersDescs[] = {
        // REBLUR
#if (NRD_MODE == OCCLUSION)
#    if (NRD_COMBINED == 1)
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR_OCCLUSION), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_OCCLUSION},
#    else
            {NRD_ID(REBLUR_DIFFUSE_OCCLUSION), nrd::Denoiser::REBLUR_DIFFUSE_OCCLUSION},
            {NRD_ID(REBLUR_SPECULAR_OCCLUSION), nrd::Denoiser::REBLUR_SPECULAR_OCCLUSION},
#    endif
#elif (NRD_MODE == SH)
#    if (NRD_COMBINED == 1)
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR_SH), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH},
#    else
            {NRD_ID(REBLUR_DIFFUSE_SH), nrd::Denoiser::REBLUR_DIFFUSE_SH},
            {NRD_ID(REBLUR_SPECULAR_SH), nrd::Denoiser::REBLUR_SPECULAR_SH},
#    endif
#elif (NRD_MODE == DIRECTIONAL_OCCLUSION)
            {NRD_ID(REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION), nrd::Denoiser::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION},
#else
#    if (NRD_COMBINED == 1)
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR},
#    else
            {NRD_ID(REBLUR_DIFFUSE), nrd::Denoiser::REBLUR_DIFFUSE},
            {NRD_ID(REBLUR_SPECULAR), nrd::Denoiser::REBLUR_SPECULAR},
#    endif
#endif

        // RELAX
#if (NRD_MODE == SH)
#    if (NRD_COMBINED == 1)
            {NRD_ID(RELAX_DIFFUSE_SPECULAR_SH), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH},
#    else
            {NRD_ID(RELAX_DIFFUSE_SH), nrd::Denoiser::RELAX_DIFFUSE_SH},
            {NRD_ID(RELAX_SPECULAR_SH), nrd::Denoiser::RELAX_SPECULAR_SH},
#    endif
#else
#    if (NRD_COMBINED == 1)
            {NRD_ID(RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR},
#    else
            {NRD_ID(RELAX_DIFFUSE), nrd::Denoiser::RELAX_DIFFUSE},
            {NRD_ID(RELAX_SPECULAR), nrd::Denoiser::RELAX_SPECULAR},
#    endif
#endif

        // SIGMA
#if (NRD_MODE < OCCLUSION)
            {NRD_ID(SIGMA_SHADOW), SIGMA_VARIANT},
#endif

            // REFERENCE
            {NRD_ID(REFERENCE), nrd::Denoiser::REFERENCE},
        };

        nrd::InstanceCreationDesc instanceCreationDesc = {};
        instanceCreationDesc.denoisers = denoisersDescs;
        instanceCreationDesc.denoisersNum = helper::GetCountOf(denoisersDescs);

        nrd::IntegrationCreationDesc desc = {};
        desc.name = "NRD";
        desc.queuedFrameNum = GetQueuedFrameNum();
        desc.enableDescriptorCaching = NRD_ALLOW_DESCRIPTOR_CACHING;
        desc.promoteFloat16to32 = NRD_PROMOTE_FLOAT16_TO_32;
        desc.demoteFloat32to16 = NRD_DEMOTE_FLOAT32_TO_16;
        desc.resourceWidth = (uint16_t)m_RenderResolution.x;
        desc.resourceHeight = (uint16_t)m_RenderResolution.y;

        nri::VideoMemoryInfo videoMemoryInfo1 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

        NRI_ABORT_ON_FALSE(m_NRD.Initialize(desc, instanceCreationDesc, *m_Device, NRI));

        nri::VideoMemoryInfo videoMemoryInfo2 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

        printf("NRD: allocated %.2f Mb for REBLUR, RELAX, SIGMA and REFERENCE denoisers\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));
    }

#if 0
        // README "Memory requirements" table generator
        printf("| %10s | %36s | %16s | %16s | %16s |\n", "Resolution", "Denoiser", "Working set (Mb)", "Persistent (Mb)", "Aliasable (Mb)");
        printf("|------------|--------------------------------------|------------------|------------------|------------------|\n");

        for (uint32_t j = 0; j < 3; j++)
        {
            const char* resolution = "1080p";
            uint16_t w = 1920;
            uint16_t h = 1080;

            if (j == 1)
            {
                resolution = "1440p";
                w = 2560;
                h = 1440;
            }
            else if (j == 2)
            {
                resolution = "2160p";
                w = 3840;
                h = 2160;
            }

            for (uint32_t i = 0; i <= (uint32_t)nrd::Denoiser::REFERENCE; i++)
            {
                nrd::Denoiser denoiser = (nrd::Denoiser)i;
                const char* methodName = nrd::GetDenoiserString(denoiser);

                const nrd::DenoiserDesc denoiserDesc = {0, denoiser};

                nrd::InstanceCreationDesc instanceCreationDesc = {};
                instanceCreationDesc.denoisers = &denoiserDesc;
                instanceCreationDesc.denoisersNum = 1;

                nrd::IntegrationCreationDesc desc = {};
                desc.name = "Unused";
                desc.queuedFrameNum = GetQueuedFrameNum();
                desc.enableDescriptorCaching = NRD_ALLOW_DESCRIPTOR_CACHING;
                desc.promoteFloat16to32 = NRD_PROMOTE_FLOAT16_TO_32;
                desc.demoteFloat32to16 = NRD_DEMOTE_FLOAT32_TO_16;
                desc.resourceWidth = w;
                desc.resourceHeight = h;

                nrd::Integration instance;
                NRI_ABORT_ON_FALSE( instance.Initialize(desc, instanceCreationDesc, *m_Device, NRI, NRI) );

                printf("| %10s | %36s | %16.2f | %16.2f | %16.2f |\n", i == 0 ? resolution : "", methodName, instance.GetTotalMemoryUsageInMb(), instance.GetPersistentMemoryUsageInMb(), instance.GetAliasableMemoryUsageInMb());

                instance.Destroy();
            }

            if (j != 2)
                printf("| %10s | %36s | %16s | %16s | %16s |\n", "", "", "", "", "");
        }

        __debugbreak();
#endif

    LoadScene();

    if (m_SceneFile.find("BistroInterior") != std::string::npos)
        AddInnerGlassSurfaces();

    GenerateAnimatedCubes();

    nri::Format swapChainFormat = CreateSwapChain();
    CreateCommandBuffers();
    CreatePipelineLayoutAndDescriptorPool();
    CreatePipelines();
    CreateAccelerationStructures();
    CreateSamplers();
    CreateResources(swapChainFormat);
    CreateDescriptorSets();

    UploadStaticData();

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
    m_Scene.UnloadTextureData();
    m_Scene.UnloadGeometryData();

    m_SettingsDefault = m_Settings;
    m_ShowValidationOverlay = m_DebugNRD;

    nri::VideoMemoryInfo videoMemoryInfo = {};
    NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo);
    printf("Allocated %.2f Mb\n", videoMemoryInfo.usageSize / (1024.0f * 1024.0f));

    return InitUI(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    const QueuedFrame& queuedFrame = m_QueuedFrames[frameIndex % GetQueuedFrameNum()];
    if (frameIndex >= GetQueuedFrameNum()) {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - GetQueuedFrameNum());
        NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
    }
}

void Sample::PrepareFrame(uint32_t frameIndex) {
    nri::nriBeginAnnotation("Prepare frame", nri::BGRA_UNUSED);

    m_ForceHistoryReset = false;
    m_SettingsPrev = m_Settings;
    m_Camera.SavePreviousState();

    if (IsKeyToggled(Key::Tab))
        m_ShowUi = !m_ShowUi;
    if (IsKeyToggled(Key::F1))
        m_Settings.debug = step(0.5f, 1.0f - m_Settings.debug);
    if (IsKeyToggled(Key::F3))
        m_Settings.emission = !m_Settings.emission;
    if (IsKeyToggled(Key::Space))
        m_Settings.pauseAnimation = !m_Settings.pauseAnimation;
    if (IsKeyToggled(Key::PageDown) || IsKeyToggled(Key::Num3)) {
        m_Settings.denoiser++;
        if (m_Settings.denoiser > DENOISER_REFERENCE)
            m_Settings.denoiser = DENOISER_REBLUR;
    }
    if (IsKeyToggled(Key::PageUp) || IsKeyToggled(Key::Num9)) {
        m_Settings.denoiser--;
        if (m_Settings.denoiser < DENOISER_REBLUR)
            m_Settings.denoiser = DENOISER_REFERENCE;
    }

    BeginUI();
    if (!IsKeyPressed(Key::LAlt) && m_ShowUi) {
        static const char* onScreenModes[] = {
#if (NRD_MODE == OCCLUSION)
            "Diffuse occlusion",
            "Specular occlusion",
#elif (NRD_MODE == DIRECTIONAL_OCCLUSION)
            "Diffuse occlusion",
#else
            "Final",
            "Denoised diffuse",
            "Denoised specular",
            "Diffuse occlusion",
            "Specular occlusion",
            "Shadow",
            "Base color",
            "Normal",
            "Roughness",
            "Metalness",
            "Material ID",
            "PSR throughput",
            "World units",
            "Instance index",
            "UV",
            "Curvature",
            "Mip level (primary)",
            "Mip level (specular)",
#endif
        };

        static std::array<const char*, 4> nrdModes = {
            "NORMAL",
            "SH",
            "OCCLUSION"
            "DIRECTIONAL_OCCLUSION"};

        const nrd::LibraryDesc& nrdLibraryDesc = nrd::GetLibraryDesc();

        char buf[256];
        snprintf(buf, sizeof(buf) - 1, "NRD v%u.%u.%u (%u.%u) - %s [Tab]", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild, nrdLibraryDesc.normalEncoding, nrdLibraryDesc.roughnessEncoding, nrdModes[NRD_MODE]);

        ImGui::SetNextWindowPos(ImVec2(m_Settings.windowAlignment ? 5.0f : GetOutputResolution().x - m_UiWidth - 5.0f, 5.0f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f));
        ImGui::Begin(buf, nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        {
            float avgFrameTime = m_Timer.GetVerySmoothedFrameTime();
            snprintf(buf, sizeof(buf), "%.1f FPS (%.2f ms)", 1000.0f / avgFrameTime, avgFrameTime);

            ImVec4 colorFps = UI_GREEN;
            if (avgFrameTime > 1000.0f / 59.5f)
                colorFps = UI_YELLOW;
            if (avgFrameTime > 1000.0f / 29.5f)
                colorFps = UI_RED;

            float lo = avgFrameTime * 0.5f;
            float hi = avgFrameTime * 1.5f;

            const uint32_t N = helper::GetCountOf(m_FrameTimes);
            uint32_t head = frameIndex % N;
            m_FrameTimes[head] = m_Timer.GetFrameTime();
            ImGui::PushStyleColor(ImGuiCol_Text, colorFps);
            ImGui::PlotLines("##Plot", m_FrameTimes.data(), N, head, buf, lo, hi, ImVec2(0.0f, 70.0f));
            ImGui::PopStyleColor();

            if (IsButtonPressed(Button::Right)) {
                ImGui::Text("Move - W/S/A/D");
                ImGui::Text("Accelerate - MOUSE SCROLL");
            } else {
                // "Camera" section
                ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                bool isUnfolded = ImGui::CollapsingHeader("CAMERA (press RIGHT MOUSE BOTTON for free-fly mode)", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                ImGui::PushID("CAMERA");
                if (isUnfolded) {
                    static const char* motionMode[] = {
                        "Left / Right",
                        "Up / Down",
                        "Forward / Backward",
                        "Mixed",
                        "Pan",
                    };

                    static const char* mvType[] = {
                        "2D",
                        "2.5D",
                    };

                    ImGui::Combo("On screen", &m_Settings.onScreen, onScreenModes, helper::GetCountOf(onScreenModes));
                    ImGui::Checkbox("Ortho", &m_Settings.ortho);
                    ImGui::SameLine();
                    ImGui::Checkbox("+Z", &m_PositiveZ);
                    ImGui::SameLine();
                    ImGui::Checkbox("rZ", &m_ReversedZ);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, (!m_Settings.cameraJitter && (m_Settings.TAA || IsDlssEnabled())) ? UI_RED : UI_DEFAULT);
                    ImGui::Checkbox("Jitter", &m_Settings.cameraJitter);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.animatedObjects && !m_Settings.pauseAnimation && m_Settings.mvType == MV_2D) ? UI_RED : UI_DEFAULT);
                    ImGui::Combo("MV", &m_Settings.mvType, mvType, helper::GetCountOf(mvType));
                    ImGui::PopStyleColor();

                    ImGui::SliderFloat("FOV (deg)", &m_Settings.camFov, 1.0f, 160.0f, "%.1f");
                    ImGui::SliderFloat("Exposure", &m_Settings.exposure, 0.0f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

                    if (m_DLRR) {
                        ImGui::Checkbox("DLSS-RR", &m_Settings.RR);
                        ImGui::SameLine();
                    }
                    if (m_DLSR && !m_Settings.RR) {
                        ImGui::Checkbox("DLSS-SR", &m_Settings.SR);
                        ImGui::SameLine();
                    }
                    if (!m_Settings.SR) {
                        ImGui::Checkbox("TAA", &m_Settings.TAA);
                        ImGui::SameLine();
                    }
                    ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                    if (m_Settings.RR)
                        m_Settings.resolutionScale = 1.0f; // TODO: RR doesn't support DRS
                    else
                        ImGui::SliderFloat("Resolution scale (%)", &m_Settings.resolutionScale, m_MinResolutionScale, 1.0f, "%.3f");

                    ImGui::SliderFloat("Aperture (cm)", &m_DofAperture, 0.0f, 100.0f, "%.2f");
                    ImGui::SliderFloat("Focal distance (m)", &m_DofFocalDistance, NEAR_Z, 10.0f, "%.3f");

                    ImGui::Checkbox("FPS cap", &m_Settings.limitFps);
                    if (m_Settings.limitFps) {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                        ImGui::SliderFloat("Max FPS", &m_Settings.maxFps, 30.0f, 120.0f, "%.0f");
                    }

                    ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.motionStartTime > 0.0 ? UI_YELLOW : UI_DEFAULT);
                    bool isPressed = ImGui::Button("Animation");
                    ImGui::PopStyleColor();
                    if (isPressed)
                        m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? 0.0 : -1.0;
                    if (m_Settings.motionStartTime > 0.0) {
                        ImGui::SameLine();
                        ImGui::Checkbox("Linear", &m_Settings.linearMotion);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                        ImGui::Combo("Mode", &m_Settings.motionMode, motionMode, helper::GetCountOf(motionMode));
                        ImGui::SliderFloat("Slower / Faster", &m_Settings.emulateMotionSpeed, -10.0f, 10.0f);
                    }
                }
                ImGui::PopID();

                // "Materials" section
                ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                isUnfolded = ImGui::CollapsingHeader("MATERIALS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                ImGui::PushID("MATERIALS");
                if (isUnfolded) {
                    static const char* forcedMaterial[] = {
                        "None",
                        "Gypsum",
                        "Cobalt",
                    };

                    ImGui::SliderFloat2("Roughness / Metalness", &m_Settings.roughnessOverride, 0.0f, 1.0f, "%.3f");
                    ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.emissiveObjects && !m_Settings.emission) ? UI_YELLOW : UI_DEFAULT);
                    ImGui::Checkbox("Emission [F3]", &m_Settings.emission);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                    ImGui::Combo("Material", &m_Settings.forcedMaterial, forcedMaterial, helper::GetCountOf(forcedMaterial));
                    if (m_Settings.emission)
                        ImGui::SliderFloat("Emission intensity", &m_Settings.emissionIntensity, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::PopID();

                // "Hair" section
                if (m_SceneFile.find("Hair") != std::string::npos) {
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("HAIR", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("HAIR");
                    if (isUnfolded) {
                        ImGui::SliderFloat2("Beta", m_HairBetas.a, 0.01f, 1.0f, "%.3f");
                        ImGui::ColorEdit3("Base color", m_HairBaseColor.a, ImGuiColorEditFlags_Float);
                    }
                    ImGui::PopID();
                }

                if (m_Settings.onScreen == 11)
                    ImGui::SliderFloat("Units in 1 meter", &m_Settings.meterToUnitsMultiplier, 0.001f, 100.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
                else {
                    // "World" section
                    snprintf(buf, sizeof(buf) - 1, "WORLD%s", (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateScene) ? (m_Settings.pauseAnimation ? " (SPACE - unpause)" : " (SPACE - pause)") : "");

                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("WORLD");
                    if (isUnfolded) {
                        ImGui::Checkbox("Animate sun", &m_Settings.animateSun);
                        if (m_Scene.animations.size() > 0) {
                            ImGui::SameLine();
                            ImGui::Checkbox("Animate scene", &m_Settings.animateScene);
                        }

                        if (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateScene) {
                            ImGui::SameLine();
                            ImGui::Checkbox("Pause", &m_Settings.pauseAnimation);
                        }

                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                        ImGui::SliderFloat("Sun size (deg)", &m_Settings.sunAngularDiameter, 0.0f, 3.0f, "%.1f");

                        ImGui::SliderFloat2("Sun position (deg)", &m_Settings.sunAzimuth, -180.0f, 180.0f, "%.2f");
                        if (!m_Settings.pauseAnimation && (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateScene))
                            ImGui::SliderFloat("Slower / Faster", &m_Settings.animationSpeed, -10.0f, 10.0f);

                        ImGui::Checkbox("Objects", &m_Settings.animatedObjects);
                        if (m_Settings.animatedObjects) {
                            ImGui::SameLine();
                            ImGui::Checkbox("9", &m_Settings.nineBrothers);
                            ImGui::SameLine();
                            ImGui::Checkbox("Blink", &m_Settings.blink);
                            ImGui::SameLine();
                            ImGui::Checkbox("Emissive", &m_Settings.emissiveObjects);
                            ImGui::SameLine();
                            ImGui::Checkbox("Glass", &m_GlassObjects);
                            if (!m_Settings.nineBrothers)
                                ImGui::SliderInt("Object number", &m_Settings.animatedObjectNum, 1, (int32_t)MAX_ANIMATED_INSTANCE_NUM);
                            ImGui::SliderFloat("Object scale", &m_Settings.animatedObjectScale, 0.1f, 2.0f);
                        }

                        if (m_Settings.animateScene && m_Scene.animations[m_Settings.activeAnimation].durationMs != 0.0f) {
                            char animationLabel[128];
                            snprintf(animationLabel, sizeof(animationLabel), "Animation %.1f sec (%%)", 0.001f * m_Scene.animations[m_Settings.activeAnimation].durationMs / (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed)));
                            ImGui::SliderFloat(animationLabel, &m_Settings.animationProgress, 0.0f, 99.999f);

                            if (m_Scene.animations.size() > 1) {
                                char items[1024] = {'\0'};
                                size_t offset = 0;
                                char* iterator = items;
                                for (auto animation : m_Scene.animations) {
                                    const size_t size = std::min(sizeof(items), animation.name.length() + 1);
                                    memcpy(iterator + offset, animation.name.c_str(), size);
                                    offset += animation.name.length() + 1;
                                }
                                ImGui::Combo("Animated scene", &m_Settings.activeAnimation, items, helper::GetCountOf(m_Scene.animations));
                            }
                        }
                    }
                    ImGui::PopID();

                    // "Path tracer" section
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("PATH TRACER", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("PATH TRACER");
                    if (isUnfolded) {
                        const float sceneRadiusInMeters = m_Scene.aabb.GetRadius() / m_Settings.meterToUnitsMultiplier;

                        static const char* resolution[] = {
                            "Full",
                            "Full (probabilistic)",
                            "Half",
                        };

#if (NRD_MODE < OCCLUSION)
                        ImGui::SliderInt2("Samples / Bounces", &m_Settings.rpp, 1, 8);
#else
                        ImGui::SliderInt("Samples", &m_Settings.rpp, 1, 8);
#endif
                        ImGui::SliderFloat("AO / SO range (m)", &m_Settings.hitDistScale, 0.01f, sceneRadiusInMeters, "%.2f");
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.denoiser == DENOISER_REFERENCE && m_Settings.tracingMode > RESOLUTION_FULL_PROBABILISTIC) ? UI_YELLOW : UI_DEFAULT);
                        ImGui::Combo("Resolution", &m_Settings.tracingMode, resolution, helper::GetCountOf(resolution));
                        ImGui::PopStyleColor();

                        ImGui::Checkbox("Diff", &m_Settings.indirectDiffuse);
                        ImGui::SameLine();
                        ImGui::Checkbox("Spec", &m_Settings.indirectSpecular);
                        ImGui::SameLine();
                        ImGui::Checkbox("Trim lobe", &m_Settings.specularLobeTrimming);
                        ImGui::SameLine();
                        ImGui::Checkbox("Normal map", &m_Settings.normalMap);

#if (NRD_MODE < OCCLUSION)
                        const float3& sunDirection = GetSunDirection();
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, sunDirection.z > 0.0f ? UI_DEFAULT : (m_Settings.importanceSampling ? UI_GREEN : UI_YELLOW));
                        ImGui::Checkbox("IS", &m_Settings.importanceSampling);
                        ImGui::PopStyleColor();

                        ImGui::Checkbox("L1 (prev frame)", &m_Settings.usePrevFrame);
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.SHARC ? UI_GREEN : UI_YELLOW);
                        ImGui::Checkbox("L2 (SHARC)", &m_Settings.SHARC);
                        ImGui::PopStyleColor();
#endif
                        if (m_Settings.tracingMode != RESOLUTION_HALF) {
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.PSR ? UI_GREEN : UI_YELLOW);
                            ImGui::Checkbox("PSR", &m_Settings.PSR);
                            ImGui::PopStyleColor();
                        }
                    }
                    ImGui::PopID();

                    // "NRD" section
                    static const char* denoiser[] = {
#if (NRD_MODE == OCCLUSION)
                        "REBLUR_OCCLUSION",
                        "(unsupported)",
#elif (NRD_MODE == SH)
                        "REBLUR_SH",
                        "RELAX_SH",
#elif (NRD_MODE == DIRECTIONAL_OCCLUSION)
                        "REBLUR_DIRECTIONAL_OCCLUSION",
                        "(unsupported)",
#else
                        "REBLUR",
                        "RELAX",
#endif
                        "REFERENCE",
                    };
                    snprintf(buf, sizeof(buf) - 1, "NRD/%s [PgDown / PgUp]", denoiser[m_Settings.denoiser]);

                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("NRD");
                    if (m_Settings.RR)
                        ImGui::Text("Pass-through mode...");
                    else if (isUnfolded) {
                        static const char* hitDistanceReconstructionMode[] = {
                            "Off",
                            "3x3",
                            "5x5",
                        };

                        if (m_DebugNRD) {
                            ImGui::PushStyleColor(ImGuiCol_Text, m_ShowValidationOverlay ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("Validation overlay", &m_ShowValidationOverlay);
                            ImGui::PopStyleColor();
                        }

                        if (ImGui::Button("<<")) {
                            m_Settings.denoiser--;
                            if (m_Settings.denoiser < DENOISER_REBLUR)
                                m_Settings.denoiser = DENOISER_REFERENCE;
                        }

                        ImGui::SameLine();
                        if (ImGui::Button(">>")) {
                            m_Settings.denoiser++;
                            if (m_Settings.denoiser > DENOISER_REFERENCE)
                                m_Settings.denoiser = DENOISER_REBLUR;
                        }

                        ImGui::SameLine();
                        m_ForceHistoryReset = ImGui::Button("Reset");

                        if (m_Settings.denoiser == DENOISER_REBLUR) {
                            nrd::ReblurSettings defaults = GetDefaultReblurSettings();

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
                                defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                            }

                            bool isSame = true;
                            if (m_ReblurSettings.antilagSettings.luminanceSigmaScale != defaults.antilagSettings.luminanceSigmaScale)
                                isSame = false;
                            else if (m_ReblurSettings.antilagSettings.luminanceSensitivity != defaults.antilagSettings.luminanceSensitivity)
                                isSame = false;
                            else if (m_ReblurSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                isSame = false;
                            else if (m_ReblurSettings.historyFixBasePixelStride != defaults.historyFixBasePixelStride)
                                isSame = false;
                            else if (m_ReblurSettings.minBlurRadius != defaults.minBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.maxBlurRadius != defaults.maxBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.minHitDistanceWeight != defaults.minHitDistanceWeight)
                                isSame = false;
                            else if (m_ReblurSettings.lobeAngleFraction != defaults.lobeAngleFraction)
                                isSame = false;
                            else if (m_ReblurSettings.roughnessFraction != defaults.roughnessFraction)
                                isSame = false;
                            else if (m_ReblurSettings.responsiveAccumulationRoughnessThreshold != defaults.responsiveAccumulationRoughnessThreshold)
                                isSame = false;
                            else if (m_ReblurSettings.planeDistanceSensitivity != defaults.planeDistanceSensitivity)
                                isSame = false;
                            else if (m_ReblurSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                isSame = false;
                            else if (m_ReblurSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                isSame = false;
                            else if (m_ReblurSettings.enablePerformanceMode != defaults.enablePerformanceMode)
                                isSame = false;
                            else if (m_ReblurSettings.usePrepassOnlyForSpecularMotionEstimation != defaults.usePrepassOnlyForSpecularMotionEstimation)
                                isSame = false;
                            else if ((int32_t)m_ReblurSettings.maxStabilizedFrameNum < m_Settings.maxAccumulatedFrameNum)
                                isSame = false;

                            bool hasSpatial = m_ReblurSettings.minBlurRadius + m_ReblurSettings.maxBlurRadius != 0.0f
                                || m_ReblurSettings.diffusePrepassBlurRadius != 0.0f
                                || m_ReblurSettings.specularPrepassBlurRadius != 0.0f;
                            ImGui::SameLine();
                            if (ImGui::Button(hasSpatial ? "No spatial" : "Spatial")) {
                                if (hasSpatial) {
                                    m_ReblurSettings.minBlurRadius = 0.0f;
                                    m_ReblurSettings.maxBlurRadius = 0.0f;
                                    m_ReblurSettings.diffusePrepassBlurRadius = 0.0f;
                                    m_ReblurSettings.specularPrepassBlurRadius = 0.0f;
                                } else {
                                    m_ReblurSettings.minBlurRadius = defaults.minBlurRadius;
                                    m_ReblurSettings.maxBlurRadius = defaults.maxBlurRadius;
                                    m_ReblurSettings.diffusePrepassBlurRadius = defaults.diffusePrepassBlurRadius;
                                    m_ReblurSettings.specularPrepassBlurRadius = defaults.specularPrepassBlurRadius;
                                }
                            }

                            bool hasFastHistory = m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum;
                            ImGui::SameLine();
                            if (ImGui::Button(hasFastHistory ? "No fast" : "Fast")) {
                                if (hasFastHistory)
                                    m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                else
                                    m_Settings.maxFastAccumulatedFrameNum = defaults.maxFastAccumulatedFrameNum;
                            }

                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                            if (ImGui::Button("Defaults") || frameIndex == 0) {
                                m_ReblurSettings = defaults;
                                m_ReblurSettings.maxStabilizedFrameNum = m_Settings.maxAccumulatedFrameNum;
                            }
                            ImGui::PopStyleColor();

                            ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.adaptiveAccumulation ? UI_GREEN : UI_YELLOW);
                            ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);
                            ImGui::PopStyleColor();
                            ImGui::SameLine();
                            ImGui::Checkbox("Anti-firefly", &m_ReblurSettings.enableAntiFirefly);

                            ImGui::Checkbox("Performance mode", &m_ReblurSettings.enablePerformanceMode);
                            if (m_Settings.SHARC && m_Settings.adaptiveAccumulation) {
                                ImGui::SameLine();
                                ImGui::Checkbox("SHARC boost", &m_Settings.boost);
                            }
#if (NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION)
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, m_Resolve ? UI_GREEN : UI_RED);
                            ImGui::Checkbox("Resolve", &m_Resolve);
                            ImGui::PopStyleColor();
#endif

                            ImGui::BeginDisabled(m_Settings.adaptiveAccumulation);
                            ImGui::SliderInt2("Accumulation (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d");
#if (NRD_MODE != OCCLUSION)
                            ImGui::SliderInt("Stabilization (frames)", (int32_t*)&m_ReblurSettings.maxStabilizedFrameNum, 0, m_Settings.maxAccumulatedFrameNum, "%d");
#endif
                            ImGui::EndDisabled();

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
                                ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                {
                                    int32_t v = (int32_t)m_ReblurSettings.hitDistanceReconstructionMode;
                                    ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                    m_ReblurSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                }
                                ImGui::PopStyleColor();
                            }

#if (NRD_MODE < OCCLUSION)
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.diffusePrepassBlurRadius != 0.0f && m_ReblurSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                            ImGui::SliderFloat2("Pre-pass radius (px)", &m_ReblurSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PopStyleColor();
#endif

                            ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.minBlurRadius < 0.5f ? UI_RED : UI_DEFAULT);
                            ImGui::SliderFloat("Min blur radius (px)", &m_ReblurSettings.minBlurRadius, 0.0f, 10.0f, "%.1f");
                            ImGui::PopStyleColor();

                            ImGui::SliderFloat("Max blur radius (px)", &m_ReblurSettings.maxBlurRadius, 0.0f, 60.0f, "%.1f");
                            ImGui::SliderFloat("Lobe fraction", &m_ReblurSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Roughness fraction", &m_ReblurSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Min hitT weight", &m_ReblurSettings.minHitDistanceWeight, 0.01f, 0.2f, "%.2f");
                            ImGui::SliderInt("History fix frames", (int32_t*)&m_ReblurSettings.historyFixFrameNum, 0, 5);
                            ImGui::SliderInt("History fix stride", (int32_t*)&m_ReblurSettings.historyFixBasePixelStride, 1, 20);
                            ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.5f);
                            ImGui::SliderFloat("Responsive accumulation roughness threshold", &m_ReblurSettings.responsiveAccumulationRoughnessThreshold, 0.0f, 1.0f, "%.2f");

                            if (m_ReblurSettings.maxAccumulatedFrameNum && m_ReblurSettings.maxStabilizedFrameNum) {
                                ImGui::Text("ANTI-LAG:");
                                ImGui::SliderFloat("Sigma scale", &m_ReblurSettings.antilagSettings.luminanceSigmaScale, 1.0f, 5.0f, "%.1f");
                                ImGui::SliderFloat("Sensitivity", &m_ReblurSettings.antilagSettings.luminanceSensitivity, 1.0f, 5.0f, "%.1f");
                            }
                        } else if (m_Settings.denoiser == DENOISER_RELAX) {
                            nrd::RelaxSettings defaults = GetDefaultRelaxSettings();

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
                                defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                            }

                            bool isSame = true;
                            if (m_RelaxSettings.antilagSettings.accelerationAmount != defaults.antilagSettings.accelerationAmount)
                                isSame = false;
                            else if (m_RelaxSettings.antilagSettings.spatialSigmaScale != defaults.antilagSettings.spatialSigmaScale)
                                isSame = false;
                            else if (m_RelaxSettings.antilagSettings.temporalSigmaScale != defaults.antilagSettings.temporalSigmaScale)
                                isSame = false;
                            else if (m_RelaxSettings.antilagSettings.resetAmount != defaults.antilagSettings.resetAmount)
                                isSame = false;
                            else if (m_RelaxSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                isSame = false;
                            else if (m_RelaxSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                isSame = false;
                            else if (m_RelaxSettings.minHitDistanceWeight != defaults.minHitDistanceWeight)
                                isSame = false;
                            else if (m_RelaxSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                isSame = false;
                            else if (m_RelaxSettings.historyFixBasePixelStride != defaults.historyFixBasePixelStride)
                                isSame = false;
                            else if (m_RelaxSettings.historyFixEdgeStoppingNormalPower != defaults.historyFixEdgeStoppingNormalPower)
                                isSame = false;
                            else if (m_RelaxSettings.diffusePhiLuminance != defaults.diffusePhiLuminance)
                                isSame = false;
                            else if (m_RelaxSettings.specularPhiLuminance != defaults.specularPhiLuminance)
                                isSame = false;
                            else if (m_RelaxSettings.lobeAngleFraction != defaults.lobeAngleFraction)
                                isSame = false;
                            else if (m_RelaxSettings.roughnessFraction != defaults.roughnessFraction)
                                isSame = false;
                            else if (m_RelaxSettings.specularVarianceBoost != defaults.specularVarianceBoost)
                                isSame = false;
                            else if (m_RelaxSettings.specularLobeAngleSlack != defaults.specularLobeAngleSlack)
                                isSame = false;
                            else if (m_RelaxSettings.historyClampingColorBoxSigmaScale != defaults.historyClampingColorBoxSigmaScale)
                                isSame = false;
                            else if (m_RelaxSettings.spatialVarianceEstimationHistoryThreshold != defaults.spatialVarianceEstimationHistoryThreshold)
                                isSame = false;
                            else if (m_RelaxSettings.atrousIterationNum != defaults.atrousIterationNum)
                                isSame = false;
                            else if (m_RelaxSettings.diffuseMinLuminanceWeight != defaults.diffuseMinLuminanceWeight)
                                isSame = false;
                            else if (m_RelaxSettings.specularMinLuminanceWeight != defaults.specularMinLuminanceWeight)
                                isSame = false;
                            else if (m_RelaxSettings.depthThreshold != defaults.depthThreshold)
                                isSame = false;
                            else if (m_RelaxSettings.confidenceDrivenRelaxationMultiplier != defaults.confidenceDrivenRelaxationMultiplier)
                                isSame = false;
                            else if (m_RelaxSettings.confidenceDrivenLuminanceEdgeStoppingRelaxation != defaults.confidenceDrivenLuminanceEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.confidenceDrivenNormalEdgeStoppingRelaxation != defaults.confidenceDrivenNormalEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.luminanceEdgeStoppingRelaxation != defaults.luminanceEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.normalEdgeStoppingRelaxation != defaults.normalEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.roughnessEdgeStoppingRelaxation != defaults.roughnessEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                isSame = false;
                            else if (m_RelaxSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                isSame = false;
                            else if (m_RelaxSettings.enableRoughnessEdgeStopping != defaults.enableRoughnessEdgeStopping)
                                isSame = false;

                            bool hasSpatial = m_RelaxSettings.diffusePhiLuminance != 0.0f
                                || m_RelaxSettings.specularPhiLuminance != 0.0f
                                || m_RelaxSettings.diffusePrepassBlurRadius != 0.0f
                                || m_RelaxSettings.specularPrepassBlurRadius != 0.0f
                                || m_RelaxSettings.spatialVarianceEstimationHistoryThreshold != 0;
                            ImGui::SameLine();
                            if (ImGui::Button(hasSpatial ? "No spatial" : "Spatial")) {
                                if (hasSpatial) {
                                    m_RelaxSettings.diffusePhiLuminance = 0.0f;
                                    m_RelaxSettings.specularPhiLuminance = 0.0f;
                                    m_RelaxSettings.diffusePrepassBlurRadius = 0.0f;
                                    m_RelaxSettings.specularPrepassBlurRadius = 0.0f;
                                    m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = 0;
                                } else {
                                    m_RelaxSettings.diffusePhiLuminance = defaults.diffusePhiLuminance;
                                    m_RelaxSettings.specularPhiLuminance = defaults.specularPhiLuminance;
                                    m_RelaxSettings.diffusePrepassBlurRadius = defaults.diffusePrepassBlurRadius;
                                    m_RelaxSettings.specularPrepassBlurRadius = defaults.specularPrepassBlurRadius;
                                    m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = defaults.spatialVarianceEstimationHistoryThreshold;
                                }
                            }

                            bool hasFastHistory = m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum;
                            ImGui::SameLine();
                            if (ImGui::Button(hasFastHistory ? "No fast" : "Fast")) {
                                if (hasFastHistory)
                                    m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                else
                                    m_Settings.maxFastAccumulatedFrameNum = defaults.diffuseMaxFastAccumulatedFrameNum;
                            }

                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                            if (ImGui::Button("Defaults") || frameIndex == 0)
                                m_RelaxSettings = defaults;
                            ImGui::PopStyleColor();

                            ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.adaptiveAccumulation ? UI_GREEN : UI_YELLOW);
                            ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);
                            ImGui::PopStyleColor();
                            ImGui::SameLine();
                            ImGui::Checkbox("Anti-firefly", &m_RelaxSettings.enableAntiFirefly);

                            ImGui::Checkbox("Roughness edge stopping", &m_RelaxSettings.enableRoughnessEdgeStopping);
                            if (m_Settings.SHARC) {
                                ImGui::SameLine();
                                ImGui::Checkbox("SHARC boost", &m_Settings.boost);
                            }
#if (NRD_MODE == SH)
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, m_Resolve ? UI_GREEN : UI_RED);
                            ImGui::Checkbox("Resolve", &m_Resolve);
                            ImGui::PopStyleColor();
#endif

                            ImGui::BeginDisabled(m_Settings.adaptiveAccumulation);
                            ImGui::SliderInt2("Accumulation (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d");
                            ImGui::EndDisabled();

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
                                ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                {
                                    int32_t v = (int32_t)m_RelaxSettings.hitDistanceReconstructionMode;
                                    ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                    m_RelaxSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                }
                                ImGui::PopStyleColor();
                            }

#if (NRD_MODE < OCCLUSION)
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.diffusePrepassBlurRadius != 0.0f && m_RelaxSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                            ImGui::SliderFloat2("Pre-pass radius (px)", &m_RelaxSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PopStyleColor();
#endif

                            ImGui::SliderInt("A-trous iterations", (int32_t*)&m_RelaxSettings.atrousIterationNum, 2, 8);
                            ImGui::SliderFloat2("Diff-Spec luma weight", &m_RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderFloat2("Min luma weight", &m_RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Depth threshold", &m_RelaxSettings.depthThreshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                            ImGui::SliderFloat("Lobe fraction", &m_RelaxSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Roughness fraction", &m_RelaxSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Min hitT weight", &m_RelaxSettings.minHitDistanceWeight, 0.01f, 0.2f, "%.2f");
                            ImGui::SliderFloat("Spec variance boost", &m_RelaxSettings.specularVarianceBoost, 0.0f, 8.0f, "%.2f");
                            ImGui::SliderFloat("Clamping sigma scale", &m_RelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderInt("History threshold", (int32_t*)&m_RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, 10);
                            ImGui::Text("Luminance / Normal / Roughness:");
                            ImGui::SliderFloat3("Relaxation", &m_RelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f, "%.2f");

                            ImGui::Text("HISTORY FIX:");
                            ImGui::SliderFloat("Normal weight power", &m_RelaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 128.0f, "%.1f");
                            ImGui::SliderInt("Frames", (int32_t*)&m_RelaxSettings.historyFixFrameNum, 0, 5);
                            ImGui::SliderInt("Stride", (int32_t*)&m_RelaxSettings.historyFixBasePixelStride, 1, 20);

                            ImGui::Text("ANTI-LAG:");
                            ImGui::SliderFloat("Acceleration amount", &m_RelaxSettings.antilagSettings.accelerationAmount, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat2("S/T sigma scales", &m_RelaxSettings.antilagSettings.spatialSigmaScale, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderFloat("Reset amount", &m_RelaxSettings.antilagSettings.resetAmount, 0.0f, 1.0f, "%.2f");
                        } else if (m_Settings.denoiser == DENOISER_REFERENCE) {
                            float t = (float)m_ReferenceSettings.maxAccumulatedFrameNum;
                            ImGui::SliderFloat("Accumulation (frames)", &t, 0.0f, nrd::REFERENCE_MAX_HISTORY_FRAME_NUM, "%.0f", ImGuiSliderFlags_Logarithmic);
                            m_ReferenceSettings.maxAccumulatedFrameNum = (int32_t)t;
                        }
                    }
                    ImGui::PopID();

                    // NRD/SIGMA
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("NRD/SIGMA", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("NRD/SIGMA");
                    if (m_Settings.RR)
                        ImGui::Text("Pass-through mode...");
                    else if (isUnfolded) {
                        ImGui::BeginDisabled(m_Settings.adaptiveAccumulation);
                        ImGui::SliderInt("Stabilization (frames)", (int32_t*)&m_SigmaSettings.maxStabilizedFrameNum, 0, nrd::SIGMA_MAX_HISTORY_FRAME_NUM, "%d");
                        ImGui::EndDisabled();
                    }
                    ImGui::PopID();

                    // "Other" section
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("OTHER", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("OTHER");
                    if (isUnfolded) {
                        ImGui::SliderFloat("Debug [F1]", &m_Settings.debug, 0.0f, 1.0f, "%.6f");
                        ImGui::SliderFloat("Input / Denoised", &m_Settings.separator, 0.0f, 1.0f, "%.2f");

                        if (ImGui::Button(m_Settings.windowAlignment ? ">>" : "<<"))
                            m_Settings.windowAlignment = !m_Settings.windowAlignment;

                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, m_IsReloadShadersSucceeded ? UI_DEFAULT : UI_RED);
                        if (ImGui::Button("Reload shaders")) {
                            int result = 0;
#ifdef _WIN32 // TODO: can be made Linux friendly too
#    ifdef _DEBUG
                            std::string sampleShaders = "_Bin\\Debug\\ShaderMake.exe";
                            std::string nrdShaders = "_Bin\\Debug\\ShaderMake.exe";
#    else
                            std::string sampleShaders = "_Bin\\Release\\ShaderMake.exe";
                            std::string nrdShaders = "_Bin\\Release\\ShaderMake.exe";
#    endif

                            sampleShaders +=
                                " --useAPI --flatten --stripReflection --WX --colorize"
                                " --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128"
                                " --binary"
                                " --shaderModel 6_6"
                                " --sourceDir Shaders"
                                " --ignoreConfigDir"
                                " -c Shaders/Shaders.cfg"
                                " -o _Shaders"
                                " -I Shaders"
                                " -I External"
                                " -I " STRINGIFY(ML_SOURCE_DIR) " -I " STRINGIFY(NRD_SOURCE_DIR) " -I " STRINGIFY(NRI_SOURCE_DIR) " -I " STRINGIFY(SHARC_SOURCE_DIR) " -D NRD_NORMAL_ENCODING=" STRINGIFY(NRD_NORMAL_ENCODING) " -D NRD_ROUGHNESS_ENCODING=" STRINGIFY(NRD_ROUGHNESS_ENCODING);

                            nrdShaders +=
                                " --useAPI --flatten --stripReflection --WX --colorize"
                                " --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128"
                                " --binary --header"
                                " --allResourcesBound"
                                " --vulkanVersion 1.2"
                                " --sourceDir External/NRD/Shaders/Source"
                                " --ignoreConfigDir"
                                " -c External/NRD/Shaders/Shaders.cfg"
                                " -o _Shaders"
                                " -I " STRINGIFY(ML_SOURCE_DIR)
                                    " -I External/NRD/Shaders/Include"
                                    " -I External/NRD/Shaders/Resources"
                                    " -D NRD_NORMAL_ENCODING=" STRINGIFY(NRD_NORMAL_ENCODING) " -D NRD_ROUGHNESS_ENCODING=" STRINGIFY(NRD_ROUGHNESS_ENCODING) " -D NRD_INTERNAL";

                            if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12) {
                                std::string dxil = " -p DXIL --compiler \"" STRINGIFY(DXC_PATH) "\"";
                                sampleShaders += dxil;
                                nrdShaders += dxil;
                            } else {
                                std::string spirv = " -p SPIRV --compiler \"" STRINGIFY(DXC_SPIRV_PATH) "\" --hlsl2021";
                                sampleShaders += spirv;
                                nrdShaders += spirv;
                            }

                            printf("Compiling sample shaders...\n");
                            result = system(sampleShaders.c_str());
                            if (!result) {
                                printf("Compiling NRD shaders...\n");
                                result = system(nrdShaders.c_str());
                            }

                            if (result)
                                SetForegroundWindow(GetConsoleWindow());

                            m_IsReloadShadersSucceeded = !result;

#    undef SAMPLE_SHADERS
#    undef NRD_SHADERS
#endif

                            if (!result)
                                CreatePipelines();

                            printf("Ready!\n");
                        }
                        ImGui::PopStyleColor();

                        ImGui::SameLine();
                        if (ImGui::Button("Defaults")) {
                            m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                            m_Settings = m_SettingsDefault;
                            m_RelaxSettings = GetDefaultRelaxSettings();
                            m_ReblurSettings = GetDefaultReblurSettings();
                            m_ForceHistoryReset = true;
                        }
                    }
                    ImGui::PopID();

                    // "Tests" section
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("TESTS [F2]", ImGuiTreeNodeFlags_CollapsingHeader);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("TESTS");
                    if (isUnfolded) {
                        float buttonWidth = 25.0f * float(GetWindowResolution().x) / float(GetOutputResolution().x);

                        char s[64];
                        std::string sceneName = std::string(utils::GetFileName(m_SceneFile));
                        size_t dotPos = sceneName.find_last_of(".");
                        if (dotPos != std::string::npos)
                            sceneName = sceneName.substr(0, dotPos) + ".bin";
                        const std::string path = utils::GetFullPath(sceneName, utils::DataFolder::TESTS);
                        const uint32_t testByteSize = sizeof(m_Settings) + Camera::GetStateSize();

                        // Get number of tests
                        if (m_TestNum == uint32_t(-1)) {
                            FILE* fp = fopen(path.c_str(), "rb");
                            if (fp) {
// Use this code to convert tests to reflect new Settings and Camera layouts
#if 0
                                    typedef Settings SettingsOld; // adjust if needed
                                    typedef Camera CameraOld; // adjust if needed

                                    const uint32_t oldItemSize = sizeof(SettingsOld) + CameraOld::GetStateSize();

                                    fseek(fp, 0, SEEK_END);
                                    m_TestNum = ftell(fp) / oldItemSize;
                                    fseek(fp, 0, SEEK_SET);

                                    FILE* fpNew;
                                    fopen_s(&fpNew, (path + ".new").c_str(), "wb");

                                    for (uint32_t i = 0; i < m_TestNum && fpNew; i++)
                                    {
                                        SettingsOld settingsOld;
                                        fread_s(&settingsOld, sizeof(SettingsOld), 1, sizeof(SettingsOld), fp);

                                        CameraOld cameraOld;
                                        fread_s(cameraOld.GetState(), CameraOld::GetStateSize(), 1, CameraOld::GetStateSize(), fp);

                                        // Convert Old to New here
                                        m_Settings = settingsOld;
                                        m_Camera.state = cameraOld.state;

                                        // ...

                                        fwrite(&m_Settings, 1, sizeof(m_Settings), fpNew);
                                        fwrite(m_Camera.GetState(), 1, Camera::GetStateSize(), fpNew);
                                    }

                                    fclose(fp);
                                    fclose(fpNew);

                                    __debugbreak();
#endif

                                fseek(fp, 0, SEEK_END);
                                m_TestNum = ftell(fp) / testByteSize;
                                fclose(fp);
                            } else
                                m_TestNum = 0;
                        }

                        // Adjust current test index
                        bool isTestChanged = false;
                        if (IsKeyToggled(Key::F2) && m_TestNum) {
                            m_LastSelectedTest++;
                            isTestChanged = true;
                        }

                        if (m_LastSelectedTest == uint32_t(-1) || !m_TestNum)
                            m_LastSelectedTest = uint32_t(-1);
                        else
                            m_LastSelectedTest %= m_TestNum;

                        // Main buttons
                        uint32_t i = 0;
                        for (; i < m_TestNum; i++) {
                            snprintf(s, sizeof(s), "%u", i + 1);

                            if (i % 14 != 0)
                                ImGui::SameLine();

                            bool isColorChanged = false;
                            if (m_improveMeTests && std::find(m_improveMeTests->begin(), m_improveMeTests->end(), i + 1) != m_improveMeTests->end()) {
                                ImGui::PushStyleColor(ImGuiCol_Text, UI_RED);
                                isColorChanged = true;
                            } else if (m_checkMeTests && std::find(m_checkMeTests->begin(), m_checkMeTests->end(), i + 1) != m_checkMeTests->end()) {
                                ImGui::PushStyleColor(ImGuiCol_Text, UI_YELLOW);
                                isColorChanged = true;
                            }

                            if (ImGui::Button(i == m_LastSelectedTest ? "*" : s, ImVec2(buttonWidth, 0.0f)) || isTestChanged) {
                                uint32_t test = isTestChanged ? m_LastSelectedTest : i;
                                FILE* fp = fopen(path.c_str(), "rb");

                                if (fp && fseek(fp, test * testByteSize, SEEK_SET) == 0) {
                                    size_t elemNum = fread(&m_Settings, sizeof(m_Settings), 1, fp);
                                    if (elemNum == 1)
                                        elemNum = fread(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);

                                    m_LastSelectedTest = test;

                                    // File read error
                                    if (elemNum != 1) {
                                        m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                                        m_Settings = m_SettingsDefault;
                                    }

                                    // Reset some settings to defaults to avoid a potential confusion
                                    m_Settings.debug = 0.0f;
                                    m_Settings.denoiser = DENOISER_REBLUR;
                                    m_Settings.RR = m_DLRR;
                                    m_Settings.SR = m_DLSR;
                                    m_Settings.TAA = true;
                                    m_Settings.cameraJitter = true;
                                    m_Settings.onScreen = clamp(m_Settings.onScreen, 0, (int32_t)helper::GetCountOf(onScreenModes));

                                    m_ForceHistoryReset = true;
                                }

                                if (fp)
                                    fclose(fp);

                                isTestChanged = false;
                            }

                            if (isColorChanged)
                                ImGui::PopStyleColor();
                        }

                        if (i % 14 != 0)
                            ImGui::SameLine();

                        // "Add" button
                        if (ImGui::Button("Add")) {
                            FILE* fp = fopen(path.c_str(), "ab");

                            if (fp) {
                                m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? -1.0 : 0.0;

                                fwrite(&m_Settings, sizeof(m_Settings), 1, fp);
                                fwrite(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);
                                fclose(fp);

                                m_TestNum = uint32_t(-1);
                            }
                        }

                        if ((i + 1) % 14 != 0)
                            ImGui::SameLine();

                        // "Del" button
                        snprintf(s, sizeof(s), "Del %u", m_LastSelectedTest + 1);
                        if (m_TestNum != uint32_t(-1) && m_LastSelectedTest != uint32_t(-1) && ImGui::Button(s)) {
                            std::vector<uint8_t> data;
                            utils::LoadFile(path, data);

                            FILE* fp = fopen(path.c_str(), "wb");

                            if (fp) {
                                for (i = 0; i < m_TestNum; i++) {
                                    if (i != m_LastSelectedTest)
                                        fwrite(&data[i * testByteSize], 1, testByteSize, fp);
                                }

                                fclose(fp);

                                m_TestNum = uint32_t(-1);
                            }
                        }
                    }
                    ImGui::PopID();
                }
            }
            m_UiWidth = ImGui::GetWindowWidth();
        }
        ImGui::End();
    }
    EndUI();

    // Animate scene and update camera
    cBoxf cameraLimits = m_Scene.aabb;
    cameraLimits.Scale(2.0f);

    CameraDesc desc = {};
    desc.limits = cameraLimits;
    desc.aspectRatio = float(GetOutputResolution().x) / float(GetOutputResolution().y);
    desc.horizontalFov = degrees(atan(tan(radians(m_Settings.camFov) * 0.5f) * desc.aspectRatio * 9.0f / 16.0f) * 2.0f); // recalculate to ultra-wide if needed
    desc.nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    desc.farZ = 10000.0f * m_Settings.meterToUnitsMultiplier;
    desc.isCustomMatrixSet = false; // No camera animation hooked up
    desc.isPositiveZ = m_PositiveZ;
    desc.isReversedZ = m_ReversedZ;
    desc.orthoRange = m_Settings.ortho ? tan(radians(m_Settings.camFov) * 0.5f) * 3.0f * m_Settings.meterToUnitsMultiplier : 0.0f;
    desc.backwardOffset = CAMERA_BACKWARD_OFFSET;
    GetCameraDescFromInputDevices(desc);

    if (m_Settings.motionStartTime > 0.0) {
        float time = float(m_Timer.GetTimeStamp() - m_Settings.motionStartTime);
        float amplitude = 40.0f * m_Camera.state.motionScale;
        float period = 0.0003f * time * (m_Settings.emulateMotionSpeed < 0.0f ? 1.0f / (1.0f + abs(m_Settings.emulateMotionSpeed)) : (1.0f + m_Settings.emulateMotionSpeed));

        float3 localPos = m_Camera.state.mWorldToView.Row(0).xyz;
        if (m_Settings.motionMode == 1)
            localPos = m_Camera.state.mWorldToView.Row(1).xyz;
        else if (m_Settings.motionMode == 2)
            localPos = m_Camera.state.mWorldToView.Row(2).xyz;
        else if (m_Settings.motionMode == 3) {
            float3 rows[3] = {m_Camera.state.mWorldToView.Row(0).xyz, m_Camera.state.mWorldToView.Row(1).xyz, m_Camera.state.mWorldToView.Row(2).xyz};
            float f = sin(Pi(period * 3.0f));
            localPos = normalize(f < 0.0f ? lerp(rows[1], rows[0], float3(abs(f))) : lerp(rows[1], rows[2], float3(f)));
        }

        if (m_Settings.motionMode == 4) {
            float3 axisX = m_Camera.state.mWorldToView.Row(0).xyz;
            float3 axisY = m_Camera.state.mWorldToView.Row(1).xyz;
            float2 v = Rotate(float2(1.0f, 0.0f), fmod(Pi(period * 2.0f), Pi(2.0f)));
            localPos = (axisX * v.x + axisY * v.y) * amplitude / Pi(1.0f);
        } else
            localPos *= amplitude * (m_Settings.linearMotion ? WaveTriangle(period) - 0.5f : sin(Pi(period)) * 0.5f);

        desc.dUser = localPos - m_PrevLocalPos;
        m_PrevLocalPos = localPos;
    } else if (m_Settings.motionStartTime == -1.0) {
        m_Settings.motionStartTime = m_Timer.GetTimeStamp();
        m_PrevLocalPos = float3::Zero();
    }

    m_Camera.Update(desc, frameIndex);

    // Animate scene
    const float animationSpeed = m_Settings.pauseAnimation ? 0.0f : (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
    const float animationDelta = animationSpeed * m_Timer.GetFrameTime() * 0.001f;

    for (size_t i = 0; i < m_Scene.animations.size(); i++)
        m_Scene.Animate(animationSpeed, m_Timer.GetFrameTime(), m_Settings.animationProgress, (int32_t)i);

    // Animate sun
    if (m_Settings.animateSun) {
        static float sunAzimuthPrev = 0.0f;
        static double sunMotionStartTime = 0.0;
        if (m_Settings.animateSun != m_SettingsPrev.animateSun) {
            sunAzimuthPrev = m_Settings.sunAzimuth;
            sunMotionStartTime = m_Timer.GetTimeStamp();
        }
        double t = m_Timer.GetTimeStamp() - sunMotionStartTime;
        if (!m_Settings.pauseAnimation)
            m_Settings.sunAzimuth = sunAzimuthPrev + (float)sin(t * animationSpeed * 0.0003) * 10.0f;
    }

    // Animate objects
    const float scale = m_Settings.animatedObjectScale * m_Settings.meterToUnitsMultiplier / 2.0f;
    if (m_Settings.nineBrothers) {
        const float3& vRight = m_Camera.state.mViewToWorld[0].xyz;
        const float3& vTop = m_Camera.state.mViewToWorld[1].xyz;
        const float3& vForward = m_Camera.state.mViewToWorld[2].xyz;

        float3 basePos = float3(m_Camera.state.globalPosition);

#if (USE_CAMERA_ATTACHED_REFLECTION_TEST == 1)
        m_Settings.animatedObjectNum = 3;

        for (int32_t i = -1; i <= 1; i++) {
            const uint32_t index = i + 1;

            float x = float(i) * 3.0f;
            float y = (i == 0) ? -1.5f : 0.0f;
            float z = (i == 0) ? 1.0f : 3.0f;

            x *= scale;
            y *= scale;
            z *= m_PositiveZ ? scale : -scale;

            float3 pos = basePos + vRight * x + vTop * y + vForward * z;

            utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[index].instanceID];
            instance.position = double3(pos);
            instance.rotation = m_Camera.state.mViewToWorld;
            instance.rotation.SetTranslation(float3::Zero());
            instance.rotation.AddScale(scale);
        }
#else
        m_Settings.animatedObjectNum = 9;

        for (int32_t i = -1; i <= 1; i++) {
            for (int32_t j = -1; j <= 1; j++) {
                const uint32_t index = (i + 1) * 3 + (j + 1);

                float x = float(i) * scale * 4.0f;
                float y = float(j) * scale * 4.0f;
                float z = 10.0f * (m_PositiveZ ? scale : -scale);

                float3 pos = basePos + vRight * x + vTop * y + vForward * z;

                utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[index].instanceID];
                instance.position = double3(pos);
                instance.rotation = m_Camera.state.mViewToWorld;
                instance.rotation.SetTranslation(float3::Zero());
                instance.rotation.AddScale(scale);
            }
        }
#endif
    } else if (m_Settings.animatedObjects) {
        for (int32_t i = 0; i < m_Settings.animatedObjectNum; i++) {
            float3 position;
            float4x4 transform = m_AnimatedInstances[i].Animate(animationDelta, scale, position);

            utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[i].instanceID];
            instance.rotation = transform;
            instance.position = double3(position);
        }
    }

    // Adjust settings if tracing mode has been changed to / from "probabilistic sampling"
    if (m_Settings.tracingMode != m_SettingsPrev.tracingMode && (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC || m_SettingsPrev.tracingMode == RESOLUTION_FULL_PROBABILISTIC)) {
        nrd::ReblurSettings reblurDefaults = {};
        nrd::ReblurSettings relaxDefaults = {};

        if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        } else {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.diffusePrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.diffusePrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        }
    }

    // Print out information
    if (m_SettingsPrev.resolutionScale != m_Settings.resolutionScale || m_SettingsPrev.tracingMode != m_Settings.tracingMode || m_SettingsPrev.rpp != m_Settings.rpp || frameIndex == 0) {
        std::array<uint32_t, 4> rppScale = {2, 1, 2, 2};
        std::array<float, 4> wScale = {1.0f, 1.0f, 0.5f, 0.5f};
        std::array<float, 4> hScale = {1.0f, 1.0f, 1.0f, 0.5f};

        uint32_t pw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
        uint32_t ph = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
        uint32_t iw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale * wScale[m_Settings.tracingMode] + 0.5f);
        uint32_t ih = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale * hScale[m_Settings.tracingMode] + 0.5f);
        uint32_t rayNum = m_Settings.rpp * rppScale[m_Settings.tracingMode];
        float rpp = float(iw * ih * rayNum) / float(pw * ph);

        printf(
            "Output          : %ux%u\n"
            "  Primary rays  : %ux%u\n"
            "  Indirect rays : %ux%u x %u ray(s)\n"
            "  Indirect rpp  : %.2f\n",
            GetOutputResolution().x, GetOutputResolution().y,
            pw, ph,
            iw, ih, rayNum,
            rpp);
    }

    if (m_SettingsPrev.denoiser != m_Settings.denoiser || m_SettingsPrev.RR != m_Settings.RR || frameIndex == 0) {
        m_checkMeTests = nullptr;
        m_improveMeTests = nullptr;

        if (m_SceneFile.find("BistroInterior") != std::string::npos) {
            m_checkMeTests = &interior_checkMeTests;

            if (m_Settings.denoiser == DENOISER_REBLUR)
                m_improveMeTests = &REBLUR_interior_improveMeTests;
            else if (m_Settings.denoiser == DENOISER_RELAX)
                m_improveMeTests = &RELAX_interior_improveMeTests;

            if (m_Settings.RR)
                m_improveMeTests = &DLRR_interior_improveMeTests;
        }
    }

    // Global history reset
    if (m_SettingsPrev.denoiser != m_Settings.denoiser)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.denoiser == DENOISER_REFERENCE && m_SettingsPrev.tracingMode != m_Settings.tracingMode)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.ortho != m_Settings.ortho)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.onScreen != m_Settings.onScreen)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.RR != m_Settings.RR)
        m_ForceHistoryReset = true;
    if (frameIndex == 0)
        m_ForceHistoryReset = true;

    float sunCurr = smoothstep(-0.9f, 0.05f, sin(radians(m_Settings.sunElevation)));
    float sunPrev = smoothstep(-0.9f, 0.05f, sin(radians(m_SettingsPrev.sunElevation)));
    float resetHistoryFactor = 1.0f - smoothstep(0.0f, 0.2f, abs(sunCurr - sunPrev));

    float emiCurr = m_Settings.emission * m_Settings.emissionIntensity;
    float emiPrev = m_SettingsPrev.emission * m_SettingsPrev.emissionIntensity;
    if (emiCurr != emiPrev)
        resetHistoryFactor *= lerp(1.0f, 0.5f, abs(emiCurr - emiPrev) / max(emiCurr, emiPrev));

    if (m_ForceHistoryReset)
        resetHistoryFactor = 0.0f;

    // NRD common settings
    if (m_Settings.adaptiveAccumulation) {
        bool isFastHistoryEnabled = m_Settings.maxAccumulatedFrameNum > m_Settings.maxFastAccumulatedFrameNum;
        float fps = 1000.0f / m_Timer.GetVerySmoothedFrameTime();
        fps = min(fps, 121.0f);

        // REBLUR / RELAX
        float accumulationTime = nrd::REBLUR_DEFAULT_ACCUMULATION_TIME * ((m_Settings.boost && m_Settings.SHARC) ? 0.667f : 1.0f);
        int32_t maxAccumulatedFrameNum = max(nrd::GetMaxAccumulatedFrameNum(accumulationTime, fps), 1u);

        m_Settings.maxAccumulatedFrameNum = min(maxAccumulatedFrameNum, MAX_HISTORY_FRAME_NUM);
        m_Settings.maxFastAccumulatedFrameNum = isFastHistoryEnabled ? m_Settings.maxAccumulatedFrameNum / 5 : MAX_HISTORY_FRAME_NUM;

        m_ReblurSettings.maxStabilizedFrameNum = m_Settings.maxAccumulatedFrameNum;

        // SIGMA
        uint32_t maxSigmaStabilizedFrames = nrd::GetMaxAccumulatedFrameNum(nrd::SIGMA_DEFAULT_ACCUMULATION_TIME, fps);

        m_SigmaSettings.maxStabilizedFrameNum = min(maxSigmaStabilizedFrames, nrd::SIGMA_MAX_HISTORY_FRAME_NUM);
    }

    uint32_t maxAccumulatedFrameNum = uint32_t(m_Settings.maxAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    uint32_t maxFastAccumulatedFrameNum = uint32_t(m_Settings.maxFastAccumulatedFrameNum * resetHistoryFactor + 0.5f);

    m_ReblurSettings.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_ReblurSettings.maxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
    m_ReblurSettings.checkerboardMode = (m_Settings.tracingMode == RESOLUTION_HALF && !m_Settings.RR) ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
    m_ReblurSettings.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
    m_ReblurSettings.minMaterialForSpecular = MATERIAL_ID_METAL;

    m_RelaxSettings.diffuseMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
    m_RelaxSettings.specularMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_RelaxSettings.specularMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
    m_RelaxSettings.checkerboardMode = (m_Settings.tracingMode == RESOLUTION_HALF && !m_Settings.RR) ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
    m_RelaxSettings.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
    m_RelaxSettings.minMaterialForSpecular = MATERIAL_ID_METAL;

    UpdateConstantBuffer(frameIndex, resetHistoryFactor);
    GatherInstanceData();

    nri::nriEndAnnotation();
}

void Sample::LoadScene() {
    // Proxy geometry, which will be instancinated
    std::string sceneFile = utils::GetFullPath("Cubes/Cubes.gltf", utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING));

    m_ProxyInstancesNum = helper::GetCountOf(m_Scene.instances);

    // The scene
    sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING));

    // Some scene dependent settings
    m_ReblurSettings = GetDefaultReblurSettings();
    m_RelaxSettings = GetDefaultRelaxSettings();

    if (m_SceneFile.find("BistroInterior") != std::string::npos) {
        m_Settings.exposure = 80.0f;
        m_Settings.emission = true;
        m_Settings.animatedObjectScale = 0.5f;
        m_Settings.sunElevation = 7.0f;
    } else if (m_SceneFile.find("BistroExterior") != std::string::npos) {
        m_Settings.exposure = 50.0f;
        m_Settings.emission = true;
    } else if (m_SceneFile.find("Hair") != std::string::npos) {
        m_Settings.exposure = 2.0f;
        m_Settings.bounceNum = 4;
    } else if (m_SceneFile.find("ShaderBalls") != std::string::npos)
        m_Settings.exposure = 1.7f;
}

void Sample::AddInnerGlassSurfaces() {
    // IMPORTANT: this is only valid for non-merged instances, when each instance represents a single object
    // TODO: try thickness emulation in TraceTransparent shader

    size_t instanceNum = m_Scene.instances.size();
    for (size_t i = 0; i < instanceNum; i++) {
        const utils::Instance& instance = m_Scene.instances[i];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        // Skip non-transparent objects
        if (!material.IsTransparent())
            continue;

        const utils::MeshInstance& meshInstance = m_Scene.meshInstances[instance.meshInstanceIndex];
        const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];
        float3 size = mesh.aabb.vMax - mesh.aabb.vMin;
        size *= instance.rotation.GetScale();

        // Skip too thin objects
        float minSize = min(size.x, min(size.y, size.z));
        if (minSize < GLASS_THICKNESS * 2.0f)
            continue;

        // Skip objects, which look "merged"
        /*
        float maxSize = max(size.x, max(size.y, size.z));
        if (maxSize > 0.5f)
            continue;
        */

        utils::Instance innerInstance = instance;
        innerInstance.scale = (size - GLASS_THICKNESS) / (size + 1e-15f);

        m_Scene.instances.push_back(innerInstance);
    }
}

void Sample::GenerateAnimatedCubes() {
    for (uint32_t i = 0; i < MAX_ANIMATED_INSTANCE_NUM; i++) {
        float3 position = lerp(m_Scene.aabb.vMin, m_Scene.aabb.vMax, Rng::Hash::GetFloat4(m_RngState).xyz);

        AnimatedInstance animatedInstance = {};
        animatedInstance.instanceID = helper::GetCountOf(m_Scene.instances);
        animatedInstance.basePosition = position;
        animatedInstance.durationSec = Rng::Hash::GetFloat(m_RngState) * 10.0f + 5.0f;
        animatedInstance.progressedSec = animatedInstance.durationSec * Rng::Hash::GetFloat(m_RngState);
        animatedInstance.rotationAxis = normalize(float3(Rng::Hash::GetFloat4(m_RngState).xyz) * 2.0f - 1.0f);
        animatedInstance.elipseAxis = (float3(Rng::Hash::GetFloat4(m_RngState).xyz) * 2.0f - 1.0f) * 5.0f;
        animatedInstance.reverseDirection = Rng::Hash::GetFloat(m_RngState) < 0.5f;
        animatedInstance.reverseRotation = Rng::Hash::GetFloat(m_RngState) < 0.5f;
        m_AnimatedInstances.push_back(animatedInstance);

        utils::Instance instance = m_Scene.instances[i % m_ProxyInstancesNum];
        instance.allowUpdate = true;

        m_Scene.instances.push_back(instance);
    }
}

nri::Format Sample::CreateSwapChain() {
    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.window = GetWindow();
    swapChainDesc.queue = m_GraphicsQueue;
    swapChainDesc.format = ALLOW_HDR ? nri::SwapChainFormat::BT709_G10_16BIT : nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.verticalSyncInterval = m_VsyncInterval;
    swapChainDesc.width = (uint16_t)GetWindowResolution().x;
    swapChainDesc.height = (uint16_t)GetWindowResolution().y;
    swapChainDesc.textureNum = GetSwapChainFrameNum();
    swapChainDesc.queuedFrameNum = GetQueuedFrameNum();

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));
    m_IsSrgb = swapChainDesc.format != nri::SwapChainFormat::BT709_G10_16BIT;

    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
    const nri::TextureDesc& swapChainTextureDesc = NRI.GetTextureDesc(*swapChainTextures[0]);
    nri::Format swapChainFormat = swapChainTextureDesc.format;

    m_SwapChainBuffers.clear();
    for (uint32_t i = 0; i < swapChainTextureNum; i++) {
        m_SwapChainBuffers.emplace_back();
        BackBuffer& backBuffer = m_SwapChainBuffers.back();

        backBuffer = {};
        backBuffer.texture = swapChainTextures[i];
        backBuffer.attachmentFormat = swapChainFormat;

        char name[32];
        snprintf(name, sizeof(name), "Texture::SwapChain#%u", i);
        NRI.SetDebugName(backBuffer.texture, name);

        nri::Texture2DViewDesc textureViewDesc = {backBuffer.texture, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, backBuffer.colorAttachment));
    }

    return swapChainFormat;
}

void Sample::CreateCommandBuffers() {
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }
}

void Sample::CreatePipelineLayoutAndDescriptorPool() {
    // SET_GLOBAL
    const nri::DescriptorRangeDesc descriptorRanges0[] = {
        {0, 3, nri::DescriptorType::SAMPLER, nri::StageBits::COMPUTE_SHADER},
    };

    // SET_OTHER
    const nri::DescriptorRangeDesc descriptorRanges1[] = {
        {0, 12, nri::DescriptorType::TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
        {0, 13, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
    };

    // SET_RAY_TRACING
    const uint32_t textureNum = helper::GetCountOf(m_Scene.materials) * TEXTURES_PER_MATERIAL;
    nri::DescriptorRangeDesc descriptorRanges2[] = {
        {0, 2, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::StageBits::COMPUTE_SHADER},
        {2, 3, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
        {5, textureNum, nri::DescriptorType::TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND | nri::DescriptorRangeBits::VARIABLE_SIZED_ARRAY},
    };

    // SET_MORPH
    const nri::DescriptorRangeDesc descriptorRanges3[] = {
        {0, 3, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
        {0, 2, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
    };

    // SET_SHARC
    const nri::DescriptorRangeDesc descriptorRanges4[] = {
        {0, 4, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
    };

    nri::DynamicConstantBufferDesc dynamicConstantBuffer = {0, nri::StageBits::COMPUTE_SHADER};

    const nri::DescriptorSetDesc descriptorSetDescs[] = {
        {SET_GLOBAL, descriptorRanges0, helper::GetCountOf(descriptorRanges0), &dynamicConstantBuffer, 1},
        {SET_OTHER, descriptorRanges1, helper::GetCountOf(descriptorRanges1), nullptr, 0},
        {SET_RAY_TRACING, descriptorRanges2, helper::GetCountOf(descriptorRanges2)},
        {SET_MORPH, descriptorRanges3, helper::GetCountOf(descriptorRanges3), &dynamicConstantBuffer, 1},
        {SET_SHARC, descriptorRanges4, helper::GetCountOf(descriptorRanges4)},
    };

    { // Pipeline layout
        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};

        uint32_t setNum = 1;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.dynamicConstantBufferMaxNum += descriptorSetDescs[SET_GLOBAL].dynamicConstantBufferNum * setNum;
        descriptorPoolDesc.samplerMaxNum += descriptorSetDescs[SET_GLOBAL].ranges[0].descriptorNum * GetQueuedFrameNum() * setNum;

        setNum = (uint32_t)DescriptorSet::MAX_NUM - 6; // exclude non-SET_OTHER sets
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.textureMaxNum += descriptorSetDescs[SET_OTHER].ranges[0].descriptorNum * setNum;
        descriptorPoolDesc.storageTextureMaxNum += descriptorSetDescs[SET_OTHER].ranges[1].descriptorNum * setNum;

        setNum = 1;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.accelerationStructureMaxNum += descriptorSetDescs[SET_RAY_TRACING].ranges[0].descriptorNum * setNum;
        descriptorPoolDesc.structuredBufferMaxNum += descriptorSetDescs[SET_RAY_TRACING].ranges[1].descriptorNum * setNum;
        descriptorPoolDesc.textureMaxNum += descriptorSetDescs[SET_RAY_TRACING].ranges[2].descriptorNum * setNum;

        setNum = 2;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.dynamicConstantBufferMaxNum += descriptorSetDescs[SET_MORPH].dynamicConstantBufferNum * setNum;
        descriptorPoolDesc.structuredBufferMaxNum += descriptorSetDescs[SET_MORPH].ranges[0].descriptorNum * setNum;
        descriptorPoolDesc.storageStructuredBufferMaxNum += descriptorSetDescs[SET_MORPH].ranges[1].descriptorNum * setNum;

        setNum = 2;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.storageStructuredBufferMaxNum += descriptorSetDescs[SET_SHARC].ranges[0].descriptorNum * setNum;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    }
}

void Sample::CreatePipelines() {
    if (!m_Pipelines.empty()) {
        NRI.WaitForIdle(*m_GraphicsQueue);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(*m_Pipelines[i]);
        m_Pipelines.clear();

        m_NRD.CreatePipelines();
    }

    utils::ShaderCodeStorage shaderCodeStorage;

    nri::ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = m_PipelineLayout;

    nri::Pipeline* pipeline = nullptr;
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    { // Pipeline::MorphMeshUpdateVertices
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "MorphMeshUpdateVertices.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::MorphMeshUpdatePrimitives
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "MorphMeshUpdatePrimitives.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::SharcUpdate
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "SharcUpdate.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::SharcResolve
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "SharcResolve.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::SharcHashCopy
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "SharcHashCopy.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::TraceOpaque
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "TraceOpaque.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Composition
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Composition.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::TraceTransparent
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "TraceTransparent.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Taa
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "TAA.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Final
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Final.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DlssBefore
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "DlssBefore.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DlssAfter
        pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "DlssAfter.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }
}

void Sample::CreateAccelerationStructures() {
    // Temp resources created as "dedicated", since they are destroyed immediately after use
    double stamp1 = m_Timer.GetTimeStamp();

    // Prepare
    std::vector<uint32_t> uniqueDynamicMeshInstances;
    std::array<std::vector<uint32_t>, 4> instanceIndices; // opaque, transparent, emissive, other
    uint64_t uploadSize = 0;
    uint64_t geometryOffset = 0;
    uint32_t geometryNum = 0;

    for (uint32_t i = m_ProxyInstancesNum; i < m_Scene.instances.size(); i++) {
        utils::Instance& instance = m_Scene.instances[i];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        if (material.IsOff())
            continue;

        uint32_t appearanceNum = 1;
        if (instance.allowUpdate) {
            if (std::find(uniqueDynamicMeshInstances.begin(), uniqueDynamicMeshInstances.end(), instance.meshInstanceIndex) != uniqueDynamicMeshInstances.end())
                continue;

            uniqueDynamicMeshInstances.push_back(instance.meshInstanceIndex);
            instanceIndices[3].push_back(i);
        } else {
            if (!material.IsTransparent()) {
                instanceIndices[0].push_back(i);
                m_OpaqueObjectsNum++;
            } else {
                instanceIndices[1].push_back(i);
                m_TransparentObjectsNum++;
            }

            if (material.IsEmissive()) {
                instanceIndices[2].push_back(i);
                m_EmissiveObjectsNum++;
                appearanceNum++;
            }
        }

        if (!appearanceNum)
            continue;

        const utils::MeshInstance& meshInstance = m_Scene.meshInstances[instance.meshInstanceIndex];
        const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];

        uint16_t vertexStride = mesh.HasMorphTargets() ? sizeof(float16_t4) : sizeof(float[3]);
        uint64_t vertexDataSize = mesh.vertexNum * vertexStride;
        uint64_t indexDataSize = helper::Align(mesh.indexNum * sizeof(utils::Index), 4);
        uint64_t transformDataSize = instance.allowUpdate ? 0 : sizeof(nri::TransformMatrix);

        vertexDataSize *= appearanceNum;
        indexDataSize *= appearanceNum;
        transformDataSize *= appearanceNum;

        uploadSize += vertexDataSize + indexDataSize + transformDataSize;
        geometryOffset += transformDataSize;

        geometryNum += appearanceNum;
    }

    { // AccelerationStructure::TLAS_World
        nri::AllocateAccelerationStructureDesc allocateAccelerationStructureDesc = {};
        allocateAccelerationStructureDesc.desc.type = nri::AccelerationStructureType::TOP_LEVEL;
        allocateAccelerationStructureDesc.desc.flags = TLAS_BUILD_BITS;
        allocateAccelerationStructureDesc.desc.geometryOrInstanceNum = helper::GetCountOf(m_Scene.instances);
        allocateAccelerationStructureDesc.memoryLocation = nri::MemoryLocation::DEVICE;

        nri::AccelerationStructure* accelerationStructure = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAccelerationStructure(*m_Device, allocateAccelerationStructureDesc, accelerationStructure));
        m_AccelerationStructures.push_back(accelerationStructure);

        // Descriptor::World_AccelerationStructure
        nri::Descriptor* descriptor = nullptr;
        NRI.CreateAccelerationStructureDescriptor(*accelerationStructure, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    { // AccelerationStructure::TLAS_Emissive
        nri::AllocateAccelerationStructureDesc allocateAccelerationStructureDesc = {};
        allocateAccelerationStructureDesc.desc.type = nri::AccelerationStructureType::TOP_LEVEL;
        allocateAccelerationStructureDesc.desc.flags = TLAS_BUILD_BITS;
        allocateAccelerationStructureDesc.desc.geometryOrInstanceNum = helper::GetCountOf(m_Scene.instances);
        allocateAccelerationStructureDesc.memoryLocation = nri::MemoryLocation::DEVICE;

        nri::AccelerationStructure* accelerationStructure = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAccelerationStructure(*m_Device, allocateAccelerationStructureDesc, accelerationStructure));
        m_AccelerationStructures.push_back(accelerationStructure);

        // Descriptor::Light_AccelerationStructure
        nri::Descriptor* descriptor = nullptr;
        NRI.CreateAccelerationStructureDescriptor(*accelerationStructure, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    // Create temp buffer for indices, vertices and transforms in UPLOAD heap
    nri::Buffer* uploadBuffer = nullptr;
    {
        nri::AllocateBufferDesc allocateBufferDesc = {};
        allocateBufferDesc.desc = {uploadSize, 0, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT};
        allocateBufferDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        allocateBufferDesc.dedicated = true;

        NRI_ABORT_ON_FAILURE(NRI.AllocateBuffer(*m_Device, allocateBufferDesc, uploadBuffer));
    }

    // Create BOTTOM_LEVEL acceleration structures
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    uint64_t scratchSize = 0;

    uint8_t* uploadData = (uint8_t*)NRI.MapBuffer(*uploadBuffer, 0, nri::WHOLE_SIZE);
    assert(uploadData);

    uint64_t primitivesNum = 0;
    std::vector<nri::BuildBottomLevelAccelerationStructureDesc> buildBottomLevelAccelerationStructureDescs;
    std::vector<bool> isDeformable;

    std::vector<nri::BottomLevelGeometryDesc> geometries;
    geometries.reserve(geometryNum); // reallocation is NOT allowed!

    for (uint32_t mode = 0; mode < instanceIndices.size(); mode++) {
        size_t geometryObjectBase = geometries.size();

        for (uint32_t i : instanceIndices[mode]) {
            const utils::Instance& instance = m_Scene.instances[i];
            const utils::Material& material = m_Scene.materials[instance.materialIndex];
            utils::MeshInstance& meshInstance = m_Scene.meshInstances[instance.meshInstanceIndex];
            const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];

            if (mode == 3)
                meshInstance.blasIndex = (uint32_t)m_AccelerationStructures.size();

            // Copy geometry to temp buffer
            uint16_t vertexStride = mesh.HasMorphTargets() ? sizeof(float16_t4) : sizeof(float[3]);
            uint64_t vertexDataSize = mesh.vertexNum * vertexStride;
            uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);

            uint8_t* p = uploadData + geometryOffset;
            for (uint32_t v = 0; v < mesh.vertexNum; v++) {
                if (mesh.HasMorphTargets())
                    memcpy(p, &m_Scene.morphVertices[mesh.morphTargetVertexOffset + v].pos, vertexStride);
                else
                    memcpy(p, m_Scene.vertices[mesh.vertexOffset + v].pos, vertexStride);
                p += vertexStride;
            }

            memcpy(p, &m_Scene.indices[mesh.indexOffset], indexDataSize);

            // Copy transform to temp buffer
            uint64_t transformOffset = 0;
            if (mode != 3) {
                float4x4 mObjectToWorld = instance.rotation;

                if (any(instance.scale != 1.0f)) {
                    float4x4 translation;
                    translation.SetupByTranslation(float3(instance.position) - mesh.aabb.GetCenter());

                    float4x4 translationInv = translation;
                    translationInv.InvertOrtho();

                    float4x4 scale;
                    scale.SetupByScale(instance.scale);

                    mObjectToWorld = mObjectToWorld * translationInv * scale * translation;
                }

                mObjectToWorld.AddTranslation(float3(instance.position));
                mObjectToWorld.Transpose3x4();

                transformOffset = geometries.size() * sizeof(nri::TransformMatrix);
                memcpy(uploadData + transformOffset, mObjectToWorld.a, sizeof(nri::TransformMatrix));
            }

            // Add geometry object
            nri::BottomLevelGeometryDesc& bottomLevelGeometry = geometries.emplace_back();
            bottomLevelGeometry = {};
            bottomLevelGeometry.type = nri::BottomLevelGeometryType::TRIANGLES;
            bottomLevelGeometry.flags = material.IsAlphaOpaque() ? nri::BottomLevelGeometryBits::NONE : nri::BottomLevelGeometryBits::OPAQUE_GEOMETRY;
            bottomLevelGeometry.triangles.vertexBuffer = uploadBuffer;
            bottomLevelGeometry.triangles.vertexOffset = geometryOffset;
            bottomLevelGeometry.triangles.vertexNum = mesh.vertexNum;
            bottomLevelGeometry.triangles.vertexStride = vertexStride;
            bottomLevelGeometry.triangles.vertexFormat = mesh.HasMorphTargets() ? nri::Format::RGBA16_SFLOAT : nri::Format::RGB32_SFLOAT;
            bottomLevelGeometry.triangles.indexBuffer = uploadBuffer;
            bottomLevelGeometry.triangles.indexOffset = geometryOffset + vertexDataSize;
            bottomLevelGeometry.triangles.indexNum = mesh.indexNum;
            bottomLevelGeometry.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

            if (mode != 3) {
                bottomLevelGeometry.triangles.transformBuffer = uploadBuffer;
                bottomLevelGeometry.triangles.transformOffset = transformOffset;
            } else {
                // Create BLAS
                nri::AllocateAccelerationStructureDesc allocateAccelerationStructureDesc = {};
                allocateAccelerationStructureDesc.desc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                allocateAccelerationStructureDesc.desc.flags = mesh.HasMorphTargets() ? BLAS_DEFORMABLE_MESH_BUILD_BITS : BLAS_RIGID_MESH_BUILD_BITS;
                allocateAccelerationStructureDesc.desc.geometryOrInstanceNum = 1;
                allocateAccelerationStructureDesc.desc.geometries = &bottomLevelGeometry;
                allocateAccelerationStructureDesc.memoryLocation = nri::MemoryLocation::DEVICE;
                allocateAccelerationStructureDesc.dedicated = true;

                nri::AccelerationStructure* accelerationStructure = nullptr;
                NRI_ABORT_ON_FAILURE(NRI.AllocateAccelerationStructure(*m_Device, allocateAccelerationStructureDesc, accelerationStructure));
                m_AccelerationStructures.push_back(accelerationStructure);

                // Save build parameters
                nri::BuildBottomLevelAccelerationStructureDesc& buildBottomLevelAccelerationStructureDesc = buildBottomLevelAccelerationStructureDescs.emplace_back();
                buildBottomLevelAccelerationStructureDesc = {};
                buildBottomLevelAccelerationStructureDesc.dst = accelerationStructure;
                buildBottomLevelAccelerationStructureDesc.geometryNum = 1;
                buildBottomLevelAccelerationStructureDesc.geometries = &geometries[geometries.size() - 1];
                buildBottomLevelAccelerationStructureDesc.scratchBuffer = nullptr;
                buildBottomLevelAccelerationStructureDesc.scratchOffset = scratchSize;
                isDeformable.push_back(mesh.HasMorphTargets());

                // Update scratch
                uint64_t buildSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
                scratchSize += helper::Align(buildSize, deviceDesc.memoryAlignment.scratchBufferOffset);

                if (mesh.HasMorphTargets()) {
                    uint64_t updateSize = NRI.GetAccelerationStructureUpdateScratchBufferSize(*accelerationStructure);
                    m_MorphMeshScratchSize += helper::Align(max(buildSize, updateSize), deviceDesc.memoryAlignment.scratchBufferOffset);
                }
            }

            // Update geometry offset
            geometryOffset += vertexDataSize + helper::Align(indexDataSize, 4);
            primitivesNum += mesh.indexNum / 3;
        }

        if (mode != 3) {
            uint32_t geometryObjectsNum = (uint32_t)(geometries.size() - geometryObjectBase);
            if (geometryObjectsNum) {
                // Create BLAS
                nri::AllocateAccelerationStructureDesc allocateAccelerationStructureDesc = {};
                allocateAccelerationStructureDesc.desc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                allocateAccelerationStructureDesc.desc.flags = BLAS_RIGID_MESH_BUILD_BITS;
                allocateAccelerationStructureDesc.desc.geometryOrInstanceNum = geometryObjectsNum;
                allocateAccelerationStructureDesc.desc.geometries = &geometries[geometryObjectBase];
                allocateAccelerationStructureDesc.memoryLocation = nri::MemoryLocation::DEVICE;
                allocateAccelerationStructureDesc.dedicated = true;

                nri::AccelerationStructure* accelerationStructure = nullptr;
                NRI_ABORT_ON_FAILURE(NRI.AllocateAccelerationStructure(*m_Device, allocateAccelerationStructureDesc, accelerationStructure));
                m_AccelerationStructures.push_back(accelerationStructure);

                // Save build parameters
                nri::BuildBottomLevelAccelerationStructureDesc& buildBottomLevelAccelerationStructureDesc = buildBottomLevelAccelerationStructureDescs.emplace_back();
                buildBottomLevelAccelerationStructureDesc = {};
                buildBottomLevelAccelerationStructureDesc.dst = accelerationStructure;
                buildBottomLevelAccelerationStructureDesc.geometryNum = geometryObjectsNum;
                buildBottomLevelAccelerationStructureDesc.geometries = &geometries[geometryObjectBase];
                buildBottomLevelAccelerationStructureDesc.scratchBuffer = nullptr;
                buildBottomLevelAccelerationStructureDesc.scratchOffset = scratchSize;
                isDeformable.push_back(false);

                // Update scratch
                uint64_t size = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
                scratchSize += helper::Align(size, deviceDesc.memoryAlignment.scratchBufferOffset);
            } else {
                // Needed only to preserve order
                m_AccelerationStructures.push_back(nullptr);
            }
        }
    }

    // Create temp resources
    uint32_t blasNum = (uint32_t)buildBottomLevelAccelerationStructureDescs.size();

    nri::Buffer* scratchBuffer = nullptr;
    {
        nri::AllocateBufferDesc allocateBufferDesc = {};
        allocateBufferDesc.desc = {scratchSize, 0, nri::BufferUsageBits::SCRATCH_BUFFER};
        allocateBufferDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        allocateBufferDesc.dedicated = true;

        NRI_ABORT_ON_FAILURE(NRI.AllocateBuffer(*m_Device, allocateBufferDesc, scratchBuffer));
    }

    nri::Buffer* readbackBuffer = nullptr;
    {
        nri::AllocateBufferDesc allocateBufferDesc = {};
        allocateBufferDesc.desc = {blasNum * sizeof(uint64_t), 0, nri::BufferUsageBits::NONE};
        allocateBufferDesc.memoryLocation = nri::MemoryLocation::HOST_READBACK;
        allocateBufferDesc.dedicated = true;

        NRI_ABORT_ON_FAILURE(NRI.AllocateBuffer(*m_Device, allocateBufferDesc, readbackBuffer));
    }

    nri::QueryPool* queryPool = nullptr;
    {
        nri::QueryPoolDesc queryPoolDesc = {};
        queryPoolDesc.queryType = nri::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE;
        queryPoolDesc.capacity = blasNum;

        NRI_ABORT_ON_FAILURE(NRI.CreateQueryPool(*m_Device, queryPoolDesc, queryPool));
    }

    nri::CommandAllocator* commandAllocator = nullptr;
    NRI.CreateCommandAllocator(*m_GraphicsQueue, commandAllocator);

    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    double stamp2 = m_Timer.GetTimeStamp();

    { // Build BLASes
        // Record building commands
        NRI.BeginCommandBuffer(*commandBuffer, nullptr);
        {
            std::vector<nri::BufferBarrierDesc> bufferBarriers;
            std::vector<nri::AccelerationStructure*> blases;

            // Barriers (write) and patch scratch buffer
            for (size_t i = 0; i < blasNum; i++) {
                auto& desc = buildBottomLevelAccelerationStructureDescs[i];
                desc.scratchBuffer = scratchBuffer;

                nri::BufferBarrierDesc bufferBarrier = {};
                bufferBarrier.buffer = NRI.GetAccelerationStructureBuffer(*desc.dst);
                bufferBarrier.after = {nri::AccessBits::ACCELERATION_STRUCTURE_WRITE, nri::StageBits::ACCELERATION_STRUCTURE};

                bufferBarriers.push_back(bufferBarrier);
                blases.push_back(desc.dst);
            }

            nri::BarrierGroupDesc barrierGroupDesc = {};
            barrierGroupDesc.bufferNum = (uint32_t)bufferBarriers.size();
            barrierGroupDesc.buffers = bufferBarriers.data();

            NRI.CmdBarrier(*commandBuffer, barrierGroupDesc);

            // Build everything in one go
            NRI.CmdBuildBottomLevelAccelerationStructures(*commandBuffer, buildBottomLevelAccelerationStructureDescs.data(), (uint32_t)buildBottomLevelAccelerationStructureDescs.size());

            // Barriers (read)
            for (nri::BufferBarrierDesc& bufferBarrier : bufferBarriers) {
                bufferBarrier.before = bufferBarrier.after;
                bufferBarrier.after = {nri::AccessBits::ACCELERATION_STRUCTURE_READ, nri::StageBits::ACCELERATION_STRUCTURE};
            }

            NRI.CmdBarrier(*commandBuffer, barrierGroupDesc);

            // Emit sizes for compaction
            NRI.CmdResetQueries(*commandBuffer, *queryPool, 0, blasNum);
            NRI.CmdWriteAccelerationStructuresSizes(*commandBuffer, blases.data(), blasNum, *queryPool, 0);
            NRI.CmdCopyQueries(*commandBuffer, *queryPool, 0, blasNum, *readbackBuffer, 0);
        }
        NRI.EndCommandBuffer(*commandBuffer);

        // Submit
        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);

        // Wait idle
        NRI.WaitForIdle(*m_GraphicsQueue);
    }

    // Compact BLASes
    std::vector<nri::AccelerationStructure*> compactedBlases;
    {
        uint64_t* sizes = (uint64_t*)NRI.MapBuffer(*readbackBuffer, 0, nri::WHOLE_SIZE);

        // Record compaction commands
        NRI.BeginCommandBuffer(*commandBuffer, nullptr);
        {
            for (uint32_t i = 0; i < blasNum; i++) {
                const nri::BuildBottomLevelAccelerationStructureDesc& blasBuildDesc = buildBottomLevelAccelerationStructureDescs[i];

                nri::AllocateAccelerationStructureDesc allocateAccelerationStructureDesc = {};
                allocateAccelerationStructureDesc.desc.optimizedSize = sizes[i];
                allocateAccelerationStructureDesc.desc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                allocateAccelerationStructureDesc.desc.flags = isDeformable[i] ? BLAS_DEFORMABLE_MESH_BUILD_BITS : BLAS_RIGID_MESH_BUILD_BITS;
                allocateAccelerationStructureDesc.desc.geometryOrInstanceNum = blasBuildDesc.geometryNum;
                allocateAccelerationStructureDesc.desc.geometries = blasBuildDesc.geometries;
                allocateAccelerationStructureDesc.memoryLocation = nri::MemoryLocation::DEVICE;

                nri::AccelerationStructure* compactedBlas = nullptr;
                NRI_ABORT_ON_FAILURE(NRI.AllocateAccelerationStructure(*m_Device, allocateAccelerationStructureDesc, compactedBlas));
                compactedBlases.push_back(compactedBlas);

                nri::AccelerationStructure* tempBlas = blasBuildDesc.dst;
                NRI.CmdCopyAccelerationStructure(*commandBuffer, *compactedBlas, *tempBlas, nri::CopyMode::COMPACT);
            }
        }
        NRI.EndCommandBuffer(*commandBuffer);

        // Submit
        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);

        // Wait idle
        NRI.WaitForIdle(*m_GraphicsQueue);
    }

    double buildTime = m_Timer.GetTimeStamp() - stamp2;

    // Cleanup
    for (uint32_t i = 0; i < blasNum; i++) {
        const nri::BuildBottomLevelAccelerationStructureDesc& blasBuildDesc = buildBottomLevelAccelerationStructureDescs[i];

        nri::AccelerationStructure* tempBlas = blasBuildDesc.dst;
        NRI.DestroyAccelerationStructure(*tempBlas);

        nri::AccelerationStructure* compactedBlas = compactedBlases[i];
        std::replace(m_AccelerationStructures.begin(), m_AccelerationStructures.end(), tempBlas, compactedBlas);
    }

    NRI.UnmapBuffer(*uploadBuffer);
    NRI.UnmapBuffer(*readbackBuffer);

    NRI.DestroyQueryPool(*queryPool);

    NRI.DestroyBuffer(*readbackBuffer);
    NRI.DestroyBuffer(*scratchBuffer);
    NRI.DestroyBuffer(*uploadBuffer);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    double totalTime = m_Timer.GetTimeStamp() - stamp1;

    printf(
        "Scene stats:\n"
        "  Instances     : %zu\n"
        "  Meshes        : %zu\n"
        "  Vertices      : %zu\n"
        "  Primitives    : %zu\n"
        "BVH stats:\n"
        "  Total time    : %.2f ms\n"
        "  Building time : %.2f ms\n"
        "  Scratch size  : %.2f Mb\n"
        "  BLAS num      : %u\n"
        "  Geometries    : %zu\n"
        "  Primitives    : %zu\n",
        m_Scene.instances.size(), m_Scene.meshes.size(), m_Scene.vertices.size(), m_Scene.primitives.size(),
        totalTime, buildTime, scratchSize / (1024.0 * 1024.0),
        blasNum, geometries.size(), primitivesNum);
}

void Sample::CreateSamplers() {
    nri::Descriptor* descriptor = nullptr;

    { // Descriptor::LinearMipmapLinear_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::LinearMipmapNearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::NEAREST};
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::NearestMipmapNearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.filters = {nri::Filter::NEAREST, nri::Filter::NEAREST, nri::Filter::NEAREST};
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }
}

inline nri::Format ConvertFormatToTextureStorageCompatible(nri::Format format) {
    switch (format) {
        case nri::Format::D16_UNORM:
            return nri::Format::R16_UNORM;
        case nri::Format::D24_UNORM_S8_UINT:
            return nri::Format::R24_UNORM_X8;
        case nri::Format::D32_SFLOAT:
            return nri::Format::R32_SFLOAT;
        case nri::Format::D32_SFLOAT_S8_UINT_X24:
            return nri::Format::R32_SFLOAT_X8_X24;
        case nri::Format::RGBA8_SRGB:
            return nri::Format::RGBA8_UNORM;
        case nri::Format::BGRA8_SRGB:
            return nri::Format::BGRA8_UNORM;
        default:
            return format;
    }
}

void Sample::CreateResources(nri::Format swapChainFormat) {
    // TODO: DLSS doesn't support R16 UNORM/SNORM
#if (NRD_MODE == OCCLUSION)
    const nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::R16_SFLOAT : nri::Format::R16_UNORM;
#elif (NRD_MODE == DIRECTIONAL_OCCLUSION)
    const nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::RGBA16_SFLOAT : nri::Format::RGBA16_SNORM;
#else
    const nri::Format dataFormat = nri::Format::RGBA16_SFLOAT;
#endif

#if (NRD_NORMAL_ENCODING == 0)
    const nri::Format normalFormat = nri::Format::RGBA8_UNORM;
#elif (NRD_NORMAL_ENCODING == 1)
    const nri::Format normalFormat = nri::Format::RGBA8_SNORM;
#elif (NRD_NORMAL_ENCODING == 2)
    const nri::Format normalFormat = nri::Format::R10_G10_B10_A2_UNORM;
#elif (NRD_NORMAL_ENCODING == 3)
    const nri::Format normalFormat = nri::Format::RGBA16_UNORM;
#elif (NRD_NORMAL_ENCODING == 4)
    const nri::Format normalFormat = nri::Format::RGBA16_SFLOAT; // TODO: RGBA16_SNORM can't be used, because NGX doesn't support it
#endif

    const nri::Format taaFormat = nri::Format::RGBA16_SFLOAT; // required for new TAA even in LDR mode (RGBA16_UNORM can't be used)
    const nri::Format colorFormat = USE_LOW_PRECISION_FP_FORMATS ? nri::Format::R11_G11_B10_UFLOAT : nri::Format::RGBA16_SFLOAT;
    const nri::Format criticalColorFormat = nri::Format::RGBA16_SFLOAT; // TODO: R9_G9_B9_E5_UFLOAT?
    const nri::Format shadowFormat = SIGMA_TRANSLUCENT ? nri::Format::RGBA8_UNORM : nri::Format::R8_UNORM;

    const uint16_t w = (uint16_t)m_RenderResolution.x;
    const uint16_t h = (uint16_t)m_RenderResolution.y;
    const uint64_t instanceNum = m_Scene.instances.size() + MAX_ANIMATED_INSTANCE_NUM;
    const uint64_t instanceDataSize = instanceNum * sizeof(InstanceData);
    const uint64_t worldScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_World));
    const uint64_t lightScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_Emissive));

    std::vector<DescriptorDesc> descriptorDescs;

    m_InstanceData.resize(instanceNum);
    m_WorldTlasData.resize(instanceNum);
    m_LightTlasData.resize(instanceNum);

    // Buffers (DEVICE, read-only)
    CreateBuffer(descriptorDescs, "Buffer::InstanceData", nri::Format::UNKNOWN, instanceDataSize / sizeof(InstanceData), sizeof(InstanceData),
        nri::BufferUsageBits::SHADER_RESOURCE);
    CreateBuffer(descriptorDescs, "Buffer::MorphMeshIndices", nri::Format::UNKNOWN, m_Scene.morphMeshTotalIndicesNum, sizeof(utils::Index),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT);
    CreateBuffer(descriptorDescs, "Buffer::MorphMeshVertices", nri::Format::UNKNOWN, m_Scene.morphVertices.size(), sizeof(utils::MorphVertex),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT);

    // Buffers (DEVICE)
    CreateBuffer(descriptorDescs, "Buffer::MorphedPositions", nri::Format::UNKNOWN, m_Scene.morphedVerticesNum * MAX_ANIMATION_HISTORY_FRAME_NUM, sizeof(float16_t4),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT);
    CreateBuffer(descriptorDescs, "Buffer::MorphedAttributes", nri::Format::UNKNOWN, m_Scene.morphedVerticesNum, sizeof(MorphedAttributes),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::MorphedPrimitivePrevPositions", nri::Format::UNKNOWN, m_Scene.morphedPrimitivesNum, sizeof(MorphedPrimitivePrevPositions),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::PrimitiveData", nri::Format::UNKNOWN, m_Scene.totalInstancedPrimitivesNum, sizeof(PrimitiveData),
        nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::SharcHashEntries", nri::Format::UNKNOWN, SHARC_CAPACITY, sizeof(uint64_t),
        nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::SharcHashCopyOffset", nri::Format::UNKNOWN, SHARC_CAPACITY, sizeof(uint32_t),
        nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::SharcVoxelDataPing", nri::Format::UNKNOWN, SHARC_CAPACITY, sizeof(uint32_t) * 4,
        nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::SharcVoxelDataPong", nri::Format::UNKNOWN, SHARC_CAPACITY, sizeof(uint32_t) * 4,
        nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::WorldScratch", nri::Format::UNKNOWN, worldScratchBufferSize, 1,
        nri::BufferUsageBits::SCRATCH_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::LightScratch", nri::Format::UNKNOWN, lightScratchBufferSize, 1,
        nri::BufferUsageBits::SCRATCH_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::MorphMeshScratch", nri::Format::UNKNOWN, m_MorphMeshScratchSize, 1,
        nri::BufferUsageBits::SCRATCH_BUFFER);

    // Textures (DEVICE)
    CreateTexture(descriptorDescs, "Texture::ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Mv", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Normal_Roughness", normalFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::PsrThroughput", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::BaseColor_Metalness", nri::Format::RGBA8_SRGB, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectLighting", colorFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectEmission", colorFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Shadow", shadowFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Penumbra", nri::Format::R16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Translucency", shadowFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Validation", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Composed", criticalColorFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ComposedDiff", colorFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ComposedSpec_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::TaaHistory", taaFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TaaHistoryPrev", taaFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);

    // Created unconditionally, unfortunately...
    CreateTexture(descriptorDescs, "Texture::RRGuide_DiffAlbedo", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::RRGuide_SpecAlbedo", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::RRGuide_SpecHitDistance", nri::Format::R16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::RRGuide_Normal_Roughness", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::DlssOutput", criticalColorFormat, (uint16_t)GetOutputResolution().x, (uint16_t)GetOutputResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);

    CreateTexture(descriptorDescs, "Texture::PreFinal", criticalColorFormat, (uint16_t)GetOutputResolution().x, (uint16_t)GetOutputResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::Final", swapChainFormat, (uint16_t)GetWindowResolution().x, (uint16_t)GetWindowResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::COPY_SOURCE);

#if (NRD_MODE == SH)
    CreateTexture(descriptorDescs, "Texture::Unfiltered_DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
#endif

    for (const utils::Texture* texture : m_Scene.textures)
        CreateTexture(descriptorDescs, "", texture->GetFormat(), texture->GetWidth(), texture->GetHeight(), texture->GetMipNum(), texture->GetArraySize(), nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    // Create descriptors
    nri::Descriptor* descriptor = nullptr;
    {
        const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

        nri::BufferViewDesc constantBufferViewDesc = {};
        constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
        constantBufferViewDesc.buffer = NRI.GetStreamerConstantBuffer(*m_Streamer);

        constantBufferViewDesc.size = helper::Align(sizeof(GlobalConstants), deviceDesc.memoryAlignment.constantBufferOffset);
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(constantBufferViewDesc, descriptor));
        m_Descriptors.push_back(descriptor);

        constantBufferViewDesc.size = helper::Align(sizeof(MorphMeshUpdateVerticesConstants), deviceDesc.memoryAlignment.constantBufferOffset);
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(constantBufferViewDesc, descriptor));
        m_Descriptors.push_back(descriptor);

        constantBufferViewDesc.size = helper::Align(sizeof(MorphMeshUpdatePrimitivesConstants), deviceDesc.memoryAlignment.constantBufferOffset);
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(constantBufferViewDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    for (const DescriptorDesc& desc : descriptorDescs) {
        if (desc.textureUsage == nri::TextureUsageBits::NONE) {
            if (desc.bufferUsage == nri::BufferUsageBits::CONSTANT_BUFFER) {
                // Constant buffer views are not stored in m_Descriptors
            } else {
                NRI.SetDebugName((nri::Object*)desc.resource, desc.debugName);

                if (desc.bufferUsage & nri::BufferUsageBits::SHADER_RESOURCE) {
                    const nri::BufferViewDesc viewDesc = {(nri::Buffer*)desc.resource, nri::BufferViewType::SHADER_RESOURCE, desc.format};
                    NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, descriptor));
                    m_Descriptors.push_back(descriptor);
                }
                if (desc.bufferUsage & nri::BufferUsageBits::SHADER_RESOURCE_STORAGE) {
                    const nri::BufferViewDesc viewDesc = {(nri::Buffer*)desc.resource, nri::BufferViewType::SHADER_RESOURCE_STORAGE, desc.format};
                    NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, descriptor));
                    m_Descriptors.push_back(descriptor);
                }
            }
        } else {
            NRI.SetDebugName((nri::Object*)desc.resource, desc.debugName);

            nri::Texture2DViewDesc viewDesc = {(nri::Texture*)desc.resource, desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_2D, desc.format};
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
            m_Descriptors.push_back(descriptor);

            if (desc.textureUsage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE) {
                viewDesc.format = ConvertFormatToTextureStorageCompatible(desc.format);
                viewDesc.viewType = desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D;
                NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
                m_Descriptors.push_back(descriptor);
            }
        }
    }
}

void Sample::CreateDescriptorSets() {
    nri::DescriptorSet* descriptorSet = nullptr;

    { // DescriptorSet::Global0
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_GLOBAL, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* samplers[] = {
            Get(Descriptor::LinearMipmapLinear_Sampler),
            Get(Descriptor::LinearMipmapNearest_Sampler),
            Get(Descriptor::NearestMipmapNearest_Sampler),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {samplers, helper::GetCountOf(samplers)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);

        nri::Descriptor* constantBuffer = Get(Descriptor::Global_ConstantBuffer);
        NRI.UpdateDynamicConstantBuffers(*descriptorSet, 0, 1, &constantBuffer);
    }

    { // DescriptorSet::TraceOpaque1
        const nri::Descriptor* resources[] = {
            Get(Descriptor::ComposedDiff_Texture),
            Get(Descriptor::ComposedSpec_ViewZ_Texture),
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::ScramblingRanking)),
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::SobolSequence)),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::DirectEmission_StorageTexture),
            Get(Descriptor::PsrThroughput_StorageTexture),
            Get(Descriptor::Unfiltered_Penumbra_StorageTexture),
            Get(Descriptor::Unfiltered_Translucency_StorageTexture),
            Get(Descriptor::Unfiltered_Diff_StorageTexture),
            Get(Descriptor::Unfiltered_Spec_StorageTexture),
#if (NRD_MODE == SH)
            Get(Descriptor::Unfiltered_DiffSh_StorageTexture),
            Get(Descriptor::Unfiltered_SpecSh_StorageTexture),
#endif
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Composition1
        const nri::Descriptor* resources[] = {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::DirectLighting_Texture),
            Get(Descriptor::DirectEmission_Texture),
            Get(Descriptor::PsrThroughput_Texture),
            Get(Descriptor::Shadow_Texture),
            Get(Descriptor::Diff_Texture),
            Get(Descriptor::Spec_Texture),
#if (NRD_MODE == SH)
            Get(Descriptor::DiffSh_Texture),
            Get(Descriptor::SpecSh_Texture),
#endif
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::ComposedDiff_StorageTexture),
            Get(Descriptor::ComposedSpec_ViewZ_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::TraceTransparent1
        const nri::Descriptor* resources[] = {
            Get(Descriptor::ComposedDiff_Texture),
            Get(Descriptor::ComposedSpec_ViewZ_Texture),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::Composed_StorageTexture),
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Taa1a
        const nri::Descriptor* resources[] = {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_Texture),
            Get(Descriptor::TaaHistoryPrev_Texture),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::TaaHistory_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Taa1b
        const nri::Descriptor* resources[] = {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_Texture),
            Get(Descriptor::TaaHistory_Texture),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::TaaHistoryPrev_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Final1
        const nri::Descriptor* resources[] = {
            Get(Descriptor::PreFinal_Texture),
            Get(Descriptor::Composed_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DlssBefore1
        const nri::Descriptor* resources[] = {
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::Unfiltered_Spec_Texture),
        };

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::RRGuide_DiffAlbedo_StorageTexture),
            Get(Descriptor::RRGuide_SpecAlbedo_StorageTexture),
            Get(Descriptor::RRGuide_SpecHitDistance_StorageTexture),
            Get(Descriptor::RRGuide_Normal_Roughness_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DlssAfter1
        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::DlssOutput_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 1, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::RayTracing2
        const nri::Descriptor* accelerationStructures[] = {
            Get(Descriptor::World_AccelerationStructure),
            Get(Descriptor::Light_AccelerationStructure)};

        const nri::Descriptor* structuredBuffers[] = {
            Get(Descriptor::InstanceData_Buffer),
            Get(Descriptor::PrimitiveData_Buffer),
            Get(Descriptor::MorphedPrimitivePrevData_Buffer),
        };

        std::vector<nri::Descriptor*> textures(m_Scene.materials.size() * TEXTURES_PER_MATERIAL);
        for (size_t i = 0; i < m_Scene.materials.size(); i++) {
            const size_t index = i * TEXTURES_PER_MATERIAL;
            const utils::Material& material = m_Scene.materials[i];

            textures[index] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.baseColorTexIndex));
            textures[index + 1] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.roughnessMetalnessTexIndex));
            textures[index + 2] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.normalTexIndex));
            textures[index + 3] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.emissiveTexIndex));
        }

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_RAY_TRACING, &descriptorSet, 1, helper::GetCountOf(textures)));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {accelerationStructures, helper::GetCountOf(accelerationStructures)},
            {structuredBuffers, helper::GetCountOf(structuredBuffers)},
            {textures.data(), helper::GetCountOf(textures)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::MorphTargetPose3
        const nri::Descriptor* resources[] = {
            Get(Descriptor::MorphMeshVertices_Buffer)};

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::MorphedPositions_StorageBuffer),
            Get(Descriptor::MorphedAttributes_StorageBuffer),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_MORPH, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)}};

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);

        nri::Descriptor* constantBuffer = Get(Descriptor::MorphTargetPose_ConstantBuffer);
        NRI.UpdateDynamicConstantBuffers(*descriptorSet, 0, 1, &constantBuffer);
    }

    { // DescriptorSet::MorphTargetUpdatePrimitives3
        const nri::Descriptor* resources[] = {
            Get(Descriptor::MorphMeshIndices_Buffer),
            Get(Descriptor::MorphedPositions_Buffer),
            Get(Descriptor::MorphedAttributes_Buffer)};

        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::PrimitiveData_StorageBuffer),
            Get(Descriptor::MorphedPrimitivePrevData_StorageBuffer)};

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_MORPH, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {resources, helper::GetCountOf(resources)},
            {storageResources, helper::GetCountOf(storageResources)}};

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);

        nri::Descriptor* constantBuffer = Get(Descriptor::MorphTargetUpdatePrimitives_ConstantBuffer);
        NRI.UpdateDynamicConstantBuffers(*descriptorSet, 0, 1, &constantBuffer);
    }

    { // DescriptorSet::SharcPing4
        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::SharcHashEntries_StorageBuffer),
            Get(Descriptor::SharcHashCopyOffset_StorageBuffer),
            Get(Descriptor::SharcVoxelDataPing_StorageBuffer),
            Get(Descriptor::SharcVoxelDataPong_StorageBuffer),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_SHARC, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::SharcPong4
        const nri::Descriptor* storageResources[] = {
            Get(Descriptor::SharcHashEntries_StorageBuffer),
            Get(Descriptor::SharcHashCopyOffset_StorageBuffer),
            Get(Descriptor::SharcVoxelDataPong_StorageBuffer),
            Get(Descriptor::SharcVoxelDataPing_StorageBuffer),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_SHARC, &descriptorSet, 1, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {storageResources, helper::GetCountOf(storageResources)},
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }
}

void Sample::CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, nri::Dim_t width, nri::Dim_t height, nri::Mip_t mipNum, nri::Dim_t arraySize, nri::TextureUsageBits usage, nri::AccessBits access) {
    nri::AllocateTextureDesc allocateTextureDesc = {};
    allocateTextureDesc.desc.type = nri::TextureType::TEXTURE_2D;
    allocateTextureDesc.desc.usage = usage;
    allocateTextureDesc.desc.format = format;
    allocateTextureDesc.desc.width = width;
    allocateTextureDesc.desc.height = height;
    allocateTextureDesc.desc.depth = 1;
    allocateTextureDesc.desc.mipNum = mipNum;
    allocateTextureDesc.desc.layerNum = arraySize;
    allocateTextureDesc.desc.sampleNum = 1;
    allocateTextureDesc.memoryLocation = nri::MemoryLocation::DEVICE;

    nri::Texture* texture = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateTexture(*m_Device, allocateTextureDesc, texture));
    m_Textures.push_back(texture);

    if (access != nri::AccessBits::UNKNOWN) {
        nri::Layout layout = nri::Layout::SHADER_RESOURCE;
        if (access & nri::AccessBits::COPY_SOURCE)
            layout = nri::Layout::COPY_SOURCE;
        else if (access & nri::AccessBits::COPY_DESTINATION)
            layout = nri::Layout::COPY_DESTINATION;
        else if (access & nri::AccessBits::SHADER_RESOURCE_STORAGE)
            layout = nri::Layout::SHADER_RESOURCE_STORAGE;

        nri::TextureBarrierDesc transition = nri::TextureBarrierFromUnknown(texture, {access, layout});
        m_TextureStates.push_back(transition);
    }

    descriptorDescs.push_back({debugName, texture, format, usage, nri::BufferUsageBits::NONE, arraySize > 1});
}

void Sample::CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage) {
    if (!elements)
        elements = 1;

    nri::AllocateBufferDesc allocateBufferDesc = {};
    allocateBufferDesc.desc.size = elements * stride;
    allocateBufferDesc.desc.structureStride = format == nri::Format::UNKNOWN ? stride : 0;
    allocateBufferDesc.desc.usage = usage;
    allocateBufferDesc.memoryLocation = nri::MemoryLocation::DEVICE;

    nri::Buffer* buffer = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateBuffer(*m_Device, allocateBufferDesc, buffer));
    m_Buffers.push_back(buffer);

    if (!(usage & nri::BufferUsageBits::SCRATCH_BUFFER))
        descriptorDescs.push_back({debugName, buffer, format, nri::TextureUsageBits::NONE, usage, false});
}

void Sample::UploadStaticData() {
    std::vector<PrimitiveData> primitiveData(m_Scene.totalInstancedPrimitivesNum);

    for (utils::MeshInstance& meshInstance : m_Scene.meshInstances) {
        utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];
        uint32_t triangleNum = mesh.indexNum / 3;
        uint32_t staticPrimitiveOffset = mesh.indexOffset / 3;

        for (uint32_t j = 0; j < triangleNum; j++) {
            uint32_t staticPrimitiveIndex = staticPrimitiveOffset + j;

            const utils::UnpackedVertex& v0 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[staticPrimitiveIndex * 3]];
            const utils::UnpackedVertex& v1 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[staticPrimitiveIndex * 3 + 1]];
            const utils::UnpackedVertex& v2 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[staticPrimitiveIndex * 3 + 2]];

            float2 n0 = Packing::EncodeUnitVector(float3(v0.N), true);
            float2 n1 = Packing::EncodeUnitVector(float3(v1.N), true);
            float2 n2 = Packing::EncodeUnitVector(float3(v2.N), true);

            float2 t0 = Packing::EncodeUnitVector(float3(v0.T) + 1e-6f, true);
            float2 t1 = Packing::EncodeUnitVector(float3(v1.T) + 1e-6f, true);
            float2 t2 = Packing::EncodeUnitVector(float3(v2.T) + 1e-6f, true);

            PrimitiveData& data = primitiveData[meshInstance.primitiveOffset + j];
            data.uv0 = Packing::float2_to_float16_t2(float2(v0.uv[0], v0.uv[1]));
            data.uv1 = Packing::float2_to_float16_t2(float2(v1.uv[0], v1.uv[1]));
            data.uv2 = Packing::float2_to_float16_t2(float2(v2.uv[0], v2.uv[1]));

            data.n0 = Packing::float2_to_float16_t2(float2(n0.x, n0.y));
            data.n1 = Packing::float2_to_float16_t2(float2(n1.x, n1.y));
            data.n2 = Packing::float2_to_float16_t2(float2(n2.x, n2.y));

            data.t0 = Packing::float2_to_float16_t2(float2(t0.x, t0.y));
            data.t1 = Packing::float2_to_float16_t2(float2(t1.x, t1.y));
            data.t2 = Packing::float2_to_float16_t2(float2(t2.x, t2.y));

            data.bitangentSign_unused = Packing::float2_to_float16_t2(float2(v0.T[3], 0.0f));

            const utils::Primitive& primitive = m_Scene.primitives[staticPrimitiveIndex];
            data.worldArea = primitive.worldArea;
            data.uvArea = primitive.uvArea;
        }
    }

    // Gather subresources for read-only textures
    std::vector<nri::TextureSubresourceUploadDesc> subresources;
    for (const utils::Texture* texture : m_Scene.textures) {
        for (uint32_t layer = 0; layer < texture->GetArraySize(); layer++) {
            for (uint32_t mip = 0; mip < texture->GetMipNum(); mip++) {
                nri::TextureSubresourceUploadDesc subresource;
                texture->GetSubresource(subresource, mip, layer);

                subresources.push_back(subresource);
            }
        }
    }

    // Gather upload data for read-only textures
    std::vector<nri::TextureUploadDesc> textureUploadDescs;
    size_t subresourceOffset = 0;

    for (size_t i = 0; i < m_Scene.textures.size(); i++) {
        const utils::Texture* texture = m_Scene.textures[i];
        textureUploadDescs.push_back({&subresources[subresourceOffset], Get((Texture)((size_t)Texture::MaterialTextures + i)), {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}});

        nri::Mip_t mipNum = texture->GetMipNum();
        nri::Dim_t arraySize = texture->GetArraySize();
        subresourceOffset += size_t(arraySize) * size_t(mipNum);
    }

    // Append textures without data to initialize initial state
    for (const nri::TextureBarrierDesc& state : m_TextureStates) {
        nri::TextureUploadDesc desc = {};
        desc.after = {state.after.access, state.after.layout};
        desc.texture = (nri::Texture*)state.texture;

        textureUploadDescs.push_back(desc);
    }

    std::vector<utils::Index> morphMeshIndices(m_Scene.morphMeshTotalIndicesNum);
    uint32_t morphMeshIndexOffset = 0;

    // Compact static base pose data
    for (uint32_t morphMeshIndex : m_Scene.morphMeshes) {
        const utils::Mesh& mesh = m_Scene.meshes[morphMeshIndex];
        memcpy(morphMeshIndices.data() + morphMeshIndexOffset, &m_Scene.indices[mesh.indexOffset], mesh.indexNum * sizeof(m_Scene.indices[mesh.indexOffset]));
        morphMeshIndexOffset += mesh.indexNum;
    }

    // Buffer data
    nri::BufferUploadDesc bufferUploadDescs[] = {
        {primitiveData.data(), Get(Buffer::PrimitiveData), {nri::AccessBits::SHADER_RESOURCE}},
        {morphMeshIndices.data(), Get(Buffer::MorphMeshIndices), {nri::AccessBits::SHADER_RESOURCE}},
        {m_Scene.morphVertices.data(), Get(Buffer::MorphMeshVertices), {nri::AccessBits::SHADER_RESOURCE}}};

    // Upload data and apply states
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, textureUploadDescs.data(), helper::GetCountOf(textureUploadDescs), bufferUploadDescs, helper::GetCountOf(bufferUploadDescs)));
}

void Sample::GatherInstanceData() {
    bool isAnimatedObjects = m_Settings.animatedObjects;
    if (m_Settings.blink) {
        double period = 0.0003 * m_Timer.GetTimeStamp() * (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
        isAnimatedObjects &= WaveTriangle(period) > 0.5;
    }

    uint64_t staticInstanceCount = m_Scene.instances.size() - m_AnimatedInstances.size();
    uint64_t instanceCount = staticInstanceCount + (isAnimatedObjects ? m_Settings.animatedObjectNum : 0);
    uint32_t instanceIndex = 0;

    m_InstanceData.clear();
    m_WorldTlasData.clear();
    m_LightTlasData.clear();

    float4x4 mCameraTranslation = float4x4::Identity();
    mCameraTranslation.AddTranslation(m_Camera.GetRelative(double3::Zero()));
    mCameraTranslation.Transpose3x4();

    // Add static opaque (includes emissives)
    if (m_OpaqueObjectsNum) {
        nri::TopLevelInstance& topLevelInstance = m_WorldTlasData.emplace_back();
        topLevelInstance = {};
        memcpy(topLevelInstance.transform, mCameraTranslation.a, sizeof(topLevelInstance.transform));
        topLevelInstance.instanceId = instanceIndex;
        topLevelInstance.mask = FLAG_NON_TRANSPARENT;
        topLevelInstance.shaderBindingTableLocalOffset = 0;
        topLevelInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        topLevelInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_MergedOpaque));

        instanceIndex += m_OpaqueObjectsNum;
    }

    // Add static transparent
    if (m_TransparentObjectsNum) {
        nri::TopLevelInstance& topLevelInstance = m_WorldTlasData.emplace_back();
        topLevelInstance = {};
        memcpy(topLevelInstance.transform, mCameraTranslation.a, sizeof(topLevelInstance.transform));
        topLevelInstance.instanceId = instanceIndex;
        topLevelInstance.mask = FLAG_TRANSPARENT;
        topLevelInstance.shaderBindingTableLocalOffset = 0;
        topLevelInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        topLevelInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_MergedTransparent));

        instanceIndex += m_TransparentObjectsNum;
    }

    // Add static emissives (only emissives in a separate TLAS)
    if (m_EmissiveObjectsNum) {
        nri::TopLevelInstance& topLevelInstance = m_LightTlasData.emplace_back();
        topLevelInstance = {};
        memcpy(topLevelInstance.transform, mCameraTranslation.a, sizeof(topLevelInstance.transform));
        topLevelInstance.instanceId = instanceIndex;
        topLevelInstance.mask = FLAG_NON_TRANSPARENT;
        topLevelInstance.shaderBindingTableLocalOffset = 0;
        topLevelInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        topLevelInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_MergedEmissive));

        instanceIndex += m_EmissiveObjectsNum;
    }

    // Gather instance data and add dynamic objects
    // IMPORTANT: instance data order must match geometry layout in BLAS-es
    for (uint32_t mode = (uint32_t)AccelerationStructure::BLAS_MergedOpaque; mode <= (uint32_t)AccelerationStructure::BLAS_Other; mode++) {
        for (size_t i = m_ProxyInstancesNum; i < instanceCount; i++) {
            utils::Instance& instance = m_Scene.instances[i];
            const utils::Material& material = m_Scene.materials[instance.materialIndex];

            if (material.IsOff())
                continue;

            if (mode == (uint32_t)AccelerationStructure::BLAS_MergedOpaque) {
                if (instance.allowUpdate || material.IsTransparent())
                    continue;
            } else if (mode == (uint32_t)AccelerationStructure::BLAS_MergedTransparent) {
                if (instance.allowUpdate || !material.IsTransparent())
                    continue;
            } else if (mode == (uint32_t)AccelerationStructure::BLAS_MergedEmissive) {
                if (instance.allowUpdate || !material.IsEmissive())
                    continue;
            } else {
                if (!instance.allowUpdate)
                    continue;
            }

            float4x4 mObjectToWorld = float4x4::Identity();
            float4x4 mOverloadedMatrix = float4x4::Identity();
            bool isLeftHanded = false;

            if (instance.allowUpdate) {
                const utils::MeshInstance& meshInstance = m_Scene.meshInstances[instance.meshInstanceIndex];
                const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];

                // Current & previous transform
                mObjectToWorld = instance.rotation;
                float4x4 mObjectToWorldPrev = instance.rotationPrev;

                if (any(instance.scale != 1.0f)) {
                    float4x4 translation;
                    translation.SetupByTranslation(float3(instance.position) - mesh.aabb.GetCenter());

                    float4x4 scale;
                    scale.SetupByScale(instance.scale);

                    float4x4 translationInv = translation;
                    translationInv.InvertOrtho();

                    float4x4 transform = translationInv * (scale * translation);

                    mObjectToWorld = mObjectToWorld * transform;
                    mObjectToWorldPrev = mObjectToWorldPrev * transform;
                }

                mObjectToWorld.AddTranslation(m_Camera.GetRelative(instance.position));
                mObjectToWorldPrev.AddTranslation(m_Camera.GetRelative(instance.positionPrev));

                if (mesh.HasMorphTargets())
                    mOverloadedMatrix = mObjectToWorldPrev;
                else {
                    // World to world (previous state) transform
                    // FP64 used to avoid imprecision problems on close up views (InvertOrtho can't be used due to scaling factors)
                    double4x4 dmWorldToObject = double4x4(mObjectToWorld);
                    dmWorldToObject.Invert();

                    double4x4 dmObjectToWorldPrev = double4x4(mObjectToWorldPrev);
                    mOverloadedMatrix = float4x4(dmObjectToWorldPrev * dmWorldToObject);
                }

                // Update previous state
                instance.positionPrev = instance.position;
                instance.rotationPrev = instance.rotation;
            } else {
                mObjectToWorld = mCameraTranslation;

                // Static geometry doesn't have "prev" transformation, reuse this matrix to pass object rotation needed for normals
                mOverloadedMatrix = instance.rotation;

                // Transform can be left-handed (mirroring), in this case normals need flipping
                isLeftHanded = instance.rotation.IsLeftHanded();
            }

            mObjectToWorld.Transpose3x4();
            mOverloadedMatrix.Transpose3x4();

            // Add instance data
            const utils::MeshInstance& meshInstance = m_Scene.meshInstances[instance.meshInstanceIndex];
            uint32_t baseTextureIndex = instance.materialIndex * TEXTURES_PER_MATERIAL;
            float3 scale = instance.rotation.GetScale();

            uint32_t flags = 0;
            if (!instance.allowUpdate)
                flags |= FLAG_STATIC;
            if (meshInstance.morphedVertexOffset != utils::InvalidIndex)
                flags |= FLAG_DEFORMABLE;
            if (material.isHair)
                flags |= FLAG_HAIR;
            if (material.isLeaf)
                flags |= FLAG_LEAF;
            if (material.IsTransparent())
                flags |= FLAG_TRANSPARENT;
            if (i >= staticInstanceCount) {
                if (m_Settings.emission && m_Settings.emissiveObjects && (i % 3 == 0))
                    flags |= FLAG_FORCED_EMISSION;
                else if (m_GlassObjects && (i % 4 == 0))
                    flags |= FLAG_TRANSPARENT;
            }

            if (!(flags & FLAG_TRANSPARENT))
                flags |= FLAG_NON_TRANSPARENT;

            InstanceData& instanceData = m_InstanceData.emplace_back();
            instanceData = {};
            instanceData.mOverloadedMatrix0 = mOverloadedMatrix.Col(0);
            instanceData.mOverloadedMatrix1 = mOverloadedMatrix.Col(1);
            instanceData.mOverloadedMatrix2 = mOverloadedMatrix.Col(2);
            instanceData.baseColorAndMetalnessScale = material.baseColorAndMetalnessScale;
            instanceData.emissionAndRoughnessScale = material.emissiveAndRoughnessScale;
            instanceData.textureOffsetAndFlags = baseTextureIndex | (flags << FLAG_FIRST_BIT);
            instanceData.primitiveOffset = meshInstance.primitiveOffset;
            instanceData.morphedPrimitiveOffset = meshInstance.morphedPrimitiveOffset;
            instanceData.scale = (isLeftHanded ? -1.0f : 1.0f) * max(scale.x, max(scale.y, scale.z));

            // Add dynamic geometry
            if (instance.allowUpdate) {
                nri::TopLevelInstance topLevelInstance = {};
                memcpy(topLevelInstance.transform, mObjectToWorld.a, sizeof(topLevelInstance.transform));
                topLevelInstance.instanceId = instanceIndex++;
                topLevelInstance.mask = flags;
                topLevelInstance.shaderBindingTableLocalOffset = 0;
                topLevelInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE | (material.IsAlphaOpaque() ? nri::TopLevelInstanceBits::NONE : nri::TopLevelInstanceBits::FORCE_OPAQUE);
                topLevelInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_AccelerationStructures[meshInstance.blasIndex]);

                m_WorldTlasData.push_back(topLevelInstance);

                if (flags == FLAG_FORCED_EMISSION || material.IsEmissive())
                    m_LightTlasData.push_back(topLevelInstance);
            }
        }
    }

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    {
        nri::DataSize dataChunk = {};
        dataChunk.data = m_InstanceData.data();
        dataChunk.size = m_InstanceData.size() * sizeof(InstanceData);

        nri::StreamBufferDataDesc streamBufferDataDesc = {};
        streamBufferDataDesc.dataChunks = &dataChunk;
        streamBufferDataDesc.dataChunkNum = 1;
        streamBufferDataDesc.dstBuffer = Get(Buffer::InstanceData);

        NRI.StreamBufferData(*m_Streamer, streamBufferDataDesc);
    }

    {
        nri::DataSize dataChunk = {};
        dataChunk.data = m_WorldTlasData.data();
        dataChunk.size = m_WorldTlasData.size() * sizeof(nri::TopLevelInstance);

        nri::StreamBufferDataDesc streamBufferDataDesc = {};
        streamBufferDataDesc.dataChunks = &dataChunk;
        streamBufferDataDesc.dataChunkNum = 1;
        streamBufferDataDesc.placementAlignment = deviceDesc.memoryAlignment.accelerationStructureOffset;

        m_WorldTlasDataLocation = NRI.StreamBufferData(*m_Streamer, streamBufferDataDesc);
    }

    {
        nri::DataSize dataChunk = {};
        dataChunk.data = m_LightTlasData.data();
        dataChunk.size = m_LightTlasData.size() * sizeof(nri::TopLevelInstance);

        nri::StreamBufferDataDesc streamBufferDataDesc = {};
        streamBufferDataDesc.dataChunks = &dataChunk;
        streamBufferDataDesc.dataChunkNum = 1;
        streamBufferDataDesc.placementAlignment = deviceDesc.memoryAlignment.accelerationStructureOffset;

        m_LightTlasDataLocation = NRI.StreamBufferData(*m_Streamer, streamBufferDataDesc);
    }
}

void GetBasis(float3 N, float3& T, float3& B) {
    float sz = sign(N.z);
    float a = 1.0f / (sz + N.z);
    float ya = N.y * a;
    float b = N.x * ya;
    float c = N.x * sz;

    T = float3(c * N.x * a - 1.0f, sz * b, c);
    B = float3(b, N.y * ya - sz, N.y);
}

void Sample::UpdateConstantBuffer(uint32_t frameIndex, float resetHistoryFactor) {
    float3 sunDirection = GetSunDirection();
    float3 sunT, sunB;
    GetBasis(sunDirection, sunT, sunB);

    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectWprev = uint32_t(m_RenderResolution.x * m_SettingsPrev.resolutionScale + 0.5f);
    uint32_t rectHprev = uint32_t(m_RenderResolution.y * m_SettingsPrev.resolutionScale + 0.5f);

    float2 renderSize = float2(float(m_RenderResolution.x), float(m_RenderResolution.y));
    float2 outputSize = float2(float(GetOutputResolution().x), float(GetOutputResolution().y));
    float2 windowSize = float2(float(GetWindowResolution().x), float(GetWindowResolution().y));
    float2 rectSize = float2(float(rectW), float(rectH));
    float2 rectSizePrev = float2(float(rectWprev), float(rectHprev));
    float2 jitter = (m_Settings.cameraJitter ? m_Camera.state.viewportJitter : 0.0f) / rectSize;

    float3 viewDir = float3(m_Camera.state.mViewToWorld[2].xyz) * (m_PositiveZ ? -1.0f : 1.0f);
    float3 cameraGlobalPos = float3(m_Camera.state.globalPosition);
    float3 cameraGlobalPosPrev = float3(m_Camera.statePrev.globalPosition);

    float emissionIntensity = m_Settings.emissionIntensity * float(m_Settings.emission);
    float nearZ = (m_PositiveZ ? 1.0f : -1.0f) * NEAR_Z * m_Settings.meterToUnitsMultiplier;
    float baseMipBias = ((m_Settings.TAA || IsDlssEnabled()) ? -0.5f : 0.0f) + log2f(m_Settings.resolutionScale);
    float mipBias = baseMipBias + log2f(renderSize.x / outputSize.x);

    uint32_t onScreen = m_Settings.onScreen + (NRD_MODE >= OCCLUSION ? SHOW_AMBIENT_OCCLUSION : 0); // preserve original mapping

    float fps = 1000.0f / m_Timer.GetSmoothedFrameTime();
    fps = min(fps, 121.0f);
    float otherMaxAccumulatedFrameNum = fps * ACCUMULATION_TIME;
    otherMaxAccumulatedFrameNum = min(otherMaxAccumulatedFrameNum, float(MAX_HISTORY_FRAME_NUM));
    otherMaxAccumulatedFrameNum *= resetHistoryFactor;

    uint32_t sharcMaxAccumulatedFrameNum = (uint32_t)(otherMaxAccumulatedFrameNum * (m_Settings.boost ? 0.667f : 1.0f) + 0.5f);
    float taaMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.5f;
    float prevFrameMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.3f;

    nrd::HitDistanceParameters hitDistanceParameters = {};
    hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;

    float minProbability = 0.0f;
    if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC) {
        nrd::HitDistanceReconstructionMode mode = nrd::HitDistanceReconstructionMode::OFF;
        if (m_Settings.denoiser == DENOISER_REBLUR)
            mode = m_ReblurSettings.hitDistanceReconstructionMode;
        else if (m_Settings.denoiser == DENOISER_RELAX)
            mode = m_RelaxSettings.hitDistanceReconstructionMode;

        // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
        if (mode == nrd::HitDistanceReconstructionMode::AREA_3X3)
            minProbability = 1.0f / 4.0f;
        else if (mode == nrd::HitDistanceReconstructionMode::AREA_5X5)
            minProbability = 1.0f / 16.0f;
    }

    float project[3];
    float4 frustum;
    uint32_t flags = 0;
    DecomposeProjection(STYLE_D3D, STYLE_D3D, m_Camera.state.mViewToClip, &flags, nullptr, nullptr, frustum.a, project, nullptr);
    float orthoMode = (flags & PROJ_ORTHO) == 0 ? 0.0f : -1.0f;

    nri::DisplayDesc displayDesc = {};
    NRI.GetDisplayDesc(*m_SwapChain, displayDesc);

    m_SdrScale = displayDesc.sdrLuminance / 80.0f;

    GlobalConstants constants;
    {
        constants.gViewToWorld = m_Camera.state.mViewToWorld;
        constants.gViewToClip = m_Camera.state.mViewToClip;
        constants.gWorldToView = m_Camera.state.mWorldToView;
        constants.gWorldToViewPrev = m_Camera.statePrev.mWorldToView;
        constants.gWorldToClip = m_Camera.state.mWorldToClip;
        constants.gWorldToClipPrev = m_Camera.statePrev.mWorldToClip;
        constants.gHitDistParams = float4(hitDistanceParameters.A, hitDistanceParameters.B, hitDistanceParameters.C, hitDistanceParameters.D);
        constants.gCameraFrustum = frustum;
        constants.gSunBasisX = float4(sunT, 0.0f);
        constants.gSunBasisY = float4(sunB, 0.0f);
        constants.gSunDirection = float4(sunDirection, 0.0f);
        constants.gCameraGlobalPos = float4(cameraGlobalPos, CAMERA_RELATIVE);
        constants.gCameraGlobalPosPrev = float4(cameraGlobalPosPrev, 0.0f);
        constants.gViewDirection = float4(viewDir, 0.0f);
        constants.gHairBaseColor = pow(m_HairBaseColor, float4(2.2f));
        constants.gHairBetas = m_HairBetas;
        constants.gWindowSize = windowSize;
        constants.gOutputSize = outputSize;
        constants.gRenderSize = renderSize;
        constants.gRectSize = rectSize;
        constants.gInvWindowSize = float2(1.0f, 1.0f) / windowSize;
        constants.gInvOutputSize = float2(1.0f, 1.0f) / outputSize;
        constants.gInvRenderSize = float2(1.0f, 1.0f) / renderSize;
        constants.gInvRectSize = float2(1.0f, 1.0f) / rectSize;
        constants.gRectSizePrev = rectSizePrev;
        constants.gNearZ = nearZ;
        constants.gEmissionIntensity = emissionIntensity;
        constants.gJitter = jitter;
        constants.gSeparator = m_Settings.separator;
        constants.gRoughnessOverride = m_Settings.roughnessOverride;
        constants.gMetalnessOverride = m_Settings.metalnessOverride;
        constants.gUnitToMetersMultiplier = 1.0f / m_Settings.meterToUnitsMultiplier;
        constants.gIndirectDiffuse = m_Settings.indirectDiffuse ? 1.0f : 0.0f;
        constants.gIndirectSpecular = m_Settings.indirectSpecular ? 1.0f : 0.0f;
        constants.gTanSunAngularRadius = tan(radians(m_Settings.sunAngularDiameter * 0.5f));
        constants.gTanPixelAngularRadius = tan(0.5f * radians(m_Settings.camFov) / rectSize.x);
        constants.gDebug = m_Settings.debug;
        constants.gPrevFrameConfidence = (m_Settings.usePrevFrame && NRD_MODE < OCCLUSION && !m_Settings.RR) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f;
        constants.gMinProbability = minProbability;
        constants.gUnproject = 1.0f / (0.5f * rectH * project[1]);
        constants.gAperture = m_DofAperture * 0.01f;
        constants.gFocalDistance = m_DofFocalDistance;
        constants.gFocalLength = (0.5f * (35.0f * 0.001f)) / tan(radians(m_Settings.camFov * 0.5f)); // for 35 mm sensor size (aka old-school 35 mm film)
        constants.gTAA = (m_Settings.denoiser != DENOISER_REFERENCE && m_Settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f;
        constants.gHdrScale = displayDesc.isHDR ? displayDesc.maxLuminance / 80.0f : 1.0f;
        constants.gExposure = m_Settings.exposure;
        constants.gMipBias = mipBias;
        constants.gOrthoMode = orthoMode;
        constants.gSharcMaxAccumulatedFrameNum = sharcMaxAccumulatedFrameNum;
        constants.gDenoiserType = (uint32_t)m_Settings.denoiser;
        constants.gDisableShadowsAndEnableImportanceSampling = (sunDirection.z < 0.0f && m_Settings.importanceSampling && NRD_MODE < OCCLUSION) ? 1 : 0;
        constants.gOnScreen = onScreen;
        constants.gFrameIndex = frameIndex;
        constants.gForcedMaterial = m_Settings.forcedMaterial;
        constants.gUseNormalMap = m_Settings.normalMap ? 1 : 0;
        constants.gTracingMode = m_Settings.RR ? RESOLUTION_FULL_PROBABILISTIC : m_Settings.tracingMode;
        constants.gSampleNum = m_Settings.rpp;
        constants.gBounceNum = m_Settings.bounceNum;
        constants.gResolve = (m_Settings.denoiser == DENOISER_REFERENCE || m_Settings.RR) ? false : m_Resolve;
        constants.gPSR = m_Settings.PSR && m_Settings.tracingMode != RESOLUTION_HALF;
        constants.gSHARC = m_Settings.SHARC;
        constants.gValidation = m_ShowValidationOverlay && m_Settings.denoiser != DENOISER_REFERENCE && m_Settings.separator != 1.0f;
        constants.gTrimLobe = m_Settings.specularLobeTrimming ? 1 : 0;
        constants.gSR = (m_Settings.SR && !m_Settings.RR) ? 1 : 0;
        constants.gRR = m_Settings.RR ? 1 : 0;
        constants.gIsSrgb = (m_IsSrgb && (onScreen == SHOW_FINAL || onScreen == SHOW_BASE_COLOR)) ? 1 : 0;
    }

    m_GlobalConstantBufferOffset = NRI.StreamConstantData(*m_Streamer, &constants, sizeof(constants));
}

uint16_t Sample::BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM>& transitions) {
    uint32_t n = 0;

    for (uint32_t i = 0; i < stateNum; i++) {
        const TextureState& state = states[i];
        nri::TextureBarrierDesc& transition = GetState(state.texture);

        bool isStateChanged = transition.after.access != state.after.access || transition.after.layout != state.after.layout;
        bool isStorageBarrier = transition.after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE && state.after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE;
        if (isStateChanged || isStorageBarrier)
            transitions[n++] = nri::TextureBarrierFromState(transition, {state.after.access, state.after.layout});
    }

    return (uint16_t)n;
}

void Sample::RestoreBindings(nri::CommandBuffer& commandBuffer, bool isEven, bool needRayTracing) {
    NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
    NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
    NRI.CmdSetDescriptorSet(commandBuffer, SET_GLOBAL, *Get(DescriptorSet::Global0), &m_GlobalConstantBufferOffset);

    if (needRayTracing) {
        NRI.CmdSetDescriptorSet(commandBuffer, SET_RAY_TRACING, *Get(DescriptorSet::RayTracing2), nullptr);
        NRI.CmdSetDescriptorSet(commandBuffer, SET_SHARC, isEven ? *Get(DescriptorSet::SharcPing4) : *Get(DescriptorSet::SharcPong4), nullptr);
    }
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::nriBeginAnnotation("Render frame", nri::BGRA_UNUSED);

    std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM> optimizedTransitions = {};

    bool wantPrintf = IsButtonPressed(Button::Middle) || IsKeyToggled(Key::P);
    bool isEven = !(frameIndex & 0x1);
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];
    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;

    // Sizes
    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectGridW = (rectW + 15) / 16;
    uint32_t rectGridH = (rectH + 15) / 16;
    uint32_t outputGridW = (GetOutputResolution().x + 15) / 16;
    uint32_t outputGridH = (GetOutputResolution().y + 15) / 16;
    uint32_t windowGridW = (GetWindowResolution().x + 15) / 16;
    uint32_t windowGridH = (GetWindowResolution().y + 15) / 16;

    // NRD user pool
    nrd::UserPool userPool = {};
    {
        // Common
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_MV, &GetState(Texture::Mv));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_NORMAL_ROUGHNESS, &GetState(Texture::Normal_Roughness));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_VIEWZ, &GetState(Texture::ViewZ));

        // (Optional) Needed to allow IN_MV modification on the NRD side
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_BASECOLOR_METALNESS, &GetState(Texture::BaseColor_Metalness));

        // (Optional) Validation
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_VALIDATION, &GetState(Texture::Validation));

        // Diffuse
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, &GetState(Texture::Unfiltered_Diff));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, &GetState(Texture::Diff));

        // Diffuse occlusion
#if (NRD_MODE == OCCLUSION)
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_DIFF_HITDIST, &GetState(Texture::Unfiltered_Diff));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_HITDIST, &GetState(Texture::Diff));
#endif

        // Diffuse SH
#if (NRD_MODE == SH)
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH0, &GetState(Texture::Unfiltered_Diff));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH1, &GetState(Texture::Unfiltered_DiffSh));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH0, &GetState(Texture::Diff));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH1, &GetState(Texture::DiffSh));
#endif

        // Diffuse directional occlusion
#if (NRD_MODE == DIRECTIONAL_OCCLUSION)
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_DIFF_DIRECTION_HITDIST, &GetState(Texture::Unfiltered_Diff));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_DIRECTION_HITDIST, &GetState(Texture::Diff));
#endif

        // Specular
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, &GetState(Texture::Unfiltered_Spec));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, &GetState(Texture::Spec));

        // Specular occlusion
#if (NRD_MODE == OCCLUSION)
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_SPEC_HITDIST, &GetState(Texture::Unfiltered_Spec));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_HITDIST, &GetState(Texture::Spec));
#endif

        // Specular SH
#if (NRD_MODE == SH)
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH0, &GetState(Texture::Unfiltered_Spec));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH1, &GetState(Texture::Unfiltered_SpecSh));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH0, &GetState(Texture::Spec));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH1, &GetState(Texture::SpecSh));
#endif

        // SIGMA
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_PENUMBRA, &GetState(Texture::Unfiltered_Penumbra));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_TRANSLUCENCY, &GetState(Texture::Unfiltered_Translucency));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, &GetState(Texture::Shadow));

        // REFERENCE
        nrd::Integration_SetResource(userPool, nrd::ResourceType::IN_SIGNAL, &GetState(Texture::Composed));
        nrd::Integration_SetResource(userPool, nrd::ResourceType::OUT_SIGNAL, &GetState(Texture::Composed));
    }

    // NRD common settings
    nrd::CommonSettings commonSettings = {};
    memcpy(commonSettings.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));
    memcpy(commonSettings.viewToClipMatrixPrev, &m_Camera.statePrev.mViewToClip, sizeof(m_Camera.statePrev.mViewToClip));
    memcpy(commonSettings.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
    memcpy(commonSettings.worldToViewMatrixPrev, &m_Camera.statePrev.mWorldToView, sizeof(m_Camera.statePrev.mWorldToView));
    commonSettings.motionVectorScale[0] = 1.0f / float(rectW);
    commonSettings.motionVectorScale[1] = 1.0f / float(rectH);
    commonSettings.motionVectorScale[2] = m_Settings.mvType != MV_2D ? 1.0f : 0.0f;
    commonSettings.cameraJitter[0] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.x : 0.0f;
    commonSettings.cameraJitter[1] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.y : 0.0f;
    commonSettings.cameraJitterPrev[0] = m_Settings.cameraJitter ? m_Camera.statePrev.viewportJitter.x : 0.0f;
    commonSettings.cameraJitterPrev[1] = m_Settings.cameraJitter ? m_Camera.statePrev.viewportJitter.y : 0.0f;
    commonSettings.resourceSize[0] = (uint16_t)m_RenderResolution.x;
    commonSettings.resourceSize[1] = (uint16_t)m_RenderResolution.y;
    commonSettings.resourceSizePrev[0] = (uint16_t)m_RenderResolution.x;
    commonSettings.resourceSizePrev[1] = (uint16_t)m_RenderResolution.y;
    commonSettings.rectSize[0] = (uint16_t)(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    commonSettings.rectSize[1] = (uint16_t)(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    commonSettings.rectSizePrev[0] = (uint16_t)(m_RenderResolution.x * m_SettingsPrev.resolutionScale + 0.5f);
    commonSettings.rectSizePrev[1] = (uint16_t)(m_RenderResolution.y * m_SettingsPrev.resolutionScale + 0.5f);
    commonSettings.viewZScale = 1.0f;
    commonSettings.denoisingRange = GetDenoisingRange();
    commonSettings.disocclusionThreshold = 0.01f;
    commonSettings.disocclusionThresholdAlternate = 0.05f;
    commonSettings.splitScreen = (m_Settings.denoiser == DENOISER_REFERENCE || m_Settings.RR) ? 1.0f : m_Settings.separator;
    commonSettings.printfAt[0] = wantPrintf ? (uint16_t)ImGui::GetIO().MousePos.x : 9999;
    commonSettings.printfAt[1] = wantPrintf ? (uint16_t)ImGui::GetIO().MousePos.y : 9999;
    commonSettings.debug = m_Settings.debug;
    commonSettings.frameIndex = frameIndex;
    commonSettings.accumulationMode = m_ForceHistoryReset ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.isMotionVectorInWorldSpace = false;
    commonSettings.isBaseColorMetalnessAvailable = true;
    commonSettings.enableValidation = m_ShowValidationOverlay;

#if (NRD_NORMAL_ENCODING == 2)
    commonSettings.strandMaterialID = MATERIAL_ID_HAIR;
    commonSettings.strandThickness = STRAND_THICKNESS;

#    if (USE_CAMERA_ATTACHED_REFLECTION_TEST == 1)
    commonSettings.cameraAttachedReflectionMaterialID = MATERIAL_ID_SELF_REFLECTION;
#    endif
#endif

    m_NRD.NewFrame();
    m_NRD.SetCommonSettings(commonSettings);

    const uint32_t dummyDynamicConstantOffset = 0;

    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        //======================================================================================================================================
        // Resolution independent
        //======================================================================================================================================

        { // Copy upload requests to destinations
            helper::Annotation annotation(NRI, commandBuffer, "Streamer");

            // TODO: is barrier from "SHADER_RESOURCE" to "COPY_DESTINATION" needed here for "Buffer::InstanceData"?

            NRI.CmdCopyStreamedData(commandBuffer, *m_Streamer);
        }

        // All-in-one pipeline layout
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);

        NRI.CmdSetDescriptorSet(commandBuffer, SET_GLOBAL, *Get(DescriptorSet::Global0), &m_GlobalConstantBufferOffset);

        // Update morph animation
        if (m_Settings.activeAnimation < m_Scene.animations.size() && m_Scene.animations[m_Settings.activeAnimation].morphMeshInstances.size() && (!m_Settings.pauseAnimation || !m_SettingsPrev.pauseAnimation || frameIndex == 0)) {
            const utils::Animation& animation = m_Scene.animations[m_Settings.activeAnimation];
            uint32_t animCurrBufferIndex = frameIndex & 0x1;
            uint32_t animPrevBufferIndex = frameIndex == 0 ? animCurrBufferIndex : 1 - animCurrBufferIndex;

            { // Update vertices
                helper::Annotation annotation(NRI, commandBuffer, "Morph mesh: update vertices");

                { // Transitions
                    const nri::BufferBarrierDesc bufferTransitions[] = {
                        // Output
                        {Get(Buffer::MorphedPositions), {nri::AccessBits::SHADER_RESOURCE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                        {Get(Buffer::MorphedAttributes), {nri::AccessBits::SHADER_RESOURCE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                    };

                    nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, bufferTransitions, helper::GetCountOf(bufferTransitions), nullptr, 0};
                    NRI.CmdBarrier(commandBuffer, transitionBarriers);
                }

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::MorphMeshUpdateVertices));

                for (const utils::WeightTrackMorphMeshIndex& weightTrackMeshInstance : animation.morphMeshInstances) {
                    const utils::WeightsAnimationTrack& weightsTrack = animation.weightTracks[weightTrackMeshInstance.weightTrackIndex];
                    const utils::MeshInstance& meshInstance = m_Scene.meshInstances[weightTrackMeshInstance.meshInstanceIndex];
                    const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];

                    uint32_t numShaderMorphTargets = min((uint32_t)(weightsTrack.activeValues.size()), MORPH_MAX_ACTIVE_TARGETS_NUM);
                    float totalWeight = 0.f;
                    for (uint32_t i = 0; i < numShaderMorphTargets; i++)
                        totalWeight += weightsTrack.activeValues[i].second;
                    float renormalizeScale = 1.0f / totalWeight;

                    MorphMeshUpdateVerticesConstants constants = {};
                    {
                        for (uint32_t i = 0; i < numShaderMorphTargets; i++) {
                            uint32_t morphTargetIndex = weightsTrack.activeValues[i].first;
                            uint32_t morphTargetVertexOffset = mesh.morphTargetVertexOffset + morphTargetIndex * mesh.vertexNum;

                            constants.gIndices[i / MORPH_ELEMENTS_PER_ROW_NUM].a[i % MORPH_ELEMENTS_PER_ROW_NUM] = morphTargetVertexOffset;
                            constants.gWeights[i / MORPH_ELEMENTS_PER_ROW_NUM].a[i % MORPH_ELEMENTS_PER_ROW_NUM] = renormalizeScale * weightsTrack.activeValues[i].second;
                        }
                        constants.gNumWeights = numShaderMorphTargets;
                        constants.gNumVertices = mesh.vertexNum;
                        constants.gPositionCurrFrameOffset = m_Scene.morphedVerticesNum * animCurrBufferIndex + meshInstance.morphedVertexOffset;
                        constants.gAttributesOutputOffset = meshInstance.morphedVertexOffset;
                    }

                    uint32_t dynamicConstantBufferOffset = NRI.StreamConstantData(*m_Streamer, &constants, sizeof(constants));
                    NRI.CmdSetDescriptorSet(commandBuffer, SET_MORPH, *Get(DescriptorSet::MorphTargetPose3), &dynamicConstantBufferOffset);

                    NRI.CmdDispatch(commandBuffer, {(mesh.vertexNum + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE, 1, 1});
                }

                { // Transitions
                    const nri::BufferBarrierDesc bufferTransitions[] = {
                        // Input
                        {Get(Buffer::MorphedPositions), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE}},
                        {Get(Buffer::MorphedAttributes), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE}},

                        // Output
                        {Get(Buffer::PrimitiveData), {nri::AccessBits::SHADER_RESOURCE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                        {Get(Buffer::MorphedPrimitivePrevPositions), {nri::AccessBits::SHADER_RESOURCE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                    };

                    nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, bufferTransitions, helper::GetCountOf(bufferTransitions), nullptr, 0};
                    NRI.CmdBarrier(commandBuffer, transitionBarriers);
                }
            }

            { // Update primitives
                helper::Annotation annotation(NRI, commandBuffer, "Morph mesh: update primitives");

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::MorphMeshUpdatePrimitives));

                for (const utils::WeightTrackMorphMeshIndex& weightTrackMeshInstance : animation.morphMeshInstances) {
                    const utils::MeshInstance& meshInstance = m_Scene.meshInstances[weightTrackMeshInstance.meshInstanceIndex];
                    const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];
                    uint32_t numPrimitives = mesh.indexNum / 3;

                    MorphMeshUpdatePrimitivesConstants constants = {};
                    {
                        constants.gPositionFrameOffsets.x = m_Scene.morphedVerticesNum * animCurrBufferIndex + meshInstance.morphedVertexOffset;
                        constants.gPositionFrameOffsets.y = m_Scene.morphedVerticesNum * animPrevBufferIndex + meshInstance.morphedVertexOffset;
                        constants.gNumPrimitives = numPrimitives;
                        constants.gIndexOffset = mesh.morphMeshIndexOffset;
                        constants.gAttributesOffset = meshInstance.morphedVertexOffset;
                        constants.gPrimitiveOffset = meshInstance.primitiveOffset;
                        constants.gMorphedPrimitiveOffset = meshInstance.morphedPrimitiveOffset;
                    }

                    uint32_t dynamicConstantBufferOffset = NRI.StreamConstantData(*m_Streamer, &constants, sizeof(constants));
                    NRI.CmdSetDescriptorSet(commandBuffer, SET_MORPH, *Get(DescriptorSet::MorphTargetUpdatePrimitives3), &dynamicConstantBufferOffset);

                    NRI.CmdDispatch(commandBuffer, {(numPrimitives + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE, 1, 1});
                }
            }

            { // Update BLAS
                helper::Annotation annotation(NRI, commandBuffer, "Morph mesh: BLAS");

                const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

                // Do build if the animation gets paused
                bool doBuild = m_Settings.pauseAnimation && !m_SettingsPrev.pauseAnimation;

                size_t scratchOffset = 0;
                for (const utils::WeightTrackMorphMeshIndex& weightTrackMeshInstance : animation.morphMeshInstances) {
                    const utils::MeshInstance& meshInstance = m_Scene.meshInstances[weightTrackMeshInstance.meshInstanceIndex];
                    const utils::Mesh& mesh = m_Scene.meshes[meshInstance.meshIndex];

                    nri::AccelerationStructure* accelerationStructure = m_AccelerationStructures[meshInstance.blasIndex];

                    nri::BottomLevelGeometryDesc bottomLevelGeometry = {};
                    bottomLevelGeometry.type = nri::BottomLevelGeometryType::TRIANGLES;
                    bottomLevelGeometry.flags = nri::BottomLevelGeometryBits::NONE; // will be set in TLAS instance
                    bottomLevelGeometry.triangles.vertexBuffer = Get(Buffer::MorphedPositions);
                    bottomLevelGeometry.triangles.vertexStride = sizeof(float16_t4);
                    bottomLevelGeometry.triangles.vertexOffset = bottomLevelGeometry.triangles.vertexStride * (m_Scene.morphedVerticesNum * animCurrBufferIndex + meshInstance.morphedVertexOffset);
                    bottomLevelGeometry.triangles.vertexNum = mesh.vertexNum;
                    bottomLevelGeometry.triangles.vertexFormat = nri::Format::RGBA16_SFLOAT;
                    bottomLevelGeometry.triangles.indexBuffer = Get(Buffer::MorphMeshIndices);
                    bottomLevelGeometry.triangles.indexOffset = mesh.morphMeshIndexOffset * sizeof(utils::Index);
                    bottomLevelGeometry.triangles.indexNum = mesh.indexNum;
                    bottomLevelGeometry.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

                    nri::BuildBottomLevelAccelerationStructureDesc buildBottomLevelAccelerationStructureDesc = {};
                    buildBottomLevelAccelerationStructureDesc.dst = accelerationStructure;
                    buildBottomLevelAccelerationStructureDesc.geometryNum = 1;
                    buildBottomLevelAccelerationStructureDesc.geometries = &bottomLevelGeometry;
                    buildBottomLevelAccelerationStructureDesc.scratchBuffer = Get(Buffer::MorphMeshScratch);
                    buildBottomLevelAccelerationStructureDesc.scratchOffset = scratchOffset;

                    if (doBuild) {
                        uint64_t size = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
                        scratchOffset += helper::Align(size, deviceDesc.memoryAlignment.scratchBufferOffset);
                        ;
                    } else {
                        buildBottomLevelAccelerationStructureDesc.src = accelerationStructure;

                        uint64_t size = NRI.GetAccelerationStructureUpdateScratchBufferSize(*accelerationStructure);
                        scratchOffset += helper::Align(size, deviceDesc.memoryAlignment.scratchBufferOffset);
                        ;
                    }

                    NRI.CmdBuildBottomLevelAccelerationStructures(commandBuffer, &buildBottomLevelAccelerationStructureDesc, 1);
                }

                { // Transitions
                    const nri::BufferBarrierDesc bufferTransitions[] = {
                        {Get(Buffer::PrimitiveData), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE}},
                        {Get(Buffer::MorphedPrimitivePrevPositions), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE}},
                    };

                    nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, bufferTransitions, helper::GetCountOf(bufferTransitions), nullptr, 0};
                    NRI.CmdBarrier(commandBuffer, transitionBarriers);
                }
            }
        }

        { // TLAS
            helper::Annotation annotation(NRI, commandBuffer, "TLAS");

            nri::BuildTopLevelAccelerationStructureDesc buildTopLevelAccelerationStructureDescs[2] = {};
            {
                buildTopLevelAccelerationStructureDescs[0].dst = Get(AccelerationStructure::TLAS_World);
                buildTopLevelAccelerationStructureDescs[0].instanceNum = (uint32_t)m_WorldTlasData.size();
                buildTopLevelAccelerationStructureDescs[0].instanceBuffer = m_WorldTlasDataLocation.buffer;
                buildTopLevelAccelerationStructureDescs[0].instanceOffset = m_WorldTlasDataLocation.offset;
                buildTopLevelAccelerationStructureDescs[0].scratchBuffer = Get(Buffer::WorldScratch);
                buildTopLevelAccelerationStructureDescs[0].scratchOffset = 0;

                buildTopLevelAccelerationStructureDescs[1].dst = Get(AccelerationStructure::TLAS_Emissive);
                buildTopLevelAccelerationStructureDescs[1].instanceNum = (uint32_t)m_LightTlasData.size();
                buildTopLevelAccelerationStructureDescs[1].instanceBuffer = m_LightTlasDataLocation.buffer;
                buildTopLevelAccelerationStructureDescs[1].instanceOffset = m_LightTlasDataLocation.offset;
                buildTopLevelAccelerationStructureDescs[1].scratchBuffer = Get(Buffer::LightScratch);
                buildTopLevelAccelerationStructureDescs[1].scratchOffset = 0;
            }

            NRI.CmdBuildTopLevelAccelerationStructures(commandBuffer, buildTopLevelAccelerationStructureDescs, helper::GetCountOf(buildTopLevelAccelerationStructureDescs));

            { // Transitions
                const nri::BufferBarrierDesc transition = {Get(Buffer::InstanceData), {nri::AccessBits::COPY_DESTINATION}, {nri::AccessBits::SHADER_RESOURCE}};

                nri::BarrierGroupDesc barrierGroupDesc = {};
                barrierGroupDesc.buffers = &transition;
                barrierGroupDesc.bufferNum = 1;

                NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
            }
        }

        // Must be bound here, after updating "Buffer::InstanceData"
        NRI.CmdSetDescriptorSet(commandBuffer, SET_RAY_TRACING, *Get(DescriptorSet::RayTracing2), nullptr);
        NRI.CmdSetDescriptorSet(commandBuffer, SET_SHARC, isEven ? *Get(DescriptorSet::SharcPing4) : *Get(DescriptorSet::SharcPong4), nullptr);

        //======================================================================================================================================
        // Render resolution
        //======================================================================================================================================

        // SHARC
        nri::Buffer* sharcBufferToClear = isEven ? Get(Buffer::SharcVoxelDataPong) : Get(Buffer::SharcVoxelDataPing);
        if (m_Settings.SHARC && NRD_MODE < OCCLUSION) {
            helper::Annotation sharc(NRI, commandBuffer, "Radiance cache");

            const nri::BufferBarrierDesc transitions[] = {
                {Get(Buffer::SharcHashEntries), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                {Get(Buffer::SharcHashCopyOffset), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                {Get(Buffer::SharcVoxelDataPing), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
                {Get(Buffer::SharcVoxelDataPong), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
            };

            nri::BarrierGroupDesc barrierGroupDesc = {};
            barrierGroupDesc.buffers = transitions;
            barrierGroupDesc.bufferNum = (uint16_t)helper::GetCountOf(transitions);

            { // Update
                helper::Annotation annotation(NRI, commandBuffer, "SHARC - Update");

                NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::SharcUpdate));

                uint32_t w = (m_RenderResolution.x / SHARC_DOWNSCALE + 15) / 16;
                uint32_t h = (m_RenderResolution.y / SHARC_DOWNSCALE + 15) / 16;

                NRI.CmdDispatch(commandBuffer, {w, h, 1});
            }

            { // Resolve
                helper::Annotation annotation(NRI, commandBuffer, "SHARC - Resolve");

                NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::SharcResolve));

                NRI.CmdDispatch(commandBuffer, {(SHARC_CAPACITY + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE, 1, 1});
            }

            { // Hash copy
                helper::Annotation annotation(NRI, commandBuffer, "SHARC - Hash copy");

                NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::SharcHashCopy));

                NRI.CmdDispatch(commandBuffer, {(SHARC_CAPACITY + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE, 1, 1});
            }
        }

        { // Trace opaque
            helper::Annotation annotation(NRI, commandBuffer, "Trace opaque");

            const TextureState transitions[] = {
                // Input
                {Texture::ComposedDiff, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                // Output
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::PsrThroughput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Unfiltered_Penumbra, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Unfiltered_Translucency, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Unfiltered_Spec, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
#if (NRD_MODE == SH)
                {Texture::Unfiltered_DiffSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Unfiltered_SpecSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
#endif
            };
            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::TraceOpaque));
            NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::TraceOpaque1), &dummyDynamicConstantOffset);

            uint32_t rectWmod = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
            uint32_t rectHmod = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
            uint32_t rectGridWmod = (rectWmod + 15) / 16;
            uint32_t rectGridHmod = (rectHmod + 15) / 16;

            NRI.CmdDispatch(commandBuffer, {rectGridWmod, rectGridHmod, 1});
        }

#if (NRD_MODE < OCCLUSION)
        { // Shadow denoising
            helper::Annotation annotation(NRI, commandBuffer, "Shadow denoising");

            float3 sunDir = GetSunDirection();

            m_SigmaSettings.lightDirection[0] = sunDir.x;
            m_SigmaSettings.lightDirection[1] = sunDir.y;
            m_SigmaSettings.lightDirection[2] = sunDir.z;

            nrd::Identifier denoiser = NRD_ID(SIGMA_SHADOW);

            m_NRD.SetDenoiserSettings(denoiser, &m_SigmaSettings);
            m_NRD.Denoise(&denoiser, 1, commandBuffer, userPool, NRD_RESTORE_INITIAL_STATE);
        }
#endif

        { // Opaque Denoising
            helper::Annotation annotation(NRI, commandBuffer, "Opaque denoising");

            if (m_Settings.denoiser == DENOISER_REBLUR || m_Settings.denoiser == DENOISER_REFERENCE) {
                nrd::HitDistanceParameters hitDistanceParameters = {};
                hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;
                m_ReblurSettings.hitDistanceParameters = hitDistanceParameters;

                nrd::ReblurSettings settings = m_ReblurSettings;
#if (NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION)
                // High quality SG resolve allows to use more relaxed normal weights
                if (m_Resolve)
                    settings.lobeAngleFraction *= 1.333f;
#endif

#if (NRD_MODE == OCCLUSION)
#    if (NRD_COMBINED == 1)
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SPECULAR_OCCLUSION)};
#    else
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_OCCLUSION), NRD_ID(REBLUR_SPECULAR_OCCLUSION)};
#    endif
#elif (NRD_MODE == SH)
#    if (NRD_COMBINED == 1)
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SPECULAR_SH)};
#    else
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SH), NRD_ID(REBLUR_SPECULAR_SH)};
#    endif
#elif (NRD_MODE == DIRECTIONAL_OCCLUSION)
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION)};
#else
#    if (NRD_COMBINED == 1)
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SPECULAR)};
#    else
                const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE), NRD_ID(REBLUR_SPECULAR)};
#    endif
#endif

                for (uint32_t i = 0; i < helper::GetCountOf(denoisers); i++)
                    m_NRD.SetDenoiserSettings(denoisers[i], &settings);

                m_NRD.Denoise(denoisers, helper::GetCountOf(denoisers), commandBuffer, userPool, NRD_RESTORE_INITIAL_STATE);
            } else if (m_Settings.denoiser == DENOISER_RELAX) {
                nrd::RelaxSettings settings = m_RelaxSettings;
#if (NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION)
                // High quality SG resolve allows to use more relaxed normal weights
                if (m_Resolve)
                    settings.lobeAngleFraction *= 1.333f;
#endif

#if (NRD_COMBINED == 1)
#    if (NRD_MODE == SH)
                const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE_SPECULAR_SH)};
#    else
                const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE_SPECULAR)};
#    endif
#else
#    if (NRD_MODE == SH)
                const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE_SH), NRD_ID(RELAX_SPECULAR_SH)};
#    else
                const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE), NRD_ID(RELAX_SPECULAR)};
#    endif
#endif

                for (uint32_t i = 0; i < helper::GetCountOf(denoisers); i++)
                    m_NRD.SetDenoiserSettings(denoisers[i], &settings);

                m_NRD.Denoise(denoisers, helper::GetCountOf(denoisers), commandBuffer, userPool, NRD_RESTORE_INITIAL_STATE);
            }
        }

        RestoreBindings(commandBuffer, isEven);

        { // Composition
            helper::Annotation annotation(NRI, commandBuffer, "Composition");

            const TextureState transitions[] = {
                // Input
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::PsrThroughput, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Shadow, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Diff, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Spec, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
#if (NRD_MODE == SH)
                {Texture::DiffSh, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::SpecSh, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
#endif
                // Output
                {Texture::ComposedDiff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
            };
            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Composition));
            NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::Composition1), &dummyDynamicConstantOffset);

            NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
        }

        { // Trace transparent
            helper::Annotation annotation(NRI, commandBuffer, "Trace transparent");

            const TextureState transitions[] = {
                // Input
                {Texture::ComposedDiff, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                // Output
                {Texture::Composed, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
            };

            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::TraceTransparent));
            NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::TraceTransparent1), &dummyDynamicConstantOffset);

            NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
        }

        if (m_Settings.denoiser == DENOISER_REFERENCE) { // Reference
            helper::Annotation annotation(NRI, commandBuffer, "Reference accumulation");

            nrd::CommonSettings modifiedCommonSettings = commonSettings;
            modifiedCommonSettings.splitScreen = m_Settings.separator;

            nrd::Identifier denoiser = NRD_ID(REFERENCE);

            m_NRD.SetCommonSettings(modifiedCommonSettings);
            m_NRD.SetDenoiserSettings(denoiser, &m_ReferenceSettings);
            m_NRD.Denoise(&denoiser, 1, commandBuffer, userPool, NRD_RESTORE_INITIAL_STATE);
        }

        RestoreBindings(commandBuffer, isEven);

        //======================================================================================================================================
        // Output resolution
        //======================================================================================================================================

        const Texture taaSrc = isEven ? Texture::TaaHistoryPrev : Texture::TaaHistory;
        const Texture taaDst = isEven ? Texture::TaaHistory : Texture::TaaHistoryPrev;

        if (IsDlssEnabled()) {
            // Before DLSS
            if (m_Settings.SR) {
                helper::Annotation annotation(NRI, commandBuffer, "Before DLSS");

                const TextureState transitions[] = {
                    // Input
                    {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::Unfiltered_Spec, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    // Output
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                    {Texture::RRGuide_DiffAlbedo, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                    {Texture::RRGuide_SpecAlbedo, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                    {Texture::RRGuide_SpecHitDistance, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                    {Texture::RRGuide_Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                };
                nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
                NRI.CmdBarrier(commandBuffer, transitionBarriers);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DlssBefore));
                NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::DlssBefore1), &dummyDynamicConstantOffset);

                NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
            }

            { // DLSS
                helper::Annotation annotation(NRI, commandBuffer, "DLSS");

                const TextureState transitions[] = {
                    // Input
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::RRGuide_DiffAlbedo, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::RRGuide_SpecAlbedo, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::RRGuide_SpecHitDistance, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::RRGuide_Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    {Texture::Composed, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                    // Output
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                };
                nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
                NRI.CmdBarrier(commandBuffer, transitionBarriers);

                bool resetHistory = m_ForceHistoryReset || m_Settings.SR != m_SettingsPrev.SR || m_Settings.RR != m_SettingsPrev.RR;

                nri::DispatchUpscaleDesc dispatchUpscaleDesc = {};
                dispatchUpscaleDesc.output = {Get(Texture::DlssOutput), Get(Descriptor::DlssOutput_StorageTexture)};
                dispatchUpscaleDesc.input = {Get(Texture::Composed), Get(Descriptor::Composed_Texture)};
                dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)rectW, (nri::Dim_t)rectH};
                dispatchUpscaleDesc.cameraJitter = {-m_Camera.state.viewportJitter.x, -m_Camera.state.viewportJitter.y};
                dispatchUpscaleDesc.mvScale = {1.0f, 1.0f};
                dispatchUpscaleDesc.flags = resetHistory ? nri::DispatchUpscaleBits::RESET_HISTORY : nri::DispatchUpscaleBits::NONE;

                if (m_Settings.RR) {
                    dispatchUpscaleDesc.guides.denoiser.mv = {Get(Texture::Mv), Get(Descriptor::Mv_Texture)};
                    dispatchUpscaleDesc.guides.denoiser.depth = {Get(Texture::ViewZ), Get(Descriptor::ViewZ_Texture)};
                    dispatchUpscaleDesc.guides.denoiser.diffuseAlbedo = {Get(Texture::RRGuide_DiffAlbedo), Get(Descriptor::RRGuide_DiffAlbedo_Texture)};
                    dispatchUpscaleDesc.guides.denoiser.specularAlbedo = {Get(Texture::RRGuide_SpecAlbedo), Get(Descriptor::RRGuide_SpecAlbedo_Texture)};
                    dispatchUpscaleDesc.guides.denoiser.normalRoughness = {Get(Texture::RRGuide_Normal_Roughness), Get(Descriptor::RRGuide_Normal_Roughness_Texture)};
                    dispatchUpscaleDesc.guides.denoiser.specularMvOrHitT = {Get(Texture::RRGuide_SpecHitDistance), Get(Descriptor::RRGuide_SpecHitDistance_Texture)};

                    memcpy(&dispatchUpscaleDesc.settings.dlrr.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
                    memcpy(&dispatchUpscaleDesc.settings.dlrr.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));

                    NRI.CmdDispatchUpscale(commandBuffer, *m_DLRR, dispatchUpscaleDesc);
                } else {
                    dispatchUpscaleDesc.guides.upscaler.mv = {Get(Texture::Mv), Get(Descriptor::Mv_Texture)};
                    dispatchUpscaleDesc.guides.upscaler.depth = {Get(Texture::ViewZ), Get(Descriptor::ViewZ_Texture)};

                    if (m_DLSR && upscalerType == nri::UpscalerType::FSR) // workaround for "conditional expression is constant"
                    {
                        dispatchUpscaleDesc.settings.fsr.zNear = 0.1f;
                        dispatchUpscaleDesc.settings.fsr.verticalFov = radians(m_Settings.camFov);
                        dispatchUpscaleDesc.settings.fsr.frameTime = m_Timer.GetSmoothedFrameTime();
                        dispatchUpscaleDesc.settings.fsr.viewSpaceToMetersFactor = 1.0f;
                        dispatchUpscaleDesc.settings.fsr.sharpness = 0.0f;
                    }

                    NRI.CmdDispatchUpscale(commandBuffer, *m_DLSR, dispatchUpscaleDesc);
                }
            }

            RestoreBindings(commandBuffer, isEven);

            { // After DLSS
                helper::Annotation annotation(NRI, commandBuffer, "After Dlss");

                const TextureState transitions[] = {
                    // Output
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
                };
                nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
                NRI.CmdBarrier(commandBuffer, transitionBarriers);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DlssAfter));
                NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::DlssAfter1), &dummyDynamicConstantOffset);

                NRI.CmdDispatch(commandBuffer, {outputGridW, outputGridH, 1});
            }
        } else { // TAA
            helper::Annotation annotation(NRI, commandBuffer, "TAA");

            const TextureState transitions[] = {
                // Input
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Composed, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {taaSrc, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                // Output
                {taaDst, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
            };
            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Taa));
            NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(isEven ? DescriptorSet::Taa1a : DescriptorSet::Taa1b), &dummyDynamicConstantOffset);

            NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
        }

        { // NIS
            helper::Annotation annotation(NRI, commandBuffer, "NIS");

            const TextureState transitions[] = {
                // Input
                {IsDlssEnabled() ? Texture::DlssOutput : taaDst, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                // Output
                {Texture::PreFinal, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
            };

            const nri::BufferBarrierDesc buffers[] = {
                {sharcBufferToClear, {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::COPY_DESTINATION}},
            };

            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, buffers, helper::GetCountOf(buffers), optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            nri::DispatchUpscaleDesc dispatchUpscaleDesc = {};
            dispatchUpscaleDesc.settings.nis.sharpness = NIS_SHARPNESS;
            dispatchUpscaleDesc.output = {Get(Texture::PreFinal), Get(Descriptor::PreFinal_StorageTexture)};

            if (IsDlssEnabled()) {
                dispatchUpscaleDesc.input = {Get(Texture::DlssOutput), Get(Descriptor::DlssOutput_Texture)};
                dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
            } else {
                dispatchUpscaleDesc.input = {Get(taaDst), isEven ? Get(Descriptor::TaaHistory_Texture) : Get(Descriptor::TaaHistoryPrev_Texture)};
                dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)rectW, (nri::Dim_t)rectH};
            }

            NRI.CmdDispatchUpscale(commandBuffer, *m_NIS, dispatchUpscaleDesc);

            RestoreBindings(commandBuffer, isEven, false);
        }

        // SHARC clear (for the next frame)
        if (m_Settings.SHARC && NRD_MODE < OCCLUSION) {
            helper::Annotation annotation(NRI, commandBuffer, "SHARC - Clear");

            NRI.CmdZeroBuffer(commandBuffer, *sharcBufferToClear, 0, SHARC_CAPACITY * sizeof(uint32_t) * 4);
        }

        //======================================================================================================================================
        // Window resolution
        //======================================================================================================================================

        { // Final
            helper::Annotation annotation(NRI, commandBuffer, "Final");

            const TextureState transitions[] = {
                // Input
                {Texture::PreFinal, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Composed, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                {Texture::Validation, nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE},
                // Output
                {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE},
            };
            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Final));
            NRI.CmdSetDescriptorSet(commandBuffer, SET_OTHER, *Get(DescriptorSet::Final1), &dummyDynamicConstantOffset);

            NRI.CmdDispatch(commandBuffer, {windowGridW, windowGridH, 1});
        }

        uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
        assert(backBufferIndex != nri::OUT_OF_DATE && "Oops, unhandled swap chain is out of date!");

        const BackBuffer& backBuffer = m_SwapChainBuffers[backBufferIndex];

        { // Copy to back-buffer
            helper::Annotation annotation(NRI, commandBuffer, "Copy to back buffer");

            const nri::TextureBarrierDesc transitions[] = {
                nri::TextureBarrierFromState(GetState(Texture::Final), {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE}),
                nri::TextureBarrierFromUnknown(backBuffer.texture, {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION}),
            };
            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, transitions, (uint16_t)helper::GetCountOf(transitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            NRI.CmdCopyTexture(commandBuffer, *backBuffer.texture, nullptr, *Get(Texture::Final), nullptr);
        }

        { // UI
            nri::TextureBarrierDesc before = {};
            before.texture = backBuffer.texture;
            before.before = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
            before.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};

            nri::BarrierGroupDesc transitionBarriers = {nullptr, 0, nullptr, 0, &before, 1};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            nri::AttachmentsDesc desc = {};
            desc.colors = &backBuffer.colorAttachment;
            desc.colorNum = 1;

            NRI.CmdBeginRendering(commandBuffer, desc);
            RenderUI(commandBuffer, *m_Streamer, backBuffer.attachmentFormat, m_SdrScale, m_IsSrgb);
            NRI.CmdEndRendering(commandBuffer);

            const nri::TextureBarrierDesc after = nri::TextureBarrierFromState(before, {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT, nri::StageBits::ALL});
            transitionBarriers = {nullptr, 0, nullptr, 0, &after, 1};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);
        }
    }
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::FenceSubmitDesc signalFence = {};
        signalFence.fence = m_FrameFence;
        signalFence.value = 1 + frameIndex;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = &signalFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    NRI.StreamerFinalize(*m_Streamer);

    nri::nriEndAnnotation();

    // Present
    nri::nriBeginAnnotation("Present", nri::BGRA_UNUSED);

    NRI.QueuePresent(*m_SwapChain);

    nri::nriEndAnnotation();

    // Cap FPS if requested
    nri::nriBeginAnnotation("FPS cap", nri::BGRA_UNUSED);

    float msLimit = m_Settings.limitFps ? 1000.0f / m_Settings.maxFps : 0.0f;
    double lastFrameTimeStamp = m_Timer.GetLastFrameTimeStamp();

    while (m_Timer.GetTimeStamp() - lastFrameTimeStamp < msLimit)
        ;

    nri::nriEndAnnotation();
}

SAMPLE_MAIN(Sample, 0);