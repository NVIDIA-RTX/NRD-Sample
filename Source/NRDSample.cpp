// © 2022 NVIDIA Corporation

#include "NRIFramework.h"

#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIWrapperVK.h"

// NRD and NRI-based integration
#include "NRD.h"
#include "NRDIntegration.hpp"

#ifdef _WIN32
#    undef APIENTRY
#    include <windows.h> // SetForegroundWindow, GetConsoleWindow
#endif

//=================================================================================
// Settings
//=================================================================================

// NRD mode and other shared settings are here
#include "../Shaders/Include/Shared.hlsli"

constexpr uint32_t MAX_ANIMATED_INSTANCE_NUM = 512;
constexpr auto BLAS_RIGID_MESH_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE | nri::AccelerationStructureBits::ALLOW_COMPACTION;
constexpr auto TLAS_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE;
constexpr float ACCUMULATION_TIME = 0.33f;     // seconds
constexpr float NEAR_Z = 0.001f;               // m
constexpr float GLASS_THICKNESS = 0.002f;      // m
constexpr float CAMERA_BACKWARD_OFFSET = 0.0f; // m, 3rd person camera offset
constexpr float NIS_SHARPNESS = 0.2f;
constexpr bool CAMERA_RELATIVE = true;
constexpr bool ALLOW_BLAS_MERGING = true;
constexpr bool ALLOW_HDR = NRIF_PLATFORM == NRIF_WINDOWS; // use "WIN + ALT + B" to switch HDR mode
constexpr bool USE_LOW_PRECISION_FP_FORMATS = true;       // saves a bit of memory and performance
constexpr uint8_t DLSS_PRESET = 0;
constexpr nri::UpscalerType upscalerType = nri::UpscalerType::DLSR;
constexpr int32_t MAX_HISTORY_FRAME_NUM = (int32_t)std::min(60u, std::min(nrd::REBLUR_MAX_HISTORY_FRAME_NUM, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
constexpr uint32_t TEXTURES_PER_MATERIAL = 4;
constexpr uint32_t MAX_TEXTURE_TRANSITIONS_NUM = 32;
constexpr uint32_t DYNAMIC_CONSTANT_BUFFER_SIZE = 1024 * 1024; // 1MB

#if (SIGMA_TRANSLUCENCY == 1)
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

//=================================================================================

// UI
#define UI_YELLOW            ImVec4(1.0f, 0.9f, 0.0f, 1.0f)
#define UI_GREEN             ImVec4(0.5f, 0.9f, 0.0f, 1.0f)
#define UI_RED               ImVec4(1.0f, 0.1f, 0.0f, 1.0f)
#define UI_HEADER            ImVec4(0.7f, 1.0f, 0.7f, 1.0f)
#define UI_HEADER_BACKGROUND ImVec4(0.7f * 0.3f, 1.0f * 0.3f, 0.7f * 0.3f, 1.0f)
#define UI_DEFAULT           ImGui::GetStyleColorVec4(ImGuiCol_Text)

enum class AccelerationStructure : uint32_t {
    TLAS_World,
    TLAS_Emissive,

    BLAS_MergedOpaque,
    BLAS_MergedTransparent,
    BLAS_MergedEmissive,
    BLAS_Other // many
};

enum class Buffer : uint32_t {
    InstanceData,
    PrimitiveData,
    SharcHashEntries,
    SharcAccumulated,
    SharcResolved,
    WorldScratch,
    LightScratch,

    MAX_NUM
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
    Gradient_StoredPing,
    Gradient_StoredPong,
    Gradient_Ping,
    Gradient_Pong,

    // History
    ComposedDiff,
    ComposedSpec_ViewZ,
    TaaHistoryPing,
    TaaHistoryPong,

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

    // RR guides
    RRGuide_DiffAlbedo,
    RRGuide_SpecAlbedo,
    RRGuide_SpecHitDistance,
    RRGuide_Normal_Roughness, // only RGBA16f encoding is supported

    // Read-only (must be last)
    BaseReadOnlyTexture,
};

enum class Descriptor : uint32_t {
    // Constant buffer
    Constant_Buffer,

    // Acceleration structures
    TLAS_World,
    TLAS_Emissive,

    // Buffers and textures
    BaseBuffer = TLAS_Emissive + 1,
    BaseTexture = BaseBuffer + (uint32_t)Buffer::MAX_NUM * 2,
    BaseReadOnlyTexture = BaseTexture + (uint32_t)Texture::BaseReadOnlyTexture * 2,
};

enum class Pipeline : uint32_t {
    SharcUpdate,
    SharcResolve,
    ConfidenceBlur,
    TraceOpaque,
    Composition,
    TraceTransparent,
    Taa,
    Final,
    DlssBefore,
    DlssAfter,

    MAX_NUM
};

enum class DescriptorSet : uint32_t {
    // SET_OTHER
    SharcUpdatePing,
    SharcUpdatePong,
    ConfidenceBlurPing,
    ConfidenceBlurPong,
    TraceOpaque,
    Composition,
    TraceTransparent,
    TaaPing,
    TaaPong,
    Final,
    DlssBefore,
    DlssAfter,

    // SET_RAY_TRACING
    RayTracing, // must be first after "SET_OTHER"

    // SET_SHARC
    Sharc,

    MAX_NUM
};

// NRD sample doesn't use several instances of the same denoiser in one NRD instance (like REBLUR_DIFFUSE x 3),
// thus we can use fields of "nrd::Denoiser" enum as unique identifiers
#define NRD_ID(x) nrd::Identifier(nrd::Denoiser::x)

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
    float emissionIntensityLights = 1.0f;
    float emissionIntensityCubes = 1.0f;
    float debug = 0.0f;
    float meterToUnitsMultiplier = 1.0f;
    float emulateMotionSpeed = 1.0f;
    float animatedObjectScale = 1.0f;
    float separator = 0.0f;
    float animationProgress = 0.0f;
    float animationSpeed = 0.0f;
    float hitDistScale = 3.0f;
    float resolutionScale = 1.0f;
    float sharpness = 0.15f;

    int32_t maxAccumulatedFrameNum = 31;
    int32_t maxFastAccumulatedFrameNum = 7;
    int32_t onScreen = 0;
    int32_t forcedMaterial = 0;
    int32_t animatedObjectNum = 5;
    uint32_t activeAnimation = 0;
    int32_t motionMode = 0;
    int32_t denoiser = DENOISER_REBLUR;
    int32_t rpp = 1;
    int32_t bounceNum = 1;
    int32_t tracingMode = 0;
    int32_t mvType = 0;

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
    bool emission = true;
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
    bool confidence = true;
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

    inline float4x4 Animate(float elapsedSeconds, float scale, float3& position) {
        float angle = progressedSec / durationSec;
        angle = Pi(angle * 2.0f - 1.0f);

        float3 localPosition;
        localPosition.x = cos(reverseDirection ? -angle : angle);
        localPosition.y = sin(reverseDirection ? -angle : angle);
        localPosition.z = localPosition.y;

        position = basePosition + localPosition * elipseAxis;

        float4x4 transform;
        transform.SetupByRotation(reverseRotation ? -angle : angle, rotationAxis);
        transform.AddScale(scale);

        progressedSec = fmod(progressedSec + elapsedSeconds, durationSec);

        return transform;
    }
};

static inline nri::TextureBarrierDesc TextureBarrierFromUnknown(nri::Texture* texture, nri::AccessLayoutStage after) {
    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
    textureBarrier.before.access = nri::AccessBits::NONE;
    textureBarrier.before.layout = nri::Layout::UNDEFINED;
    textureBarrier.before.stages = nri::StageBits::NONE;
    textureBarrier.after = after;

    return textureBarrier;
}

static inline nri::TextureBarrierDesc TextureBarrierFromState(nri::TextureBarrierDesc& prevState, nri::AccessLayoutStage after) {
    prevState.before = prevState.after;
    prevState.after = after;

    return prevState;
}

class Sample : public SampleBase {
public:
    inline Sample() {
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

    inline nri::DescriptorSet*& Get(DescriptorSet index) {
        return m_DescriptorSets[(uint32_t)index];
    }

    inline nri::AccelerationStructure*& Get(AccelerationStructure index) {
        return m_AccelerationStructures[(uint32_t)index];
    }

    inline nri::Descriptor*& GetDescriptor(Descriptor index) {
        return m_Descriptors[(uint32_t)index];
    }

    inline nri::Descriptor*& GetDescriptor(Buffer buffer) {
        return m_Descriptors[(uint32_t)Descriptor::BaseBuffer + (uint32_t)buffer * 2];
    }

    inline nri::Descriptor*& GetStorageDescriptor(Buffer buffer) {
        return m_Descriptors[(uint32_t)Descriptor::BaseBuffer + (uint32_t)buffer * 2 + 1];
    }

    inline nri::Descriptor*& GetDescriptor(Texture texture) {
        return m_Descriptors[(uint32_t)Descriptor::BaseTexture + (uint32_t)texture * 2];
    }

    inline nri::Descriptor*& GetStorageDescriptor(Texture texture) {
        return m_Descriptors[(uint32_t)Descriptor::BaseTexture + (uint32_t)texture * 2 + 1];
    }

    inline nri::Descriptor*& GetDescriptorForReadOnlyTexture(uint32_t index) {
        return m_Descriptors[(uint32_t)Descriptor::BaseReadOnlyTexture + index];
    }

    inline nrd::Resource GetNrdResource(Texture index) {
        nri::TextureBarrierDesc* textureState = &m_TextureStates[(uint32_t)index];

        nrd::Resource resource = {};
        resource.state = textureState->after;
        resource.userArg = textureState;
        resource.nri.texture = textureState->texture;

        return resource;
    }

    inline void Denoise(const nrd::Identifier* denoisers, uint32_t denoiserNum, nri::CommandBuffer& commandBuffer) {
        // Fill resource snapshot
        nrd::ResourceSnapshot resourceSnapshot = {};
        {
            resourceSnapshot.restoreInitialState = false;

            // Common
            resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, GetNrdResource(Texture::Mv));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, GetNrdResource(Texture::Normal_Roughness));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, GetNrdResource(Texture::ViewZ));

            // (Optional) Validation
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_VALIDATION, GetNrdResource(Texture::Validation));

            // Diffuse
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, GetNrdResource(Texture::Unfiltered_Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, GetNrdResource(Texture::Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_CONFIDENCE, GetNrdResource(Texture::Gradient_Pong));

            // Specular
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, GetNrdResource(Texture::Unfiltered_Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, GetNrdResource(Texture::Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_CONFIDENCE, GetNrdResource(Texture::Gradient_Pong));

#if (NRD_MODE == SH)
            // Diffuse SH
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_SH0, GetNrdResource(Texture::Unfiltered_Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_SH1, GetNrdResource(Texture::Unfiltered_DiffSh));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_SH0, GetNrdResource(Texture::Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_SH1, GetNrdResource(Texture::DiffSh));

            // Specular SH
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_SH0, GetNrdResource(Texture::Unfiltered_Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_SH1, GetNrdResource(Texture::Unfiltered_SpecSh));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_SH0, GetNrdResource(Texture::Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_SH1, GetNrdResource(Texture::SpecSh));
#endif

            // SIGMA
            resourceSnapshot.SetResource(nrd::ResourceType::IN_PENUMBRA, GetNrdResource(Texture::Unfiltered_Penumbra));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_TRANSLUCENCY, GetNrdResource(Texture::Unfiltered_Translucency));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, GetNrdResource(Texture::Shadow));

            // REFERENCE
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SIGNAL, GetNrdResource(Texture::Composed));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SIGNAL, GetNrdResource(Texture::Composed));
        }

        // Denoise
        m_NRD.Denoise(denoisers, denoiserNum, commandBuffer, resourceSnapshot);

        // Retrieve state
        if (!resourceSnapshot.restoreInitialState) {
            for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
                nri::TextureBarrierDesc* state = (nri::TextureBarrierDesc*)resourceSnapshot.unique[i].userArg;
                state->before = state->after;
                state->after = resourceSnapshot.unique[i].state;
            }
        }
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
        defaults.checkerboardMode = nrd::CheckerboardMode::OFF;
        defaults.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
        defaults.minMaterialForSpecular = MATERIAL_ID_METAL;
        defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        defaults.diffuseMaxAccumulatedFrameNum = m_RelaxSettings.diffuseMaxAccumulatedFrameNum;
        defaults.specularMaxAccumulatedFrameNum = m_RelaxSettings.specularMaxAccumulatedFrameNum;
        defaults.diffuseMaxFastAccumulatedFrameNum = m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum;
        defaults.specularMaxFastAccumulatedFrameNum = m_RelaxSettings.specularMaxFastAccumulatedFrameNum;
        defaults.fastHistoryClampingSigmaScale = 1.5f;

        // Helps to mitigate fireflies emphasized by DLSS
        // defaults.enableAntiFirefly = m_DlssQuality != -1 && IsDlssEnabled(); // TODO: currently doesn't help in this case, but makes the image darker

        return defaults;
    }

    inline nrd::ReblurSettings GetDefaultReblurSettings() const {
        nrd::ReblurSettings defaults = {};
        defaults.checkerboardMode = nrd::CheckerboardMode::OFF;
        defaults.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
        defaults.minMaterialForSpecular = MATERIAL_ID_METAL;
        defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        defaults.maxAccumulatedFrameNum = m_ReblurSettings.maxAccumulatedFrameNum;
        defaults.maxFastAccumulatedFrameNum = m_ReblurSettings.maxFastAccumulatedFrameNum;
        defaults.maxStabilizedFrameNum = m_ReblurSettings.maxStabilizedFrameNum;
        defaults.fastHistoryClampingSigmaScale = 1.5f;

        return defaults;
    }

    inline float3 GetSunDirection() const {
        float3 sunDirection;
        sunDirection.x = cos(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.y = sin(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.z = sin(radians(m_Settings.sunElevation));

        return sunDirection;
    }

    inline uint2 GetSharcDims() const {
        return 16 * uint2((m_RenderResolution / SHARC_DOWNSCALE + 15) / 16);
    }

    bool Initialize(nri::GraphicsAPI graphicsAPI, bool) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    void LoadScene();
    void AddInnerGlassSurfaces();
    void GenerateAnimatedCubes();
    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipelineLayoutAndDescriptorPool();
    void CreatePipelines(bool recreate);
    void CreateAccelerationStructures();
    void CreateResourcesAndDescriptors(nri::Format swapChainFormat);
    void CreateDescriptorSets();
    void CreateTexture(Texture texture, const char* debugName, nri::Format format, nri::Dim_t width, nri::Dim_t height, nri::Dim_t mipNum, nri::Dim_t arraySize, bool isReadOnly, nri::AccessBits initialAccess);
    void CreateBuffer(Buffer buffer, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage);
    void UploadStaticData();
    void UpdateConstantBuffer(uint32_t frameIndex, uint32_t maxAccumulatedFrameNum);
    void RestoreBindings(nri::CommandBuffer& commandBuffer);
    void GatherInstanceData();
    uint32_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM>& transitions);

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
    nri::Upscaler* m_DLSR = nullptr;
    nri::Upscaler* m_DLRR = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    std::array<nri::Upscaler*, 2> m_NIS = {};
    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureBarrierDesc> m_TextureStates;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_AccelerationStructures;
    std::vector<SwapChainTexture> m_SwapChainTextures;

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
    float4 m_HairBaseColor = float4(0.1f, 0.1f, 0.1f, 1.0f);
    float3 m_PrevLocalPos = {};
    float2 m_HairBetas = float2(0.25f, 0.3f);
    uint2 m_RenderResolution = {};
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
    bool m_IsSrgb = false;
    bool m_GlassObjects = false;
    bool m_IsReloadShadersSucceeded = true;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocator);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        for (uint32_t i = 0; i < m_Textures.size(); i++)
            NRI.DestroyTexture(m_Textures[i]);

        for (uint32_t i = 0; i < m_Buffers.size(); i++)
            NRI.DestroyBuffer(m_Buffers[i]);

        for (uint32_t i = 0; i < m_Descriptors.size(); i++)
            NRI.DestroyDescriptor(m_Descriptors[i]);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(m_Pipelines[i]);

        for (uint32_t i = 0; i < m_AccelerationStructures.size(); i++)
            NRI.DestroyAccelerationStructure(m_AccelerationStructures[i]);

        NRI.DestroyPipelineLayout(m_PipelineLayout);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
        NRI.DestroyFence(m_FrameFence);
    }

    if (NRI.HasUpscaler()) {
        NRI.DestroyUpscaler(m_NIS[0]);
        NRI.DestroyUpscaler(m_NIS[1]);
        NRI.DestroyUpscaler(m_DLSR);
        NRI.DestroyUpscaler(m_DLRR);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(m_Streamer);

    m_NRD.Destroy();

    DestroyImgui();

    nri::nriDestroyDevice(m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    Rng::Hash::Initialize(m_RngState, 106937, 69);

    // Adapters
    nri::AdapterDesc adapterDesc[4] = {};
    uint32_t adapterDescsNum = helper::GetCountOf(adapterDesc);
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(adapterDesc, adapterDescsNum));

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_ENABLE_COMMAND_BUFFER_EMULATION;
    deviceCreationDesc.disableD3D12EnhancedBarriers = D3D12_DISABLE_ENHANCED_BARRIERS;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &adapterDesc[std::min(m_AdapterIndex, adapterDescsNum - 1)];
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::UpscalerInterface), (nri::UpscalerInterface*)&NRI));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Create streamer
        nri::StreamerDesc streamerDesc = {};
        streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::DEVICE_UPLOAD;
        streamerDesc.constantBufferSize = DYNAMIC_CONSTANT_BUFFER_SIZE;
        streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::DEVICE_UPLOAD;
        streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT};
        streamerDesc.queuedFrameNum = GetQueuedFrameNum();
        NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));
    }

    { // Create upscaler: NIS
        nri::UpscalerDesc upscalerDesc = {};
        upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
        upscalerDesc.type = nri::UpscalerType::NIS;

        upscalerDesc.flags = nri::UpscalerBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_NIS[0]));

        upscalerDesc.flags = nri::UpscalerBits::HDR;
        NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_NIS[1]));
    }

    // Create upscalers: DLSR and DLRR
    m_RenderResolution = GetOutputResolution();

    if (m_DlssQuality != -1) {
        const nri::UpscalerBits upscalerFlags = nri::UpscalerBits::DEPTH_INFINITE | nri::UpscalerBits::HDR;

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
            upscalerDesc.preset = DLSS_PRESET;
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
        }
    }

    // Initialize NRD: REBLUR, RELAX and SIGMA in one instance
    {
        const nrd::DenoiserDesc denoisersDescs[] = {
        // REBLUR
#if (NRD_MODE == SH)
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR_SH), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH},
#else
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR},
#endif

        // RELAX
#if (NRD_MODE == SH)
            {NRD_ID(RELAX_DIFFUSE_SPECULAR_SH), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH},
#else
            {NRD_ID(RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR},
#endif

            // SIGMA
            {NRD_ID(SIGMA_SHADOW), SIGMA_VARIANT},

            // REFERENCE
            {NRD_ID(REFERENCE), nrd::Denoiser::REFERENCE},
        };

        nrd::InstanceCreationDesc instanceCreationDesc = {};
        instanceCreationDesc.denoisers = denoisersDescs;
        instanceCreationDesc.denoisersNum = helper::GetCountOf(denoisersDescs);

        nrd::IntegrationCreationDesc desc = {};
        strcpy(desc.name, "NRD");
        desc.queuedFrameNum = GetQueuedFrameNum();
        desc.enableWholeLifetimeDescriptorCaching = true;
        desc.demoteFloat32to16 = false;
        desc.resourceWidth = (uint16_t)m_RenderResolution.x;
        desc.resourceHeight = (uint16_t)m_RenderResolution.y;
        desc.autoWaitForIdle = false;

        nri::VideoMemoryInfo videoMemoryInfo1 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

        if (m_NRD.Recreate(desc, instanceCreationDesc, m_Device) != nrd::Result::SUCCESS)
            return false;

        nri::VideoMemoryInfo videoMemoryInfo2 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

        printf("NRD: allocated %.2f Mb for REBLUR, RELAX, SIGMA and REFERENCE denoisers\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));
    }

    LoadScene();

    if (m_SceneFile.find("BistroInterior") != std::string::npos)
        AddInnerGlassSurfaces();

    GenerateAnimatedCubes();

    m_Pipelines.resize((size_t)Pipeline::MAX_NUM);
    m_DescriptorSets.resize((size_t)DescriptorSet::MAX_NUM);
    m_Buffers.resize((size_t)Buffer::MAX_NUM);
    m_Textures.resize((size_t)Texture::BaseReadOnlyTexture + m_Scene.textures.size());
    m_TextureStates.resize((size_t)Texture::BaseReadOnlyTexture); // no read-only textures
    m_Descriptors.resize((size_t)Descriptor::BaseReadOnlyTexture + m_Scene.textures.size());
    m_AccelerationStructures.resize((size_t)AccelerationStructure::BLAS_Other);

    nri::Format swapChainFormat = CreateSwapChain();
    CreateCommandBuffers();
    CreatePipelineLayoutAndDescriptorPool();
    CreatePipelines(false);
    CreateAccelerationStructures();
    CreateResourcesAndDescriptors(swapChainFormat);
    CreateDescriptorSets();

    UploadStaticData();
    m_Scene.UnloadTextureData();
    m_Scene.UnloadGeometryData();

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);

    m_SettingsDefault = m_Settings;
    m_ShowValidationOverlay = m_DebugNRD;

    nri::VideoMemoryInfo videoMemoryInfo = {};
    NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo);
    printf("Allocated %.2f Mb\n", videoMemoryInfo.usageSize / (1024.0f * 1024.0f));

    return InitImgui(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    const QueuedFrame& queuedFrame = m_QueuedFrames[frameIndex % GetQueuedFrameNum()];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
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

    ImGui::NewFrame();
    if (!IsKeyPressed(Key::LAlt) && m_ShowUi) {
        static std::array<const char*, 2> nrdModes = {
            "NORMAL",
            "SH",
        };

        const nrd::LibraryDesc& nrdLibraryDesc = *nrd::GetLibraryDesc();

        char buf[256];
        snprintf(buf, sizeof(buf) - 1, "NRD v%u.%u.%u (%u.%u) - %s [Tab]", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild, (uint32_t)nrdLibraryDesc.normalEncoding, (uint32_t)nrdLibraryDesc.roughnessEncoding, nrdModes[NRD_MODE]);

        ImGui::SetNextWindowPos(ImVec2(m_Settings.windowAlignment ? 5.0f : GetOutputResolution().x - m_UiWidth - 5.0f, 5.0f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f));
        ImGui::Begin(buf, nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        {
            float avgFrameTime = m_Timer.GetVerySmoothedFrameTime();
            snprintf(buf, sizeof(buf), "%.1f FPS (%.2f ms) @ %up", 1000.0f / avgFrameTime, avgFrameTime, m_OutputResolution.y);

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

                    ImGui::Checkbox("Ortho", &m_Settings.ortho);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, (!m_Settings.cameraJitter && (m_Settings.TAA || IsDlssEnabled())) ? UI_RED : UI_DEFAULT);
                    ImGui::Checkbox("Jitter", &m_Settings.cameraJitter);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
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
                        ImGui::SliderFloat2("Emission intensity", &m_Settings.emissionIntensityLights, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::PopID();

                // "Hair" section
                if (m_SceneFile.find("Claire") != std::string::npos) {
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
                            ImGui::Combo("Animated scene", (int32_t*)&m_Settings.activeAnimation, items, helper::GetCountOf(m_Scene.animations));
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
                    const float3& sunDirection = GetSunDirection();

                    ImGui::SliderInt("Bounces", &m_Settings.bounceNum, 1, 8);
                    ImGui::SliderFloat("HitT scale (m)", &m_Settings.hitDistScale, 0.01f, sceneRadiusInMeters, "%.2f");

                    ImGui::Checkbox("Normal map", &m_Settings.normalMap);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, sunDirection.z > 0.0f ? UI_DEFAULT : (m_Settings.importanceSampling ? UI_GREEN : UI_YELLOW));
                    ImGui::Checkbox("IS", &m_Settings.importanceSampling);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::Checkbox("L1 (prev frame)", &m_Settings.usePrevFrame);
                }
                ImGui::PopID();

                // "NRD" section
                static const char* denoiser[] = {
#if (NRD_MODE == SH)
                    "REBLUR_SH",
                    "RELAX_SH",
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
                    ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.adaptiveAccumulation ? UI_GREEN : UI_YELLOW);
                    ImGui::Checkbox("Adaptive", &m_Settings.adaptiveAccumulation);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.confidence ? UI_GREEN : UI_YELLOW);
                    ImGui::Checkbox("Confidence", &m_Settings.confidence);
                    ImGui::PopStyleColor();

#if (NRD_MODE == SH)
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, m_Resolve ? UI_GREEN : UI_RED);
                    ImGui::Checkbox("Resolve", &m_Resolve);
                    ImGui::PopStyleColor();
#endif

                    if (m_DebugNRD) {
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, m_ShowValidationOverlay ? UI_YELLOW : UI_DEFAULT);
                        ImGui::Checkbox("Debug overlay", &m_ShowValidationOverlay);
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

                        bool isSame = !memcmp(&m_ReblurSettings, &defaults, sizeof(defaults));
                        bool hasSpatial = m_ReblurSettings.minBlurRadius + m_ReblurSettings.maxBlurRadius != 0.0f
                            || m_ReblurSettings.diffusePrepassBlurRadius != 0.0f
                            || m_ReblurSettings.specularPrepassBlurRadius != 0.0f;

                        ImGui::SameLine();
                        if (ImGui::Button(hasSpatial ? "No spatial" : "Spatial")) {
                            if (hasSpatial) {
                                m_ReblurSettings.minBlurRadius = 0.0f;
                                m_ReblurSettings.maxBlurRadius = 0.0f;
                            } else {
                                m_ReblurSettings.minBlurRadius = defaults.minBlurRadius;
                                m_ReblurSettings.maxBlurRadius = defaults.maxBlurRadius;
                            }
                        }

                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                        if (ImGui::Button("Defaults") || frameIndex == 0) {
                            m_ReblurSettings = defaults;
                            m_ReblurSettings.maxStabilizedFrameNum = m_Settings.maxAccumulatedFrameNum;
                        }
                        ImGui::PopStyleColor();

                        ImGui::Checkbox("Anti-firefly", &m_ReblurSettings.enableAntiFirefly);

                        ImGui::BeginDisabled(m_Settings.adaptiveAccumulation);
                        ImGui::SliderInt2("Accumulation (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d");
                        ImGui::SliderInt("Stabilization (frames)", (int32_t*)&m_ReblurSettings.maxStabilizedFrameNum, 0, m_Settings.maxAccumulatedFrameNum, "%d");
                        ImGui::EndDisabled();

                        ImGui::SliderFloat2("Pre-pass radius (px)", &m_ReblurSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                        ImGui::SliderFloat2("Blur radius (px)", &m_ReblurSettings.minBlurRadius, 0.0f, 60.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
                        ImGui::SliderFloat("Lobe fraction", &m_ReblurSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Roughness fraction", &m_ReblurSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Min hitT weight", &m_ReblurSettings.minHitDistanceWeight, 0.01f, 0.2f, "%.2f");
                        ImGui::SliderInt("History fix frames", (int32_t*)&m_ReblurSettings.historyFixFrameNum, 0, 5);
                        ImGui::SliderInt("History fix stride", (int32_t*)&m_ReblurSettings.historyFixBasePixelStride, 1, 20);
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.5f);
                        ImGui::SliderFloat("Responsive accumulation roughness threshold", &m_ReblurSettings.responsiveAccumulationSettings.roughnessThreshold, 0.0f, 1.0f, "%.2f");

                        if (m_ReblurSettings.maxAccumulatedFrameNum && m_ReblurSettings.maxStabilizedFrameNum) {
                            ImGui::Text("ANTI-LAG:");
                            ImGui::SliderFloat("Sigma scale", &m_ReblurSettings.antilagSettings.luminanceSigmaScale, 1.0f, 5.0f, "%.1f");
                            ImGui::SliderFloat("Sensitivity", &m_ReblurSettings.antilagSettings.luminanceSensitivity, 1.0f, 5.0f, "%.1f");
                        }
                    } else if (m_Settings.denoiser == DENOISER_RELAX) {
                        nrd::RelaxSettings defaults = GetDefaultRelaxSettings();

                        bool isSame = !memcmp(&m_RelaxSettings, &defaults, sizeof(defaults));
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
                                m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = 0;
                            } else {
                                m_RelaxSettings.diffusePhiLuminance = defaults.diffusePhiLuminance;
                                m_RelaxSettings.specularPhiLuminance = defaults.specularPhiLuminance;
                                m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = defaults.spatialVarianceEstimationHistoryThreshold;
                            }
                        }

                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                        if (ImGui::Button("Defaults") || frameIndex == 0)
                            m_RelaxSettings = defaults;
                        ImGui::PopStyleColor();

                        ImGui::Checkbox("Anti-firefly", &m_RelaxSettings.enableAntiFirefly);

                        ImGui::BeginDisabled(m_Settings.adaptiveAccumulation);
                        ImGui::SliderInt2("Accumulation (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d");
                        ImGui::EndDisabled();

                        ImGui::SliderFloat2("Pre-pass radius (px)", &m_RelaxSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");

                        ImGui::SliderInt("A-trous iterations", (int32_t*)&m_RelaxSettings.atrousIterationNum, 2, 8);
                        ImGui::SliderFloat2("Diff-Spec luma weight", &m_RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f, "%.1f");
                        ImGui::SliderFloat2("Min luma weight", &m_RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Depth threshold", &m_RelaxSettings.depthThreshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                        ImGui::SliderFloat("Lobe fraction", &m_RelaxSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Roughness fraction", &m_RelaxSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Min hitT weight", &m_RelaxSettings.minHitDistanceWeight, 0.01f, 0.2f, "%.2f");
                        ImGui::SliderFloat("Spec variance boost", &m_RelaxSettings.specularVarianceBoost, 0.0f, 8.0f, "%.2f");
                        ImGui::SliderFloat("Clamping sigma scale", &m_RelaxSettings.fastHistoryClampingSigmaScale, 0.0f, 3.0f, "%.1f");
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
                        std::string sampleShaders;

                        bool isTool = std::string(STRINGIFY(SHADERMAKE_PATH)) == "ShaderMake";
                        if (isTool) {
#ifdef _DEBUG
                            sampleShaders = "_Bin\\Debug\\ShaderMake.exe";
#else
                            sampleShaders = "_Bin\\Release\\ShaderMake.exe";
#endif
                        } else
                            sampleShaders = STRINGIFY(SHADERMAKE_PATH);

                        // clang-format off
                        sampleShaders +=
                            " --flatten --stripReflection --WX --colorize"
                            " --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128"
                            " --binary"
                            " --shaderModel 6_6"
                            " --sourceDir Shaders"
                            " --ignoreConfigDir"
                            " -c Shaders/Shaders.cfg"
                            " -o _Shaders"
                            " -I Shaders"
                            " -I External"
                            " -I " STRINGIFY(ML_SOURCE_DIR)
                            " -I " STRINGIFY(NRD_SOURCE_DIR)
                            " -I " STRINGIFY(NRI_SOURCE_DIR)
                            " -I " STRINGIFY(SHARC_SOURCE_DIR)
                            " -I " STRINGIFY(RTXCR_SOURCE_DIR)
                            " -D RTXCR_INTEGRATION=" STRINGIFY(RTXCR_INTEGRATION);
                        // clang-format on

                        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
                            sampleShaders += " -p DXIL --compiler \"" STRINGIFY(SHADERMAKE_DXC_PATH) "\"";
                        else
                            sampleShaders += " -p SPIRV --compiler \"" STRINGIFY(SHADERMAKE_DXC_VK_PATH) "\"";

                        printf("Compiling sample shaders...\n");
                        int32_t result = system(sampleShaders.c_str());
#ifdef _WIN32
                        if (result)
                            SetForegroundWindow(GetConsoleWindow());
#endif

                        m_IsReloadShadersSucceeded = !result;

                        if (!result)
                            CreatePipelines(true);

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
                    const float buttonWidth = 25.0f;

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
                                m_Settings.RR = false;
                                m_Settings.SR = m_DLSR;
                                m_Settings.TAA = true;
                                m_Settings.cameraJitter = true;

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
            m_UiWidth = ImGui::GetWindowWidth();
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();

    // Animate scene and update camera
    cBoxf cameraLimits = m_Scene.aabb;
    cameraLimits.Scale(4.0f);

    CameraDesc desc = {};
    desc.limits = cameraLimits;
    desc.aspectRatio = float(GetOutputResolution().x) / float(GetOutputResolution().y);
    desc.horizontalFov = degrees(atan(tan(radians(m_Settings.camFov) * 0.5f) * desc.aspectRatio * 9.0f / 16.0f) * 2.0f); // recalculate to ultra-wide if needed
    desc.nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    desc.farZ = 10000.0f * m_Settings.meterToUnitsMultiplier;
    desc.isCustomMatrixSet = false; // No camera animation hooked up
    desc.isPositiveZ = true;
    desc.isReversedZ = false;
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
            float f = fmod(Pi(period * 2.0f), Pi(2.0f));
            float3 axisX = m_Camera.state.mWorldToView.Row(0).xyz;
            float3 axisY = m_Camera.state.mWorldToView.Row(1).xyz;
            float2 v = Rotate(float2(1.0f, 0.0f), f);
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

        m_Settings.animatedObjectNum = 9;

        for (int32_t i = -1; i <= 1; i++) {
            for (int32_t j = -1; j <= 1; j++) {
                const uint32_t index = (i + 1) * 3 + (j + 1);

                float x = float(i) * scale * 4.0f;
                float y = float(j) * scale * 4.0f;
                float z = 10.0f * scale;

                float3 pos = basePos + vRight * x + vTop * y + vForward * z;

                utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[index].instanceID];
                instance.position = double3(pos);
                instance.rotation = m_Camera.state.mViewToWorld;
                instance.rotation.SetTranslation(float3::Zero());
                instance.rotation.AddScale(scale);
            }
        }
    } else if (m_Settings.animatedObjects) {
        for (int32_t i = 0; i < m_Settings.animatedObjectNum; i++) {
            float3 position;
            float4x4 transform = m_AnimatedInstances[i].Animate(animationDelta, scale, position);

            utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[i].instanceID];
            instance.rotation = transform;
            instance.position = double3(position);
        }
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

    // Global history reset: incompatible state changes
    if (m_SettingsPrev.denoiser != m_Settings.denoiser)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.ortho != m_Settings.ortho)
        m_ForceHistoryReset = true;
    if (m_SettingsPrev.RR != m_Settings.RR)
        m_ForceHistoryReset = true;
    if (frameIndex == 0)
        m_ForceHistoryReset = true;

    // Global history reset: emission intensity
    float a = float(m_Settings.emission) * max(m_Settings.emissionIntensityLights, m_Settings.emissionIntensityCubes);
    float b = float(m_SettingsPrev.emission) * max(m_SettingsPrev.emissionIntensityLights, m_SettingsPrev.emissionIntensityCubes);
    a = log2(1.0f + a);
    b = log2(1.0f + b);
    float d = abs(a - b) * 1000.0f / m_Timer.GetVerySmoothedFrameTime(); // make FPS-independent
    float resetHistoryFactor = 1.0f / (1.0f + 0.2f * d);

    if (m_ForceHistoryReset)
        resetHistoryFactor = 0.0f;

    // NRD common settings
    if (m_Settings.adaptiveAccumulation) {
        float fps = 1000.0f / m_Timer.GetVerySmoothedFrameTime();
        fps = min(fps, 121.0f);

        // REBLUR / RELAX
        int32_t maxAccumulatedFrameNum = max(nrd::GetMaxAccumulatedFrameNum(ACCUMULATION_TIME, fps), 1u);

        m_Settings.maxAccumulatedFrameNum = min(maxAccumulatedFrameNum, MAX_HISTORY_FRAME_NUM);
        m_Settings.maxFastAccumulatedFrameNum = m_Settings.maxAccumulatedFrameNum / 5;

        m_ReblurSettings.maxStabilizedFrameNum = m_Settings.maxAccumulatedFrameNum;

        // SIGMA
        uint32_t maxSigmaStabilizedFrames = nrd::GetMaxAccumulatedFrameNum(nrd::SIGMA_DEFAULT_ACCUMULATION_TIME, fps);

        m_SigmaSettings.maxStabilizedFrameNum = min(maxSigmaStabilizedFrames, nrd::SIGMA_MAX_HISTORY_FRAME_NUM);
    }

    uint32_t maxAccumulatedFrameNum = uint32_t(m_Settings.maxAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    uint32_t maxFastAccumulatedFrameNum = uint32_t(m_Settings.maxFastAccumulatedFrameNum * resetHistoryFactor + 0.5f);

    m_ReblurSettings.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_ReblurSettings.maxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;

    m_RelaxSettings.diffuseMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
    m_RelaxSettings.specularMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
    m_RelaxSettings.specularMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;

    UpdateConstantBuffer(frameIndex, maxAccumulatedFrameNum);
    GatherInstanceData();

    nri::nriEndAnnotation();
}

void Sample::LoadScene() {
    // Proxy geometry, which will be instancinated
    std::string sceneFile = utils::GetFullPath("Cubes/Cubes.gltf", utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING));

    m_ProxyInstancesNum = helper::GetCountOf(m_Scene.instances);

    // The scene
    if (m_SceneFile.find("Claire") != std::string::npos) {
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/Claire_PonyTail.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/Claire_HairMain_less_strands.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/Claire_BabyHairFront.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/Claire_BabyHairBack.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/ClaireCombined_No_Hair.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/brow/eyebrows.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/hairtie/hairtie.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/glass_lens/glass_lens.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/glass_frame/glass_frame.gltf", m_Scene, !ALLOW_BLAS_MERGING));
        NRI_ABORT_ON_FALSE(utils::LoadScene("_Data/Scenes/Claire/Claire/shirt/shirt.gltf", m_Scene, !ALLOW_BLAS_MERGING));
    } else {
        sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
        NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING));
    }

    // Some scene dependent settings
    m_ReblurSettings = GetDefaultReblurSettings();
    m_RelaxSettings = GetDefaultRelaxSettings();

    m_Settings.emission = true;
    if (m_SceneFile.find("BistroInterior") != std::string::npos) {
        m_Settings.exposure = 80.0f;
        m_Settings.animatedObjectScale = 0.5f;
        m_Settings.sunElevation = 7.0f;
    } else if (m_SceneFile.find("BistroExterior") != std::string::npos)
        m_Settings.exposure = 50.0f;
    else if (m_SceneFile.find("Hair") != std::string::npos) {
        m_Settings.exposure = 1.3f;
        m_Settings.bounceNum = 4;
    } else if (m_SceneFile.find("Claire") != std::string::npos) {
        m_Settings.exposure = 1.3f;
        m_Settings.bounceNum = 4;
        m_Settings.meterToUnitsMultiplier = 100.0f;
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
        float scale = 2.0f + (Rng::Hash::GetFloat(m_RngState) - 0.5f) * 2.0f;

        AnimatedInstance animatedInstance = {};
        animatedInstance.instanceID = helper::GetCountOf(m_Scene.instances);
        animatedInstance.basePosition = position;
        animatedInstance.durationSec = Rng::Hash::GetFloat(m_RngState) * 10.0f + 5.0f;
        animatedInstance.progressedSec = animatedInstance.durationSec * Rng::Hash::GetFloat(m_RngState);
        animatedInstance.rotationAxis = normalize(float3(Rng::Hash::GetFloat4(m_RngState).xyz) * 2.0f - 1.0f);
        animatedInstance.elipseAxis = (float3(Rng::Hash::GetFloat4(m_RngState).xyz) * 2.0f - 1.0f) * scale;
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
    swapChainDesc.flags = (m_Vsync ? nri::SwapChainBits::VSYNC : nri::SwapChainBits::NONE) | nri::SwapChainBits::ALLOW_TEARING;
    swapChainDesc.width = (uint16_t)GetOutputResolution().x;
    swapChainDesc.height = (uint16_t)GetOutputResolution().y;
    swapChainDesc.textureNum = GetOptimalSwapChainTextureNum();
    swapChainDesc.queuedFrameNum = GetQueuedFrameNum();

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));
    m_IsSrgb = swapChainDesc.format != nri::SwapChainFormat::BT709_G10_16BIT;

    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);

    const nri::TextureDesc& swapChainTextureDesc = NRI.GetTextureDesc(*swapChainTextures[0]);
    nri::Format swapChainFormat = swapChainTextureDesc.format;

    for (uint32_t i = 0; i < swapChainTextureNum; i++) {
        nri::TextureViewDesc textureViewDesc = {swapChainTextures[i], nri::TextureView::COLOR_ATTACHMENT, swapChainFormat};

        nri::Descriptor* colorAttachment = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateTextureView(textureViewDesc, colorAttachment));

        nri::Fence* acquireSemaphore = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, acquireSemaphore));

        nri::Fence* releaseSemaphore = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, releaseSemaphore));

        SwapChainTexture& swapChainTexture = m_SwapChainTextures.emplace_back();

        swapChainTexture = {};
        swapChainTexture.acquireSemaphore = acquireSemaphore;
        swapChainTexture.releaseSemaphore = releaseSemaphore;
        swapChainTexture.texture = swapChainTextures[i];
        swapChainTexture.colorAttachment = colorAttachment;
        swapChainTexture.attachmentFormat = swapChainFormat;

        char name[32];
        snprintf(name, sizeof(name), "Texture::SwapChain#%u", i);
        NRI.SetDebugName(swapChainTexture.texture, name);
    }

    return swapChainFormat;
}

void Sample::CreateCommandBuffers() {
    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }
}

void Sample::CreatePipelineLayoutAndDescriptorPool() {
    // SET_OTHER
    const nri::DescriptorRangeDesc otherRanges[] = {
        {0, 16, nri::DescriptorType::TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
        {0, 16, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND},
    };

    // SET_RAY_TRACING
    const uint32_t textureNum = helper::GetCountOf(m_Scene.materials) * TEXTURES_PER_MATERIAL;
    nri::DescriptorRangeDesc rayTracingRanges[] = {
        {0, textureNum, nri::DescriptorType::TEXTURE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND | nri::DescriptorRangeBits::VARIABLE_SIZED_ARRAY},
    };

    // SET_SHARC
    const nri::DescriptorRangeDesc sharcRanges[] = {
        {0, 3, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
    };

    // SET_ROOT
    nri::RootDescriptorDesc rootDescriptors[] = {
        {0, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::COMPUTE_SHADER},
        {0, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::StageBits::COMPUTE_SHADER},
        {1, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::StageBits::COMPUTE_SHADER},
        {2, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
        {3, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
    };

    nri::RootConstantDesc rootConstant = {1, sizeof(uint32_t), nri::StageBits::COMPUTE_SHADER};

    nri::SamplerDesc samplerLinearMipmapLinear = {};
    samplerLinearMipmapLinear.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
    samplerLinearMipmapLinear.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
    samplerLinearMipmapLinear.mipMax = 16.0f;

    nri::SamplerDesc samplerLinearMipmapNearest = {};
    samplerLinearMipmapNearest.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
    samplerLinearMipmapNearest.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::NEAREST};
    samplerLinearMipmapNearest.mipMax = 16.0f;

    nri::SamplerDesc samplerNearestMipmapNearest = {};
    samplerNearestMipmapNearest.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
    samplerNearestMipmapNearest.filters = {nri::Filter::NEAREST, nri::Filter::NEAREST, nri::Filter::NEAREST};
    samplerNearestMipmapNearest.mipMax = 16.0f;

    nri::SamplerDesc samplerLinearClamp = {};
    samplerLinearClamp.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
    samplerLinearClamp.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};

    nri::SamplerDesc samplerNearestClamp = {};
    samplerNearestClamp.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
    samplerNearestClamp.filters = {nri::Filter::NEAREST, nri::Filter::NEAREST, nri::Filter::NEAREST};

    nri::RootSamplerDesc rootSamplers[] = {
        {0, samplerLinearMipmapLinear, nri::StageBits::COMPUTE_SHADER},
        {1, samplerLinearMipmapNearest, nri::StageBits::COMPUTE_SHADER},
        {2, samplerNearestMipmapNearest, nri::StageBits::COMPUTE_SHADER},
        {3, samplerLinearClamp, nri::StageBits::COMPUTE_SHADER},
        {4, samplerNearestClamp, nri::StageBits::COMPUTE_SHADER},
    };

    const nri::DescriptorSetDesc descriptorSetDescs[] = {
        {SET_OTHER, otherRanges, helper::GetCountOf(otherRanges)},
        {SET_RAY_TRACING, rayTracingRanges, helper::GetCountOf(rayTracingRanges)},
        {SET_SHARC, sharcRanges, helper::GetCountOf(sharcRanges)},
    };

    { // Pipeline layout
        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.rootRegisterSpace = SET_ROOT;
        pipelineLayoutDesc.rootConstants = &rootConstant;
        pipelineLayoutDesc.rootConstantNum = 1;
        pipelineLayoutDesc.rootDescriptors = rootDescriptors;
        pipelineLayoutDesc.rootDescriptorNum = helper::GetCountOf(rootDescriptors);
        pipelineLayoutDesc.rootSamplers = rootSamplers;
        pipelineLayoutDesc.rootSamplerNum = helper::GetCountOf(rootSamplers);
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};

        uint32_t setNum = (uint32_t)DescriptorSet::RayTracing;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.textureMaxNum += otherRanges[0].descriptorNum * setNum;
        descriptorPoolDesc.storageTextureMaxNum += otherRanges[1].descriptorNum * setNum;

        setNum = 1;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.textureMaxNum += rayTracingRanges[0].descriptorNum * setNum;

        setNum = 1;
        descriptorPoolDesc.descriptorSetMaxNum += setNum;
        descriptorPoolDesc.storageStructuredBufferMaxNum += sharcRanges[0].descriptorNum * setNum;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
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

        uint16_t vertexStride = sizeof(float[3]);
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
        nri::AccelerationStructureDesc accelerationStructureDesc = {};
        accelerationStructureDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
        accelerationStructureDesc.flags = TLAS_BUILD_BITS;
        accelerationStructureDesc.geometryOrInstanceNum = helper::GetCountOf(m_Scene.instances);

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedAccelerationStructure(*m_Device, NriDeviceHeap, accelerationStructureDesc, Get(AccelerationStructure::TLAS_World)));
    }

    { // AccelerationStructure::TLAS_Emissive
        nri::AccelerationStructureDesc accelerationStructureDesc = {};
        accelerationStructureDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
        accelerationStructureDesc.flags = TLAS_BUILD_BITS;
        accelerationStructureDesc.geometryOrInstanceNum = helper::GetCountOf(m_Scene.instances);

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedAccelerationStructure(*m_Device, NriDeviceHeap, accelerationStructureDesc, Get(AccelerationStructure::TLAS_Emissive)));
    }

    // Create temp buffer for indices, vertices and transforms in UPLOAD heap
    nri::Buffer* uploadBuffer = nullptr;
    {
        nri::BufferDesc bufferDesc = {uploadSize, 0, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT};

        NRI_ABORT_ON_FAILURE(NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::HOST_UPLOAD, 0.0f, bufferDesc, uploadBuffer));
    }

    // Create BOTTOM_LEVEL acceleration structures
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    uint64_t scratchSize = 0;

    uint8_t* uploadData = (uint8_t*)NRI.MapBuffer(*uploadBuffer, 0, nri::WHOLE_SIZE);
    assert(uploadData);

    uint64_t primitivesNum = 0;
    std::vector<nri::BuildBottomLevelAccelerationStructureDesc> buildBottomLevelAccelerationStructureDescs;

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
            uint16_t vertexStride = sizeof(float[3]);
            uint64_t vertexDataSize = mesh.vertexNum * vertexStride;
            uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);

            uint8_t* p = uploadData + geometryOffset;
            for (uint32_t v = 0; v < mesh.vertexNum; v++) {
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
            bottomLevelGeometry.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
            bottomLevelGeometry.triangles.indexBuffer = uploadBuffer;
            bottomLevelGeometry.triangles.indexOffset = geometryOffset + vertexDataSize;
            bottomLevelGeometry.triangles.indexNum = mesh.indexNum;
            bottomLevelGeometry.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

            if (mode != 3) {
                bottomLevelGeometry.triangles.transformBuffer = uploadBuffer;
                bottomLevelGeometry.triangles.transformOffset = transformOffset;
            } else {
                // Create BLAS
                nri::AccelerationStructureDesc accelerationStructureDesc = {};
                accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                accelerationStructureDesc.flags = BLAS_RIGID_MESH_BUILD_BITS;
                accelerationStructureDesc.geometryOrInstanceNum = 1;
                accelerationStructureDesc.geometries = &bottomLevelGeometry;

                nri::AccelerationStructure* accelerationStructure = nullptr;
                NRI_ABORT_ON_FAILURE(NRI.CreateCommittedAccelerationStructure(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, accelerationStructureDesc, accelerationStructure));
                m_AccelerationStructures.push_back(accelerationStructure);

                // Save build parameters
                nri::BuildBottomLevelAccelerationStructureDesc& buildBottomLevelAccelerationStructureDesc = buildBottomLevelAccelerationStructureDescs.emplace_back();
                buildBottomLevelAccelerationStructureDesc = {};
                buildBottomLevelAccelerationStructureDesc.dst = accelerationStructure;
                buildBottomLevelAccelerationStructureDesc.geometryNum = 1;
                buildBottomLevelAccelerationStructureDesc.geometries = &geometries[geometries.size() - 1];
                buildBottomLevelAccelerationStructureDesc.scratchBuffer = nullptr;
                buildBottomLevelAccelerationStructureDesc.scratchOffset = scratchSize;

                // Update scratch
                uint64_t buildSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
                scratchSize += helper::Align(buildSize, deviceDesc.memoryAlignment.scratchBufferOffset);
            }

            // Update geometry offset
            geometryOffset += vertexDataSize + helper::Align(indexDataSize, 4);
            primitivesNum += mesh.indexNum / 3;
        }

        if (mode != 3) {
            uint32_t geometryObjectsNum = (uint32_t)(geometries.size() - geometryObjectBase);
            if (geometryObjectsNum) {
                // Create BLAS
                nri::AccelerationStructureDesc accelerationStructureDesc = {};
                accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                accelerationStructureDesc.flags = BLAS_RIGID_MESH_BUILD_BITS;
                accelerationStructureDesc.geometryOrInstanceNum = geometryObjectsNum;
                accelerationStructureDesc.geometries = &geometries[geometryObjectBase];

                nri::AccelerationStructure*& accelerationStructure = Get((AccelerationStructure)((uint32_t)AccelerationStructure::BLAS_MergedOpaque + mode));
                NRI_ABORT_ON_FAILURE(NRI.CreateCommittedAccelerationStructure(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, accelerationStructureDesc, accelerationStructure));

                // Save build parameters
                nri::BuildBottomLevelAccelerationStructureDesc& buildBottomLevelAccelerationStructureDesc = buildBottomLevelAccelerationStructureDescs.emplace_back();
                buildBottomLevelAccelerationStructureDesc = {};
                buildBottomLevelAccelerationStructureDesc.dst = accelerationStructure;
                buildBottomLevelAccelerationStructureDesc.geometryNum = geometryObjectsNum;
                buildBottomLevelAccelerationStructureDesc.geometries = &geometries[geometryObjectBase];
                buildBottomLevelAccelerationStructureDesc.scratchBuffer = nullptr;
                buildBottomLevelAccelerationStructureDesc.scratchOffset = scratchSize;

                // Update scratch
                uint64_t size = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
                scratchSize += helper::Align(size, deviceDesc.memoryAlignment.scratchBufferOffset);
            }
        }
    }

    // Create temp resources
    uint32_t blasNum = (uint32_t)buildBottomLevelAccelerationStructureDescs.size();

    nri::Buffer* scratchBuffer = nullptr;
    {
        nri::BufferDesc bufferDesc = {scratchSize, 0, nri::BufferUsageBits::SCRATCH_BUFFER};

        NRI_ABORT_ON_FAILURE(NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, bufferDesc, scratchBuffer));
    }

    nri::Buffer* readbackBuffer = nullptr;
    {
        nri::BufferDesc bufferDesc = {blasNum * sizeof(uint64_t), 0, nri::BufferUsageBits::NONE};

        NRI_ABORT_ON_FAILURE(NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::HOST_READBACK, 0.0f, bufferDesc, readbackBuffer));
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

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.bufferNum = (uint32_t)bufferBarriers.size();
            barrierDesc.buffers = bufferBarriers.data();

            NRI.CmdBarrier(*commandBuffer, barrierDesc);

            // Build everything in one go
            NRI.CmdBuildBottomLevelAccelerationStructures(*commandBuffer, buildBottomLevelAccelerationStructureDescs.data(), (uint32_t)buildBottomLevelAccelerationStructureDescs.size());

            // Barriers (read)
            for (nri::BufferBarrierDesc& bufferBarrier : bufferBarriers) {
                bufferBarrier.before = bufferBarrier.after;
                bufferBarrier.after = {nri::AccessBits::ACCELERATION_STRUCTURE_READ, nri::StageBits::ACCELERATION_STRUCTURE};
            }

            NRI.CmdBarrier(*commandBuffer, barrierDesc);

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
        NRI.QueueWaitIdle(m_GraphicsQueue);
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

                nri::AccelerationStructureDesc accelerationStructureDesc = {};
                accelerationStructureDesc.optimizedSize = sizes[i];
                accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
                accelerationStructureDesc.flags = BLAS_RIGID_MESH_BUILD_BITS;
                accelerationStructureDesc.geometryOrInstanceNum = blasBuildDesc.geometryNum;
                accelerationStructureDesc.geometries = blasBuildDesc.geometries;

                nri::AccelerationStructure* compactedBlas = nullptr;
                NRI_ABORT_ON_FAILURE(NRI.CreatePlacedAccelerationStructure(*m_Device, NriDeviceHeap, accelerationStructureDesc, compactedBlas));
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
        NRI.QueueWaitIdle(m_GraphicsQueue);
    }

    double buildTime = m_Timer.GetTimeStamp() - stamp2;

    // Cleanup
    for (uint32_t i = 0; i < blasNum; i++) {
        const nri::BuildBottomLevelAccelerationStructureDesc& blasBuildDesc = buildBottomLevelAccelerationStructureDescs[i];

        nri::AccelerationStructure* tempBlas = blasBuildDesc.dst;
        NRI.DestroyAccelerationStructure(tempBlas);

        nri::AccelerationStructure* compactedBlas = compactedBlases[i];
        std::replace(m_AccelerationStructures.begin(), m_AccelerationStructures.end(), tempBlas, compactedBlas);
    }

    NRI.UnmapBuffer(*uploadBuffer);
    NRI.UnmapBuffer(*readbackBuffer);

    NRI.DestroyQueryPool(queryPool);

    NRI.DestroyBuffer(readbackBuffer);
    NRI.DestroyBuffer(scratchBuffer);
    NRI.DestroyBuffer(uploadBuffer);

    NRI.DestroyCommandBuffer(commandBuffer);
    NRI.DestroyCommandAllocator(commandAllocator);

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

void Sample::CreatePipelines(bool recreate) {
    if (recreate) {
        NRI.DeviceWaitIdle(m_Device);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(m_Pipelines[i]);

        m_NRD.RecreatePipelines();
    }

    utils::ShaderCodeStorage shaderCodeStorage;

    nri::ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = m_PipelineLayout;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "SharcUpdate.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::SharcUpdate)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "SharcResolve.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::SharcResolve)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "ConfidenceBlur.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::ConfidenceBlur)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "TraceOpaque.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::TraceOpaque)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Composition.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::Composition)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "TraceTransparent.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::TraceTransparent)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Taa.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::Taa)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Final.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::Final)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "DlssBefore.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::DlssBefore)));

    pipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "DlssAfter.cs", shaderCodeStorage);
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, Get(Pipeline::DlssAfter)));
}

void Sample::CreateResourcesAndDescriptors(nri::Format swapChainFormat) {
    const nrd::LibraryDesc& nrdLibraryDesc = *nrd::GetLibraryDesc();
    nri::Format normalFormat = nri::Format::RGBA16_SFLOAT; // TODO: RGBA16_SNORM can't be used, because NGX doesn't support it
    switch (nrdLibraryDesc.normalEncoding) {
        case nrd::NormalEncoding::RGBA8_UNORM:
            normalFormat = nri::Format::RGBA8_UNORM;
            break;
        case nrd::NormalEncoding::RGBA8_SNORM:
            normalFormat = nri::Format::RGBA8_SNORM;
            break;
        case nrd::NormalEncoding::R10_G10_B10_A2_UNORM:
            normalFormat = nri::Format::R10_G10_B10_A2_UNORM;
            break;
        case nrd::NormalEncoding::RGBA16_UNORM:
            normalFormat = nri::Format::RGBA16_UNORM;
            break;
        default:
            break;
    }

    constexpr nri::Format dataFormat = nri::Format::RGBA16_SFLOAT;
    constexpr nri::Format taaFormat = nri::Format::RGBA16_SFLOAT; // required for new TAA even in LDR mode (RGBA16_UNORM can't be used)
    constexpr nri::Format colorFormat = USE_LOW_PRECISION_FP_FORMATS ? nri::Format::R11_G11_B10_UFLOAT : nri::Format::RGBA16_SFLOAT;
    constexpr nri::Format criticalColorFormat = nri::Format::RGBA16_SFLOAT; // TODO: R9_G9_B9_E5_UFLOAT?
    constexpr nri::Format shadowFormat = SIGMA_TRANSLUCENCY ? nri::Format::RGBA8_UNORM : nri::Format::R8_UNORM;

    nri::Dim_t w = (nri::Dim_t)m_RenderResolution.x;
    nri::Dim_t h = (nri::Dim_t)m_RenderResolution.y;
    nri::Dim_t rrw = m_DlssQuality == -1 ? 1 : w;
    nri::Dim_t rrh = m_DlssQuality == -1 ? 1 : h;

    uint64_t instanceNum = m_Scene.instances.size() + MAX_ANIMATED_INSTANCE_NUM;
    uint64_t instanceDataSize = instanceNum * sizeof(InstanceData);
    uint64_t worldScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_World));
    uint64_t lightScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_Emissive));

    m_InstanceData.resize(instanceNum);
    m_WorldTlasData.resize(instanceNum);
    m_LightTlasData.resize(instanceNum);

    // Buffers
    CreateBuffer(Buffer::InstanceData, "InstanceData", instanceDataSize / sizeof(InstanceData), sizeof(InstanceData), nri::BufferUsageBits::SHADER_RESOURCE);
    CreateBuffer(Buffer::PrimitiveData, "PrimitiveData", m_Scene.totalInstancedPrimitivesNum, sizeof(PrimitiveData), nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(Buffer::SharcHashEntries, "SharcHashEntries", SHARC_CAPACITY, sizeof(uint64_t), nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(Buffer::SharcAccumulated, "SharcAccumulated", SHARC_CAPACITY, sizeof(uint32_t) * 4, nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(Buffer::SharcResolved, "SharcResolved", SHARC_CAPACITY, sizeof(uint32_t) * 4, nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(Buffer::WorldScratch, "WorldScratch", worldScratchBufferSize, 1, nri::BufferUsageBits::SCRATCH_BUFFER);
    CreateBuffer(Buffer::LightScratch, "LightScratch", lightScratchBufferSize, 1, nri::BufferUsageBits::SCRATCH_BUFFER);

    // Textures
    CreateTexture(Texture::ViewZ, "ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Mv, "Mv", nri::Format::RGBA16_SFLOAT, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Normal_Roughness, "Normal_Roughness", normalFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::PsrThroughput, "PsrThroughput", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::BaseColor_Metalness, "BaseColor_Metalness", nri::Format::RGBA8_SRGB, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::DirectLighting, "DirectLighting", colorFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::DirectEmission, "DirectEmission", colorFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Shadow, "Shadow", shadowFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Diff, "Diff", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Spec, "Spec", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Unfiltered_Penumbra, "Unfiltered_Penumbra", nri::Format::R16_SFLOAT, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Unfiltered_Diff, "Unfiltered_Diff", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Unfiltered_Spec, "Unfiltered_Spec", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Unfiltered_Translucency, "Unfiltered_Translucency", shadowFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Validation, "Validation", nri::Format::RGBA8_UNORM, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Composed, "Composed", criticalColorFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Gradient_StoredPing, "Gradient_StoredPing", nri::Format::RGBA16_SFLOAT, (nri::Dim_t)GetSharcDims().x, (nri::Dim_t)GetSharcDims().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Gradient_StoredPong, "Gradient_StoredPong", nri::Format::RGBA16_SFLOAT, (nri::Dim_t)GetSharcDims().x, (nri::Dim_t)GetSharcDims().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Gradient_Ping, "Gradient_Ping", nri::Format::RGBA16_SFLOAT, (nri::Dim_t)GetSharcDims().x, (nri::Dim_t)GetSharcDims().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Gradient_Pong, "Gradient_Pong", nri::Format::RGBA16_SFLOAT, (nri::Dim_t)GetSharcDims().x, (nri::Dim_t)GetSharcDims().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::ComposedDiff, "ComposedDiff", colorFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(Texture::ComposedSpec_ViewZ, "ComposedSpec_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(Texture::TaaHistoryPing, "TaaHistoryPing", taaFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::TaaHistoryPong, "TaaHistoryPong", taaFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::DlssOutput, "DlssOutput", criticalColorFormat, (nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::PreFinal, "PreFinal", criticalColorFormat, (nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Final, "Final", swapChainFormat, (nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y, 1, 1, false, nri::AccessBits::COPY_SOURCE);
#if (NRD_MODE == SH)
    CreateTexture(Texture::Unfiltered_DiffSh, "Unfiltered_DiffSh", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::Unfiltered_SpecSh, "Unfiltered_SpecSh", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::DiffSh, "DiffSh", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(Texture::SpecSh, "SpecSh", dataFormat, w, h, 1, 1, false, nri::AccessBits::SHADER_RESOURCE);
#endif
    CreateTexture(Texture::RRGuide_DiffAlbedo, "RRGuide_DiffAlbedo", nri::Format::R10_G10_B10_A2_UNORM, rrw, rrh, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(Texture::RRGuide_SpecAlbedo, "RRGuide_SpecAlbedo", nri::Format::R10_G10_B10_A2_UNORM, rrw, rrh, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(Texture::RRGuide_SpecHitDistance, "RRGuide_SpecHitDistance", nri::Format::R16_SFLOAT, rrw, rrh, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(Texture::RRGuide_Normal_Roughness, "RRGuide_Normal_Roughness", nri::Format::RGBA16_SFLOAT, rrw, rrh, 1, 1, false, nri::AccessBits::SHADER_RESOURCE_STORAGE);

    for (size_t i = 0; i < m_Scene.textures.size(); i++) {
        const utils::Texture* texture = m_Scene.textures[i];
        CreateTexture((Texture)((size_t)Texture::BaseReadOnlyTexture + i), "", texture->GetFormat(), texture->GetWidth(), texture->GetHeight(), texture->GetMipNum(), texture->GetArraySize(), true, nri::AccessBits::NONE);
    }

    { // Descriptor::Constant_Buffer
        const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

        size_t maxSize = sizeof(GlobalConstants);

        nri::BufferViewDesc bufferViewDesc = {};
        bufferViewDesc.type = nri::BufferView::CONSTANT_BUFFER;
        bufferViewDesc.buffer = NRI.GetStreamerConstantBuffer(*m_Streamer);
        bufferViewDesc.size = helper::Align((uint32_t)maxSize, deviceDesc.memoryAlignment.constantBufferOffset);

        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, GetDescriptor(Descriptor::Constant_Buffer)));
    }

    // Descriptor::TLAS_World
    NRI.CreateAccelerationStructureDescriptor(*Get(AccelerationStructure::TLAS_World), GetDescriptor(Descriptor::TLAS_World));

    // Descriptor::TLAS_Emissive
    NRI.CreateAccelerationStructureDescriptor(*Get(AccelerationStructure::TLAS_Emissive), GetDescriptor(Descriptor::TLAS_Emissive));
}

void Sample::CreateDescriptorSets() {
    // Ping
    const nri::Descriptor* SharcUpdatePing_Textures[] = {
        GetDescriptor(Texture::Gradient_StoredPing),
    };

    const nri::Descriptor* SharcUpdatePing_StorageTextures[] = {
        GetStorageDescriptor(Texture::Gradient_StoredPong),
        GetStorageDescriptor(Texture::Gradient_Ping),
    };

    const nri::Descriptor* ConfidenceBlurPing_Textures[] = {
        GetDescriptor(Texture::Gradient_Ping),
    };

    const nri::Descriptor* ConfidenceBlurPing_StorageTextures[] = {
        GetStorageDescriptor(Texture::Gradient_Pong),
    };

    const nri::Descriptor* TaaPing_Textures[] = {
        GetDescriptor(Texture::Mv),
        GetDescriptor(Texture::Composed),
        GetDescriptor(Texture::TaaHistoryPong),
    };

    const nri::Descriptor* TaaPing_StorageTextures[] = {
        GetStorageDescriptor(Texture::TaaHistoryPing),
    };

    // Pong
    const nri::Descriptor* SharcUpdatePong_Textures[] = {
        GetDescriptor(Texture::Gradient_StoredPong),
    };

    const nri::Descriptor* SharcUpdatePong_StorageTextures[] = {
        GetStorageDescriptor(Texture::Gradient_StoredPing),
        GetStorageDescriptor(Texture::Gradient_Ping),
    };

    const nri::Descriptor* ConfidenceBlurPong_Textures[] = {
        GetDescriptor(Texture::Gradient_Pong),
    };

    const nri::Descriptor* ConfidenceBlurPong_StorageTextures[] = {
        GetStorageDescriptor(Texture::Gradient_Ping),
    };

    const nri::Descriptor* TaaPong_Textures[] = {
        GetDescriptor(Texture::Mv),
        GetDescriptor(Texture::Composed),
        GetDescriptor(Texture::TaaHistoryPing),
    };

    const nri::Descriptor* TaaPong_StorageTextures[] = {
        GetStorageDescriptor(Texture::TaaHistoryPong),
    };

    // Other
    const nri::Descriptor* TraceOpaque_Textures[] = {
        GetDescriptor(Texture::ComposedDiff),
        GetDescriptor(Texture::ComposedSpec_ViewZ),
        GetDescriptorForReadOnlyTexture(utils::StaticTexture::ScramblingRanking),
        GetDescriptorForReadOnlyTexture(utils::StaticTexture::SobolSequence),
    };

    const nri::Descriptor* TraceOpaque_StorageTextures[] = {
        GetStorageDescriptor(Texture::Mv),
        GetStorageDescriptor(Texture::ViewZ),
        GetStorageDescriptor(Texture::Normal_Roughness),
        GetStorageDescriptor(Texture::BaseColor_Metalness),
        GetStorageDescriptor(Texture::DirectLighting),
        GetStorageDescriptor(Texture::DirectEmission),
        GetStorageDescriptor(Texture::PsrThroughput),
        GetStorageDescriptor(Texture::Unfiltered_Penumbra),
        GetStorageDescriptor(Texture::Unfiltered_Translucency),
        GetStorageDescriptor(Texture::Unfiltered_Diff),
        GetStorageDescriptor(Texture::Unfiltered_Spec),
#if (NRD_MODE == SH)
        GetStorageDescriptor(Texture::Unfiltered_DiffSh),
        GetStorageDescriptor(Texture::Unfiltered_SpecSh),
#endif
    };

    const nri::Descriptor* Composition_Textures[] = {
        GetDescriptor(Texture::ViewZ),
        GetDescriptor(Texture::Normal_Roughness),
        GetDescriptor(Texture::BaseColor_Metalness),
        GetDescriptor(Texture::DirectLighting),
        GetDescriptor(Texture::DirectEmission),
        GetDescriptor(Texture::PsrThroughput),
        GetDescriptor(Texture::Shadow),
        GetDescriptor(Texture::Diff),
        GetDescriptor(Texture::Spec),
#if (NRD_MODE == SH)
        GetDescriptor(Texture::DiffSh),
        GetDescriptor(Texture::SpecSh),
#endif
    };

    const nri::Descriptor* Composition_StorageTextures[] = {
        GetStorageDescriptor(Texture::ComposedDiff),
        GetStorageDescriptor(Texture::ComposedSpec_ViewZ),
    };

    const nri::Descriptor* TraceTransparent_Textures[] = {
        GetDescriptor(Texture::ComposedDiff),
        GetDescriptor(Texture::ComposedSpec_ViewZ),
    };

    const nri::Descriptor* TraceTransparent_StorageTextures[] = {
        GetStorageDescriptor(Texture::Composed),
        GetStorageDescriptor(Texture::Mv),
        GetStorageDescriptor(Texture::Normal_Roughness),
    };

    const nri::Descriptor* Final_Textures[] = {
        GetDescriptor(Texture::PreFinal),
        GetDescriptor(Texture::Composed),
        GetDescriptor(Texture::Validation),
    };

    const nri::Descriptor* Final_StorageTextures[] = {
        GetStorageDescriptor(Texture::Final),
    };

    const nri::Descriptor* DlssBefore_Textures[] = {
        GetDescriptor(Texture::Normal_Roughness),
        GetDescriptor(Texture::BaseColor_Metalness),
        GetDescriptor(Texture::Unfiltered_Spec),
    };

    const nri::Descriptor* DlssBefore_StorageTextures[] = {
        GetStorageDescriptor(Texture::ViewZ),
        GetStorageDescriptor(Texture::RRGuide_DiffAlbedo),
        GetStorageDescriptor(Texture::RRGuide_SpecAlbedo),
        GetStorageDescriptor(Texture::RRGuide_SpecHitDistance),
        GetStorageDescriptor(Texture::RRGuide_Normal_Roughness),
    };

    const nri::Descriptor* DlssAfter_StorageTextures[] = {
        GetStorageDescriptor(Texture::DlssOutput),
    };

    std::vector<nri::Descriptor*> RayTracing_Textures(m_Scene.materials.size() * TEXTURES_PER_MATERIAL);
    for (size_t i = 0; i < m_Scene.materials.size(); i++) {
        const size_t index = i * TEXTURES_PER_MATERIAL;
        const utils::Material& material = m_Scene.materials[i];

        RayTracing_Textures[index] = GetDescriptorForReadOnlyTexture(material.baseColorTexIndex);
        RayTracing_Textures[index + 1] = GetDescriptorForReadOnlyTexture(material.roughnessMetalnessTexIndex);
        RayTracing_Textures[index + 2] = GetDescriptorForReadOnlyTexture(material.normalTexIndex);
        RayTracing_Textures[index + 3] = GetDescriptorForReadOnlyTexture(material.emissiveTexIndex);
    }

    const nri::Descriptor* Sharc_StorageBuffers[] = {
        GetStorageDescriptor(Buffer::SharcHashEntries),
        GetStorageDescriptor(Buffer::SharcAccumulated),
        GetStorageDescriptor(Buffer::SharcResolved),
    };

    // Allocate and update everything in one go
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::SharcUpdatePing), 2, 0));    // and pong
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::ConfidenceBlurPing), 2, 0)); // and pong
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::TraceOpaque), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::Composition), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::TraceTransparent), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::TaaPing), 2, 0)); // and pong
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::Final), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::DlssBefore), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_OTHER, &Get(DescriptorSet::DlssAfter), 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_RAY_TRACING, &Get(DescriptorSet::RayTracing), 1, helper::GetCountOf(RayTracing_Textures)));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, SET_SHARC, &Get(DescriptorSet::Sharc), 1, 0));

    std::vector<nri::UpdateDescriptorRangeDesc> updateDescriptorRangeDescs;
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::SharcUpdatePing), 0, 0, SharcUpdatePing_Textures, helper::GetCountOf(SharcUpdatePing_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::SharcUpdatePing), 1, 0, SharcUpdatePing_StorageTextures, helper::GetCountOf(SharcUpdatePing_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::SharcUpdatePong), 0, 0, SharcUpdatePong_Textures, helper::GetCountOf(SharcUpdatePong_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::SharcUpdatePong), 1, 0, SharcUpdatePong_StorageTextures, helper::GetCountOf(SharcUpdatePong_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::ConfidenceBlurPing), 0, 0, ConfidenceBlurPing_Textures, helper::GetCountOf(ConfidenceBlurPing_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::ConfidenceBlurPing), 1, 0, ConfidenceBlurPing_StorageTextures, helper::GetCountOf(ConfidenceBlurPing_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::ConfidenceBlurPong), 0, 0, ConfidenceBlurPong_Textures, helper::GetCountOf(ConfidenceBlurPong_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::ConfidenceBlurPong), 1, 0, ConfidenceBlurPong_StorageTextures, helper::GetCountOf(ConfidenceBlurPong_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TraceOpaque), 0, 0, TraceOpaque_Textures, helper::GetCountOf(TraceOpaque_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TraceOpaque), 1, 0, TraceOpaque_StorageTextures, helper::GetCountOf(TraceOpaque_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::Composition), 0, 0, Composition_Textures, helper::GetCountOf(Composition_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::Composition), 1, 0, Composition_StorageTextures, helper::GetCountOf(Composition_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TraceTransparent), 0, 0, TraceTransparent_Textures, helper::GetCountOf(TraceTransparent_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TraceTransparent), 1, 0, TraceTransparent_StorageTextures, helper::GetCountOf(TraceTransparent_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TaaPing), 0, 0, TaaPing_Textures, helper::GetCountOf(TaaPing_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TaaPing), 1, 0, TaaPing_StorageTextures, helper::GetCountOf(TaaPing_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TaaPong), 0, 0, TaaPong_Textures, helper::GetCountOf(TaaPong_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::TaaPong), 1, 0, TaaPong_StorageTextures, helper::GetCountOf(TaaPong_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::Final), 0, 0, Final_Textures, helper::GetCountOf(Final_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::Final), 1, 0, Final_StorageTextures, helper::GetCountOf(Final_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::DlssBefore), 0, 0, DlssBefore_Textures, helper::GetCountOf(DlssBefore_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::DlssBefore), 1, 0, DlssBefore_StorageTextures, helper::GetCountOf(DlssBefore_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::DlssAfter), 1, 0, DlssAfter_StorageTextures, helper::GetCountOf(DlssAfter_StorageTextures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::RayTracing), 0, 0, RayTracing_Textures.data(), helper::GetCountOf(RayTracing_Textures)});
    updateDescriptorRangeDescs.push_back({Get(DescriptorSet::Sharc), 0, 0, Sharc_StorageBuffers, helper::GetCountOf(Sharc_StorageBuffers)});

    NRI.UpdateDescriptorRanges(updateDescriptorRangeDescs.data(), helper::GetCountOf(updateDescriptorRangeDescs));
}

void Sample::CreateTexture(Texture texture, const char* debugName, nri::Format format, nri::Dim_t width, nri::Dim_t height, nri::Dim_t mipNum, nri::Dim_t arraySize, bool isReadOnly, nri::AccessBits initialAccess) {
    nri::TextureDesc desc = {};
    desc.type = nri::TextureType::TEXTURE_2D;
    desc.usage = nri::TextureUsageBits::SHADER_RESOURCE | (isReadOnly ? nri::TextureUsageBits::NONE : nri::TextureUsageBits::SHADER_RESOURCE_STORAGE);
    desc.format = format;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.mipNum = mipNum;
    desc.layerNum = arraySize;
    desc.sampleNum = 1;

    NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, desc, Get(texture)));

    NRI.SetDebugName((nri::Object*)Get(texture), debugName);

    int32_t index = (int32_t)texture - (int32_t)Texture::BaseReadOnlyTexture;
    nri::TextureViewDesc viewDesc = {Get(texture), arraySize > 1 ? nri::TextureView::TEXTURE_ARRAY : nri::TextureView::TEXTURE, desc.format};
    NRI_ABORT_ON_FAILURE(NRI.CreateTextureView(viewDesc, index >= 0 ? GetDescriptorForReadOnlyTexture((uint32_t)index) : GetDescriptor(texture)));

    if (desc.usage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE) {
        const nri::FormatProps* formatProps = nriGetFormatProps(desc.format);

        viewDesc.format = formatProps->isSrgb ? nri::Format((uint8_t)desc.format - 1) : desc.format; // demote sRGB to UNORM
        viewDesc.type = arraySize > 1 ? nri::TextureView::STORAGE_TEXTURE_ARRAY : nri::TextureView::STORAGE_TEXTURE;
        NRI_ABORT_ON_FAILURE(NRI.CreateTextureView(viewDesc, GetStorageDescriptor(texture)));
    }

    if (initialAccess != nri::AccessBits::NONE) { // initial state
        nri::Layout layout = nri::Layout::SHADER_RESOURCE;
        if (initialAccess & nri::AccessBits::COPY_SOURCE)
            layout = nri::Layout::COPY_SOURCE;
        else if (initialAccess & nri::AccessBits::COPY_DESTINATION)
            layout = nri::Layout::COPY_DESTINATION;
        else if (initialAccess & nri::AccessBits::SHADER_RESOURCE_STORAGE)
            layout = nri::Layout::SHADER_RESOURCE_STORAGE;

        GetState(texture) = TextureBarrierFromUnknown(Get(texture), {initialAccess, layout});
    }
}

void Sample::CreateBuffer(Buffer buffer, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage) {
    if (!elements)
        elements = 1;

    nri::BufferDesc desc = {};
    desc.size = elements * stride;
    desc.structureStride = stride;
    desc.usage = usage;

    NRI_ABORT_ON_FAILURE(NRI.CreatePlacedBuffer(*m_Device, NriDeviceHeap, desc, Get(buffer)));

    NRI.SetDebugName((nri::Object*)Get(buffer), debugName);

    if (desc.usage & nri::BufferUsageBits::SHADER_RESOURCE) {
        const nri::BufferViewDesc viewDesc = {Get(buffer), nri::BufferView::STRUCTURED_BUFFER};
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, GetDescriptor(buffer)));
    }

    if (desc.usage & nri::BufferUsageBits::SHADER_RESOURCE_STORAGE) {
        const nri::BufferViewDesc viewDesc = {Get(buffer), nri::BufferView::STORAGE_STRUCTURED_BUFFER};
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, GetStorageDescriptor(buffer)));
    }
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
            const utils::Primitive& primitive = m_Scene.primitives[staticPrimitiveIndex];

            data.uv0 = float16_t2(float2(v0.uv[0], v0.uv[1]));
            data.uv1 = float16_t2(float2(v1.uv[0], v1.uv[1]));
            data.uv2 = float16_t2(float2(v2.uv[0], v2.uv[1]));
            data.worldArea = primitive.worldArea;

            data.n0 = float16_t2(float2(n0.x, n0.y));
            data.n1 = float16_t2(float2(n1.x, n1.y));
            data.n2 = float16_t2(float2(n2.x, n2.y));
            data.uvArea = primitive.uvArea;

            data.t0 = float16_t2(float2(t0.x, t0.y));
            data.t1 = float16_t2(float2(t1.x, t1.y));
            data.t2 = float16_t2(float2(t2.x, t2.y));
            data.bitangentSign = v0.T[3];
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
        textureUploadDescs.push_back({&subresources[subresourceOffset], Get((Texture)((size_t)Texture::BaseReadOnlyTexture + i)), {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}});

        nri::Dim_t mipNum = texture->GetMipNum();
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

    // Buffer data
    nri::BufferUploadDesc bufferUploadDescs[] = {
        {primitiveData.data(), Get(Buffer::PrimitiveData), {nri::AccessBits::SHADER_RESOURCE}},
    };

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
            } else if (!instance.allowUpdate)
                continue;

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

                // World to world (previous state) transform
                // FP64 used to avoid imprecision problems on close up views (InvertOrtho can't be used due to scaling factors)
                double4x4 dmWorldToObject = double4x4(mObjectToWorld);
                dmWorldToObject.Invert();

                double4x4 dmObjectToWorldPrev = double4x4(mObjectToWorldPrev);
                mOverloadedMatrix = float4x4(dmObjectToWorldPrev * dmWorldToObject);

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
            bool isForcedEmission = m_Settings.emission && m_Settings.emissiveObjects && (i % 3 == 0);

            uint32_t flags = 0;
            if (!instance.allowUpdate)
                flags |= FLAG_STATIC;
            if (material.isHair)
                flags |= FLAG_HAIR;
            if (material.isLeaf)
                flags |= FLAG_LEAF;
            if (material.isSkin)
                flags |= FLAG_SKIN;
            if (material.IsTransparent())
                flags |= FLAG_TRANSPARENT;
            if (i >= staticInstanceCount) {
                if (isForcedEmission)
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
            instanceData.baseColorAndMetalnessScale = float16_t4(material.baseColorAndMetalnessScale);
            instanceData.emissionAndRoughnessScale = float16_t4(material.emissiveAndRoughnessScale);
            instanceData.normalUvScale = float16_t2(material.normalUvScale);
            instanceData.textureOffsetAndFlags = baseTextureIndex | (flags << FLAG_FIRST_BIT);
            instanceData.primitiveOffset = meshInstance.primitiveOffset;
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

                if (isForcedEmission || material.IsEmissive())
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

static inline void GetBasis(float3 N, float3& T, float3& B) {
    float sz = sign(N.z);
    float a = 1.0f / (sz + N.z);
    float ya = N.y * a;
    float b = N.x * ya;
    float c = N.x * sz;

    T = float3(c * N.x * a - 1.0f, sz * b, c);
    B = float3(b, N.y * ya - sz, N.y);
}

void Sample::UpdateConstantBuffer(uint32_t frameIndex, uint32_t maxAccumulatedFrameNum) {
    float3 sunDirection = GetSunDirection();
    float3 sunT, sunB;
    GetBasis(sunDirection, sunT, sunB);

    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectWprev = uint32_t(m_RenderResolution.x * m_SettingsPrev.resolutionScale + 0.5f);
    uint32_t rectHprev = uint32_t(m_RenderResolution.y * m_SettingsPrev.resolutionScale + 0.5f);

    float2 renderSize = float2(float(m_RenderResolution.x), float(m_RenderResolution.y));
    float2 outputSize = float2(float(GetOutputResolution().x), float(GetOutputResolution().y));
    float2 rectSize = float2(float(rectW), float(rectH));
    float2 rectSizePrev = float2(float(rectWprev), float(rectHprev));
    float2 jitter = (m_Settings.cameraJitter ? m_Camera.state.viewportJitter : 0.0f) / rectSize;
    float2 jitterPrev = (m_Settings.cameraJitter ? m_Camera.statePrev.viewportJitter : 0.0f) / rectSizePrev;

    float3 viewDir = -float3(m_Camera.state.mViewToWorld[2].xyz);
    float3 cameraGlobalPos = float3(m_Camera.state.globalPosition);
    float3 cameraGlobalPosPrev = float3(m_Camera.statePrev.globalPosition);

    float nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    float baseMipBias = ((m_Settings.TAA || IsDlssEnabled()) ? -0.5f : 0.0f) + log2f(m_Settings.resolutionScale);
    float mipBias = baseMipBias + log2f(renderSize.x / outputSize.x);

    float fps = 1000.0f / m_Timer.GetSmoothedFrameTime();
    fps = min(fps, 121.0f);

    float taaMaxAccumulatedFrameNum = maxAccumulatedFrameNum * 0.5f;
    float prevFrameMaxAccumulatedFrameNum = maxAccumulatedFrameNum * 0.3f;

    nrd::ReblurHitDistanceParameters hitDistanceParameters = {};
    hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;

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
        constants.gWorldToClip = m_Camera.state.mWorldToClip;
        constants.gWorldToViewPrev = m_Camera.statePrev.mWorldToView;
        constants.gWorldToClipPrev = m_Camera.statePrev.mWorldToClip;
        constants.gViewToWorldPrev = m_Camera.statePrev.mViewToWorld;
        constants.gHitDistSettings = float4(hitDistanceParameters.A, hitDistanceParameters.B, hitDistanceParameters.C, hitDistanceParameters.D);
        constants.gCameraFrustum = frustum;
        constants.gSunBasisX = float4(sunT, 0.0f);
        constants.gSunBasisY = float4(sunB, 0.0f);
        constants.gSunDirection = float4(sunDirection, 0.0f);
        constants.gCameraGlobalPos = float4(cameraGlobalPos, CAMERA_RELATIVE);
        constants.gCameraGlobalPosPrev = float4(cameraGlobalPosPrev, 0.0f);
        constants.gViewDirection = float4(viewDir, 0.0f);
        constants.gHairBaseColor = m_HairBaseColor;
        constants.gHairBetas = m_HairBetas;
        constants.gOutputSize = outputSize;
        constants.gRenderSize = renderSize;
        constants.gRectSize = rectSize;
        constants.gInvOutputSize = float2(1.0f, 1.0f) / outputSize;
        constants.gInvRenderSize = float2(1.0f, 1.0f) / renderSize;
        constants.gInvRectSize = float2(1.0f, 1.0f) / rectSize;
        constants.gRectSizePrev = rectSizePrev;
        constants.gInvSharcRenderSize = 1.0f / float2((float)GetSharcDims().x, (float)GetSharcDims().y);
        constants.gJitter = jitter;
        constants.gJitterPrev = jitterPrev;
        constants.gEmissionIntensityLights = m_Settings.emission ? m_Settings.emissionIntensityLights : 0.0f;
        constants.gEmissionIntensityCubes = m_Settings.emission ? m_Settings.emissionIntensityCubes : 0.0f;
        constants.gNearZ = nearZ;
        constants.gSeparator = USE_SHARC_DEBUG == 0 ? m_Settings.separator : 1.0f;
        constants.gRoughnessOverride = m_Settings.roughnessOverride;
        constants.gMetalnessOverride = m_Settings.metalnessOverride;
        constants.gUnitToMetersMultiplier = 1.0f / m_Settings.meterToUnitsMultiplier;
        constants.gTanSunAngularRadius = tan(radians(m_Settings.sunAngularDiameter * 0.5f));
        constants.gTanPixelAngularRadius = tan(0.5f * radians(m_Settings.camFov) / rectSize.x);
        constants.gDebug = m_Settings.debug;
        constants.gPrevFrameConfidence = (m_Settings.usePrevFrame && !m_Settings.RR && m_Settings.denoiser != DENOISER_REFERENCE) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f;
        constants.gUnproject = 1.0f / (0.5f * rectH * project[1]);
        constants.gAperture = m_DofAperture * 0.01f;
        constants.gFocalDistance = m_DofFocalDistance;
        constants.gFocalLength = (0.5f * (35.0f * 0.001f)) / tan(radians(m_Settings.camFov * 0.5f)); // for 35 mm sensor size (aka old-school 35 mm film)
        constants.gTAA = (m_Settings.denoiser != DENOISER_REFERENCE && m_Settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f;
        constants.gHdrScale = displayDesc.isHDR ? displayDesc.maxLuminance / 80.0f : 1.0f;
        constants.gExposure = m_Settings.exposure;
        constants.gMipBias = mipBias;
        constants.gOrthoMode = orthoMode;
        constants.gMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
        constants.gDenoiserType = (uint32_t)m_Settings.denoiser;
        constants.gDisableShadowsAndEnableImportanceSampling = (sunDirection.z < 0.0f && m_Settings.importanceSampling) ? 1 : 0;
        constants.gFrameIndex = frameIndex;
        constants.gForcedMaterial = m_Settings.forcedMaterial;
        constants.gUseNormalMap = m_Settings.normalMap ? 1 : 0;
        constants.gBounceNum = m_Settings.bounceNum;
        constants.gResolve = (m_Settings.denoiser == DENOISER_REFERENCE || m_Settings.RR) ? false : m_Resolve;
        constants.gValidation = m_ShowValidationOverlay && m_Settings.denoiser != DENOISER_REFERENCE && m_Settings.separator != 1.0f;
        constants.gSR = (m_Settings.SR && !m_Settings.RR) ? 1 : 0;
        constants.gRR = m_Settings.RR ? 1 : 0;
        constants.gIsSrgb = m_IsSrgb ? 1 : 0;
    }

    m_GlobalConstantBufferOffset = NRI.StreamConstantData(*m_Streamer, &constants, sizeof(constants));
}

uint32_t Sample::BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM>& transitions) {
    uint32_t n = 0;

    for (uint32_t i = 0; i < stateNum; i++) {
        const TextureState& state = states[i];
        nri::TextureBarrierDesc& transition = GetState(state.texture);

        bool isStateChanged = transition.after.access != state.after.access || transition.after.layout != state.after.layout;
        bool isStorageBarrier = transition.after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE && state.after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE;
        if (isStateChanged || isStorageBarrier)
            transitions[n++] = TextureBarrierFromState(transition, {state.after.access, state.after.layout});
    }

    return n;
}

void Sample::RestoreBindings(nri::CommandBuffer& commandBuffer) {
    NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
    NRI.CmdSetPipelineLayout(commandBuffer, nri::BindPoint::COMPUTE, *m_PipelineLayout);

    nri::SetRootDescriptorDesc root0 = {0, GetDescriptor(Descriptor::Constant_Buffer), m_GlobalConstantBufferOffset};
    NRI.CmdSetRootDescriptor(commandBuffer, root0);

    // TODO: ray tracing related resources are not always needed, but absence of root descriptors leads to a silent crash inside VK validation
    nri::SetDescriptorSetDesc rayTracingSet = {SET_RAY_TRACING, Get(DescriptorSet::RayTracing)};
    NRI.CmdSetDescriptorSet(commandBuffer, rayTracingSet);

    nri::SetDescriptorSetDesc sharcSet = {SET_SHARC, Get(DescriptorSet::Sharc)};
    NRI.CmdSetDescriptorSet(commandBuffer, sharcSet);

    nri::SetRootDescriptorDesc root1 = {1, GetDescriptor(Descriptor::TLAS_World)};
    NRI.CmdSetRootDescriptor(commandBuffer, root1);

    nri::SetRootDescriptorDesc root2 = {2, GetDescriptor(Descriptor::TLAS_Emissive)};
    NRI.CmdSetRootDescriptor(commandBuffer, root2);

    nri::SetRootDescriptorDesc root3 = {3, GetDescriptor(Buffer::InstanceData)};
    NRI.CmdSetRootDescriptor(commandBuffer, root3);

    nri::SetRootDescriptorDesc root4 = {4, GetDescriptor(Buffer::PrimitiveData)};
    NRI.CmdSetRootDescriptor(commandBuffer, root4);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::nriBeginAnnotation("Render frame", nri::BGRA_UNUSED);

    std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM> optimizedTransitions = {};

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

    // NRD common settings
    nrd::CommonSettings commonSettings = {};
    memcpy(commonSettings.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));
    memcpy(commonSettings.viewToClipMatrixPrev, &m_Camera.statePrev.mViewToClip, sizeof(m_Camera.statePrev.mViewToClip));
    memcpy(commonSettings.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
    memcpy(commonSettings.worldToViewMatrixPrev, &m_Camera.statePrev.mWorldToView, sizeof(m_Camera.statePrev.mWorldToView));
    commonSettings.motionVectorScale[0] = 1.0f / float(rectW);
    commonSettings.motionVectorScale[1] = 1.0f / float(rectH);
    commonSettings.motionVectorScale[2] = 1.0f;
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
    commonSettings.disocclusionThresholdAlternate = 0.1f; // for hair
    commonSettings.splitScreen = (m_Settings.denoiser == DENOISER_REFERENCE || m_Settings.RR || USE_SHARC_DEBUG != 0) ? 1.0f : m_Settings.separator;
    commonSettings.debug = m_Settings.debug;
    commonSettings.frameIndex = frameIndex;
    commonSettings.accumulationMode = m_ForceHistoryReset ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.isMotionVectorInWorldSpace = false;
    commonSettings.isHistoryConfidenceAvailable = m_Settings.confidence;
    commonSettings.enableValidation = m_ShowValidationOverlay;

    const nrd::LibraryDesc& nrdLibraryDesc = *nrd::GetLibraryDesc();
    if (nrdLibraryDesc.normalEncoding == nrd::NormalEncoding::R10_G10_B10_A2_UNORM) {
        commonSettings.strandMaterialID = MATERIAL_ID_HAIR;
        commonSettings.strandThickness = STRAND_THICKNESS * m_Settings.meterToUnitsMultiplier;
    }

    m_NRD.NewFrame();
    m_NRD.SetCommonSettings(commonSettings);

    // RECORDING START
    NRI.BeginCommandBuffer(commandBuffer, nullptr);

    //======================================================================================================================================
    // Resolution independent
    //======================================================================================================================================

    { // Copy upload requests to destinations
        helper::Annotation annotation(NRI, commandBuffer, "Streamer");

        { // Transitions
            const nri::BufferBarrierDesc transitions[] = {
                {Get(Buffer::InstanceData), {nri::AccessBits::SHADER_RESOURCE}, {nri::AccessBits::COPY_DESTINATION}},
                {Get(Buffer::SharcAccumulated), {nri::AccessBits::NONE}, {nri::AccessBits::COPY_DESTINATION}},
            };

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = transitions;
            barrierDesc.bufferNum = frameIndex == 0 ? 2 : 1;

            NRI.CmdBarrier(commandBuffer, barrierDesc);
        }

        NRI.CmdCopyStreamedData(commandBuffer, *m_Streamer);
    }

    { // TLAS and SHARC clear
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

        if (frameIndex == 0)
            NRI.CmdZeroBuffer(commandBuffer, *Get(Buffer::SharcAccumulated), 0, nri::WHOLE_SIZE);

        { // Transitions
            const nri::BufferBarrierDesc transitions[] = {
                {Get(Buffer::InstanceData), {nri::AccessBits::COPY_DESTINATION}, {nri::AccessBits::SHADER_RESOURCE}},
                {Get(Buffer::SharcAccumulated), {nri::AccessBits::COPY_DESTINATION}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
            };

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = transitions;
            barrierDesc.bufferNum = frameIndex == 0 ? 2 : 1;

            NRI.CmdBarrier(commandBuffer, barrierDesc);
        }
    }

    //======================================================================================================================================
    // Render resolution
    //======================================================================================================================================

    RestoreBindings(commandBuffer);

    { // SHARC
        helper::Annotation sharc(NRI, commandBuffer, "SHARC & History confidence");

        const nri::BufferBarrierDesc bufferTransitions[] = {
            {Get(Buffer::SharcHashEntries), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
            {Get(Buffer::SharcAccumulated), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
            {Get(Buffer::SharcResolved), {nri::AccessBits::SHADER_RESOURCE_STORAGE}, {nri::AccessBits::SHADER_RESOURCE_STORAGE}},
        };

        nri::BarrierDesc bufferBarrierDesc = {};
        bufferBarrierDesc.buffers = bufferTransitions;
        bufferBarrierDesc.bufferNum = helper::GetCountOf(bufferTransitions);

        { // Update
            helper::Annotation annotation(NRI, commandBuffer, "SHARC - Update");

            const Texture prevRadiance = isEven ? Texture::Gradient_StoredPong : Texture::Gradient_StoredPing;
            const Texture currRadiance = isEven ? Texture::Gradient_StoredPing : Texture::Gradient_StoredPong;
            const TextureState transitions[] = {
                // Input
                {prevRadiance, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                // Output
                {currRadiance, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                {Texture::Gradient_Ping, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            };

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = optimizedTransitions.data();
            barrierDesc.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdBarrier(commandBuffer, barrierDesc);

            nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(isEven ? DescriptorSet::SharcUpdatePing : DescriptorSet::SharcUpdatePong)};
            NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::SharcUpdate));
            NRI.CmdDispatch(commandBuffer, {GetSharcDims().x / 16, GetSharcDims().y / 16, 1});
            NRI.CmdBarrier(commandBuffer, bufferBarrierDesc);
        }

        { // Resolve
            helper::Annotation annotation(NRI, commandBuffer, "SHARC - Resolve");

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::SharcResolve));
            NRI.CmdDispatch(commandBuffer, {(SHARC_CAPACITY + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE, 1, 1});
            NRI.CmdBarrier(commandBuffer, bufferBarrierDesc);
        }

        { // History confidence
            helper::Annotation annotation(NRI, commandBuffer, "History confidence - Blur");

            // Blur
            for (uint32_t i = 0; i < 5u; i++) { // must be odd
                const TextureState transitions[] = {
                    // Input
                    {i % 2 == 0 ? Texture::Gradient_Ping : Texture::Gradient_Pong, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                    // Output
                    {i % 2 == 0 ? Texture::Gradient_Pong : Texture::Gradient_Ping, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                };

                nri::BarrierDesc barrierDesc = {};
                barrierDesc.textures = optimizedTransitions.data();
                barrierDesc.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdBarrier(commandBuffer, barrierDesc);

                nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(i % 2 == 0 ? DescriptorSet::ConfidenceBlurPing : DescriptorSet::ConfidenceBlurPong)};
                NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

                uint32_t step = 1 + i;
                nri::SetRootConstantsDesc rootConstants = {0, &step, 4};
                NRI.CmdSetRootConstants(commandBuffer, rootConstants);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::ConfidenceBlur));
                NRI.CmdDispatch(commandBuffer, {GetSharcDims().x / 16, GetSharcDims().y / 16, 1});
            }
        }
    }

    { // Trace opaque
        helper::Annotation annotation(NRI, commandBuffer, "Trace opaque");

        const TextureState transitions[] = {
            // Input
            {Texture::ComposedDiff, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::ComposedSpec_ViewZ, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            // Output
            {Texture::Mv, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::ViewZ, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::BaseColor_Metalness, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::DirectLighting, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::DirectEmission, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::PsrThroughput, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Unfiltered_Penumbra, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Unfiltered_Translucency, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Unfiltered_Diff, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Unfiltered_Spec, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
#if (NRD_MODE == SH)
            {Texture::Unfiltered_DiffSh, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Unfiltered_SpecSh, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
#endif
        };
        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::TraceOpaque)};
        NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

        uint32_t rectWmod = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
        uint32_t rectHmod = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
        uint32_t rectGridWmod = (rectWmod + 15) / 16;
        uint32_t rectGridHmod = (rectHmod + 15) / 16;

        NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::TraceOpaque));
        NRI.CmdDispatch(commandBuffer, {rectGridWmod, rectGridHmod, 1});
    }

    { // Shadow denoising
        helper::Annotation annotation(NRI, commandBuffer, "Shadow denoising");

        float3 sunDir = GetSunDirection();

        m_SigmaSettings.lightDirection[0] = sunDir.x;
        m_SigmaSettings.lightDirection[1] = sunDir.y;
        m_SigmaSettings.lightDirection[2] = sunDir.z;

        nrd::Identifier denoiser = NRD_ID(SIGMA_SHADOW);

        m_NRD.SetDenoiserSettings(denoiser, &m_SigmaSettings);

        Denoise(&denoiser, 1, commandBuffer);
    }

    { // Opaque denoising
        helper::Annotation annotation(NRI, commandBuffer, "Opaque denoising");

        if (m_Settings.denoiser == DENOISER_REBLUR || m_Settings.denoiser == DENOISER_REFERENCE) {
            nrd::ReblurHitDistanceParameters hitDistanceParameters = {};
            hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;
            m_ReblurSettings.hitDistanceParameters = hitDistanceParameters;

            nrd::ReblurSettings settings = m_ReblurSettings;
#if (NRD_MODE == SH)
            // High quality SG resolve allows to use more relaxed normal weights
            if (m_Resolve)
                settings.lobeAngleFraction *= 1.333f;
#endif

#if (NRD_MODE == SH)
            const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SPECULAR_SH)};
#else
            const nrd::Identifier denoisers[] = {NRD_ID(REBLUR_DIFFUSE_SPECULAR)};
#endif

            for (uint32_t i = 0; i < helper::GetCountOf(denoisers); i++)
                m_NRD.SetDenoiserSettings(denoisers[i], &settings);

            Denoise(denoisers, helper::GetCountOf(denoisers), commandBuffer);
        } else if (m_Settings.denoiser == DENOISER_RELAX) {
            nrd::RelaxSettings settings = m_RelaxSettings;
#if (NRD_MODE == SH)
            // High quality SG resolve allows to use more relaxed normal weights
            if (m_Resolve)
                settings.lobeAngleFraction *= 1.333f;
#endif

#if (NRD_MODE == SH)
            const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE_SPECULAR_SH)};
#else
            const nrd::Identifier denoisers[] = {NRD_ID(RELAX_DIFFUSE_SPECULAR)};
#endif

            for (uint32_t i = 0; i < helper::GetCountOf(denoisers); i++)
                m_NRD.SetDenoiserSettings(denoisers[i], &settings);

            Denoise(denoisers, helper::GetCountOf(denoisers), commandBuffer);
        }
    }

    RestoreBindings(commandBuffer);

    { // Composition
        helper::Annotation annotation(NRI, commandBuffer, "Composition");

        const TextureState transitions[] = {
            // Input
            {Texture::ViewZ, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::BaseColor_Metalness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::DirectLighting, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::DirectEmission, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::PsrThroughput, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Shadow, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Diff, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Spec, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
#if (NRD_MODE == SH)
            {Texture::DiffSh, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::SpecSh, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
#endif
            // Output
            {Texture::ComposedDiff, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::ComposedSpec_ViewZ, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
        };
        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::Composition)};
        NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

        NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Composition));
        NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
    }

    { // Trace transparent
        helper::Annotation annotation(NRI, commandBuffer, "Trace transparent");

        const TextureState transitions[] = {
            // Input
            {Texture::ComposedDiff, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::ComposedSpec_ViewZ, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            // Output
            {Texture::Composed, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Mv, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            {Texture::Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
        };

        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::TraceTransparent)};
        NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

        NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::TraceTransparent));
        NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
    }

    if (m_Settings.denoiser == DENOISER_REFERENCE) { // Reference
        helper::Annotation annotation(NRI, commandBuffer, "Reference accumulation");

        nrd::CommonSettings modifiedCommonSettings = commonSettings;
        modifiedCommonSettings.splitScreen = m_Settings.separator;

        nrd::Identifier denoiser = NRD_ID(REFERENCE);

        m_NRD.SetCommonSettings(modifiedCommonSettings);
        m_NRD.SetDenoiserSettings(denoiser, &m_ReferenceSettings);

        Denoise(&denoiser, 1, commandBuffer);

        RestoreBindings(commandBuffer);
    }

    //======================================================================================================================================
    // Output resolution
    //======================================================================================================================================

    const Texture taaHistoryInput = isEven ? Texture::TaaHistoryPong : Texture::TaaHistoryPing;
    const Texture taaHistoryOutput = isEven ? Texture::TaaHistoryPing : Texture::TaaHistoryPong;

    if (IsDlssEnabled()) {
        // Before DLSS
        if (m_Settings.SR) {
            helper::Annotation annotation(NRI, commandBuffer, "Before DLSS");

            const TextureState transitions[] = {
                // Input
                {Texture::Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::BaseColor_Metalness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::Unfiltered_Spec, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                // Output
                {Texture::ViewZ, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                {Texture::RRGuide_DiffAlbedo, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                {Texture::RRGuide_SpecAlbedo, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                {Texture::RRGuide_SpecHitDistance, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
                {Texture::RRGuide_Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            };
            nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::DlssBefore)};
            NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DlssBefore));
            NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
        }

        { // DLSS
            helper::Annotation annotation(NRI, commandBuffer, "DLSS");

            const TextureState transitions[] = {
                // Input
                {Texture::ViewZ, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::Mv, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::RRGuide_DiffAlbedo, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::RRGuide_SpecAlbedo, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::RRGuide_SpecHitDistance, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::RRGuide_Normal_Roughness, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                {Texture::Composed, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
                // Output
                {Texture::DlssOutput, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            };
            nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            bool resetHistory = m_ForceHistoryReset || m_Settings.SR != m_SettingsPrev.SR || m_Settings.RR != m_SettingsPrev.RR;

            nri::DispatchUpscaleDesc dispatchUpscaleDesc = {};
            dispatchUpscaleDesc.output = {Get(Texture::DlssOutput), GetStorageDescriptor(Texture::DlssOutput)};
            dispatchUpscaleDesc.input = {Get(Texture::Composed), GetDescriptor(Texture::Composed)};
            dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)rectW, (nri::Dim_t)rectH};
            dispatchUpscaleDesc.cameraJitter = {-m_Camera.state.viewportJitter.x, -m_Camera.state.viewportJitter.y};
            dispatchUpscaleDesc.mvScale = {1.0f, 1.0f};
            dispatchUpscaleDesc.flags = resetHistory ? nri::DispatchUpscaleBits::RESET_HISTORY : nri::DispatchUpscaleBits::NONE;

            if (m_Settings.RR) {
                dispatchUpscaleDesc.guides.denoiser.mv = {Get(Texture::Mv), GetDescriptor(Texture::Mv)};
                dispatchUpscaleDesc.guides.denoiser.depth = {Get(Texture::ViewZ), GetDescriptor(Texture::ViewZ)};
                dispatchUpscaleDesc.guides.denoiser.diffuseAlbedo = {Get(Texture::RRGuide_DiffAlbedo), GetDescriptor(Texture::RRGuide_DiffAlbedo)};
                dispatchUpscaleDesc.guides.denoiser.specularAlbedo = {Get(Texture::RRGuide_SpecAlbedo), GetDescriptor(Texture::RRGuide_SpecAlbedo)};
                dispatchUpscaleDesc.guides.denoiser.normalRoughness = {Get(Texture::RRGuide_Normal_Roughness), GetDescriptor(Texture::RRGuide_Normal_Roughness)};
                dispatchUpscaleDesc.guides.denoiser.specularMvOrHitT = {Get(Texture::RRGuide_SpecHitDistance), GetDescriptor(Texture::RRGuide_SpecHitDistance)};

                memcpy(&dispatchUpscaleDesc.settings.dlrr.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
                memcpy(&dispatchUpscaleDesc.settings.dlrr.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));

                NRI.CmdDispatchUpscale(commandBuffer, *m_DLRR, dispatchUpscaleDesc);
            } else {
                dispatchUpscaleDesc.guides.upscaler.mv = {Get(Texture::Mv), GetDescriptor(Texture::Mv)};
                dispatchUpscaleDesc.guides.upscaler.depth = {Get(Texture::ViewZ), GetDescriptor(Texture::ViewZ)};

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

            RestoreBindings(commandBuffer);
        }

        { // After DLSS
            helper::Annotation annotation(NRI, commandBuffer, "After Dlss");

            const TextureState transitions[] = {
                // Output
                {Texture::DlssOutput, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
            };
            nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
            NRI.CmdBarrier(commandBuffer, transitionBarriers);

            nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::DlssAfter)};
            NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DlssAfter));
            NRI.CmdDispatch(commandBuffer, {outputGridW, outputGridH, 1});
        }
    } else { // TAA
        helper::Annotation annotation(NRI, commandBuffer, "TAA");

        const TextureState transitions[] = {
            // Input
            {Texture::Mv, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Composed, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {taaHistoryInput, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            // Output
            {taaHistoryOutput, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
        };
        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(isEven ? DescriptorSet::TaaPing : DescriptorSet::TaaPong)};
        NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

        NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Taa));
        NRI.CmdDispatch(commandBuffer, {rectGridW, rectGridH, 1});
    }

    { // NIS
        helper::Annotation annotation(NRI, commandBuffer, "NIS");

        const TextureState transitions[] = {
            // Input
            {IsDlssEnabled() ? Texture::DlssOutput : taaHistoryOutput, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            // Output
            {Texture::PreFinal, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
        };

        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::DispatchUpscaleDesc dispatchUpscaleDesc = {};
        dispatchUpscaleDesc.settings.nis.sharpness = NIS_SHARPNESS;
        dispatchUpscaleDesc.output = {Get(Texture::PreFinal), GetStorageDescriptor(Texture::PreFinal)};

        if (IsDlssEnabled()) {
            dispatchUpscaleDesc.input = {Get(Texture::DlssOutput), GetDescriptor(Texture::DlssOutput)};
            dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
        } else {
            dispatchUpscaleDesc.input = {Get(taaHistoryOutput), GetDescriptor(isEven ? Texture::TaaHistoryPing : Texture::TaaHistoryPong)};
            dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)rectW, (nri::Dim_t)rectH};
        }

        NRI.CmdDispatchUpscale(commandBuffer, *m_NIS[m_SdrScale > 1.0f ? 1 : 0], dispatchUpscaleDesc);

        RestoreBindings(commandBuffer);
    }

    { // Final
        helper::Annotation annotation(NRI, commandBuffer, "Final");

        const TextureState transitions[] = {
            // Input
            {Texture::PreFinal, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Composed, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            {Texture::Validation, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE}},
            // Output
            {Texture::Final, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE}},
        };
        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, optimizedTransitions.data(), BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::SetDescriptorSetDesc otherSet = {SET_OTHER, Get(DescriptorSet::Final)};
        NRI.CmdSetDescriptorSet(commandBuffer, otherSet);

        NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Final));
        NRI.CmdDispatch(commandBuffer, {outputGridW, outputGridH, 1});
    }

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    nri::Result result = NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);
    if (result == nri::Result::OUT_OF_DATE)
        printf("Oops, unhandled out of date!\n");

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    { // Copy to back-buffer
        helper::Annotation annotation(NRI, commandBuffer, "Copy to back buffer");

        const nri::TextureBarrierDesc transitions[] = {
            TextureBarrierFromState(GetState(Texture::Final), {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE}),
            TextureBarrierFromUnknown(swapChainTexture.texture, {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION}),
        };
        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, transitions, (uint16_t)helper::GetCountOf(transitions)};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        NRI.CmdCopyTexture(commandBuffer, *swapChainTexture.texture, nullptr, *Get(Texture::Final), nullptr);
    }

    { // UI
        nri::TextureBarrierDesc before = {};
        before.texture = swapChainTexture.texture;
        before.before = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        before.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};

        nri::BarrierDesc transitionBarriers = {nullptr, 0, nullptr, 0, &before, 1};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

        nri::AttachmentDesc attachmentDesc = {};
        attachmentDesc.descriptor = swapChainTexture.colorAttachment;

        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colors = &attachmentDesc;
        renderingDesc.colorNum = 1;

        CmdCopyImguiData(commandBuffer, *m_Streamer);

        NRI.CmdBeginRendering(commandBuffer, renderingDesc);
        {
            CmdDrawImgui(commandBuffer, swapChainTexture.attachmentFormat, m_SdrScale, m_IsSrgb);
        }
        NRI.CmdEndRendering(commandBuffer);

        const nri::TextureBarrierDesc after = TextureBarrierFromState(before, {nri::AccessBits::NONE, nri::Layout::PRESENT, nri::StageBits::NONE});
        transitionBarriers = {nullptr, 0, nullptr, 0, &after, 1};
        NRI.CmdBarrier(commandBuffer, transitionBarriers);
    }

    // RECORDING END
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::FenceSubmitDesc frameFence = {};
        frameFence.fence = m_FrameFence;
        frameFence.value = 1 + frameIndex;

        nri::FenceSubmitDesc textureAcquiredFence = {};
        textureAcquiredFence.fence = swapChainAcquireSemaphore;
        textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

        nri::FenceSubmitDesc renderingFinishedFence = {};
        renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

        nri::FenceSubmitDesc signalFences[] = {renderingFinishedFence, frameFence};

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.waitFences = &textureAcquiredFence;
        queueSubmitDesc.waitFenceNum = 1;
        queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = signalFences;
        queueSubmitDesc.signalFenceNum = helper::GetCountOf(signalFences);

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    NRI.EndStreamerFrame(*m_Streamer);

    nri::nriEndAnnotation();

    // Present
    nri::nriBeginAnnotation("Present", nri::BGRA_UNUSED);
    {
        NRI.QueuePresent(*m_SwapChain, *swapChainTexture.releaseSemaphore);
    }
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