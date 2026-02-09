# NRD SAMPLE

NRD Sample is a high-performance playground and reference implementation for path tracing in games. It provides a comprehensive environment to see [*NRD (NVIDIA Real-time Denoisers)*](https://github.com/NVIDIA-RTX/NRD) in action across all possible use cases and compare it directly with *DLSS-RR*. While *NRD* and *DLSS-RR* are highly competitive today, effective comparison requires enabling *NRD* "SH" mode (set via the `NRD_MODE` macro in `Shared.hlsli`). The sample is designed to demonstrate production-ready path tracing best practices suitable for real-time gaming that balance visual fidelity with high performance. A core focus of the project is high-performance glass and transparency rendering. Built on the [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIA-RTX/NRI), the sample is natively cross-platform, supporting both *D3D12* and *Vulkan*. The ideas from the sample are already used in several *AAA* games and game mods.

Features:
- best-in-class performance:
  - RTX 4080 @ 1440p (native) shows ~150 FPS at day time and ~110 FPS at night time in *Bistro* scene (when the importance sampling comes into play)
- fast path tracing based on *Trace Ray Inline* supporting:
  - tracing variants:
    - full resolution: probabilistic diffuse or specular lobe selection at the primary hit (recommended main use case)
    - half resolution: 0.5 diffuse ray + 0.5 specular ray (checkerboard rendering)
    - full resolution: 1 diffuse ray + 1 specular ray
  - material models:
    - "base color - metalness"
    - RTXCR hair and skin
  - smart levels of radiance caching, allowing to do only 1 bounce:
    - L1 radiance cache: previously denoised frame (essential for refractions)
    - L2 radiance cache: [*SHARC*](https://github.com/NVIDIA-RTX/SHARC) (global voxel-based cache)
  - simple and robust importance sampling scheme designed to work with generic emissive pixels, i.e. not analytical lights (not RESTIR)
  - realistic fast and robust glass with multi-bounce reflections and refractions not requiring denoising
  - several rays per pixel and bounces (mostly for comparison)
  - extremely fast implementation of Ray Cones for mip level calculation
  - curvature estimation
- *NRD* denoising:
  - simple integration based on [*NRD Integration*](https://github.com/NVIDIA-RTX/NRD/tree/master/Integration) layer
  - *RELAX* and *REBLUR* for radiance
  - *SIGMA* for sun shadows
  - Spherical Harmonics (Gaussians) mode essential for upscalers
  - history confidence bumping IQ of dynamic lighting to a new level
  - *NRD* debug validation layer
  - denoising best practices including hair denoising
  - occlusion-only modes
  - reference accumulation
  - NRD unit tests
  - NRD special tests
- native integration of *DLSS-SR*, *DLSS-RR*, *FSR* and *XeSS* via *NRIUpscaler* extension (not StreamLine)
 - NOTE: don't forget to modify `NRD_MODE` in `Shared.hlsli` to `SH` to unclock Spherical Harmonics (Gaussians) resolve mode essential for image quality!

GitHub branches:
- "best practices" `simplex` branch
  - optimized for recommended usage (probabilistic split, `HitDistanceReconstructionMode::AREA_3X3` reconstruction). Recommended for learning NRD integration and borrowing PT code. It offers better performance and has less code
- "all-in-one" `main` branch
  - contains every possible option and setting, used primarily for NRD development and maintenance.

## HOW TO BUILD

- Install [*Cmake*](https://cmake.org/download/) 3.30+
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
    - To build the binary with static MSVC runtime, add `-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` parameter when deploying the project
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

*CMake* options:

- `USE_MINIMAL_DATA=OFF` - download minimal resource package (90MB)
- `RTXCR_INTEGRATION` - use RTXCR for hair and skin rendering, download sample scene

## HOW TO RUN

- Run `3-Run` script and answer the cmdline questions to set the runtime parameters
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- The executables can be found in `_Bin`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run` script)

Requirements:
- any GPU supporting "trace ray inline"

## USAGE

By default, NRD is used in common mode. To switch to "SH" or "occlusion-only" modes, modify the `NRD_MODE` macro in `Shared.hlsli` to one of the following: `NORMAL`, `OCCLUSION`, `SH`, or `DIRECTIONAL_OCCLUSION`. RELAX doesn't support AO / SO denoising. If RELAX is the current denoiser, ambient term will be flat.

Controls:
- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- F2 - go to next test (only if *TESTS* section is unfolded)
- F3 - toggle emission
- Tab - UI toggle
- Space - animation toggle
- PgUp/PgDown - switch between denoisers
