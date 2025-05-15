# NRD SAMPLE

All-in-one repository including all relevant pieces to see [*NRD (NVIDIA Real-time Denoisers)*](https://github.com/NVIDIA-RTX/NRD) in action. The sample is cross-platform, it's based on [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIA-RTX/NRI) to bring cross-GraphicsAPI support.

*NRD sample* is a land for high performance path tracing for games. Some features to highlight:
- minimalistic path tracer utilizing *Trace Ray Inline*
- HALF resolution (checkerboard), FULL resolution and FULL resolution tracing with PROBABILISTIC diffuse / specular selection at the primary hit
- NRD denoising, including occlusion-only and spherical harmonics (spherical gaussians) modes
- overhead-free multi-bounce propagation (even in case of a single bounce) based on reusing the previously denoised frame
- SHARC radiance cache
- reference accumulation
- several rays per pixel and bounces
- realistic glass with multi-bounce reflections and refractions
- mip level calculation
- curvature estimation
- native integration of DLSS-SR and DLSS-RR via NGX API (not StreamLine)

# BUILD INSTRUCTIONS

- Install [*Cmake*](https://cmake.org/download/) 3.16+
- Install on
    - Windows: latest *WindowsSDK* and *VulkanSDK*
    - Linux (x86-64): latest *VulkanSDK*, *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
    - Linux (aarch64): find a precompiled binary for [*DXC*](https://github.com/microsoft/DirectXShaderCompiler), *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
    - To build the binary with static MSVC runtime, add `-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` parameter when deploying the project
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

### CMAKE OPTIONS

- `USE_MINIMAL_DATA=OFF` - download minimal resource package (90MB)
- `DISABLE_SHADER_COMPILATION=OFF` - disable compilation of shaders (shaders can be built on other platform)

# HOW TO RUN

- Run `3-Run NRD sample` script and answer the cmdline questions to set the runtime parameters
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- The executables can be found in `_Bin`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run NRD sample` script)

### REQUIREMENTS

Any ray tracing compatible GPU.

### USAGE

- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- F2 - go to next test (only if *TESTS* section is unfolded)
- F3 - toggle emission
- Tab - UI toggle
- Space - animation toggle
- PgUp/PgDown - switch between denoisers

By default *NRD* is used in common mode. But it can also be used in occlusion-only (including directional) and SH (spherical harmonics) modes in the sample. To change the behavior `NRD_MODE` macro needs to be changed from `NORMAL` to `OCCLUSION`, `SH` or `DIRECTIONAL_OCCLUSION` in `Shared.hlsli`.

Notes:
- RELAX doesn't support AO / SO denoising. If RELAX is the current denoiser, ambient term will be flat, but energy correct.
