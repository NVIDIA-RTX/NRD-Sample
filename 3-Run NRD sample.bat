@echo off

set DIR_BIN=%cd%\_Bin\Release

if not exist "%DIR_BIN%" (
    set DIR_BIN=%cd%\_Bin\Debug
)

if not exist "%DIR_BIN%" (
    echo The project is not compiled!
    exit /b
)
echo Running NRD sample from '%DIR_BIN%'...

set API=D3D12
echo.
echo 1 - D3D12
echo 2 - VULKAN
:CHOOSE_API
    set /P M=Choose API [1-2]:
    if %M%==1 (
        set API=D3D12
        goto RESOLUTION
    )
    if %M%==2 (
        set API=VULKAN
        goto RESOLUTION
    )
    goto CHOOSE_API

:RESOLUTION
set WIDTH=1920
set HEIGHT=1080
echo.
echo 1 - 1080p
echo 2 - 1440p
echo 3 - 2160p
echo 4 - 1080p (ultra wide)
echo 5 - 1440p (ultra wide)
:CHOOSE_RESOLUTION
    set /P M=Choose resolution [1-5]:
    if %M%==1 (
        set WIDTH=1920
        set HEIGHT=1080
        goto DLSS
    )
    if %M%==2 (
        set WIDTH=2560
        set HEIGHT=1440
        goto DLSS
    )
    if %M%==3 (
        set WIDTH=3840
        set HEIGHT=2160
        goto DLSS
    )
    if %M%==4 (
        set WIDTH=2560
        set HEIGHT=1080
        goto DLSS
    )
    if %M%==5 (
        set WIDTH=3440
        set HEIGHT=1440
        goto DLSS
    )
    goto CHOOSE_RESOLUTION

:DLSS
set DLSS=-1
echo.
echo 1 - off
echo 2 - ultra performance
echo 3 - performance
echo 4 - balanced
echo 5 - quality
echo 6 - aa
:CHOOSE_DLSS
    set /P M=DLSS mode [1-6]:
    if %M%==1 (
        set DLSS=-1
        goto SCENE
    )
    if %M%==2 (
        set DLSS=0
        goto SCENE
    )
    if %M%==3 (
        set DLSS=1
        goto SCENE
    )
    if %M%==4 (
        set DLSS=2
        goto SCENE
    )
    if %M%==5 (
        set DLSS=3
        goto SCENE
    )
    if %M%==6 (
        set DLSS=4
        goto SCENE
    )
    goto CHOOSE_DLSS

:SCENE
set SCENE=Bistro\BistroInterior.gltf
echo.
echo 1 - Bistro (interior)
echo 2 - Bistro (exterior)
echo 3 - Shader balls
echo 4 - Kitchen
:CHOOSE_SCENE
    set /P M=Choose scene [1-4]:
    if %M%==1 (
        set SCENE=Bistro\BistroInterior.gltf
        goto RUN
    )
    if %M%==2 (
        set SCENE=Bistro\BistroExterior.gltf
        goto RUN
    )
    if %M%==3 (
        set SCENE=ShaderBalls\ShaderBalls.gltf
        goto RUN
    )
    if %M%==4 (
        set SCENE=Kitchen\Kitchen.gltf
        goto RUN
    )
    goto CHOOSE_SCENE

:RUN
start "NRD sample" "%DIR_BIN%\NRDSample.exe" --width=%WIDTH% --height=%HEIGHT% --api=%API% --scene=%SCENE% --dlssQuality=%DLSS% --debugNRD

exit /b
