cmake_minimum_required(VERSION 3.30)

include(FetchContent)

# Options
option(DISABLE_SHADER_COMPILATION "Disable shader compilation" OFF)
option(USE_MINIMAL_DATA "Use minimal '_Data' (90MB)" OFF)
set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")

# Cached
set(DXC_CUSTOM_PATH "custom/path/to/dxc" CACHE STRING "This DXC will be used if Vulkan SDK is not installed")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")

# Create project
project(NRDSample LANGUAGES C CXX)

# Globals?
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# Download dependencies (Packman)
if(WIN32)
    set(PACKMAN_EXT ".cmd")
endif()

if((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    if(WIN32)
        set(PACKMAN_PLATFORM "windows-x86_64")
    else()
        set(PACKMAN_PLATFORM "linux-x86_64")
    endif()
elseif((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set(PACKMAN_PLATFORM "linux-aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

message("Packman platform: ${PACKMAN_PLATFORM}")

if(USE_MINIMAL_DATA)
    set(NRI_DATA_VERSION 2.3_minimal)
    message("Using minimal '_Data'")
else()
    set(NRI_DATA_VERSION 2.3)
endif()

execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/packman${PACKMAN_EXT}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -p ${PACKMAN_PLATFORM} -t nri_data_version=${NRI_DATA_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PACKMAN_RESULT
)

if(NOT "${PACKMAN_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Packman failed(code = ${PACKMAN_RESULT})")
endif()

# Download dependencies
set(DEPS)

# SHARC
FetchContent_Declare(
    sharc
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/SHARC.git
    GIT_TAG main
    GIT_SHALLOW 1
)
list(APPEND DEPS sharc)

if(DEPS)
    message("Downloading dependencies:")
    message(STATUS "${DEPS} ...")

    FetchContent_MakeAvailable(${DEPS})
endif()

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(MSVC)
    set(COMPILE_OPTIONS /W4 /WX /wd4324 /wd4834)
else()
    message(WARNING "Unknown compiler!")
endif()

# Compile definitions
if(WIN32)
    set(COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
endif()

# External/NRIFramework
set(NRIF_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(NRI_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
option(NRI_ENABLE_NONE_SUPPORT "" OFF)
option(NRI_ENABLE_D3D11_SUPPORT "" OFF)
option(NRI_ENABLE_NIS_SDK "" ON)
option(NRI_ENABLE_NGX_SDK "" ON)
option(NRI_ENABLE_FFX_SDK "" ON)
option(NRI_ENABLE_XESS_SDK "" ON)
add_subdirectory("External/NRIFramework")

# External/NRD
set(NRD_DXC_CUSTOM_PATH ${DXC_CUSTOM_PATH} CACHE STRING "")
set(NRD_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(NRD_NORMAL_ENCODING "2" CACHE STRING "")
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "")
option(NRD_DISABLE_SHADER_COMPILATION "" ${DISABLE_SHADER_COMPILATION})
add_subdirectory("External/NRD")

# Move dependencies into "External"
function(fix_folders _DIRECTORY _FOLDER)
    get_property(_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)

    foreach(_TARGET IN LISTS _TARGETS)
        get_property(_FOLDER_ORIGINAL TARGET ${_TARGET} PROPERTY FOLDER)
        set_property(TARGET ${_TARGET} PROPERTY FOLDER "${_FOLDER}/${_FOLDER_ORIGINAL}")
    endforeach()

    get_property(SUBDIRS DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)

    foreach(SUBDIR IN LISTS SUBDIRS)
        fix_folders("${SUBDIR}" "${_FOLDER}")
    endforeach()
endfunction()

fix_folders("External/NRIFramework" "External")
fix_folders("External/NRD" "External")

# Get source directories for 3rd parties
get_target_property(ML_SOURCE_DIR MathLib INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(NRI_SOURCE_DIR NRI SOURCE_DIR)
get_target_property(NRD_SOURCE_DIR NRD SOURCE_DIR)

# NRD sample
file(GLOB NRD_SAMPLE_SOURCE "Source/*.cpp")
source_group("" FILES ${NRD_SAMPLE_SOURCE})

add_executable(${PROJECT_NAME} ${NRD_SAMPLE_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE
    "Source"
    "External"
    "$ENV{VULKAN_SDK}/include"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${COMPILE_DEFINITIONS}
    PROJECT_NAME=${PROJECT_NAME}
    NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING}
    DXC_PATH=${DXC_PATH}
    DXC_SPIRV_PATH=${DXC_SPIRV_PATH}
    ML_SOURCE_DIR=${ML_SOURCE_DIR}
    NRI_SOURCE_DIR=${NRI_SOURCE_DIR}/Include
    NRD_SOURCE_DIR=${NRD_SOURCE_DIR}/Shaders/Include
    SHARC_SOURCE_DIR=${sharc_SOURCE_DIR}/Include
)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    NRIFramework
    NRD
    NRDIntegration
)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Sample")
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Copy arguments for Visual Studio Smart Command Line Arguments extension
if(WIN32 AND MSVC)
    configure_file(.args "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.args.json" COPYONLY)
endif()

# Shaders
file(GLOB_RECURSE SHADERS
    "Shaders/*.hlsl"
    "Shaders/*.hlsli"
)
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

set(SHADERMAKE_GENERAL_ARGS
    --useAPI --flatten --stripReflection --WX
    --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128 # TODO: must be in sync with "VK_BINDING_OFFSETS"
    --binary
    --shaderModel 6_6
    --sourceDir "Shaders"
    --ignoreConfigDir
    -c "Shaders/Shaders.cfg"
    -o "${SHADER_OUTPUT_PATH}"
    -I "Shaders"
    -I "External"
    -I "${ML_SOURCE_DIR}"
    -I "${NRD_SOURCE_DIR}/Shaders/Include"
    -I "${NRI_SOURCE_DIR}/Include"
    -I "${sharc_SOURCE_DIR}/Include"
    -D NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    -D NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING}
)

if(WIN32)
    add_custom_target(${PROJECT_NAME}Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXIL --compiler "${DXC_PATH}"
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}" --hlsl2021
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
else()
    add_custom_target(${PROJECT_NAME}Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}" --hlsl2021
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
endif()

set_property(TARGET ${PROJECT_NAME}Shaders PROPERTY FOLDER "Sample")
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}Shaders)
