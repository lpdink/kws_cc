cmake_minimum_required(VERSION 3.14.3)
project(mini_demo VERSION 0.0.1)
enable_language(ASM)

set(CMAKE_DEBUG_TARGET_PROPERTIES
    INCLUDE_DIRECTORIES
    COMPILE_DEFINITIONS
    POSITION_INDEPENDENT_CODE
    CONTAINER_SIZE_REQUIRED
    LIB_VERSION
)

include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)

include_directories(${common_incs})

if (${OS_TYPE} MATCHES "HLOS")
    # for android device
    add_executable(speech_backend_android_exec
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils
    )

    set(ignoreMe ${DSP_TYPE})
    add_library(speech_backend_idl
        SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/speech_backend_stub
    )

    build_idl(inc/speech_backend.idl speech_backend_idl)
    SET(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_LIBRARY} -lc++")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++")

    include_directories(
        ${common_incs}
    )

    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_options(speech_backend_android_exec)
    link_custom_library(speech_backend_android_exec ${dsprpc})
    link_custom_library(speech_backend_idl ${dsprpc})
    add_dependencies(speech_backend_android_exec speech_backend_idl)
    target_link_libraries(speech_backend_android_exec speech_backend_idl)

else()
    #################################################
    #Hexagon Dynamic Library
    #################################################
    add_library(speech_backend_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/speech_backend_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/conv2d
    )
    include_directories(
        ${common_incs}
    )
    target_include_directories(speech_backend_skel PUBLIC ../inc)

    target_link_libraries(speech_backend_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(speech_backend_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    build_idl(inc/speech_backend.idl speech_backend_skel)

    #With Hexagon Tools 8.4.0x, new symbols are added for c++17 compliance,
    #whose definitions are not present on targets earlier to saipan.
    #Link this weak_refs.so to your library for weak definitions of these new symbols.
    string(FIND ${PREBUILT_LIB_DIR} "toolv81" is_toolv81)
    string(FIND ${PREBUILT_LIB_DIR} "toolv82" is_toolv82)
    string(FIND ${PREBUILT_LIB_DIR} "toolv83" is_toolv83)

    if (${is_toolv81} EQUAL -1 AND ${is_toolv82} EQUAL -1 AND ${is_toolv83} EQUAL -1)
        target_link_libraries(speech_backend_skel ${HEXAGON_SDK_ROOT}/libs/weak_refs/ship/hexagon_toolv84/weak_refs.a)
    endif()

endif()
