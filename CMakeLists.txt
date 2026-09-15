cmake_minimum_required(VERSION 3.10)

project(REGENA LANGUAGES CXX)

include_directories(include/)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimised Release build (-O3 -DNDEBUG for GCC)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    message(STATUS "  You can specify with the flag -DCMAKE_BUILD_TYPE=<Debug|Release|MinSizeRel|RelWithDebInfo>")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

message(STATUS "CMake build type is set to ${CMAKE_BUILD_TYPE}")

find_package(Threads REQUIRED)

# Determine system architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
    set(ARCHITECTURE "amd64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    set(ARCHITECTURE "arm64")
else()
    set(ARCHITECTURE "unknown")
endif()

message(STATUS "Detected architecture: ${ARCHITECTURE}")

# SSE support (default: OFF; amd64 only)
option(ENABLE_SSE "Enable SSE support" OFF)

if (ENABLE_SSE)
    if (ARCHITECTURE STREQUAL "amd64")
        set(SSE_SUPPORT 1)
    else()
        message(WARNING "SSE support currently only available on amd64 platforms. Turning OFF SSE support.")
        set(SSE_SUPPORT 0)
    endif()
else()
    set(SSE_SUPPORT 0)
endif()

ADD_DEFINITIONS(-DSSE_SUPPORT=${SSE_SUPPORT})

# Check whether we need to link against the rt library
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <time.h>
int main() {
    struct timespec ts;
    return clock_gettime(CLOCK_REALTIME, &ts);
}
" HAVE_CLOCK_GETTIME)

set(regena_lib_header
        include/auxillary.h
        include/arguments.h
        include/genotype.h
        include/genomult.h
        include/helper.h
        include/io.h
        include/mailman.h
        include/matmult.h
        include/printable.h
        include/std.h
        include/storage.h
        include/functions.h
        include/vectorfn.h
        include/statsfn.h
)

set(regena_lib_source
        src/auxillary.cpp
        src/genotype.cpp
        src/genomult.cpp
        src/matmult.cpp
        src/storage.cpp
        src/functions.cpp
)

add_library(regena_lib STATIC ${regena_lib_header} ${regena_lib_source})
target_include_directories(regena_lib INTERFACE include)
target_link_libraries(regena_lib PRIVATE Threads::Threads)
IF (NOT HAVE_CLOCK_GETTIME)
    target_link_libraries(regena_lib PRIVATE rt)
ENDIF ()

add_executable(REGENA src/regena.cpp)
target_link_libraries(REGENA PRIVATE regena_lib)

# Double precision. The headers test `#ifdef USE_DOUBLE`, so any definition of the
# macro selects double; REGENA is always built (and was validated) in double precision.
target_compile_definitions(regena_lib PUBLIC USE_DOUBLE=1)
