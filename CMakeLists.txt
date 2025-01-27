cmake_minimum_required(VERSION 3.20)

# This should go before project declaration.
if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif ()

project(real-time-video-stab
        VERSION 0.1
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_LIST
        src/*.cpp
        src/*.h
)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
)
