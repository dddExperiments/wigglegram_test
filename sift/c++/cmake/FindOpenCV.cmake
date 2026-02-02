# Helper module to find OpenCV on Windows if not already found
# It searches common installation paths and sets OpenCV_DIR accordingly.

if(NOT OpenCV_FOUND)
    set(OPENCV_SEARCH_PATHS
        "C:/opencv/build"
        "C:/tools/opencv/build"
        "C:/Program Files/opencv/build"
        "D:/opencv/build"
        $ENV{OpenCV_DIR}
    )

    foreach(path ${OPENCV_SEARCH_PATHS})
        if(EXISTS "${path}/OpenCVConfig.cmake")
            message(STATUS "Found OpenCVConfig.cmake in ${path}")
            set(OpenCV_DIR "${path}" CACHE PATH "Path to OpenCV build directory" FORCE)
            break()
        endif()
    endforeach()
endif()

# Delegate to standard config search
find_package(OpenCV CONFIG QUIET)

if(OpenCV_FOUND)
    set(OpenCV_FOUND TRUE)
    message(STATUS "OpenCV found: ${OpenCV_DIR}")
else()
    if(OpenCV_FIND_REQUIRED)
        message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR to the directory containing OpenCVConfig.cmake")
    endif()
endif()
