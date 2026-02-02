
# Python CMakeLists.txt

# We assume this is called via add_subdirectory from the parent C++ project
# So we have access to variables and targets like 'websiftgpu_core' and 'pybind11'.

# Verify we have pybind11
if(NOT TARGET pybind11::module)
    message(FATAL_ERROR "pybind11 not found. This project should be built from the root c++/CMakeLists.txt")
endif()

# Define the python module
# Note: linking websiftgpu_core (OBJECT lib) gives us usage requirements (includes, libs)
# But we need to explicitly include the objects to compile them into the module.
pybind11_add_module(websiftgpu_py 
    src/python_bindings.cpp 
)

# Link dependencies (propagates webgpu, threads, system libs)
target_link_libraries(websiftgpu_py PRIVATE websiftgpu_core)

# Helper to copy the DLL (defined in parent)
if(COMMAND target_copy_webgpu_binaries)
    target_copy_webgpu_binaries(websiftgpu_py)
else()
    message(WARNING "target_copy_webgpu_binaries function not found. WGPU DLL might not be copied.")
endif()

# Output directory: explicitly put it in the python/ root (where this CMakeLists.txt is)
# CMAKE_CURRENT_SOURCE_DIR is python/. 
set_target_properties(websiftgpu_py PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Copy wgpu_native.dll to this directory as well
# We can use the logic from parent but adapted
if(WGPU_LIB)
    # WGPU_LIB is likely the .lib file. We want the .dll.
    # We can infer it or use wgpu_native_SOURCE_DIR
    add_custom_command(TARGET websiftgpu_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll"
        "$<TARGET_FILE_DIR:websiftgpu_py>"
    )
endif()
