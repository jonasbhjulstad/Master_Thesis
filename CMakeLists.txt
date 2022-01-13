cmake_minimum_required(VERSION 3.10)
project(FIPOPT)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
set(FIPOPT_INCLUDE "${PROJECT_SOURCE_DIR}/include")
find_package(pybind11 REQUIRED)
set(Python_ADDITIONAL_VERSIONS 3.4)
find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUTEST_STATIC_LIB_PATH "$ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a")
set(CUTEST_INCLUDE_PATH "$ENV{CUTEST}/include")
set(SIF_DATA_PATH "${PROJECT_SOURCE_DIR}/Data/SIF/Problem")

set(SIF_DECODER_FLAGS "")

set(CMAKE_CXX_FLAGS_DEBUG "-g -fPIC -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-msse2 -mfpmath=sse -march=native -O3 -fPIC -lgfortran")
add_library(CutestLib STATIC IMPORTED GLOBAL)
set_target_properties(CutestLib PROPERTIES
    IMPORTED_LOCATION ${CUTEST_STATIC_LIB_PATH})

    if (NOT SIF_PROBLEM)
    set(SIF_PROBLEM "HS12.SIF")
    endif()

set(SIF_OBJECTS "${SIF_DATA_PATH}/ELFUN.o" "${SIF_DATA_PATH}/EXTER.o" "${SIF_DATA_PATH}/GROUP.o" "${SIF_DATA_PATH}/RANGE.o")

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/Data" ${CUTEST_INCLUDE_PATH} "${PROJECT_SOURCE_DIR}/extern" "${PROJECT_SOURCE_DIR}/extern/pybind11/include")
link_libraries("gfortran" Eigen3::Eigen)

#Macro:
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

add_subdirectory(Data)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(Plot)

# first we can indicate the documentation build as an option and set it to ON by default
# option(BUILD_DOC "Build documentation" ON)

# # check if Doxygen is installed
# find_package(Doxygen)
# if (DOXYGEN_FOUND)
#     # set input and output files
#     set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
#     set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#     # request to configure the file
#     configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#     message("Doxygen build started")

#     # note the option ALL which allows to build the docs together with the application
#     add_custom_target( doc_doxygen ALL
#         COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Generating API documentation with Doxygen"
#         VERBATIM )
# else (DOXYGEN_FOUND)
#   message("Doxygen need to be installed to generate the doxygen documentation")
# endif (DOXYGEN_FOUND)


