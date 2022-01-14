cmake_minimum_required(VERSION 3.10)
project(FIPOPT)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
set(FIPOPT_INCLUDE "${PROJECT_SOURCE_DIR}/include")
find_package(pybind11 REQUIRED)
set(Python_ADDITIONAL_VERSIONS 3.4)
find_package(PythonLibs REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUTEST_STATIC_LIB_PATH "$ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a")
set(CUTEST_INCLUDE_PATH "$ENV{CUTEST}/include")
set(SIF_DATA_PATH "${PROJECT_SOURCE_DIR}/Data/SIF/Problem")

set(CMAKE_CXX_FLAGS_DEBUG "-g -fPIC -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-msse2 -mfpmath=sse -march=native -O3")
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



