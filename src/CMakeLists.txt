add_library(SIF_Instantiations STATIC SIF_Instantiator.cpp ${SIF_Objects})
# target_include_directories(SIF_Instantiations PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH} ${FIPOPT_INCLUDE})
# target_link_libraries(SIF_Instantiations PUBLIC CutestLib "-lgfortran") #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )

SUBDIRLIST(FIPOPT_SRC_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(folder ${FIPOPT_SRC_FOLDERS})
    add_subdirectory(${folder})
ENDFOREACH()


