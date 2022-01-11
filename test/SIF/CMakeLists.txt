add_executable(Run_SIF Run_SIF.cpp ${SIF_OBJECTS})
target_include_directories(Run_SIF PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
target_link_libraries(Run_SIF PUBLIC "gfortran" CutestLib SIF_Instantiations) #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )
add_dependencies(Run_SIF Preloader_execute)