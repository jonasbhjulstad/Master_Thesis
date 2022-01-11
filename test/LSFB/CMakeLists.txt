# add_executable(LSFB_Sparse LSFB_Sparse.cpp ${SIF_OBJECTS})
# target_include_directories(LSFB_Sparse PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
# target_link_libraries(LSFB_Sparse PUBLIC "gfortran" CutestLib SIF_Instantiations) #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )
# add_dependencies(LSFB_Sparse Preloader_execute)