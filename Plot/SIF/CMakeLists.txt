pybind11_add_module(Binder_SIF Binder.cpp ${SIF_OBJECTS})
target_include_directories(Binder_SIF PRIVATE ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
target_link_libraries(Binder_SIF PRIVATE "gfortran" CutestLib SIF_Instantiations) #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )
add_dependencies(Binder_SIF SIF_fortran_compile)
add_dependencies(Binder_SIF Preloader_execute)
add_dependencies(Binder_SIF Run_SIF)


