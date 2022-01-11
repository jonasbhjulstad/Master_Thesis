# add_executable(Cutest_c Cutest_constrained_test.cpp)

# add_executable(Cutest_Loader Cutest_SIF_Loader_test.cpp ELFUN.o RANGE.o GROUP.o)
# target_link_libraries(Cutest_Loader PUBLIC CutestLib "-lgfortran" Eigen3::Eigen lib_Common)


# add_executable(Cutest_constrained_test Cutest_constrained_test.cpp ${SIF_OBJECTS})
# target_include_directories(Cutest_constrained_test PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
# target_link_libraries(Cutest_constrained_test PUBLIC "gfortran" CutestLib SIF_Instantiations) #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )
# add_dependencies(Cutest_constrained_test Preloader_execute)