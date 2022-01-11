# add_executable(QP_constrained QP_constrained_test.cpp)
# add_executable(objective_SIF objective_SIF_test.cpp
# $ENV{MASTSIF}/ELFUN.o 
# $ENV{MASTSIF}/GROUP.o 
# $ENV{MASTSIF}/RANGE.o)

# target_link_libraries(objective_SIF PUBLIC "$ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a" "-lgfortran")
# target_link_libraries(objective_constrained_SIF PUBLIC gfortran CutestLib)


add_executable(SIF_test objective_SIF_test.cpp ${SIF_OBJECTS})
target_include_directories(SIF_test PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
target_link_libraries(SIF_test PUBLIC "gfortran" CutestLib SIF_Instantiations) #"-L $ENV{CUTEST}/objects/pc64.lnx.gfo/double/libcutest.a"  "libcutest" )
add_dependencies(SIF_test Preloader_execute)