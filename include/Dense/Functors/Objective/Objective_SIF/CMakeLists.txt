# add_library(lib_Objective_SIF INTERFACE)

# target_link_libraries(Load_parameters PRIVATE "-lgfortran" ${SIF_PROBLEM_STATIC_LIB_PATH})

# add_custom_target(Run_Load_parameters
#   COMMAND Load_parameters
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMENT "Loading SIF-parameters to header-file in ${CMAKE_CURRENT_SOURCE_DIR}"
#   # SOURCES ${Run_Load_parameters_SOURCES}
# )

# target_link_libraries(lib_Objective_SIF INTERFACE lib_Objective_Memoized)
