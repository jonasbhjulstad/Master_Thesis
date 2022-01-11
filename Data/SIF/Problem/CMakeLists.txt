add_custom_target(Decoder ALL
    COMMAND bash -c "${CMAKE_CURRENT_SOURCE_DIR}/DimensionPreloader.sh ${SIF_PROBLEM}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running sifdecoder.."
    )
add_custom_target(SIF_fortran_compile ALL
    COMMAND bash -c "gfortran -c *.f"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS Decoder
    COMMENT "Compiling SIF-objects..")

add_custom_target(SIF_Clean ALL
    COMMAND touch probname.txt
    COMMAND echo ${SIF_PROBLEM} > probname.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    
add_executable(Dimension_Preloader Dimension_Preloader.cpp)
target_include_directories(Dimension_Preloader PUBLIC ${CUTEST_INCLUDE_PATH} ${SIF_DATA_PATH})
target_link_libraries(Dimension_Preloader PUBLIC CutestLib "-lgfortran")

add_custom_target(Preloader_execute ALL
    COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/Dimension_Preloader ${CMAKE_CURRENT_SOURCE_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/include/SIF_Dimensions/"
    DEPENDS Dimension_Preloader SIF_fortran_compile
    COMMENT "Preloading dimensions..")