SUBDIRLIST(FIPOPT_TEST_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(folder ${FIPOPT_TEST_FOLDERS})
    add_subdirectory(${folder})
ENDFOREACH()

