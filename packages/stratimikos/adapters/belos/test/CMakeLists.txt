
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EpetraExt)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_ThyraEpetraAdapters)
IF (${PACKAGE_NAME}_ENABLE_EpetraExt AND ${PACKAGE_NAME}_ENABLE_ThyraEpetraAdapters)
  ADD_SUBDIRECTORIES(MVOPTester LOWSFactory)
ENDIF()
