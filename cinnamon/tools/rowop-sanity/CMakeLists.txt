project(rowop-sanity)

add_executable(${PROJECT_NAME}
  rowop-sanity.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    BitsCodegen
    LLVMSupport
)
