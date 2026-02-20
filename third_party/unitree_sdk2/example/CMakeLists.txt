# Go2-focused, presence-guarded example list.
set(UNITREE_SDK2_EXAMPLES
  helloworld
  wireless_controller
  jsonize
  state_machine
  go2
  b2
  h1
  g1
  go2w
  b2w
  a2
)

foreach(example_name IN LISTS UNITREE_SDK2_EXAMPLES)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/CMakeLists.txt")
    add_subdirectory(${example_name})
  endif()
endforeach()
