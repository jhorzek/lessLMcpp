cmake_minimum_required(VERSION 3.20.0)

# define project:
project(lessLMcpp VERSION 1.0)

# specify dependencies:
find_package(Armadillo REQUIRED)

add_subdirectory("external/lessOptimizers")

# set compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add all src files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# create target:
add_executable(${PROJECT_NAME} ${SRC_FILES})

# include armadillo
target_include_directories(${PROJECT_NAME} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ARMADILLO_LIBRARIES})

# include lessOptimizers
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    lessOptimizers)
