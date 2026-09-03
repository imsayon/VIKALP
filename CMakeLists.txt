cmake_minimum_required(VERSION 3.20)
project(vikalp LANGUAGES CXX)

add_library(vikalp_contracts src/contracts/Model.cpp)
target_include_directories(vikalp_contracts PUBLIC include)
target_compile_features(vikalp_contracts PUBLIC cxx_std_20)

include(CTest)
if(BUILD_TESTING)
    add_executable(vikalp_contracts_test tests/contracts/contracts_test.cpp)
    target_link_libraries(vikalp_contracts_test PRIVATE vikalp_contracts)
    add_test(NAME contracts COMMAND vikalp_contracts_test)
endif()
