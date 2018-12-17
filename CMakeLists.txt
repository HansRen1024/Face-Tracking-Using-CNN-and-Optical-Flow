#Set minimum version requered
cmake_minimum_required(VERSION 2.4.6)
#just to avoid the warning
if(COMMAND cmake_policy)
     cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
#set project name
project(ncnn_mtcnn_tld)
#Append path to the module path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
#OpenCV
find_package(OpenCV REQUIRED)
#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/./bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/./lib)
#set the include directories
include_directories (
	${PROJECT_SOURCE_DIR}/./include
	${OpenCV_INCLUDE_DIRS}
	/home/hans/ncnn-20181019/build/install/include)
ADD_DEFINITIONS(
	-O3 -fPIC -fopenmp -std=c++11
	)
#compile tld libraries
add_library(tld_utils ${PROJECT_SOURCE_DIR}/ ./src/tld_utils.cpp)
add_library(LKTracker ${PROJECT_SOURCE_DIR}/ ./src/LKTracker.cpp)
add_library(ferNN ${PROJECT_SOURCE_DIR}/ ./src/FerNNClassifier.cpp)
add_library(tld ${PROJECT_SOURCE_DIR}/ ./src/TLD.cpp)
#compile shared library
add_library(ncnn_mtcnn_tld_so SHARED ./src/ncnn_mtcnn_tld_so.cpp)
FIND_LIBRARY(NCNN_LIB ncnn /home/hans/ncnn-20181019/build/install/lib NO_DEFAULT_PATH)
target_link_libraries(ncnn_mtcnn_tld_so tld LKTracker ferNN tld_utils ${OpenCV_LIBS} ${NCNN_LIB})
#executables
add_executable(demo ${PROJECT_SOURCE_DIR}/ ./src/demo.cpp)
#link the libraries
target_link_libraries(demo ncnn_mtcnn_tld_so ${OpenCV_LIBS})
#set optimization level 
set(CMAKE_BUILD_TYPE Release)

