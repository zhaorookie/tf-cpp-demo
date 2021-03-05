project(tf-cpp-demo)
cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++11)

set(TENSORFLOW_ROOT_DIR /home/ubuntu/learn/tensorflow)

find_package(OpenCV REQUIRED)

# add other include dir if needed
include_directories(
    ${TENSORFLOW_ROOT_DIR}
    ${TENSORFLOW_ROOT_DIR}/bazel-include
    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen/proto
    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/eigen
    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/absl
    /usr/include/eigen3 
    /usr/local/include/tf2
    /usr/local/include/tf2/tensorflow/contrib/makefile/downloads/absl
    ${OpenCV_INCLUDE_DIRS}
)

aux_source_directory(./src DIR_SRCS)

link_directories(/home/ubuntu/learn/tensorflow/lib)

add_executable(tf-cpp-demo ${DIR_SRCS})

# 上面的link_directories 是为了能够找到libtensorflow_cc.so, libtensorflow_framework.so
target_link_libraries(tf-cpp-demo
    tensorflow_cc
    tensorflow_framework
    ${OpenCV_LIBS}
)

# 若是不用link_directories，可以这么写
#[[
set(TENSORFLOW_LIBS
    /usr/local/lib/libtensorflow_cc.so
    /usr/local/lib//libtensorflow_framework.so)

target_link_libraries(tf-cpp-demo
    ${TENSORFLOW_LIBS}
    ${OpenCV_LIBS}
)
]]
