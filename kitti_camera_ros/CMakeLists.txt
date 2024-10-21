cmake_minimum_required(VERSION 2.8.3)
project(kitti_camera_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
  vision_msgs
  cv_bridge
  )

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}${OpenCV_INCLUDE_DIRS})
# include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

catkin_package()

add_executable(kitti_camera_ros src/kitti_camera_ros.cpp)
target_link_libraries(kitti_camera_ros ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
# target_link_libraries(kitti_camera_ros ${catkin_LIBRARIES} ${PCL_LIBRARIES})
if(catkin_EXPORTED_TARGETS)
  add_dependencies(kitti_camera_ros ${catkin_EXPORTED_TARGETS})
endif()
