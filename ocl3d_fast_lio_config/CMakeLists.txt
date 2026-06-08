cmake_minimum_required(VERSION 2.8.3)
project(ocl3d_fast_lio_config)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
