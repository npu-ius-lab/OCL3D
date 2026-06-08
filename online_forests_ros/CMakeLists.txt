cmake_minimum_required(VERSION 2.8.3)
project(online_forests_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  autoware_tracker
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

catkin_package(
  INCLUDE_DIRS include
  )

include(CheckIncludeFileCXX)
check_include_file_cxx(gmm/gmm.h HAVE_GMM)
check_include_file_cxx(libconfig.h++ HAVE_LIBCONFIG)

if(HAVE_GMM AND HAVE_LIBCONFIG)
  add_executable(online_forests_ros
    src/online_forests/classifier.cpp
    src/online_forests/hyperparameters.cpp
    src/online_forests/onlinenode.cpp
    src/online_forests/onlinetree.cpp
    src/online_forests/utilities.cpp
    src/online_forests/data.cpp
    #  src/online_forests/Online-Forest.cpp
    src/online_forests/onlinerf.cpp
    src/online_forests/randomtest.cpp
    src/online_forests_ros.cpp
    )

  target_link_libraries(online_forests_ros
    ${catkin_LIBRARIES}
    config++
    blas
    )

  if(catkin_EXPORTED_TARGETS)
    add_dependencies(online_forests_ros
      ${catkin_EXPORTED_TARGETS}
      )
  endif()
else()
  message(WARNING "Skipping C++ online_forests_ros target because gmm and/or libconfig headers are missing. The Python IMF_iusl.py node is still available.")
endif()

catkin_install_python(PROGRAMS
  scripts/IMF_iusl.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
