/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 */

#ifndef _VISUALIZEDETECTEDOBJECTS_H
#define _VISUALIZEDETECTEDOBJECTS_H

#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <std_msgs/Header.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include "autoware_tracker/DetectedObject.h"
#include "autoware_tracker/DetectedObjectArray.h"

#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <Eigen/Dense> 
#define __APP_NAME__ "visualize_detected_objects"

class VisualizeDetectedObjects
{
private:
  const double arrow_height_;
  const double label_height_;
  const double label_height_forests_;
  const double object_max_linear_size_ = 50.;
  double object_speed_threshold_;
  double arrow_speed_threshold_;
  double marker_display_duration_;
  double marker_display_duration_forests;

  int marker_id_;

  std_msgs::ColorRGBA label_color_, label_color_forests,box_color_, hull_color_, arrow_color_, centroid_color_, model_color_;

  std::string input_topic_, ros_namespace_;

  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_detected_objects_;
  ros::Subscriber subscriber_detected_objects_forests;

  ros::Publisher publisher_markers_;
  ros::Publisher publisher_markers_forests;

  visualization_msgs::MarkerArray ObjectsToLabels(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToLabels_forests(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToArrows(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToBoxes(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToModels(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToModels_forests(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToHulls(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToCentroids(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray ObjectsToCentroids_forests(const autoware_tracker::DetectedObjectArray &in_objects);
  
  visualization_msgs::MarkerArray ObjectsToBoxes_forests(const autoware_tracker::DetectedObjectArray &in_objects);

  visualization_msgs::MarkerArray estimate_vis(const autoware_tracker::DetectedObjectArray &in_objects);
  visualization_msgs::Marker getMarker(std::vector<geometry_msgs::Point> vertices,std_msgs::Header header);
  std::vector<geometry_msgs::Point> getEightVertices(const autoware_tracker::DetectedObject &in_object, double rot_z, geometry_msgs::Pose pose);
  double convertRot_z2Rot_y(double rot_z);
  double quat2rot_z(const autoware_tracker::DetectedObject &in_object);
  geometry_msgs::Vector3 estimate_size(const autoware_tracker::DetectedObject &in_object);
  double estimate_rot_z(const autoware_tracker::DetectedObject &in_object,double rot_z);



  std::string ColorToString(const std_msgs::ColorRGBA &in_color);

  void DetectedObjectsCallback(const autoware_tracker::DetectedObjectArray &in_objects);

   void ForestCallback(const autoware_tracker::DetectedObjectArray &in_objects);

  bool IsObjectValid(const autoware_tracker::DetectedObject &in_object);

  float CheckColor(double value);

  float CheckAlpha(double value);

  std_msgs::ColorRGBA ParseColor(const std::vector<double> &in_color);

public:
  VisualizeDetectedObjects();
};

#endif  // _VISUALIZEDETECTEDOBJECTS_H
