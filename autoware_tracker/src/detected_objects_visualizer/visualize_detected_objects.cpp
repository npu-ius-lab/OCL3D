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

#include "visualize_detected_objects.h"

VisualizeDetectedObjects::VisualizeDetectedObjects() : arrow_height_(0.5), label_height_(1.0),label_height_forests_(1.5)
{
        ros::NodeHandle private_nh_;//("~");

        private_nh_.param<double>("object_speed_threshold", object_speed_threshold_, 0.1);
        ROS_INFO("[%s] object_speed_threshold: %.2f", __APP_NAME__, object_speed_threshold_);

        private_nh_.param<double>("arrow_speed_threshold", arrow_speed_threshold_, 0.25);
        ROS_INFO("[%s] arrow_speed_threshold: %.2f", __APP_NAME__, arrow_speed_threshold_);

        private_nh_.param<double>("marker_display_duration", marker_display_duration_, 0.5);
        private_nh_.param<double>("marker_display_duration", marker_display_duration_forests, 0.5);
        ROS_INFO("[%s] marker_display_duration: %.2f", __APP_NAME__, marker_display_duration_);

        std::vector<double> color;
        std::vector<double> color_forests;
        private_nh_.param<std::vector<double> >("label_color", color, {255.,255.,255.,1.0});
        private_nh_.param<std::vector<double> >("label_color_forests", color_forests, {0.,255.,0.,1.0});
        label_color_ = ParseColor(color);
        label_color_forests = ParseColor(color_forests);
        ROS_INFO("[%s] label_color: %s", __APP_NAME__, ColorToString(label_color_).c_str());
        ROS_INFO("[%s] label_color: %s", __APP_NAME__, ColorToString(label_color_forests).c_str());

        private_nh_.param<std::vector<double> >("arrow_color", color, {0.,255.,0.,0.8});
        arrow_color_ = ParseColor(color);
        ROS_INFO("[%s] arrow_color: %s", __APP_NAME__, ColorToString(arrow_color_).c_str());

        private_nh_.param<std::vector<double> >("hull_color", color, {51.,204.,51.,0.8});
        hull_color_ = ParseColor(color);
        ROS_INFO("[%s] hull_color: %s", __APP_NAME__, ColorToString(hull_color_).c_str());

        private_nh_.param<std::vector<double> >("box_color", color, {51.,128.,204.,0.8});
        box_color_ = ParseColor(color);
        ROS_INFO("[%s] box_color: %s", __APP_NAME__, ColorToString(box_color_).c_str());

        private_nh_.param<std::vector<double> >("model_color", color, {190.,190.,190.,0.5});
        model_color_ = ParseColor(color);
        ROS_INFO("[%s] model_color: %s", __APP_NAME__, ColorToString(model_color_).c_str());

        private_nh_.param<std::vector<double> >("centroid_color", color, {77.,121.,255.,0.8});
        centroid_color_ = ParseColor(color);
        ROS_INFO("[%s] centroid_color: %s", __APP_NAME__, ColorToString(centroid_color_).c_str());

        subscriber_detected_objects_ = node_handle_.subscribe("autoware_tracker/tracker/objects", 1, &VisualizeDetectedObjects::DetectedObjectsCallback, this);
        // subscriber_detected_objects_ = node_handle_.subscribe("autoware_tracker/cluster/objects_Predict", 1, &VisualizeDetectedObjects::DetectedObjectsCallback, this);

        subscriber_detected_objects_forests = node_handle_.subscribe("/online_random_forest/rf_label", 1, &VisualizeDetectedObjects::ForestCallback, this);

        publisher_markers_ = node_handle_.advertise<visualization_msgs::MarkerArray>("autoware_tracker/visualizer/objects", 1);
        publisher_markers_forests = node_handle_.advertise<visualization_msgs::MarkerArray>("autoware_tracker/visualizer/forests_objects", 1);
        

}

std::string VisualizeDetectedObjects::ColorToString(const std_msgs::ColorRGBA &in_color)
{
        std::stringstream stream;

        stream << "{R:" << std::fixed << std::setprecision(1) << in_color.r*255 << ", ";
        stream << "G:" << std::fixed << std::setprecision(1) << in_color.g*255 << ", ";
        stream << "B:" << std::fixed << std::setprecision(1) << in_color.b*255 << ", ";
        stream << "A:" << std::fixed << std::setprecision(1) << in_color.a << "}";
        return stream.str();
}

float VisualizeDetectedObjects::CheckColor(double value)
{
        float final_value;
        if (value > 255.)
                final_value = 1.f;
        else if (value < 0)
                final_value = 0.f;
        else
                final_value = value/255.f;
        return final_value;
}

float VisualizeDetectedObjects::CheckAlpha(double value)
{
        float final_value;
        if (value > 1.)
                final_value = 1.f;
        else if (value < 0.1)
                final_value = 0.1f;
        else
                final_value = value;
        return final_value;
}

std_msgs::ColorRGBA VisualizeDetectedObjects::ParseColor(const std::vector<double> &in_color)
{
        std_msgs::ColorRGBA color;
        if (in_color.size() == 4) //r,g,b,a
        {
                color.r = CheckColor(in_color[0]);
                color.g = CheckColor(in_color[1]);
                color.b = CheckColor(in_color[2]);
                color.a = CheckAlpha(in_color[3]);
        }
        return color;
}

void VisualizeDetectedObjects::DetectedObjectsCallback(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray label_markers, arrow_markers, centroid_markers, polygon_hulls, bounding_boxes,
                                        object_models;

        visualization_msgs::MarkerArray visualization_markers;

        marker_id_ = 0;

        label_markers = ObjectsToLabels(in_objects);
        arrow_markers = ObjectsToArrows(in_objects);
        polygon_hulls = ObjectsToHulls(in_objects);
        bounding_boxes = ObjectsToBoxes(in_objects);
        object_models = ObjectsToModels(in_objects);
        centroid_markers = ObjectsToCentroids(in_objects);

        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             label_markers.markers.begin(), label_markers.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             arrow_markers.markers.begin(), arrow_markers.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             polygon_hulls.markers.begin(), polygon_hulls.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             bounding_boxes.markers.begin(), bounding_boxes.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             object_models.markers.begin(), object_models.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             centroid_markers.markers.begin(), centroid_markers.markers.end());

        // std::cerr << "results: " <<visualization_markers << std::endl;

        publisher_markers_.publish(visualization_markers);

}

geometry_msgs::Vector3 VisualizeDetectedObjects::estimate_size(const autoware_tracker::DetectedObject &in_object){
        geometry_msgs::Vector3 dimensions;



        if(in_object.label == "0"){
                dimensions.x = 3.5133945;
                dimensions.y = 1.64369525;
                dimensions.z = 1.53620525;
        } else if(in_object.label == "1"){
                dimensions.x = 1.02429157;
                dimensions.y = 0.78691077;
                dimensions.z = 1.85780369;
        } else if(in_object.label == "2"){
                dimensions.x = 1.7911454;
                dimensions.y = 0.78559974;
                dimensions.z = 1.83601675;
        } else{
                std::cout<< "unknow label"<<std::endl;
        }
        return dimensions;
}

double VisualizeDetectedObjects::estimate_rot_z(const autoware_tracker::DetectedObject &in_object,double rot_z){
        double rot_z_out;
        if(in_object.dimensions.x > in_object.dimensions.y){
                rot_z_out = rot_z;
        } else{
                rot_z_out = rot_z + M_PI / 2;
        }
        
        return rot_z_out;     
}

double VisualizeDetectedObjects::quat2rot_z(const autoware_tracker::DetectedObject &in_object){
        double rot_z;
        tf::Quaternion q(in_object.pose.orientation.x,
                                        in_object.pose.orientation.y,
                                        in_object.pose.orientation.z,
                                        in_object.pose.orientation.w);
        double roll, pitch, yaw;

        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

        rot_z = yaw;
        return rot_z;
          
}

double VisualizeDetectedObjects::convertRot_z2Rot_y(double rot_z){
        double rot_y;

        rot_y = -rot_z - M_PI / 2;

        return rot_y;
}



std::vector<geometry_msgs::Point> VisualizeDetectedObjects::getEightVertices(const autoware_tracker::DetectedObject &in_object, double rot_z, geometry_msgs::Pose pose) {
        std::vector<geometry_msgs::Point> vertices(8);
        geometry_msgs::Vector3 dimensions = estimate_size(in_object);
        // 提取尺寸信息
        double l = dimensions.x;
        double w = dimensions.y;
        double h = dimensions.z;

        geometry_msgs::Vector3 dimensions_cluster;
        dimensions_cluster.x = std::max(in_object.dimensions.x, in_object.dimensions.y);
        dimensions_cluster.y = std::min(in_object.dimensions.x, in_object.dimensions.y);  
        

        std::vector<double> center = {pose.position.x,pose.position.y,pose.position.z - 0.5 * h};
        
        

        if (dimensions.x > dimensions_cluster.x){
                center[0] = pose.position.x + 0.5 * (dimensions.x - dimensions_cluster.x) * cos(rot_z);
                center[1] = pose.position.y + 0.5 * (dimensions.x - dimensions_cluster.x) * sin(rot_z);
        }    
        
        double x_corners[8] = {l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2};
        double y_corners[8] = {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2};
        double z_corners[8] = {0, 0, 0, 0, h, h, h, h};
        // Stack the corners into a 3D matrix (3, 8) in object coordinate system.

        Eigen::MatrixXd corners_3d(3, 8);
        for (int i = 0; i < 8; ++i) {
                corners_3d(0, i) = x_corners[i];
                corners_3d(1, i) = y_corners[i];
                corners_3d(2, i) = z_corners[i];
        }

        // Define the rotation matrix to transform from object coordinate to Velodyne coordinate.
        Eigen::Matrix3d R;
        R << cos(rot_z), -sin(rot_z), 0,
                sin(rot_z), cos(rot_z),  0,
                0,           0,          1;
        // Perform the coordinate transformation and translation.
        Eigen::MatrixXd transformed_corners_3d = R * corners_3d;

        for (int i = 0; i < 8; ++i) {
                transformed_corners_3d(0, i) += center[0];
                transformed_corners_3d(1, i) += center[1];
                transformed_corners_3d(2, i) += center[2];
        }



        for (int i = 0; i < 8; ++i) {
                vertices[i].x = transformed_corners_3d(0, i);
                vertices[i].y = transformed_corners_3d(1, i);
                vertices[i].z = transformed_corners_3d(2, i);
        }

    return vertices;
}

visualization_msgs::Marker VisualizeDetectedObjects::getMarker(std::vector<geometry_msgs::Point> vertices,std_msgs::Header header){
        visualization_msgs::Marker marker;
        marker.lifetime = ros::Duration(marker_display_duration_);
        marker.header = header;
        marker.ns = "rf_bounding_boxes";
        marker.id = marker_id_++;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.1;

        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0; 

        marker.points.push_back(vertices[0]);
        marker.points.push_back(vertices[1]);

        marker.points.push_back(vertices[1]);
        marker.points.push_back(vertices[2]);       
        
        marker.points.push_back(vertices[2]);
        marker.points.push_back(vertices[3]); 

        marker.points.push_back(vertices[3]);
        marker.points.push_back(vertices[0]); 

        marker.points.push_back(vertices[0]);
        marker.points.push_back(vertices[4]); 

        marker.points.push_back(vertices[1]);
        marker.points.push_back(vertices[5]);

        marker.points.push_back(vertices[2]);
        marker.points.push_back(vertices[6]); 

        marker.points.push_back(vertices[3]);
        marker.points.push_back(vertices[7]); 


        marker.points.push_back(vertices[4]);
        marker.points.push_back(vertices[5]); 

        marker.points.push_back(vertices[5]);
        marker.points.push_back(vertices[6]); 

        marker.points.push_back(vertices[6]);
        marker.points.push_back(vertices[7]);

        marker.points.push_back(vertices[7]);
        marker.points.push_back(vertices[4]);

        return marker;
}

// std::vector<geometry_msgs::Point> VisualizeDetectedObjects::estimateVertices(geometry_msgs::Vector3 dimensions, double rot_z, geometry_msgs::Pose pose){

// }
visualization_msgs::MarkerArray VisualizeDetectedObjects::estimate_vis(const autoware_tracker::DetectedObjectArray &in_objects){
        visualization_msgs::MarkerArray markers;

        for (size_t i = 0; i < in_objects.objects.size(); i++)
        {       
                autoware_tracker::DetectedObject object = in_objects.objects[i];
                visualization_msgs::Marker marker;
                

                // 四元数转欧拉角
                double rot_z = quat2rot_z(object);
                rot_z = estimate_rot_z(object,rot_z);
                std::cout << "estimated rot z is " <<rot_z << std::endl;
                // 计算八个顶点
                std::vector<geometry_msgs::Point> vertices = getEightVertices(object,rot_z,object.pose);

                marker = getMarker(vertices,in_objects.header);

                markers.markers.push_back(marker);
        }
        return markers;

}



//yao
void VisualizeDetectedObjects::ForestCallback(const autoware_tracker::DetectedObjectArray &in_objects)
{       
        visualization_msgs::MarkerArray label_markers, arrow_markers, centroid_markers, polygon_hulls, bounding_boxes,bounding_boxes_esti,object_models;

        visualization_msgs::MarkerArray visualization_markers;
        

        marker_id_ = 0;

        label_markers = ObjectsToLabels_forests(in_objects);
        centroid_markers = ObjectsToCentroids_forests(in_objects);

        bounding_boxes = ObjectsToBoxes_forests(in_objects);
        bounding_boxes_esti = estimate_vis(in_objects);

        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             label_markers.markers.begin(), label_markers.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             centroid_markers.markers.begin(), centroid_markers.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             bounding_boxes.markers.begin(), bounding_boxes.markers.end());
        visualization_markers.markers.insert(visualization_markers.markers.end(),
                                             bounding_boxes_esti.markers.begin(), bounding_boxes_esti.markers.end());
        // std::cerr << "results: " <<visualization_markers << std::endl;

        publisher_markers_forests.publish(visualization_markers);

}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToCentroids(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray centroid_markers;
        for (auto const &object: in_objects.objects)
        {
                // if (IsObjectValid(object))
                {
                        visualization_msgs::Marker centroid_marker;
                        // std::cout<<"id is "<<object.id<<",position is :"<<object.pose.position.x<<" "<<object.pose.position.y<<std::endl;
                        centroid_marker.lifetime = ros::Duration(marker_display_duration_);

                        centroid_marker.header = in_objects.header;
                        centroid_marker.type = visualization_msgs::Marker::SPHERE;
                        centroid_marker.action = visualization_msgs::Marker::ADD;
                        centroid_marker.pose = object.pose;
                        centroid_marker.ns = "centroid_markers";

                        centroid_marker.scale.x = 0.5;
                        centroid_marker.scale.y = 0.5;
                        centroid_marker.scale.z = 0.5;

                        if (object.color.a == 0)
                        {
                                centroid_marker.color = centroid_color_;
                        }
                        else
                        {
                                centroid_marker.color = object.color;
                        }
                        centroid_marker.id = marker_id_++;
                        centroid_markers.markers.push_back(centroid_marker);
                }
        }
        return centroid_markers;
}//ObjectsToCentroids

//yao
visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToCentroids_forests(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray centroid_markers;
        for (auto const &object: in_objects.objects)
        {
                // if (IsObjectValid(object))
                {
                        visualization_msgs::Marker centroid_marker;
                        centroid_marker.lifetime = ros::Duration(marker_display_duration_forests);

                        centroid_marker.header = in_objects.header;
                        centroid_marker.type = visualization_msgs::Marker::SPHERE;
                        centroid_marker.action = visualization_msgs::Marker::ADD;
                        centroid_marker.pose = object.pose;

                        // std::cout << "pose in visual " << object.pose <<std::endl;
                        centroid_marker.ns = "centroid_markers";

                        centroid_marker.scale.x = 0.5;
                        centroid_marker.scale.y = 0.5;
                        centroid_marker.scale.z = 0.5;

                        if (object.color.a == 0)
                        {
                                centroid_marker.color = centroid_color_;
                        }
                        else
                        {
                                centroid_marker.color = object.color;
                        }
                        centroid_marker.id = marker_id_++;
                        centroid_markers.markers.push_back(centroid_marker);
                }
        }
        return centroid_markers;
}//ObjectsToCentroids


visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToBoxes_forests(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray object_boxes;

        for (auto const &object: in_objects.objects)
        {
                // if (IsObjectValid(object) &&
                //     (object.pose_reliable || object.label != "9") &&
                //     (object.dimensions.x + object.dimensions.y + object.dimensions.z) < object_max_linear_size_)
                // {
                        visualization_msgs::Marker box;

                        box.lifetime = ros::Duration(marker_display_duration_);
                        box.header = in_objects.header;
                        box.type = visualization_msgs::Marker::CUBE;
                        box.action = visualization_msgs::Marker::ADD;
                        box.ns = "box_markers";
                        box.id = marker_id_++;
                        box.pose.position = object.pose.position;
                        // box.scale = object.dimensions;
                        box.pose.orientation = object.pose.orientation;
                        box.scale = estimate_size(object);
                        
                        
                        
                        tf::Quaternion q(object.pose.orientation.x,
                                        object.pose.orientation.y,
                                        object.pose.orientation.z,
                                        object.pose.orientation.w);
                        double roll, pitch, yaw;
                        
                        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
                        std::cout << "in visualize yaw is " << yaw << std::endl;
                        
                        double rot_z = estimate_rot_z(object,yaw);
                        geometry_msgs::Quaternion quat = tf::createQuaternionMsgFromYaw(rot_z);

                        box.pose.orientation = quat;
                        if (object.color.a == 0)
                        {
                                box.color = box_color_;
                        }
                        else
                        {
                                box.color = object.color;
                        }

                        object_boxes.markers.push_back(box);
                // }
        }
        return object_boxes;
}//ObjectsToBoxes

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToBoxes(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray object_boxes;

        for (auto const &object: in_objects.objects)
        {
                if (IsObjectValid(object) &&
                    (object.pose_reliable || object.label != "9") &&
                    (object.dimensions.x + object.dimensions.y + object.dimensions.z) < object_max_linear_size_)
                {
                        visualization_msgs::Marker box;

                        box.lifetime = ros::Duration(marker_display_duration_);
                        box.header = in_objects.header;
                        box.type = visualization_msgs::Marker::CUBE;
                        box.action = visualization_msgs::Marker::ADD;
                        box.ns = "box_markers";
                        box.id = marker_id_++;
                        box.scale = object.dimensions;
                        box.pose.position = object.pose.position;

                        if (object.pose_reliable)
                                box.pose.orientation = object.pose.orientation;

                        if (object.color.a == 0)
                        {
                                box.color = box_color_;
                        }
                        else
                        {
                                box.color = object.color;
                        }

                        object_boxes.markers.push_back(box);
                }
        }
        return object_boxes;
}//ObjectsToBoxes

visualization_msgs::MarkerArray
VisualizeDetectedObjects::ObjectsToModels(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray object_models;

        for (auto const &object: in_objects.objects)
        {
                if (IsObjectValid(object) &&
                    object.label != "9" &&
                    (object.dimensions.x + object.dimensions.y + object.dimensions.z) < object_max_linear_size_)
                {
                        visualization_msgs::Marker model;

                        model.lifetime = ros::Duration(marker_display_duration_);
                        model.header = in_objects.header;
                        //model.type = visualization_msgs::Marker::MESH_RESOURCE;
                        model.action = visualization_msgs::Marker::ADD;
                        model.ns = "model_markers";
                        model.mesh_use_embedded_materials = false;
                        model.color = model_color_;
                        
                        model.scale.x = 1;
                        model.scale.y = 1;
                        model.scale.z = 1;
                        model.id = marker_id_++;
                        model.pose.position = object.pose.position;
                        model.pose.position.z-= object.dimensions.z/2;

                        if (object.pose_reliable)
                                model.pose.orientation = object.pose.orientation;

                        object_models.markers.push_back(model);
                }
        }
        return object_models;
}//ObjectsToModels

visualization_msgs::MarkerArray
VisualizeDetectedObjects::ObjectsToHulls(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray polygon_hulls;

        for (auto const &object: in_objects.objects)
        {
                // if (IsObjectValid(object) && !object.convex_hull.polygon.points.empty() && object.label == "9")
                if (IsObjectValid(object) && !object.convex_hull.polygon.points.empty())
                {
                        visualization_msgs::Marker hull;
                        hull.lifetime = ros::Duration(marker_display_duration_);
                        hull.header = in_objects.header;
                        hull.type = visualization_msgs::Marker::LINE_STRIP;
                        hull.action = visualization_msgs::Marker::ADD;
                        hull.ns = "hull_markers";
                        hull.id = marker_id_++;
                        hull.scale.x = 0.2;

                        
                        for(auto const &point: object.convex_hull.polygon.points)
                        {
                                geometry_msgs::Point tmp_point;
                                tmp_point.x = point.x;
                                tmp_point.y = point.y;
                                tmp_point.z = point.z;
                                hull.points.push_back(tmp_point);
                        }

                        if (object.color.a == 0)
                        {
                                hull.color = hull_color_;
                        } 
                        else
                        {
                                hull.color = object.color;
                        }

                        if(!object.label.empty())
                                if(object.label == "0") {
                                        hull.text = "Car ";
                                } else if(object.label == "1") {
                                        hull.text = "Pedestrian ";
                                } else if(object.label == "2") {
                                        hull.text = "Cyclist ";
                                } else if(object.label == "9") {
                                        hull.text = "unknown ";
                                }

                        if(!object.label.empty()){
                                  std::string text = "\n<" + std::to_string(object.id) + ">" ;
                                  hull.text += text;
                                }

                        polygon_hulls.markers.push_back(hull);
                }

                
        }
        // std::cout << "################"<<polygon_hulls<< "################" << std::endl;
        return polygon_hulls;
}

visualization_msgs::MarkerArray
VisualizeDetectedObjects::ObjectsToArrows(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray arrow_markers;
        for (auto const &object: in_objects.objects)
        {
                if (IsObjectValid(object) && object.pose_reliable)
                {
                        double velocity = object.velocity.linear.x;

                        if (abs(velocity) >= arrow_speed_threshold_)
                        {
                                visualization_msgs::Marker arrow_marker;
                                arrow_marker.lifetime = ros::Duration(marker_display_duration_);

                                tf::Quaternion q(object.pose.orientation.x,
                                                 object.pose.orientation.y,
                                                 object.pose.orientation.z,
                                                 object.pose.orientation.w);
                                double roll, pitch, yaw;

                                tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

                                // in the case motion model fit opposite direction
                                if (velocity < -0.1)
                                {
                                        yaw += M_PI;
                                        // normalize angle
                                        while (yaw > M_PI)
                                                yaw -= 2. * M_PI;
                                        while (yaw < -M_PI)
                                                yaw += 2. * M_PI;
                                }

                                tf::Matrix3x3 obs_mat;
                                tf::Quaternion q_tf;

                                obs_mat.setEulerYPR(yaw, 0, 0); // yaw, pitch, roll
                                obs_mat.getRotation(q_tf);

                                arrow_marker.header = in_objects.header;
                                arrow_marker.ns = "arrow_markers";
                                arrow_marker.action = visualization_msgs::Marker::ADD;
                                arrow_marker.type = visualization_msgs::Marker::ARROW;

                                // green
                                if (object.color.a == 0)
                                {
                                        arrow_marker.color = arrow_color_;
                                }
                                else
                                {
                                        arrow_marker.color = object.color;
                                }
                                arrow_marker.id = marker_id_++;

                                // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
                                arrow_marker.pose.position.x = object.pose.position.x;
                                arrow_marker.pose.position.y = object.pose.position.y;
                                arrow_marker.pose.position.z = arrow_height_;

                                arrow_marker.pose.orientation.x = q_tf.getX();
                                arrow_marker.pose.orientation.y = q_tf.getY();
                                arrow_marker.pose.orientation.z = q_tf.getZ();
                                arrow_marker.pose.orientation.w = q_tf.getW();

                                // Set the scale of the arrow -- 1x1x1 here means 1m on a side
                                arrow_marker.scale.x = 3;
                                arrow_marker.scale.y = 0.1;
                                arrow_marker.scale.z = 0.1;

                                arrow_markers.markers.push_back(arrow_marker);
                        }//velocity threshold
                }//valid object
        }//end for
        return arrow_markers;
}//ObjectsToArrows

visualization_msgs::MarkerArray
VisualizeDetectedObjects::ObjectsToLabels(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray label_markers;
        for (auto const &object: in_objects.objects)
        {
                if (IsObjectValid(object))
                {
                        visualization_msgs::Marker label_marker;

                        label_marker.lifetime = ros::Duration(marker_display_duration_);
                        label_marker.header = in_objects.header;
                        label_marker.ns = "label_markers";
                        label_marker.action = visualization_msgs::Marker::ADD;
                        label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                        label_marker.scale.x = 1.5;
                        label_marker.scale.y = 1.5;
                        label_marker.scale.z = 1.5;

                        label_marker.color = label_color_;

                        label_marker.id = marker_id_++;

                        // Object Class if available
                        if(!object.label.empty() && object.label != "9")
                                if(object.label == "0") {
                                        label_marker.text = "Car ";
                                } else if(object.label == "1") {
                                        label_marker.text = "Pedestrian ";
                                } else if(object.label == "2") {
                                        label_marker.text = "Cyclist ";
                                }


                        std::stringstream distance_stream;
                        distance_stream << std::fixed << std::setprecision(1)
                                        << sqrt((object.pose.position.x * object.pose.position.x) +
                                (object.pose.position.y * object.pose.position.y));
                        std::string distance_str = distance_stream.str() + " m";
                        //label_marker.text += distance_str;

                        if (object.velocity_reliable)
                        {
                                double velocity = object.velocity.linear.x;
                                if (velocity < -0.1)
                                {
                                        velocity *= -1;
                                }

                                if (abs(velocity) < object_speed_threshold_)
                                {
                                        velocity = 0.0;
                                }

                                tf::Quaternion q(object.pose.orientation.x, object.pose.orientation.y,
                                                 object.pose.orientation.z, object.pose.orientation.w);

                                double roll, pitch, yaw;
                                tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

                                
                                // convert m/s to km/h
                                std::stringstream kmh_velocity_stream;
                                kmh_velocity_stream << std::fixed << std::setprecision(1) << (velocity * 3.6);
                                // std::string text = "\n<" + std::to_string(object.id) + "> " + kmh_velocity_stream.str() + " km/h";
                                // std::string text = "\n<" + std::to_string(object.id) + "> ";
                                // label_marker.text += text;
                                if(!object.label.empty() && object.label != "9"){
                                  std::string text = "\n<" + std::to_string(object.id) + ">" ;
                                  label_marker.text += text;

                                }
                        }

                        label_marker.pose.position.x = object.pose.position.x;
                        label_marker.pose.position.y = object.pose.position.y;
                        label_marker.pose.position.z = label_height_;
                        label_marker.scale.z = 1.0;
                        if (!label_marker.text.empty())
                                label_markers.markers.push_back(label_marker);
                }
        } // end in_objects.objects loop

        return label_markers;
}//ObjectsToLabels
//yao
visualization_msgs::MarkerArray
VisualizeDetectedObjects::ObjectsToLabels_forests(const autoware_tracker::DetectedObjectArray &in_objects)
{
        visualization_msgs::MarkerArray label_markers;
        for (auto const &object: in_objects.objects)
        {
                // if (IsObjectValid(object))
                {       
                        visualization_msgs::Marker label_marker;

                        label_marker.lifetime = ros::Duration(marker_display_duration_forests);
                        label_marker.header = in_objects.header;
                        label_marker.ns = "label_markers";
                        label_marker.action = visualization_msgs::Marker::ADD;
                        label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                        label_marker.scale.x = 0.5;
                        label_marker.scale.y = 0.5;
                        label_marker.scale.z = 0.5;
                        
                        label_marker.color = label_color_forests;

                        label_marker.id = marker_id_++;

                        // Object Class if available
                        if(!object.label.empty() && object.label != "9.0")
                                if(object.label == "0") {
                                        label_marker.text = "Car ";
                                } else if(object.label == "1") {
                                        label_marker.text = "Pedestrian ";
                                } else if(object.label == "2") {
                                        label_marker.text = "Cyclist ";
                                }

                                tf::Quaternion q(object.pose.orientation.x,
                                        object.pose.orientation.y,
                                        object.pose.orientation.z,
                                        object.pose.orientation.w);
                                double roll, pitch, yaw;

                                tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

                                
                                if(!object.label.empty() && object.label != "9"){
                                        std::string text = "\n<" + std::to_string(object.id) + ">";
                                        label_marker.text += text;
                                        std::string text_yaw = "\n" + std::to_string(yaw * 57.3);
                                        label_marker.text += text_yaw;
                                        std::string text_dims = "\n" + std::to_string(object.dimensions.x) + " " + std::to_string(object.dimensions.y) + " "+ std::to_string(object.dimensions.z);
                                        label_marker.text += text_dims;
                                        std::string result_file_path = "/home/lbh/online_learning_ws_mini/src/autoware_tracker/src/detected_objects_visualizer/orient.txt";
                                        std::ofstream outputfile(result_file_path, std::ofstream::out | std::ofstream::app);
                                        outputfile << std::to_string(yaw * 57.3) + " "
                                        << std::to_string(object.dimensions.x) + " " + std::to_string(object.dimensions.y) + " "+ std::to_string(object.dimensions.z) + " "
                                        << object.label + "\n";
                                        outputfile.close();
                                }
                                

                        label_marker.pose.position.x = object.pose.position.x;
                        label_marker.pose.position.y = object.pose.position.y;
                        label_marker.pose.position.z = label_height_forests_;

                        
                        

                        // label_marker.scale.z = 1.0;
                        if (!label_marker.text.empty())
                                label_markers.markers.push_back(label_marker);
                }
        } // end in_objects.objects loop

        return label_markers;
}//ObjectsToLabels


bool VisualizeDetectedObjects::IsObjectValid(const autoware_tracker::DetectedObject &in_object)
{
        if (!in_object.valid ||
            std::isnan(in_object.pose.orientation.x) ||
            std::isnan(in_object.pose.orientation.y) ||
            std::isnan(in_object.pose.orientation.z) ||
            std::isnan(in_object.pose.orientation.w) ||
            std::isnan(in_object.pose.position.x) ||
            std::isnan(in_object.pose.position.y) ||
            std::isnan(in_object.pose.position.z) ||
            (in_object.pose.position.x == 0.) ||
            (in_object.pose.position.y == 0.) ||
            (in_object.dimensions.x <= 0.) ||
            (in_object.dimensions.y <= 0.) ||
            (in_object.dimensions.z <= 0.)
            )
        {
                return false;
        }
        return true;
}//end IsObjectValid
