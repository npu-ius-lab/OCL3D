/**
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, Zhi Yan
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/String.h>
#include "point_cloud_features/point_cloud_features.h"
#include "autoware_tracker/DetectedObjectArray.h"
#include <ctime>
#include "pointnet_3d_box_stamped/PointNet3DBoxStamped.h"
#include "pointnet_3d_box_stamped/PointNet3DBoxStampedArray.h"
static ros::Subscriber object_sub;
static ros::Publisher features_pub;


void Callback(const autoware_tracker::DetectedObjectArray::ConstPtr& objects_msg){
                // std::cout << "!!!!!!!!!! in feature extract frame_out !!!!!!!!!!!!!!"<< objects_msg->frame_out  << std::endl;
                // std::cout << "!!!!!!!!!! in feature extract seq !!!!!!!!!!!!!!"<< objects_msg->header.seq  << std::endl;
                // std::cout << "!!!!!!!!!! in feature extract objects[i] !!!!!!!!!!!!!!"<< objects_msg->objects[0].header.seq  << std::endl;
                pointnet_3d_box_stamped::PointNet3DBoxStampedArray features_msg_array;
                // std_msgs::String features_msg;

                int number_of_samples = 0;
                std::vector<float> covariance, moit, slice, intensity, features_dig;
                int minimum_points=5; // The minimum points that a cluster should contain, e.g. 3 for PCA.
                bool number_of_points=true, min_distance=true, covariance_mat3D=true, normalized_MOIT=true, slice_feature=true, intensity_distribution=true;
                int number_of_samples_count = 0;
                int number_of_car_count = 0;
                int number_of_ped_count = 0;
                int number_of_cyc_count = 0;
                ros::Rate loop_rate(10); 
                clock_t start = clock(); 
                srand(time(0)); // Use the time function to get a "seed” value for srand
                
                for(int i = 0; i < objects_msg->objects.size(); i++) {
                        if(objects_msg->objects[i].pointcloud.data.size()/32 >= minimum_points) {
                                
                                pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
                                pcl::fromROSMsg(objects_msg->objects[i].pointcloud, *pc);
                                covariance.clear();
                                moit.clear();
                                slice.clear();
                                intensity.clear();
                                features_dig.clear();
                                if(number_of_points) {
                                        features_dig.push_back(numberOfPoints(pc));
                                }
                                if(min_distance) {
                                        features_dig.push_back(minDistance(pc));
                                }
                                if(covariance_mat3D) {
                                        covarianceMat3D(pc, covariance);
                                        features_dig.insert(features_dig.end(), covariance.begin(), covariance.end());
                                }
                                if(normalized_MOIT) {
                                        normalizedMOIT(pc, moit);
                                        features_dig.insert(features_dig.end(), moit.begin(), moit.end());
                                }
                                if(slice_feature) {
                                        sliceFeature(pc, 10, slice);
                                        features_dig.insert(features_dig.end(), slice.begin(), slice.end());
                                }
                                if(intensity_distribution) {
                                        intensityDistribution(pc, 25, intensity);
                                        features_dig.insert(features_dig.end(), intensity.begin(), intensity.end());
                                }



                                if (objects_msg->objects[i].label != "9"){
                                        pointnet_3d_box_stamped::PointNet3DBoxStamped features_msg;
                                        features_msg.label = objects_msg->objects[i].label;
                                        features_msg.id = objects_msg->objects[i].id;
                                        features_msg.pose = objects_msg->objects[i].pose;
                                        features_msg.header = objects_msg->objects[i].header;
                                        features_msg.dimensions = objects_msg->objects[i].dimensions;
                                        features_msg.score = objects_msg->objects[i].score;
                                        features_msg.flag = objects_msg->objects[i].flag;
                                        
                                        for(int j = 0; j < features_dig.size(); j++) {
                                                features_msg.features.push_back(features_dig[j]);
                                
                                        }


                                        if(objects_msg->objects[i].label.compare("0") == 0) {
                                                number_of_car_count++;
                                        } else if (objects_msg->objects[i].label.compare("1") == 0) {
                                                number_of_ped_count++;
                                        } else if (objects_msg->objects[i].label.compare("2") == 0) {
                                                number_of_cyc_count++;
                                        }
                                        number_of_samples_count++;
                                        number_of_samples++;
                                        features_msg_array.fea_boxes.push_back(features_msg);
                                }
                                
                        }
                }
                
                features_msg_array.header = objects_msg->header;

                features_msg_array.number_of_samples = number_of_samples;
                features_msg_array.fea_dimensions = features_dig.size();
                features_msg_array.Classes = 3;
                features_msg_array.FeatureMinIndex = 1;

                features_msg_array.frame_out =  objects_msg->frame_out;
                // std::cerr << features_msg_array.header.stamp << std::endl;
                // std::cout << "******in point cloud feature features_msg_array.header.seq****** " << features_msg_array.header.seq<< std::endl;
                // std::cout << "******in point cloud feature features_msg_array.fea_boxes[0].header.seq****** " << features_msg_array.fea_boxes[0].header.seq<< std::endl;        
                // std::cerr << features_msg_array.fea_boxes[0].header.stamp << std::endl;        
                features_pub.publish(features_msg_array);
                loop_rate.sleep();
        }  
int main(int argc, char **argv) {
        autoware_tracker::DetectedObjectArray::ConstPtr objects_msg;
        ros::init(argc, argv, "point_cloud_features_");
        ros::NodeHandle private_nh("~");
        features_pub = private_nh.advertise<pointnet_3d_box_stamped::PointNet3DBoxStampedArray>("features_global", 100); 
        object_sub = private_nh.subscribe("/autoware_tracker/cluster/objects", 100, Callback);
        
        ros::spin();
}

