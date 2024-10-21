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

int main(int argc, char **argv) {
        int minimum_points; // The minimum points that a cluster should contain, e.g. 3 for PCA.
        bool number_of_points, min_distance, covariance_mat3D, normalized_MOIT, slice_feature, intensity_distribution;
        std::vector<float> covariance, moit, slice, intensity, features_dig;
        autoware_tracker::DetectedObjectArray::ConstPtr objects_msg;

        ros::init(argc, argv, "point_cloud_features");
        ros::NodeHandle private_nh("~");

        ros::Publisher features_pub = private_nh.advertise<std_msgs::String>("features", 100, false); // c.f. https://github.com/amirsaffari/online-random-forests#data-format

        private_nh.param<int>("minimum_points", minimum_points, 5);
        private_nh.param<bool>("number_of_points", number_of_points, true);
        private_nh.param<bool>("min_distance", min_distance, true);
        private_nh.param<bool>("covariance_mat3D", covariance_mat3D, true);
        private_nh.param<bool>("normalized_MOIT", normalized_MOIT, true);
        private_nh.param<bool>("slice_feature", slice_feature, true);
        private_nh.param<bool>("intensity_distribution", intensity_distribution, true);

        int number_of_samples_count = 0;
        int number_of_car_count = 0;
        int number_of_ped_count = 0;
        int number_of_cyc_count = 0;

        while (ros::ok()) {
                objects_msg = ros::topic::waitForMessage<autoware_tracker::DetectedObjectArray>("/autoware_tracker/tracker/examples"); // process blocked waiting
                
                std_msgs::String features_msg;
                int number_of_samples = 0;

                double start_time,current_time;
                start_time = ros::WallTime::now().toSec();
                srand(time(0)); // Use the time function to get a "seed” value for srand

                for(int i = 0; i < objects_msg->objects.size(); i++) {
                        if(objects_msg->objects[i].pointcloud.data.size()/32 >= minimum_points) {
                                if(objects_msg->objects[i].label.compare("unknown") == 0) continue;

                                // downsampling for training
             
                                
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

                 
                                // features_msg.data += objects_msg->objects[i].label + std::to_string(objects_msg->objects[i].id); // 0:car, 1:pedestrian, 2:cyclist !!!!!!!!
                                if (objects_msg->objects[i].label != "9"){
                                features_msg.data += objects_msg->objects[i].label ;
                                for(int j = 0; j < features_dig.size(); j++) {
                                        features_msg.data += " " + std::to_string(j+1) + ":" + std::to_string(features_dig[j]);
                                }


                                features_msg.data += "\n";

                                if(objects_msg->objects[i].label.compare("0") == 0) {
                                        number_of_car_count++;
                                } else if (objects_msg->objects[i].label.compare("1") == 0) {
                                        number_of_ped_count++;
                                } else if (objects_msg->objects[i].label.compare("2") == 0) {
                                        number_of_cyc_count++;
                                }
                                number_of_samples_count++;
                                number_of_samples++;
                                }
                        }
                }

                features_msg.data.insert(0, std::to_string(number_of_samples) + " " + std::to_string(features_dig.size()) + " 3 1\n"); // Samples + Features + Classes + FeatureMinIndex
                if(number_of_samples > 0) {
                        features_pub.publish(features_msg);
                }
                current_time = ros::WallTime::now().toSec();
                std::cout << "Pointcloud feature extractor: " << current_time - start_time << "s" << std::endl;


                ros::spinOnce();
        }

        return EXIT_SUCCESS;
}
