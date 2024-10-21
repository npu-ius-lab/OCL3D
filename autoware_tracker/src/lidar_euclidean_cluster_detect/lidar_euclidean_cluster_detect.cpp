#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>

#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl-1.8/pcl/sample_consensus/method_types.h>
#include <pcl-1.8/pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

#include "autoware_tracker/Centroids.h"
#include "autoware_tracker/CloudCluster.h"
#include "autoware_tracker/CloudClusterArray.h"
#include "autoware_tracker/DetectedObject.h"
#include "autoware_tracker/DetectedObjectArray.h"

// #include <vector_map/vector_map.h>

#include <tf/tf.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>

#if (CV_MAJOR_VERSION == 3)
#include "gencolors.cpp"
#else
#include <opencv2/contrib/contrib.hpp>
#include <autoware_tracker/DetectedObjectArray.h>
#endif

#include "cluster.h"

// yang21itsc
#include <vision_msgs/Detection2DArray.h>
#include <message_filters/subscriber.h>
// #include <message_filters/cache.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <algorithm>
#include "../libkitti/kitti.h"
// yang21itsc

#define __APP_NAME__ "euclidean_clustering"

// yang21itsc
Calibration *calib;
float iou_threshold = 0.5;
// yang21itsc

#include <dirent.h>

using namespace cv;
bool iskitti;
ros::Publisher _pub_cluster_cloud;
ros::Publisher _pub_ground_cloud;
ros::Publisher _centroid_pub;

ros::Publisher _pub_clusters_message;

ros::Publisher _pub_points_lanes_cloud;

ros::Publisher _pub_detected_objects;

std_msgs::Header _velodyne_header;

std::string _output_frame;

static bool _downsample_cloud;
static bool _pose_estimation;
static double _leaf_size;
static int _cluster_size_min;
static int _cluster_size_max;
static const double _initial_quat_w = 1.0;

static bool _remove_ground;  // only ground

bool use_callback = true;
bool use_camera = true;

static bool _use_diffnormals;

static double _clip_min_height;
static double _clip_max_height;

static bool _keep_lanes;
static double _keep_lane_left_distance;
static double _keep_lane_right_distance;

static double _remove_points_min;
static double _remove_points_max;
static double _cluster_merge_threshold;
static double _clustering_distance;

static std::chrono::system_clock::time_point _start, _end;

std::vector<std::vector<geometry_msgs::Point> > _way_area_points;
std::vector<cv::Scalar> _colors;
pcl::PointCloud<pcl::PointXYZI> _sensor_cloud;
visualization_msgs::Marker _visualization_marker;

static bool _use_multiple_thres;
std::vector<double> _clustering_distances;
std::vector<double> _clustering_ranges;

tf::StampedTransform *_transform;
tf::StampedTransform *_velodyne_output_transform;
tf::TransformListener *_transform_listener;
tf::TransformListener *_vectormap_transform_listener;

int count = -1;

tf::StampedTransform findTransform(const std::string &in_target_frame, const std::string &in_source_frame)
{
        tf::StampedTransform transform;

        try
        {
                // What time should we use?
                _vectormap_transform_listener->lookupTransform(in_target_frame, in_source_frame, ros::Time(0), transform);
        }
        catch (tf::TransformException ex)
        {
                ROS_ERROR("%s", ex.what());
                return transform;
        }

        return transform;
}

geometry_msgs::Point transformPoint(const geometry_msgs::Point& point, const tf::Transform& tf)
{
        tf::Point tf_point;
        tf::pointMsgToTF(point, tf_point);

        tf_point = tf * tf_point;

        geometry_msgs::Point ros_point;
        tf::pointTFToMsg(tf_point, ros_point);

        return ros_point;
}

void transformBoundingBox(const jsk_recognition_msgs::BoundingBox &in_boundingbox,
                          jsk_recognition_msgs::BoundingBox &out_boundingbox, const std::string &in_target_frame,
                          const std_msgs::Header &in_header)
{
        geometry_msgs::PoseStamped pose_in, pose_out;
        pose_in.header = in_header;
        pose_in.pose = in_boundingbox.pose;
        try
        {
                _transform_listener->transformPose(in_target_frame, ros::Time(), pose_in, in_header.frame_id, pose_out);
        }
        catch (tf::TransformException &ex)
        {
                ROS_ERROR("transformBoundingBox: %s", ex.what());
        }
        out_boundingbox.pose = pose_out.pose;
        out_boundingbox.header = in_header;
        out_boundingbox.header.frame_id = in_target_frame;
        out_boundingbox.dimensions = in_boundingbox.dimensions;
        out_boundingbox.value = in_boundingbox.value;
        out_boundingbox.label = in_boundingbox.label;
}

// yang21itsc
/* y = P_rect_2 * R0_rect * Tr_velo_to_cam * x
Eigen::Vector3d projection(const Eigen::Vector4d &point) {
        Eigen::Vector3d projected_point = calib->GetProjCam2() * calib->getR_rect() * calib->getTr_velo_cam() * point;
        return Eigen::Vector3d(int(projected_point[0] / projected_point[2] + 0.5),
                               int(projected_point[1] / projected_point[2] + 0.5),
                               1);
}*/

Eigen::Vector3d projection(const Eigen::Vector4d &point) {
        Eigen::Vector3d projected_point = calib->GetProjCam2() * calib->getR_rect() * calib->getTr_velo_cam() * point;
        return Eigen::Vector3d(float(projected_point[0] / projected_point[2]),
                               float(projected_point[1] / projected_point[2]),
                               1);
}
// yang21itsc


void publishDetectedObjects(const autoware_tracker::CloudClusterArray &in_clusters,
                            const vision_msgs::Detection2DArrayConstPtr& in_image_detections)
{       
        autoware_tracker::DetectedObjectArray detected_objects;
        detected_objects.header = in_clusters.header;
        detected_objects.frame_out = count;
        std::cout <<"detected_objects.frame_out is " << detected_objects.frame_out << std::endl;
        for(size_t i = 0; i < in_clusters.clusters.size(); i++) {
                // Size limitation is not reasonable, but it can increase fps.
                double length = std::max(in_clusters.clusters[i].bounding_box.dimensions.x,in_clusters.clusters[i].bounding_box.dimensions.y);
                double width = std::min(in_clusters.clusters[i].bounding_box.dimensions.x,in_clusters.clusters[i].bounding_box.dimensions.y);
                double height = in_clusters.clusters[i].bounding_box.dimensions.z;
                if(length < 0.1
                || length > 5.0
                || width < 0.1
                || width > 2.5
                || height < 0.8
                || height > 2.0
                || length * width * height > 15) continue;
 
                autoware_tracker::DetectedObject detected_object;
                detected_object.header = in_clusters.header;
                detected_object.label = "9";
                detected_object.score = 1.;

                std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > vertices_3dbox;
                Eigen::Vector4d left_back_down(in_clusters.clusters[i].centroid_point.point.x - in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y - in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z - in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d right_back_down(in_clusters.clusters[i].centroid_point.point.x + in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y - in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z - in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d right_front_down(in_clusters.clusters[i].centroid_point.point.x + in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y + in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z - in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d left_front_down(in_clusters.clusters[i].centroid_point.point.x - in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y + in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z - in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d left_back_up(in_clusters.clusters[i].centroid_point.point.x - in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y - in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z + in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d right_back_up(in_clusters.clusters[i].centroid_point.point.x + in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y - in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z + in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d right_front_up(in_clusters.clusters[i].centroid_point.point.x + in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y + in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z + in_clusters.clusters[i].dimensions.z/2, 1);
                Eigen::Vector4d left_front_up(in_clusters.clusters[i].centroid_point.point.x - in_clusters.clusters[i].dimensions.x/2, in_clusters.clusters[i].centroid_point.point.y + in_clusters.clusters[i].dimensions.y/2, in_clusters.clusters[i].centroid_point.point.z + in_clusters.clusters[i].dimensions.z/2, 1);
                vertices_3dbox.push_back(left_back_down);
                vertices_3dbox.push_back(right_back_down);
                vertices_3dbox.push_back(right_front_down);
                vertices_3dbox.push_back(left_front_down);
                vertices_3dbox.push_back(left_back_up);
                vertices_3dbox.push_back(right_back_up);
                vertices_3dbox.push_back(right_front_up);
                vertices_3dbox.push_back(left_front_up);


                std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > projected_vertices;
                //std::cerr << "-------MUST BE 8-------vertices_3dbox.size() = " << vertices_3dbox.size() << std::endl;
                for(size_t j = 0; j < vertices_3dbox.size(); j++) {
                        projected_vertices.push_back(projection(vertices_3dbox[j]));
                }
                Eigen::Vector3d projected_min_point(DBL_MAX, DBL_MAX, 1);
                Eigen::Vector3d projected_max_point(-DBL_MAX, -DBL_MAX, 1);
                for(size_t j = 0; j < projected_vertices.size(); j++) {
                        if(projected_min_point[0] > projected_vertices[j][0]) {
                                projected_min_point[0] = projected_vertices[j][0];
                        }
                        if(projected_min_point[1] > projected_vertices[j][1]) {
                                projected_min_point[1] = projected_vertices[j][1];
                        }
                        if(projected_max_point[0] < projected_vertices[j][0]) {
                                projected_max_point[0] = projected_vertices[j][0];
                        }
                        if(projected_max_point[1] < projected_vertices[j][1]) {
                                projected_max_point[1] = projected_vertices[j][1];
                        }
                }
                detected_object.space_frame = in_clusters.header.frame_id;
                detected_object.pose = in_clusters.clusters[i].bounding_box.pose;
                detected_object.dimensions = in_clusters.clusters[i].dimensions;
                detected_object.pointcloud = in_clusters.clusters[i].cloud;
                detected_object.convex_hull = in_clusters.clusters[i].convex_hull;
                detected_object.valid = true;
                float iou_max = 0;
                int image_height,image_width;

                // if (iskitti){
                //         image_height = 375;
                //         image_width = 1242;
                // }else{
                //         image_height = 1200;
                //         image_width = 1920;
                // }

                image_height = 375;
                image_width = 1242;
                
   
                for(size_t j = 0; j < in_image_detections->detections.size(); j++) {//对于一帧中所有2d框
                        if(in_image_detections->detections[j].results[0].score > 0.3) {//置信度>0.5
                                int im_min_x = in_image_detections->detections[j].bbox.center.x - (in_image_detections->detections[j].bbox.size_x / 2);
                                int im_min_y = in_image_detections->detections[j].bbox.center.y - (in_image_detections->detections[j].bbox.size_y / 2);
                                int im_max_x = in_image_detections->detections[j].bbox.center.x + (in_image_detections->detections[j].bbox.size_x / 2);
                                int im_max_y = in_image_detections->detections[j].bbox.center.y + (in_image_detections->detections[j].bbox.size_y / 2);

                                int x0 = std::max(int(projected_min_point[0]), im_min_x);
                                int y0 = std::max(int(projected_min_point[1]), im_min_y);
                                int x1 = std::min(int(projected_max_point[0]), im_max_x);
                                int y1 = std::min(int(projected_max_point[1]), im_max_y);

                                int inter_area = std::abs(std::max(x1-x0, 0) * std::max(y1-y0, 0));

                                if(inter_area > 0) {
                                        int box0 = std::abs(int(projected_max_point[0] - projected_min_point[0]) *
                                                            int(projected_max_point[1] - projected_min_point[1]));//聚类
                                        int box1 = std::abs(int(im_max_x - im_min_x) *
                                                            int(im_max_y - im_min_y));


                                        float iou = inter_area / float(box0 + box1 - inter_area);
                                        
                                        if(iou > iou_max) {
                                                iou_max = iou;
                                                if(iou_max > iou_threshold && in_clusters.clusters[i].centroid_point.point.x > 0) {//iou>0.5，给label
                                                        detected_object.label = std::to_string(in_image_detections->detections[j].results[0].id); // 0:car, 1:pedestrian, 2:cyclist
                                                        detected_object.score = in_image_detections->detections[j].results[0].score;
                                                        detected_object.image_frame = std::to_string(j); //
                                                        // detected_objects.objects.push_back(detected_object);
                               
                                                }
                                        }
                                }
                        }
                }

                
                detected_objects.objects.push_back(detected_object);
                
        }

        for(size_t n = 0; n < in_image_detections->detections.size(); n++) {
                float iou_max_ = 0;
                for(size_t m = 0; m < detected_objects.objects.size(); m++) {
                        if(detected_objects.objects[m].image_frame == std::to_string(n)) {
                                if(detected_objects.objects[m].score > iou_max_) {
                                        iou_max_ = detected_objects.objects[m].score;
                                }
                        }
                }
                
                for(size_t k = 0; k < detected_objects.objects.size(); k++) {
       
                        if(detected_objects.objects[k].image_frame == std::to_string(n)) {
                                if(detected_objects.objects[k].score < iou_max_) {
                                        
                                        std::cout << "set to 9" << std::endl;
                                        detected_objects.objects[k].label = "9";
                                        detected_objects.objects[k].score = 1.;
                                } else{
                                  detected_objects.objects[k].score = iou_max_;
                                }
                        }
                }
        }

        _pub_detected_objects.publish(detected_objects);
}
// yang21itsc

void publishCloudClusters(const ros::Publisher *in_publisher, const autoware_tracker::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header,
                          const vision_msgs::Detection2DArrayConstPtr& in_image_detections)
{
        
        if (in_target_frame != in_header.frame_id)
        {
                autoware_tracker::CloudClusterArray clusters_transformed;
                clusters_transformed.header = in_header;
                clusters_transformed.header.frame_id = in_target_frame;
                for (auto i = in_clusters.clusters.begin(); i != in_clusters.clusters.end(); i++)
                {
                        autoware_tracker::CloudCluster cluster_transformed;
                        cluster_transformed.header = in_header;
                        try
                        {
                                _transform_listener->lookupTransform(in_target_frame, _velodyne_header.frame_id, ros::Time(),
                                                                     *_transform);
                                pcl_ros::transformPointCloud(in_target_frame, *_transform, i->cloud, cluster_transformed.cloud);
                                _transform_listener->transformPoint(in_target_frame, ros::Time(), i->min_point, in_header.frame_id,
                                                                    cluster_transformed.min_point);
                                _transform_listener->transformPoint(in_target_frame, ros::Time(), i->max_point, in_header.frame_id,
                                                                    cluster_transformed.max_point);
                                _transform_listener->transformPoint(in_target_frame, ros::Time(), i->avg_point, in_header.frame_id,
                                                                    cluster_transformed.avg_point);
                                _transform_listener->transformPoint(in_target_frame, ros::Time(), i->centroid_point, in_header.frame_id,
                                                                    cluster_transformed.centroid_point);

                                cluster_transformed.dimensions = i->dimensions;
                                cluster_transformed.eigen_values = i->eigen_values;
                                cluster_transformed.eigen_vectors = i->eigen_vectors;

                                cluster_transformed.convex_hull = i->convex_hull;
                                cluster_transformed.bounding_box.pose.position = i->bounding_box.pose.position;
                                if(_pose_estimation)
                                {
                                        cluster_transformed.bounding_box.pose.orientation = i->bounding_box.pose.orientation;
                                }
                                else
                                {
                                        cluster_transformed.bounding_box.pose.orientation.w = _initial_quat_w;
                                }
                                clusters_transformed.clusters.push_back(cluster_transformed);
                        }
                        catch (tf::TransformException &ex)
                        {
                                ROS_ERROR("publishCloudClusters: %s", ex.what());
                        }
                }
                in_publisher->publish(clusters_transformed);
                publishDetectedObjects(clusters_transformed, in_image_detections);
        } else
        {
                in_publisher->publish(in_clusters);
                publishDetectedObjects(in_clusters, in_image_detections);
        }
}

void publishCentroids(const ros::Publisher *in_publisher, const autoware_tracker::Centroids &in_centroids,
                      const std::string &in_target_frame, const std_msgs::Header &in_header)
{
        if (in_target_frame != in_header.frame_id)
        {
                autoware_tracker::Centroids centroids_transformed;
                centroids_transformed.header = in_header;
                centroids_transformed.header.frame_id = in_target_frame;
                for (auto i = centroids_transformed.points.begin(); i != centroids_transformed.points.end(); i++)
                {
                        geometry_msgs::PointStamped centroid_in, centroid_out;
                        centroid_in.header = in_header;
                        centroid_in.point = *i;
                        try
                        {
                                _transform_listener->transformPoint(in_target_frame, ros::Time(), centroid_in, in_header.frame_id,
                                                                    centroid_out);

                                centroids_transformed.points.push_back(centroid_out.point);
                        }
                        catch (tf::TransformException &ex)
                        {
                                ROS_ERROR("publishCentroids: %s", ex.what());
                        }
                }
                in_publisher->publish(centroids_transformed);
        } else
        {
                in_publisher->publish(in_centroids);
        }
}

void publishCloud(const ros::Publisher *in_publisher, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{
        if (in_publisher->getNumSubscribers() == 0)
                return;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
        cloud_msg.header = _velodyne_header;
        in_publisher->publish(cloud_msg);
}

void publishColorCloud(const ros::Publisher *in_publisher,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
        cloud_msg.header = _velodyne_header;
        in_publisher->publish(cloud_msg);
}

void keepLanePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr, float in_left_lane_threshold = 1.5,
                    float in_right_lane_threshold = 1.5)
{
        pcl::PointIndices::Ptr far_indices(new pcl::PointIndices);
        for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
        {
                pcl::PointXYZI current_point;
                current_point.x = in_cloud_ptr->points[i].x;
                current_point.y = in_cloud_ptr->points[i].y;
                current_point.z = in_cloud_ptr->points[i].z;

                if (current_point.y > (in_left_lane_threshold) || current_point.y < -1.0 * in_right_lane_threshold)
                {
                        far_indices->indices.push_back(i);
                }
        }
        out_cloud_ptr->points.clear();
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(in_cloud_ptr);
        extract.setIndices(far_indices);
        extract.setNegative(true);        // true removes the indices, false leaves only the indices
        extract.filter(*out_cloud_ptr);
}


std::vector<ClusterPtr> clusterAndColor(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                                        pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr,
                                        autoware_tracker::Centroids &in_out_centroids,
                                        double in_max_cluster_distance = 0.5)
{
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);

        // create 2d pc
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*in_cloud_ptr, *cloud_2d);
        // // make it flat
        for (size_t i = 0; i < cloud_2d->points.size(); i++)
        {
                cloud_2d->points[i].z = 0;
        }

        if (cloud_2d->points.size() > 0)
                tree->setInputCloud(cloud_2d);

        std::vector<pcl::PointIndices> cluster_indices;

        // perform clustering on 2d cloud
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(in_max_cluster_distance);          //
        ec.setMinClusterSize(_cluster_size_min);
        ec.setMaxClusterSize(_cluster_size_max);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_2d);
        ec.extract(cluster_indices);
        // use indices on 3d cloud

        /////////////////////////////////
        //---	3. Color clustered points
        /////////////////////////////////
        unsigned int k = 0;
        // pcl::PointCloud<pcl::PointXYZI>::Ptr final_cluster (new pcl::PointCloud<pcl::PointXYZI>);

        std::vector<ClusterPtr> clusters;
        // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);//coord + color
        // cluster
        for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
                ClusterPtr cluster(new Cluster());
                cluster->SetCloud(in_cloud_ptr, it->indices, _velodyne_header, k, (int) _colors[k].val[0],
                                  (int) _colors[k].val[1],
                                  (int) _colors[k].val[2], "", _pose_estimation);
                clusters.push_back(cluster);

                k++;
        }
        // std::cout << "Clusters: " << k << std::endl;
        return clusters;
}

void checkClusterMerge(size_t in_cluster_id, std::vector<ClusterPtr> &in_clusters,
                       std::vector<bool> &in_out_visited_clusters, std::vector<size_t> &out_merge_indices,
                       double in_merge_threshold)
{
        // std::cout << "checkClusterMerge" << std::endl;
        pcl::PointXYZI point_a = in_clusters[in_cluster_id]->GetCentroid();
        for (size_t i = 0; i < in_clusters.size(); i++)
        {
                if (i != in_cluster_id && !in_out_visited_clusters[i])
                {
                        pcl::PointXYZI point_b = in_clusters[i]->GetCentroid();
                        double distance = sqrt(pow(point_b.x - point_a.x, 2) + pow(point_b.y - point_a.y, 2));
                        if (distance <= in_merge_threshold)
                        {
                                in_out_visited_clusters[i] = true;
                                out_merge_indices.push_back(i);
                                // std::cout << "Merging " << in_cluster_id << " with " << i << " dist:" << distance << std::endl;
                                checkClusterMerge(i, in_clusters, in_out_visited_clusters, out_merge_indices, in_merge_threshold);
                        }
                }
        }
}

void mergeClusters(const std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                   std::vector<size_t> in_merge_indices, const size_t &current_index,
                   std::vector<bool> &in_out_merged_clusters)
{
        // std::cout << "mergeClusters:" << in_merge_indices.size() << std::endl;
        pcl::PointCloud<pcl::PointXYZI> sum_cloud;
        pcl::PointCloud<pcl::PointXYZI> mono_cloud;
        ClusterPtr merged_cluster(new Cluster());
        for (size_t i = 0; i < in_merge_indices.size(); i++)
        {
                sum_cloud += *(in_clusters[in_merge_indices[i]]->GetCloud());
                in_out_merged_clusters[in_merge_indices[i]] = true;
        }
        std::vector<int> indices(sum_cloud.points.size(), 0);
        for (size_t i = 0; i < sum_cloud.points.size(); i++)
        {
                indices[i] = i;
        }

        if (sum_cloud.points.size() > 0)
        {
                pcl::copyPointCloud(sum_cloud, mono_cloud);
                merged_cluster->SetCloud(mono_cloud.makeShared(), indices, _velodyne_header, current_index,
                                         (int) _colors[current_index].val[0], (int) _colors[current_index].val[1],
                                         (int) _colors[current_index].val[2], "", _pose_estimation);
                out_clusters.push_back(merged_cluster);
        }
}

void checkAllForMerge(std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                      float in_merge_threshold)
{
        // std::cout << "checkAllForMerge" << std::endl;
        std::vector<bool> visited_clusters(in_clusters.size(), false);
        std::vector<bool> merged_clusters(in_clusters.size(), false);
        size_t current_index = 0;
        for (size_t i = 0; i < in_clusters.size(); i++)
        {
                if (!visited_clusters[i])
                {
                        visited_clusters[i] = true;
                        std::vector<size_t> merge_indices;
                        checkClusterMerge(i, in_clusters, visited_clusters, merge_indices, in_merge_threshold);
                        mergeClusters(in_clusters, out_clusters, merge_indices, current_index++, merged_clusters);
                }
        }
        for (size_t i = 0; i < in_clusters.size(); i++)
        {
                // check for clusters not merged, add them to the output
                if (!merged_clusters[i])
                {
                        out_clusters.push_back(in_clusters[i]);
                }
        }

        // ClusterPtr cluster(new Cluster());
}

void segmentByDistance(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr,
                       autoware_tracker::Centroids &in_out_centroids, autoware_tracker::CloudClusterArray &in_out_clusters)
{
        // cluster the pointcloud according to the distance of the points using different thresholds (not only one for the
        // entire pc)
        // in this way, the points farther in the pc will also be clustered

        // 0 => 0-15m d=0.5
        // 1 => 15-30 d=1
        // 2 => 30-45 d=1.6
        // 3 => 45-60 d=2.1
        // 4 => >60   d=2.6

        std::vector<ClusterPtr> all_clusters;

        if (!_use_multiple_thres)
        {
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);

                for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
                {
                        pcl::PointXYZI current_point;
                        current_point.x = in_cloud_ptr->points[i].x;
                        current_point.y = in_cloud_ptr->points[i].y;
                        current_point.z = in_cloud_ptr->points[i].z;
                        current_point.intensity = in_cloud_ptr->points[i].intensity;

                        cloud_ptr->points.push_back(current_point);
                }

                all_clusters = clusterAndColor(cloud_ptr, out_cloud_ptr, in_out_centroids, _clustering_distance);
        } else
        {
                std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_segments_array(5);
                for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
                {
                        pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                        cloud_segments_array[i] = tmp_cloud;
                }

                for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
                {
                        pcl::PointXYZI current_point;
                        current_point.x = in_cloud_ptr->points[i].x;
                        current_point.y = in_cloud_ptr->points[i].y;
                        current_point.z = in_cloud_ptr->points[i].z;
                        current_point.intensity = in_cloud_ptr->points[i].intensity;

                        float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

                        if (origin_distance < _clustering_ranges[0])
                        {
                                cloud_segments_array[0]->points.push_back(current_point);
                        }
                        else if (origin_distance < _clustering_ranges[1])
                        {
                                cloud_segments_array[1]->points.push_back(current_point);

                        }else if (origin_distance < _clustering_ranges[2])
                        {
                                cloud_segments_array[2]->points.push_back(current_point);

                        }else if (origin_distance < _clustering_ranges[3])
                        {
                                cloud_segments_array[3]->points.push_back(current_point);

                        }else
                        {
                                cloud_segments_array[4]->points.push_back(current_point);
                        }
                }

                std::vector<ClusterPtr> local_clusters;
                for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
                {
                        local_clusters = clusterAndColor(cloud_segments_array[i], out_cloud_ptr, in_out_centroids, _clustering_distances[i]);
                        all_clusters.insert(all_clusters.end(), local_clusters.begin(), local_clusters.end());
                }
        }

        // Clusters can be merged or checked in here
        //....
        // check for mergable clusters
        std::vector<ClusterPtr> mid_clusters;
        std::vector<ClusterPtr> final_clusters;

        if (all_clusters.size() > 0)
                checkAllForMerge(all_clusters, mid_clusters, _cluster_merge_threshold);
        else
                mid_clusters = all_clusters;

        if (mid_clusters.size() > 0)
                checkAllForMerge(mid_clusters, final_clusters, _cluster_merge_threshold);
        else
                final_clusters = mid_clusters;

        // Get final PointCloud to be published
        for (unsigned int i = 0; i < final_clusters.size(); i++)
        {
                *out_cloud_ptr = *out_cloud_ptr + *(final_clusters[i]->GetCloud());

                jsk_recognition_msgs::BoundingBox bounding_box = final_clusters[i]->GetBoundingBox();
                geometry_msgs::PolygonStamped polygon = final_clusters[i]->GetPolygon();
                jsk_rviz_plugins::Pictogram pictogram_cluster;
                pictogram_cluster.header = _velodyne_header;

                // PICTO
                pictogram_cluster.mode = pictogram_cluster.STRING_MODE;
                pictogram_cluster.pose.position.x = final_clusters[i]->GetMaxPoint().x;
                pictogram_cluster.pose.position.y = final_clusters[i]->GetMaxPoint().y;
                pictogram_cluster.pose.position.z = final_clusters[i]->GetMaxPoint().z;
                tf::Quaternion quat(0.0, -0.7, 0.0, 0.7);
                tf::quaternionTFToMsg(quat, pictogram_cluster.pose.orientation);
                pictogram_cluster.size = 4;
                std_msgs::ColorRGBA color;
                color.a = 1;
                color.r = 1;
                color.g = 1;
                color.b = 1;
                pictogram_cluster.color = color;
                pictogram_cluster.character = std::to_string(i);
                // PICTO

                // pcl::PointXYZI min_point = final_clusters[i]->GetMinPoint();
                // pcl::PointXYZI max_point = final_clusters[i]->GetMaxPoint();
                pcl::PointXYZI center_point = final_clusters[i]->GetCentroid();
                geometry_msgs::Point centroid;
                centroid.x = center_point.x;
                centroid.y = center_point.y;
                centroid.z = center_point.z;
                bounding_box.header = _velodyne_header;
                polygon.header = _velodyne_header;

                if (final_clusters[i]->IsValid())
                {

                        in_out_centroids.points.push_back(centroid);

                        autoware_tracker::CloudCluster cloud_cluster;
                        final_clusters[i]->ToROSMessage(_velodyne_header, cloud_cluster);
                        in_out_clusters.clusters.push_back(cloud_cluster);
                }
        }
}

void removeFloor(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr out_nofloor_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr out_onlyfloor_cloud_ptr, float in_max_height = 0.2,
                 float in_floor_max_angle = 0.1)
{

        pcl::SACSegmentation<pcl::PointXYZI> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setAxis(Eigen::Vector3f(0, 0, 1));
        seg.setEpsAngle(in_floor_max_angle);

        seg.setDistanceThreshold(in_max_height);                    // floor distance
        seg.setOptimizeCoefficients(true);
        seg.setInputCloud(in_cloud_ptr);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
        {
                std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
        }

        // REMOVE THE FLOOR FROM THE CLOUD
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(in_cloud_ptr);
        extract.setIndices(inliers);
        extract.setNegative(true);                    // true removes the indices, false leaves only the indices
        extract.filter(*out_nofloor_cloud_ptr);

        // EXTRACT THE FLOOR FROM THE CLOUD
        extract.setNegative(false);                    // true removes the indices, false leaves only the indices
        extract.filter(*out_onlyfloor_cloud_ptr);
}




void downsampleCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr, float in_leaf_size = 0.2)
{
        pcl::VoxelGrid<pcl::PointXYZI> sor;
        sor.setInputCloud(in_cloud_ptr);
        sor.setLeafSize((float) in_leaf_size, (float) in_leaf_size, (float) in_leaf_size);
        sor.filter(*out_cloud_ptr);
}

void clipCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
               pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr, float in_min_height = -1.3, float in_max_height = 0.5)
{
        out_cloud_ptr->points.clear();
        for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
        {
                if (in_cloud_ptr->points[i].z >= in_min_height && in_cloud_ptr->points[i].z <= in_max_height)
                {
                        out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
                } 
        }
}

void differenceNormalsSegmentation(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr)
{
        float small_scale = 0.5;
        float large_scale = 2.0;
        float angle_threshold = 0.5;
        pcl::search::Search<pcl::PointXYZI>::Ptr tree;
        if (in_cloud_ptr->isOrganized())
        {
                tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZI>());
        } else
        {
                tree.reset(new pcl::search::KdTree<pcl::PointXYZI>(false));
        }

        // Set the input pointcloud for the search tree
        tree->setInputCloud(in_cloud_ptr);

        pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointNormal> normal_estimation;
        // pcl::gpu::NormalEstimation<pcl::PointXYZI, pcl::PointNormal> normal_estimation;
        normal_estimation.setInputCloud(in_cloud_ptr);
        normal_estimation.setSearchMethod(tree);

        normal_estimation.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());

        pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);

        normal_estimation.setRadiusSearch(small_scale);
        normal_estimation.compute(*normals_small_scale);

        normal_estimation.setRadiusSearch(large_scale);
        normal_estimation.compute(*normals_large_scale);

        pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud(new pcl::PointCloud<pcl::PointNormal>);
        pcl::copyPointCloud<pcl::PointXYZI, pcl::PointNormal>(*in_cloud_ptr, *diffnormals_cloud);

        // Create DoN operator
        pcl::DifferenceOfNormalsEstimation<pcl::PointXYZI, pcl::PointNormal, pcl::PointNormal> diffnormals_estimator;
        diffnormals_estimator.setInputCloud(in_cloud_ptr);
        diffnormals_estimator.setNormalScaleLarge(normals_large_scale);
        diffnormals_estimator.setNormalScaleSmall(normals_small_scale);

        diffnormals_estimator.initCompute();

        diffnormals_estimator.computeFeature(*diffnormals_cloud);

        pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
        range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
                                          new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, angle_threshold)));
        // Build the filter
        pcl::ConditionalRemoval<pcl::PointNormal> cond_removal;
        cond_removal.setCondition(range_cond);
        cond_removal.setInputCloud(diffnormals_cloud);

        pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

        // Apply filter
        cond_removal.filter(*diffnormals_cloud_filtered);

        pcl::copyPointCloud<pcl::PointNormal, pcl::PointXYZI>(*diffnormals_cloud, *out_cloud_ptr);
}

void removePointsUpTo(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr, const double min_dist, const double max_dist)
{
        out_cloud_ptr->points.clear();
        for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
        {
                float origin_distance = sqrt(pow(in_cloud_ptr->points[i].x, 2) + pow(in_cloud_ptr->points[i].y, 2));
                if (origin_distance > min_dist && origin_distance < max_dist)
                {
                        out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
                }
        }
}


void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud,
                       const vision_msgs::Detection2DArrayConstPtr& in_image_detections)
{
        //_start = std::chrono::system_clock::now();
        count += 1;
        std::cout << " in callback " << count << std::endl;
        
        std::cerr << in_sensor_cloud->header.stamp << "---" << in_image_detections->header.stamp << std::endl;

        double start_time;
        double current_time;

        pcl::PointCloud<pcl::PointXYZI>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr removed_points_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr inlanes_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr nofloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr onlyfloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr diffnormals_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr clipped_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr colored_clustered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);

        autoware_tracker::Centroids centroids;
        autoware_tracker::CloudClusterArray cloud_clusters;

        pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);

        _velodyne_header = in_sensor_cloud->header;

        // remove points too close or too far
        if (_remove_points_min > 0.0)
        {
                removePointsUpTo(current_sensor_cloud_ptr, removed_points_cloud_ptr, _remove_points_min, _remove_points_max);
        }
        else
        {
                removed_points_cloud_ptr = current_sensor_cloud_ptr;
        }

        // downsample cloud
        if (_downsample_cloud)
                downsampleCloud(removed_points_cloud_ptr, downsampled_cloud_ptr, _leaf_size);
        else
                downsampled_cloud_ptr = removed_points_cloud_ptr;

        // height clip
        clipCloud(downsampled_cloud_ptr, clipped_cloud_ptr, _clip_min_height, _clip_max_height);

        // left and right points filter
        if (_keep_lanes)
                keepLanePoints(clipped_cloud_ptr, inlanes_cloud_ptr, _keep_lane_left_distance, _keep_lane_right_distance);
        else
                inlanes_cloud_ptr = clipped_cloud_ptr;

        // remove ground points
        if (_remove_ground)
        {
                removeFloor(inlanes_cloud_ptr, nofloor_cloud_ptr, onlyfloor_cloud_ptr);
                publishCloud(&_pub_ground_cloud, onlyfloor_cloud_ptr);
        }
        else
        {
                
                nofloor_cloud_ptr = inlanes_cloud_ptr;
        }
        publishCloud(&_pub_points_lanes_cloud, nofloor_cloud_ptr);

        // publishCloud(&_pub_noground_cloud, nofloor_cloud_ptr);
        
        // normal filter
        if (_use_diffnormals)
                differenceNormalsSegmentation(nofloor_cloud_ptr, diffnormals_cloud_ptr);
        else
                diffnormals_cloud_ptr = nofloor_cloud_ptr;

        // main cluster
        start_time = ros::WallTime::now().toSec();
        segmentByDistance(diffnormals_cloud_ptr, colored_clustered_cloud_ptr, centroids, cloud_clusters);
        publishColorCloud(&_pub_cluster_cloud, colored_clustered_cloud_ptr);
        current_time = ros::WallTime::now().toSec();
        ROS_INFO("cluster: %lf s", current_time - start_time);

        centroids.header = _velodyne_header;
        publishCentroids(&_centroid_pub, centroids, _output_frame, _velodyne_header);

        cloud_clusters.header = _velodyne_header;
        publishCloudClusters(&_pub_clusters_message, cloud_clusters, _output_frame, _velodyne_header, in_image_detections);
}

int main(int argc, char **argv)
{
        // Initialize ROS
        ros::init(argc, argv, "euclidean_cluster");
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

        ros::NodeHandle nh;

        tf::StampedTransform transform;
        tf::TransformListener listener;
        tf::TransformListener vectormap_tf_listener;

        _vectormap_transform_listener = &vectormap_tf_listener;
        _transform = &transform;
        _transform_listener = &listener;

        ROS_INFO("[%s] generating colors ...", __APP_NAME__);
                                      #if (CV_MAJOR_VERSION == 3)
        generateColors(_colors, 255);
                                      #else
        cv::generateColors(_colors, 255);
                                      #endif
        ROS_INFO("[%s] generating colors done!", __APP_NAME__);


        _pub_cluster_cloud = nh.advertise<sensor_msgs::PointCloud2>("autoware_tracker/cluster/points_cluster", 10);
        _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("autoware_tracker/cluster/points_ground", 10);
        // _pub_noground_cloud = nh.advertise<sensor_msgs::PointCloud2>("autoware_tracker/cluster/points_noground", 1);
        _centroid_pub = nh.advertise<autoware_tracker::Centroids>("autoware_tracker/cluster/cluster_centroids", 10);

        _pub_points_lanes_cloud = nh.advertise<sensor_msgs::PointCloud2>("autoware_tracker/cluster/points_lanes", 10);
        _pub_clusters_message = nh.advertise<autoware_tracker::CloudClusterArray>("autoware_tracker/cluster/cloud_clusters", 10);
        _pub_detected_objects = nh.advertise<autoware_tracker::DetectedObjectArray>("autoware_tracker/cluster/objects", 10);

        std::string points_topic = "/points_raw";
        if (nh.getParam("autoware_tracker/cluster/points_node", points_topic)) {
                ROS_INFO("[%s] Setting points_node to %s", __APP_NAME__, points_topic.c_str());
        } else {
                ROS_INFO("[%s] No points_node received, defaulting to /points_raw", __APP_NAME__);
        }
        // yang21itsc
        std::string image_detections_topic = "/image_detections";
        if (nh.getParam("autoware_tracker/cluster/label_source", image_detections_topic)) {
                ROS_INFO("[%s] Setting label_source to %s", __APP_NAME__, image_detections_topic.c_str());
        } else {
                ROS_INFO("[%s] No label_source received, defaulting to /image_detections", __APP_NAME__);
        }
        std::string extrinsic_calibration_file = "calib.txt";
        if (nh.getParam("autoware_tracker/cluster/extrinsic_calibration", extrinsic_calibration_file)) {
                ROS_INFO("[%s] Setting points_node to %s", __APP_NAME__, extrinsic_calibration_file.c_str());
        } else {
                ROS_INFO("[%s] No points_node received, defaulting to calib.txt", __APP_NAME__);
        }

        if (nh.getParam("autoware_tracker/cluster/iou_threshold", iou_threshold)) {
                ROS_INFO("[%s] Setting points_node to %f", __APP_NAME__, iou_threshold);
        } else {
                ROS_INFO("[%s] No points_node received, defaulting to 0.5", __APP_NAME__);
        }

        calib = new Calibration(extrinsic_calibration_file);
        // yang21itsc

        _use_diffnormals = false;
        if (nh.getParam("autoware_tracker/cluster/use_diffnormals", _use_diffnormals)) {
                if (_use_diffnormals) {
                        ROS_INFO("[%s] Applying difference of normals on clustering pipeline", __APP_NAME__);
                } else {
                        ROS_INFO("[%s] Difference of Normals will not be used.", __APP_NAME__);
                }
        }

        /* Initialize tuning parameter */
        
        nh.param("autoware_tracker/cluster/use_callback", use_callback, use_callback);
        ROS_INFO("[%s] use_callback: %d", __APP_NAME__, use_callback);

        nh.param("autoware_tracker/cluster/use_camera", use_camera, use_camera);
        ROS_INFO("[%s] use_camera: %d", __APP_NAME__, use_camera);

        nh.param<std::string>("autoware_tracker/cluster/output_frame", _output_frame, "velodyne");
        ROS_INFO("[%s] output_frame: %s", __APP_NAME__, _output_frame.c_str());
        nh.param("autoware_tracker/cluster/downsample_cloud", _downsample_cloud, false);
        ROS_INFO("[%s] downsample_cloud: %d", __APP_NAME__, _downsample_cloud);
        nh.param("autoware_tracker/cluster/remove_ground", _remove_ground, true);
        ROS_INFO("[%s] remove_ground: %d", __APP_NAME__, _remove_ground);
        nh.param("autoware_tracker/cluster/leaf_size", _leaf_size, 0.1);
        ROS_INFO("[%s] leaf_size: %f", __APP_NAME__, _leaf_size);
        nh.param("autoware_tracker/cluster/cluster_size_min", _cluster_size_min, 20);
        ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);
        nh.param("autoware_tracker/cluster/cluster_size_max", _cluster_size_max, 100000);
        ROS_INFO("[%s] cluster_size_max: %d", __APP_NAME__, _cluster_size_max);
        nh.param("autoware_tracker/cluster/pose_estimation", _pose_estimation, false);
        ROS_INFO("[%s] pose_estimation: %d", __APP_NAME__, _pose_estimation);
        nh.param("autoware_tracker/cluster/clip_min_height", _clip_min_height, -1.3);
        ROS_INFO("[%s] clip_min_height: %f", __APP_NAME__, _clip_min_height);
        nh.param("autoware_tracker/cluster/clip_max_height", _clip_max_height, 0.5);
        ROS_INFO("[%s] clip_max_height: %f", __APP_NAME__, _clip_max_height);
        nh.param("autoware_tracker/cluster/keep_lanes", _keep_lanes, false);
        ROS_INFO("[%s] keep_lanes: %d", __APP_NAME__, _keep_lanes);
        nh.param("autoware_tracker/cluster/keep_lane_left_distance", _keep_lane_left_distance, 5.0);
        ROS_INFO("[%s] keep_lane_left_distance: %f", __APP_NAME__, _keep_lane_left_distance);
        nh.param("autoware_tracker/cluster/keep_lane_right_distance", _keep_lane_right_distance, 5.0);
        ROS_INFO("[%s] keep_lane_right_distance: %f", __APP_NAME__, _keep_lane_right_distance);
        nh.param("autoware_tracker/cluster/cluster_merge_threshold", _cluster_merge_threshold, 1.5);
        ROS_INFO("[%s] cluster_merge_threshold: %f", __APP_NAME__, _cluster_merge_threshold);
        nh.param("autoware_tracker/cluster/clustering_distance", _clustering_distance, 0.75);
        ROS_INFO("[%s] clustering_distance: %f", __APP_NAME__, _clustering_distance);
        nh.param("autoware_tracker/cluster/remove_points_min", _remove_points_min, 0.0);
        ROS_INFO("[%s] remove_points_min: %f", __APP_NAME__, _remove_points_min);
        nh.param("autoware_tracker/cluster/remove_points_max", _remove_points_max, 0.0);
        ROS_INFO("[%s] remove_points_max: %f", __APP_NAME__, _remove_points_max);

        nh.param("autoware_tracker/cluster/use_multiple_thres", _use_multiple_thres, false);
        ROS_INFO("[%s] use_multiple_thres: %d", __APP_NAME__, _use_multiple_thres);

        //lbh
        std::string velodyne_dir = "velodyne_dir_path";
        nh.param<std::string>("autoware_tracker/cluster/velodyne_dir", velodyne_dir, "velodyne_dir_path");
        ROS_INFO("[%s] use_velodyne_dir: %s", __APP_NAME__, velodyne_dir.c_str());

        std::string camera_dir = "camera_dir_path";
        nh.param<std::string>("autoware_tracker/cluster/camera_dir", camera_dir, "camera_dir_path");
        ROS_INFO("[%s] use_camera_dir: %s", __APP_NAME__, camera_dir.c_str());
        double frequency = 5.0;
        nh.param<double>("autoware_tracker/cluster/frequency", frequency, frequency);    
        ROS_INFO("[%s] use_frequency: %f", __APP_NAME__, frequency);
        nh.param<bool>("autoware_tracker/cluster/iskitti", iskitti, true);    
        ROS_INFO("[%s] iskitti: %d", __APP_NAME__, iskitti);

        std::vector<double> cluster_distance = {0.3,0.5,0.8,1.4,2.4};
        std::vector<double> cluster_range = {15,30,45,60};
        nh.getParam("autoware_tracker/cluster/cluster_distances", cluster_distance);


        


        double timestamp;
        if (_use_multiple_thres)
        {       
                if (iskitti)
                {


                        _clustering_distances = cluster_distance;
                        _clustering_ranges = cluster_range;

                }
                else{
                        _clustering_distances.push_back(0.4);
                        _clustering_distances.push_back(0.6);
                        _clustering_distances.push_back(0.8);
                        _clustering_distances.push_back(1.0);
                        _clustering_distances.push_back(1.2);//10.25 1.2 



                        _clustering_ranges.push_back(4);
                        _clustering_ranges.push_back(8);
                        _clustering_ranges.push_back(12);
                        _clustering_ranges.push_back(16);  
                }



        }
        for (const auto &item : cluster_distance) {
                ROS_INFO("cluster_distances: %f", item);
        }

        nh.getParam("autoware_tracker/cluster/cluster_ranges", cluster_range);

        for (const auto &item : cluster_range) {
                ROS_INFO("cluster_ranges: %f", item);
        }

        message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub(nh, points_topic, 10);
        message_filters::Subscriber<vision_msgs::Detection2DArray> image_detections_sub(nh, image_detections_topic, 10);
        
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, vision_msgs::Detection2DArray> MySyncPolicy;
        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), points_sub, image_detections_sub);
        sync.registerCallback(boost::bind(&velodyne_callback, _1, _2));


        ros::spin();
}
