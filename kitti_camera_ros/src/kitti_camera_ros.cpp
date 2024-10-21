// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
// PCL
#include <pcl_conversions/pcl_conversions.h>
// C++
#include <dirent.h>
// Rui
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
// CV Bridge
#include <cv_bridge/cv_bridge.h>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int count = -1;

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

int main(int argc, char **argv) {
        double frequency;
        std::string camera_dir;
        std::string img_path;
        std::string scence;
        double timestamp;
        
        ros::init(argc, argv, "kitti_camera_ros");
        ros::NodeHandle private_nh("~");

        ros::Publisher camera_pub = private_nh.advertise<vision_msgs::Detection2DArray>("/image_detections", 100, true);
        ros::Publisher img_pub = private_nh.advertise<sensor_msgs::Image>("/image_vis", 100, true);

        private_nh.param<double>("frequency", frequency, 1);
        private_nh.param<std::string>("camera_dir", camera_dir, "camera_dir_path");
        private_nh.param<std::string>("img_path", img_path, "img_path");
        private_nh.param<std::string>("scence", scence, "0000");
        ros::Rate loop_rate(frequency);
        
        //vision_msgs::Detection2DArray detection_results;
        camera_dir = camera_dir + scence + "/";
        img_path = img_path + scence + "/";
        struct dirent **filelist;
        int n_file = scandir(camera_dir.c_str(), &filelist, NULL, alphasort);
        if(n_file == -1) {
                ROS_ERROR_STREAM("[kitti_camera_ros] Could not open directory: " << camera_dir);
                return EXIT_FAILURE;
        } else {
                ROS_INFO_STREAM("[kitti_camera_ros] Load camera files in " << camera_dir);
                ROS_INFO_STREAM("[kitti_camera_ros] frequency (loop rate): " << frequency);
        }

        int i_file = 2; // 0 = . 1 = ..
        while(ros::ok() && i_file < n_file) {
                vision_msgs::Detection2DArray detection_results;

                /*** Camera ***/
                std::string s = camera_dir + filelist[i_file]->d_name;
                
                std::fstream camera_txt(s.c_str(), std::ios::in | std::ios::binary);
                std::string filename(filelist[i_file]->d_name);
                
                // 查找最后一个点的位置
                size_t dotPosition = filename.find_last_of('.');

                // 提取文件名（不包括扩展名）
                std::string name = filename.substr(0, dotPosition);

        
                std::string path_img = img_path + name + ".png";  

                if (fileExists(path_img)) {
                        
                }else{
                        path_img = img_path + name + ".jpg"; 
                }
                std::cout << path_img<< std::endl;

                //std::cerr << "s: " << s.c_str() << std::endl;
                if(!camera_txt.good()) {
                        ROS_ERROR_STREAM("[kitti_camera_ros] Could not read file: " << s);
                        return EXIT_FAILURE;
                } else {
                        camera_txt >> timestamp;
                        ros::Time timestamp_ros(timestamp == 0 ? ros::TIME_MIN.toSec() : timestamp);
                        detection_results.header.stamp = timestamp_ros;

                        cv::Mat image = cv::imread(path_img);
                        if (image.empty()) {
                        ROS_ERROR_STREAM("[kitti_camera_ros] Could not read image: " << path_img);
                        return EXIT_FAILURE;
                        }

                        for(int i = 0; camera_txt.good() && !camera_txt.eof(); i++) {
                                vision_msgs::Detection2D detection;
                                vision_msgs::ObjectHypothesisWithPose result;
                                camera_txt >> detection.bbox.center.x;
                                camera_txt >> detection.bbox.center.y;
                                camera_txt >> detection.bbox.size_x;
                                camera_txt >> detection.bbox.size_y;
                                camera_txt >> result.id;
                                camera_txt >> result.score;
                                detection.results.push_back(result);
                                detection_results.detections.push_back(detection);
                                // std::cout<<"detection " << detection << std::endl;

                                cv::Rect bbox(detection.bbox.center.x - detection.bbox.size_x / 2, detection.bbox.center.y - detection.bbox.size_y / 2,detection.bbox.size_x, detection.bbox.size_y);

                                std::string label = std::to_string(result.id);  

                                if (label == "0"){
                                        label = "Car";
                                        cv::rectangle(image, bbox, cv::Scalar(0,0,255), 2);
                                }else if (label == "1")
                                {
                                        label = "Pedestrian";
                                        cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
                                }else if (label == "2")
                                {
                                        label = "Cyclist";
                                        cv::rectangle(image, bbox, cv::Scalar(255,0,0), 2);
                                }
                                
                                
                                
                                

                                // Draw the label
                                int baseline = 2;

                                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 2, 3, &baseline);
                                //make the linewidth wider

                                cv::Point labelPos(bbox.x, bbox.y);
                                // cv::rectangle(image, labelPos, cv::Point(labelPos.x + labelSize.width, labelPos.y + labelSize.height), cv::Scalar(0, 255, 0), cv::FILLED);
                                cv::putText(image, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 1);
                                
                        }

                        camera_txt.close();
                        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
                        img_pub.publish(*image_msg);
                        camera_pub.publish(detection_results);
                        count += 1;
                        std::cout << "in kitti camera ros" << count<< std::endl;
                }

                ros::spinOnce();
                loop_rate.sleep();
                i_file++;
        }

        for(int i = 2; i < n_file; i++) {
                free(filelist[i]);
        }
        free(filelist);

        return EXIT_SUCCESS;
}
