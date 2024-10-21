// (c) 2020 Zhi Yan, Rui Yang
// This code is licensed under MIT license (see LICENSE.txt for details)
#define GMM_USES_BLAS

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h>
// Online Random Forests
#include "online_forests/onlinetree.h"
#include "online_forests/onlinerf.h"
//autoware
#include "autoware_tracker/DetectedObject.h"
#include "autoware_tracker/DetectedObjectArray.h"
#include "pointnet_3d_box_stamped/PointNet3DBoxStamped.h"
#include "pointnet_3d_box_stamped/PointNet3DBoxStampedArray.h"
#include <pthread.h>
#include "tf/tf.h"


// #include <tf/tf.h>
ros::Publisher _pub_rf_objects;
int count_ = 0;
int count_callback = 0;

void pub_msg(vector<Result> _results){
  //lbh
  count_ += 1;
  std::cout<<"count_online forest pub time is " <<  count_ << std::endl;
  
  autoware_tracker::DetectedObjectArray ORF_obj_array;
  ORF_obj_array.header =  _results[0].sing_sample.header;
  for(int i = 0; i < _results.size(); i++){
    float socre = _results[i].confidence[_results[i].prediction];
    autoware_tracker::DetectedObject ORF_obj;
    if (socre > 0.5){
      // std::cout << "test result" << _results[i].prediction << std::endl;
      // std::cout << " result x y" << _results[i].sing_sample.position_x<<_results[i].sing_sample.position_y << std::endl;


      // std::cout << "sing_sample.pose " << _results[i].sing_sample.pose << std::endl;
      // double rotationY = 0;
      // // 欧拉角
      // // //四元数转欧拉角
      // tf::Quaternion q;
      // tf::quaternionMsgToTF(_results[i].sing_sample.pose.orientation, q);
      // std::cout << "orign orientation " << _results[i].sing_sample.pose.orientation<<std::endl;
      // // 将四元数转换为旋转矩阵
      // tf::Matrix3x3 rotation(q);

      // // 将旋转矩阵转换为欧拉角
      // double roll, pitch, yaw;
      // rotation.getRPY(roll, pitch, yaw);
      // rotationY = -yaw ;
      // std::cout << " ou la jiao orign " << rotationY << std::endl;
      // rotationY = -yaw ;
      // std::cout << " ou la jiao after " << rotationY << std::endl;    


      // // 四元数
      // tf::Quaternion bounding_box_quat = tf::createQuaternionFromRPY(0.0, 0.0,  - rotationY);
      // tf::quaternionTFToMsg(bounding_box_quat, _results[i].sing_sample.pose.orientation);

      // std::cout << "bounding_box_quat" << _results[i].sing_sample.pose.orientation <<std::endl;


      ORF_obj.id = _results[i].sing_sample.id;

      ORF_obj.score = _results[i].confidence[_results[i].prediction];
      ORF_obj.label = std::to_string(_results[i].prediction);
      
      ORF_obj.pose = _results[i].sing_sample.pose;



      
      std::cout << "ORF_obj.pose " << ORF_obj.pose << std::endl;

      ORF_obj.dimensions = _results[i].sing_sample.dimensions;


      ORF_obj.header = _results[i].sing_sample.header;
      ORF_obj_array.objects.push_back(ORF_obj);
    }
  }
  _pub_rf_objects.publish(ORF_obj_array);
}

int main(int argc, char **argv) {
  std::ofstream icra_log;
  std::string log_name = "orf_time_log_"+std::to_string(ros::WallTime::now().toSec());
  
  std::string conf_file_name;
  std::string model_file_name;
  string model_file_name_save;
  int mode; // 1 - train, 2 - test, 3 - train and test.
  int minimum_samples;
  int total_samples = 0;
  ros::init(argc, argv, "online_forests_ros");
  ros::NodeHandle nh, private_nh("~");
  _pub_rf_objects = nh.advertise<autoware_tracker::DetectedObjectArray>("/online_random_forest/rf_label", 1);

  if(private_nh.getParam("conf_file_name", conf_file_name)) {
    ROS_INFO("Got param 'conf_file_name': %s", conf_file_name.c_str());
  } else {
    ROS_ERROR("Failed to get param 'conf_file_name'");
    exit(EXIT_SUCCESS);
  }

  if(private_nh.getParam("model_file_name", model_file_name)) {
    ROS_INFO("Got param 'model_file_name': %s", model_file_name.c_str());
  } else {
    ROS_ERROR("Failed to get param 'model_file_name'");
    exit(EXIT_SUCCESS);
  }


  if(private_nh.getParam("mode", mode)) {
    ROS_INFO("Got param 'mode': %d", mode);
  } else {
    ROS_ERROR("Failed to get param 'mode'");
    exit(EXIT_SUCCESS);
  }

  private_nh.param<int>("minimum_samples", minimum_samples, 1);
  Hyperparameters hp(conf_file_name);
  std_msgs::String::ConstPtr features_test;
  std_msgs::String::ConstPtr features_train;
  pointnet_3d_box_stamped::PointNet3DBoxStampedArray::ConstPtr features_train_new_msgs;
  
  //correct total
  int correct_total= 0;
  int numsamples_total = 0;
  while (ros::ok()) {
    features_train_new_msgs= ros::topic::waitForMessage<pointnet_3d_box_stamped::PointNet3DBoxStampedArray>("/point_cloud_features_global/features_global"); // process blocked waiting
    count_callback += 1;
    std::cout<<"count_callback in forest  time is " <<  count_callback << std::endl;
    // features_train= ros::topic::waitForMessage<std_msgs::String>("/point_cloud_features_global/features_global"); // process blocked waiting

    // features_train = ros::topic::waitForMessage<std_msgs::String>("/point_cloud_features/features"); // process blocked waiting
    // Creating the train data
    DataSet dataset_tr;
    dataset_tr.loadLIBSVM4(*features_train_new_msgs);
    std::cout << " 96 " <<std::endl;

    // Creating the test data
    DataSet dataset_ts;
    // dataset_ts.loadLIBSVM3(features_test->data);
    // dataset_ts.loadLIBSVM(hp.testData);
    //保存结果
    int correct_cur = 0;
    vector<Result> results;
    if(features_train_new_msgs->number_of_samples >= minimum_samples) {
      OnlineRF model(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange); // TOTEST: OnlineTree

      icra_log.open(log_name, std::ofstream::out | std::ofstream::app);
      time_t start_time = ros::WallTime::now().toSec();

      switch(mode) {
      case 1: // train only
        if(access( model_file_name.c_str(), F_OK ) != -1){
          model.loadForest(model_file_name);
        }
        
        model.train(dataset_tr);
        model.writeForest(model_file_name); //turning off the writing function
        break;
      case 2: // test only
        model.loadForest(model_file_name);
        results = model.test(dataset_tr);
        
        for(int i = 0;i < results.size();i++ ){
          if(results[i].correct_samples == 1)
          {
            std::cout<< "label is " <<results[i].prediction<<std::endl;
            correct_cur++;
          }
        }
        std::cout<< "current correct nums is " <<correct_cur<<std::endl;
        std::cout<< "current  samples nums is " <<results.size()<<std::endl;
        numsamples_total += results.size();
        correct_total += correct_cur;
        std::cout<< "current total correct  is " <<correct_total<<std::endl;
        std::cout<< "current total samples nums is " <<numsamples_total<<std::endl;   
        std::cout<< "total rate is " << correct_total * 1.0 / numsamples_total<<std::endl;      
        pub_msg(results);
        break;

      case 3: //train and test
        results = model.trainAndTest(dataset_tr, dataset_ts);
        pub_msg(results);
        break;
      default:
        ROS_ERROR("Unknown 'mode'");
      }

      std::cout << "[online_forests_ros] Training time: " << ros::WallTime::now().toSec() - start_time << " s" << std::endl;
      icra_log << (total_samples+=dataset_tr.m_numSamples) << " " << ros::WallTime::now().toSec()-start_time << "\n";
      icra_log.close();
    }

    ros::spinOnce();
  }

  return EXIT_SUCCESS;
}
