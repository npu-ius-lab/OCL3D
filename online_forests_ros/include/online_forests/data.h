#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <vector>
#include <gmm/gmm.h>
#include <string>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include "pointnet_3d_box_stamped/PointNet3DBoxStamped.h"
#include "pointnet_3d_box_stamped/PointNet3DBoxStampedArray.h"
using namespace std;
using namespace gmm;

// TYPEDEFS
typedef int Label;
typedef double Weight;

typedef rsvector<double> SparseVector;
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
// DATA CLASSES
class Sample {
public:
    SparseVector x;
    Label y;
    Weight w;
    
    int seq;
    float stamp;

    geometry_msgs::Pose pose;
    geometry_msgs::Vector3 dimensions;
    int id;
    float position_x;
    float position_y;
    float position_z;

    // float dimensions_x;
    // float dimensions_y;
    // float dimensions_z;

    std_msgs::Header header;
    void disp() {
        cout << "Sample: y = " << y << ", w = " << w << ", x = ";
        cout << x << endl;
    }
};

class DataSet {
public:
    vector<Sample> m_samples;
    int m_numSamples;
    int m_numFeatures;
    int m_numClasses;

    vector<double> m_minFeatRange;
    vector<double> m_maxFeatRange;

    void findFeatRange();

    void loadLIBSVM(string filename);
    void loadLIBSVM2(string data);
    //lbh loadLIBSVM3 used for testing and visual 
    void loadLIBSVM3(string data);

    void loadLIBSVM4(const pointnet_3d_box_stamped::PointNet3DBoxStampedArray& data); //new msgs

};

class Result {
public:
    vector<double> confidence;
    Sample sing_sample;
    int correct_samples;
    int prediction;
};

#endif /* DATA_H_ */
