#! /usr/bin/env python
# coding:utf-8
# Created by Yao dexin

from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torch.utils.data
from unicodedata import name
from numpy.core.fromnumeric import shape
from std_msgs.msg import String
import time
import numpy as np
import rospy
import ros_numpy as rnp
from sensor_msgs.msg import PointCloud2
import torch.nn.parallel
import torch.utils.data
import sys
from autoware_tracker.msg import DetectedObjectArray,DetectedObject
# sys.path.append("/home/tianbot/efficent_online_ws/src/pointnet.pytorch")
from torch.autograd import Variable
# from pointnet.dataset import ShapeNetDataset
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc = torch.nn.Linear(1024, 61)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.fc(x)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=3, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(61, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        trans_feat = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

def callback(data):

    start = time.time()
    minimum_points = 5
    number_of_samples_count = 0
    number_of_samples = 0
    number_of_car_count = 0
    number_of_ped_count = 0
    number_of_cyc_count = 0
    pub_msg = String()
    pub_msg.data += "61 3 1\n"
    for object in data.objects:
        if len(object.pointcloud.data)/32 >= minimum_points:
            if (object.label == 'unknown'):
                continue

            _rec_data = object.pointcloud
            convert_data = rnp.numpify(_rec_data)
            points=np.zeros((convert_data.shape[0],4))
            points[:,0]=convert_data['x']
            points[:,1]=convert_data['y']
            points[:,2]=convert_data['z']
            # points[:,3]=convert_data['label']
            # print(points[:,0:3].T.shape)
            _data = torch.from_numpy(np.array([points[:,0:3].T]))
            _data = _data.cuda()
            _data = _data.type(torch.cuda.FloatTensor)
            # print("data_input_size",_data.size())
            out, _2, pointfea = cls(_data)
            # print("result_size",pointfea.size())

            pointfea = pointfea.tolist()
            # print(pointfea)
            # pub_msg.data += object.label + str(object.id)
            pub_msg.data += object.label
            for j in range(len(pointfea[0])):
                pub_msg.data += " " + str(j+1) + ":" + str(pointfea[0][j])
            pub_msg.data += "\n"

            if object.label == '0':
                number_of_car_count += 1
            elif object.label == '1':
                number_of_ped_count += 1
            elif object.label == '2':
                number_of_cyc_count += 1
            number_of_samples_count += 1
            number_of_samples += 1
    pub_msg.data = str(number_of_samples)+" " + pub_msg.data
    end = time.time()
    print('in learning features time is ',end - start)
    if number_of_samples > 0:      
        pub.publish(pub_msg)

if __name__ == '__main__':
    pretrained_dict = torch.load('/home/tianbot/online_learning_ws_mini/src/point_cloud_features/scripts/cls_model_249.pth') #模型load
    cls = PointNetCls(k = 3)
    cls.cuda()
    cls.load_state_dict(pretrained_dict,strict=False)
    cls.eval()
    rospy.init_node("pointnet_fea_ros")


    pub =  rospy.Publisher("/point_cloud_features/features", String, queue_size=10)
    sub = rospy.Subscriber("/autoware_tracker/tracker/examples", DetectedObjectArray, callback)

    rospy.spin()