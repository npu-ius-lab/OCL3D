#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import joblib
import csv
from random import seed
from math import sqrt
import rospy
from std_msgs.msg import String
from autoware_tracker.msg import DetectedObjectArray,DetectedObject
from pointnet_3d_box_stamped.msg import PointNet3DBoxStampedArray,PointNet3DBoxStamped
import random
import matplotlib.pyplot as plt
import threading
import time
from rospy.rostime import Time
import copy
import os   
from river import evaluate
from river import forest
from river import metrics



import sys
sys.setrecursionlimit(1000000)


def isValidToken(token):
    if token and len(token) > 100:
        return True
    else:
        return False

def loadmsg(msg):
    tokens = msg.data.split('\n')

    results = []
    for token in tokens:
        if not isValidToken(token):
            continue
        res = {}
        res['label'] = token[0]

        space_idx = token.find(' ',0)

        res['id'] = token[1:space_idx]

        data_str = token[space_idx + 1:]
        dict_data = {}

        key_value_pairs = data_str.split()
        for pair in key_value_pairs:
            key, value = pair.split(':')
            key = str(int(key) - 1)
            dict_data[key] = float(value)
        res['feature'] = dict_data
        results.append(res)
    return results


sample_count = 0
def do_train_task():
    global sample_count
    while not rospy.is_shutdown():
        print('*'*15,'online incremental training starting','*'*15)
        point_feature = rospy.wait_for_message('/point_cloud_features/features', String)
        start = time.time()
        if point_feature.data:
            samples = loadmsg(point_feature)
            for sample in samples:
                model_train.learn_one(sample['feature'],sample['label'])
                sample_count += 1
        end = time.time()
  
        print(f'learned {sample_count} samples')



eval_dict = {}
eval_dict['test_samples'] = 0
eval_dict['no_det'] = 0
eval_dict['callback_cn'] = 0
eval_dict['wrong_det'] = 0
eval_dict['epoch'] = 0
eval_dict['confusion_matrix'] = metrics.ConfusionMatrix()
def features_callback(features_msg):
    global save_dir

    global eval_dict,evalkitti
    eval_dict['callback_cn'] += 1

    print('*'*15,'online incremental testing starting','*'*15,'with ',eval_dict['callback_cn'],' callback function')
    
    if eval_dict['callback_cn'] % 4722 == 0:
        eval_dict['epoch'] = eval_dict['callback_cn'] // 4722
        print('*'*15,'run out of dataset for ',eval_dict['epoch'],' epoches','*'*15)
        print(eval_dict['confusion_matrix'])
        with open(os.path.join(save_dir,'epoch_' + str(eval_dict['epoch'] - 1).zfill(2) + '.txt'),'w') as f:
            f.write(str(eval_dict['confusion_matrix']))
            print('*'*15,'save confusion_matrix for ',eval_dict['epoch'],' epoches','*'*15)   
            save_model_path = os.path.join(save_dir,'epoch_'+ str(eval_dict['epoch'] - 1).zfill(2) + '.pth')
            joblib.dump(model_train,save_model_path)
    
    if eval_dict['callback_cn'] // 4721 == 5:
        with open(os.path.join(save_dir,'epoch_04.txt'),'w') as f:
            f.write(str(eval_dict['confusion_matrix']))
            print('*'*15,'save confusion_matrix for ',5,' epoches','*'*15)        
            save_model_path = os.path.join(save_dir,'epoch_04.pth') 
            joblib.dump(model_train,save_model_path)

    print('current epoch is ',eval_dict['epoch'],'current callback_cn is ',eval_dict['callback_cn'])
    rf_msg_array = DetectedObjectArray()
    result = []
    if (features_msg.number_of_samples != 0 ):
        for data in features_msg.fea_boxes:
            print(features_msg.frame_out,data.header.seq)
            eval_dict['test_samples'] += 1 #统计测试次数
            x = {}
            for i, value in enumerate(data.features):
                x[str(i)] = value
            res = {}
            predict = model_train.predict_proba_one(x) #预测的结果
            try:
                res['predict'] = max(predict, key=predict.get)
            except:
                eval_dict['no_det'] += 1#统计未分类的次数
                print('can not get predict may be empty')
                continue

            if res['predict'] != data.label:
                eval_dict['wrong_det'] += 1
            
            eval_dict['confusion_matrix'].update(data.label,res['predict'])

            res['conf'] =  predict[res['predict']]
            res['pose'] = data.pose
            res['dimensions'] = data.dimensions
            res['frame'] = features_msg.frame_out
            

            result.append(res)
            rf_msg = DetectedObject()
            rf_msg.header = features_msg.header
 
            rf_msg.label = res['predict']
            rf_msg.pose= data.pose
            rf_msg.dimensions = data.dimensions
            rf_msg.header.frame_id = "velodyne"
            rf_msg_array.objects.append(rf_msg)
        rate = (eval_dict['test_samples'] - eval_dict['wrong_det'] - eval_dict['no_det']) / eval_dict['test_samples'] * 100
        print(f'total rate is {rate}% with test ',eval_dict['test_samples'], 'samples,wrong det ',eval_dict['wrong_det'], 'samples,no det', eval_dict['no_det'],'samples')
        


    else:
        print('get empty frame ')
    
 

    rf_msg_array.frame_out = features_msg.frame_out

    rf_msg_array.header.frame_id = "velodyne"
    RF_label_pub.publish(rf_msg_array)
    


evalkitti = True


save_dir = '/home/tianbot/online_learning_ws_mini/data_kitti/iusl_key_frame/IMF_workdir/iusl_11_5_pointnet/clear'
if __name__ == '__main__':
    seed(1)
    rospy.init_node("random_forest_node_online")
    seq = rospy.get_param('/random_forest_node_online/scence')
    noise = rospy.get_param('/random_forest_node_online/noise')
    kitti = rospy.get_param('/random_forest_node_online/kitti')
    model_train = forest.AMFClassifier(
        n_estimators=50,
        use_aggregation=True,
        dirichlet=0.5,
        seed=1
    )
    RF_label_pub = rospy.Publisher("/online_random_forest/rf_label", DetectedObjectArray, queue_size=10)

    load_weights = False

    if load_weights:
        model_train = joblib.load('/home/tianbot/online_learning_ws_mini/data_kitti/iusl_key_frame/IMF_workdir/iusl_11_5_pointnet/clear/epoch_04.pth')#clear


    feature_sub = rospy.Subscriber("/point_cloud_features_global/features_global", PointNet3DBoxStampedArray, features_callback,queue_size=100)
    print('*'*15,'start online incremental learning','*'*15)
    try:
        train_thread = threading.Thread(target=do_train_task)
        train_thread.start()
    except:
        print ("Error: can't start thread")



