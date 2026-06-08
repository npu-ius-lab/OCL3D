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
import time
from rospy.rostime import Time
import copy
import os   
from river import evaluate
from river import forest
from river import metrics



import sys
sys.setrecursionlimit(1000000)

PERSON_LABEL = "1"
UNKNOWN_LABEL = "9"
KNOWN_LABELS = (PERSON_LABEL, UNKNOWN_LABEL)


def normalize_label(label):
    return PERSON_LABEL if str(label) == PERSON_LABEL else UNKNOWN_LABEL


class ClassReservoirReplayBuffer:
    def __init__(self, capacity_per_class, seed_value=1):
        self.capacity_per_class = max(0, int(capacity_per_class))
        self.buffers = {label: [] for label in KNOWN_LABELS}
        self.seen = {label: 0 for label in KNOWN_LABELS}
        self.rng = random.Random(seed_value)

    def add(self, x, y):
        y = normalize_label(y)
        if self.capacity_per_class <= 0:
            return

        self.seen[y] += 1
        sample = (dict(x), y)
        buf = self.buffers[y]
        if len(buf) < self.capacity_per_class:
            buf.append(sample)
            return

        replace_idx = self.rng.randrange(self.seen[y])
        if replace_idx < self.capacity_per_class:
            buf[replace_idx] = sample

    def sample(self, samples_per_class):
        samples_per_class = max(0, int(samples_per_class))
        if samples_per_class <= 0:
            return []

        replay = []
        for label in KNOWN_LABELS:
            buf = self.buffers[label]
            if not buf:
                continue
            k = min(samples_per_class, len(buf))
            replay.extend(self.rng.sample(buf, k))
        self.rng.shuffle(replay)
        return replay

    def sizes(self):
        return {label: len(self.buffers[label]) for label in KNOWN_LABELS}


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
replay_sample_count = 0


def learn_with_replay(x, y):
    global sample_count, replay_sample_count

    y = normalize_label(y)
    model_train.learn_one(x, y)
    sample_count += 1
    replay_buffer.add(x, y)

    for replay_x, replay_y in replay_buffer.sample(replay_samples_per_class):
        model_train.learn_one(replay_x, replay_y)
        replay_sample_count += 1



eval_dict = {}
eval_dict['test_samples'] = 0
eval_dict['no_det'] = 0
eval_dict['callback_cn'] = 0
eval_dict['wrong_det'] = 0
eval_dict['epoch'] = 0
eval_dict['confusion_matrix'] = metrics.ConfusionMatrix()
def features_callback(features_msg):
    global save_dir, fallback_label

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
    rf_msg_array.header = features_msg.header
    result = []
    if (features_msg.number_of_samples != 0 ):
        for data in features_msg.fea_boxes:
            print(features_msg.frame_out,data.header.seq)
            eval_dict['test_samples'] += 1 #统计测试次数
            x = {}
            for i, value in enumerate(data.features):
                x[str(i)] = value
            label = normalize_label(data.label)
            res = {}
            predict = model_train.predict_proba_one(x) #预测的结果
            try:
                normalized_predict = {}
                for pred_label, score in predict.items():
                    pred_label = normalize_label(pred_label)
                    normalized_predict[pred_label] = normalized_predict.get(pred_label, 0.0) + score
                res['predict'] = max(normalized_predict, key=normalized_predict.get)
                predict = normalized_predict
                res['conf'] =  predict[res['predict']]
            except:
                eval_dict['no_det'] += 1#统计未分类的次数
                res['predict'] = normalize_label(fallback_label or data.label)
                res['conf'] = 0.0
                print('can not get predict may be empty, using fallback label', res['predict'])

            if res['predict'] != label:
                eval_dict['wrong_det'] += 1
            
            eval_dict['confusion_matrix'].update(label,res['predict'])
            learn_with_replay(x, label)

            res['pose'] = data.pose
            res['dimensions'] = data.dimensions
            res['frame'] = features_msg.frame_out
            

            result.append(res)
            rf_msg = DetectedObject()
            rf_msg.header = data.header if data.header.frame_id else features_msg.header
 
            rf_msg.label = res['predict']
            rf_msg.score = res['conf']
            rf_msg.pose= data.pose
            rf_msg.dimensions = data.dimensions
            rf_msg.valid = True
            rf_msg_array.objects.append(rf_msg)
        rate = (eval_dict['test_samples'] - eval_dict['wrong_det'] - eval_dict['no_det']) / eval_dict['test_samples'] * 100
        print(f'total rate is {rate}% with test ',eval_dict['test_samples'], 'samples,wrong det ',eval_dict['wrong_det'], 'samples,no det', eval_dict['no_det'],'samples')
        print('learned current samples', sample_count, 'replay samples', replay_sample_count, 'buffer sizes', replay_buffer.sizes())
        


    else:
        print('get empty frame ')
    
 

    rf_msg_array.frame_out = features_msg.frame_out

    RF_label_pub.publish(rf_msg_array)
    


evalkitti = True


save_dir = os.path.expanduser('~/ocl3d_imf_workdir')
fallback_label = ''
replay_buffer_size_per_class = 512
replay_samples_per_class = 8
if __name__ == '__main__':
    seed(1)
    rospy.init_node("random_forest_node_online")
    seq = rospy.get_param('/random_forest_node_online/scence')
    noise = rospy.get_param('/random_forest_node_online/noise')
    kitti = rospy.get_param('/random_forest_node_online/kitti')
    save_dir = os.path.expanduser(rospy.get_param('~save_dir', save_dir))
    fallback_label = normalize_label(rospy.get_param('~fallback_label', fallback_label))
    replay_buffer_size_per_class = rospy.get_param('~replay_buffer_size_per_class', replay_buffer_size_per_class)
    replay_samples_per_class = rospy.get_param('~replay_samples_per_class', replay_samples_per_class)
    os.makedirs(save_dir, exist_ok=True)
    replay_buffer = ClassReservoirReplayBuffer(replay_buffer_size_per_class, seed_value=1)
    rospy.loginfo(
        "online random forest labels=[%s,%s] fallback=%s replay_buffer_size_per_class=%s replay_samples_per_class=%s",
        PERSON_LABEL,
        UNKNOWN_LABEL,
        fallback_label,
        replay_buffer_size_per_class,
        replay_samples_per_class,
    )
    model_train = forest.AMFClassifier(
        n_estimators=50,
        use_aggregation=True,
        dirichlet=0.5,
        seed=1
    )
    RF_label_pub = rospy.Publisher("/online_random_forest/rf_label", DetectedObjectArray, queue_size=10)

    load_weights = rospy.get_param('~load_weights', False)
    model_file_name = os.path.expanduser(rospy.get_param('~model_file_name', ''))

    if load_weights and model_file_name:
        model_train = joblib.load(model_file_name)


    feature_sub = rospy.Subscriber("/point_cloud_features_global/features_global", PointNet3DBoxStampedArray, features_callback,queue_size=100)
    print('*'*15,'start online incremental learning','*'*15)
    rospy.spin()
