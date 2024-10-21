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

def loadtxt(path):
    results = []
    with open(path,'r') as f:
        lines = f.readlines()
        
        for line in lines:
            tokens = line.split('\n')
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

    global eval_dict
    eval_dict['callback_cn'] += 1
    
    print('*'*15,'online incremental testing starting','*'*15,'with ',eval_dict['callback_cn'],' callback function')



    print('current epoch is ',eval_dict['epoch'],'current callback_cn is ',eval_dict['callback_cn'])
    rf_msg_array = DetectedObjectArray()
    result = []
    balance_weight = 0.6
    if (features_msg.number_of_samples != 0 ):

        for data in features_msg.fea_boxes:
            print(features_msg.frame_out,data.header.seq)
            eval_dict['test_samples'] += 1 
            x = {}
            for i, value in enumerate(data.features):
                x[str(i)] = value

            res = {}
            predict = model_train.predict_proba_one(x)
            try:
                res['predict'] = max(predict, key=predict.get)
            except:
                eval_dict['no_det'] += 1
                print('can not get predict may be empty')
                continue
            res['conf'] =  data.score
            if data.flag:
                proba = (balance_weight * data.score) / ((1 - balance_weight) * predict[res['predict']]) 
                if proba > 1 or res['predict'] is None:
                    res['predict'] = data.label
                    res['conf'] = data.score
                else:
                    pass

            if res['predict'] != data.label:
                eval_dict['wrong_det'] += 1

            eval_dict['confusion_matrix'].update(data.label,res['predict'])

            
            res['pose'] = data.pose
            res['dimensions'] = data.dimensions
            res['frame'] = features_msg.frame_out
            res['flag'] = data.flag
            
            result.append(res)
            rf_msg = DetectedObject()
            rf_msg.header = features_msg.header
       
            rf_msg.label = res['predict']
            rf_msg.flag = data.flag
            rf_msg.pose= data.pose
            rf_msg.dimensions = data.dimensions
            rf_msg.header.frame_id = "velodyne"
            rf_msg_array.objects.append(rf_msg)
        rate = (eval_dict['test_samples'] - eval_dict['wrong_det'] - eval_dict['no_det']) / eval_dict['test_samples'] * 100
        print(f'total rate is {rate}% with test ',eval_dict['test_samples'], 'samples,wrong det ',eval_dict['wrong_det'], 'samples,no det', eval_dict['no_det'],'samples')
        


        
    else:
        print('get empty frame ')
    
    print(sample_nums)
    if eval_dict['callback_cn'] % sample_nums == 0:
        eval_dict['epoch'] = eval_dict['callback_cn'] // sample_nums
        print('*'*15,'run out of dataset for ',eval_dict['epoch'],' epoches','*'*15)
        print(eval_dict['confusion_matrix'])
        with open(os.path.join(save_dir,'epoch_' + str(eval_dict['epoch'] - 1).zfill(2) + '.txt'),'w') as f:
            f.write(str(eval_dict['confusion_matrix']))
            print('*'*15,'save confusion_matrix for ',eval_dict['epoch'],' epoches','*'*15)   
            save_model_path = os.path.join(save_dir,'epoch_'+ str(eval_dict['epoch'] - 1).zfill(2) + '.pth')
            joblib.dump(model_train,save_model_path)
    
    if eval_dict['callback_cn'] // sample_nums == 5:
        with open(os.path.join(save_dir,'epoch_04.txt'),'w') as f:
            f.write(str(eval_dict['confusion_matrix']))
            print('*'*15,'save confusion_matrix for ',5,' epoches','*'*15)        
            save_model_path = os.path.join(save_dir,'epoch_04.pth') 
            joblib.dump(model_train,save_model_path)    

    rf_msg_array.frame_out = features_msg.frame_out
    rf_msg_array.header.frame_id = "velodyne"
    RF_label_pub.publish(rf_msg_array)




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
    car_dict = {'0': 0.5158819556236267, '1': 0.46460089087486267, '2': -0.46693626046180725, '3': 0.037744879722595215, '4': -0.2983928620815277, '5': -0.6285813450813293, '6': -1.1644072532653809, '7': -0.018713094294071198, '8': 0.16956248879432678, '9': 1.1744883060455322, '10': 0.38838469982147217, '11': 0.01991802453994751, '12': 0.7275718450546265, '13': 0.3433399498462677, '14': -0.25049862265586853, '15': 1.0639324188232422, '16': 0.3432977497577667, '17': -0.6357495188713074, '18': 0.7574660181999207, '19': -0.2220393419265747, '20': -0.8823767304420471, 
    '21': -0.45736610889434814, '22': 0.3128064274787903, '23': -0.6317265033721924, '24': 0.40848419070243835, '25': -0.0589144304394722, '26': 0.55250084400177, '27': 0.43164190649986267, '28': -0.8620143532752991, '29': 0.11800336092710495, '30': -0.7851263880729675, '31': -0.8576516509056091, '32': -0.7491078972816467, '33': 0.4367942214012146, '34': -1.070931077003479, '35': -0.5368137955665588, '36': -0.4858717620372772, '37': -0.7191183567047119, '38': -0.09514150023460388, '39': -1.0629186630249023, '40': 0.3304608464241028, '41': -0.011352424509823322, 
    '42': -0.8085067272186279, '43': -0.23460181057453156, '44': 0.5494269132614136, '45': 0.8756844997406006, '46': -0.22693978250026703, '47': 0.2476288378238678, '48': 0.06766197085380554, '49': -0.10988238453865051, '50': 0.7098011374473572, '51': 0.4041394293308258, '52': -0.12019500881433487, '53': 0.592487633228302, '54': -0.30186572670936584, '55': 1.0386040210723877, '56': 0.6525334119796753, '57': 1.0875600576400757, '58': 0.38144344091415405, '59': -0.16054996848106384, '60': 0.0925634503364563}
    ped_dict = {'0': -0.21227599680423737, '1': -0.4151626527309418, '2': -0.1069561243057251, '3': 1.210011601448059, '4': 0.2640308439731598, '5': 0.11661799997091293, '6': 0.2629205882549286, '7': 1.2273460626602173, '8': -0.11567755043506622, '9': 1.1059306859970093, '10': -0.0942678451538086, '11': 0.0700659528374672, '12': 0.607570230960846, '13': 0.9353980422019958, '14': -0.48701751232147217, '15': -0.9032691717147827, '16': 1.0914511680603027, '17': 0.22874665260314941, '18': 0.5646218657493591, '19': 0.15286794304847717, '20': -0.41836169362068176, 
    '21': -0.1377459466457367, '22': -0.19572634994983673, '23': -1.4874659776687622, '24': 1.1871007680892944, '25': 0.5249699354171753, '26': 0.4045926630496979, '27': 1.0787533521652222, '28': -0.09275994449853897, '29': 1.340361475944519, '30': -0.03441968932747841, '31': 0.4857548177242279, '32': -0.0017625633627176285, '33': -0.27819839119911194, '34': 0.02089362032711506, '35': 0.2205306440591812, '36': 0.948790431022644, '37': 0.22333122789859772, '38': 1.3984094858169556, '39': -0.0030909087508916855, '40': 0.33895719051361084, '41': -0.1437559276819229, 
    '42': -0.13629932701587677, '43': 0.7038841247558594, '44': 0.5387366414070129, '45': 0.4342055022716522, '46': -0.15432500839233398, '47': -0.44320690631866455, '48': -0.4718688130378723, '49': -0.1324043571949005, '50': 0.7783534526824951, '51': 0.5155394077301025, '52': -0.6723066568374634, '53': -0.6478142142295837, '54': 0.740097165107727, '55': 0.7457046508789062, '56': 0.4294067621231079, '57': 0.03140582889318466, '58': -0.41063278913497925, '59': -1.6458427906036377, '60': 0.40228694677352905}
    cyc_dict = {'0': 0.01252756081521511, '1': -0.25056198239326477, '2': -0.13016536831855774, '3': 0.6320714354515076, '4': 0.004072541370987892, '5': -0.1283288598060608, '6': 0.4032718539237976, '7': 1.3788244724273682, '8': 0.14251235127449036, '9': 1.3944952487945557, '10': 0.34041398763656616, '11': 0.12452966719865799, '12': 1.0774633884429932, '13': 1.1644033193588257, '14': -0.2877926528453827, '15': -0.6435136795043945, '16': 0.9150018692016602, '17': 0.16668811440467834, '18': 0.6119072437286377, '19': 0.5563963055610657, '20': -0.7057510018348694, '21': 0.06354548782110214, '22': 0.23681186139583588, '23': -1.1762173175811768, '24': 0.9022822380065918, '25': 0.1198912039399147, '26': 0.721418023109436, '27': 0.7665329575538635, '28': 0.027366913855075836, '29': 1.39188551902771, '30': -0.09084278345108032, '31': -0.006816159002482891, '32': -0.168186217546463, '33': -0.4281992018222809, '34': -0.3422708213329315, '35': -0.0611138716340065, '36': 0.6478186845779419, '37': 0.200370192527771, '38': 1.3196529150009155, '39': 0.04501120001077652, '40': 0.5854645371437073, '41': 0.2086259126663208, '42': -0.12350621074438095, '43': 0.9290116429328918, '44': 0.7482350468635559, '45': 0.5309380292892456, '46': -0.011018412187695503, '47': 0.03336665779352188, '48': -0.41810035705566406, '49': -0.4202966094017029, '50': 0.7590486407279968, '51': 0.7828821539878845, '52': -0.29635414481163025, '53': -0.20043615996837616, '54': 0.7245672941207886, '55': 0.8686888217926025, '56': 1.0337045192718506, '57': -0.06447068601846695, '58': -0.48897314071655273, '59': -1.272535800933838, '60': 0.3591497242450714}
    
    
    RF_label_pub = rospy.Publisher("/online_random_forest/rf_label", DetectedObjectArray, queue_size=10)
    
    init_train = True
    if init_train:
        model_train.learn_one(car_dict, '0')        
        model_train.learn_one(ped_dict, '1') 
        model_train.learn_one(cyc_dict, '2') 

    

    load_weights = False
    if load_weights:
        model_train = joblib.load(f'/home/tianbot/online_learning_ws_mini/src/kitti-object-eval-python/result_dir/kitti_dt/pillars_imf/{seq}/epoch_00.pth')
    

    
    gt_dir = f'/home/tianbot/online_learning_ws_mini/src/kitti-object-eval-python/groundtruth_dir/{seq}'

    sample_nums = len(os.listdir(gt_dir))
    feature_sub = rospy.Subscriber("/point_cloud_features_global/features_global", PointNet3DBoxStampedArray, features_callback,queue_size=100)#pointnet

    print('*'*15,'start online incremental learning','*'*15)
    try:
        train_thread = threading.Thread(target=do_train_task)
        train_thread.start()
    except:
        print ("Error: can't start thread")



