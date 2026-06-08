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
import json
from collections import deque
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


def optional_known_label(label):
    label = str(label)
    return label if label in KNOWN_LABELS else ''


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
online_sample_count = 0
dropped_online_sample_count = 0
model_lock = None
replay_buffer_lock = None
training_queue_lock = None
training_queue_event = None
training_queue = None
latest_features_lock = None
latest_features_event = None
latest_features_msg = None


def enqueue_training_sample(x, y):
    global online_sample_count, dropped_online_sample_count

    if training_queue is None:
        return

    with training_queue_lock:
        if len(training_queue) == training_queue.maxlen:
            dropped_online_sample_count += 1
        training_queue.append((dict(x), normalize_label(y)))
        online_sample_count += 1
        training_queue_event.set()


def learn_with_replay(x, y):
    global sample_count, replay_sample_count

    y = normalize_label(y)
    with model_lock:
        model_train.learn_one(x, y)
    sample_count += 1

    with replay_buffer_lock:
        replay_buffer.add(x, y)
        replay_samples = replay_buffer.sample(replay_samples_per_class)

    for replay_x, replay_y in replay_samples:
        with model_lock:
            model_train.learn_one(replay_x, replay_y)
        replay_sample_count += 1


def get_replay_buffer_sizes():
    with replay_buffer_lock:
        return replay_buffer.sizes()


def get_training_queue_size():
    with training_queue_lock:
        return len(training_queue)


def do_train_task():
    while not rospy.is_shutdown():
        training_queue_event.wait(0.1)
        if rospy.is_shutdown():
            break

        sample = None
        with training_queue_lock:
            if training_queue:
                sample = training_queue.popleft()
            if not training_queue:
                training_queue_event.clear()

        if sample is None:
            continue

        learn_with_replay(sample[0], sample[1])


def fallback_prediction(label):
    if fallback_label:
        return normalize_label(fallback_label)
    return normalize_label(label)


def load_initial_samples(sample_file):
    if not sample_file:
        return []

    sample_file = os.path.expanduser(sample_file)
    if not os.path.exists(sample_file):
        rospy.logwarn("initial sample file does not exist: %s", sample_file)
        return []

    samples = []
    with open(sample_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                label = normalize_label(row['label'])
                features = row['features']
                x = {str(i): float(value) for i, value in enumerate(features)}
                samples.append((x, label))
            except Exception as exc:
                rospy.logwarn("skip bad initial sample %s:%d: %s", sample_file, line_no, exc)

    return samples


def train_initial_samples(sample_file, epochs):
    samples = load_initial_samples(sample_file)
    if not samples:
        rospy.logwarn("online RF starts without offline initial samples")
        return

    epochs = max(1, int(epochs))
    label_counts = {label: 0 for label in KNOWN_LABELS}
    for _, label in samples:
        label_counts[label] += 1

    for _ in range(epochs):
        for x, y in samples:
            learn_with_replay(x, y)

    rospy.loginfo(
        "online RF pre-trained from offline samples file=%s samples=%d label_counts=%s epochs=%d current_samples=%d replay_samples=%d replay_buffer=%s",
        os.path.expanduser(sample_file),
        len(samples),
        label_counts,
        epochs,
        sample_count,
        replay_sample_count,
        replay_buffer.sizes(),
    )



eval_dict = {}
eval_dict['test_samples'] = 0
eval_dict['no_det'] = 0
eval_dict['callback_cn'] = 0
eval_dict['wrong_det'] = 0
eval_dict['epoch'] = 0
eval_dict['confusion_matrix'] = metrics.ConfusionMatrix()

def features_callback(features_msg):
    global latest_features_msg

    with latest_features_lock:
        latest_features_msg = features_msg
        latest_features_event.set()


def do_predict_task():
    global latest_features_msg
    global save_dir, fallback_label
    global eval_dict,evalkitti

    while not rospy.is_shutdown():
        latest_features_event.wait(0.1)
        if rospy.is_shutdown():
            break

        with latest_features_lock:
            features_msg = latest_features_msg
            latest_features_msg = None
            latest_features_event.clear()

        if features_msg is None:
            continue

        eval_dict['callback_cn'] += 1

        if eval_dict['callback_cn'] % 30 == 1:
            rospy.loginfo(
                "online RF callback=%d test_samples=%d online_samples=%d dropped_online_samples=%d current_samples=%d replay_samples=%d buffer_sizes=%s training_queue=%d",
                eval_dict['callback_cn'],
                eval_dict['test_samples'],
                online_sample_count,
                dropped_online_sample_count,
                sample_count,
                replay_sample_count,
                get_replay_buffer_sizes(),
                get_training_queue_size(),
            )
        
        if eval_dict['callback_cn'] % 4722 == 0:
            eval_dict['epoch'] = eval_dict['callback_cn'] // 4722
            print('*'*15,'run out of dataset for ',eval_dict['epoch'],' epoches','*'*15)
            print(eval_dict['confusion_matrix'])
            with open(os.path.join(save_dir,'epoch_' + str(eval_dict['epoch'] - 1).zfill(2) + '.txt'),'w') as f:
                f.write(str(eval_dict['confusion_matrix']))
                print('*'*15,'save confusion_matrix for ',eval_dict['epoch'],' epoches','*'*15)   
                save_model_path = os.path.join(save_dir,'epoch_'+ str(eval_dict['epoch'] - 1).zfill(2) + '.pth')
                with model_lock:
                    joblib.dump(model_train,save_model_path)
        
        if eval_dict['callback_cn'] // 4721 == 5:
            with open(os.path.join(save_dir,'epoch_04.txt'),'w') as f:
                f.write(str(eval_dict['confusion_matrix']))
                print('*'*15,'save confusion_matrix for ',5,' epoches','*'*15)        
                save_model_path = os.path.join(save_dir,'epoch_04.pth') 
                with model_lock:
                    joblib.dump(model_train,save_model_path)

        rf_msg_array = DetectedObjectArray()
        rf_msg_array.header = features_msg.header
        result = []
        if (features_msg.number_of_samples != 0 ):
            for data in features_msg.fea_boxes:
                eval_dict['test_samples'] += 1 #统计测试次数
                x = {}
                for i, value in enumerate(data.features):
                    x[str(i)] = value
                label = normalize_label(data.label)
                res = {}
                with model_lock:
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
                    res['predict'] = fallback_prediction(data.label)
                    res['conf'] = 0.0
                    rospy.logwarn_throttle(2.0, 'can not get predict may be empty, using fallback label %s', res['predict'])

                if res['predict'] != label:
                    eval_dict['wrong_det'] += 1

                eval_dict['confusion_matrix'].update(label,res['predict'])
                enqueue_training_sample(x, label)

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
            if eval_dict['callback_cn'] % 30 == 1:
                rospy.loginfo(
                    "online RF rate=%.2f%% test=%d wrong=%d no_det=%d current_samples=%d replay_samples=%d online_samples=%d dropped_online_samples=%d",
                    rate,
                    eval_dict['test_samples'],
                    eval_dict['wrong_det'],
                    eval_dict['no_det'],
                    sample_count,
                    replay_sample_count,
                    online_sample_count,
                    dropped_online_sample_count,
                )
        else:
            rospy.logdebug('get empty frame')

        rf_msg_array.frame_out = features_msg.frame_out
        RF_label_pub.publish(rf_msg_array)
    


evalkitti = True


save_dir = os.path.expanduser('~/ocl3d_imf_workdir')
fallback_label = ''
replay_buffer_size_per_class = 512
replay_samples_per_class = 2
rf_n_estimators = 25
online_training_queue_size = 256
initial_samples_file = ''
initial_train_epochs = 1
if __name__ == '__main__':
    seed(1)
    rospy.init_node("random_forest_node_online")
    seq = rospy.get_param('/random_forest_node_online/scence')
    noise = rospy.get_param('/random_forest_node_online/noise')
    kitti = rospy.get_param('/random_forest_node_online/kitti')
    save_dir = os.path.expanduser(rospy.get_param('~save_dir', save_dir))
    fallback_label = optional_known_label(rospy.get_param('~fallback_label', fallback_label))
    replay_buffer_size_per_class = rospy.get_param('~replay_buffer_size_per_class', replay_buffer_size_per_class)
    replay_samples_per_class = rospy.get_param('~replay_samples_per_class', replay_samples_per_class)
    rf_n_estimators = rospy.get_param('~rf_n_estimators', rf_n_estimators)
    online_training_queue_size = rospy.get_param('~online_training_queue_size', online_training_queue_size)
    initial_samples_file = rospy.get_param('~initial_samples_file', initial_samples_file)
    initial_train_epochs = rospy.get_param('~initial_train_epochs', initial_train_epochs)
    os.makedirs(save_dir, exist_ok=True)

    model_lock = threading.Lock()
    replay_buffer_lock = threading.Lock()
    training_queue_lock = threading.Lock()
    training_queue_event = threading.Event()
    training_queue = deque(maxlen=max(1, int(online_training_queue_size)))
    latest_features_lock = threading.Lock()
    latest_features_event = threading.Event()
    replay_buffer = ClassReservoirReplayBuffer(replay_buffer_size_per_class, seed_value=1)
    rospy.loginfo(
        "online random forest labels=[%s,%s] fallback=%s replay_buffer_size_per_class=%s replay_samples_per_class=%s rf_n_estimators=%s online_training_queue_size=%s initial_samples_file=%s initial_train_epochs=%s",
        PERSON_LABEL,
        UNKNOWN_LABEL,
        fallback_label or "upstream_label",
        replay_buffer_size_per_class,
        replay_samples_per_class,
        rf_n_estimators,
        online_training_queue_size,
        initial_samples_file,
        initial_train_epochs,
    )
    model_train = forest.AMFClassifier(
        n_estimators=rf_n_estimators,
        use_aggregation=True,
        dirichlet=0.5,
        seed=1
    )
    RF_label_pub = rospy.Publisher("/online_random_forest/rf_label", DetectedObjectArray, queue_size=1)

    load_weights = rospy.get_param('~load_weights', False)
    model_file_name = os.path.expanduser(rospy.get_param('~model_file_name', ''))

    if load_weights and model_file_name:
        model_train = joblib.load(model_file_name)

    train_initial_samples(initial_samples_file, initial_train_epochs)

    feature_sub = rospy.Subscriber("/point_cloud_features_global/features_global", PointNet3DBoxStampedArray, features_callback, queue_size=1, buff_size=2**20)
    train_thread = threading.Thread(target=do_train_task)
    train_thread.daemon = True
    train_thread.start()
    predict_thread = threading.Thread(target=do_predict_task)
    predict_thread.daemon = True
    predict_thread.start()
    rospy.loginfo('start asynchronous online incremental learning')
    rospy.spin()
