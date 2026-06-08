#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import random

import rospy
from pointnet_3d_box_stamped.msg import PointNet3DBoxStampedArray


PERSON_LABEL = "1"
UNKNOWN_LABEL = "9"
KNOWN_LABELS = (PERSON_LABEL, UNKNOWN_LABEL)


def normalize_label(label):
    return PERSON_LABEL if str(label) == PERSON_LABEL else UNKNOWN_LABEL


class ClassReservoir:
    def __init__(self, capacity_per_class, seed_value=1):
        self.capacity_per_class = max(0, int(capacity_per_class))
        self.buffers = {label: [] for label in KNOWN_LABELS}
        self.seen = {label: 0 for label in KNOWN_LABELS}
        self.rng = random.Random(seed_value)

    def add(self, sample):
        label = normalize_label(sample["label"])
        self.seen[label] += 1
        buf = self.buffers[label]
        if len(buf) < self.capacity_per_class:
            buf.append(sample)
            return

        replace_idx = self.rng.randrange(self.seen[label])
        if replace_idx < self.capacity_per_class:
            buf[replace_idx] = sample

    def full(self):
        return all(len(self.buffers[label]) >= self.capacity_per_class for label in KNOWN_LABELS)

    def sizes(self):
        return {label: len(self.buffers[label]) for label in KNOWN_LABELS}

    def samples(self):
        rows = []
        for label in KNOWN_LABELS:
            rows.extend(self.buffers[label])
        self.rng.shuffle(rows)
        return rows


def sample_from_box(box, frame_header):
    label = normalize_label(box.label)
    return {
        "label": label,
        "features": [float(value) for value in box.features],
        "id": int(box.id),
        "score": float(box.score),
        "stamp": float(frame_header.stamp.to_sec()),
        "frame_id": box.header.frame_id or frame_header.frame_id,
        "pose": {
            "position": {
                "x": float(box.pose.position.x),
                "y": float(box.pose.position.y),
                "z": float(box.pose.position.z),
            },
            "orientation": {
                "x": float(box.pose.orientation.x),
                "y": float(box.pose.orientation.y),
                "z": float(box.pose.orientation.z),
                "w": float(box.pose.orientation.w),
            },
        },
        "dimensions": {
            "x": float(box.dimensions.x),
            "y": float(box.dimensions.y),
            "z": float(box.dimensions.z),
        },
    }


class InitialSampleCollector:
    def __init__(self):
        self.output_file = os.path.expanduser(rospy.get_param("~output_file"))
        self.samples_per_class = int(rospy.get_param("~samples_per_class", 100))
        self.max_frames = int(rospy.get_param("~max_frames", 0))
        self.topic = rospy.get_param("~feature_topic", "/point_cloud_features_global/features_global")
        self.buffer = ClassReservoir(self.samples_per_class, seed_value=1)
        self.frames = 0
        self.done = False
        self.sub = rospy.Subscriber(self.topic, PointNet3DBoxStampedArray, self.callback, queue_size=1, buff_size=2**20)
        rospy.loginfo(
            "collect initial RF samples topic=%s output=%s samples_per_class=%d max_frames=%d",
            self.topic,
            self.output_file,
            self.samples_per_class,
            self.max_frames,
        )

    def callback(self, msg):
        if self.done:
            return
        if msg.number_of_samples == 0:
            return

        self.frames += 1
        for box in msg.fea_boxes:
            if not box.features:
                continue
            self.buffer.add(sample_from_box(box, msg.header))

        if self.frames % 30 == 1:
            rospy.loginfo(
                "collect initial RF samples frames=%d sizes=%s seen=%s",
                self.frames,
                self.buffer.sizes(),
                self.buffer.seen,
            )

        if self.buffer.full() or (self.max_frames > 0 and self.frames >= self.max_frames):
            self.write_samples()
            self.done = True
            rospy.signal_shutdown("initial sample collection complete")

    def write_samples(self):
        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        rows = self.buffer.samples()
        with open(self.output_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        rospy.loginfo(
            "wrote initial RF samples output=%s rows=%d sizes=%s seen=%s",
            self.output_file,
            len(rows),
            self.buffer.sizes(),
            self.buffer.seen,
        )


if __name__ == "__main__":
    rospy.init_node("collect_initial_rf_samples")
    InitialSampleCollector()
    rospy.spin()
