#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR_CLASSES = {2, 5, 7}  # car, bus, truck

OCL3D_CAR = 0
OCL3D_PEDESTRIAN = 1
OCL3D_CYCLIST = 2
OCL3D_UNKNOWN = 9


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def union_box(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def make_detection(box, cls_id, score):
    x1, y1, x2, y2 = box
    detection = Detection2D()
    detection.bbox.center.x = (x1 + x2) / 2.0
    detection.bbox.center.y = (y1 + y2) / 2.0
    detection.bbox.center.theta = 0.0
    detection.bbox.size_x = max(0.0, x2 - x1)
    detection.bbox.size_y = max(0.0, y2 - y1)

    result = ObjectHypothesisWithPose()
    result.id = int(cls_id)
    result.score = float(score)
    detection.results.append(result)
    return detection


class YoloV8CameraRos:
    def __init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'ultralytics' is required. Install with: "
                "python3 -m pip install --user ultralytics"
            ) from exc

        self.image_topic = rospy.get_param("~image_topic", "/hikrobot_camera/rgb/compressed")
        self.model_path = rospy.get_param("~model", "yolov8n.pt")
        self.confidence = float(rospy.get_param("~confidence", 0.25))
        self.nms_iou = float(rospy.get_param("~iou", 0.45))
        self.cyclist_iou = float(rospy.get_param("~cyclist_iou", 0.4))
        self.device = rospy.get_param("~device", "")
        self.person_unknown_only = bool(rospy.get_param("~person_unknown_only", True))
        self.publish_person_when_cyclist = bool(rospy.get_param("~publish_person_when_cyclist", False))
        self.visualize = bool(rospy.get_param("~visualize", True))

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.detections_pub = rospy.Publisher("/image_detections", Detection2DArray, queue_size=10)
        self.image_pub = rospy.Publisher("/image_vis", Image, queue_size=2)
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(
            "[yolov8_camera_ros] model=%s image_topic=%s conf=%.2f iou=%.2f cyclist_iou=%.2f device=%s person_unknown_only=%s",
            self.model_path,
            self.image_topic,
            self.confidence,
            self.nms_iou,
            self.cyclist_iou,
            self.device if self.device else "auto",
            self.person_unknown_only,
        )

    def image_callback(self, msg):
        frame = self.decode_image(msg)
        if frame is None:
            return

        classes = [COCO_PERSON] if self.person_unknown_only else [COCO_PERSON, COCO_BICYCLE, 2, 5, 7]
        kwargs = {
            "conf": self.confidence,
            "iou": self.nms_iou,
            "verbose": False,
            "classes": classes,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)
        detections, draw_items = self.convert_results(results[0])

        out = Detection2DArray()
        out.header = msg.header
        out.detections = detections
        self.detections_pub.publish(out)

        if self.visualize and self.image_pub.get_num_connections() > 0:
            vis = frame.copy()
            self.draw_detections(vis, draw_items)
            image_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            image_msg.header = msg.header
            self.image_pub.publish(image_msg)

    @staticmethod
    def decode_image(msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn_throttle(2.0, "[yolov8_camera_ros] failed to decode compressed image")
        return frame

    def convert_results(self, result):
        cars = []
        persons = []
        bicycles = []

        if result.boxes is None:
            return [], []

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(xyxy, cls, conf):
            item = (tuple(float(v) for v in box), float(score))
            if self.person_unknown_only:
                if cls_id == COCO_PERSON:
                    persons.append(item)
            elif cls_id in COCO_CAR_CLASSES:
                cars.append(item)
            elif cls_id == COCO_PERSON:
                persons.append(item)
            elif cls_id == COCO_BICYCLE:
                bicycles.append(item)

        detections = []
        draw_items = []
        used_persons = set()
        used_bicycles = set()

        if not self.person_unknown_only:
            for person_idx, (person_box, person_score) in enumerate(persons):
                best_bicycle_idx = None
                best_iou = -math.inf
                for bicycle_idx, (bicycle_box, _) in enumerate(bicycles):
                    if bicycle_idx in used_bicycles:
                        continue
                    iou = box_iou(person_box, bicycle_box)
                    if iou >= self.cyclist_iou and iou > best_iou:
                        best_iou = iou
                        best_bicycle_idx = bicycle_idx

                if best_bicycle_idx is None:
                    continue

                bicycle_box, bicycle_score = bicycles[best_bicycle_idx]
                cyclist_box = union_box(person_box, bicycle_box)
                cyclist_score = min(person_score, bicycle_score)
                detections.append(make_detection(cyclist_box, OCL3D_CYCLIST, cyclist_score))
                draw_items.append((cyclist_box, "Cyclist", cyclist_score, (255, 0, 0)))
                used_persons.add(person_idx)
                used_bicycles.add(best_bicycle_idx)

            for box, score in cars:
                detections.append(make_detection(box, OCL3D_CAR, score))
                draw_items.append((box, "Car", score, (0, 0, 255)))

        for person_idx, (box, score) in enumerate(persons):
            if person_idx in used_persons and not self.publish_person_when_cyclist:
                continue
            detections.append(make_detection(box, OCL3D_PEDESTRIAN, score))
            draw_items.append((box, "Pedestrian", score, (0, 255, 0)))

        return detections, draw_items

    @staticmethod
    def draw_detections(image, draw_items):
        for box, label, score, color in draw_items:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = "{} {:.2f}".format(label, score)
            text_y = max(18, y1 - 5)
            cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main():
    rospy.init_node("yolov8_camera_ros")
    try:
        YoloV8CameraRos()
    except RuntimeError as exc:
        rospy.logfatal("[yolov8_camera_ros] %s", exc)
        raise SystemExit(1)
    rospy.spin()


if __name__ == "__main__":
    main()
