#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
DATA_PATH='/home/tianbot/efficient_online_learning/src/datasets/2011_09_26_drive_0005_sync/'

if __name__=='__main__':
    rospy.init_node('kitti_node',anonymous=True)
    cam_pub=rospy.Publisher('kitti_cam',Image,queue_size=10)
    bridge=CvBridge()
    rate=rospy.Rate(5)
    frame=0

    while not rospy.is_shutdown():
        img=cv2.imread(DATA_PATH+'image_00/data/' + str(frame).zfill(10)+".png")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
                
        im = Image()
        im.encoding = 'rgb8'            
        im.header.stamp = rospy.Time.now()
        im.header.frame_id = 'result'    
        im.height = image.shape[0]
        im.width = image.shape[1]
        im.step = image.shape[1] * image.shape[2]
        im.data = np.array(image).tostring()

        cam_pub.publish(im)
        # rospy.loginfo('camera iamge published')
        rate.sleep()
        frame+=1
        frame%=154

