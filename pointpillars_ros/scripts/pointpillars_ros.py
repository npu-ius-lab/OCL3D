#! /usr/bin/env python
# coding:utf-8
import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2,PointField,Image
from sensor_msgs import point_cloud2
import copy
import os
import cv2
import time
from mmdet3d.apis import init_model, inference_detector,show_result_meshlab
from mmdet3d.core.points import get_points_type
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import math
from geometry_msgs.msg import Point
import open3d as o3d
from mmdet3d.models import build_model


class pointpillar_ros():
    def __init__(self):
        
        self.root_file = '/home/tianbot/online_learning_git/src/pointpillars_ros/weights'
        self.config_file = os.path.join(self.root_file,'config.py')
        self.checkpoint_file = os.path.join(self.root_file,'epoch_80.pth')

        

        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        self.model.eval()
        rospy.init_node("pointpillars")

        scence = rospy.get_param('/pointpillars/scence', '0000')
        self.output = rospy.get_param('/pointpillars/output', False)
        self.result_dir = f'/home/tianbot/online_learning_git/src/pointpillars_ros/result/{scence}'
        self.calib_path = f'/home/tianbot/online_learning_git/src/autoware_tracker/config/calib/{scence}.txt' 
        
        if not os.path.exists(self.result_dir):
           os.mkdir(self.result_dir)
        
        self.box =  rospy.Publisher("/pp_boxes", PointCloud2, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/points_raw", PointCloud2, self.detections)
        self.ego_car_pub = rospy.Publisher("/ego_car", Marker, queue_size=10)
        self.points_in_cam = rospy.Publisher('/points_cam',PointCloud2,queue_size=1)

        self.publisher_vis_box = rospy.Publisher('/detected_objects_vis_box',BoundingBoxArray,queue_size=1)
        self.publisher_vis_txt = rospy.Publisher('/detected_objects_vis_txt',MarkerArray,queue_size=1)
        self.score_thr = 0.0
        self.max_marker = 0
        self.range = [0, -39.68, -3, 69.12, 39.68, 1]
        
        rospy.spin()


    def read_calib(self,path_calib):
            with open(path_calib, 'r') as f:
                lines = f.readlines()
                P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                R0 = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
                V2C = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            return P2,R0,V2C


        
    def publish_ego_car(self):
        #publish left and right 45 degree FOV lines and ego car model mesh
        
        #marker_array=MarkerArray()

        marker=Marker()
        marker.header.frame_id='velodyne'
        marker.header.stamp=rospy.Time.now()

        marker.id=10000
        marker.action=Marker.ADD
        marker.lifetime=rospy.Duration()
        marker.type=Marker.LINE_STRIP


        marker.color.r=0.0
        marker.color.g=1.0
        marker.color.b=0.0
        marker.color.a=1.0
        marker.scale.x=0.2


        marker.points=[]
        marker.points.append(Point(30,-30,0))
        marker.points.append(Point(0,0,0))
        marker.points.append(Point(30,30,0))

        self.ego_car_pub.publish(marker)        


    def crop_pc(self,point_cloud,pred_bbox):
        pcd = o3d.geometry.PointCloud()
        if point_cloud.shape[1] == 4:
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
            colors = np.hstack((point_cloud[:,3].reshape(-1,1),point_cloud[:,3].reshape(-1,1),point_cloud[:,3].reshape(-1,1)))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6])
        vis = o3d.visualization.Visualizer()
        vis.create_window() 
        box = o3d.geometry.OrientedBoundingBox()

        box.center = np.array([pred_bbox[0],pred_bbox[1],pred_bbox[2] + 0.5 *pred_bbox[5]])

        box.extent = np.array([pred_bbox[3],pred_bbox[4],pred_bbox[5]])
        rot_y = -pred_bbox[6] - math.pi/2
        box.R = R.from_euler('xyz', [0, 0, rot_y], degrees=False).as_matrix()


        box.rotate(box.R,box.center)

        points_in_box = pcd.crop(box)

        ###vis
        cropped_pcd = o3d.geometry.PointCloud(points_in_box)
        vis.add_geometry(box)
        vis.add_geometry(cropped_pcd)
        
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        ###vis
        numpy_xyz = np.asarray(points_in_box.points)
        numpy_i = np.asarray(points_in_box.colors)[:,0].reshape(-1,1)
        numpy_pcd = np.hstack([numpy_xyz,numpy_i])

        return numpy_pcd

    def det3d_postprocess_vis(self,results,pc, data,score_thr, CLASSES,header):
      """

      Args:
          result (_type_): _description_
          score_thr (_type_): _description_
          CLASSES (_type_): _description_
          frame_id (str, optional): _description_. Defaults to "map".

      Returns:
          _type_: _description_
      """

      result = results[0]

      if 'pts_bbox' in result.keys():
          pred_bboxes = result['pts_bbox']['boxes_3d'].tensor.numpy()
          pred_scores = result['pts_bbox']['scores_3d'].numpy()
          pred_labels = result['pts_bbox']['labels_3d'].numpy()
      else:
          pred_bboxes = result['boxes_3d'].tensor.numpy()
          pred_scores = result['scores_3d'].numpy()
          pred_labels = result['labels_3d'].numpy()
      

      print(len(pred_bboxes),len(pred_bboxes),len(pred_scores))
      # 框
      inds = np.where(pred_scores > score_thr)
      pred_scores = pred_scores[inds] ##重要

      pred_bboxes = pred_bboxes[inds]

      detected_object_array = BoundingBoxArray()
      detected_object_array.header = header
      pred_labels = pred_labels[inds]
      print(len(pred_bboxes),len(pred_bboxes),len(pred_scores))
      

  
      arr_marker = MarkerArray()
    
      for i in range(len(pred_bboxes)):
            pred_bbox = pred_bboxes[i]

            if (not self.isinsidecam(pred_bbox)):
                continue
            
            detected_object = BoundingBox()
            detected_object.header = header
            detected_object.value = pred_scores[i]
            label = 0
            

            if CLASSES[pred_labels[i]] == 'Car':
                label = 0
            elif CLASSES[pred_labels[i]] == 'Pedestrian':
                label = 1
            elif CLASSES[pred_labels[i]] == 'Cyclist':
                label = 2

            detected_object.label = label

            # xyz
            detected_object.pose.position.x = pred_bbox[0]
            detected_object.pose.position.y = pred_bbox[1]
            detected_object.pose.position.z = pred_bbox[2]

            # lwh
            detected_object.dimensions.x = pred_bbox[3]
            detected_object.dimensions.y = pred_bbox[4]
            detected_object.dimensions.z = pred_bbox[5]

            # yaw
            yaw = pred_bbox[6]
            
            detected_object.pose.position.z += detected_object.dimensions.z / 2


            # get quaternion from yaw axis euler angles
            r = R.from_euler('z', yaw, degrees=False)
            q = r.as_quat()
            detected_object.pose.orientation.x = q[0]
            detected_object.pose.orientation.y = q[1]
            detected_object.pose.orientation.z = q[2]
            detected_object.pose.orientation.w = q[3]

            detected_object_array.boxes.append(detected_object)

            ####out_result###
            
            if (self.output):
                self.tokitti(detected_object,header)



            ####标签#####
            marker = Marker()
            marker.header.frame_id = 'velodyne'
            # marker.header.stamp = header.stamp
            marker.type = Marker.TEXT_VIEW_FACING
            marker.ns = "PP"
            marker.id = i
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration(0.1)
            marker.color.r = 0
            marker.color.b = 0
            marker.color.g = 255
            marker.color.a = 0.8  
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 1.0

            marker.pose.position.x = pred_bbox[0]
            marker.pose.position.y = pred_bbox[1]
            marker.pose.position.z = pred_bbox[2]
            marker.text = CLASSES[pred_labels[i]]    
            arr_marker.markers.append(marker)
      return detected_object_array,arr_marker


    
    def camera2pixel(self,pc,P2):
        pc_hom = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32))) # [N, 4]
        pc_hom_C2 = np.dot(pc_hom, P2.T) # [N, 3] 
        pc_hom_C2 = (pc_hom_C2[:, :2].T / pc_hom_C2[:, 2]).T
        
        max_x = max(pc_hom_C2[:,0])
        min_x = min(pc_hom_C2[:,0])
        max_y = max(pc_hom_C2[:,1])
        min_y = min(pc_hom_C2[:,1])
        return max_x,max_y,min_x,min_y


    def isinsidecam(self,pred_bbox):
        x = pred_bbox[0]
        y = pred_bbox[1]
        if (x + y) >= 0 and (x - y) >= 0:
            return True
        else:
            return False

    
    def lidar_to_camera_pose(self,pose, r_rect, velo2cam):
        """Convert points in lidar coordinate to camera coordinate.

        Note:
            This function is for KITTI only.

        Args:
            points (np.ndarray, shape=[N, 3]): Points in lidar coordinate.
            r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
                specific camera coordinate (e.g. CAM2) to CAM0.
            velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
                camera coordinate to lidar coordinate.

        Returns:
            np.ndarray, shape=[N, 3]: Points in lidar coordinate.
        """
        
        points = np.zeros(shape=(1,4))

        points[0][0] = pose.position.x
        points[0][1] = pose.position.y
        points[0][2] = pose.position.z
        points[0][3] = 1
        
        
        if r_rect.shape[-1] == 3:
            r_rect = np.vstack([r_rect,np.array([0,0,0])])
            r_rect = np.hstack([r_rect,np.array([0,0,0,1]).reshape(-1,1)])
            r_rect = np.eye(4)


        lidar_points = np.dot(points, np.dot(velo2cam.T, r_rect.T)) #right
        return lidar_points[..., :3]


        
          




    def cut_pc(self,point_cloud,calib_path):
        x_min = self.range[0]
        y_min = self.range[1]
        z_min = self.range[2]
        x_max = self.range[3]
        y_max = self.range[4]
        z_max = self.range[5]

        selected_points = (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) & \
            (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) & \
            (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
        cropped_point_cloud = point_cloud[selected_points]



        return cropped_point_cloud
    
    def convert2lidar(self,pose):
        res = np.zeros(shape=(1,3))
        res[0][0] = -pose.position.y
        res[0][1] = -pose.position.z
        res[0][2] = pose.position.x

        return res

    def tokitti(self,result,header):
        res = {}
        frame = str(header.seq).zfill(6)
        txt_path = os.path.join(self.result_dir,frame + '.txt')

        res['length'] = result.dimensions.x
        res['width'] = result.dimensions.y
        res['height'] = result.dimensions.z
        
        res['pose'] = copy.deepcopy(result.pose)
        
        
        P2,R0,V2C = self.read_calib(self.calib_path)
        V2C = np.vstack([V2C,[0,0,0,1]])
        
        res['camera_location'] = self.lidar_to_camera_pose(res['pose'],R0,V2C)
        
        orientation = res['pose'].orientation

        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler_angles = r.as_euler('xyz', degrees=False) 
        roll, pitch, yaw = euler_angles
        res['rot_y'] = -yaw - math.pi/2
        if result.label == 0:
           res['label'] = 'Car'
        elif result.label == 1:
           res['label'] = 'Pedestrian'
        else:
           res['label'] = 'Cyclist'
        res['score'] = result.value
        with open(txt_path,'a+') as f:
            f.write(res['label'])
            f.write(' -1 -1 -10 -1 -1 -1 100 ')
            f.write(str(res['height']) + ' ')
            f.write(str(res['width']) + ' ')
            f.write(str(res['length']) + ' ')

            f.write(str(res['camera_location'][0][0]) + ' ')
            f.write(str(res['camera_location'][0][1] + res['height'] / 2) + ' ')
            f.write(str(res['camera_location'][0][2]) + ' ')
            f.write(str(res['rot_y']) + ' ' )
            f.write(str(res['score']) + '\n')



    def read_pc(self,path_pc):
        points = np.fromfile(path_pc, dtype=np.float32).reshape(-1, 4)
        return points
    def detections(self,data):

        pc = ros_numpy.numpify(data)
        points=np.zeros((pc.shape[0],4))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        points[:,3]=pc['intensity']


        pc_in = points
        points_class = get_points_type('LIDAR')
        points_mmdet3d = points_class(points, points_dim=pc_in.shape[-1], attribute_dims=None)

        result, data_ = inference_detector(self.model, points_mmdet3d)
    
        result_visbox,marker_arr = self.det3d_postprocess_vis(result,pc_in, data_,self.score_thr, self.model.CLASSES,data.header)

        self.publisher_vis_box.publish(result_visbox)
        self.publisher_vis_txt.publish(marker_arr)
        self.publish_ego_car()
        
      



if __name__ == '__main__': 
  try:
    clip = pointpillar_ros()
  except rospy.ROSInterruptException:
    pass




