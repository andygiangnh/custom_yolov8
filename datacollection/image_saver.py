#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import numpy as np
from random import randint

# Instantiate CvBridge
bridge = CvBridge()

name_prefix = 'glacier'

def image_callback(msg):
    global name_prefix
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        img_name = f'{name_prefix}_{randint(1, 1000)}.jpg'
        cv2.imwrite(img_name, cv2_img)
        
        print(f'image {img_name} is saved!')
        # Shutdown ros
        rospy.signal_shutdown("Image saved, shutting down.")

# this method used for depth camera
def image_callback2(msg):
    print("Received an image from depth camera!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "passthrough")

        depth_array = np.array(cv2_img, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        # print(msg.encoding)        

    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite(f"depth{randint(1, 100)}.png", depth_array*255)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/rgb/image_raw"
    image_topic_depth = "/camera/depth/image_raw"
    # Set up your subscriber and define its callback
    sub1 = None
    sub1 = rospy.Subscriber(image_topic, Image, image_callback)
    
    # sub2 = rospy.Subscriber(image_topic_depth, Image, image_callback2)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()