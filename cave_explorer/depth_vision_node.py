#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image

class Node:
    """ROS node class."""

    def __init__(self, conf, labels, threshold_vals):
        """Constructor for the Node class."""
        rospy.init_node('node_name')  # Replace 'node_name' with your node's name

        self.conf = conf
        self.labels = labels
        self.threshold_vals = threshold_vals

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Publisher for the camera detections
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # Read in computer vision model (simple starting point)
        self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        # self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)
        self.computer_vision_model_ = YOLO(self.computer_vision_model_filename_)  # Load your custom trained model
        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/depth/image_raw", Image, self.image_callback, queue_size=1)


    def image_callback(self, image_msg):
        """Callback function for the topic subscriber.

        Args:
            msg: The received message.
        """
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        depth_array = np.array(image, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        depth_array = (depth_array*255).astype(np.uint8)
        depth_array = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)

        # Retrieve the pre-trained model
        model = self.computer_vision_model_

        # Perform object detection on an image
        results = model.predict(depth_array, conf=self.conf)

        # num_detections = len(detections)
        num_detections = len(results)

        if num_detections > 0:
            self.artifact_found_ = True
        else:
            self.artifact_found_ = False

        # Draw custom bounding boxes for objects with confidence > 80%
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)  # Get confidence score
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates
                label = box.cls.item()  # Convert Tensor to a standard Python type
                index = int(label)
                
                # Draw the bounding box
                if confidence > self.threshold_vals[index]:
                    cv2.rectangle(depth_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    y_text = int(y1) - 10
                    if y_text >= 0:
                        cv2.putText(depth_array, f"{self.labels[index]} {confidence:.2f}", (int(x1), y_text), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(depth_array, encoding="bgr8")
        self.image_detections_pub_.publish(image_detection_message)

        rospy.loginfo('image_callback')
        rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))

if __name__ == '__main__':
    node = Node(0.8, ['alien','glacier','mushroom','white sphere', 'green rock'], [0.8, 0.8, 0.8, 0.8, 0.8])
    rospy.spin()