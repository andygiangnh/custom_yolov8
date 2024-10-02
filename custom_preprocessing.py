import cv2
import os

def convert_to_greyscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_dir, filename), grey_img)

# Convert training images
convert_to_greyscale('E:\\UTS\\Semester 2\\49274_Space_Robotics\\Assignment3\\yolov8\\train\\images', \
    'E:\\UTS\\Semester 2\\49274_Space_Robotics\\Assignment3\\yolov8\\images_greyscale\\train')

# Convert validation images
convert_to_greyscale('F:\\UTS\\Semester 2\\49274_Space_Robotics\\Assignment3\\yolov8\\val\\images', \
    'E:\\UTS\\Semester 2\\49274_Space_Robotics\\Assignment3\\yolov8\\images_greyscale\\val')
