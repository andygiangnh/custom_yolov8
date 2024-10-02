from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("yolov8s_custom.pt")  # Load your custom trained model
labels = ['alien','green rock','glacier','mushroom','whitedot','stop']

# Load an image
# image_path = "/home/andy/mysource/yolov8/test/alien_11.jpg"
# image_path = "/home/andy/mysource/yolov8/test/greenrock_62.jpg"
image_path = "/home/andy/mysource/yolov8/test/glacier_694.jpg"
# image_path = "/home/andy/mysource/yolov8/test/mushroom_81.jpg"
# image_path = "/home/andy/mysource/yolov8/test/whitedot_196.jpg"
# image_path = "/home/andy/mysource/yolov8/test/stop_483.jpg"
image = cv2.imread(image_path)

# grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Replicate the grayscale image to have three channels
# gray_3_channel = cv2.merge([grey_img, grey_img, grey_img])

# Perform object detection on an image
# results = model.predict(gray_3_channel, conf=0.8)
results = model.predict(image, conf=0.4)
print(f'result found: {len(results)}')
# results[0].show()

# Draw custom bounding boxes for objects with confidence > 80%
for result in results:
    print(f'result boxes {len(result.boxes)}')
    for box in result.boxes:
        confidence = float(box.conf)  # Get confidence score
        print(f'confidence score {confidence}')
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates
        label = box.cls.item()  # Convert Tensor to a standard Python type
        
        # Draw the bounding box
        y_text = int(y1) - 10
        if y_text <= 0:
            y_text = 10
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{labels[int(label)]} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the image with bounding boxes
cv2.imwrite("output.jpg", image)
cv2.imshow("Detected Objects", image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
