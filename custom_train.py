from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")

# Define data augmentation parameters
augmentation_params = {
    'flipud': 0.5,  # Vertical flip with 50% probability
    'fliplr': 0.5,  # Horizontal flip with 50% probability
    'degrees': 10.0,  # Rotate by ±10 degrees
    'scale': 0.1,  # Scale by ±10%
    'hsv_h': 0.015,  # Adjust hue by ±1.5%
    'hsv_s': 0.7,  # Adjust saturation by ±70%
    'hsv_v': 0.4,  # Adjust value (brightness) by ±40%
    'mosaic': 1.0,  # Enable mosaic augmentation
    'mixup': 0.5  # Enable mixup augmentation with 50% probability
}

# Train the model
train_results = model.train(
    data="data_custom.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    save_period=1, 
    augment=True,  # Enable augmentation
    **augmentation_params  # Pass augmentation parameters
)

# Evaluate model performance on the validation set
metrics = model.val()

# enable INT8 quantization
model.export(format="onnx", int8=True)

# Save the trained model
model.save("yolov8s_custom.pt")