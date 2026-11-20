from ultralytics import YOLO

model = YOLO("helmet_detection_best.pt")

image_path = "helmet_yolo_dataset/images/test/BikesHelmets604.png"

results = model.predict(
    image_path,
    imgsz=640,
    conf=0.05,
    save=True,
    project="new_model_test",
    name="prediction",
    exist_ok=True
)

print("Classes:", model.names)
print("Total detections:", len(results[0].boxes))
print("Saved prediction image in: new_model_test/prediction/")