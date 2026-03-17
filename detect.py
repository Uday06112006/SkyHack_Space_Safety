from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    degrees=180,
    translate=0.2,
    scale=0.5,
    shear=10,
    perspective=0.001
)
results = model.predict(
    source="/content/dataset/train/images/001-29-_jpeg.rf.07313a59b01c605e18ac4ced2e53961e.jpg",
    conf=0.15,
    show_labels=True,
    show_conf=False,
    save=True
)
