import cv2
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.predict(source='sample.jpg', save=True)

if __name__ == "__main__":