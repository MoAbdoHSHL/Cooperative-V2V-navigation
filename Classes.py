#!/usr/bin/env python3
import os
import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
from ultralytics import YOLO

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

class_names = ['pathholde', 'Manholde', 'Duck', 'Snow', 'Stop']

def initialize_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        buffer_count=3,
        transform=Transform(hflip=1, vflip=1)
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({"ScalerCrop": (0, 0, 4608, 2592)})
    return picam2

def initialize_model():
    model = YOLO("/home/mohamed/Desktop/Code/Cooperative-V2V-navigation/runs/detect/train2/weights/best.pt").to("cpu")
    model.fuse()
    model.amp = True
    _ = model(np.zeros((160, 160, 3), dtype=np.uint8), imgsz=160, verbose=False)
    return model

class FrameCaptureThread(threading.Thread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            frame = self.camera.capture_array()
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False

def process_frame(frame, model):
    results = model(frame, imgsz=160, conf=0.5, half=True, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated_frame

def main():
    camera = initialize_camera()
    model = initialize_model()
    frame_thread = FrameCaptureThread(camera)
    frame_thread.start()

    try:
        frame_count = 0
        while True:
            frame = frame_thread.get_frame()
            if frame is None:
                continue

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            start_time = time.time()
            annotated = process_frame(frame, model)
            print(f"Inference time: {time.time() - start_time:.2f} sec")

            cv2.imshow("Road Hazard Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        frame_thread.stop()
        frame_thread.join()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
