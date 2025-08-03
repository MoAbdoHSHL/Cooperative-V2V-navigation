import cv2
from ultralytics import YOLO

model = YOLO("D:/Project_Work/MyDataSet/runs/detect/train2/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

class_names = ['patholde', 'Manholde', 'Duck', 'Snow', 'Stop']

print("🚀 Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break


    results = model.predict(frame, imgsz=640, conf=0.4)  


    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls_id]} ({conf:.2f})"

          
            x1, y1, x2, y2 = map(int, box.xyxy[0])
         
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Detection - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection ended.")

