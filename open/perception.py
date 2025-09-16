import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# Video source (0 = default webcam, or path to video file)
VIDEO_SOURCE = 0

# MQTT broker configuration
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 8080        # WebSocket port
MQTT_TOPIC = "traffic/edge/metrics"

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # small YOLO model

# Initialize MQTT client with WebSocket transport
mqttc = mqtt.Client(transport="websockets")
mqttc.connect(, MQTT_PORT, 60)

# Time of last publish
last_publish = time.time()

# Open video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame, imgsz=640)

    # Count vehicles (car, bike, bus, truck)
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # car, bike, bus, truck
                count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Publish every second
    now = time.time()
    if now - last_publish > 1:
        data = {
            "intersection_id": "INT_001",
            "approach_id": "North",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "vehicle_count": count,
            "queue_len_m": count * 5  # assume 5m per vehicle
        }
        mqttc.publish(MQTT_TOPIC, json.dumps(data))
        print("Published:", data)
        last_publish = now

    # Show video
    cv2.imshow("Traffic Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
