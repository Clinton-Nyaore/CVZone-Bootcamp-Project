from ultralytics import YOLO
import numpy as np

def detect_objects(frame):
    """Perform object detection on the frame using the YOLO model."""
    model = YOLO("./models/nano_detect_goats.pt")
    results = model.predict(frame)[0]
    detections = np.empty((0, 5))
    detections_list = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x, y, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf_f = box.conf[0]
        w, h = x2 - x, y2 - y

        if conf_f > 0.5:
            new_array = np.array([x, y, x2, y2, conf_f])
            detections = np.vstack((detections, new_array))
            conf = conf_f
            detections_list.append([x, y, w, h])

    return conf, detections, detections_list