import cv2
import datetime
import numpy as np
from centroidtracker import CentroidTracker
import imutils
from itertools import combinations
import math

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Initialize CentroidTracker
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    cap = cv2.VideoCapture('testvideo2.mp4')
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    skip_frames = 5  # Process every 5th frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        total_frames += 1
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip frame

        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to person in COCO
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, x + w, y + h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        rects = [boxes[i] for i in indexes.flatten()]

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)

        objects = tracker.update(boundingboxes)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(objects.items(), 2):
            distance = calculate_distance(p1, p2)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for id, box in objects.items():
            color = (0, 0, 255) if id in red_zone_list else (0, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Social Distancing", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
