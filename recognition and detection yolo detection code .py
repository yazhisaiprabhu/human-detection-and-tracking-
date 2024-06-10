import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from collections import deque



# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video capture
cap = cv2.VideoCapture('/content/Video.mp4')  # Provide your video file here
output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize a set to store unique object IDs
unique_ids = set()

# Tracking information
trackers = []
trackableObjects = {}
nextObjectID = 0

class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.disappeared = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
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
            if confidence > 0.5 and classes[class_id] == 'person':
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    rects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            rects.append([x, y, x+w, y+h])
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update trackers
    objects = {}
    for rect in rects:
        x1, y1, x2, y2 = rect
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        min_dist = float('inf')
        objectID = None
        for oid, to in trackableObjects.items():
            dist = np.linalg.norm(np.array(to.centroids[-1]) - np.array(centroid))
            if dist < min_dist:
                min_dist = dist
                objectID = oid

        if objectID is None or min_dist > 50:
            objectID = nextObjectID
            nextObjectID += 1

        if objectID in trackableObjects:
            trackableObjects[objectID].centroids.append(centroid)
            trackableObjects[objectID].disappeared = 0
        else:
            trackableObjects[objectID] = TrackableObject(objectID, centroid)

        objects[objectID] = centroid
        unique_ids.add(objectID)  # Add to unique ID set

    # Display the results
    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

    cv2_imshow(frame)
    output_video.write(frame)

cap.release()
output_video.release()
cv2.destroyAllWindows()

# Print the total number of unique students
print(f"Total number of unique students: {len(unique_ids)}")
