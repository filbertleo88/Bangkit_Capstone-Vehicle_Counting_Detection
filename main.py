import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import csv
from firebase import firebase

cap = cv2.VideoCapture("videos/test2.mp4")  # For Video
# cap = cv2.VideoCapture(1) # For Webcam

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Custom model
model = YOLO("models/model_custom.pt")

# Custom classes
classNames=["big bus", "big truck", "bus-l-", "bus-s-", "car", "mid truck", "small bus", "small truck",
            "truck-l-", "truck-m-", "truck-s-", "truck-xl-"]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line
limits = [30, 450, 568, 450]
limits1 = [680, 450, 1300, 450]

# Counter
totalCount = []
totalCount1 = []

capacity = input("Enter rest area capacity: ")
firebase =firebase.FirebaseApplication('https://interstate-now-default-rtdb.asia-southeast1.firebasedatabase.app/',None)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            label = f'{currentClass} {conf}'

            if currentClass == 'car':
                color = (0, 204, 255)
            elif currentClass == "big bus" or currentClass == "bus-l-" or currentClass == "bus-s-"\
                  or currentClass == "small bus":
                color = (222, 82, 175)
            elif currentClass == "big truck" or currentClass == "mid truck" or currentClass == "big bus"\
                  or currentClass == "small truck" or currentClass == "truck-l-" or currentClass == "truck-m-"\
                    or currentClass == "truck-s-" or currentClass == "truck-xl-":
                color = (0, 149, 255)
            else:
                color = (255, 255, 255)

            if currentClass in classNames and conf > 0.4:
                cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=5, colorR=color)
                cvzone.putTextRect(img, f"{label} ", (max(0, x1), max(35, y1)), scale=1, thickness=2,
                                   offset=5, colorR=color)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)


    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        # Center Point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Counting
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # Change line color
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if limits1[0] < cx < limits1[2] and limits1[1] - 20 < cy < limits1[3] + 20:
            if totalCount1.count(id) == 0:
                totalCount1.append(id)
                # Change line color
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)

    total_vehicle_in = str(len(totalCount))
    total_vehicle_out = str(len(totalCount1))
    total_capacity = int(capacity) - int(total_vehicle_in) + int(total_vehicle_out)

    cv2.putText(img, f"vehicle in: {total_vehicle_in}", (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    cv2.putText(img, f"vehicle out: {total_vehicle_out}",(800,100),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),3)

    print("In:"+total_vehicle_in)
    print("Out"+total_vehicle_out)

    # CSV
    output_file = 'output.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['In', 'Out', 'Capacity'])
        writer.writerow([total_vehicle_in,total_vehicle_out,total_capacity])
    csvfile.close()

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # # Firebase
    result=firebase.put('/count','capacity',int(total_capacity))
    result=firebase.put('/count','in',int(total_vehicle_in))
    result=firebase.put('/count','out',int(total_vehicle_out))
