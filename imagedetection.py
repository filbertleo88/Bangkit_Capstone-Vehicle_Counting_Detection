from ultralytics import YOLO
import cv2

model = YOLO("models/custom1.pt")
results = model("image/cctv2.jpg", show=True)
cv2.waitKey(0)