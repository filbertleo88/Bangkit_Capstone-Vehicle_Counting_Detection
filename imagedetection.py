from ultralytics import YOLO
import cv2

model = YOLO("models/custom.pt")
results = model("image/cctv.jpg", show=True)
cv2.waitKey(0)