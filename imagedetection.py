from ultralytics import YOLO
import cv2

model = YOLO("models/model_custom.pt")
results = model("image/sample4.jpg", show=True)
cv2.waitKey(0)