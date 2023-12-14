import cv2
from ultralytics import YOLO
import numpy as np

import torch
# print(torch.backends.mps.is_available())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
model = YOLO("yolov8m.pt")

#load class list
# classes = []
# with open("deepFaceDetection/data/classes.txt","r") as file_object:
#     for class_name in file_object.readlines():
#         class_name = class_name.strip()
#         classes.append(class_name)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not success:
        print("the capturing is not successful")
        break
    
    result = model(frame, device="mps")
    # print (result)
    result = result[0]
    if len(result) > 0:
        print("this is the result[0]", result[0])
    else:
        print("No results found in the 'result' array")
    bboxes = result.boxes.xyxy
    clas = result.boxes.cls
    # print("class:",clas)


    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,225),2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()