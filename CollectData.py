import os
import cv2
import numpy as np
import csv
import math

from AdfansedKonkuterWision.DriverDrowsinessDetection.DriverDrowsinessDet2.FaceHandsDetector import Detector


DATA_FILE = "coordsNeut.csv"
CLASS = "Neutral"

dt = Detector()
cap = cv2.VideoCapture(0)

if os.path.exists(DATA_FILE) is False:
    file_created = False
else:
    file_created = True

save_data = False
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    pressedKey = cv2.waitKey(1) & 0xFF
    if success is False:
        break
    img = cv2.resize(img, (1280, 720))
    h, w, _ = img.shape

    img, face_mesh_row, hands_row, face_mesh_list, hands_list_right, hands_list_left = dt.get_data(img)
    # print(hands_list_right)
    num_row = len(face_mesh_list) + len(hands_list_right) + len(hands_list_left)

    if file_created is False:
        columns = ['class']
        for val in range(1, num_row + 1):
            columns += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

        with open(DATA_FILE, mode="w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(columns)

    else:
        if pressedKey == ord('s'):
            save_data = True

        if save_data:
            cv2.putText(img, f"Saving data for class: {CLASS}", (500, 80), cv2.FONT_HERSHEY_PLAIN, 2, (125, 0, 255), 2)
            if len(face_mesh_row) != 0:
                data_row = face_mesh_row + hands_row
                data_row.insert(0, CLASS)

                with open(DATA_FILE, mode="a", newline="") as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(data_row)

    cv2.imshow("res", img)
    if pressedKey == 27:
        break

cap.release()
cv2.destroyAllWindows()
