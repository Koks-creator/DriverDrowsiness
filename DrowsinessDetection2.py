import time
import os
import math
import pytz
import cv2
import pickle
import numpy as np
import pandas as pd
import pyglet


from AdfansedKonkuterWision.DriverDrowsinessDetection.DriverDrowsinessDet2.FaceHandsDetector import Detector


def play_sound(file="elo2.wav"):
    current_path = os.getcwd() + "\\"
    sound = pyglet.media.load(current_path + file)
    sound.play()


def get_bbox(face_m_list: list) -> list:
    region = np.array(face_m_list)

    x1 = int(np.min(region[:, 0]) * w)
    x2 = int(np.max(region[:, 0]) * w)
    y1 = int(np.min(region[:, 1]) * h)
    y2 = int(np.max(region[:, 1]) * h)

    bboxw = x2 - x1
    bboxh = y2 - y1

    return [int(x1), int(y1), bboxw, bboxh]


def draw_bbox(img: np.array) -> list:
    x1, y1, bboxw, bboxh = get_bbox(face_mesh_list)

    cv2.rectangle(img, (x1, y1), (x1 + bboxw, y1 + bboxh), (0, 200, 0), 2)
    cv2.line(img, (x1, y1), (x1 + 100, y1), (0, 0, 200), 15)
    cv2.line(img, (x1, y1), (x1, y1 + 100), (0, 0, 200), 15)

    cv2.line(img, (x1, (y1 + bboxh)), (x1 + 100, (y1 + bboxh)), (0, 0, 200), 15)
    cv2.line(img, (x1, (y1 + bboxh)), (x1, (y1 + bboxh) - 100), (0, 0, 200), 15)

    cv2.line(img, (x1 + bboxw, (y1 + bboxh)), (x1 + bboxw, (y1 + bboxh) - 100), (0, 0, 200), 15)
    cv2.line(img, ((x1 + bboxw), (y1 + bboxh)), ((x1 + bboxw) - 100, (y1 + bboxh)), (0, 0, 200), 15)

    cv2.line(img, ((x1 + bboxw), y1), ((x1 + bboxw) - 100, y1), (0, 0, 200), 15)
    cv2.line(img, ((x1 + bboxw), y1), ((x1 + bboxw), y1 + 100), (0, 0, 200), 15)

    return [x1, y1, bboxw, bboxh]


def closed_eyes(img: np.array) -> str:
    color = (0, 255, 255)
    eyes_dict = {
        "LeftEye": [159, 145, 33, 133],
        # "RightEye": [374, 386, 362, 359]
    }

    left_points = []
    values = list(eyes_dict.values())

    for index in values[0]:
        l_index = index

        left_point = np.multiply(face_mesh_list[l_index][:2], [w, h])
        left_point = tuple(map(int, left_point))
        left_points.append(left_point)

        cv2.circle(img, left_point, 5, color, -1)

    cv2.line(img, left_points[0], left_points[1], color, 3)
    cv2.line(img, left_points[2], left_points[3], color, 3)

    v_t, v_l, h_l, h_r = left_points

    v_len = math.hypot(v_t[0] - v_l[0], v_t[1] - v_l[1])
    h_len = math.hypot(h_l[0] - h_r[0], h_l[1] - h_r[1])

    ratio = v_len / h_len

    if ratio < 0.19:
        return "Closed"
    else:
        return "Opened"


tz = pytz.timezone("Poland")
interval = 60
thr_yawns = 1
end_time = time.time() + interval

cap = cv2.VideoCapture(0)
det = Detector()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

closed_eyes_frames = 0
yawns = []
yawning_frames = 0
lowered_head_frames = 0
status = "Unknown"
p_time = 0
font_color = (0, 0, 200)
border_color = (0, 200, 0)
while True:
    success, img = cap.read()
    if success is False:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    h, w, _ = img.shape

    img, face_mesh_row, hands_row, face_mesh_list, hands_list_right, hands_list_left = det.get_data(img, draw=False)

    if len(face_mesh_row) != 0:
        x1, y1, bboxw, bboxh = draw_bbox(img)

        nose_point = np.multiply(face_mesh_list[4][:2], [w, h])
        nose_point = tuple(map(int, nose_point))

        cv2.circle(img, nose_point, 5, border_color, -1)
        cv2.line(img, (0, h - int(h // 2.75)), (w, h - int(h // 2.75)), border_color, 3)

        if nose_point[1] > h - int(h // 2.75):
            border_color = (0, 0, 200)
            lowered_head_frames += 1
            if lowered_head_frames % 10 == 0:
                play_sound()
        else:
            border_color = (0, 200, 0)
            lowered_head_frames = 0

        data_row = face_mesh_row + hands_row
        x = pd.DataFrame([data_row])
        body_lang_class = model.predict(x)[0]
        body_lang_prob = model.predict_proba(x)[0]
        prob = body_lang_prob[np.argmax(body_lang_prob)]

        if prob > 0.4:
            if body_lang_class == "Yawning":
                yawning_frames += 1
                if yawning_frames == 20:
                    yawns.append(1)
            else:
                yawning_frames = 0

            eyes_status = closed_eyes(img)
            if eyes_status == "Closed" and body_lang_class != "Yawning":
                closed_eyes_frames += 1
                if closed_eyes_frames == 4:
                    play_sound()
                    closed_eyes_frames = 0
            else:
                closed_eyes_frames = 0

            label_x, label_y = x1, y1
            cv2.rectangle(img, (label_x - 5, label_y - 75), (label_x + 220, label_y - 45), (255, 255, 255), -1)
            cv2.putText(img, f"{body_lang_class}: {int(prob * 100)}%", (label_x + 3, label_y - 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, font_color, 2)
            cv2.rectangle(img, (label_x - 5, label_y - 40), (label_x + 220, label_y - 10), (255, 255, 255), -1)
            cv2.putText(img, f"Eyes: {eyes_status}", (label_x + 3, label_y - 15), cv2.FONT_HERSHEY_PLAIN, 1.8,
                        font_color, 2)
            cv2.rectangle(img, (label_x - 5, label_y - 75), (label_x + 220, label_y - 10), (0, 0, 200), 2)
            cv2.line(img, (label_x - 5, label_y - 43), (label_x + 220, label_y - 43), (0, 0, 200), 4)

        cur_time = time.time()

        if int(end_time - cur_time) == 0:
            if len(yawns) >= thr_yawns:
                status = "Sleepy"
            else:
                status = "Normal"

            end_time = cur_time + interval
            yawns.clear()

        cv2.putText(img, f"Status: {status}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, font_color, 2)
        mins, secs = divmod(int(end_time - cur_time), 60)
        cv2.putText(img, f"Yawns: {len(yawns)}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2.5, font_color, 2)
        cv2.putText(img, f"Timer: {mins}:{secs}", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2.5, font_color, 2)

    c_time = time.time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time
    cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, font_color, 2)

    cv2.imshow("Res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
