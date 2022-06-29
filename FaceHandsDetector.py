from dataclasses import dataclass
import os
import numpy as np
import mediapipe as mp
import cv2
import csv
import pickle
import pandas as pd
import math


@dataclass
class Detector:
    min_confidence_face: float = 0.5
    min_confidence_hands: float = 0.5

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=min_confidence_face)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=min_confidence_hands, max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils

    def get_data(self, img: np.array, draw=True) -> list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False  # lepsza wydajnosc

        result_fm_class = self.face_mesh.process(img)
        result_fm = result_fm_class.multi_face_landmarks
        result_hand_class = self.hands.process(img)
        result_hand = result_hand_class.multi_hand_landmarks

        face_mesh_list = []
        hands_list_left = [[0.0, 0.0, 0.0, 0.0] for _ in range(21)]
        hands_list_right = [[0.0, 0.0, 0.0, 0.0] for _ in range(21)]
        processed_fm_list = []
        processed_h_list = []
        if result_fm:
            for landmarks in result_fm:
                if draw:
                    self.mp_drawing.draw_landmarks(img, landmarks, self.mp_face_mesh.FACE_CONNECTIONS)

                for landmark in landmarks.landmark:
                    face_mesh_list.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

            if result_hand:
                for hand_type, hand_lms in zip(result_hand_class.multi_handedness, result_hand):
                    if draw:
                        self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    # print(hand_type)
                    for index, lm in enumerate(hand_lms.landmark):
                        if hand_type.classification[0].label == "Right":
                            hands_list_right[index] = [lm.x, lm.y, lm.z, lm.visibility]

                        if hand_type.classification[0].label == "Left":
                            hands_list_left[index] = [lm.x, lm.y, lm.z, lm.visibility]

            processed_fm_list = list(np.array([face_mesh_list]).flatten())
            hands_list = hands_list_right + hands_list_left
            processed_h_list = list(np.array([hands_list]).flatten())

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return [img, processed_fm_list, processed_h_list, face_mesh_list, hands_list_right, hands_list_left]


if __name__ == '__main__':
    xd = Detector()
    cap = cv2.VideoCapture(0)

    # with open("chuj16.pkl", "rb") as f:
    #     model = pickle.load(f)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1280, 720))
        h, w, _ = img.shape
        if success is False:
            break

        img, face_mesh_row, hands_row, face_mesh_list, hands_list_right, hands_list_left = xd.get_data(img)
        # print(len(face_mesh_row))
        # if len(face_mesh_row) != 0:
        #     data_row = face_mesh_row + hands_row
        #     x = pd.DataFrame([data_row])
        #     body_lang_class = model.predict(x)[0]
        #     body_lang_prob = model.predict_proba(x)[0]
        #     # print(body_lang_class, body_lang_prob)
        #     prob = body_lang_prob[np.argmax(body_lang_prob)]
        #     if prob > 0.4:
        #         print(prob, body_lang_class)
        #         cv2.putText(img, f"{body_lang_class} {prob}", (500, 80), cv2.FONT_HERSHEY_PLAIN, 2, (125, 0, 255), 2)

        cv2.imshow("res", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



