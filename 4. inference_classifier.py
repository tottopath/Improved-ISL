import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N',
               14: '0', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z'}

current_alphabet = None
start_time = None

while True:

    data_aux_left = []
    data_aux_right = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                if i < 21:
                    data_aux_left.append(hand_landmarks.landmark[i].x)
                    data_aux_left.append(hand_landmarks.landmark[i].y)
                else:
                    data_aux_right.append(hand_landmarks.landmark[i].x)
                    data_aux_right.append(hand_landmarks.landmark[i].y)

    data_aux = data_aux_left + data_aux_right

    if len(data_aux) > 0:
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        if predicted_character == current_alphabet:
            if start_time is None:
                start_time = time.time()
            else:
                if time.time() - start_time >= 3:
                    with open('speak.txt', 'a') as f:
                        f.write(predicted_character)
                    print(f"Added '{predicted_character}' to speak.txt")
                    start_time = None
        else:
            current_alphabet = predicted_character
            start_time = time.time()

        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        with open('speak.txt', 'a') as f:
            f.write(' ')
        print("Added space to speak.txt")
    elif key == ord('q'):
        break

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

from englisttohindi.englisttohindi import EngtoHindi

with open('speak.txt', 'r') as file:
    contents = file.read()
    print(contents)
    
trans = EngtoHindi(contents)
new = trans.convert
print(new)

with open('Speak.txt', 'a', encoding="utf-8") as file:
    file.write('\n')
    print(new)
    file.write(new)