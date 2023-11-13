import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui



VOLUME_RANGE = [0, 100]



current_volume = VOLUME_RANGE[1] // 2



mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands



cap = cv2.VideoCapture(0)


# Main loop

with mp_hands.Hands(

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5) as hands:


    while cap.isOpened():

        success, image = cap.read()

        if not success:

            print("nenacita se frame webky.")

            break


        #  convefrtovani do mediapipu

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # processovani pomoci mediapipe hand modelu

        results = hands.process(image_rgb)


        # landmarky ruky

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)



                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pointer_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]


                distance = math.sqrt(
                    (thumb.x - pointer_finger.x) ** 2 + (thumb.y - pointer_finger.y) ** 2)


                mapped_volume = np.interp(distance, [0, 0.2], VOLUME_RANGE)

                mapped_volume = int(mapped_volume)

                if mapped_volume != current_volume:
                    current_volume = mapped_volume
                    pyautogui.press('volumeup', presses=current_volume)


        cv2.putText(image, f"Volume: {current_volume}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Display

        cv2.imshow('Hand Tracking', image)



        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


cap.release()