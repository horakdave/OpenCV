import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import threading
from pyvirtualcam import Camera

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_detection = mp_face.FaceDetection()
pose = mp_pose.Pose()
hands = mp_hands.Hands()

def capture_screen(camera):
    try:
        while True:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # flip frame horizontally
            frame = cv2.flip(frame, 1)

            faces = face_detection.process(frame)
            if faces.detections:
                for detection in faces.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, bbox, (255, 0, 255), 2)

            pose_landmarks = pose.process(frame)
            if pose_landmarks.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            hand_landmarks = hands.process(frame)
            if hand_landmarks.multi_hand_landmarks:
                for landmarks in hand_landmarks.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # (swap red and blue channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, (640, 480))

            # send frame to virtual camera
            camera.send(frame)

            cv2.imshow("Live Screen Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

with Camera(width=640, height=480, fps=30) as camera:
    screen_capture_thread = threading.Thread(target=capture_screen, args=(camera,))
    screen_capture_thread.start()

    screen_capture_thread.join()

cv2.destroyAllWindows()
