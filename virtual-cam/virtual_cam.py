import cv2
import numpy as np
import pyvirtualcam
import mediapipe as mp

width, height = 640, 480

#cap = cam(double)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
face_detection = mp_face.FaceDetection(min_detection_confidence=0.3)
hands = mp_hands.Hands()

with pyvirtualcam.Camera(width=width, height=height, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose Detection
        pose_results = pose.process(frame_rgb)

        # Face Detection
        face_results = face_detection.process(frame_rgb)

        # Hand Detection
        hands_results = hands.process(frame_rgb)

        # blank image(for drawing)
        blank_image = np.zeros_like(frame)

        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                blank_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(blank_image, bbox, (0, 255, 0), 2)

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cam.send(blank_image)

        cv2.imshow('Virtual Camera', blank_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
