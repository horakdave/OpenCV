import cv2
import numpy as np
import pyvirtualcam
import mediapipe as mp

width, height = 640, 480

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose()
face_detection = mp_face.FaceDetection(min_detection_confidence=0.3)
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

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

        #pose detection
        pose_results = pose.process(frame_rgb)

        #face detection
        face_results = face_detection.process(frame_rgb)

        #hand detection
        hands_results = hands.process(frame_rgb)

        #face mesh
        face_mesh_results = face_mesh.process(frame_rgb)
        landmarks_frame = np.zeros_like(frame)
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(landmarks_frame, (x, y), 1, (255, 0, 0), 1)
        #pose mesh
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                landmarks_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #hands mesh
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    landmarks_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #show only landmarks image
        cam.send(landmarks_frame)

        cv2.imshow('Virtual Camera with Landmarks Detection', landmarks_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
