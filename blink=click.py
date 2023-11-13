import cv2
import pyautogui
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

BLINK_RATIO_THRESHOLD = 0.2  # Upravte podle potreby

blink_counter = 0
prev_blink_state = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            eye_landmarks = [
                (x + int(w * 0.3), y + int(h * 0.5)),  # Leve oko
                (x + int(w * 0.7), y + int(h * 0.5))   # Prave oke
            ]

            eye_width = abs(eye_landmarks[1][0] - eye_landmarks[0][0])
            eye_height = abs(eye_landmarks[1][1] - eye_landmarks[0][1])

            blink_ratio = eye_height / eye_width

            if blink_ratio > BLINK_RATIO_THRESHOLD:
                if not prev_blink_state:
                    blink_counter += 1
                    prev_blink_state = True
            else:
                prev_blink_state = False

    if blink_counter > 0:
        pyautogui.click()
        blink_counter = 0

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
