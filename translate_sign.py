#translate hand sign
import cv2
import mediapipe as mp
import numpy as np
import joblib  # โหลดโมเดล

# โหลดโมเดล KNN
model = joblib.load("hand_sign_model.pkl")
label_names = ["1", "2", "3", "4", "i love you"]

# ตั้งค่า Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ดึงค่าตำแหน่งมือ
            landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

            # ตรวจสอบว่ามือข้างไหน
            hand_type = handedness.classification[0].label  # 'Left' หรือ 'Right'
            hand_label = f"{hand_type} Hand"

            # ทำนายผลลัพธ์
            if len(landmark_list) == model.n_features_in_:
                prediction = model.predict([landmark_list])
                predicted_text = f"{label_names[prediction[0]]} ({hand_label})"
                cv2.putText(image, predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Hand Sign Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()