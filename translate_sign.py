import cv2  # ใช้ OpenCV สำหรับการจับภาพจากกล้อง
import mediapipe as mp  # ใช้ Mediapipe สำหรับตรวจจับมือ
import joblib  # ใช้สำหรับโหลดโมเดลที่ถูกบันทึกไว้
import numpy as np  # ใช้ NumPy สำหรับการจัดการข้อมูลทางคณิตศาสตร์

# โหลดโมเดล KNN และ Label Encoder ที่ถูกบันทึกไว้
model = joblib.load("hand_sign_model.pkl")  # โหลดโมเดล KNN
encoder = joblib.load("label_encoder.pkl")  # โหลดตัวเข้ารหัส label

# ตั้งค่า Mediapipe เพื่อใช้ในการตรวจจับมือ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)  # เปิดกล้องสำหรับการจับภาพจากกล้อง

while cap.isOpened():  # ตรวจสอบว่าเปิดกล้องสำเร็จ
    ret, frame = cap.read()  # อ่านข้อมูลจากกล้อง
    if not ret:  # ถ้าไม่สามารถอ่านภาพได้ ให้หยุด
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB
    results = hands.process(image)  # ตรวจจับมือในภาพ
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # แปลงภาพกลับเป็น BGR

    if results.multi_hand_landmarks:  # ถ้ามีการตรวจจับมือ
        for hand_landmarks in results.multi_hand_landmarks:  # ตรวจจับทุกมือที่พบ
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  
            # วาด landmarks ของมือ

            # ดึงข้อมูลตำแหน่ง landmarks ของมือ (x, y) ของทุกจุด
            landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

            # ตรวจสอบว่าเรามีจำนวน features ที่ตรงกับที่โมเดลต้องการ
            if len(landmark_list) == model.n_features_in_:
                prediction = model.predict([landmark_list])  # ใช้โมเดลทำนายผลลัพธ์
                confidence = max(model.predict_proba([landmark_list])[0])  # ค่าความมั่นใจของโมเดล
                predicted_text = encoder.inverse_transform(prediction)[0]  
                # แปลงผลลัพธ์จากตัวเลขเป็นข้อความ

                # สร้างพื้นหลังสีดำสำหรับข้อความ
                cv2.rectangle(frame, (40, 20), (400, 120), (0, 0, 0), -1)  
                
                # แสดงผลลัพธ์การทำนายและค่าความมั่นใจบนหน้าจอ
                cv2.putText(frame, f"Prediction: {predicted_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Sign Detection", frame)  # แสดงภาพที่มีการทำนายผลบนหน้าจอ
    if cv2.waitKey(1) & 0xFF == ord('q'):  # ถ้าผู้ใช้กด 'q' ให้หยุดโปรแกรม
        break

cap.release()  # ปิดกล้อง
cv2.destroyAllWindows()  # ปิดหน้าต่างที่แสดงภาพจาก OpenCV
