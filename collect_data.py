import cv2  # ใช้ OpenCV สำหรับการจัดการภาพและกล้อง
import mediapipe as mp  # ใช้ Mediapipe สำหรับตรวจจับมือ
import pandas as pd  # ใช้ Pandas สำหรับจัดการข้อมูล CSV
import os  # ใช้สำหรับตรวจสอบการมีอยู่ของไฟล์

# ตั้งค่า Mediapipe เพื่อใช้งาน Hand Landmark Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# กำหนดชื่อไฟล์ CSV สำหรับเก็บข้อมูล
csv_file = "sign_language_data.csv"

# เช็คว่าไฟล์ CSV มีอยู่หรือไม่
if os.path.exists(csv_file):
    try:
        # ถ้ามีไฟล์ CSV ให้โหลดข้อมูลจากไฟล์
        df = pd.read_csv(csv_file)
        data, labels = df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist()
        print(f"พบข้อมูลเก่า {len(labels)} รายการ กำลังเพิ่มข้อมูลใหม่...")
    except pd.errors.EmptyDataError:
        # ถ้าไฟล์ว่างจะเริ่มต้นเก็บข้อมูลใหม่
        data, labels = [], []
else:
    # ถ้าไม่มีไฟล์เริ่มต้นเก็บข้อมูลใหม่
    data, labels = [], []

# ให้ผู้ใช้ป้อนชื่อท่าทางที่จะบันทึก
cap = cv2.VideoCapture(0)  # เปิดกล้อง
label_name = input("ป้อนชื่อท่าทางที่ต้องการบันทึก : ").strip()  # ป้อนชื่อท่าทาง

while True:
    ret, frame = cap.read()  # อ่านภาพจากกล้อง
    if not ret:
        break  # ถ้าไม่สามารถอ่านภาพได้ ให้หยุด

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB
    results = hands.process(image)  # ประมวลผลการตรวจจับมือจากภาพ
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # แปลงภาพกลับเป็น BGR

    if results.multi_hand_landmarks:  # ถ้ามีการตรวจจับมือ
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # วาด landmarks ของมือ

    # แสดงข้อความบนภาพ
    cv2.putText(image, f"Label: {label_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collect Data", image)  # แสดงภาพที่มีการวาด landmarks
    key = cv2.waitKey(1)  # รอรับการกดปุ่มจากผู้ใช้

    if key == ord('s') and results.multi_hand_landmarks:  # ถ้าผู้ใช้กด 's' และมีการตรวจจับมือ
        for hand_landmarks in results.multi_hand_landmarks:
            # ดึงข้อมูล landmarks (ตำแหน่ง x, y) ของมือ
            landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
            data.append(landmark_list)  # เก็บข้อมูล landmarks
            labels.append(label_name)  # เก็บ label
            print(f"บันทึกท่า '{label_name}' สำเร็จ")

    elif key == ord('c'):  # ถ้าผู้ใช้กด 'c' ให้เปลี่ยน label
        label_name = input("เปลี่ยนชื่อท่าทางใหม่: ").strip()  # ป้อนชื่อท่าทางใหม่
        print(f"กำลังบันทึกข้อมูลสำหรับ '{label_name}'")

    elif key == ord('q'):  # ถ้าผู้ใช้กด 'q' ให้หยุดโปรแกรม
        break

cap.release()  # ปิดการใช้งานกล้อง
cv2.destroyAllWindows()  # ปิดหน้าต่างการแสดงภาพจาก OpenCV

# สร้าง DataFrame จากข้อมูลที่เก็บมา
df_new = pd.DataFrame(data)
df_new["label"] = labels  # เพิ่มคอลัมน์ label ลงใน DataFrame

# บันทึกข้อมูลทั้งหมดลงในไฟล์ CSV
df_new.to_csv(csv_file, index=False)  # บันทึก DataFrame ลงไฟล์ CSV
print(f"ข้อมูลทั้งหมดถูกบันทึกลง {csv_file} แล้ว!")  # แจ้งผู้ใช้ว่าเสร็จสิ้นการบันทึกข้อมูล
