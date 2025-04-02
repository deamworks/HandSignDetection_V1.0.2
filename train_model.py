import pandas as pd  # ใช้ Pandas สำหรับจัดการข้อมูลในรูปแบบ DataFrame
import joblib  # ใช้สำหรับบันทึกโมเดลและตัวเข้ารหัสลงไฟล์
from sklearn.model_selection import train_test_split  # ใช้สำหรับแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
from sklearn.neighbors import KNeighborsClassifier  # ใช้สำหรับสร้างโมเดล K-Nearest Neighbors (KNN)

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("sign_language_data.csv")  # อ่านข้อมูลจากไฟล์ CSV
X = df.iloc[:, :-1].values  # ข้อมูลคุณลักษณะ (ไม่รวมคอลัมน์สุดท้าย)
y = df.iloc[:, -1].values  # คอลัมน์ label (คอลัมน์สุดท้าย)

# แปลง label เป็นตัวเลข (จากข้อความเป็นตัวเลข)
from sklearn.preprocessing import LabelEncoder  # ใช้ LabelEncoder ในการแปลง label
encoder = LabelEncoder()  # สร้างตัวเข้ารหัส label
y = encoder.fit_transform(y)  # แปลง label ให้เป็นตัวเลข

# แบ่งข้อมูลเป็นชุดฝึก (training) และชุดทดสอบ (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # ใช้ 80% สำหรับฝึก และ 20% สำหรับทดสอบ

# สร้างโมเดล KNN
model = KNeighborsClassifier(n_neighbors=3)  # สร้างโมเดล KNN โดยเลือกจำนวนเพื่อนบ้าน (k) เป็น 3
model.fit(X_train, y_train)  # ฝึกโมเดลด้วยข้อมูลชุดฝึก

# บันทึกโมเดลและตัวเข้ารหัส label
joblib.dump(model, "hand_sign_model.pkl")  # บันทึกโมเดล KNN ลงในไฟล์ hand_sign_model.pkl
joblib.dump(encoder, "label_encoder.pkl")  # บันทึก LabelEncoder ลงในไฟล์ label_encoder.pkl

print("โมเดลถูกบันทึกเป็น hand_sign_model.pkl แล้ว!")  # แจ้งเตือนเมื่อบันทึกโมเดลเสร็จ
print("Label Encoder ถูกบันทึกเป็น label_encoder.pkl แล้ว!")  # แจ้งเตือนเมื่อบันทึกตัวเข้ารหัสเสร็จ

