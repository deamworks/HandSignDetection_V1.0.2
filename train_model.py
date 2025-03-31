#train model
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# โหลดข้อมูล
df = pd.read_csv("sign_language_data.csv")
X = df.iloc[:, :-1].values  # ค่าตำแหน่งมือ
y = df.iloc[:, -1].values  # ป้ายกำกับ

# สร้างโมเดล KNN (ง่ายและเร็ว)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

#  บันทึกโมเดล
joblib.dump(model, "hand_sign_model.pkl")
print(" โมเดลถูกบันทึกเป็น hand_sign_model.pkl แล้ว!")