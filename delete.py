import pandas as pd

# โหลดข้อมูล
file_path = "sign_language_data.csv"
df = pd.read_csv(file_path)

# แสดงตัวอย่างข้อมูลก่อนลบ
print("ข้อมูลก่อนลบ:")
print(df.head())

# ลบข้อมูลที่มี Label เป็น "สวัสดี"
df = df[df["label"] != "Thank you"]

# บันทึกไฟล์ใหม่ (เขียนทับไฟล์เดิม)
df.to_csv(file_path, index=False)

print("ลบข้อมูลสำเร็จ และบันทึกไฟล์ใหม่แล้ว!")
