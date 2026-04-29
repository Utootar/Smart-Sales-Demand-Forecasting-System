import pandas as pd
from sqlalchemy import create_engine

print("--- 🚀 เริ่มต้นการอัปโหลดข้อมูล ---")

# Step 1: อ่านไฟล์
print("1. กำลังอ่านไฟล์ Sre....")
try:
    df = pd.read_csv('Sre.csv', encoding='utf-8', low_memory=False)
    print(f"✅ อ่านไฟล์สำเร็จ! เจอข้อมูลทั้งหมด {len(df)} แถว")
except Exception as e:
    print(f"❌ พังตั้งแต่อ่านไฟล์ CSV! (หาไฟล์ไม่เจอ หรือพิมพ์ชื่อผิด): {e}")
    exit()

# Step 2: ต่อท่อ MySQL
print("2. กำลังต่อท่อเข้า MySQL (lfgoon)...")
try:
    engine = create_engine('mysql+pymysql://root:56472000sql@localhost/lfgoon')
    # เทสการเชื่อมต่อ
    connection = engine.connect()
    connection.close()
    print("✅ ต่อท่อ MySQL สำเร็จ! รหัสผ่านถูกต้อง!")
except Exception as e:
    print(f"❌ ท่อแตก! ต่อ MySQL ไม่ได้ เช็กรหัสผ่านด่วน: {e}")
    exit()

# Step 3: ดันข้อมูล
print("3. กำลังดันข้อมูลครึ่งล้านแถวลงตาราง Sreport (นั่งจิบน้ำรอเลยสัส)...")
try:
    df.to_sql('Sre', con=engine, if_exists='replace', index=False, chunksize=10000)
    print("🎉 โคตรแจ่ม!! ดันข้อมูลเข้า MySQL เสร็จสมบูรณ์ 100% แล้วสัส!!")
except Exception as e:
    print(f"❌ พังตอนดันข้อมูลลง Database!: {e}")