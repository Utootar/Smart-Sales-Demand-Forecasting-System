import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. โหลดข้อมูล (สอนตอง: ใส่ low_memory=False กันมันฟ้อง Error เรื่องแรม)
df = pd.read_csv('Amazon Sale Report.csv', low_memory=False)

# 2. จัดการข้อมูลเบื้องต้น
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df['Amount'] = df['Amount'].fillna(0) # แทนค่าว่างด้วย 0
df['Qty'] = df['Qty'].fillna(0)

# 3. รวมข้อมูลเป็นรายวัน (สอนตอง: ใช้รายวันเพื่อให้มี Data เยอะพอจะทำ ML)
daily_df = df.groupby('Date')[['Qty', 'Amount']].sum().sort_index()

# 4. สร้าง "สมอง" ให้โมเดล (Feature Engineering)
daily_df['day_of_week'] = daily_df.index.dayofweek
daily_df['day'] = daily_df.index.day
daily_df['month'] = daily_df.index.month

# สอนตอง: สร้าง Lag คือเอา "ยอดขายเมื่อวาน" (lag_1) มาช่วยทำนาย "ยอดขายวันนี้"
daily_df['lag_1'] = daily_df['Qty'].shift(1)
daily_df['lag_7'] = daily_df['Qty'].shift(7)

# ลบแถวที่ว่างจากการ shift ทิ้ง
daily_df = daily_df.dropna()

# 5. กำหนดตัวแปร X (ปัจจัย) และ Y (สิ่งที่จะทำนาย)
features = ['day_of_week', 'day', 'month', 'lag_1', 'lag_7']
X = daily_df[features]
y = daily_df['Qty'] # เราจะทำนาย "จำนวนชิ้นที่ขายได้"

# 6. แบ่งข้อมูล (Time Series ห้ามใช้ shuffle=True นะสัส เดี๋ยวลำดับพัง)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7. สร้างและฝึกโมเดล Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. พยากรณ์และวัดผล
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'--- สรุปผลของตอง ---')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score (ความแม่นยำ): {r2:.2f}')

# 9. ทำกราฟสวยๆ ไปส่งอาจารย์ (สอนตอง: เอาผลพยากรณ์มาเทียบกับค่าจริง)
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Sales', color='blue', marker='o')
plt.plot(y_pred, label='Predicted Sales', color='red', linestyle='--', marker='x')
plt.title('Amazon Sales Demand Forecasting by TONG')
plt.legend()
plt.savefig('tong_forecast_result.png')
print("\nsuccess ไปดูไฟล์ชื่อ tong_forecast_result.png")