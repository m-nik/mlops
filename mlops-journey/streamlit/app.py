import streamlit as st
import joblib
import numpy as np

# بارگذاری مدل
pipeline = joblib.load('linear_pipeline.joblib')

# عنوان
st.title("پیش‌بینی قیمت خانه در California")

# گرفتن ورودی از کاربر
MedInc = st.slider('میانگین درآمد خانوار (x10,000$)', 0.0, 15.0, 3.5)
HouseAge = st.slider('سن خانه', 1, 50, 20)
AveRooms = st.slider('میانگین تعداد اتاق', 1.0, 10.0, 5.0)
AveBedrms = st.slider('میانگین تعداد اتاق خواب', 0.5, 5.0, 1.0)
Population = st.slider('جمعیت منطقه', 100, 35000, 1000)
AveOccup = st.slider('میانگین افراد هر خانه', 0.5, 10.0, 3.0)
Latitude = st.slider('عرض جغرافیایی', 32.0, 42.0, 34.0)
Longitude = st.slider('طول جغرافیایی', -125.0, -114.0, -120.0)

# ساخت آرایه ورودی
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

# پیش‌بینی و نمایش نتیجه
if st.button('پیش‌بینی کن'):
    prediction = pipeline.predict(input_data)
    st.success(f'🏠 قیمت پیش‌بینی‌شده خانه: {prediction[0]:.2f} x 100,000 $')
