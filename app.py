import streamlit as st
import numpy as np
import pickle

# ======= Load model & scaler =======
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# ======= Mapping Kota (Nama → Encoding) =======
city_mapping = {
    "Jakarta": 0,
    "Bogor": 1,
    "Depok": 2,
    "Tangerang": 3,
    "Bekasi": 4,
    "Serpong": 5,
    "Cibubur": 6,
    "Cikarang": 7,
    "BSD": 8
}

# ======= UI =======
st.title("🏠 Prediksi Harga Rumah di Jabodetabek")
st.markdown("Masukkan data properti rumah, lalu sistem akan memprediksi harganya menggunakan model Machine Learning.")

with st.form("form_prediksi"):
    st.subheader("Form Input Data Properti Rumah")

    col1, col2 = st.columns(2)
    with col1:
        city_name = st.selectbox("Kota", list(city_mapping.keys()))
        city = city_mapping[city_name]

        bedrooms = st.number_input("Jumlah Kamar Tidur", 0, 20, 3)
        bathrooms = st.number_input("Jumlah Kamar Mandi", 0, 20, 2)
        land_size = st.number_input("Luas Tanah (m2)", 0, 10000, 100)
        building_size = st.number_input("Luas Bangunan (m2)", 0, 10000, 90)
        floors = st.number_input("Jumlah Lantai", 1, 5, 2)

    with col2:
        carports = st.number_input("Carport (Tempat parkir mobil semi-terbuka)", 0, 5, 1)
        maid_bedrooms = st.number_input("Kamar Pembantu", 0, 5, 0)
        maid_bathrooms = st.number_input("Kamar Mandi Pembantu", 0, 5, 0)
        building_age = st.number_input("Umur Bangunan (tahun)", 0, 100, 5)
        garages = st.number_input("Garasi (Tertutup penuh)", 0, 10, 1)
        furnishing = st.selectbox("Furnishing (encoded)", [0, 1, 2, 3])

    property_type = 0  # Asumsikan rumah
    property_condition = st.selectbox("Kondisi Properti (encoded)", [0, 1, 2, 3, 4, 5, 6])
    year_built = 0  # default dummy

    submitted = st.form_submit_button("🔍 Prediksi Harga")

    if submitted:
        input_data = np.array([[
            city, property_type, bedrooms, bathrooms, land_size, building_size,
            carports, maid_bedrooms, maid_bathrooms, floors,
            building_age, year_built, property_condition, garages, furnishing
        ]])

        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            st.success(f"💰 Estimasi Harga Rumah: Rp {int(prediction):,}")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
