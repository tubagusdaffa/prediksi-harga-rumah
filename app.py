import streamlit as st
import numpy as np
import pickle

# Load model dan scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping
city_mapping = {
    "Jakarta": 0, "Bogor": 1, "Depok": 2, "Tangerang": 3,
    "Bekasi": 4, "Serpong": 5, "Cibubur": 6, "Cikarang": 7, "BSD": 8
}
furnishing_labels = {
    0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully-Furnished", 3: "Custom"
}
condition_labels = {
    0: "Butuh renovasi", 1: "Renovasi ringan", 2: "Layak huni",
    3: "Baru direnovasi", 4: "Baru", 5: "Premium", 6: "Lainnya"
}

# UI
st.title("🏡 Prediksi Harga Rumah di Jabodetabek")
st.markdown("Masukkan data rumah, lalu sistem akan memprediksi harganya dan mengklasifikasikan tipenya.")

with st.form("form_prediksi"):
    st.subheader("Formulir Properti Rumah")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Kota", list(city_mapping.keys()))
        bedrooms = st.number_input("Kamar Tidur", 1, 10, 3)
        bathrooms = st.number_input("Kamar Mandi", 1, 10, 2)
        land_size = st.number_input("Luas Tanah (m²)", 10, 1000, 72)
        building_size = st.number_input("Luas Bangunan (m²)", 10, 1000, 60)
        floors = st.selectbox("Jumlah Lantai", [1, 2, 3])

    with col2:
        carports = st.number_input("Carport", 0, 3, 1)
        maid_bedrooms = st.number_input("Kamar Pembantu", 0, 2, 0)
        maid_bathrooms = st.number_input("KM Pembantu", 0, 2, 0)
        building_age = st.number_input("Umur Bangunan", 0, 100, 10)
        garages = st.number_input("Garasi", 0, 3, 1)
        furnishing = st.selectbox("Furnishing", list(furnishing_labels.keys()), format_func=lambda x: furnishing_labels[x])

    condition = st.selectbox("Kondisi Properti", list(condition_labels.keys()), format_func=lambda x: condition_labels[x])

    submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        data = np.array([[
            city_mapping[city], 0, bedrooms, bathrooms, land_size, building_size,
            carports, maid_bedrooms, maid_bathrooms, floors, building_age, 0,
            condition, garages, furnishing
        ]])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]

        # Kategori rumah berdasarkan harga
        if prediction < 500_000_000:
            kategori = "Murah"
        elif prediction <= 2_000_000_000:
            kategori = "Menengah"
        else:
            kategori = "Mewah"

        # Tampilkan hasil
        st.success(f"💰 Estimasi Harga: Rp {int(prediction):,}")
        st.info(f"🏷️ Kategori Rumah: **{kategori}**")

        # Fitur tambahan opsional
        estimasi_pajak = prediction * 0.05
        st.caption(f"💸 Estimasi Pajak Pembelian: Rp {int(estimasi_pajak):,}")
