import streamlit as st
import pandas as pd

st.title("Analisis Pengaruh Pendidikan terhadap Pengangguran")

uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Tambahkan kolom rata-rata pendidikan (simulasi bobot tahunan)
    df['Pendidikan_RataRata'] = (
        df['Tidak/belum pernah sekolah'] * 0 +
        df['Tidak/belum tamat SD'] * 3 +
        df['SD'] * 6 +
        df['SLTP'] * 9 +
        df['SLTA Umum/SMU'] * 12 +
        df['SLTA Kejuruan/SMK'] * 12 +
        df['Akademi/Diploma'] * 14 +
        df['Universitas'] * 16
    ) / df['Total']

    st.subheader("Data dengan Pendidikan Rata-Rata:")
    st.dataframe(df[['Periode', 'Pendidikan_RataRata']].head())

    st.line_chart(df.set_index('Periode')[['Pendidikan_RataRata']])
