# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard SmartDemand-ID",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT DATA & ASET ---
@st.cache_data
def load_raw_data():
    hist_df = None
    forecast_df = None
    try:
        hist_df = pd.read_csv('master_table_fix.csv')
        forecast_df = pd.read_csv('hasil_prediksi_12_bulan.csv')
        return hist_df, forecast_df
    except:
        st.error("Gagal memuat data. Pastikan file tersedia.")
        return None, None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        return model
    except:
        st.error("Model tidak ditemukan.")
        return None

# --- MEMUAT DATA ---
df_hist_raw, df_forecast_raw = load_raw_data()
model = load_model()
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json"

if df_hist_raw is None or df_forecast_raw is None or model is None:
    st.stop()

# --- PREPROCESS ---
df_hist = df_hist_raw.copy()
df_forecast = df_forecast_raw.copy()

df_hist['tanggal'] = pd.to_datetime(df_hist['tanggal'], errors='coerce')
df_forecast['tanggal'] = pd.to_datetime(df_forecast['tanggal'], errors='coerce')

for col in ['harga_beras', 'harga_beras_bulan_lalu', 'stok_beras_ton']:
    if col in df_hist.columns:
        df_hist[col] = df_hist[col].astype(str).str.replace(',', '', regex=False)
        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

# --- SIDEBAR ---
with st.sidebar:
    st.header("Navigasi Dashboard")
    list_provinsi = sorted(df_hist['provinsi'].unique())
    selected_province = st.selectbox('Pilih Provinsi:', options=list_provinsi)

    hist_dates = df_hist[df_hist['provinsi'] == selected_province]['tanggal'].dropna()
    pred_dates = df_forecast[df_forecast['provinsi'] == selected_province]['tanggal'].dropna()
    combined_dates = pd.concat([hist_dates, pred_dates]).drop_duplicates().sort_values()

    selected_historical_date = None
    if not combined_dates.empty:
        selected_historical_date = st.selectbox(
            'Pilih Tanggal Historis:',
            options=combined_dates,
            index=len(combined_dates) - 1,
            format_func=lambda x: x.strftime('%B %Y')
        )
    else:
        st.warning("Tidak ada data tersedia.")

    st.info("Dashboard ini untuk analisis harga beras.")

# --- JUDUL ---
st.title("🍚 Dashboard SmartDemand-ID")
st.markdown("Analisis Prediktif Harga Beras Nasional")

# --- KPI DATA ---
selected_date_data = None
selected_date_data_df = df_hist[(df_hist['provinsi'] == selected_province) & (df_hist['tanggal'] == selected_historical_date)]

if not selected_date_data_df.empty:
    selected_date_data = selected_date_data_df.iloc[0]
else:
    selected_date_data_df = df_forecast[(df_forecast['provinsi'] == selected_province) & (df_forecast['tanggal'] == selected_historical_date)]
    if not selected_date_data_df.empty:
        selected_date_data = selected_date_data_df.iloc[0]

latest_hist_date = df_hist[df_hist['provinsi'] == selected_province]['tanggal'].dropna().sort_values().iloc[-1]
target_prediction_date = selected_historical_date + pd.DateOffset(months=1)
target_prediction_date = target_prediction_date.replace(day=1)

predicted_data_for_target_date = df_forecast[(df_forecast['provinsi'] == selected_province) & (df_forecast['tanggal'] == target_prediction_date)]
if not predicted_data_for_target_date.empty:
    predicted_data_for_target_date = predicted_data_for_target_date.iloc[0]
else:
    predicted_data_for_target_date = None

# --- KPI DISPLAY ---
kpi1, kpi2, kpi3 = st.columns(3)

harga_value = None
if selected_date_data is not None:
    if 'harga_beras' in selected_date_data:
        harga_value = pd.to_numeric(selected_date_data['harga_beras'], errors='coerce')
    elif 'harga_prediksi' in selected_date_data:
        harga_value = pd.to_numeric(selected_date_data['harga_prediksi'], errors='coerce')

if harga_value is not None:
    kpi1.metric(f"Harga ({selected_historical_date.strftime('%B %Y')})", f"Rp {harga_value:,.0f}")
else:
    kpi1.metric("Harga", "N/A")

selected_price_numeric = harga_value if harga_value is not None else None

if predicted_data_for_target_date is not None and 'harga_prediksi' in predicted_data_for_target_date:
    pred_price_numeric = pd.to_numeric(predicted_data_for_target_date['harga_prediksi'], errors='coerce')
    delta_value = f"Rp {pred_price_numeric - selected_price_numeric:,.0f}" if selected_price_numeric is not None else "N/A"
    label_kpi2 = "Harga Bulan Depan"
    if selected_historical_date == latest_hist_date:
        label_kpi2 = "Harga Prediksi Bulan Depan"
    kpi2.metric(f"{label_kpi2} ({target_prediction_date.strftime('%B %Y')})", f"Rp {pred_price_numeric:,.0f}", delta=delta_value)
else:
    kpi2.metric(f"Harga Bulan Depan ({target_prediction_date.strftime('%B %Y')})", "N/A", "N/A")

kpi3.metric("Akurasi Model (MAE)", "~ Rp 161", help="Mean Absolute Error")

# --- CHART ---
st.markdown("---")
hist_province = df_hist[df_hist['provinsi'] == selected_province]
forecast_province = df_forecast[df_forecast['provinsi'] == selected_province]

fig_ts = go.Figure()
if not hist_province.empty:
    fig_ts.add_trace(go.Scatter(x=hist_province['tanggal'], y=hist_province['harga_beras'], mode='lines', name='Harga Historis', line=dict(color='royalblue')))
if not forecast_province.empty:
    fig_ts.add_trace(go.Scatter(x=forecast_province['tanggal'], y=forecast_province['harga_prediksi'], mode='lines+markers', name='Harga Prediksi', line=dict(dash='dash', color='red')))
fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_ts, use_container_width=True)
