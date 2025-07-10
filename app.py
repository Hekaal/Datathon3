# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard SmartDemand-ID v3.0 (AI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI-FUNGSI UNTUK MEMUAT ASET ---
@st.cache_data
def load_raw_data():
    """Memuat dan memproses data CSV."""
    try:
        # Menggunakan file data final yang memiliki fitur unsupervised
        hist_df = pd.read_csv('master_table_modified_unsupervised.csv')
        forecast_df = pd.read_csv('hasil_prediksi_dengan_unsupervised.csv')
        
        # Pra-pemrosesan data agar konsisten
        hist_df['tanggal'] = pd.to_datetime(hist_df['tanggal'], errors='coerce')
        forecast_df['tanggal'] = pd.to_datetime(forecast_df['tanggal'], errors='coerce')
        
        for col in ['harga_beras', 'harga_beras_bulan_lalu', 'stok_beras_ton']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        
        return hist_df, forecast_df
    except Exception as e:
        st.error(f"Gagal memuat data CSV: {e}")
        return None, None

@st.cache_resource
def load_ml_models():
    """Memuat semua model machine learning (non-NLP)."""
    try:
        kmeans = joblib.load("kmeans_model.joblib")
        isoforest = joblib.load("isolation_forest_model.joblib")
        # Model prediktif tidak lagi dipanggil langsung, jadi kita bisa hapus dari sini
        # scaler juga tidak digunakan secara langsung di app, jadi bisa dihapus
        return kmeans, isoforest
    except Exception as e:
        st.error(f"Gagal memuat model .joblib: {e}")
        return None, None

@st.cache_resource
def load_nlp_model():
    """Memuat model dan tokenizer NLP dari Hugging Face Hub."""
    try:
        # Ganti dengan ID repo Hugging Face Anda yang benar
        model_path = "Haekhal/ringkasan-beras-nlp"  # <-- PASTIKAN INI BENAR
        
        st.info(f"Memuat model AI dari Hugging Face Hub: {model_path}...")
        # Pipeline akan menangani download dan caching secara otomatis
        summarizer = pipeline("summarization", model=model_path)
        st.success("Model AI berhasil dimuat!")
        return summarizer
    except Exception as e:
        st.error(f"Gagal memuat model NLP dari Hugging Face Hub. Pastikan ID Repo benar dan publik. Error: {e}")
        return None

# --- FUNGSI GENERASI RINGKASAN AI ---
def generate_ai_summary(summarizer, data_dict):
    """Menghasilkan ringkasan menggunakan model NLP dari pipeline."""
    if summarizer is None:
        return "Model AI tidak berhasil dimuat."
    if not data_dict or pd.isna(data_dict.get('harga_sekarang')):
        return "Data tidak cukup untuk menghasilkan ringkasan AI."

    # Format input sesuai dengan yang digunakan saat pelatihan
    input_text = (
        f"provinsi: {data_dict.get('provinsi', '')}; "
        f"harga_sekarang: {int(data_dict.get('harga_sekarang', 0))}; "
        f"harga_prediksi: {int(data_dict.get('harga_prediksi', 0))}; "
        f"persen_perubahan: {data_dict.get('persen_perubahan', 0):.1f}; "
        f"cluster: {int(data_dict.get('cluster_kmeans', -1))}; "
        f"anomali: {int(data_dict.get('anomaly_isolation_forest', 0))}"
    )
    
    try:
        result = summarizer(input_text, max_length=128, min_length=20, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Gagal menghasilkan ringkasan: {e}"

# --- MEMUAT SEMUA ASET ---
df_hist, df_forecast = load_raw_data()
kmeans_model, iso_model = load_ml_models()
summarizer_pipeline = load_nlp_model()
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json"

if df_hist is None or kmeans_model is None or summarizer_pipeline is None:
    st.error("Gagal memuat komponen penting aplikasi. Aplikasi berhenti.")
    st.stop()

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Navigasi Dashboard")
    list_provinsi = sorted(df_hist['provinsi'].unique())
    selected_province = st.selectbox('Pilih Provinsi:', options=list_provinsi)
    
    combined_dates = pd.concat([
        df_hist[df_hist['provinsi'] == selected_province]['tanggal'],
        df_forecast[df_forecast['provinsi'] == selected_province]['tanggal']
    ]).dropna().drop_duplicates().sort_values()

    selected_date = st.selectbox(
        'Pilih Tanggal Analisis:',
        options=combined_dates,
        index=len(combined_dates) - 1 if not combined_dates.empty else 0,
        format_func=lambda x: x.strftime('%B %Y')
    )

# --- JUDUL UTAMA ---
st.title("🍚 Dashboard SmartDemand-ID v3.0 (AI)")
st.markdown("### Analisis dan Prediksi Harga Beras dengan Ringkasan AI")

# --- PERSIAPAN DATA TERPUSAT UNTUK ANALISIS ---
analysis_data = {'provinsi': selected_province}
if selected_date:
    row, harga_col = (None, None)
    
    temp_df_hist = df_hist[(df_hist['tanggal'] == selected_date) & (df_hist['provinsi'] == selected_province)]
    if not temp_df_hist.empty:
        row, harga_col = temp_df_hist.iloc[0], 'harga_beras'
    else:
        temp_df_forecast = df_forecast[(df_forecast['tanggal'] == selected_date) & (df_forecast['provinsi'] == selected_province)]
        if not temp_df_forecast.empty:
            row, harga_col = temp_df_forecast.iloc[0], 'harga_prediksi'

    if row is not None:
        analysis_data['harga_sekarang'] = row.get(harga_col)
        analysis_data['cluster_kmeans'] = row.get('cluster_kmeans')
        analysis_data['anomaly_isolation_forest'] = row.get('anomaly_isolation_forest')

    next_month_date = (selected_date + pd.DateOffset(months=1)).replace(day=1)
    next_row, next_harga_col = (None, None)
    
    temp_df_next_forecast = df_forecast[(df_forecast['tanggal'] == next_month_date) & (df_forecast['provinsi'] == selected_province)]
    if not temp_df_next_forecast.empty:
        next_row, next_harga_col = temp_df_next_forecast.iloc[0], 'harga_prediksi'
    else:
        temp_df_next_hist = df_hist[(df_hist['tanggal'] == next_month_date) & (df_hist['provinsi'] == selected_province)]
        if not temp_df_next_hist.empty:
            next_row, next_harga_col = temp_df_next_hist.iloc[0], 'harga_beras'
            
    if next_row is not None:
        analysis_data['harga_prediksi'] = next_row.get(next_harga_col)
        
    harga_s, harga_p = analysis_data.get('harga_sekarang'), analysis_data.get('harga_prediksi')
    analysis_data['persen_perubahan'] = ((harga_p - harga_s) / harga_s) * 100 if pd.notnull(harga_s) and pd.notnull(harga_p) and harga_s > 0 else 0.0

# --- RINGKASAN OTOMATIS DENGAN AI ---
st.subheader("📝 Ringkasan Analisis oleh AI")
with st.spinner('🤖 AI sedang menganalisis dan menulis ringkasan... (pertama kali mungkin butuh beberapa saat)'):
    with st.container(border=True):
        summary_text = generate_ai_summary(summarizer_pipeline, analysis_data)
        st.markdown(summary_text)

# --- VISUALISASI DATA ---
st.markdown("---")
kpi1, kpi2, kpi3 = st.columns(3)
harga_sekarang = analysis_data.get('harga_sekarang')
harga_prediksi = analysis_data.get('harga_prediksi')
kpi1.metric("Harga Saat Ini", f"Rp {harga_sekarang:,.0f}" if pd.notnull(harga_sekarang) else "N/A", help=f"Data untuk {selected_date.strftime('%B %Y') if selected_date else ''}")
kpi2.metric("Harga Prediksi Bulan Depan", f"Rp {harga_prediksi:,.0f}" if pd.notnull(harga_prediksi) else "N/A", f"{harga_prediksi - harga_sekarang:,.0f}" if pd.notnull(harga_prediksi) and pd.notnull(harga_sekarang) else "N/A")
kpi3.metric("Akurasi Model (MAE)", "~ Rp 110", help="Estimasi rata-rata selisih prediksi dari harga aktual.")

col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader(f"Grafik Harga di {selected_province}")
    hist_province = df_hist[df_hist['provinsi'] == selected_province]
    forecast_province = df_forecast[df_forecast['provinsi'] == selected_province]
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=hist_province['tanggal'], y=hist_province['harga_beras'], mode='lines', name='Harga Historis', line=dict(color='royalblue')))
    fig_ts.add_trace(go.Scatter(x=forecast_province['tanggal'], y=forecast_province['harga_prediksi'], mode='lines+markers', name='Harga Prediksi', line=dict(dash='dash', color='red')))
    fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader(f"Analisis Risiko di {selected_province}")
    # Analisis Unsupervised (Momentum, Cluster, Anomaly) bisa ditambahkan di sini jika diinginkan
    st.markdown("Analisis risiko seperti momentum tren, segmentasi klaster, dan deteksi anomali kini telah diintegrasikan langsung ke dalam ringkasan AI di atas untuk memberikan wawasan yang lebih komprehensif.")

# --- PETA RISIKO NASIONAL ---
st.subheader("Peta Risiko Kenaikan Harga Nasional")

map_data = []
# Pastikan tanggal terpilih valid sebelum memproses
if selected_date:
    # Hitung tanggal bulan berikutnya sekali saja di luar loop
    next_month_date = (selected_date + pd.DateOffset(months=1)).replace(day=1)
    
    with st.spinner("Menyiapkan data peta risiko untuk semua provinsi..."):
        # Loop melalui setiap provinsi yang unik untuk mengumpulkan datanya masing-masing
        for prov in df_hist['provinsi'].unique():
            current_price, next_month_price = None, None

            # 1. Ambil harga saat ini untuk provinsi 'prov'
            row_hist = df_hist[(df_hist['tanggal'] == selected_date) & (df_hist['provinsi'] == prov)]
            if not row_hist.empty:
                current_price = row_hist.iloc[0]['harga_beras']
            else:
                row_forecast = df_forecast[(df_forecast['tanggal'] == selected_date) & (df_forecast['provinsi'] == prov)]
                if not row_forecast.empty:
                    current_price = row_forecast.iloc[0]['harga_prediksi']

            # 2. Ambil harga bulan depan untuk provinsi 'prov'
            next_row_forecast = df_forecast[(df_forecast['tanggal'] == next_month_date) & (df_forecast['provinsi'] == prov)]
            if not next_row_forecast.empty:
                next_month_price = next_row_forecast.iloc[0]['harga_prediksi']
            else:
                next_row_hist = df_hist[(df_hist['tanggal'] == next_month_date) & (df_hist['provinsi'] == prov)]
                if not next_row_hist.empty:
                    next_month_price = next_row_hist.iloc[0]['harga_beras']

            # 3. Hitung persen kenaikan jika data lengkap, jika tidak, beri nilai 0
            if pd.notnull(current_price) and pd.notnull(next_month_price) and current_price > 0:
                kenaikan_persen = ((next_month_price - current_price) / current_price) * 100
                map_data.append({'provinsi': prov, 'kenaikan_persen': kenaikan_persen})
            else:
                map_data.append({'provinsi': prov, 'kenaikan_persen': 0})

# Tampilkan peta jika data berhasil dikumpulkan
if map_data:
    df_map = pd.DataFrame(map_data)
    # Menentukan rentang warna agar lebih informatif
    color_midpoint = 0 
    color_range = [-10, 10] # Misalnya, dari -10% hingga 10%

    fig_map = px.choropleth(
        df_map, 
        geojson=geojson_url, 
        locations='provinsi', 
        featureidkey="properties.Propinsi",
        color='kenaikan_persen', 
        color_continuous_scale="RdYlGn_r", # Merah-Kuning-Hijau (dibalik)
        range_color=color_range,
        scope="asia",
        labels={'kenaikan_persen': 'Perubahan Harga (%)'}
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        margin={"r":0, "t":30, "l":0, "b":0},
        coloraxis_colorbar=dict(title="Perubahan (%)"),
        title=f"Prediksi Perubahan Harga untuk {next_month_date.strftime('%B %Y')}"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Tidak dapat membuat peta risiko karena data tanggal tidak valid.")
