import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Klaster Kesejahteraan Provinsi Indonesia",
    page_icon="🗺️",
    layout="wide",
)

# Load & prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("project1_merge.csv", sep=";")
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "Percentage of Poor People - March": "persen_miskin_maret",
            "Percentage of Poor People - September": "persen_miskin_sept",
            "Poverty Line - March (Rp)": "gk_maret",
            "Poverty Line - September (Rp)": "gk_sept",
            "Number of Poor People - March (thousand)": "jml_miskin_maret",
            "Number of Poor People - September (thousand)": "jml_miskin_sept",
            "Indeks Pembangunan Manusia": "ipm",
        }
    )
    return df


@st.cache_data
def load_geojson():
    with open("indonesia (1).geojson", "r") as f:
        return json.load(f)


df = load_data()
geojson = load_geojson()

FEATURES = ["persen_miskin_sept", "gk_sept", "ipm"]
FEATURE_LABELS = {
    "persen_miskin_sept": "Persentase Kemiskinan (%)",
    "gk_sept": "Garis Kemiskinan (Rp)",
    "ipm": "IPM",
}

# Mapping nama provinsi (data BPS) -> nama wilayah di geojson
NAME_FIX = {
    "DI Yogyakarta": "Yogyakarta",
    "DKI Jakarta": "Jakarta Raya",
    "Kepulauan Bangka Belitung": "Bangka-Belitung",
}
# Provinsi hasil pemekaran 2022 digabung ke provinsi induk (geojson belum punya batasnya)
MERGE_TO_PARENT = {
    "Papua Barat Daya": "Papua Barat",
    "Papua Selatan": "Papua",
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
n_clusters = st.sidebar.slider("Jumlah Klaster (k)", min_value=2, max_value=6, value=4)
st.sidebar.caption(
    "Klaster dibuat dengan K-Means berdasarkan Persentase Kemiskinan (Sept), "
    "Garis Kemiskinan (Sept), dan IPM tahun 2025 — data BPS."
)

# ---------------------------------------------------------------------------
# Clustering (per-provinsi, 38 wilayah — dipakai untuk tabel & ranking)
# ---------------------------------------------------------------------------
X_scaled = StandardScaler().fit_transform(df[FEATURES])
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df["cluster"])

# Beri label bermakna berdasarkan rata-rata IPM tiap cluster (rendah -> tinggi)
cluster_order = df.groupby("cluster")["ipm"].mean().sort_values().index.tolist()
label_pool = [
    "Kritis",
    "Perlu Prioritas",
    "Berkembang",
    "Sejahtena & Maju",
    "Sangat Sejahtera",
    "Istimewa",
]
label_map = {c: label_pool[i] for i, c in enumerate(cluster_order)}
df["cluster_label"] = df["cluster"].map(label_map)

# ---------------------------------------------------------------------------
# Data untuk peta (38 provinsi -> digabung ke 34 wilayah geojson)
# ---------------------------------------------------------------------------
df["geo_key"] = df["Provinsi"].replace(NAME_FIX).replace(MERGE_TO_PARENT)
df_map = df.groupby("geo_key")[FEATURES].mean().reset_index()

X_map_scaled = StandardScaler().fit_transform(df_map[FEATURES])
km_map = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_map["cluster"] = km_map.fit_predict(X_map_scaled)
cluster_order_map = df_map.groupby("cluster")["ipm"].mean().sort_values().index.tolist()
label_map_map = {c: label_pool[i] for i, c in enumerate(cluster_order_map)}
df_map["cluster_label"] = df_map["cluster"].map(label_map_map)

geo_states = {f["properties"]["state"] for f in geojson["features"]}
unmatched = set(df_map["geo_key"]) - geo_states

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🗺️ Klaster Kesejahteraan Provinsi Indonesia")
st.markdown(
    "Analisis clustering (K-Means) terhadap 38 provinsi berdasarkan **tingkat kemiskinan** "
    "dan **Indeks Pembangunan Manusia (IPM)** — Sumber: BPS, 2025."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Provinsi", len(df))
col2.metric("Jumlah Klaster", n_clusters)
col3.metric("Silhouette Score", f"{sil_score:.3f}")
col4.metric("Rata-rata IPM Nasional", f"{df['ipm'].mean():.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Peta choropleth
# ---------------------------------------------------------------------------
st.subheader("Peta Sebaran Klaster")
st.caption(
    "Provinsi hasil pemekaran Papua (2022) digabung ke provinsi induk untuk keperluan "
    "visualisasi peta, karena batas administratif barunya belum tersedia di data geografis ini."
)

fig_map = px.choropleth(
    df_map,
    geojson=geojson,
    locations="geo_key",
    featureidkey="properties.state",
    color="cluster_label",
    hover_name="geo_key",
    hover_data={c: ":.2f" for c in FEATURES} | {"geo_key": False},
    category_orders={"cluster_label": label_pool},
    color_discrete_sequence=px.colors.sequential.RdYlGn[::-1] if n_clusters <= 4 else px.colors.qualitative.Set2,
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Perbandingan cluster & ranking
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Karakteristik Tiap Klaster")
    summary = (
        df.groupby("cluster_label")[FEATURES]
        .mean()
        .reindex(label_map[c] for c in cluster_order)
        .rename(columns=FEATURE_LABELS)
        .round(2)
    )
    st.dataframe(summary, use_container_width=True)

    fig_bar = px.bar(
        df.groupby("cluster_label")["persen_miskin_sept"].mean().reindex(
            label_map[c] for c in cluster_order
        ).reset_index(),
        x="cluster_label",
        y="persen_miskin_sept",
        labels={"cluster_label": "Klaster", "persen_miskin_sept": "Rata-rata Kemiskinan (%)"},
        color="cluster_label",
        color_discrete_sequence=px.colors.sequential.RdYlGn[::-1],
    )
    fig_bar.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("🔎 Provinsi Prioritas Tertinggi")
    st.caption("Diurutkan dari skor prioritas tertinggi (kemiskinan tinggi + IPM rendah)")
    df["skor_prioritas"] = df["persen_miskin_sept"] - df["ipm"] / 10
    ranking = (
        df[["Provinsi", "persen_miskin_sept", "ipm", "cluster_label", "skor_prioritas"]]
        .sort_values("skor_prioritas", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    st.dataframe(
        ranking.rename(
            columns={
                "persen_miskin_sept": "Kemiskinan (%)",
                "ipm": "IPM",
                "cluster_label": "Klaster",
                "skor_prioritas": "Skor Prioritas",
            }
        ).round(2),
        use_container_width=True,
        height=490,
    )

st.divider()

# ---------------------------------------------------------------------------
# Explorer per provinsi
# ---------------------------------------------------------------------------
st.subheader("🔍 Cek Provinsi Tertentu")
provinsi_pilihan = st.selectbox("Pilih provinsi", sorted(df["Provinsi"].unique()))
row = df[df["Provinsi"] == provinsi_pilihan].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Klaster", row["cluster_label"])
c2.metric("IPM", f"{row['ipm']:.2f}")
c3.metric("Kemiskinan (Sept)", f"{row['persen_miskin_sept']:.2f}%")
c4.metric("Garis Kemiskinan", f"Rp {row['gk_sept']:,.0f}")

st.caption(
    "Dibuat sebagai proyek portofolio Data Science · Sumber data: BPS (2025) · "
    "Metodologi: K-Means clustering dengan standardisasi fitur"
)
