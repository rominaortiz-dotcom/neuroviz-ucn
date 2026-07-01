import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroViz UCN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILOS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .main { background-color: #FAFAFA; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #FAFAFA !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

    /* Títulos */
    .page-title   { font-size:1.8rem; font-weight:800; color:#0D0D0D; margin-bottom:0; letter-spacing:-0.5px; }
    .page-sub     { font-size:0.95rem; color:#8F9FBF; margin-top:0.2rem; margin-bottom:0; }
    .study-title  { font-size:1.3rem; font-weight:700; color:#0511F2; margin:0.5rem 0; border-left:4px solid #0511F2; padding-left:0.6rem; }
    .sec-header   { font-size:1.0rem; font-weight:600; color:#0D0D0D; margin:0.6rem 0 0.2rem 0; }
    .card         { background:#fff; border-radius:10px; padding:1.2rem; border:1px solid #eee;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06); margin-bottom:0.5rem; }
    .tag-low      { background:#BF7330; color:white; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
    .tag-med      { background:#8F9FBF; color:white; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
    .tag-high     { background:#0511F2; color:white; border-radius:4px; padding:2px 8px; font-size:0.8rem; }

    /* Métricas */
    div[data-testid="stMetricValue"]  { font-size:1.5rem !important; font-weight:700 !important; color:#0511F2 !important; }
    div[data-testid="stMetricLabel"]  { font-size:0.78rem !important; color:#8F9FBF !important; }
    div[data-testid="stMetricDelta"]  { font-size:0.82rem !important; }

    /* Tabs */
    button[data-baseweb="tab"] { font-size:0.88rem; font-weight:600; }

    /* Divider */
    hr { border-color: #eee; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── PALETA ───────────────────────────────────────────────────────────────────
COL_CONG    = "#0511F2"
COL_INCONG  = "#BF7330"
COL_MERIT   = "#8C5C20"
COL_NEUTRAL = "#8F9FBF"
COL_DARK    = "#0D0D0D"
COLOR_GRUPO = {"LOW": "#BF7330", "MEDIUM": "#8F9FBF", "HIGH": "#0511F2"}

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
@st.cache_data
def load_tal():
    cb = pd.read_excel("FRE_UNIQ_CB (1).xlsx")
    cm = pd.read_excel("FRE_UNIQ_CM (1).xlsx")
    ca = pd.read_excel("FRE_UNIQ_CA (1).xlsx")
    muestra = pd.read_excel("MUESTRAS_DESCRIP.xlsx")

    cb.columns = ["Categoria","Palabra","Frecuencia","Total"]
    cm.columns = ["Categoria","Palabra","Frecuencia","Total"]
    ca.columns = ["Categoria","Palabra","Frecuencia","Total"]

    ca_top = ca.dropna(subset=["Categoria"])[["Categoria","Total"]].sort_values("Total", ascending=False).head(12).reset_index(drop=True)
    cb_top = cb.dropna(subset=["Categoria"])[["Categoria","Total"]].sort_values("Total", ascending=False).head(12).reset_index(drop=True)
    cm_top = cm.dropna(subset=["Categoria"])[["Categoria","Total"]].sort_values("Total", ascending=False).head(12).reset_index(drop=True)

    muestra.columns = ["Edad","Sexo","Genero","Ciudad","Nacionalidad","Educacion","EstadoCivil","NSE"]
    muestra["Ciudad"] = muestra["Ciudad"].str.strip().str.title()

    return ca_top, cb_top, cm_top, muestra

@st.cache_data
def load_all():
    df = pd.read_excel("D_scores_final.xlsx", skiprows=2, header=0)
    df = df.dropna(subset=["id"])
    df = df[df["id"] != "id"]
    for col in ["media_RT_cong","media_RT_incong","D_score",
                "n400_cong","n400_incong","porp_correctas","prop_errores"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["RT_cong_ms"]   = df["media_RT_cong"]   / 10000
    df["RT_incong_ms"] = df["media_RT_incong"] / 10000
    df["delta_RT"]     = df["RT_incong_ms"] - df["RT_cong_ms"]
    df["delta_N400"]   = df["n400_incong"]  - df["n400_cong"]
    df["grupo_orden"]  = df["grupo_meritocracia"].map({"LOW":1,"MEDIUM":2,"HIGH":3})
    df = df.sort_values("grupo_orden")

    erp = pd.read_excel("Base_datos_lpp_n400_mert.xlsx", sheet_name="1er")
    erp["delta_N400"] = erp["N400_Incong"] - erp["N400_Cong"]
    erp["delta_LPP"]  = erp["LPP_Incong"]  - erp["LPP_Cong"]

    controls = pd.read_excel("CONTROLES PLANILLA JOSEFINA  BAPQ CON ITEMS INVERTIDOS (1).xlsx")
    bapq_cols = [f"BAPQ_{i}" for i in range(1,37)]
    controls["BAPQ_total"] = controls[bapq_cols].mean(axis=1)
    aloof_i = [c for c in [f"BAPQ_{i}" for i in [1,5,9,12,16,18,23,25,29,31,34,36]] if c in controls.columns]
    rigid_i = [c for c in [f"BAPQ_{i}" for i in [2,6,10,13,17,20,24,27,30,33]]       if c in controls.columns]
    prag_i  = [c for c in [f"BAPQ_{i}" for i in [3,7,11,14,15,19,21,22,26,28,32,35]] if c in controls.columns]
    controls["Aloof"]     = controls[aloof_i].mean(axis=1)
    controls["Rigid"]     = controls[rigid_i].mean(axis=1)
    controls["Pragmatic"] = controls[prag_i].mean(axis=1)

    autistas = pd.read_excel("BBDD JOSÉ M.xlsx")
    aq_cols  = [f"AQ_{str(i).zfill(2)}" for i in range(1,47)]
    autistas["AQ_total"] = autistas[aq_cols].mean(axis=1)
    sexo_col = [c for c in autistas.columns if "SEXO" in c.upper()][0]

    return df, erp, controls, autistas, sexo_col

try:
    df, erp, controls, autistas, sexo_col = load_all()
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    logo_path = "logo_lnc.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=160)
    
    st.markdown("### NeuroViz UCN")
    st.markdown("*Dashboard de Ciencia Abierta*")
    st.divider()

    seccion = st.radio("Navegar", [
        "🏠  Inicio",
        "🧪  Estímulos IAT — TAL",
        "📊  Estudio 1 — Meritocracia",
        "🔬  Estudio 2 — Fenotipo Autista",
    ])
    st.divider()
    st.markdown("**Muestra actual**")
    st.markdown(f"IAT + ERP: **{len(df)}** part.")
    st.markdown(f"ERP completo: **{len(erp)}** part.")
    st.markdown(f"Controles BAPQ: **{len(controls)}**")
    st.markdown(f"Autistas AQ: **{int(autistas['AQ_total'].notna().sum())}**")
    st.divider()
    st.markdown("**Paleta de colores**")
    st.markdown("""
<div style="font-size:0.78rem; line-height:1.9;">
  <span style="background:#BF7330;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> LOW meritocracia<br>
  <span style="background:#8F9FBF;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> MEDIUM meritocracia<br>
  <span style="background:#0511F2;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> HIGH meritocracia<br>
  <hr style="border-color:#333;margin:4px 0;">
  <span style="background:#0511F2;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> Congruente (IAT)<br>
  <span style="background:#BF7330;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> Incongruente (IAT)<br>
  <span style="background:#8C5C20;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> Score_Merit<br>
  <span style="background:#8F9FBF;border-radius:3px;padding:1px 7px;color:white;font-size:0.7rem;">■</span> Neutral / BAPQ
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.caption("Laboratorio de Neurociencia Cognitiva\nUCN · Antofagasta · 2026")

# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if "Inicio" in seccion:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if os.path.exists("logo_lnc.png"):
            st.image(Image.open("logo_lnc.png"), width=120)
    with col_title:
        st.markdown('<p class="page-title">NeuroViz UCN</p>', unsafe_allow_html=True)
        st.markdown('<p class="page-sub">Dashboard Interactivo para la Comunicación Científica del Laboratorio de Neurociencia Cognitiva UCN</p>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color:#0511F2; margin-top:0;">📊 Estudio 1 — Meritocracia</h4>
            <p style="font-size:0.9rem; color:#444;">
            Examina cómo las creencias meritocráticas modulan la respuesta cognitiva y neurofisiológica 
            ante información sobre desigualdad social, combinando <b>EEG de 64 canales</b>, 
            <b>pupilometría</b> y un <b>Test de Asociación Implícita (IAT)</b> en un diseño 2×2 mixto.
            </p>
            <p style="font-size:0.85rem; color:#8F9FBF; margin-bottom:0;">
            Romina Ortiz Hidalgo · Doctorado en Psicología UCN-UTA
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color:#8C5C20; margin-top:0;">🔬 Estudio 2 — Fenotipo Autista</h4>
            <p style="font-size:0.9rem; color:#444;">
            Explora perfiles del fenotipo autista ampliado en población adulta mediante el 
            <b>BAPQ</b> (controles, N=245) y el <b>AQ</b> (adultos autistas, N=77), 
            con foco en regulación emocional, flexibilidad cognitiva y camuflaje social.
            </p>
            <p style="font-size:0.85rem; color:#8F9FBF; margin-bottom:0;">
            José Monroy Avendaño · Doctorado en Psicología UCN-UTA
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Estado de la muestra")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Participantes IAT+ERP", len(df), help="Base integrada D_scores_final")
    c2.metric("D-score medio", f"{df['D_score'].mean():.3f}", help="Sesgo pro-meritocrático implícito")
    c3.metric("Controles BAPQ", len(controls))
    c4.metric("Adultos autistas AQ", int(autistas["AQ_total"].notna().sum()))

    st.divider()
    st.markdown("### Nuestro Laboratorio")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""
        <div class="card">
            <h4 style="color:#0D0D0D; margin-top:0;">Laboratorio de Neurociencia Cognitiva — LNC UCN</h4>
            <p style="font-size:0.92rem; color:#444; line-height:1.6;">
            El Laboratorio de Neurociencia Cognitiva (LNC) de la Escuela de Psicología de la UCN 
            se especializa en la investigación del funcionamiento cognitivo y emocional del cerebro humano. 
            A través de técnicas como el electroencefalograma (EEG), eye-tracker y protocolos conductuales 
            innovadores, nos esforzamos por revelar los mecanismos cerebrales subyacentes a la cognición 
            y su relación con el comportamiento humano.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        if os.path.exists("logo_lnc.png"):
            st.image(Image.open("logo_lnc.png"), width=180)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card" style="border-left:4px solid #0511F2; min-height:160px;">
            <h2 style="color:#0511F2; margin-top:0; font-size:2rem;">01</h2>
            <p style="font-size:0.88rem; color:#444; line-height:1.5; margin-bottom:0;">
            <b>Pioneros</b> en el estudio de neurociencia cognitiva en el norte grande del país.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card" style="border-left:4px solid #8C5C20; min-height:160px;">
            <h2 style="color:#8C5C20; margin-top:0; font-size:2rem;">02</h2>
            <p style="font-size:0.88rem; color:#444; line-height:1.5; margin-bottom:0;">
            Buscamos avanzar en <b>metodologías innovadoras</b> para obtener una comprensión profunda 
            del funcionamiento de los mecanismos cerebrales subyacentes a la cognición y su relación 
            con el comportamiento humano.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card" style="border-left:4px solid #8F9FBF; min-height:160px;">
            <h2 style="color:#8F9FBF; margin-top:0; font-size:2rem;">03</h2>
            <p style="font-size:0.88rem; color:#444; line-height:1.5; margin-bottom:0;">
            Desarrollo de técnicas de <b>electroencefalograma (EEG)</b>, <b>eye-tracker</b> 
            y protocolos conductuales innovadores para el estudio del cerebro humano.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Distribución de grupos — Estudio 1")
    col3, col4 = st.columns(2)
    with col3:
        gc = df["grupo_meritocracia"].value_counts().reindex(["LOW","MEDIUM","HIGH"]).reset_index()
        gc.columns = ["Grupo","N"]
        fig = px.bar(gc, x="Grupo", y="N", color="Grupo",
            color_discrete_map=COLOR_GRUPO, text="N",
            labels={"N":"Participantes"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=280, margin=dict(t=20,b=20),
            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribución por terciles de la Escala de Creencias Meritocráticas.")
    with col4:
        fig2 = px.histogram(df, x="D_score", nbins=10,
            color="grupo_meritocracia", color_discrete_map=COLOR_GRUPO,
            category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
            labels={"D_score":"D-score","grupo_meritocracia":"Grupo"})
        fig2.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Sin sesgo")
        fig2.update_layout(height=280, margin=dict(t=20,b=20),
            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"D-score: M={df['D_score'].mean():.3f}, rango {df['D_score'].min():.2f} a {df['D_score'].max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# ESTUDIO 1 — MERITOCRACIA
# ══════════════════════════════════════════════════════════════════════════════
elif "Estímulos IAT" in seccion:
    ca_top, cb_top, cm_top, muestra_tal = load_tal()

    st.markdown('<p class="study-title" style="color:#8C5C20; border-color:#8C5C20;">🧪 Construcción del IAT — Tarea de Asociación Libre (TAL)</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Estudio previo para identificar los estímulos del IAT · N=71 participantes · Antofagasta, 2024</p>', unsafe_allow_html=True)

    st.info("**Metodología:** Se aplicó una TAL para identificar arquetipos de trabajos asociados a Clase Alta, Clase Media y Clase Baja. Solo Clase Alta y Clase Baja se usaron como categorías del IAT. Clase Media fue variable de control.")

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("N participantes TAL", 71)
    c2.metric("Edad media", "29.6 años")
    c3.metric("Ciudad principal", "Antofagasta")
    c4.metric("NSE más frecuente", "3-4 / 6")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "⚖️  Clase Alta vs. Baja (IAT)",
        "🔵  Clase Media (Control)",
        "👥  Muestra TAL"
    ])

    with tab1:
        st.info("**T8, T10 · A2** | Las categorías Clase Alta y Clase Baja fueron seleccionadas para el IAT como categorías objetivo. Los trabajos más frecuentes definen el arquetipo de cada clase social.")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="sec-header">Clase Alta — Top trabajos asociados</p>', unsafe_allow_html=True)
            fig = px.bar(ca_top.sort_values("Total"), x="Total", y="Categoria",
                orientation="h",
                color="Total",
                color_continuous_scale=["#8F9FBF","#0511F2"],
                labels={"Total":"Frecuencia","Categoria":"Trabajo"},
                title="Arquetipo Clase Alta (usado en IAT)")
            fig.update_layout(height=420, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("T10 · A1+A2 | Los 5 más frecuentes (Médico, Empresario, Ingeniero, Abogado, Político) constituyeron los estímulos Upper Class del IAT.")

        with col2:
            st.markdown('<p class="sec-header">Clase Baja — Top trabajos asociados</p>', unsafe_allow_html=True)
            fig2 = px.bar(cb_top.sort_values("Total"), x="Total", y="Categoria",
                orientation="h",
                color="Total",
                color_continuous_scale=["#8F9FBF","#BF7330"],
                labels={"Total":"Frecuencia","Categoria":"Trabajo"},
                title="Arquetipo Clase Baja (usado en IAT)")
            fig2.update_layout(height=420, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("T10 · A1+A2 | Los 5 más frecuentes (Personal de Aseo, Obrero Construcción, Vendedor Ambulante, Vendedor, Recolectores Basura) constituyeron los estímulos Lower Class del IAT.")

        st.divider()
        # ── ESTÍMULOS SELECCIONADOS PARA EL IAT ──────────────────────────────
        st.markdown('<p class="sec-header">Estímulos seleccionados para el IAT</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F0F4FF; border-radius:8px; padding:1rem; border-left:4px solid #0511F2; margin-bottom:0.5rem;">
            <p style="font-size:0.82rem; color:#444; margin:0 0 0.5rem 0;">
            <b>Criterio de selección:</b> Se eligieron los trabajos más frecuentes por clase. Aquellos que aparecieron 
            también en la lista de Clase Media fueron <b>eliminados</b> para garantizar la polaridad conceptual 
            Upper Class / Lower Class y evitar categorías ambiguas en el IAT.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="card" style="border-left:4px solid #0511F2;">
                <h4 style="color:#0511F2; margin-top:0;">⬆️ UPPER CLASS / CLASE ALTA</h4>
                <table style="width:100%; font-size:0.85rem;">
                    <tr style="color:#8F9FBF; font-size:0.75rem;"><td><b>#</b></td><td><b>Estímulo</b></td><td><b>Frec.</b></td><td><b>✓ Único CA</b></td></tr>
                    <tr><td>1</td><td>MÉDICO</td><td>59</td><td>✓</td></tr>
                    <tr style="background:#F5F5F5;"><td>2</td><td>EMPRESARIO</td><td>40</td><td>✓</td></tr>
                    <tr><td>3</td><td><s style="color:#999;">INGENIERO</s></td><td><span style="color:#999;">38</span></td><td><span style="color:#BF7330;">✗ en CM</span></td></tr>
                    <tr style="background:#F5F5F5;"><td>4</td><td>ABOGADO</td><td>37</td><td>✓</td></tr>
                    <tr><td>5</td><td>POLÍTICO</td><td>28</td><td>✓</td></tr>
                    <tr style="background:#F5F5F5;"><td>6</td><td>GERENTE</td><td>22</td><td>✓</td></tr>
                    <tr><td>7</td><td>INVERSIONISTA</td><td>7</td><td>✓</td></tr>
                </table>
                <p style="font-size:0.75rem; color:#8F9FBF; margin:0.5rem 0 0 0;">
                INGENIERO fue eliminado por aparecer también en Clase Media (Frec.=15).
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="card" style="border-left:4px solid #BF7330;">
                <h4 style="color:#BF7330; margin-top:0;">⬇️ LOWER CLASS / CLASE BAJA</h4>
                <table style="width:100%; font-size:0.85rem;">
                    <tr style="color:#8F9FBF; font-size:0.75rem;"><td><b>#</b></td><td><b>Estímulo</b></td><td><b>Frec.</b></td><td><b>✓ Único CB</b></td></tr>
                    <tr><td>1</td><td>AUXILIAR DE ASEO</td><td>50</td><td>✓</td></tr>
                    <tr style="background:#F5F5F5;"><td>2</td><td>OBRERO</td><td>23</td><td>✓</td></tr>
                    <tr><td>3</td><td>VENDEDOR AMBULANTE</td><td>22</td><td>✓</td></tr>
                    <tr style="background:#F5F5F5;"><td>4</td><td><s style="color:#999;">VENDEDOR</s></td><td><span style="color:#999;">19</span></td><td><span style="color:#BF7330;">✗ en CM</span></td></tr>
                    <tr><td>5</td><td>RECOLECTOR DE BASURA</td><td>16</td><td>✓</td></tr>
                    <tr style="background:#F5F5F5;"><td>6</td><td>ASESORA DEL HOGAR</td><td>13</td><td>✓</td></tr>
                    <tr><td>7</td><td>CAJERO</td><td>10</td><td>✓</td></tr>
                </table>
                <p style="font-size:0.75rem; color:#8F9FBF; margin:0.5rem 0 0 0;">
                VENDEDOR fue eliminado por aparecer también en Clase Media (Frec.=23).
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<p class="sec-header">Comparación de frecuencias entre Clase Alta y Clase Baja</p>', unsafe_allow_html=True)

        ca_comp = ca_top.head(8).copy()
        ca_comp["Clase"] = "Alta"
        cb_comp = cb_top.head(8).copy()
        cb_comp["Clase"] = "Baja"
        comp = pd.concat([ca_comp, cb_comp], ignore_index=True)

        fig3 = px.bar(comp, x="Categoria", y="Total", color="Clase",
            color_discrete_map={"Alta":"#0511F2","Baja":"#BF7330"},
            barmode="group",
            labels={"Total":"Frecuencia","Categoria":"Trabajo","Clase":"Clase Social"},
            title="Top 8 trabajos por clase social")
        fig3.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40,b=20), xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("T9 · A1+A2 | Comparación directa de los arquetipos de trabajo por clase social. Azul = Clase Alta (Upper Class en IAT) · Naranja = Clase Baja (Lower Class en IAT).")

    with tab2:
        st.markdown('<p class="sec-header">Clase Media — trabajos asociados (variable de control)</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.bar(cm_top.sort_values("Total"), x="Total", y="Categoria",
                orientation="h",
                color="Total",
                color_continuous_scale=["#8F9FBF","#8C5C20"],
                labels={"Total":"Frecuencia","Categoria":"Trabajo"})
            fig.update_layout(height=400, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("""
            <div class="card" style="margin-top:20px;">
                <h4 style="color:#8C5C20; margin-top:0;">¿Por qué Clase Media como control?</h4>
                <p style="font-size:0.85rem; color:#444; line-height:1.6;">
                La Clase Media presenta trabajos de perfil intermedio (Profesor, Enfermero, Psicólogo) que no activan de forma sistemática asociaciones con mérito o esfuerzo, lo que la hace adecuada como condición de control en el diseño del IAT.
                </p>
                <p style="font-size:0.8rem; color:#8F9FBF;">
                No fue incluida como categoría objetivo del IAT final.
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.caption("T10 · A1 | Clase Media como variable de control: sus trabajos no fueron incluidos en el IAT definitivo.")

    with tab3:
        st.markdown('<p class="sec-header">Caracterización de la muestra TAL</p>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown('<p class="sec-header">Distribución de Edad</p>', unsafe_allow_html=True)
            fig = px.histogram(muestra_tal, x="Edad", nbins=15,
                color_discrete_sequence=[COL_NEUTRAL],
                labels={"Edad":"Edad (años)"})
            fig.add_vline(x=muestra_tal["Edad"].mean(), line_color=COL_CONG,
                annotation_text=f"M={muestra_tal['Edad'].mean():.1f}")
            fig.update_layout(height=260, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<p class="sec-header">Sexo</p>', unsafe_allow_html=True)
            sexo_counts = muestra_tal["Sexo"].value_counts().reset_index()
            sexo_counts.columns = ["Sexo","N"]
            fig2 = px.pie(sexo_counts, values="N", names="Sexo",
                color_discrete_sequence=[COL_CONG, COL_INCONG],
                hole=0.4)
            fig2.update_layout(height=260, margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with c3:
            st.markdown('<p class="sec-header">NSE auto-reportado (1-6)</p>', unsafe_allow_html=True)
            nse_counts = muestra_tal["NSE"].value_counts().sort_index().reset_index()
            nse_counts.columns = ["NSE","N"]
            fig3 = px.bar(nse_counts, x="NSE", y="N",
                color_discrete_sequence=[COL_MERIT],
                labels={"NSE":"Nivel Socioeconómico (1=bajo, 6=alto)","N":"N"})
            fig3.update_layout(height=260, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<p class="sec-header">Nivel Educativo</p>', unsafe_allow_html=True)
            edu_counts = muestra_tal["Educacion"].value_counts().reset_index()
            edu_counts.columns = ["Educacion","N"]
            edu_short = {
                "Educación universitaria o técnica incompleta": "Univ./Téc. incompleta",
                "Educación universitaria completa": "Univ. completa",
                "Estudios de Postgrado (Máster/magister, Doctorado o equivalente)": "Posgrado",
                "Educación media completa": "Media completa",
                "Educación técnica completa": "Técnica completa",
                "Educación media incompleta": "Media incompleta"
            }
            edu_counts["Educacion"] = edu_counts["Educacion"].map(edu_short).fillna(edu_counts["Educacion"])
            fig4 = px.bar(edu_counts.sort_values("N", ascending=True),
                x="N", y="Educacion", orientation="h",
                color_discrete_sequence=[COL_NEUTRAL],
                labels={"N":"N","Educacion":"Nivel Educativo"})
            fig4.update_layout(height=280, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig4, use_container_width=True)

        with col4:
            st.markdown('<p class="sec-header">Ciudad de residencia</p>', unsafe_allow_html=True)
            ciudad_counts = muestra_tal["Ciudad"].value_counts().reset_index()
            ciudad_counts.columns = ["Ciudad","N"]
            fig5 = px.bar(ciudad_counts.head(8).sort_values("N", ascending=True),
                x="N", y="Ciudad", orientation="h",
                color_discrete_sequence=[COL_CONG],
                labels={"N":"N","Ciudad":"Ciudad"})
            fig5.update_layout(height=280, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig5, use_container_width=True)

        st.caption("T10 · A1+A2 | Muestra TAL: N=71, edad M=29.6 años (DE=11.3), 55% hombres, 75% Antofagasta, NSE predominante 3-4/6, mayoría con educación universitaria incompleta o completa.")

elif "Estudio 1" in seccion:
    st.markdown('<p class="study-title">Estudio 1 — Meritocracia, IAT & EEG</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Romina Ortiz Hidalgo · N=16 participantes con datos completos · Muestra piloto</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  D-score por grupo",
        "⏱  Tiempos de respuesta",
        "🧠  Componentes ERP",
        "🔗  Integración"
    ])

    # ── TAB 1: D-SCORE ──────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="sec-header">D-score por grupo de meritocracia</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = px.strip(df, x="grupo_meritocracia", y="D_score",
                color="grupo_meritocracia", color_discrete_map=COLOR_GRUPO,
                category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
                labels={"grupo_meritocracia":"Grupo","D_score":"D-score"})
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Sin sesgo")
            for g, col in COLOR_GRUPO.items():
                m = df[df["grupo_meritocracia"]==g]["D_score"].mean()
                x_pos = ["LOW","MEDIUM","HIGH"].index(g)
                fig.add_shape(type="line", x0=x_pos-0.3, x1=x_pos+0.3,
                    y0=m, y1=m, line=dict(color=col, width=3))
            fig.update_layout(showlegend=False, height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("T9, T12 · A1 | H7: grupo HIGH debería mostrar D-score más positivo (mayor sesgo pro-meritocrático implícito).")

        with col2:
            fig2 = px.box(df, x="grupo_meritocracia", y="D_score",
                color="grupo_meritocracia", color_discrete_map=COLOR_GRUPO,
                category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
                points="all",
                labels={"grupo_meritocracia":"Grupo","D_score":"D-score"})
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(showlegend=False, height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        resumen_grupo = df.groupby("grupo_meritocracia")["D_score"].agg(
            N="count", Media="mean", DE="std", Min="min", Max="max"
        ).reindex(["LOW","MEDIUM","HIGH"]).round(3)
        st.dataframe(resumen_grupo, use_container_width=True)
        st.caption("H7: el grupo HIGH debería mostrar D-score más positivo (mayor sesgo pro-meritocrático implícito).")

    # ── TAB 2: TIEMPOS DE RESPUESTA ──────────────────────────────────────────
    with tab2:
        st.info("**T1, T5** · Acción: COMPARAR · Objetivo: RT congruente vs. incongruente por grupo · H1: RT_Incong > RT_Cong especialmente en grupo HIGH · Barras de error = ±1 SEM · A1")
        st.markdown('<p class="sec-header">Tiempos de respuesta por condición y grupo</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            rt_means = df[["RT_cong_ms","RT_incong_ms"]].agg(["mean","sem"]).T.reset_index()
            rt_means.columns = ["Condicion","Media","SEM"]
            rt_means["Condicion"] = ["Congruente","Incongruente"]
            fig = go.Figure()
            for _, row in rt_means.iterrows():
                color = COL_CONG if row["Condicion"]=="Congruente" else COL_INCONG
                fig.add_trace(go.Bar(x=[row["Condicion"]], y=[row["Media"]],
                    error_y=dict(type="data", array=[row["SEM"]], visible=True),
                    marker_color=color, name=row["Condicion"]))
            fig.update_layout(yaxis_title="RT medio (ms)", showlegend=False,
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"T1, T9 · A1+A2 | RT cong. M={df['RT_cong_ms'].mean():.0f} ms, RT incong. M={df['RT_incong_ms'].mean():.0f} ms. Delta={df['delta_RT'].mean():.0f} ms (±1 SEM). H1: HIGH > efecto de congruencia.")

        with col2:
            rt_long = pd.melt(df, id_vars=["id","grupo_meritocracia"],
                value_vars=["RT_cong_ms","RT_incong_ms"],
                var_name="Condicion", value_name="RT (ms)")
            rt_long["Condicion"] = rt_long["Condicion"].map(
                {"RT_cong_ms":"Congruente","RT_incong_ms":"Incongruente"})
            fig2 = px.box(rt_long, x="grupo_meritocracia", y="RT (ms)",
                color="Condicion",
                color_discrete_map={"Congruente":COL_CONG,"Incongruente":COL_INCONG},
                category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
                boxmode="group", points="all",
                labels={"grupo_meritocracia":"Grupo"})
            fig2.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        delta_rt = df.groupby("grupo_meritocracia")["delta_RT"].agg(
            ["mean","sem"]).reindex(["LOW","MEDIUM","HIGH"]).reset_index()
        fig3 = px.bar(delta_rt, x="grupo_meritocracia", y="mean",
            color="grupo_meritocracia", color_discrete_map=COLOR_GRUPO,
            error_y="sem",
            labels={"grupo_meritocracia":"Grupo","mean":"Delta RT (ms)"},
            title="Efecto de congruencia IAT por grupo (RT Incong. - RT Cong., ± 1 SEM)")
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(showlegend=False, height=320,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40,b=20))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("T1, T5 · A1 | H1: grupo HIGH → mayor Delta RT (interferencia ante estímulos incongruentes). Vistas coordinadas con tab D-score [L09].")

    # ── TAB 3: ERP ───────────────────────────────────────────────────────────
    with tab3:
        st.info("**T1, T3** · Acción: COMPARAR + EXPLORAR · Objetivo: amplitudes N400/LPP por condición y correlación con Score_Merit · H3: N400_Incong > N400_Cong · H4: meritocracia atenúa ΔN400 · A1")
        st.markdown('<p class="sec-header">Componentes N400 y LPP por grupo y condición</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            n400_long = pd.melt(df, id_vars=["id","grupo_meritocracia"],
                value_vars=["n400_cong","n400_incong"],
                var_name="Condicion", value_name="N400 (µV·ms)")
            n400_long["Condicion"] = n400_long["Condicion"].map(
                {"n400_cong":"Congruente","n400_incong":"Incongruente"})
            fig = px.box(n400_long, x="grupo_meritocracia", y="N400 (µV·ms)",
                color="Condicion",
                color_discrete_map={"Congruente":COL_CONG,"Incongruente":COL_INCONG},
                category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
                boxmode="group", points="all",
                labels={"grupo_meritocracia":"Grupo"})
            fig.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("T1 · A1 | H3: N400_Incong > N400_Cong en todos los grupos. H4: meritocracia atenúa el efecto de incongruencia neural. Boxmode=group para comparación directa.")

        with col2:
            delta_n400 = df.groupby("grupo_meritocracia")["delta_N400"].agg(
                ["mean","sem"]).reindex(["LOW","MEDIUM","HIGH"]).reset_index()
            fig2 = px.bar(delta_n400, x="grupo_meritocracia", y="mean",
                color="grupo_meritocracia", color_discrete_map=COLOR_GRUPO,
                error_y="sem",
                labels={"grupo_meritocracia":"Grupo","mean":"Delta N400 (µV·ms)"},
                title="Delta N400 por grupo (N400 Incong. - N400 Cong., ± 1 SEM)")
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(showlegend=False, height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("T1 · A1 | ΔN400 = N400_Incong − N400_Cong por grupo. ±1 SEM. H4: grupo HIGH → mayor ΔN400 (mayor procesamiento neural de la violación semántica ante incongruencia meritocrática).")

        st.divider()
        st.markdown('<p class="sec-header">Score Meritocracia vs componentes ERP (base ERP completa)</p>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(erp, x="Score_Merit", y="delta_N400",
                text="Subject", trendline="ols",
                color_discrete_sequence=[COL_MERIT],
                labels={"Score_Merit":"Score Meritocracia","delta_N400":"Delta N400 (µV·ms)"})
            fig3.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig3.update_traces(textposition="top center", textfont_size=8)
            fig3.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("T3 · A1 | Correlación OLS: Score_Merit × ΔN400. H4: pendiente negativa esperada (mayor meritocracia → menor ΔN400). Tooltip revela ID sujeto.")

        with col4:
            fig4 = px.scatter(erp, x="Score_Merit", y="delta_LPP",
                text="Subject", trendline="ols",
                color_discrete_sequence=[COL_INCONG],
                labels={"Score_Merit":"Score Meritocracia","delta_LPP":"Delta LPP (µV·ms)"})
            fig4.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig4.update_traces(textposition="top center", textfont_size=8)
            fig4.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 4: INTEGRACIÓN ───────────────────────────────────────────────────
    with tab4:
        st.info("**T6, T7** · Acción: EXPLORAR + LOCALIZAR · Objetivo: dependencias cruzadas EEG × IAT × meritocracia · Scatter con OLS · Color = grupo · Tooltip = ID sujeto · A1")
        st.markdown('<p class="sec-header">Relación D-score × Delta N400 × Grupo</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(df, x="D_score", y="delta_N400",
                color="grupo_meritocracia", symbol="grupo_meritocracia",
                color_discrete_map=COLOR_GRUPO,
                text="id", trendline="ols",
                category_orders={"grupo_meritocracia":["LOW","MEDIUM","HIGH"]},
                labels={"D_score":"D-score (IAT)","delta_N400":"Delta N400 (µV·ms)",
                        "grupo_meritocracia":"Grupo"})
            fig.update_traces(textposition="top center", textfont_size=8,
                selector=dict(mode="markers+text"))
            fig.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig.add_vline(x=0, line_dash="dot", line_color="lightgray")
            fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("T7 · A1 | Dependencias cruzadas EEG × IAT. Color = grupo. H7+H4: D-score positivo debería correlacionar con ΔN400 mayor en grupo HIGH. Tooltip: ID sujeto.")

        with col2:
            fig2 = px.scatter(erp, x="Score_Merit", y="delta_N400",
                color="delta_LPP",
                color_continuous_scale=["#8F9FBF","#8C5C20"],
                text="Subject", trendline="ols",
                labels={"Score_Merit":"Score Meritocracia",
                        "delta_N400":"Delta N400","delta_LPP":"Delta LPP"})
            fig2.update_traces(textposition="top center", textfont_size=8,
                selector=dict(mode="markers+text"))
            fig2.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig2.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("T3, T6 · A1 | Score_Merit (continuo) × ΔN400. Color = intensidad ΔLPP. H2: mayor meritocracia → menor ΔLPP (atenuación emocional). N=17 (base ERP).")

        st.divider()
        st.markdown('<p class="sec-header">Tabla integrada</p>', unsafe_allow_html=True)
        st.dataframe(
            df[["id","grupo_meritocracia","D_score","RT_cong_ms",
                "RT_incong_ms","delta_RT","n400_cong","n400_incong","delta_N400"]
            ].round(2).rename(columns={
                "id":"ID","grupo_meritocracia":"Grupo",
                "D_score":"D-score","RT_cong_ms":"RT Cong (ms)",
                "RT_incong_ms":"RT Incong (ms)","delta_RT":"Delta RT",
                "n400_cong":"N400 Cong","n400_incong":"N400 Incong","delta_N400":"Delta N400"
            }),
            use_container_width=True, height=380
        )

# ══════════════════════════════════════════════════════════════════════════════
# ESTUDIO 2 — FENOTIPO AUTISTA
# ══════════════════════════════════════════════════════════════════════════════
elif "Estudio 2" in seccion:
    st.markdown('<p class="study-title" style="color:#8C5C20; border-color:#8C5C20;">Estudio 2 — Fenotipo Autista Ampliado</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">José Monroy Avendaño · Controles N=245 (BAPQ) · Adultos autistas N=77 (AQ)</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📋  Distribuciones",
        "📐  Subescalas BAPQ",
        "👥  Demografía"
    ])

    # ── TAB 1: DISTRIBUCIONES ────────────────────────────────────────────────
    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("N Controles", len(controls))
        c2.metric("BAPQ Total (M)", f"{controls['BAPQ_total'].mean():.2f}")
        c3.metric("N Autistas", int(autistas["AQ_total"].notna().sum()))
        c4.metric("AQ Total (M)", f"{autistas['AQ_total'].mean():.2f}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="sec-header">BAPQ Total — Controles (escala 1-6)</p>', unsafe_allow_html=True)
            fig = px.histogram(controls, x="BAPQ_total", nbins=20,
                color_discrete_sequence=[COL_NEUTRAL],
                labels={"BAPQ_total":"BAPQ Total"})
            fig.add_vline(x=controls["BAPQ_total"].mean(), line_color=COL_CONG,
                annotation_text=f"M={controls['BAPQ_total'].mean():.2f}")
            fig.update_layout(height=320, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"T9, T10 · A1+A2 | BAPQ Total: M={controls['BAPQ_total'].mean():.2f}, DE={controls['BAPQ_total'].std():.2f}, rango {controls['BAPQ_total'].min():.2f}-{controls['BAPQ_total'].max():.2f}. Escala 1-6.")

        with col2:
            st.markdown('<p class="sec-header">AQ Total — Adultos Autistas (escala 1-5)</p>', unsafe_allow_html=True)
            fig2 = px.histogram(autistas.dropna(subset=["AQ_total"]), x="AQ_total", nbins=20,
                color_discrete_sequence=[COL_INCONG],
                labels={"AQ_total":"AQ Total"})
            fig2.add_vline(x=autistas["AQ_total"].mean(), line_color=COL_MERIT,
                annotation_text=f"M={autistas['AQ_total'].mean():.2f}")
            fig2.update_layout(height=320, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"T9, T10 · A1+A2 | AQ Total: M={autistas['AQ_total'].mean():.2f}, DE={autistas['AQ_total'].std():.2f}, rango {autistas['AQ_total'].min():.2f}-{autistas['AQ_total'].max():.2f}. Escala 1-5. ⚠ Escala distinta a BAPQ.")

    # ── TAB 2: SUBESCALAS ────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="sec-header">Perfil de subescalas BAPQ — Controles</p>', unsafe_allow_html=True)
        sub_data = pd.DataFrame({
            "Subescala": ["Pers. Distante"]*len(controls) +
                         ["Pers. Rígida"]*len(controls) +
                         ["Lenguaje Pragmático"]*len(controls),
            "Puntaje": list(controls["Aloof"]) + list(controls["Rigid"]) + list(controls["Pragmatic"])
        })
        fig = px.violin(sub_data, x="Subescala", y="Puntaje", color="Subescala",
            color_discrete_sequence=[COL_CONG, COL_NEUTRAL, COL_MERIT],
            box=True, points=False,
            labels={"Puntaje":"Puntaje medio (escala 1-6)"})
        fig.update_layout(showlegend=False, height=420,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("T9 · A1 | Perfil de subescalas BAPQ: Aloof M=3.40, Rigid M=3.50, Pragmatic M=3.37 (DE~0.8-1.0). Violin + boxplot revela forma distribucional que los promedios ocultan [L11].")

        st.divider()
        resumen_sub = pd.DataFrame({
            "Subescala": ["Personalidad Distante","Personalidad Rígida","Lenguaje Pragmático"],
            "Media": [controls["Aloof"].mean(), controls["Rigid"].mean(), controls["Pragmatic"].mean()],
            "DE":    [controls["Aloof"].std(),  controls["Rigid"].std(),  controls["Pragmatic"].std()],
            "Min":   [controls["Aloof"].min(),  controls["Rigid"].min(),  controls["Pragmatic"].min()],
            "Max":   [controls["Aloof"].max(),  controls["Rigid"].max(),  controls["Pragmatic"].max()],
        }).round(2)
        st.dataframe(resumen_sub, use_container_width=True, hide_index=True)

    # ── TAB 3: DEMOGRAFÍA ────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="sec-header">Comparación demográfica entre grupos</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            edad_df = pd.DataFrame({
                "Edad": list(controls["Edad"].dropna()) + list(autistas["EDAD"].dropna()),
                "Grupo": ["Controles"]*len(controls["Edad"].dropna()) +
                          ["Adultos autistas"]*len(autistas["EDAD"].dropna())
            })
            fig = px.violin(edad_df, x="Grupo", y="Edad", color="Grupo",
                color_discrete_map={"Controles":COL_NEUTRAL,"Adultos autistas":COL_INCONG},
                box=True, points=False,
                title="Distribución de Edad por grupo")
            fig.update_layout(showlegend=False, height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"T10, T11 · A1+A2 | Controles: M={controls['Edad'].mean():.1f} años (DE={controls['Edad'].std():.1f}). Autistas: M={autistas['EDAD'].mean():.1f} años (DE={autistas['EDAD'].std():.1f}). Violin + boxplot para comparación distribucional.")

        with col2:
            sexo_df = pd.DataFrame({
                "Grupo":["Controles","Controles","Adultos autistas","Adultos autistas"],
                "Sexo":["Mujer","Hombre","Mujer","Hombre"],
                "N":[(controls["Sexo"]==1).sum(),(controls["Sexo"]==2).sum(),
                     (autistas[sexo_col]==1).sum(),(autistas[sexo_col]==2).sum()]
            })
            fig2 = px.bar(sexo_df, x="Grupo", y="N", color="Sexo",
                color_discrete_map={"Mujer":COL_CONG,"Hombre":COL_MERIT},
                barmode="group", title="Distribución de Sexo por grupo",
                text="N")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)
