import html
import os
import re
import sys
import time
import unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hugging_face.huggingface import generar_informe
from hugging_face.prompts import construir_prompt

# Permite importar el paquete src cuando la app se ejecuta desde Streamlit.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src.model import predict_admission_probability, recommend_alternative_careers


# ---------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictor de Admisión DEMRE | Ciencia de Datos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Estilos
# ---------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 50%, #0D47A1 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(21, 101, 192, 0.25);
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .prob-card {
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.5rem;
    }

    .prob-high { background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%); }
    .prob-medium { background: linear-gradient(135deg, #F57F17 0%, #E65100 100%); }
    .prob-low { background: linear-gradient(135deg, #C62828 0%, #880E4F 100%); }

    .prob-val {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .prob-title {
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .profile-card {
        background: linear-gradient(135deg, #F8FBFF 0%, #EEF5FF 100%);
        border: 1px solid #D9E7FA;
        border-left: 6px solid #1E88E5;
        border-radius: 16px;
        padding: 1.4rem;
        margin: 0.8rem 0 1.2rem 0;
        box-shadow: 0 6px 16px rgba(30, 136, 229, 0.08);
    }

    .profile-tag {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: white;
        border: 1px solid #D9E7FA;
        color: #263238;
        font-size: 0.84rem;
    }

    .recommendation-card {
        background: #F8F9FA;
        padding: 1rem;
        border-left: 5px solid #2E7D32;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }

    .context-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
        min-height: 140px;
        box-shadow: 0 5px 14px rgba(15, 23, 42, 0.06);
    }

    /* Tarjetas dinámicas recuperadas desde app.py */
    .affinity-card {
        border-radius: 18px;
        padding: 1.35rem 1.25rem;
        min-height: 160px;
        box-shadow: 0 12px 28px -16px rgba(0, 0, 0, 0.65);
        margin-bottom: 0.7rem;
    }

    .affinity-card-amber {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.16) 0%, rgba(15, 23, 42, 0.88) 100%);
        border: 1px solid rgba(245, 158, 11, 0.55);
    }

    .affinity-card-indigo {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.18) 0%, rgba(15, 23, 42, 0.88) 100%);
        border: 1px solid rgba(99, 102, 241, 0.55);
    }

    .affinity-card-emerald {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.17) 0%, rgba(15, 23, 42, 0.88) 100%);
        border: 1px solid rgba(16, 185, 129, 0.55);
    }

    .affinity-card-purple {
        background: linear-gradient(145deg, rgba(168, 85, 247, 0.18) 0%, rgba(15, 23, 42, 0.88) 100%);
        border: 1px solid rgba(168, 85, 247, 0.55);
    }

    .affinity-pill {
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.65rem;
        letter-spacing: 0.02em;
    }

    .affinity-pill-amber { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.55); }
    .affinity-pill-indigo { background: rgba(99, 102, 241, 0.2); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.55); }
    .affinity-pill-emerald { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.55); }
    .affinity-pill-purple { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.55); }

    .affinity-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    .affinity-value {
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        margin-bottom: 0.45rem;
    }

    .affinity-label {
        color: #CBD5E1;
        font-size: 0.77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        overflow-wrap: anywhere;
    }

    .affinity-description {
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.35;
        margin: 0.55rem 0 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
        font-weight: 700;
    }

    /* Bento Grid Layout para Informe IA */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1.2rem;
        margin-top: 1.2rem;
        margin-bottom: 1.8rem;
    }

    .bento-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 1.5rem;
        color: #F8FAFC;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
    }

    .bento-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 35px -8px rgba(0, 0, 0, 0.5);
    }

    .bento-card-blue {
        border-left: 5px solid #3B82F6;
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-emerald {
        border-left: 5px solid #10B981;
        background: linear-gradient(145deg, rgba(6, 78, 59, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-cyan {
        border-left: 5px solid #06B6D4;
        background: linear-gradient(145deg, rgba(22, 78, 99, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-purple {
        border-left: 5px solid #8B5CF6;
        background: linear-gradient(145deg, rgba(88, 28, 135, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-amber {
        border-left: 5px solid #F59E0B;
        background: linear-gradient(145deg, rgba(120, 53, 15, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-green {
        border-left: 5px solid #22C55E;
        background: linear-gradient(145deg, rgba(20, 83, 45, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    }

    .bento-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.6rem;
        letter-spacing: -0.01em;
    }

    .bento-card-body {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #E2E8F0;
        flex-grow: 1;
    }

    .bento-col-12 { grid-column: span 12; }
    .bento-col-7 { grid-column: span 7; }
    .bento-col-6 { grid-column: span 6; }
    .bento-col-5 { grid-column: span 5; }

    @media (max-width: 900px) {
        .bento-col-7, .bento-col-6, .bento-col-5 {
            grid-column: span 12;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------
@st.cache_resource
def load_app_artifacts():
    # Localiza los modelos persistidos y los entrena solo si faltan.
    models_dir = os.path.join(BASE_DIR, "models")

    stats_path = os.path.join(models_dir, "career_stats.joblib")
    clf_path = os.path.join(models_dir, "global_classifier.joblib")
    eda_path = os.path.join(models_dir, "demre_sample_eda.joblib")

    if not (os.path.exists(stats_path) and os.path.exists(clf_path)):
        # Busca el dataset fuente en las ubicaciones compatibles con el proyecto.
        from src.model import train_and_save_model

        candidate_data_paths = [
            os.path.join(BASE_DIR, "DEMRE_1.csv"),
            os.path.join(BASE_DIR, "..", "DEMRE_1.csv"),
        ]
        data_path = next((path for path in candidate_data_paths if os.path.exists(path)), None)
        if data_path is None:
            raise FileNotFoundError(
                "No se encontró DEMRE_1.csv ni los modelos entrenados dentro de la carpeta models/."
            )
        train_and_save_model(data_path, models_dir)

    # Carga estadísticas, clasificador y muestra EDA para reutilizarlos en la sesión.
    stats_df = joblib.load(stats_path)
    model_artifact = joblib.load(clf_path)
    eda_sample = joblib.load(eda_path) if os.path.exists(eda_path) else None

    return stats_df, model_artifact, eda_sample


# ---------------------------------------------------------
# Utilidades para integrar las variables contextuales
# ---------------------------------------------------------
def normalize_column_name(value):
    """Normaliza nombres para localizar columnas equivalentes en distintos datasets."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return "".join(char.lower() for char in value if char.isalnum())


def normalize_text_value(value):
    """Normaliza valores categóricos para comparar nombres con tildes o signos distintos."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = " ".join(value.casefold().strip().split())
    return "".join(char for char in value if char.isalnum() or char.isspace())


def find_column(dataframe, aliases):
    if dataframe is None:
        return None

    # Normaliza nombres para tolerar variantes de columnas del dataset.
    normalized_columns = {normalize_column_name(col): col for col in dataframe.columns}

    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    # Usa coincidencia parcial como respaldo para nombres largos del DEMRE.
    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if len(normalized_alias) < 5:
            continue
        for normalized_col, original_col in normalized_columns.items():
            if normalized_alias in normalized_col or normalized_col in normalized_alias:
                return original_col

    return None


def clean_options(dataframe, column, fallback):
    if dataframe is None or column is None or column not in dataframe.columns:
        return fallback

    # Elimina valores vacíos y devuelve opciones ordenadas para los selectores.
    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(values) if values else fallback


def html_value(value):
    return html.escape(str(value))


def markdown_to_clean_html(text):
    if not text:
        return ""
    # Convierte el subconjunto de Markdown usado por el informe a HTML seguro para la vista.
    formatted = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    lines = formatted.split("\n")
    formatted_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("• ") or stripped.startswith("* "):
            item_text = stripped[2:].strip()
            if not in_list:
                formatted_lines.append('<ul style="margin: 0.4rem 0; padding-left: 1.2rem;">')
                in_list = True
            formatted_lines.append(f'<li style="margin-bottom: 0.35rem;">{item_text}</li>')
        else:
            if in_list:
                formatted_lines.append("</ul>")
                in_list = False
            if stripped:
                formatted_lines.append(f'<p style="margin: 0.4rem 0;">{stripped}</p>')
    if in_list:
        formatted_lines.append("</ul>")
    return "\n".join(formatted_lines)


def parse_informe_sections(informe_text):
    # Divide el informe generado por sus encabezados Markdown.
    sections = {}
    current_key = "General"
    current_lines = []

    for line in informe_text.split("\n"):
        if line.strip().startswith("#"):
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
                current_lines = []
            header_title = line.strip().lstrip("#").strip()
            current_key = header_title
        else:
            if line.strip().startswith("---") or line.strip().startswith("==="):
                continue
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def find_bento_section(sections_dict, keywords):
    for title, content in sections_dict.items():
        if any(kw.lower() in title.lower() for kw in keywords):
            return title, content
    return None, None


def render_informe_bento_grid(informe_text):
    if not informe_text:
        return

    # Organiza las secciones del informe en tarjetas de la interfaz.
    sections = parse_informe_sections(informe_text)
    if len(sections) < 2:
        st.markdown(informe_text)
        return

    title_res, content_res = find_bento_section(sections, ["resumen"])
    title_diag, content_diag = find_bento_section(sections, ["diagnóstico", "diagnostico"])
    title_comp, content_comp = find_bento_section(sections, ["comparativo"])
    title_afin, content_afin = find_bento_section(sections, ["afinidad"])
    title_fort, content_fort = find_bento_section(sections, ["fortalezas"])
    title_reco, content_reco = find_bento_section(sections, ["recomendaciones"])

    used_titles = set(t for t in [title_res, title_diag, title_comp, title_afin, title_fort, title_reco] if t)

    def _render_card(title, content, default_title):
        display_title = title or default_title
        try:
            with st.container(border=True):
                st.markdown(f"#### {display_title}")
                st.markdown(content)
        except (TypeError, AttributeError):
            st.markdown(f"#### {display_title}")
            st.markdown(content)

    # Fila 1 del Bento: Resumen (70%) + Diagnóstico (50%)
    c1, c2 = st.columns([5, 5])
    with c1:
        if content_res:
            _render_card(title_res, content_res, "📋 Resumen del perfil")
    with c2:
        if content_diag:
            _render_card(title_diag, content_diag, "📊 Diagnóstico de admisión")

    # Fila 2 del Bento: Comparativo (50%) + Afinidad (50%)
    c3, c4 = st.columns([5, 5])
    with c3:
        if content_comp:
            _render_card(title_comp, content_comp, "📈 Análisis comparativo")
    with c4:
        if content_afin:
            _render_card(title_afin, content_afin, "🧭 Afinidad contextual")

    # Fila 3 del Bento: Fortalezas (50%) + Recomendaciones (50%)
    c5, c6 = st.columns([6, 6])
    with c5:
        if content_fort:
            _render_card(title_fort, content_fort, "⭐ Fortalezas observadas")
    with c6:
        if content_reco:
            _render_card(title_reco, content_reco, "🎯 Recomendaciones")

    # Secciones adicionales fuera del esquema estándar
    for title, content in sections.items():
        if title not in used_titles and content.strip():
            _render_card(title, content, title)


def numeric_series(dataframe, column):
    if dataframe is None or column is None or column not in dataframe.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce")


def build_context_similarity(dataframe, profile, context_columns):
    """Retorna similitud por fila usando solo las variables disponibles en el EDA."""
    if dataframe is None or dataframe.empty:
        return pd.Series(dtype=float, index=getattr(dataframe, "index", None))

    # Calcula una similitud por variable y luego la promedia por registro.
    components = []

    for key, selected_value in profile.items():
        column = context_columns.get(key)
        if column is None or column not in dataframe.columns:
            continue

        if key == "income_quantile":
            values = pd.to_numeric(dataframe[column], errors="coerce")
            selected_numeric = pd.to_numeric(pd.Series([selected_value]), errors="coerce").iloc[0]
            if pd.isna(selected_numeric):
                continue
            component = (1 - (values - selected_numeric).abs() / 9).clip(lower=0, upper=1)
            components.append(component.fillna(0))
        else:
            values = dataframe[column].astype(str).str.strip().str.casefold()
            selected = str(selected_value).strip().casefold()
            components.append(values.eq(selected).astype(float))

    if not components:
        return pd.Series(0.0, index=dataframe.index)

    return pd.concat(components, axis=1).mean(axis=1)


def filter_by_target(dataframe, institution, career, institution_col, career_col):
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    # Filtra progresivamente por institución y carrera sin modificar el dataframe original.
    subset = dataframe

    if institution and institution_col and institution_col in subset.columns:
        selected_institution = normalize_text_value(institution)
        institution_values = subset[institution_col].map(normalize_text_value)
        subset = subset[institution_values == selected_institution]

    if career and career_col and career_col in subset.columns:
        selected_career = normalize_text_value(career)
        career_values = subset[career_col].map(normalize_text_value)
        subset = subset[career_values == selected_career]

    return subset


def contextual_metrics(
    dataframe,
    profile,
    context_columns,
    institution,
    career,
    institution_col,
    career_col,
    ponderado_col,
):
    # Detecta qué variables contextuales pueden calcularse con las columnas disponibles.
    available_keys = [
        key
        for key, column in context_columns.items()
        if column is not None and dataframe is not None and column in dataframe.columns
    ]

    if dataframe is None or dataframe.empty or not available_keys:
        return {
            "available_keys": available_keys,
            "target_rows": 0,
            "target_scope": "none",
            "target_affinity": None,
            "exact_matches": 0,
            "nearest_rows": 0,
            "nearest_ponderado": None,
        }

    # Mide similitud y selecciona primero la combinación exacta solicitada.
    similarity = build_context_similarity(dataframe, profile, context_columns)
    # Primero busca la combinación exacta carrera + institución. Si la muestra EDA
    # no contiene esa pareja, usa todos los registros de la carrera como respaldo.
    target = filter_by_target(
        dataframe,
        institution,
        career,
        institution_col,
        career_col,
    )
    target_scope = "career_institution"

    if target.empty and career_col and career_col in dataframe.columns:
        target = filter_by_target(
            dataframe,
            None,
            career,
            institution_col,
            career_col,
        )
        target_scope = "career"

    if target.empty and institution_col and institution_col in dataframe.columns:
        target = filter_by_target(
            dataframe,
            institution,
            None,
            institution_col,
            career_col,
        )
        target_scope = "institution"

    if target.empty:
        target_scope = "none"

    target_similarity = similarity.reindex(target.index) if not target.empty else pd.Series(dtype=float)
    target_affinity = float(target_similarity.mean() * 100) if not target_similarity.empty else None

    # Cuenta coincidencias exactas y obtiene el grupo de perfiles más cercanos.
    exact_mask = similarity.ge(0.999999)
    exact_matches = int(exact_mask.sum())

    nearest_count = max(1, min(len(dataframe), max(50, int(len(dataframe) * 0.05))))
    nearest_index = similarity.nlargest(nearest_count).index
    nearest_rows = dataframe.loc[nearest_index]

    nearest_ponderado = None
    if ponderado_col and ponderado_col in nearest_rows.columns:
        ponderado_values = numeric_series(nearest_rows, ponderado_col).dropna()
        if not ponderado_values.empty:
            nearest_ponderado = float(ponderado_values.mean())

    return {
        "available_keys": available_keys,
        "target_rows": len(target),
        "target_scope": target_scope,
        "target_affinity": target_affinity,
        "exact_matches": exact_matches,
        "nearest_rows": len(nearest_rows),
        "nearest_ponderado": nearest_ponderado,
    }


def contextual_affinity_for_target(
    dataframe,
    profile,
    context_columns,
    institution,
    career,
    institution_col,
    career_col,
):
    subset = filter_by_target(
        dataframe,
        institution,
        career,
        institution_col,
        career_col,
    )
    if subset.empty:
        subset = filter_by_target(
            dataframe,
            None,
            career,
            institution_col,
            career_col,
        )
    if subset.empty:
        return None

    similarity = build_context_similarity(subset, profile, context_columns)
    return float(similarity.mean() * 100) if not similarity.empty else None


def enrich_recommendations(
    recommendations,
    dataframe,
    profile,
    context_columns,
    institution_col,
    career_col,
):
    if recommendations is None or recommendations.empty:
        return recommendations

    # Añade afinidad contextual sin reemplazar la probabilidad original.
    enriched = recommendations.copy()
    affinities = []

    for _, row in enriched.iterrows():
        affinity = contextual_affinity_for_target(
            dataframe,
            profile,
            context_columns,
            row.get("institucion", ""),
            row.get("carrera", ""),
            institution_col,
            career_col,
        )
        affinities.append(affinity)

    enriched["afinidad_contextual"] = affinities
    probability = pd.to_numeric(enriched["probabilidad"], errors="coerce").fillna(0)
    affinity_numeric = pd.to_numeric(enriched["afinidad_contextual"], errors="coerce")

    # El índice combinado solo ordena las alternativas, no cambia su probabilidad.
    enriched["indice_combinado"] = np.where(
        affinity_numeric.notna(),
        probability * 0.75 + affinity_numeric.fillna(0) * 0.25,
        probability,
    )
    return enriched.sort_values("indice_combinado", ascending=False)


def classify_career_area(value):
    """Agrupa una carrera o tipo de programa en Ingeniería, Técnica u Otra."""
    normalized = normalize_text_value(value)
    if "ingenier" in normalized:
        return "Ingeniería"
    if any(token in normalized for token in ("tecnic", "tecnolog", "tecnico nivel superior")):
        return "Técnica"
    return "Otra"


def calculate_area_distribution(
    dataframe,
    profile,
    context_columns,
    career_col,
    career_type_col=None,
):
    """Distribución vocacional entre los perfiles históricos más cercanos."""
    empty_result = {
        "Ingeniería": None,
        "Técnica": None,
        "Otra": None,
        "sample_size": 0,
    }
    if dataframe is None or dataframe.empty:
        return empty_result

    # Selecciona la columna que permite clasificar las carreras por área.
    source_col = None
    if career_type_col and career_type_col in dataframe.columns:
        source_col = career_type_col
    elif career_col and career_col in dataframe.columns:
        source_col = career_col

    if source_col is None:
        return empty_result

    # Usa los perfiles contextualmente más cercanos para estimar la distribución de áreas.
    similarity = build_context_similarity(dataframe, profile, context_columns)
    if similarity.empty:
        return empty_result

    nearest_count = max(1, min(len(dataframe), max(50, int(len(dataframe) * 0.05))))
    nearest_index = similarity.nlargest(nearest_count).index
    nearest = dataframe.loc[nearest_index, source_col].dropna()
    if nearest.empty:
        return empty_result

    areas = nearest.map(classify_career_area)
    distribution = areas.value_counts(normalize=True).mul(100)
    return {
        "Ingeniería": float(distribution.get("Ingeniería", 0.0)),
        "Técnica": float(distribution.get("Técnica", 0.0)),
        "Otra": float(distribution.get("Otra", 0.0)),
        "sample_size": int(len(nearest)),
    }


# ---------------------------------------------------------
# Encabezado y carga
# ---------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1>🎓 Predictor de Selección Universitaria DEMRE</h1>
    <p>Predicción por puntajes, perfil contextual, análisis socioeconómico y exploración de datos en una sola aplicación.</p>
</div>
""",
    unsafe_allow_html=True,
)

try:
    stats_df, model_artifact, eda_sample = load_app_artifacts()
except Exception as exc:
    st.error(f"Error al cargar los datos y modelos: {exc}")
    st.stop()


# Columnas detectadas en la muestra EDA.
COLUMN_ALIASES = {
    "gender": ["sexo", "genero", "nombre_sexo"],
    "nationality": ["nacionalidad", "nombre_nacionalidad"],
    "health": ["salud", "nombre_cobertura_salud", "cobertura_salud"],
    "school_branch": [
        "rama_resumen",
        "rama_educacional",
        "nombre_rama_educacional",
        "tipo_ensenanza",
    ],
    "school_dependency": [
        "dependencia",
        "nombre_dependencia_establecimiento",
        "dependencia_establecimiento",
    ],
    "region": ["region", "nombre_region", "region_establecimiento", "region_egreso"],
    "income_quantile": ["cuantil_ingreso", "cuantil_ingreso_bruto_fam"],
    "living_arrangement": [
        "viven_padres",
        "vive_con_padres",
        "convivencia_familiar",
        "con_quien_vive",
    ],
    "household_head": ["jefe_familia", "jefe_hogar", "principal_sostenedor"],
    "employment": ["trabajo_remunerado", "situacion_laboral", "trabaja"],
    "financing": [
        "financiamiento",
        "descripcion_fuente_financiamiento_estudio_superior_primaria",
        "fuente_financiamiento",
    ],
}

context_columns = {
    key: find_column(eda_sample, aliases) for key, aliases in COLUMN_ALIASES.items()
}
eda_institution_col = find_column(
    eda_sample,
    ["nombre_institucion_educacion_superior", "institucion", "universidad"],
)
eda_career_col = find_column(
    eda_sample,
    ["nombre_carrera_normalizacion", "carrera", "nombre_carrera"],
)
ponderado_col = find_column(
    eda_sample,
    ["puntaje_ponderado_estimado", "puntaje_ponderado", "ponderado_total"],
)
eda_career_type_col = find_column(
    eda_sample,
    ["tipo_carrera", "tipo_programa", "nivel_carrera", "area_carrera"],
)


# ---------------------------------------------------------
# Sidebar: inputs originales + inputs recuperados de app.py
# ---------------------------------------------------------
st.sidebar.markdown("### 📝 Tus Puntajes (PAES / PSU)")
nem_score = st.sidebar.number_input(
    "Puntaje NEM", min_value=150, max_value=1000, value=650, step=5
)
leng_score = st.sidebar.number_input(
    "Competencia Lectora / Lenguaje",
    min_value=150,
    max_value=1000,
    value=620,
    step=5,
)
mate_score = st.sidebar.number_input(
    "Matemática (M1)", min_value=150, max_value=1000, value=640, step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Prueba Electiva")
opt_test = st.sidebar.radio(
    "Selecciona tu prueba específica:",
    ["Ciencias", "Historia / Ciencias Sociales"],
)

if opt_test == "Ciencias":
    cien_score = st.sidebar.number_input(
        "Puntaje Ciencias", min_value=150, max_value=1000, value=610, step=5
    )
    hycs_score = np.nan
else:
    hycs_score = st.sidebar.number_input(
        "Puntaje Historia", min_value=150, max_value=1000, value=630, step=5
    )
    cien_score = np.nan

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ Carrera e Institución Objetivo")

carreras_list = sorted(
    stats_df["nombre_carrera_normalizacion"].dropna().unique().tolist()
)

# Primero se selecciona la carrera. La institución depende de esta selección.
selected_carrera = st.sidebar.selectbox(
    "Selecciona la Carrera:",
    carreras_list,
    key="selector_carrera_objetivo",
)

filtered_instituciones = sorted(
    stats_df.loc[
        stats_df["nombre_carrera_normalizacion"] == selected_carrera,
        "nombre_institucion_educacion_superior",
    ]
    .dropna()
    .unique()
    .tolist()
)

if not filtered_instituciones:
    st.sidebar.error("No se encontraron instituciones para la carrera seleccionada.")
    st.stop()

selected_inst = st.sidebar.selectbox(
    "Selecciona la Universidad / Institución:",
    filtered_instituciones,
    key="selector_institucion_objetivo",
)

with st.sidebar.expander("🧭 Perfil contextual del postulante", expanded=True):
    st.caption(
        "Estas variables se usan para medir similitud histórica dentro de la muestra EDA. "
        "La probabilidad del modelo de admisión continúa basada en los puntajes."
    )

    u_sexo = st.selectbox(
        "Género",
        clean_options(eda_sample, context_columns["gender"], ["Femenino", "Masculino"]),
        help="Permite observar patrones históricos de participación por género.",
    )
    u_nac = st.selectbox(
        "Nacionalidad",
        clean_options(
            eda_sample,
            context_columns["nationality"],
            ["Chilena", "Extranjera"],
        ),
        help="Permite comparar perfiles nacionales y extranjeros cuando la variable existe.",
    )
    u_salud = st.selectbox(
        "Previsión de Salud",
        clean_options(eda_sample, context_columns["health"], ["FONASA", "ISAPRE"]),
        help="Se utiliza como indicador contextual socioeconómico, no como criterio de admisión.",
    )
    u_rama = st.selectbox(
        "Rama Educacional",
        clean_options(
            eda_sample,
            context_columns["school_branch"],
            ["Humanista Científico", "Técnico Profesional"],
        ),
        help="Compara la formación escolar con los perfiles históricos de cada carrera.",
    )
    u_dep = st.selectbox(
        "Dependencia Colegio",
        clean_options(
            eda_sample,
            context_columns["school_dependency"],
            ["Municipal", "Particular Subvencionado", "Particular Pagado"],
        ),
        help="Permite analizar diferencias históricas entre tipos de establecimiento.",
    )
    u_region = st.selectbox(
        "Región de Origen",
        clean_options(
            eda_sample,
            context_columns["region"],
            ["Región Metropolitana", "Valparaíso", "Biobío"],
        ),
        help="Aporta contexto territorial cuando esta variable está disponible.",
    )
    u_cuantil = st.slider(
        "Tramo Ingreso (1-10)",
        1,
        10,
        5,
        help="Se compara por cercanía con el cuantil de ingreso familiar del dataset.",
    )
    u_viven = st.selectbox(
        "Convivencia Familiar",
        clean_options(
            eda_sample,
            context_columns["living_arrangement"],
            ["Con ambos padres", "Solo con la madre", "Independiente"],
        ),
        help="Describe la estructura de convivencia registrada en el perfil.",
    )
    u_jefe = st.selectbox(
        "Jefe de Hogar",
        clean_options(
            eda_sample,
            context_columns["household_head"],
            ["Padre", "Madre", "Postulante"],
        ),
        help="Identifica la figura principal del hogar cuando el dataset la contiene.",
    )
    u_trabajo = st.selectbox(
        "Situación Laboral",
        clean_options(
            eda_sample,
            context_columns["employment"],
            ["No", "Ocasionalmente", "Sí, permanentemente"],
        ),
        help="Permite estudiar compatibilidad histórica con modalidades y duración de carrera.",
    )
    u_finan = st.selectbox(
        "Fuente Financiamiento",
        clean_options(
            eda_sample,
            context_columns["financing"],
            ["Aporte familiar", "Crédito/Beca", "Ingresos propios"],
        ),
        help="Se utiliza para analizar el contexto financiero histórico de los postulantes.",
    )

btn_calcular = st.sidebar.button(
    "🚀 Calcular admisión y afinidad",
    type="primary",
    use_container_width=True,
)

current_profile = {
    "gender": u_sexo,
    "nationality": u_nac,
    "health": u_salud,
    "school_branch": u_rama,
    "school_dependency": u_dep,
    "region": u_region,
    "income_quantile": u_cuantil,
    "living_arrangement": u_viven,
    "household_head": u_jefe,
    "employment": u_trabajo,
    "financing": u_finan,
}

current_inputs = {
    "nem_score": int(nem_score),
    "leng_score": int(leng_score),
    "mate_score": int(mate_score),
    "opt_test": opt_test,
    "hycs_score": None if np.isnan(hycs_score) else int(hycs_score),
    "cien_score": None if np.isnan(cien_score) else int(cien_score),
    "selected_inst": selected_inst,
    "selected_carrera": selected_carrera,
    "profile": current_profile.copy(),
}

if btn_calcular:
    with st.spinner("Procesando puntajes y variables contextuales..."):
        # Calcula predicción, afinidad, recomendaciones y distribución en un mismo ciclo.
        calculated_context = contextual_metrics(
            eda_sample,
            current_profile,
            context_columns,
            selected_inst,
            selected_carrera,
            eda_institution_col,
            eda_career_col,
            ponderado_col,
        )

        calculated_prediction = predict_admission_probability(
            nem_score,
            leng_score,
            mate_score,
            hycs_score if not np.isnan(hycs_score) else 0,
            cien_score if not np.isnan(cien_score) else 0,
            selected_inst,
            selected_carrera,
            stats_df,
            model_artifact,
        )

        calculated_recommendations = recommend_alternative_careers(
            nem_score,
            leng_score,
            mate_score,
            hycs_score if not np.isnan(hycs_score) else 0,
            cien_score if not np.isnan(cien_score) else 0,
            selected_carrera,
            stats_df,
            top_n=5,
        )
        calculated_recommendations = enrich_recommendations(
            calculated_recommendations,
            eda_sample,
            current_profile,
            context_columns,
            eda_institution_col,
            eda_career_col,
        )

        calculated_area_distribution = calculate_area_distribution(
            eda_sample,
            current_profile,
            context_columns,
            eda_career_col,
            eda_career_type_col,
        )

        # Guarda el último cálculo confirmado para mantenerlo estable entre recargas.
        st.session_state["admission_calculation"] = {
            "inputs": current_inputs,
            "prediction": calculated_prediction,
            "context": calculated_context,
            "recommendations": calculated_recommendations,
            "area_distribution": calculated_area_distribution,
        }

    # Genera el informe interpretativo a partir de los resultados recién calculados.
    with st.spinner("Generando informe de orientación vocacional con IA..."):
        prompt_perfil = {
            "sexo": u_sexo,
            "nacionalidad": u_nac,
            "region": u_region,
            "dependencia": u_dep,
            "rama": u_rama,
            "cuantil": u_cuantil,
            "trabajo": u_trabajo,
            "financiamiento": u_finan,
            "salud": u_salud,
            "convivencia": u_viven,
            "jefe_hogar": u_jefe,
            "carrera": selected_carrera,
            "institucion": selected_inst,
        }

        cien_val = cien_score if not np.isnan(cien_score) else None
        hycs_val = hycs_score if not np.isnan(hycs_score) else None
        prompt_puntajes = {
            "nem": nem_score,
            "lenguaje": leng_score,
            "matematica": mate_score,
            "tipo_electiva": opt_test,
            "puntaje_electiva": cien_val if opt_test == "Ciencias" else hycs_val,
        }

        recs_list = (
            calculated_recommendations.to_dict(orient="records")
            if calculated_recommendations is not None
            and not calculated_recommendations.empty
            else []
        )

        # Construye el prompt con datos estructurados y solicita el informe a HuggingFace.
        prompt_text = construir_prompt(
            perfil=prompt_perfil,
            puntajes=prompt_puntajes,
            prediccion=calculated_prediction,
            contexto=calculated_context,
            distribucion_areas=calculated_area_distribution,
            carreras_alternativas=recs_list,
        )
        informe_ia = generar_informe(prompt_text)
        st.session_state["informe_ia"] = informe_ia

    st.sidebar.success("Resultados actualizados.")

# Recupera el último cálculo para no actualizar resultados solo por cambiar un selector.
calculation = st.session_state.get("admission_calculation")
results_ready = calculation is not None

if results_ready:
    calculated_inputs = calculation["inputs"]
    has_pending_changes = current_inputs != calculated_inputs
    if has_pending_changes:
        st.sidebar.warning(
            "Hay cambios pendientes. Presiona el botón para actualizar los resultados."
        )

    # Desde aquí, todos los resultados visuales usan exclusivamente la última
    # configuración confirmada mediante el botón.
    nem_score = calculated_inputs["nem_score"]
    leng_score = calculated_inputs["leng_score"]
    mate_score = calculated_inputs["mate_score"]
    opt_test = calculated_inputs["opt_test"]
    hycs_score = (
        np.nan
        if calculated_inputs["hycs_score"] is None
        else calculated_inputs["hycs_score"]
    )
    cien_score = (
        np.nan
        if calculated_inputs["cien_score"] is None
        else calculated_inputs["cien_score"]
    )
    selected_inst = calculated_inputs["selected_inst"]
    selected_carrera = calculated_inputs["selected_carrera"]
    profile = calculated_inputs["profile"]
    u_sexo = profile["gender"]
    u_nac = profile["nationality"]
    u_salud = profile["health"]
    u_rama = profile["school_branch"]
    u_dep = profile["school_dependency"]
    u_region = profile["region"]
    u_cuantil = profile["income_quantile"]
    u_viven = profile["living_arrangement"]
    u_jefe = profile["household_head"]
    u_trabajo = profile["employment"]
    u_finan = profile["financing"]
    prediction = calculation["prediction"]
    context = calculation["context"]
    recs = calculation["recommendations"]
    area_distribution = calculation["area_distribution"]

else:
    profile = current_profile
    prediction = None
    context = None
    recs = None
    area_distribution = {
        "Ingeniería": None,
        "Técnica": None,
        "Otra": None,
        "sample_size": 0,
    }


# ---------------------------------------------------------
# Navegación
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Predictor de Selección",
        "🧭 Perfil y Afinidad Contextual",
        "📊 Análisis Exploratorio (EDA)",
        "💰 Correlaciones Socioeconómicas",
        "📁 Explorador de Datos",
    ]
)


# =========================================================
# TAB 1: Predictor
# =========================================================
with tab1:
    if not results_ready:
        st.info('Configura tus datos y presiona “🚀 Calcular admisión y afinidad” para generar la predicción.')
    else:
        # Presenta la probabilidad y las alternativas usando el cálculo confirmado.
        col_main, col_stats = st.columns([1.2, 1.0])

        with col_main:
            st.markdown(
                f"""
            <div class="prob-card {html_value(prediction['badge_class'])}">
                <div class="prob-title">{html_value(prediction['etiqueta'])}</div>
                <div class="prob-val">{html_value(prediction['probabilidad'])}%</div>
                <p>Estimación para <b>{html_value(selected_carrera)}</b> en <b>{html_value(selected_inst)}</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Tu Ponderado Estimado", f"{prediction['user_ponderado']} pts")
            with m2:
                st.metric("Puntaje Corte (P25)", f"{prediction['corte_p25']} pts")
            with m3:
                st.metric("Promedio Carrera (P50)", f"{prediction['promedio_p50']} pts")

            st.markdown("---")
            st.markdown("#### 📈 Tus puntajes vs. promedio histórico")

            user_scores = [nem_score, leng_score, mate_score]
            career_averages = [
                prediction["promedios_carrera"]["NEM"],
                prediction["promedios_carrera"]["Lenguaje"],
                prediction["promedios_carrera"]["Matemáticas"],
            ]
            categories = ["NEM", "Lenguaje / Comp. Lectora", "Matemática (M1)"]

            if opt_test == "Ciencias":
                user_scores.append(cien_score)
                career_averages.append(prediction["promedios_carrera"]["Ciencias"])
                categories.append("Ciencias")
            else:
                user_scores.append(hycs_score)
                career_averages.append(prediction["promedios_carrera"]["Historia"])
                categories.append("Historia")

            fig_comp = go.Figure(
                data=[
                    go.Bar(
                        name="Tus Puntajes",
                        x=categories,
                        y=user_scores,
                        marker_color="#1E88E5",
                    ),
                    go.Bar(
                        name="Promedio Carrera (DEMRE)",
                        x=categories,
                        y=career_averages,
                        marker_color="#FFA726",
                    ),
                ]
            )
            fig_comp.update_layout(
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_stats:
            st.markdown("#### 💡 Carreras alternativas")
            st.write(
                "Se conserva la probabilidad entregada por el modelo y se agrega la afinidad "
                "contextual únicamente para ordenar las alternativas."
            )

            if recs is not None and not recs.empty:
                for _, row in recs.iterrows():
                    affinity = row.get("afinidad_contextual")
                    affinity_text = (
                        f" | Afinidad contextual: {affinity:.1f}%"
                        if pd.notna(affinity)
                        else ""
                    )
                    st.markdown(
                        f"""
                    <div class="recommendation-card">
                        <div style="font-weight:700; color:#1E88E5;">{html_value(row['carrera'])}</div>
                        <div style="font-size:0.88rem; color:#495057;">🏛️ {html_value(row['institucion'])}</div>
                        <div style="font-size:0.85rem; margin-top:0.3rem;">
                            <b>Probabilidad estimada:</b>
                            <span style="color:#2E7D32; font-weight:700;">{html_value(row['probabilidad'])}%</span>
                            | Corte: {html_value(row['corte_estimado'])} pts{html_value(affinity_text)}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No se encontraron carreras alternativas para los puntajes ingresados.")


# =========================================================
# TAB 2: Perfil y afinidad contextual
# =========================================================
with tab2:
    if not results_ready:
        st.info('Presiona “🚀 Calcular admisión y afinidad” para generar la ficha y las tarjetas dinámicas.')
    else:
        # Muestra el perfil contextual y la comparación con registros históricos.
        st.markdown("### 🧭 Ficha vocacional y similitud histórica")
        st.caption(
            "Este bloque incorpora todos los inputs contextuales de app.py. No reemplaza la "
            "predicción de admisión; muestra qué tan frecuente es un perfil parecido en los datos históricos."
        )

        st.markdown(
            f"""
        <div class="profile-card">
            <h3 style="margin-top:0; color:#0D47A1;">📋 Perfil del postulante</h3>
            <span class="profile-tag">👤 <b>Género:</b> {html_value(u_sexo)}</span>
            <span class="profile-tag">🌎 <b>Nacionalidad:</b> {html_value(u_nac)}</span>
            <span class="profile-tag">🏥 <b>Salud:</b> {html_value(u_salud)}</span>
            <span class="profile-tag">📘 <b>Rama:</b> {html_value(u_rama)}</span>
            <span class="profile-tag">🏫 <b>Colegio:</b> {html_value(u_dep)}</span>
            <span class="profile-tag">📍 <b>Región:</b> {html_value(u_region)}</span>
            <span class="profile-tag">💰 <b>Ingreso:</b> Cuantil {html_value(u_cuantil)}</span>
            <span class="profile-tag">🏡 <b>Convivencia:</b> {html_value(u_viven)}</span>
            <span class="profile-tag">👪 <b>Jefe de hogar:</b> {html_value(u_jefe)}</span>
            <span class="profile-tag">💼 <b>Trabajo:</b> {html_value(u_trabajo)}</span>
            <span class="profile-tag">💳 <b>Financiamiento:</b> {html_value(u_finan)}</span>
            <span class="profile-tag">🎯 <b>Carrera:</b> {html_value(selected_carrera)}</span>
            <span class="profile-tag">🏛️ <b>Institución:</b> {html_value(selected_inst)}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        target_scope = context.get("target_scope", "none")
        scope_config = {
            "career_institution": (
                "Afinidad carrera/institución",
                "Registros carrera/institución",
                "Se encontraron registros históricos de la combinación exacta seleccionada.",
            ),
            "career": (
                "Afinidad con la carrera",
                "Registros de la carrera",
                "La muestra EDA no tenía la combinación exacta; se utilizaron todos los registros disponibles de la carrera.",
            ),
            "institution": (
                "Afinidad en la institución",
                "Registros de la institución",
                "La muestra EDA no tenía registros de la carrera; se utilizaron registros disponibles de la institución.",
            ),
            "none": (
                "Afinidad contextual",
                "Registros encontrados",
                "No existen registros compatibles en la muestra EDA.",
            ),
        }
        affinity_label, records_label, scope_help = scope_config[target_scope]

        st.markdown("#### 🏆 Tarjetas Dinámicas de Afinidad y Coincidencia")

        pct_affinity = context.get("target_affinity")
        pct_engineering = area_distribution.get("Ingeniería")
        pct_technical = area_distribution.get("Técnica")
        pct_other = area_distribution.get("Otra")

        def format_card_percentage(value):
            return f"{value:.1f}%" if value is not None else "Sin datos"

        card1, card2, card3, card4 = st.columns(4)
        with card1:
            st.markdown(
                f"""
                <div class="affinity-card affinity-card-amber">
                    <div class="affinity-pill affinity-pill-amber"><span class="affinity-dot" style="background:#FBBF24; box-shadow:0 0 8px #FBBF24;"></span>CARRERA CONSULTADA</div>
                    <div class="affinity-value" style="color:#FBBF24;">{html_value(format_card_percentage(pct_affinity))}</div>
                    <div class="affinity-label">{html_value(selected_carrera)}</div>
                    <p class="affinity-description">Coincidencia de tu perfil con los registros históricos disponibles para esta carrera.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card2:
            st.markdown(
                f"""
                <div class="affinity-card affinity-card-indigo">
                    <div class="affinity-pill affinity-pill-indigo"><span class="affinity-dot" style="background:#818CF8; box-shadow:0 0 8px #818CF8;"></span>ÁREA TECNOLÓGICA</div>
                    <div class="affinity-value" style="color:#A5B4FC;">{html_value(format_card_percentage(pct_engineering))}</div>
                    <div class="affinity-label">Ingenierías en general</div>
                    <p class="affinity-description">Distribución entre los perfiles históricos más cercanos al perfil ingresado.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card3:
            st.markdown(
                f"""
                <div class="affinity-card affinity-card-emerald">
                    <div class="affinity-pill affinity-pill-emerald"><span class="affinity-dot" style="background:#34D399; box-shadow:0 0 8px #34D399;"></span>ÁREA APLICADA</div>
                    <div class="affinity-value" style="color:#6EE7B7;">{html_value(format_card_percentage(pct_technical))}</div>
                    <div class="affinity-label">Carreras técnicas</div>
                    <p class="affinity-description">Participación de programas técnicos entre los perfiles históricos similares.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card4:
            st.markdown(
                f"""
                <div class="affinity-card affinity-card-purple">
                    <div class="affinity-pill affinity-pill-purple"><span class="affinity-dot" style="background:#C084FC; box-shadow:0 0 8px #C084FC;"></span>OTRAS DISCIPLINAS</div>
                    <div class="affinity-value" style="color:#C084FC;">{html_value(format_card_percentage(pct_other))}</div>
                    <div class="affinity-label">Otras áreas</div>
                    <p class="affinity-description">Participación de salud, humanidades y otras disciplinas en perfiles similares.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        detail1, detail2, detail3 = st.columns(3)
        with detail1:
            st.metric(records_label, f"{context['target_rows']:,}", help=scope_help)
        with detail2:
            st.metric(
                "Coincidencias exactas",
                f"{context['exact_matches']:,}",
                help="Filas que coinciden con todas las variables contextuales disponibles.",
            )
        with detail3:
            nearest_ponderado = context["nearest_ponderado"]
            st.metric(
                "Ponderado perfiles cercanos",
                f"{nearest_ponderado:.1f} pts" if nearest_ponderado is not None else "Sin datos",
            )

        if target_scope == "career":
            st.info(
                "La muestra EDA no contiene la combinación exacta de carrera e institución. "
                "Para evitar mostrar ‘Sin datos’, la afinidad se calculó usando todos los registros de la carrera seleccionada."
            )
        elif target_scope == "institution":
            st.warning(
                "La muestra EDA tampoco contiene registros de la carrera seleccionada. "
                "La afinidad mostrada usa los registros disponibles de la institución."
            )

        detected_labels = {
            "gender": "Género",
            "nationality": "Nacionalidad",
            "health": "Previsión de salud",
            "school_branch": "Rama educacional",
            "school_dependency": "Dependencia escolar",
            "region": "Región",
            "income_quantile": "Cuantil de ingreso",
            "living_arrangement": "Convivencia familiar",
            "household_head": "Jefe de hogar",
            "employment": "Situación laboral",
            "financing": "Financiamiento",
        }
        detected = [detected_labels[key] for key in context["available_keys"]]
        missing = [
            detected_labels[key]
            for key in detected_labels
            if key not in context["available_keys"]
        ]

        if detected:
            st.success(
                f"Variables encontradas y aplicadas en el dataset: {', '.join(detected)}."
            )
        else:
            st.warning(
                "La muestra EDA no contiene columnas compatibles con los inputs contextuales. "
                "Los campos siguen disponibles en la ficha, pero no pueden calcular afinidad histórica."
            )

        if missing:
            st.info(
                "Variables sin columna equivalente en la muestra EDA: "
                f"{', '.join(missing)}. Se omiten del cálculo, sin inventar datos."
            )

        st.markdown("#### 🔍 Influencia de los principales grupos de variables")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                f"""
            <div class="context-card">
                <b style="color: rgb(13, 71, 161)">📘 Entorno escolar</b>
                <p style="color:#263238; font-weight: bold">Rama <b>{html_value(u_rama)}</b> y dependencia <b>{html_value(u_dep)}</b>. Se comparan con la distribución histórica de la carrera elegida.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                f"""
            <div class="context-card">
                <b style="color: rgb(13, 71, 161)">🏡 Contexto familiar</b>
                <p style="color:#263238; font-weight: bold">Cuantil <b>{html_value(u_cuantil)}</b>, convivencia <b>{html_value(u_viven)}</b> y jefe de hogar <b>{html_value(u_jefe)}</b>. Se usan solo cuando existen columnas equivalentes.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                f"""
            <div class="context-card">
                <b style="color: rgb(13, 71, 161)">💼 Empleo y financiamiento</b>
                <p style="color:#263238; font-weight: bold">Situación <b>{html_value(u_trabajo)}</b> y fuente <b>{html_value(u_finan)}</b>. Complementan el análisis, pero no reducen ni aumentan directamente la probabilidad del clasificador.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Informe de IA al final de Tab 2 ──
        st.markdown("---")
        st.markdown("### 🤖 Informe de Orientación Vocacional (IA)")
        st.caption(
            "Informe generado automáticamente por un modelo de lenguaje (Qwen 2.5) "
            "a partir de todos los resultados calculados. "
            "No reemplaza orientación profesional humana."
        )

        informe_ia = st.session_state.get("informe_ia")

        if st.button("🔄 Regenerar informe IA", key="btn_regenerar_ia"):
            with st.spinner("Regenerando informe de orientación vocacional..."):
                # Reutiliza los resultados actuales para solicitar una nueva interpretación.
                prompt_perfil_regen = {
                    "sexo": u_sexo,
                    "nacionalidad": u_nac,
                    "region": u_region,
                    "dependencia": u_dep,
                    "rama": u_rama,
                    "cuantil": u_cuantil,
                    "trabajo": u_trabajo,
                    "financiamiento": u_finan,
                    "salud": u_salud,
                    "convivencia": u_viven,
                    "jefe_hogar": u_jefe,
                    "carrera": selected_carrera,
                    "institucion": selected_inst,
                }

                cien_regen = cien_score if not np.isnan(cien_score) else None
                hycs_regen = hycs_score if not np.isnan(hycs_score) else None
                prompt_puntajes_regen = {
                    "nem": nem_score,
                    "lenguaje": leng_score,
                    "matematica": mate_score,
                    "tipo_electiva": opt_test,
                    "puntaje_electiva": cien_regen if opt_test == "Ciencias" else hycs_regen,
                }

                recs_regen = (
                    recs.to_dict(orient="records")
                    if recs is not None and not recs.empty
                    else []
                )

                prompt_regen = construir_prompt(
                    perfil=prompt_perfil_regen,
                    puntajes=prompt_puntajes_regen,
                    prediccion=prediction,
                    contexto=context,
                    distribucion_areas=area_distribution,
                    carreras_alternativas=recs_regen,
                )
                informe_ia = generar_informe(prompt_regen)
                st.session_state["informe_ia"] = informe_ia

        if informe_ia:
            render_informe_bento_grid(informe_ia)
        else:
            st.info(
                "El informe de IA se genera automáticamente al presionar "
                '"🚀 Calcular admisión y afinidad". '
                "También puedes regenerarlo con el botón de arriba."
            )


# =========================================================
# TAB 3: EDA
# =========================================================
with tab3:
    # Resume la muestra EDA con métricas y gráficos de frecuencia.
    st.markdown("### 📊 Análisis exploratorio de datos del DEMRE")

    if eda_sample is None or eda_sample.empty:
        st.warning("No se encontró la muestra EDA demre_sample_eda.joblib.")
    else:
        career_col = eda_career_col
        institution_col = eda_institution_col
        nem_col = find_column(eda_sample, ["ptje_nem", "puntaje_nem"])

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Total Postulaciones Muestra", f"{len(eda_sample):,}")
        with e2:
            st.metric(
                "Carreras Registradas",
                f"{eda_sample[career_col].nunique():,}" if career_col else "Sin columna",
            )
        with e3:
            st.metric(
                "Instituciones",
                f"{eda_sample[institution_col].nunique():,}"
                if institution_col
                else "Sin columna",
            )
        with e4:
            nem_mean = numeric_series(eda_sample, nem_col).mean() if nem_col else np.nan
            st.metric(
                "Puntaje NEM Promedio",
                f"{nem_mean:.1f} pts" if pd.notna(nem_mean) else "Sin columna",
            )

        st.markdown("---")
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 🔥 Top carreras por postulaciones")
            if career_col:
                top_careers = (
                    eda_sample[career_col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(12)
                    .rename_axis("Carrera")
                    .reset_index(name="Postulaciones")
                )
                fig_careers = px.bar(
                    top_careers,
                    x="Postulaciones",
                    y="Carrera",
                    orientation="h",
                    color="Postulaciones",
                    color_continuous_scale="Blues",
                )
                fig_careers.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=450,
                    showlegend=False,
                )
                st.plotly_chart(fig_careers, use_container_width=True)
            else:
                st.info("No se encontró una columna de carrera en la muestra EDA.")

        with c_right:
            st.markdown("#### 🏛️ Top instituciones por postulaciones")
            if institution_col:
                top_institutions = (
                    eda_sample[institution_col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .rename_axis("Institución")
                    .reset_index(name="Postulaciones")
                )
                fig_institutions = px.bar(
                    top_institutions,
                    x="Postulaciones",
                    y="Institución",
                    orientation="h",
                    color="Postulaciones",
                    color_continuous_scale="Viridis",
                )
                fig_institutions.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=450,
                    showlegend=False,
                )
                st.plotly_chart(fig_institutions, use_container_width=True)
            else:
                st.info("No se encontró una columna de institución en la muestra EDA.")


# =========================================================
# TAB 4: Correlaciones socioeconómicas
# =========================================================
with tab4:
    # Compara variables académicas y socioeconómicas solo cuando están disponibles.
    st.markdown("### 💰 Correlaciones socioeconómicas y previsión")
    st.write(
        "Exploración de la relación entre ingreso, puntajes, salud, dependencia escolar "
        "y financiamiento. Los gráficos aparecen únicamente cuando existen las columnas requeridas."
    )

    if eda_sample is None or eda_sample.empty:
        st.warning("No se encontró la muestra EDA demre_sample_eda.joblib.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📉 Matriz de correlación numérica")
            corr_candidates = {
                "cuantil_ingreso_bruto_fam": "Cuantil Ingreso Fam.",
                "ptje_nem": "NEM",
                "ptje_leng": "Lenguaje",
                "ptje_mate": "Matemática",
                "ptje_cien": "Ciencias",
                "ptje_hycs": "Historia",
                "puntaje_ponderado_estimado": "Ponderado Total",
            }
            available_corr = {
                column: label
                for column, label in corr_candidates.items()
                if column in eda_sample.columns
            }

            if len(available_corr) >= 2:
                corr_data = eda_sample[list(available_corr)].apply(
                    pd.to_numeric, errors="coerce"
                )
                corr_matrix = corr_data.rename(columns=available_corr).corr()
                fig_heatmap = px.imshow(
                    corr_matrix.round(2),
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Matriz de Correlación de Pearson",
                )
                fig_heatmap.update_layout(height=450)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No hay suficientes columnas numéricas para construir la matriz.")

        with c2:
            st.markdown("#### 🏥 Ponderado promedio por previsión de salud")
            health_col = context_columns["health"]
            if health_col and ponderado_col:
                health_data = eda_sample[[health_col, ponderado_col]].copy()
                health_data[ponderado_col] = pd.to_numeric(
                    health_data[ponderado_col], errors="coerce"
                )
                health_df = (
                    health_data.dropna()
                    .groupby(health_col)[ponderado_col]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                health_df = health_df[health_df["count"] >= 50].sort_values(
                    by="mean", ascending=False
                )
                health_df.columns = [
                    "Sistema de Previsión",
                    "Ponderado Promedio",
                    "Cantidad Postulantes",
                ]

                if not health_df.empty:
                    fig_health = px.bar(
                        health_df,
                        x="Ponderado Promedio",
                        y="Sistema de Previsión",
                        orientation="h",
                        color="Ponderado Promedio",
                        color_continuous_scale="Teal",
                        text_auto=".1f",
                    )
                    fig_health.update_layout(
                        yaxis={"categoryorder": "total ascending"}, height=450
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                else:
                    st.info("No hay categorías con al menos 50 registros válidos.")
            else:
                st.info("Falta la columna de previsión o de puntaje ponderado.")

        st.markdown("---")
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### 🏫 Puntajes por dependencia escolar")
            dependency_col = context_columns["school_dependency"]
            score_aliases = {
                "NEM": ["ptje_nem", "puntaje_nem"],
                "Lenguaje": ["ptje_leng", "puntaje_lenguaje"],
                "Matemática": ["ptje_mate", "puntaje_matematica"],
                "Ponderado Promedio": [
                    "puntaje_ponderado_estimado",
                    "puntaje_ponderado",
                ],
            }
            score_columns = {
                label: find_column(eda_sample, aliases)
                for label, aliases in score_aliases.items()
            }
            score_columns = {
                label: col for label, col in score_columns.items() if col is not None
            }

            if dependency_col and score_columns:
                dep_data = eda_sample[[dependency_col, *score_columns.values()]].copy()
                for column in score_columns.values():
                    dep_data[column] = pd.to_numeric(dep_data[column], errors="coerce")
                dep_df = dep_data.groupby(dependency_col)[
                    list(score_columns.values())
                ].mean()
                dep_df = dep_df.rename(
                    columns={column: label for label, column in score_columns.items()}
                ).reset_index()
                dep_df = dep_df.rename(columns={dependency_col: "Dependencia"})

                fig_dependency = px.bar(
                    dep_df,
                    x="Dependencia",
                    y=list(score_columns.keys()),
                    barmode="group",
                    title="Promedio según tipo de colegio",
                )
                fig_dependency.update_layout(height=420)
                st.plotly_chart(fig_dependency, use_container_width=True)
            else:
                st.info("No se encontraron las columnas necesarias.")

        with c4:
            st.markdown("#### 💳 Fuente principal de financiamiento")
            financing_col = context_columns["financing"]
            if financing_col:
                financing_df = (
                    eda_sample[financing_col]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(7)
                    .rename_axis("Fuente de Financiamiento")
                    .reset_index(name="Cantidad")
                )
                fig_financing = px.pie(
                    financing_df,
                    names="Fuente de Financiamiento",
                    values="Cantidad",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_financing.update_layout(height=420)
                st.plotly_chart(fig_financing, use_container_width=True)
            else:
                st.info("No se encontró una columna de financiamiento.")

        st.markdown("---")
        st.markdown("#### 🔥 Asociación entre rama educacional y carrera")
        branch_col = context_columns["school_branch"]
        if branch_col and eda_career_col:
            top_career_names = (
                eda_sample[eda_career_col].value_counts().head(12).index.tolist()
            )
            categorical_sample = eda_sample[
                eda_sample[eda_career_col].isin(top_career_names)
            ]
            categorical_matrix = pd.crosstab(
                categorical_sample[branch_col],
                categorical_sample[eda_career_col],
                normalize="index",
            ) * 100
            if not categorical_matrix.empty:
                fig_categorical = px.imshow(
                    categorical_matrix,
                    labels={
                        "x": "Carrera",
                        "y": "Rama educacional",
                        "color": "% Asociación",
                    },
                    color_continuous_scale="Blues",
                    text_auto=".1f",
                    aspect="auto",
                )
                fig_categorical.update_layout(height=500)
                st.plotly_chart(fig_categorical, use_container_width=True)
            else:
                st.info("No existen suficientes combinaciones para la matriz categórica.")
        else:
            st.info("No se encontraron columnas compatibles de rama y carrera.")


# =========================================================
# TAB 5: Explorador de datos (multiselect recuperado)
# =========================================================
with tab5:
    # Permite inspeccionar y descargar una vista filtrada de la muestra EDA.
    st.markdown("### 📁 Explorador de datos completo")

    if eda_sample is None or eda_sample.empty:
        st.warning("No se encontró la muestra EDA demre_sample_eda.joblib.")
    else:
        preferred_defaults = [
            context_columns["gender"],
            context_columns["region"],
            context_columns["school_dependency"],
            context_columns["school_branch"],
            context_columns["income_quantile"],
            context_columns["employment"],
            context_columns["financing"],
            context_columns["health"],
            context_columns["living_arrangement"],
            ponderado_col,
            eda_institution_col,
            eda_career_col,
        ]
        default_columns = []
        for column in preferred_defaults:
            if column and column in eda_sample.columns and column not in default_columns:
                default_columns.append(column)
        if not default_columns:
            default_columns = eda_sample.columns[: min(10, len(eda_sample.columns))].tolist()

        var_filter = st.multiselect(
            "Seleccionar columnas visibles:",
            eda_sample.columns.tolist(),
            default=default_columns,
        )

        row_limit = st.slider(
            "Cantidad de filas a visualizar:",
            min_value=25,
            max_value=min(1000, max(25, len(eda_sample))),
            value=min(200, max(25, len(eda_sample))),
            step=25,
        )

        if var_filter:
            st.dataframe(
                eda_sample[var_filter].head(row_limit),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Selecciona al menos una columna para visualizar el dataset.")

        csv_data = eda_sample[var_filter].head(row_limit).to_csv(index=False).encode("utf-8") if var_filter else b""
        st.download_button(
            "⬇️ Descargar vista en CSV",
            data=csv_data,
            file_name="demre_vista_filtrada.csv",
            mime="text/csv",
            disabled=not bool(var_filter),
        )


st.markdown("---")
st.caption(
    "Samsung Innovation Campus Chile 2026 – Predictor DEMRE con perfil contextual integrado"
)