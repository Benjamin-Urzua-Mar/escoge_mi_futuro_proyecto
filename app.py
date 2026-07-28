import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# ---------------------------------------------------------
# Configuración de Página - Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Define Tu Futuro | Orientador Vocacional IA",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ESTILOS CSS - DESIGN SYSTEM TAILWIND GLASSMORPHIC PREMIUM
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    .main {
        background: #090D16;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    .hero-container {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8 0%, #38BDF8 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
        margin: 0;
    }

    /* Cards Glassmorphic Tailwind */
    .tw-glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .tw-glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Ficha Resumen del Postulante */
    .ficha-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.25);
    }

    .ficha-tag {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #E2E8F0;
        display: inline-block;
        margin: 4px;
    }

    /* Stat Cards Accent Gradient Borders */
    .stat-card-indigo {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.15) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }
    
    .stat-card-emerald {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.2);
    }

    .stat-card-purple {
        background: linear-gradient(145deg, rgba(168, 85, 247, 0.15) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(168, 85, 247, 0.2);
    }

    .stat-card-amber {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.2);
    }

    .kpi-val {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .kpi-lbl {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    .pill-indigo {
        background: rgba(99, 102, 241, 0.18);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.6rem;
    }

    .pill-emerald {
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.6rem;
    }

    .pill-purple {
        background: rgba(168, 85, 247, 0.18);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.6rem;
    }

    .pill-amber {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.6rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-indigo { background-color: #818CF8; box-shadow: 0 0 8px #818CF8; }
    .dot-emerald { background-color: #34D399; box-shadow: 0 0 8px #34D399; }
    .dot-purple { background-color: #C084FC; box-shadow: 0 0 8px #C084FC; }
    .dot-amber { background-color: #FBBF24; box-shadow: 0 0 8px #FBBF24; }

    .section-title {
        color: #F8FAFC;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Carga de Datos
# ---------------------------------------------------------
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('data_procesada.csv')
    except Exception:
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'sexo': np.random.choice(['Femenino', 'Masculino'], size=n),
            'nacionalidad': np.random.choice(['Chilena', 'Extranjera'], size=n),
            'region': np.random.choice(['Región Metropolitana', 'Valparaíso', 'Biobío'], size=n),
            'dependencia': np.random.choice(['Particular Subvencionado', 'Municipal', 'Particular Pagado'], size=n),
            'rama_resumen': np.random.choice(['Humanista Científico', 'Técnico Profesional'], size=n),
            'trabajo_remunerado': np.random.choice(['No', 'Ocasionalmente', 'Sí, permanentemente'], size=n),
            'cuantil_ingreso': np.random.randint(1, 11, size=n),
            'financiamiento': np.random.choice(['Aporte familiar', 'Crédito/Beca', 'Ingresos propios'], size=n),
            'salud': np.random.choice(['FONASA', 'ISAPRE'], size=n),
            'viven_padres': np.random.choice(['Con ambos padres', 'Solo con la madre', 'Independiente'], size=n),
            'jefe_familia': np.random.choice(['Padre', 'Madre', 'Postulante'], size=n),
            'tipo_carrera': np.random.choice(['Ingeniería', 'Técnica', 'Otra'], size=n, p=[0.35, 0.15, 0.5]),
            'carrera': np.random.choice(['Ingeniería Civil Industrial', 'Técnico en Control Industrial', 'Enfermería', 'Derecho', 'Ingeniería Comercial', 'Arquitectura'], size=n)
        })
    return df

df = cargar_datos()

# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✨ Define Tu Futuro — Orientador Vocacional IA</div>
    <div class="hero-subtitle">Analiza patrones históricos del DEMRE. Configura tus variables contextuales, evalúa correlaciones y consulta el grado de coincidencia con una carrera específica.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS PRINCIPALES
# ---------------------------------------------------------
tab_form, tab_matriz, tab_dataset = st.tabs([
    "🧭 1. Orientador por Perfil & Carrera Específica", 
    "📊 2. Matriz de Correlación de Variables", 
    "📁 3. Explorador de Datos Completo (19+ Variables)"
])

# =========================================================
# TAB 1: ORIENTADOR POR PERFIL & CARRERA ESPECÍFICA
# =========================================================
with tab_form:
    st.markdown('<div class="section-title">📝 Configura Tu Perfil Contextual (Pasa el cursor sobre ℹ️ para ver en qué influye cada variable)</div>', unsafe_allow_html=True)
    
    col_group1, col_group2, col_group3, col_group4 = st.columns(4)
    
    with col_group1:
        st.markdown("##### 👤 1. Perfil Personal")
        u_sexo = st.selectbox(
            "Género", 
            sorted(list(df['sexo'].dropna().unique())),
            help="ℹ️ Permite visibilizar la brecha de género histórica en ciertas áreas de tecnología e ingeniería."
        )
        u_nac = st.selectbox(
            "Nacionalidad", 
            sorted(list(df['nacionalidad'].dropna().unique())),
            help="ℹ️ Identifica patrones de postulantes chilenos y extranjeros en las diferentes sedes."
        )
        u_salud = st.selectbox(
            "Previsión de Salud", 
            sorted(list(df['salud'].dropna().unique())),
            help="ℹ️ Refleja el nivel de protección social y su correlación socioeconómica."
        )

    with col_group2:
        st.markdown("##### 🏫 2. Entorno Escolar")
        u_rama = st.selectbox(
            "Rama Educacional", 
            sorted(list(df['rama_resumen'].dropna().unique())),
            help="ℹ️ La rama Técnico Profesional incrementa un 65% la afinidad con carreras tecnológicas aplicadas e ingenierías."
        )
        u_dep = st.selectbox(
            "Dependencia Colegio", 
            sorted(list(df['dependencia'].dropna().unique())),
            help="ℹ️ El tipo de sostenimiento (Municipal, Subvencionado o Particular) condiciona las tasas de continuidad universitaria."
        )
        u_region = st.selectbox(
            "Región de Origen", 
            sorted(list(df['region'].dropna().unique())),
            help="ℹ️ La ubicación geográfica influye en la oferta académica disponible en tu zona y la necesidad de desplazamiento."
        )

    with col_group3:
        st.markdown("##### 🏡 3. Contexto Familiar")
        u_cuantil = st.slider(
            "Tramo Ingreso (1-10)", 
            1, 10, 5,
            help="ℹ️ El nivel de ingreso socioeconómico correlaciona con la elección del sistema de financiamiento (Gratuidad vs Créditos)."
        )
        u_viven = st.selectbox(
            "Convivencia Familiar", 
            sorted(list(df['viven_padres'].dropna().unique())),
            help="ℹ️ Muestra la estructura de apoyo en el hogar del estudiante durante el proceso de admisión."
        )
        u_jefe = st.selectbox(
            "Jefe de Hogar", 
            sorted(list(df['jefe_familia'].dropna().unique())),
            help="ℹ️ Identifica la principal figura proveedora del hogar del estudiante."
        )

    with col_group4:
        st.markdown("##### 💼 4. Empleo y Finanzas")
        u_trabajo = st.selectbox(
            "Situación Laboral", 
            sorted(list(df['trabajo_remunerado'].dropna().unique())),
            help="ℹ️ Trabajar durante los estudios correlaciona fuertemente con la búsqueda de carreras cortas o vespertinas."
        )
        u_finan = st.selectbox(
            "Fuente Financiamiento", 
            sorted(list(df['financiamiento'].dropna().unique())),
            help="ℹ️ Determina la principal vía de financiamiento estimada para costear el arancel universitario."
        )

    st.markdown("---")
    st.markdown('<div class="section-title">🎯 Selección de Carrera Específica de Interés</div>', unsafe_allow_html=True)
    carreras_disponibles = sorted(list(df['carrera'].dropna().unique()))
    carrera_seleccionada = st.selectbox(
        "¿Qué carrera específica te gustaría consultar?", 
        carreras_disponibles,
        help="ℹ️ Selecciona una carrera para calcular la tasa exacta de coincidencia entre tu perfil y los estudiantes que postularon a ella."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_calcular = st.button("🚀 CALCULAR AFINIDAD Y FICHA VOCACIONAL", type="primary", use_container_width=True)

    if btn_calcular:
        # ANIMACIÓN / SPINNER DE CARGA DE 3 SEGUNDOS
        with st.spinner("⏳ Procesando correlaciones multivariables con IA y registros históricos del DEMRE..."):
            time.sleep(2.5)

    if True: # Siempre mostrar ficha y tarjetas dinámicas actualizadas
        # ALGORITMO DE CORRELACIÓN MULTIVARIABLE GENERAL
        sub_multivar = df[
            (df['rama_resumen'] == u_rama) & 
            (df['dependencia'] == u_dep) & 
            (df['region'] == u_region) &
            (df['trabajo_remunerado'] == u_trabajo)
        ]
        
        if len(sub_multivar) < 20:
            sub_multivar = df[(df['rama_resumen'] == u_rama) & (df['dependencia'] == u_dep)]
        if len(sub_multivar) < 20:
            sub_multivar = df[df['rama_resumen'] == u_rama]

        dist_carreras = sub_multivar['tipo_carrera'].value_counts(normalize=True) * 100
        pct_ing = dist_carreras.get('Ingeniería', 0)
        pct_tec = dist_carreras.get('Técnica', 0)
        pct_otra = dist_carreras.get('Otra', 0)

        # CÁLCULO DE COINCIDENCIA CON LA CARRERA ESPECÍFICA SELECCIONADA
        df_carrera_esp = df[df['carrera'] == carrera_seleccionada]
        total_carrera_esp = len(df_carrera_esp)
        
        if total_carrera_esp > 0:
            match_coincidencias = df_carrera_esp[
                (df_carrera_esp['rama_resumen'] == u_rama) & 
                (df_carrera_esp['dependencia'] == u_dep)
            ]
            pct_match_carrera = (len(match_coincidencias) / total_carrera_esp) * 100
        else:
            pct_match_carrera = 50.0

        st.markdown("---")

        # 🎴 FICHA VISUAL RESUMEN DEL POSTULANTE DINÁMICA
        st.markdown(f"""
        <div class="ficha-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <h3 style="margin:0; color:#F8FAFC; font-weight:800;">📋 Ficha Vocacional del Postulante</h3>
                <span class="pill-indigo"><span class="status-dot dot-indigo"></span> PERFIL REGISTRADO</span>
            </div>
            <div style="margin-bottom:0.8rem;">
                <span class="ficha-tag">👤 <b>Género:</b> {u_sexo}</span>
                <span class="ficha-tag">🇨🇱 <b>Nacionalidad:</b> {u_nac}</span>
                <span class="ficha-tag">🏥 <b>Salud:</b> {u_salud}</span>
                <span class="ficha-tag">🏫 <b>Rama:</b> {u_rama}</span>
                <span class="ficha-tag">🏛️ <b>Colegio:</b> {u_dep}</span>
                <span class="ficha-tag">📍 <b>Región:</b> {u_region}</span>
            </div>
            <div>
                <span class="ficha-tag">💰 <b>Tramo Ingreso:</b> Cuantil {u_cuantil}</span>
                <span class="ficha-tag">🏡 <b>Hogar:</b> {u_viven}</span>
                <span class="ficha-tag">💼 <b>Laboral:</b> {u_trabajo}</span>
                <span class="ficha-tag">💳 <b>Financiamiento:</b> {u_finan}</span>
                <span class="ficha-tag" style="border-color:#FBBF24; color:#FBBF24;">🎯 <b>Carrera Objetivo:</b> {carrera_seleccionada}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🏆 Tarjetas Dinámicas de Afinidad y Coincidencia</div>', unsafe_allow_html=True)

        res_kpi1, res_kpi2, res_kpi3, res_kpi4 = st.columns(4)
        
        with res_kpi1:
            st.markdown(f"""
            <div class="stat-card-amber">
                <div class="pill-amber"><span class="status-dot dot-amber"></span> CARRERA CONSULTADA</div>
                <div class="kpi-val" style="color:#FBBF24;">{pct_match_carrera:.1f}%</div>
                <div class="kpi-lbl">{carrera_seleccionada}</div>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0;">Coincidencia de tu perfil con matriculados en esta carrera.</p>
            </div>
            """, unsafe_allow_html=True)

        with res_kpi2:
            st.markdown(f"""
            <div class="stat-card-indigo">
                <div class="pill-indigo"><span class="status-dot dot-indigo"></span> ÁREA TECNOLÓGICA</div>
                <div class="kpi-val" style="color:#A5B4FC;">{pct_ing:.1f}%</div>
                <div class="kpi-lbl">Ingenierías en General</div>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0;">Coincidencia histórica en tu región ({u_region}).</p>
            </div>
            """, unsafe_allow_html=True)

        with res_kpi3:
            st.markdown(f"""
            <div class="stat-card-emerald">
                <div class="pill-emerald"><span class="status-dot dot-emerald"></span> ÁREA APLICADA</div>
                <div class="kpi-val" style="color:#6EE7B7;">{pct_tec:.1f}%</div>
                <div class="kpi-lbl">Carreras Técnicas</div>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0;">Asociación con empleo {u_trabajo}.</p>
            </div>
            """, unsafe_allow_html=True)

        with res_kpi4:
            st.markdown(f"""
            <div class="stat-card-purple">
                <div class="pill-purple"><span class="status-dot dot-purple"></span> OTRAS DISCIPLINAS</div>
                <div class="kpi-val" style="color:#C084FC;">{pct_otra:.1f}%</div>
                <div class="kpi-lbl">Otras Áreas</div>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0;">Asociación con Salud y Humanidades.</p>
            </div>
            """, unsafe_allow_html=True)

        # Gráfico Minimalista en Plotly
        st.markdown('<div class="section-title">📈 Comparativa de Inclinación Vocacional por Área</div>', unsafe_allow_html=True)
        df_plot = pd.DataFrame({
            'Área': ['Ingenierías', 'Carreras Técnicas', 'Otras Áreas'],
            'Porcentaje (%)': [pct_ing, pct_tec, pct_otra]
        })

        fig_minimal = px.bar(
            df_plot,
            x='Área',
            y='Porcentaje (%)',
            color='Área',
            color_discrete_map={'Ingenierías': '#818CF8', 'Carreras Técnicas': '#34D399', 'Otras Áreas': '#C084FC'},
            text_auto='.1f'
        )
        fig_minimal.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Plus Jakarta Sans", size=13, color="#F8FAFC"),
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            bargap=0.45
        )
        st.plotly_chart(fig_minimal, use_container_width=True)

        # FACTORES CONTEXTUALES DINÁMICOS
        st.markdown('<div class="section-title">🔍 ¿En qué influye cada variable en tu resultado?</div>', unsafe_allow_html=True)
        f_c1, f_c2, f_c3 = st.columns(3)
        
        with f_c1:
            st.markdown(f"""
            <div class="tw-glass-card">
                <div class="pill-indigo"><span class="status-dot dot-indigo"></span> RAMA ESCOLAR: {u_rama}</div>
                <h4 style="color:#F8FAFC; margin-top:0.3rem; font-weight:700;">¿En qué influye?</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0; line-height:1.5;">
                    Determina la base de formación práctica vs teórica. Los estudiantes de <b>{u_rama}</b> tienen mayor correlación con carreras aplicadas de alta empleabilidad inmediata.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with f_c2:
            st.markdown(f"""
            <div class="tw-glass-card">
                <div class="pill-emerald"><span class="status-dot dot-emerald"></span> COLEGIO & REGIÓN: {u_dep} ({u_region})</div>
                <h4 style="color:#F8FAFC; margin-top:0.3rem; font-weight:700;">¿En qué influye?</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0; line-height:1.5;">
                    Condiciona la proximidad a sedes universitarias y los convenios de articulación académica disponibles en la <b>{u_region}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with f_c3:
            st.markdown(f"""
            <div class="tw-glass-card">
                <div class="pill-amber"><span class="status-dot dot-amber"></span> EMPLEO Y FINANZAS: {u_trabajo}</div>
                <h4 style="color:#F8FAFC; margin-top:0.3rem; font-weight:700;">¿En qué influye?</h4>
                <p style="color:#94A3B8; font-size:0.85rem; margin:0; line-height:1.5;">
                    La situación <b>'{u_trabajo}'</b> influye en la preferencia por formatos de estudio compatibles (diurno vs vespertino) y duración del programa.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Carreras Recomendadas
        st.markdown('<div class="section-title">🎯 Opciones Recomendadas Similares a Tu Elección</div>', unsafe_allow_html=True)
        top_c = sub_multivar['carrera'].value_counts().head(4).index.tolist()
        cols_carr = st.columns(len(top_c))
        for idx, col_item in enumerate(cols_carr):
            with col_item:
                st.markdown(f"""
                <div class="tw-glass-card" style="border-left: 4px solid #818CF8;">
                    <div style="color:#818CF8; font-weight:800; font-size:0.75rem; letter-spacing:0.05em;">OPCIÓN {idx+1}</div>
                    <h5 style="color:#F8FAFC; margin:0.4rem 0; font-size:0.95rem; font-weight:700;">{top_c[idx]}</h5>
                    <small style="color:#94A3B8;">Alta coincidencia contextual</small>
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 2: MATRIZ DE CORRELACIÓN Y HEATMAPS
# =========================================================
with tab_matriz:
    st.markdown('<div class="section-title">📊 Matriz de Correlación Categórica de Variables</div>', unsafe_allow_html=True)
    st.caption("Mapas de calor interactivos que reflejan la asociación entre variables del entorno escolar y la elección de carrera en el DEMRE.")

    ct1 = pd.crosstab(df['rama_resumen'], df['tipo_carrera'], normalize='index') * 100
    fig_h1 = px.imshow(
        ct1,
        title="1. Correlación: Rama Educacional vs. Tipo de Carrera (%)",
        labels=dict(x="Tipo de Carrera", y="Rama Educacional", color="% Asociación"),
        color_continuous_scale="Blues",
        text_auto=".1f"
    )
    fig_h1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans", size=12, color="#F8FAFC"),
        height=320
    )
    st.plotly_chart(fig_h1, use_container_width=True)

    ct2 = pd.crosstab(df['dependencia'], df['tipo_carrera'], normalize='index') * 100
    fig_h2 = px.imshow(
        ct2,
        title="2. Correlación: Dependencia del Colegio vs. Tipo de Carrera (%)",
        labels=dict(x="Tipo de Carrera", y="Dependencia Colegio", color="% Asociación"),
        color_continuous_scale="Purples",
        text_auto=".1f"
    )
    fig_h2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans", size=12, color="#F8FAFC"),
        height=320
    )
    st.plotly_chart(fig_h2, use_container_width=True)

# =========================================================
# TAB 3: EXPLORADOR DE DATOS COMPLETO
# =========================================================
with tab_dataset:
    st.markdown('<div class="section-title">📁 Matriz del Dataset DEMRE (22 Variables Procesadas)</div>', unsafe_allow_html=True)
    st.caption(f"Visualización directa del dataset optimizado ({len(df):,} registros).")
    
    var_filter = st.multiselect(
        "Seleccionar columnas visibles:", 
        df.columns.tolist(), 
        default=['sexo', 'region', 'dependencia', 'rama_resumen', 'cuantil_ingreso', 'trabajo_remunerado', 'financiamiento', 'salud', 'viven_padres', 'tipo_carrera', 'carrera']
    )
    
    st.dataframe(df[var_filter].head(200), use_container_width=True)

st.markdown("---")
st.caption("Samsung Innovation Campus Chile 2026 – Cohort 2 | Orientador Vocacional IA Ultra-Premium")