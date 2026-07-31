import pandas as pd
import numpy as np
import os

def clean_text(text):
    if not isinstance(text, str):
        return text
    replacements = {
        'Ingeniera': 'Ingeniería', 'Enfermera': 'Enfermería', 'Psicologa': 'Psicología',
        'Tecnologa': 'Tecnología', 'Mdica': 'Médica', 'Pedagoga': 'Pedagogía',
        'Educacin': 'Educación', 'Comunicacin': 'Comunicación', 'Biologa': 'Biología',
        'Qumica': 'Química', 'Geografa': 'Geografía', 'Odontologa': 'Odontología',
        'Kinesiologa': 'Kinesiología', 'Sociologa': 'Sociología', 'Filosofa': 'Filosofía',
        'Comn': 'Común', 'Fsica': 'Física', 'Agronoma': 'Agronomía',
        'Construccin': 'Construcción', 'Auditora': 'Auditoría', 'Diseo': 'Diseño'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

def load_and_preprocess_demre(filepath, sample_size=80000):
    """
    Loads DEMRE CSV dataset and preprocesses scores, text encodings, and full socio-demographic features.
    """
    print(f"Cargando dataset desde {filepath}...")
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).copy()
    
    text_cols = [
        'nombre_carrera_normalizacion', 'nombre_institucion_educacion_superior', 
        'nombre_sede', 'nombre_region', 'nombre_dependencia_establecimiento',
        'nombre_cobertura_salud', 'descripcion_fuente_financiamiento_estudio_superior_primaria',
        'descripcion_jefe_familia', 'descripcion_trabajo_remunerado', 'nombre_sexo'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
            
    score_cols = ['ptje_nem', 'ptje_leng', 'ptje_mate', 'ptje_hycs', 'ptje_cien']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(550.0)
        
    df['cuantil_ingreso_bruto_fam'] = pd.to_numeric(df['cuantil_ingreso_bruto_fam'], errors='coerce').fillna(3.0)
    
    # Binary/Categorical Encoding for ML Features
    df['colegio_particular_pagado'] = (df['nombre_dependencia_establecimiento'].str.contains('Pagado', case=False, na=False)).astype(int)
    df['colegio_subvencionado'] = (df['nombre_dependencia_establecimiento'].str.contains('Subvencionado', case=False, na=False)).astype(int)
    df['trabaja_remunerado'] = (df['descripcion_trabajo_remunerado'].str.contains('Sí|Si', case=False, na=False)).astype(int)
    df['es_femenino'] = (df['nombre_sexo'].str.contains('Femenino', case=False, na=False)).astype(int)
    
    # Max specific test score
    df['ptje_especifica_max'] = df[['ptje_hycs', 'ptje_cien']].max(axis=1)
    
    nem = df['ptje_nem']
    leng = df['ptje_leng']
    mate = df['ptje_mate']
    esp = df['ptje_especifica_max']
    carreras = df['nombre_carrera_normalizacion'].str.lower()

    is_ing = carreras.str.contains('ingenier|matemat|fisic|inform', regex=True, na=False)
    is_salud = carreras.str.contains('medicin|enfermer|salud|kinesiolog|odontol', regex=True, na=False)
    is_hum = carreras.str.contains('derecho|periodis|histori|psicolog', regex=True, na=False)
    is_com = carreras.str.contains('comercial|econom|auditor', regex=True, na=False)

    df['puntaje_ponderado_estimado'] = (
        np.where(is_ing, nem * 0.20 + leng * 0.15 + mate * 0.45 + esp * 0.20,
        np.where(is_salud, nem * 0.25 + leng * 0.15 + mate * 0.25 + df['ptje_cien'] * 0.35,
        np.where(is_hum, nem * 0.20 + leng * 0.35 + mate * 0.15 + df['ptje_hycs'] * 0.30,
        np.where(is_com, nem * 0.20 + leng * 0.25 + mate * 0.40 + esp * 0.15,
        nem * 0.20 + leng * 0.25 + mate * 0.35 + esp * 0.20))))
    )
    
    return df

def generate_career_statistics(df):
    grp = df.groupby(['nombre_institucion_educacion_superior', 'nombre_carrera_normalizacion']).agg(
        total_postulaciones=('orden_preferencia', 'count'),
        nem_prom=('ptje_nem', 'mean'),
        leng_prom=('ptje_leng', 'mean'),
        mate_prom=('ptje_mate', 'mean'),
        hycs_prom=('ptje_hycs', 'mean'),
        cien_prom=('ptje_cien', 'mean'),
        esp_max_prom=('ptje_especifica_max', 'mean'),
        ponderado_prom=('puntaje_ponderado_estimado', 'mean'),
        cuantil_ingreso_prom=('cuantil_ingreso_bruto_fam', 'mean'),
        pct_particular=('colegio_particular_pagado', 'mean'),
        ponderado_p10=('puntaje_ponderado_estimado', lambda x: float(np.percentile(x, 10))),
        ponderado_p25=('puntaje_ponderado_estimado', lambda x: float(np.percentile(x, 25))),
        ponderado_p50=('puntaje_ponderado_estimado', lambda x: float(np.percentile(x, 50))),
        ponderado_p75=('puntaje_ponderado_estimado', lambda x: float(np.percentile(x, 75))),
        ponderado_p90=('puntaje_ponderado_estimado', lambda x: float(np.percentile(x, 90)))
    ).reset_index()
    
    grp = grp[grp['total_postulaciones'] >= 3].copy()
    return grp
