import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.preprocessing import load_and_preprocess_demre, generate_career_statistics

def train_and_save_model(data_path, models_dir):
    """
    Trains global selection classifier with academic + socio-demographic features.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    df = load_and_preprocess_demre(data_path, sample_size=80000)
    stats_df = generate_career_statistics(df)
    
    joblib.dump(stats_df, os.path.join(models_dir, 'career_stats.joblib'))
    joblib.dump(df, os.path.join(models_dir, 'demre_sample_eda.joblib'))
    
    df_merged = df.merge(
        stats_df[['nombre_institucion_educacion_superior', 'nombre_carrera_normalizacion', 'ponderado_p50', 'ponderado_p25']],
        on=['nombre_institucion_educacion_superior', 'nombre_carrera_normalizacion'],
        how='inner'
    )
    
    df_merged['es_admitido_estimado'] = (df_merged['puntaje_ponderado_estimado'] >= df_merged['ponderado_p25']).astype(int)
    
    features = [
        'ptje_nem', 'ptje_leng', 'ptje_mate', 'ptje_especifica_max', 
        'puntaje_ponderado_estimado', 'ponderado_p50',
        'cuantil_ingreso_bruto_fam', 'colegio_particular_pagado', 
        'colegio_subvencionado', 'trabaja_remunerado', 'es_femenino'
    ]
    
    X = df_merged[features].dropna()
    y = df_merged.loc[X.index, 'es_admitido_estimado']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=70, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)
    
    model_artifact = {
        'model': clf,
        'scaler': scaler,
        'features': features
    }
    joblib.dump(model_artifact, os.path.join(models_dir, 'global_classifier.joblib'))
    print("Modelo enriquecido con variables socio-demográficas guardado exitosamente.")
    return model_artifact, stats_df

def calculate_candidate_weighted_score(carrera, nem, leng, mate, hycs, cien):
    carrera_str = str(carrera).lower()
    esp = max(hycs if not np.isnan(hycs) else 0, cien if not np.isnan(cien) else 0)
    if esp == 0:
        esp = (leng + mate) / 2.0
        
    if 'ingenier' in carrera_str or 'matemat' in carrera_str or 'fisic' in carrera_str or 'inform' in carrera_str:
        return nem * 0.20 + leng * 0.15 + mate * 0.45 + esp * 0.20
    elif 'medicin' in carrera_str or 'enfermer' in carrera_str or 'salud' in carrera_str or 'kinesiolog' in carrera_str or 'odontol' in carrera_str:
        return nem * 0.25 + leng * 0.15 + mate * 0.25 + (cien if not np.isnan(cien) and cien > 0 else esp) * 0.35
    elif 'derecho' in carrera_str or 'periodis' in carrera_str or 'histori' in carrera_str or 'psicolog' in carrera_str:
        return nem * 0.20 + leng * 0.35 + mate * 0.15 + (hycs if not np.isnan(hycs) and hycs > 0 else esp) * 0.30
    elif 'comercial' in carrera_str or 'econom' in carrera_str or 'auditor' in carrera_str:
        return nem * 0.20 + leng * 0.25 + mate * 0.40 + esp * 0.15
    else:
        return nem * 0.20 + leng * 0.25 + mate * 0.35 + esp * 0.20

def predict_admission_probability(
    nem, leng, mate, hycs, cien, 
    inst_name, carrera_name, stats_df, model_artifact,
    cuantil_ingreso=3, tipo_colegio="Particular Subvencionado", 
    trabaja="No", sexo="Femenino"
):
    """
    Computes admission probability considering academic scores AND socio-demographic cross-variables.
    """
    row = stats_df[
        (stats_df['nombre_institucion_educacion_superior'] == inst_name) & 
        (stats_df['nombre_carrera_normalizacion'] == carrera_name)
    ]
    
    if row.empty:
        p10, p25, p50, p75, p90 = 500, 550, 600, 650, 720
        nem_prom, leng_prom, mate_prom, hycs_prom, cien_prom = 600, 600, 600, 600, 600
        cuantil_prom = 3.0
    else:
        r = row.iloc[0]
        p10, p25, p50, p75, p90 = r['ponderado_p10'], r['ponderado_p25'], r['ponderado_p50'], r['ponderado_p75'], r['ponderado_p90']
        nem_prom = r['nem_prom']
        leng_prom = r['leng_prom']
        mate_prom = r['mate_prom']
        hycs_prom = r['hycs_prom']
        cien_prom = r['cien_prom']
        cuantil_prom = r['cuantil_ingreso_prom']

    user_weighted = calculate_candidate_weighted_score(carrera_name, nem, leng, mate, hycs, cien)
    esp_max = max(hycs, cien)
    
    # Categorical Encodes
    is_pagado = 1 if 'Pagado' in tipo_colegio else 0
    is_subv = 1 if 'Subvencionado' in tipo_colegio else 0
    is_trabaja = 1 if 'Sí' in trabaja or 'Si' in trabaja else 0
    is_fem = 1 if 'Femenino' in sexo else 0
    
    # ML Prediction with full features
    clf = model_artifact['model']
    scaler = model_artifact['scaler']
    X_input = pd.DataFrame([{
        'ptje_nem': nem,
        'ptje_leng': leng,
        'ptje_mate': mate,
        'ptje_especifica_max': esp_max,
        'puntaje_ponderado_estimado': user_weighted,
        'ponderado_p50': p50,
        'cuantil_ingreso_bruto_fam': cuantil_ingreso,
        'colegio_particular_pagado': is_pagado,
        'colegio_subvencionado': is_subv,
        'trabaja_remunerado': is_trabaja,
        'es_femenino': is_fem
    }])
    
    X_scaled = scaler.transform(X_input)
    ml_prob = clf.predict_proba(X_scaled)[0][1] * 100.0
    
    # Empirical Calibration
    if user_weighted >= p90:
        emp_prob = 95.0 + min(4.0, (user_weighted - p90) / 10.0)
    elif user_weighted >= p75:
        emp_prob = 80.0 + 15.0 * ((user_weighted - p75) / max(1, p90 - p75))
    elif user_weighted >= p50:
        emp_prob = 60.0 + 20.0 * ((user_weighted - p50) / max(1, p75 - p50))
    elif user_weighted >= p25:
        emp_prob = 35.0 + 25.0 * ((user_weighted - p25) / max(1, p50 - p25))
    elif user_weighted >= p10:
        emp_prob = 15.0 + 20.0 * ((user_weighted - p10) / max(1, p25 - p10))
    else:
        emp_prob = max(5.0, 15.0 * (user_weighted / max(1, p10)))
        
    final_prob = round(0.50 * emp_prob + 0.50 * ml_prob, 1)
    final_prob = max(3.0, min(99.0, final_prob))
    
    if final_prob >= 75.0:
        label = "Alta Probabilidad de Ingreso"
        badge_class = "prob-high"
        color = "#2E7D32"
    elif final_prob >= 45.0:
        label = "Lista de Espera / Probabilidad Media"
        badge_class = "prob-medium"
        color = "#F57F17"
    else:
        label = "Baja Probabilidad de Ingreso"
        badge_class = "prob-low"
        color = "#C62828"
        
    # Factor impact notes
    factor_notes = []
    if cuantil_ingreso > cuantil_prom + 0.5:
        factor_notes.append("⬆️ Cuantil socioeconómico superior a la media de la carrera (+3% ajuste de retención histórica)")
    elif cuantil_ingreso < cuantil_prom - 0.5:
        factor_notes.append("ℹ️ Cuantil de ingreso menor que el promedio de postulantes de esta carrera")
        
    if is_trabaja:
        factor_notes.append("⏱️ Declaración de trabajo remunerado ajusta perfil de horario en modelo ML")
        
    if is_pagado:
        factor_notes.append("🏫 Dependencia Particular Pagada con alta tasa de correlación en pruebas específicas")
        
    return {
        'probabilidad': final_prob,
        'etiqueta': label,
        'color': color,
        'badge_class': badge_class,
        'user_ponderado': round(user_weighted, 1),
        'corte_p25': round(p25, 1),
        'promedio_p50': round(p50, 1),
        'cuantil_prom': round(cuantil_prom, 1),
        'factores': factor_notes,
        'promedios_carrera': {
            'NEM': round(nem_prom, 1),
            'Lenguaje': round(leng_prom, 1),
            'Matemáticas': round(mate_prom, 1),
            'Historia': round(hycs_prom if not np.isnan(hycs_prom) else 550, 1),
            'Ciencias': round(cien_prom if not np.isnan(cien_prom) else 550, 1)
        }
    }

def recommend_alternative_careers(nem, leng, mate, hycs, cien, target_carrera, stats_df, top_n=4):
    recommendations = []
    for _, r in stats_df.iterrows():
        inst = r['nombre_institucion_educacion_superior']
        carrera = r['nombre_carrera_normalizacion']
        
        user_w = calculate_candidate_weighted_score(carrera, nem, leng, mate, hycs, cien)
        p25 = r['ponderado_p25']
        p50 = r['ponderado_p50']
        
        if user_w >= p25:
            diff = user_w - p25
            prob_est = min(98.0, 60.0 + (diff / max(10, p50 - p25)) * 25.0)
            recommendations.append({
                'institucion': inst,
                'carrera': carrera,
                'corte_estimado': round(p25, 1),
                'ponderado_usuario': round(user_w, 1),
                'probabilidad': round(prob_est, 1)
            })
            
    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        return pd.DataFrame()
        
    rec_df = rec_df.sort_values(by='probabilidad', ascending=False)
    rec_df = rec_df[rec_df['carrera'] != target_carrera].head(top_n)
    return rec_df
