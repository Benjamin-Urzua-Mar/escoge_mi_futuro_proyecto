import pandas as pd
import numpy as np

print("Cargando dataset DEMRE_1.csv con TODAS las variables sociodemográficas y territoriales...")
df = pd.read_csv('DEMRE_1.csv', sep=';', encoding='latin-1')

# Categorizar tipo de carrera
def categorizar_carrera(nombre):
    if pd.isna(nombre):
        return 'Otra'
    nombre_upper = str(nombre).upper()
    if any(k in nombre_upper for k in ['TÉCNIC', 'TECNIC', 'TECNOLOGÍA EN', 'TECNOLOGIA EN']):
        return 'Técnica'
    elif 'INGENIER' in nombre_upper or 'ING.' in nombre_upper:
        return 'Ingeniería'
    else:
        return 'Otra'

df['tipo_carrera'] = df['nombre_carrera_normalizacion'].apply(categorizar_carrera)

# Selección exhaustiva de variables
cols_map = {
    'nombre_sexo': 'sexo',
    'nombre_nacionalidad': 'nacionalidad',
    'nombre_region': 'region',
    'nombre_provincia': 'provincia',
    'nombre_comuna': 'comuna',
    'nombre_dependencia_establecimiento': 'dependencia',
    'descripcion_rama_educacional': 'rama_educacional',
    'descripcion_situacion_egreso_postulante': 'egreso',
    'descripcion_trabajo_remunerado': 'trabajo_remunerado',
    'descripcion_proseguir_estudios': 'proseguir_estudios',
    'descripcion_jefe_familia': 'jefe_familia',
    'descripcion_fuente_financiamiento_estudio_superior_primaria': 'financiamiento',
    'cuantil_ingreso_bruto_fam': 'cuantil_ingreso',
    'nombre_cobertura_salud': 'salud',
    'descripcion_viven_padres': 'viven_padres',
    'orden_preferencia': 'orden_preferencia',
    'tipo_carrera': 'tipo_carrera',
    'nombre_carrera_normalizacion': 'carrera',
    'nombre_institucion_educacion_superior': 'institucion',
    'ptje_nem': 'ptje_nem',
    'ptje_mate': 'ptje_mate'
}

existing_cols = {k: v for k, v in cols_map.items() if k in df.columns}
df_clean = df[list(existing_cols.keys())].rename(columns=existing_cols)

# Limpieza y normalización de textos
df_clean['rama_resumen'] = df_clean['rama_educacional'].fillna('Humanista científico').apply(
    lambda x: 'Técnico Profesional' if 'TÉCNICO' in str(x).upper() or 'TECNICO' in str(x).upper() else 'Humanista Científico'
)

df_clean['sexo'] = df_clean['sexo'].fillna('No informa')
df_clean['nacionalidad'] = df_clean['nacionalidad'].fillna('Chilena')
df_clean['dependencia'] = df_clean['dependencia'].fillna('Particular Subvencionado')
df_clean['trabajo_remunerado'] = df_clean['trabajo_remunerado'].fillna('No')
df_clean['cuantil_ingreso'] = pd.to_numeric(df_clean['cuantil_ingreso'], errors='coerce').fillna(5).astype(int)
df_clean['financiamiento'] = df_clean['financiamiento'].fillna('Aporte familiar')
df_clean['salud'] = df_clean['salud'].fillna('FONASA')
df_clean['viven_padres'] = df_clean['viven_padres'].fillna('Con ambos padres')
df_clean['jefe_familia'] = df_clean['jefe_familia'].fillna('Padre')

# Muestreo optimizado
if len(df_clean) > 50000:
    df_clean = df_clean.sample(n=50000, random_state=42).reset_index(drop=True)

output_path = 'data_procesada.csv'
df_clean.to_csv(output_path, index=False, encoding='utf-8')
print(f"data_procesada.csv guardada con éxito con {len(df_clean.columns)} variables y {len(df_clean)} filas.")
