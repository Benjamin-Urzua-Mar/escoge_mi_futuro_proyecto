# 🎓 Define Tu Futuro — Predictor de Selección Universitaria DEMRE

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Licencia](https://img.shields.io/badge/licencia-MIT-green.svg)](LICENSE)

**Proyecto Final del Curso de Código y Programación**  
**Samsung Innovation Campus Chile 2026 – Cohort 2**

---

## 📌 1. Resumen del proyecto

**Define Tu Futuro** es una aplicación web interactiva desarrollada con **Python, Streamlit, Plotly y scikit-learn** que permite explorar datos históricos del proceso de admisión universitaria chileno y generar una estimación orientativa para una carrera e institución determinadas.

La aplicación combina cinco componentes:

1. **Predicción académica de admisión**, basada en puntajes, percentiles históricos y un modelo Random Forest.
2. **Afinidad contextual**, que compara el perfil socioeducativo del usuario con perfiles históricos similares.
3. **Recomendaciones de carreras alternativas**, ordenadas por una combinación de probabilidad estimada y afinidad contextual.
4. **Análisis exploratorio de datos**, con indicadores, correlaciones, gráficos y descarga de vistas filtradas.
5. **Informe de orientación vocacional asistido por IA**, generado con **HuggingFace (Qwen 2.5)** e interpretado dinámicamente en un diseño **Bento Grid nativo**.

La herramienta está diseñada para apoyar la exploración vocacional y facilitar la comprensión de datos complejos. **No reemplaza las ponderaciones, vacantes, requisitos ni resultados oficiales del proceso de admisión.**

---

## 🔍 2. Problema e investigación

### 2.1 Contexto

La elección de una carrera de educación superior involucra intereses personales, rendimiento académico, contexto familiar, ubicación territorial, posibilidades de financiamiento y oferta institucional. Sin embargo, muchos estudiantes realizan esta decisión con información fragmentada o difícil de interpretar.

### 2.2 Problema identificado

Los postulantes necesitan una herramienta que reúna en un solo lugar:

- Comparación de sus puntajes con perfiles históricos.
- Exploración de carreras e instituciones compatibles.
- Información contextual sobre estudiantes con características similares.
- Visualizaciones comprensibles para usuarios sin experiencia en ciencia de datos.

### 2.3 Pregunta de investigación

> **¿Es posible utilizar puntajes académicos y patrones históricos del DEMRE para orientar la exploración de una carrera e institución, complementando el resultado con variables contextuales del postulante?**

### 2.4 Usuarios objetivo

- 🎓 Estudiantes de enseñanza media y egresados que exploran opciones de educación superior.
- 🏫 Orientadores vocacionales, docentes y establecimientos educacionales.
- 👨‍👩‍👧 Familias y apoderados que acompañan el proceso de postulación.
- 📊 Personas interesadas en explorar patrones históricos del sistema de admisión.

---

## ✨ 3. Funcionalidades principales

### 3.1 Predictor de selección

El usuario ingresa:

- Puntaje NEM.
- Competencia Lectora o Lenguaje.
- Matemática M1.
- Ciencias o Historia y Ciencias Sociales.
- Carrera objetivo.
- Institución objetivo.

La interfaz presenta:

- Probabilidad estimada de ingreso.
- Ponderado estimado del postulante.
- Referencia histórica P25.
- Mediana histórica P50.
- Comparación entre puntajes personales y promedios históricos.
- Carreras alternativas con alta compatibilidad académica.

### 3.2 Selección dependiente de carrera e institución

El flujo de selección funciona en este orden:

1. Se muestra el listado completo de carreras.
2. El usuario selecciona una carrera.
3. La aplicación filtra y muestra solamente las instituciones que disponen de esa carrera.

Esto evita combinaciones de carrera e institución inexistentes en las estadísticas del proyecto.

### 3.3 Perfil contextual

La aplicación recupera y utiliza los siguientes inputs:

- Género.
- Nacionalidad.
- Previsión de salud.
- Rama educacional.
- Dependencia del establecimiento.
- Región de origen.
- Cuantil de ingreso.
- Convivencia familiar.
- Jefe de hogar.
- Situación laboral.
- Fuente principal de financiamiento.

Estas variables se utilizan principalmente para calcular **similitud histórica contextual**. No sustituyen los puntajes académicos ni garantizan un resultado de admisión.

### 3.4 Tarjetas dinámicas de afinidad

La pestaña de perfil presenta cuatro tarjetas:

- 🟡 **Carrera consultada:** afinidad contextual con los registros históricos disponibles de la carrera e institución.
- 🔵 **Área tecnológica:** porcentaje de perfiles cercanos vinculados con ingenierías.
- 🟢 **Área aplicada:** porcentaje de perfiles cercanos vinculados con carreras técnicas.
- 🟣 **Otras disciplinas:** porcentaje correspondiente a salud, humanidades, artes, pedagogías y otras áreas.

### 3.5 Cálculo controlado mediante botón

Los resultados se calculan únicamente al presionar:

> **🚀 Calcular admisión y afinidad**

Mover un input no vuelve a ejecutar el predictor automáticamente. Los últimos resultados permanecen guardados en `st.session_state` y la aplicación informa cuando existen cambios pendientes por recalcular.

### 3.6 Análisis exploratorio

La aplicación incluye:

- Total de registros de la muestra.
- Número de carreras e instituciones.
- Puntajes promedio.
- Carreras e instituciones con más postulaciones.
- Correlaciones entre variables académicas y socioeconómicas.
- Puntaje promedio según previsión de salud.
- Comparación por dependencia escolar.
- Distribución de fuentes de financiamiento.
- Asociación entre rama educacional y carrera.
- Explorador de datos con selección de columnas y descarga en CSV.

### 3.7 Informe de orientación vocacional asistido por IA (Bento Grid)

Al presionar el botón **🚀 Calcular admisión y afinidad**, el sistema ejecuta una llamada a la **Inference API de HuggingFace** (modelo `Qwen/Qwen2.5-7B-Instruct`) utilizando un prompt enriquecido que sintetiza:

- Perfil socioeducativo completo del postulante.
- Puntajes ingresados vs promedios históricos de la carrera.
- Probabilidad estimada de ingreso y percentiles P25 (corte) y P50 (mediana).
- Factores sociodemográficos influyentes detectados por el modelo.
- Afinidad contextual y distribución de áreas (Ingeniería, Técnica, Otras).
- Recomendaciones de carreras alternativas con sus datos de corte y afinidad.

El informe se presenta automáticamente al final de la pestaña **🧭 Perfil y Afinidad Contextual** estructurado en un **diseño Bento Grid nativo en Streamlit** (`st.columns` + `st.container(border=True)`). Además, cuenta con un botón **🔄 Regenerar informe IA** para re-obtener la interpretación del modelo en cualquier momento.

---

## 🧠 4. Arquitectura de la solución

```text
Entrada del usuario
        │
        ├── Puntajes PAES/PSU
        ├── Carrera e institución
        └── Perfil contextual
        │
        ▼
Aplicación Streamlit
        │
        ├── Estadísticas por carrera e institución
        ├── Modelo Random Forest
        ├── Calibración empírica por percentiles
        ├── Motor de similitud contextual
        ├── Motor de recomendaciones
        └── Módulo HuggingFace (Inference API Qwen 2.5)
        │
        ▼
Resultados y visualizaciones
        ├── Probabilidad orientativa
        ├── Ponderado y percentiles
        ├── Tarjetas de afinidad
        ├── Informe de IA en Bento Grid nativo
        ├── Carreras alternativas
        └── Análisis exploratorio
```

---

## ⚙️ 5. Preparación y procesamiento de datos

El archivo `src/preprocessing.py` realiza las principales tareas de preparación.

### 5.1 Lectura y muestra

- El dataset DEMRE se carga desde un archivo CSV separado por `;` y codificado en `latin-1`.
- Cuando el dataset supera el límite configurado, se selecciona una muestra máxima de **80.000 registros**.
- La muestra usa `random_state=42`, lo que favorece la reproducibilidad.

### 5.2 Limpieza de texto

Se corrigen nombres con problemas frecuentes de codificación, por ejemplo:

- `Ingeniera` → `Ingeniería`
- `Psicologa` → `Psicología`
- `Educacin` → `Educación`
- `Diseo` → `Diseño`

### 5.3 Tratamiento de valores faltantes

Los puntajes faltantes de:

- NEM.
- Lenguaje.
- Matemática.
- Historia.
- Ciencias.

se reemplazan por **550 puntos**.

El cuantil de ingreso faltante se reemplaza por `3`.

> Esta imputación permite conservar registros, pero puede introducir distorsiones. Debe considerarse al interpretar los resultados.

### 5.4 Variables derivadas

El preprocesamiento crea:

- `colegio_particular_pagado`
- `colegio_subvencionado`
- `trabaja_remunerado`
- `es_femenino`
- `ptje_especifica_max`
- `puntaje_ponderado_estimado`

---

## 🧮 6. Cálculo del ponderado estimado

La aplicación no utiliza las ponderaciones oficiales particulares de cada programa. Emplea reglas generales según palabras encontradas en el nombre de la carrera.

Sea:

- `NEM`: puntaje NEM.
- `LEN`: Lenguaje o Competencia Lectora.
- `MAT`: Matemática.
- `HIS`: Historia y Ciencias Sociales.
- `CIE`: Ciencias.
- `ESP`: el mayor valor entre Historia y Ciencias.

Cuando ambas pruebas específicas son cero, `ESP` se reemplaza por el promedio entre Lenguaje y Matemática.

### 6.1 Ingenierías, informática, matemática o física

```text
Ponderado = NEM × 0,20 + LEN × 0,15 + MAT × 0,45 + ESP × 0,20
```

### 6.2 Carreras de salud

Para nombres que contienen medicina, enfermería, salud, kinesiología u odontología:

```text
Ponderado = NEM × 0,25 + LEN × 0,15 + MAT × 0,25 + CIE × 0,35
```

### 6.3 Derecho, periodismo, historia o psicología

```text
Ponderado = NEM × 0,20 + LEN × 0,35 + MAT × 0,15 + HIS × 0,30
```

### 6.4 Ingeniería Comercial, economía o auditoría

```text
Ponderado = NEM × 0,20 + LEN × 0,25 + MAT × 0,40 + ESP × 0,15
```

### 6.5 Resto de las carreras

```text
Ponderado = NEM × 0,20 + LEN × 0,25 + MAT × 0,35 + ESP × 0,20
```

---

## 📊 7. Estadísticas históricas por carrera e institución

Los registros se agrupan por:

```text
Institución + Carrera
```

Para cada combinación se calculan:

- Total de postulaciones.
- Promedios de NEM, Lenguaje, Matemática, Historia y Ciencias.
- Promedio del ponderado estimado.
- Promedio del cuantil de ingreso.
- Porcentaje de estudiantes de colegios particulares pagados.
- Percentiles P10, P25, P50, P75 y P90.

Solo se conservan combinaciones con al menos **3 registros**.

### Interpretación de percentiles

- **P10:** 10% de los ponderados se encuentra aproximadamente bajo este valor.
- **P25:** 25% se encuentra aproximadamente bajo este valor.
- **P50:** mediana de la distribución.
- **P75:** 75% se encuentra aproximadamente bajo este valor.
- **P90:** 90% se encuentra aproximadamente bajo este valor.

En la interfaz, P25 se muestra como referencia de corte. **No corresponde necesariamente al último puntaje oficial de selección informado por una institución.**

---

## 🤖 8. Entrenamiento del modelo

### 8.1 Etiqueta estimada

El dataset no utiliza una variable oficial de admisión observada. La etiqueta de entrenamiento se construye mediante:

```text
Admitido estimado = 1, cuando ponderado estimado ≥ P25
Admitido estimado = 0, cuando ponderado estimado < P25
```

Por lo tanto, el modelo aprende a aproximar una regla creada dentro del proyecto y no una admisión oficial confirmada.

### 8.2 Algoritmo

Se utiliza un:

```text
RandomForestClassifier
```

Configuración:

```text
n_estimators = 70
max_depth = 10
random_state = 42
n_jobs = -1
```

### 8.3 Variables de entrada

Antes del entrenamiento se aplica `StandardScaler`. El modelo recibe:

1. Puntaje NEM.
2. Puntaje de Lenguaje.
3. Puntaje de Matemática.
4. Mejor prueba específica.
5. Ponderado estimado del postulante.
6. P50 de la carrera e institución.
7. Cuantil de ingreso.
8. Indicador de colegio particular pagado.
9. Indicador de colegio subvencionado.
10. Indicador de trabajo remunerado.
11. Indicador de sexo femenino.

Los artefactos se guardan en:

```text
models/global_classifier.joblib
models/career_stats.joblib
models/demre_sample_eda.joblib
```

---

## 📈 9. Cálculo de la probabilidad de admisión

La probabilidad final combina una estimación del Random Forest y una calibración empírica basada en percentiles.

### 9.1 Probabilidad del modelo

```python
ml_prob = model.predict_proba(datos_usuario)[0][1] * 100
```

Representa la probabilidad asignada por el modelo a la etiqueta interna `admitido estimado`.

### 9.2 Probabilidad empírica

La posición del ponderado del usuario dentro de los percentiles genera una segunda estimación.

| Tramo | Rango empírico aproximado |
|---|---:|
| Sobre P90 | 95% a 99% |
| P75 a P90 | 80% a 95% |
| P50 a P75 | 60% a 80% |
| P25 a P50 | 35% a 60% |
| P10 a P25 | 15% a 35% |
| Bajo P10 | 5% a 15% aproximadamente |

Dentro de cada tramo se realiza una interpolación lineal.

### 9.3 Probabilidad final

```text
Probabilidad final = probabilidad empírica × 0,50
                   + probabilidad Random Forest × 0,50
```

El resultado se limita entre `3%` y `99%`.

### 9.4 Etiquetas de resultado

| Probabilidad | Etiqueta |
|---:|---|
| 75% o más | Alta Probabilidad de Ingreso |
| 45% a 74,9% | Lista de Espera / Probabilidad Media |
| Menos de 45% | Baja Probabilidad de Ingreso |

### 9.5 Valores de respaldo

Si no se encuentra una combinación en las estadísticas, el modelo utiliza:

```text
P10 = 500
P25 = 550
P50 = 600
P75 = 650
P90 = 720
Promedios de pruebas = 600
Cuantil promedio = 3
```

El selector dependiente de carrera e institución reduce la posibilidad de llegar a este caso.

---

## 🧭 10. Afinidad contextual

La afinidad contextual es independiente de la probabilidad académica principal. Su objetivo es medir cuánto se parece el perfil ingresado a perfiles históricos del dataset EDA.

### 10.1 Variables categóricas

Para género, nacionalidad, salud, rama educacional, dependencia escolar, región, convivencia, jefe de hogar, trabajo y financiamiento:

```text
Coincide = 1
No coincide = 0
```

Las comparaciones normalizan mayúsculas, minúsculas, tildes y signos para reducir falsos desacuerdos.

### 10.2 Cuantil de ingreso

El cuantil utiliza una similitud gradual:

```text
Similitud = 1 − |cuantil histórico − cuantil usuario| / 9
```

El resultado se restringe al rango de 0 a 1.

### 10.3 Similitud por registro

```text
Similitud del registro = promedio de todos los componentes disponibles
```

Todas las variables disponibles tienen el mismo peso.

### 10.4 Afinidad con la carrera consultada

La aplicación intenta utilizar, en este orden:

1. Registros de la carrera e institución seleccionadas.
2. Todos los registros disponibles de la carrera.
3. Todos los registros disponibles de la institución.

La tarjeta ámbar muestra el promedio de similitud de esos registros.

### 10.5 Coincidencias exactas

Una fila se considera coincidencia exacta cuando su similitud es prácticamente del 100%.

### 10.6 Perfiles más cercanos

Se selecciona:

```text
Máximo entre 50 registros y el 5% del dataset
```

Estos perfiles se utilizan para calcular:

- Ponderado promedio de perfiles cercanos.
- Distribución de áreas académicas.
- Apoyo al ordenamiento de carreras alternativas.

---

## 🏆 11. Tarjetas dinámicas

### 11.1 Carrera consultada

Muestra la afinidad contextual promedio con la carrera e institución seleccionadas, o con el nivel de respaldo disponible.

### 11.2 Área tecnológica

Se clasifican como ingeniería las carreras cuyo nombre contiene expresiones como:

```text
ingenier, matemat, fisic, inform
```

### 11.3 Área aplicada

Se clasifican como técnicas las carreras cuyo nombre contiene expresiones como:

```text
tecnic, tecnolog, tecnico nivel superior
```

### 11.4 Otras disciplinas

Incluye todos los programas que no quedaron clasificados como ingeniería o carrera técnica.

Los tres porcentajes se calculan sobre los perfiles históricos más cercanos y deberían sumar aproximadamente 100%, considerando redondeos.

---

## 💡 12. Carreras alternativas

Para cada carrera e institución disponible se calcula nuevamente el ponderado del usuario usando la regla asociada al nombre de esa carrera.

Una alternativa se considera cuando:

```text
Ponderado del usuario ≥ P25 de la alternativa
```

La estimación académica de la recomendación es:

```text
Probabilidad = 60 + ((ponderado usuario − P25) / max(10, P50 − P25)) × 25
```

El resultado tiene un máximo de `98%`.

Después se calcula afinidad contextual para cada alternativa y se utiliza el siguiente índice únicamente para ordenar:

```text
Índice combinado = probabilidad académica × 0,75
                 + afinidad contextual × 0,25
```

La probabilidad mostrada permanece siendo la estimación académica original; el índice combinado no sustituye ese valor.

---

## 📉 13. Análisis estadístico y visualizaciones

### 13.1 Matriz de correlación

Se utiliza la correlación de Pearson entre variables numéricas.

```text
 1  → relación lineal positiva fuerte
 0  → ausencia de relación lineal clara
-1  → relación lineal negativa fuerte
```

Una correlación no demuestra causalidad.

### 13.2 Agrupaciones descriptivas

Los gráficos de previsión de salud, dependencia escolar y financiamiento utilizan:

- Promedios.
- Conteos.
- Distribuciones porcentuales.

Son análisis descriptivos, no predicciones individuales.

### 13.3 Asociación rama-carrera

La aplicación construye una tabla cruzada normalizada por rama educacional para comparar la distribución de las carreras más frecuentes.

---

## 🖥️ 14. Interfaz de la aplicación

La aplicación está organizada en cinco pestañas:

1. **🎯 Predictor de Selección**  
   Probabilidad, ponderado, percentiles, comparación de puntajes y recomendaciones.

2. **🧭 Perfil y Afinidad Contextual**  
   Ficha del postulante, tarjetas dinámicas, coincidencias y análisis contextual.

3. **📊 Análisis Exploratorio (EDA)**  
   Indicadores generales, carreras e instituciones más frecuentes.

4. **💰 Correlaciones Socioeconómicas**  
   Correlaciones, previsión, dependencia escolar, financiamiento y rama educacional.

5. **📁 Explorador de Datos**  
   Selección de columnas, límite de filas y descarga de la vista en CSV.

La interfaz utiliza un tema oscuro, tarjetas claras de alto contraste y componentes personalizados mediante CSS.

---

## 🛠️ 15. Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Aplicación web | Streamlit |
| Procesamiento de datos | pandas, NumPy |
| Machine Learning | scikit-learn |
| Persistencia de modelos | joblib |
| Modelo de Lenguaje (IA) | HuggingFace Inference API (`Qwen/Qwen2.5-7B-Instruct`) |
| Visualizaciones | Plotly |
| Gráficos auxiliares | Matplotlib, Seaborn |
| Despliegue sugerido | Streamlit Community Cloud |

---

## 📂 16. Estructura del repositorio

```text
escoge_mi_futuro_proyecto-main/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── DEMRE_1.csv                      # No incluido necesariamente en el repositorio
│
├── hugging_face/
│   ├── __init__.py
│   ├── huggingface.py               # Cliente InferenceClient y llamada al LLM Qwen 2.5
│   └── prompts.py                   # Constructor del prompt estructurado enriquecido
│
├── src/
│   ├── model.py
│   └── preprocessing.py
│
├── models/
│   ├── career_stats.joblib
│   ├── demre_sample_eda.joblib
│   └── global_classifier.joblib
│
└── .streamlit/
    └── config.toml                  # Opcional: configuración visual
```

### Descripción de archivos

- `app.py`: interfaz principal, renderizado Bento Grid y lógica de interacción en Streamlit.
- `hugging_face/huggingface.py`: cliente de IA para interacción con la API de HuggingFace.
- `hugging_face/prompts.py`: función `construir_prompt` que sintetiza puntajes, predicción y afinidad en un prompt vocacional.
- `train_model.py`: ejecución del entrenamiento y generación de artefactos.
- `src/preprocessing.py`: carga, limpieza, transformación y estadísticas históricas.
- `src/model.py`: ponderaciones, entrenamiento, predicción y recomendaciones.
- `models/career_stats.joblib`: estadísticas por carrera e institución.
- `models/demre_sample_eda.joblib`: muestra procesada para análisis exploratorio.
- `models/global_classifier.joblib`: modelo Random Forest, escalador y lista de variables.

---

## 🚀 17. Instalación y ejecución

### 17.1 Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd escoge_mi_futuro_proyecto-main
```

### 17.2 Crear un entorno virtual

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux o macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 17.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 17.4 Ejecutar la aplicación

```bash
streamlit run app.py
```

Cuando se utiliza el archivo final trabajado con las tarjetas dinámicas:

```bash
streamlit run app_final_tarjetas.py
```

---

## 🏋️ 18. Reentrenamiento del modelo

Para generar nuevamente los artefactos:

1. Ubicar `DEMRE_1.csv` en la raíz del proyecto.
2. Verificar que tenga las columnas esperadas por `src/preprocessing.py`.
3. Ejecutar:

```bash
python train_model.py
```

El proceso generará o reemplazará los archivos de la carpeta `models/`.

La aplicación también puede intentar entrenar automáticamente cuando no encuentra los artefactos y sí encuentra el CSV requerido.

---

## ☁️ 19. Despliegue en Streamlit Cloud

1. Subir el proyecto a GitHub.
2. Ingresar a Streamlit Community Cloud.
3. Seleccionar el repositorio y la rama.
4. Definir `app.py` como archivo principal.
5. Confirmar que `requirements.txt` y los archivos `.joblib` estén disponibles.

🚀 **Aplicación publicada:** `REEMPLAZAR_CON_URL_FINAL`

> Si los artefactos exceden los límites del repositorio, deben almacenarse mediante una estrategia compatible con el entorno de despliegue.

---

## ⚠️ 20. Limitaciones y uso responsable

1. **No es un sistema oficial de admisión.**  
   La aplicación no conoce vacantes, listas de espera, ponderaciones oficiales actualizadas ni criterios especiales de cada programa.

2. **La etiqueta de entrenamiento es estimada.**  
   Se define a partir del P25 calculado por el proyecto y no de una columna oficial de estudiantes seleccionados.

3. **P25 no equivale necesariamente al último puntaje seleccionado.**

4. **Las ponderaciones son heurísticas.**  
   Se asignan por palabras presentes en el nombre de la carrera.

5. **Los valores faltantes se imputan.**  
   Los puntajes faltantes se reemplazan por 550 y el cuantil faltante por 3.

6. **El entrenamiento utiliza una muestra máxima de 80.000 filas.**

7. **Las variables contextuales no deben interpretarse como causas.**  
   Su función principal en la interfaz final es medir similitud histórica y complementar recomendaciones.

8. **La versión actual de la interfaz mantiene separadas la predicción académica y la afinidad contextual.**  
   Aunque la función del modelo admite algunos datos sociodemográficos, la llamada principal del frontend puede usar sus valores predeterminados si no se pasan explícitamente.

9. **No existen garantías individuales.**  
   El resultado debe leerse como orientación exploratoria.

Texto recomendado para la interfaz:

> **Estimación orientativa basada en ponderaciones simuladas, percentiles históricos, un modelo de clasificación y similitud contextual. No representa una garantía de selección ni reemplaza la información oficial de admisión.**

---

## ✅ 21. Mejoras implementadas durante el desarrollo

- Integración de todos los inputs contextuales del prototipo original.
- Corrección de contraste para evitar texto blanco sobre fondos claros.
- Selector de carrera ubicado antes del selector de institución.
- Filtrado automático de instituciones según la carrera elegida.
- Corrección del orden de los argumentos enviados al predictor.
- Normalización de textos para comparar nombres con tildes y formatos diferentes.
- Estrategia de respaldo para evitar resultados constantes de “Sin datos”.
- Restauración de las cuatro tarjetas dinámicas de afinidad.
- Uso de `st.session_state` para calcular solo al presionar el botón.
- Incorporación de cinco pestañas de análisis.
- Explorador de datos con descarga en CSV.
- Creación del paquete `hugging_face/` con integración de la Inference API de HuggingFace (`Qwen/Qwen2.5-7B-Instruct`).
- Refactorización del método `construir_prompt` para alimentar el LLM con la totalidad de resultados académicos, percentiles, afinidad contextual y carreras alternativas.
- Generación automática del informe al presionar **🚀 Calcular admisión y afinidad** con opción de regeneración manual (`🔄 Regenerar informe IA`).
- Presentación del informe de orientación vocacional mediante un diseño **Bento Grid nativo en Streamlit** (`st.columns` + `st.container(border=True)`).

---

## 🧪 22. Próximas mejoras sugeridas

- Incorporar ponderaciones oficiales por carrera, institución y proceso de admisión.
- Entrenar con una etiqueta real de selección o matrícula confirmada.
- Separar entrenamiento, validación y prueba.
- Guardar métricas como accuracy, precision, recall, F1, ROC-AUC y calibración.
- Agregar intervalos de confianza según el número de registros disponibles.
- Corregir la métrica de coincidencias exactas para restringirla a la carrera e institución consultadas.
- Sustituir la clasificación textual de áreas por una taxonomía académica oficial.
- Enviar explícitamente al predictor los inputs de cuantil, dependencia, trabajo y género cuando se decida que formen parte de la probabilidad principal.
- Incorporar M2 y ponderaciones específicas cuando sean necesarias.
- Añadir pruebas unitarias para preprocesamiento, ponderaciones y filtros dependientes.
- Registrar la versión del dataset y la fecha del proceso de admisión analizado.

---

## 👥 23. Integrantes del equipo y roles

- **Integrante 1:** Sebastián Morales.
- **Integrante 2:** Fabián Valdés.
- **Integrante 3:** Camila Bravo.
- **Integrante 4:** Benjamín Urzúa.

---

## 🏁 24. Conclusiones

El proyecto demuestra que es posible integrar procesamiento de datos, modelos de clasificación, percentiles históricos, similitud contextual y visualizaciones interactivas en una herramienta comprensible para estudiantes y orientadores.

Sus principales aportes son:

- Centralizar información académica y contextual en una sola aplicación.
- Facilitar la comparación entre puntajes personales y registros históricos.
- Mostrar carreras alternativas bajo criterios transparentes.
- Permitir la exploración visual de variables socioeducativas.
- Comunicar claramente las limitaciones de una estimación construida con datos históricos.

El valor principal de **Define Tu Futuro** no es reemplazar el proceso oficial, sino ofrecer una base tecnológica para una orientación vocacional más informada, trazable y accesible.

---

## 📄 25. Licencia

Este proyecto se distribuye bajo licencia MIT. Consultar el archivo `LICENSE` para más información.
