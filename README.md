# 🎓 Define Tu Futuro — Orientador Vocacional IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Licencia](https://img.shields.io/badge/licencia-MIT-green.svg)](LICENSE)

**Proyecto Final del Curso de Código y Programación**  
**Samsung Innovation Campus Chile 2026 – Cohort 2**

---

## 📌 1. Resumen del Proyecto e Introducción
La elección de una carrera de educación superior es una de las decisiones más determinantes en la vida de los jóvenes. Sin embargo, en Chile muchos estudiantes de enseñanza media enfrentan este proceso con información limitada, lo que contribuye a altas tasas de deserción durante el primer año. 

**Define Tu Futuro** es una aplicación web interactiva desarrollada con **Streamlit** e **Inteligencia Artificial (IA)** que analiza datos reales de postulantes universitarios (DEMRE) para identificar patrones clave entre quienes optan por **Carreras Técnicas** o **Ingenierías**, entregando orientación vocacional clara y no técnica.

---

## 🔍 2. Investigación

### 2.1 Contexto
Actualmente existe una amplia oferta académica en educación superior. El Ministerio de Educación destaca que la elección involucra vocación, competencias académicas y proyección laboral.

### 2.2 Problema Identificado
Muchos estudiantes eligen una carrera sin experiencias previas ni información clara sobre el perfil académico o territorial de los matriculados, lo que genera insatisfacción, cambio de carrera o abandono prematuro de los estudios.

### 2.3 Usuarios Objetivo
* 🎓 Estudiantes de enseñanza media en proceso de exploración vocacional.
* 🏫 Orientadores vocacionales y docentes de colegios.
* 👨‍👩‍👧 Apoderados y postulantes a la educación superior en Chile.

---

## 📊 3. Datos y Pregunta de Investigación

### 3.1 Fuente de Datos
Registros oficiales del proceso de admisión a la educación superior en Chile del **DEMRE** (*Departamento de Evaluación, Medición y Registro Educacional*).

### 3.2 Pregunta Principal de Investigación
> **"¿Es posible identificar patrones en estudiantes universitarios que permitan orientar la elección entre una carrera técnica o una ingeniería?"**

### 3.3 Hipótesis Preliminares
1. La elección entre una carrera técnica y una ingeniería está fuertemente correlacionada con el rendimiento en matemáticas (PAES/PSU) y el puntaje NEM.
2. Existen sesgos o patrones territoriales y de dependencia escolar (Municipal vs. Subvencionado vs. Particular) en la inclinación hacia ingenierías.

---

## 💡 4. Propuesta de Solución

### 4.1 Funcionalidades Principales
* 📊 **Indicadores Clave (KPIs):** Total de estudiantes, porcentaje de inclinación por ingenierías y promedios académicos.
* 🎛️ **Filtros Interactivos:** Por región de origen, tipo de dependencia del establecimiento escolar y rango de puntaje NEM.
* 📈 **Visualizaciones Interactivas (Plotly):** Boxplots de notas, barras comparativas por tipo de colegio y gráficos de dispersión.
* 🤖 **Explicación No Técnica apoyada por IA:** Módulo *"Define tu Futuro"* que traduce patrones numéricos complejos a síntesis amigables para el usuario.

---

## ⚙️ 7. Desarrollo Técnico

* **Lenguaje:** Python 3.10+
* **Procesamiento de Datos:** `pandas`, `numpy`
* **Aplicación Web:** `streamlit`
* **Visualizaciones:** `plotly.express`
* **Modelamiento & Clasificación:** `scikit-learn`
* **Despliegue:** Streamlit Cloud

---

## 📂 12. Estructura del Repositorio

```text
/proyectos/equipo-orientacion-vocacional/
├── README.md                 # Documentación técnica e informe
├── requirements.txt           # Librerías necesarias para ejecución
├── app.py                     # Código fuente de la aplicación Streamlit
├── preprocesar.py             # Script de preparación del dataset DEMRE
├── data_procesada.csv         # Dataset limpio optimizado
├── notebook_analisis.ipynb    # Notebook con Análisis Exploratorio de Datos (EDA)
└── .streamlit/
    └── config.toml           # Configuración de tema gráfico
```

---

## 🌐 12.1 Enlace a la Aplicación Publicada
🚀 **Aplicación Interactiva:** [Enlace a Streamlit Cloud](https://streamlit.io/) *(Reemplazar con la URL pública final de Streamlit Cloud)*

---

## 👥 11. Integrantes del Equipo y Roles

* **Integrante 1:** Product Owner & Coordinador General
* **Integrante 2:** Data Engineer (Limpieza y Preprocesamiento DEMRE)
* **Integrante 3:** Data Scientist (EDA & Análisis Estadístico)
* **Integrante 4:** Frontend Developer (Desarrollo en Streamlit & Plotly)
* **Integrante 5:** UX Writer & Especialista en Comunicación No Técnica

---

## 🏁 13. Conclusiones

1. Los datos históricos del DEMRE permiten visibilizar brechas y patrones marcados en la elección de carrera según el tipo de colegio y el desempeño en pruebas de selección.
2. La integración de herramientas interactivas como **Streamlit** y síntesis en lenguaje natural permite democratizar la ciencia de datos para orientadores y estudiantes sin conocimientos técnicos.
