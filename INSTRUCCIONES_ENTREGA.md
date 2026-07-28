# 📋 Guía Completa de Entrega y Presentación Final
## Samsung Innovation Campus Chile 2026 – Cohort 2

Esta carpeta `/proyectos/orientador-vocacional-samsung/` contiene el 100% de los entregables requeridos para la evaluación del proyecto final.

---

## 📂 Estructura de Entregables Incluidos

```text
/proyectos/orientador-vocacional-samsung/
├── README.md                 # Informe técnico oficial en las 13 secciones solicitadas
├── INSTRUCCIONES_ENTREGA.md  # Guía de despliegue, GitHub Pull Request y pitch
├── requirements.txt           # Dependencias (streamlit, pandas, plotly, scikit-learn)
├── app.py                     # Código fuente de la aplicación Streamlit Ultra-Premium
├── data_procesada.csv         # Dataset DEMRE optimizado (22 variables, 50,000 filas)
├── preprocesar.py             # Script de preparación y limpieza de datos
├── notebook_analisis.ipynb    # Notebook documentado de Análisis Exploratorio (EDA)
└── .streamlit/
    └── config.toml           # Configuración de tema visual y modo oscuro
```

---

## 🚀 1. Cómo Probar la Aplicación Localmente

En la terminal, navega a esta carpeta y ejecuta:

```bash
cd proyectos/orientador-vocacional-samsung
streamlit run app.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## 🌐 2. Pasos para la Publicación Gratuita en Streamlit Cloud

1. **Subir los cambios a GitHub:**
   - Asegúrate de subir la carpeta `proyectos/orientador-vocacional-samsung/` a tu repositorio del equipo.
2. **Desplegar en la Nube:**
   - Ingresa a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con GitHub.
   - Haz clic en **"New app"**.
   - Selecciona tu repositorio, la rama `main`, y en la ruta del archivo ingresa: `proyectos/orientador-vocacional-samsung/app.py`.
   - Haz clic en **"Deploy"**.
3. **Copiar URL Pública:**
   - Una vez desplegada, copia la URL que te asigna Streamlit Cloud (ej: `https://orientador-vocacional.streamlit.app`).
   - Pégala en el apartado de enlace público de tu `README.md`.

---

## 📤 3. Pasos para el Pull Request en GitHub (Antes del Viernes 31 de Julio)

1. Haz **Fork** o crea una nueva rama (*branch*) en el repositorio del curso.
2. Sube la carpeta `/proyectos/orientador-vocacional-samsung/`.
3. Abre un **Pull Request** hacia la rama principal del curso especificando:
   - **Título:** `PR Entrega Final Equipo Orientador Vocacional - Cohort 2`
   - **Descripción:** Incluir el enlace público de Streamlit Cloud y la lista de integrantes del equipo.

---

## 🎤 4. Guion Sugerido para la Presentación Final (5 a 7 Minutos)

* **0:00 - 1:00 (El Problema Real):** Explicar la desorientación vocacional y la deserción estudiantil en Chile.
* **1:00 - 4:00 (Demostración en Vivo):** 
  - Abrir la app desplegada en pantalla completa.
  - Rllenar la ficha vocacional con 4-5 características.
  - Mostrar la **Ficha Resumen del Postulante** y el indicador animado de carga.
  - Consultar una carrera específica y mostrar las tarjetas de afinidad en vivo.
* **4:00 - 5:30 (Explicación No Técnica):** Resaltar cómo las correlaciones contextuales ayudan a tomar decisiones sin depender de puntajes de pruebas.
* **5:30 - 7:00:** Preguntas de los docentes.
