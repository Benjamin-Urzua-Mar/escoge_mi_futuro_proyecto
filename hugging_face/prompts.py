"""
Construcción del prompt para el informe vocacional generado por IA.

El prompt recibe *todos* los resultados que ``app.py`` calcula al presionar
el botón «🚀 Calcular admisión y afinidad» y los estructura de forma que
el modelo pueda interpretarlos y generar un informe coherente.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────

def _fmt(value, suffix: str = "%", decimals: int = 1) -> str:
    """Formatea un valor numérico; devuelve 'Sin datos' si es None/NaN."""
    # Centraliza el formato para evitar cifras inconsistentes en el informe.
    if value is None:
        return "Sin datos"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "Sin datos"


def _build_carreras_txt(carreras: list[dict]) -> str:
    """Genera la sección de carreras alternativas a partir de la lista de dicts."""
    if not carreras:
        return "No se encontraron carreras alternativas con los puntajes ingresados."

    # Convierte las recomendaciones estructuradas a texto legible para el modelo.
    lines: list[str] = []
    for i, c in enumerate(carreras, start=1):
        afinidad = c.get("afinidad_contextual")
        afinidad_txt = f" | Afinidad contextual: {_fmt(afinidad)}" if afinidad is not None else ""
        lines.append(
            f"{i}. {c.get('carrera', '—')} en {c.get('institucion', '—')} "
            f"— Probabilidad estimada: {_fmt(c.get('probabilidad'))} "
            f"| Corte P25: {_fmt(c.get('corte_estimado'), ' pts')}"
            f"{afinidad_txt}"
        )
    return "\n".join(lines)


def _build_factores_txt(factores: list[str]) -> str:
    """Genera la sección de factores sociodemográficos destacados."""
    if not factores:
        return "No se identificaron factores sociodemográficos destacados."
    # Mantiene cada factor como una viñeta independiente.
    return "\n".join(f"• {f}" for f in factores)


# ─────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────

def construir_prompt(
    perfil: dict,
    puntajes: dict,
    prediccion: dict,
    contexto: dict,
    distribucion_areas: dict,
    carreras_alternativas: list[dict],
) -> str:
    """Construye el prompt completo para el modelo de IA.

    Parameters
    ----------
    perfil : dict
        Datos contextuales del estudiante (sexo, región, dependencia, etc.).
    puntajes : dict
        Puntajes PAES/PSU ingresados por el usuario (nem, lenguaje, mate,
        prueba electiva y su tipo).
    prediccion : dict
        Resultado completo de ``predict_admission_probability``.
    contexto : dict
        Resultado de ``contextual_metrics`` (target_affinity, exact_matches, etc.).
    distribucion_areas : dict
        Resultado de ``calculate_area_distribution`` (Ingeniería, Técnica, Otra).
    carreras_alternativas : list[dict]
        Lista de diccionarios con las recomendaciones enriquecidas.
    """

    # Reúne los puntajes y la prueba electiva elegida.
    prueba_electiva = puntajes.get("tipo_electiva", "Ciencias")
    ptje_electiva = puntajes.get("puntaje_electiva", "Sin datos")

    # Recupera los promedios históricos para contextualizar los puntajes.
    promedios = prediccion.get("promedios_carrera", {})

    # Construye comparaciones explícitas entre el usuario y la referencia histórica.
    def _comparar(nombre: str, usuario, promedio) -> str:
        try:
            diff = float(usuario) - float(promedio)
            signo = "+" if diff >= 0 else ""
            return f"{nombre}: usuario {_fmt(usuario, ' pts', 0)} vs promedio histórico {_fmt(promedio, ' pts')} ({signo}{diff:.0f} pts)"
        except (TypeError, ValueError):
            return f"{nombre}: sin datos suficientes para comparar"

    comparaciones = "\n".join([
        _comparar("NEM", puntajes.get("nem"), promedios.get("NEM")),
        _comparar("Lenguaje", puntajes.get("lenguaje"), promedios.get("Lenguaje")),
        _comparar("Matemática", puntajes.get("matematica"), promedios.get("Matemáticas")),
        _comparar(
            prueba_electiva,
            ptje_electiva,
            promedios.get("Ciencias") if prueba_electiva == "Ciencias" else promedios.get("Historia"),
        ),
    ])

    # Prepara recomendaciones y factores antes de interpolarlos en el prompt final.
    carreras_txt = _build_carreras_txt(carreras_alternativas)
    factores_txt = _build_factores_txt(prediccion.get("factores", []))

    return f"""Actúa como un orientador vocacional profesional chileno especializado en interpretar estadísticas históricas del DEMRE.

# REGLAS ESTRICTAS

- Tu tarea consiste ÚNICAMENTE en interpretar la información entregada por la aplicación.
- No debes calcular porcentajes ni modificar cifras.
- No debes inventar información ni datos que no estén presentes.
- No debes emitir diagnósticos clínicos ni asegurar éxito académico o laboral.
- No debes afirmar relaciones de causa y efecto.
- Todas tus conclusiones deben basarse exclusivamente en los datos proporcionados.
- La información corresponde a tendencias históricas observadas en los datos del DEMRE y NO a predicciones individuales garantizadas.

====================================================
PERFIL DEL ESTUDIANTE

Sexo: {perfil.get("sexo", "No informado")}
Nacionalidad: {perfil.get("nacionalidad", "No informado")}
Región: {perfil.get("region", "No informado")}
Dependencia educacional: {perfil.get("dependencia", "No informado")}
Rama educacional: {perfil.get("rama", "No informado")}
Cuantil de ingresos: {perfil.get("cuantil", "No informado")}
Situación laboral: {perfil.get("trabajo", "No informado")}
Sistema de financiamiento: {perfil.get("financiamiento", "No informado")}
Sistema de salud: {perfil.get("salud", "No informado")}
Convivencia: {perfil.get("convivencia", "No informado")}
Jefe de hogar: {perfil.get("jefe_hogar", "No informado")}

Carrera consultada: {perfil.get("carrera", "No informado")}
Institución consultada: {perfil.get("institucion", "No informado")}

====================================================
PUNTAJES INGRESADOS POR EL ESTUDIANTE

NEM: {_fmt(puntajes.get("nem"), " pts", 0)}
Competencia Lectora / Lenguaje: {_fmt(puntajes.get("lenguaje"), " pts", 0)}
Matemática (M1): {_fmt(puntajes.get("matematica"), " pts", 0)}
Prueba electiva ({prueba_electiva}): {_fmt(ptje_electiva, " pts", 0)}

====================================================
RESULTADO DE ADMISIÓN (calculado por el modelo)

Probabilidad estimada de ingreso: {_fmt(prediccion.get("probabilidad"))}
Clasificación: {prediccion.get("etiqueta", "Sin clasificación")}
Ponderado estimado del postulante: {_fmt(prediccion.get("user_ponderado"), " pts")}
Puntaje de corte histórico (P25): {_fmt(prediccion.get("corte_p25"), " pts")}
Mediana histórica (P50): {_fmt(prediccion.get("promedio_p50"), " pts")}

====================================================
COMPARACIÓN PUNTAJES DEL ESTUDIANTE VS PROMEDIOS HISTÓRICOS DE LA CARRERA

{comparaciones}

====================================================
FACTORES SOCIODEMOGRÁFICOS DETECTADOS POR EL MODELO

{factores_txt}

====================================================
AFINIDAD CONTEXTUAL

Afinidad con la carrera consultada: {_fmt(contexto.get("target_affinity"))}
Registros históricos encontrados: {contexto.get("target_rows", 0)}
Coincidencias exactas de perfil: {contexto.get("exact_matches", 0)}
Ponderado promedio de perfiles cercanos: {_fmt(contexto.get("nearest_ponderado"), " pts")}

Distribución de áreas entre perfiles similares:
- Ingeniería / Tecnológica: {_fmt(distribucion_areas.get("Ingeniería"))}
- Técnica / Aplicada: {_fmt(distribucion_areas.get("Técnica"))}
- Otras disciplinas: {_fmt(distribucion_areas.get("Otra"))}

====================================================
CARRERAS ALTERNATIVAS RECOMENDADAS

{carreras_txt}

====================================================

Genera el informe utilizando EXACTAMENTE los siguientes títulos y en el orden indicado.

# 📋 Resumen del perfil

Describe brevemente al estudiante integrando:
- Sus puntajes ingresados.
- Su contexto socioeducativo (región, rama, dependencia, cuantil).
- La carrera e institución consultadas.

Máximo 120 palabras.

----------------------------------------------------

# 📊 Diagnóstico de admisión

Interpreta el resultado de admisión entregado por el modelo:
- Explica qué significa la probabilidad estimada y su clasificación (Alta / Media / Baja).
- Compara el ponderado estimado del estudiante con el corte P25 y la mediana P50.
- Si el ponderado supera el P25, indícalo positivamente. Si está por debajo, explica con prudencia.
- Si existen factores sociodemográficos detectados, menciónalos.

Aclara que estos resultados representan tendencias históricas y NO garantías individuales.

No inventes cifras.

----------------------------------------------------

# 📈 Análisis comparativo

Interpreta la comparación de los puntajes del estudiante frente a los promedios históricos de la carrera:
- Destaca las pruebas donde el estudiante supera el promedio.
- Señala con prudencia las pruebas donde está por debajo.
- Si hay diferencias notables, sugiere fortalecer esas áreas.

Máximo 100 palabras.

----------------------------------------------------

# 🧭 Afinidad contextual

Explica de forma sencilla qué representa la afinidad contextual:
- Qué significa el porcentaje de afinidad con la carrera.
- Qué representan las distribuciones por área (Ingeniería, Técnica, Otras).
- Menciona la cantidad de registros históricos y coincidencias exactas si son relevantes.

Aclara que la afinidad mide similitud con perfiles históricos, no probabilidad de éxito.

Máximo 100 palabras.

----------------------------------------------------

# ⭐ Fortalezas observadas

Entrega entre cuatro y seis fortalezas.
Cada fortaleza debe justificarse con datos concretos del perfil, puntajes o resultados.
Interpreta la información de manera legible y coherente, evitando repetir la misma idea.
No inventes fortalezas.

----------------------------------------------------

# 🎯 Recomendaciones

Genera recomendaciones prácticas y personalizadas:

- Indica si la carrera consultada parece coherente con el perfil y los puntajes.
- Menciona las carreras alternativas recomendadas con sus probabilidades.
- Recomienda revisar las mallas curriculares de las opciones sugeridas.
- Recomienda investigar empleabilidad y duración de las carreras.
- Recomienda conversar con estudiantes o titulados.
- Recomienda explorar distintas instituciones que impartan las carreras de interés.

Finaliza exactamente con el siguiente texto:

"Este informe corresponde a una interpretación de tendencias históricas observadas en los datos analizados y no constituye una predicción individual sobre el desempeño o éxito futuro del postulante."

Devuelve únicamente el informe en formato Markdown.
"""