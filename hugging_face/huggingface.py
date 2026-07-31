"""
Cliente de HuggingFace Inference API para generar informes vocacionales.

Utiliza la API de chat completions con el modelo Qwen2.5-7B-Instruct.
El token se lee de la variable de entorno HF_TOKEN con fallback interno.
"""

import logging
import os

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_FALLBACK_TOKEN = "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"
_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM_MESSAGE = (
    "Eres un orientador vocacional profesional chileno. "
    "Interpretas estadísticas históricas del DEMRE de manera clara y responsable. "
    "Respondes exclusivamente en español de Chile y en formato Markdown."
)


def _get_client() -> InferenceClient:
    """Devuelve un cliente singleton de HuggingFace Inference."""
    token = os.environ.get("HF_TOKEN", _FALLBACK_TOKEN)
    return InferenceClient(token=token)


def generar_informe(prompt: str) -> str:
    """Envía el prompt al modelo y devuelve el informe generado en Markdown.

    Parameters
    ----------
    prompt : str
        Prompt completo construido por ``construir_prompt``.

    Returns
    -------
    str
        Texto del informe o mensaje de error descriptivo.
    """
    try:
        client = _get_client()
        response = client.chat_completion(
            model=_MODEL_ID,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=1500,
        )

        if response.choices:
            return response.choices[0].message.content

        logger.warning("La respuesta del modelo no contiene choices.")
        return "⚠️ No se recibió respuesta del modelo. Intenta nuevamente."

    except Exception as exc:
        logger.error("Error al generar informe con HuggingFace: %s", exc)
        return (
            "⚠️ No se pudo generar el informe de IA en este momento. "
            f"Detalle técnico: {exc}"
        )