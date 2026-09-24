import json
import os

import numpy as np
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ============================================================
# CONFIGURACIÓN NVIDIA NIM
# Se lee del archivo .env (el mismo que usa document_processor.py)
# o de variables de entorno. NO escribas la key aquí: el repo es público.
# ============================================================
NIM_URL = os.getenv("NVIDIA_NIM_URL", "").strip() or "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip() or "AQUI VA LA API"
NIM_MODEL = (
    os.getenv("NIM_MODEL", "").strip()
    or os.getenv("NVIDIA_MODEL", "").strip()
    or "meta/llama-3.3-70b-instruct"
)
NIM_TIMEOUT = 60

MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RESPUESTA_SIN_DATOS = "No encuentro esa información en el programa."

SYSTEM_PROMPT = """Eres un asistente especializado en la programación de eventos de Almería.

Responde exclusivamente utilizando la información del programa proporcionado.

No inventes nombres, eventos, fechas, horarios ni ubicaciones.

Si la respuesta no puede obtenerse de los datos proporcionados, responde:

'No encuentro esa información en el programa.'

Si existen varios eventos relevantes, puedes mostrarlos en una lista.

Responde siempre en español.

Sé claro y conciso."""

CAMPOS_TEXTO = [
    ("Evento", "nombre"),
    ("Fecha", "fecha"),
    ("Hora", "hora"),
    ("Lugar", "lugar"),
    ("Categoría", "categoria"),
    ("Descripción", "descripcion"),
]


def _valor(evento, campo):
    if not isinstance(evento, dict):
        return "No especificado"
    valor = evento.get(campo)
    if valor is None or str(valor).strip() == "":
        return "No especificado"
    return str(valor).strip()


def evento_a_texto(evento):
    return "\n".join(f"{etiqueta}: {_valor(evento, campo)}" for etiqueta, campo in CAMPOS_TEXTO)


def cargar_modelo_embeddings():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELO_EMBEDDINGS)


def _normalizar_vectores(vectores):
    vectores = np.asarray(vectores, dtype=np.float32)
    if vectores.ndim == 1:
        vectores = vectores.reshape(1, -1)
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / np.maximum(normas, 1e-12)


def crear_indice_eventos(eventos, modelo_embeddings=None):
    """Devuelve (embeddings normalizados en RAM, textos)."""
    textos = [evento_a_texto(e) for e in (eventos or [])]
    if not textos:
        return np.zeros((0, 0), dtype=np.float32), []
    if modelo_embeddings is None:
        modelo_embeddings = cargar_modelo_embeddings()
    vectores = modelo_embeddings.encode(textos, convert_to_numpy=True, show_progress_bar=False)
    return _normalizar_vectores(vectores), textos


def buscar_eventos_semanticamente(pregunta, eventos, embeddings, modelo_embeddings, top_k=5):
    if not pregunta or not eventos or modelo_embeddings is None:
        return []
    if embeddings is None or len(embeddings) == 0 or len(embeddings) != len(eventos):
        return []

    vector_pregunta = modelo_embeddings.encode([pregunta], convert_to_numpy=True, show_progress_bar=False)
    vector_pregunta = _normalizar_vectores(vector_pregunta)[0]

    similitudes = embeddings @ vector_pregunta
    k = max(1, min(int(top_k), len(eventos)))
    indices = np.argsort(-similitudes)[:k]
    return [eventos[i] for i in indices]


def generar_con_nim(messages):
    """Llama a NVIDIA NIM. Devuelve el texto o lanza RuntimeError con un mensaje seguro."""
    if not NVIDIA_API_KEY or NVIDIA_API_KEY == "AQUI VA LA API":
        raise RuntimeError("Falta configurar NVIDIA_API_KEY (variable de entorno).")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": NIM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 700,
        "stream": False,
    }

    try:
        respuesta = requests.post(NIM_URL, headers=headers, json=payload, timeout=NIM_TIMEOUT)
    except requests.exceptions.Timeout:
        raise RuntimeError("NVIDIA NIM ha tardado demasiado en responder. Inténtalo de nuevo.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con NVIDIA NIM: {type(e).__name__}")

    if respuesta.status_code != 200:
        detalle = respuesta.text[:300].replace(NVIDIA_API_KEY, "***")
        raise RuntimeError(f"NVIDIA NIM devolvió el error {respuesta.status_code}: {detalle}")

    try:
        contenido = respuesta.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("Respuesta inesperada de NVIDIA NIM.")

    contenido = (contenido or "").strip()
    return contenido or RESPUESTA_SIN_DATOS


def construir_contexto_chat(pregunta, eventos, embeddings=None, modelo_embeddings=None, top_k=5):
    """Devuelve (contexto, eventos_usados, modo). modo = 'rag' | 'completo' | 'vacio'."""
    eventos = eventos or []
    if not eventos:
        return "", [], "vacio"

    try:
        relevantes = buscar_eventos_semanticamente(pregunta, eventos, embeddings, modelo_embeddings, top_k)
    except Exception:
        relevantes = []

    if relevantes:
        contexto = "\n\n".join(evento_a_texto(e) for e in relevantes)
        return contexto, relevantes, "rag"

    # Fallback sin embeddings: todos los eventos
    return json.dumps(eventos, ensure_ascii=False), eventos, "completo"


def preguntar_chatbot(pregunta, eventos_relevantes, historial=None):
    """eventos_relevantes puede ser una lista de eventos o un contexto ya construido (str)."""
    if isinstance(eventos_relevantes, str):
        contexto = eventos_relevantes
    else:
        contexto = "\n\n".join(evento_a_texto(e) for e in (eventos_relevantes or []))

    if not contexto.strip():
        return RESPUESTA_SIN_DATOS

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for mensaje in (historial or [])[-6:]:
        if (
            isinstance(mensaje, dict)
            and mensaje.get("role") in ("user", "assistant")
            and isinstance(mensaje.get("content"), str)
        ):
            messages.append({"role": mensaje["role"], "content": mensaje["content"]})

    messages.append({
        "role": "user",
        "content": f"EVENTOS RELEVANTES:\n{contexto}\n\nPREGUNTA:\n{pregunta}",
    })

    return generar_con_nim(messages)
