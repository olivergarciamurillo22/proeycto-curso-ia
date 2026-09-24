import unicodedata

import streamlit as st

from chatbot import (
    RESPUESTA_SIN_DATOS,
    cargar_modelo_embeddings,
    construir_contexto_chat,
    crear_indice_eventos,
    preguntar_chatbot,
)

st.set_page_config(page_title="Agenda Inteligente de Almería", page_icon="🎉", layout="wide")

DATOS_DEMO = {
    "documento": {
        "titulo": "Programa de fiestas de Almería",
        "descripcion": "Datos temporales de desarrollo"
    },
    "eventos": [
        {
            "nombre": "Concierto de Rock",
            "fecha": "24 de agosto",
            "hora": "22:00",
            "lugar": "Plaza Vieja",
            "descripcion": "Concierto gratuito al aire libre.",
            "categoria": "Música"
        },
        {
            "nombre": "Actividades infantiles",
            "fecha": "24 de agosto",
            "hora": "18:00",
            "lugar": "Parque Nicolás Salmerón",
            "descripcion": "Juegos y actividades para niños.",
            "categoria": "Infantil"
        },
        {
            "nombre": "Espectáculo de flamenco",
            "fecha": "25 de agosto",
            "hora": "21:30",
            "lugar": "Plaza de la Catedral",
            "descripcion": "Actuación de flamenco.",
            "categoria": "Cultura"
        },
        {
            "nombre": "Fuegos artificiales",
            "fecha": "25 de agosto",
            "hora": "00:00",
            "lugar": "Paseo Marítimo",
            "descripcion": "Espectáculo de fuegos artificiales.",
            "categoria": "Espectáculo"
        }
    ]
}

CAMPOS_EVENTO = ["nombre", "fecha", "hora", "lugar", "descripcion", "categoria"]
TODOS = "Todos"
TOP_K = 5

ICONOS_CATEGORIA = {
    "musica": "🎵",
    "infantil": "🧸",
    "cultura": "🎭",
    "espectaculo": "🎆",
    "deporte": "⚽",
    "gastronomia": "🍽️",
}


# ============================================================
# INTEGRACIÓN PERSONA 1: solo hay que cambiar esta función.
# ============================================================
def cargar_datos():
    datos_json = DATOS_DEMO
    texto_completo = ""
    # from document_processor import procesar_pdf
    # datos_json, texto_completo = procesar_pdf("programa.pdf")
    return datos_json, texto_completo


def quitar_acentos(texto):
    texto = unicodedata.normalize("NFD", str(texto or ""))
    return "".join(c for c in texto if unicodedata.category(c) != "Mn").lower().strip()


def normalizar_evento(evento):
    if not isinstance(evento, dict):
        evento = {}
    normalizado = {}
    for campo in CAMPOS_EVENTO:
        valor = evento.get(campo)
        if valor is None:
            normalizado[campo] = None
        else:
            valor = str(valor).strip()
            normalizado[campo] = valor if valor and valor.lower() not in ("null", "none") else None
    return normalizado


def preparar_datos(datos_json):
    if not isinstance(datos_json, dict):
        datos_json = {}
    documento = datos_json.get("documento")
    if not isinstance(documento, dict):
        documento = {}
    eventos = datos_json.get("eventos")
    if not isinstance(eventos, list):
        eventos = []
    eventos = [normalizar_evento(e) for e in eventos if isinstance(e, dict)]
    eventos = [e for e in eventos if any(e.values())]
    return {
        "documento": {
            "titulo": documento.get("titulo") or "Programa de eventos",
            "descripcion": documento.get("descripcion") or "",
        },
        "eventos": eventos,
    }


def valores_unicos(eventos, campo):
    return list(dict.fromkeys(e[campo] for e in eventos if e.get(campo)))


def filtrar_eventos(eventos, texto="", fecha=TODOS, lugar=TODOS, categoria=TODOS):
    texto_busqueda = quitar_acentos(texto)
    resultado = []
    for e in eventos:
        if fecha != TODOS and e.get("fecha") != fecha:
            continue
        if lugar != TODOS and e.get("lugar") != lugar:
            continue
        if categoria != TODOS and e.get("categoria") != categoria:
            continue
        if texto_busqueda:
            contenido = " ".join(
                quitar_acentos(e.get(c)) for c in ("nombre", "descripcion", "lugar", "categoria")
            )
            if texto_busqueda not in contenido:
                continue
        resultado.append(e)
    return resultado


def mostrar_evento(evento):
    icono = ICONOS_CATEGORIA.get(quitar_acentos(evento.get("categoria")), "🎵")
    with st.container(border=True):
        st.markdown(f"### {icono} {evento.get('nombre') or 'Evento sin nombre'}")
        c1, c2 = st.columns(2)
        c1.markdown(f"📅 **Fecha:** {evento.get('fecha') or 'No especificado'}")
        c1.markdown(f"🕐 **Hora:** {evento.get('hora') or 'No especificado'}")
        c2.markdown(f"📍 **Lugar:** {evento.get('lugar') or 'No especificado'}")
        c2.markdown(f"🏷️ **Categoría:** {evento.get('categoria') or 'No especificado'}")
        if evento.get("descripcion"):
            st.write(evento["descripcion"])


@st.cache_resource(show_spinner=False)
def obtener_modelo_embeddings():
    return cargar_modelo_embeddings()


def inicializar_estado():
    if "datos_json" not in st.session_state:
        try:
            datos_json, texto_completo = cargar_datos()
        except Exception as e:
            st.session_state.error_datos = f"Error al cargar los datos del programa: {e}"
            datos_json, texto_completo = {}, ""
        st.session_state.datos_json = preparar_datos(datos_json)
        st.session_state.texto_completo = texto_completo if isinstance(texto_completo, str) else ""

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    if "eventos_indice" not in st.session_state:
        eventos = st.session_state.datos_json["eventos"]
        st.session_state.eventos_indice = eventos
        st.session_state.embeddings = None
        st.session_state.modelo_embeddings = None
        st.session_state.error_embeddings = None
        if eventos:
            try:
                with st.spinner("Preparando la búsqueda inteligente..."):
                    modelo = obtener_modelo_embeddings()
                    embeddings, _ = crear_indice_eventos(eventos, modelo)
                st.session_state.modelo_embeddings = modelo
                st.session_state.embeddings = embeddings
            except Exception as e:
                st.session_state.error_embeddings = f"{type(e).__name__}: {e}"


# ============================================================
# INTERFAZ
# ============================================================
inicializar_estado()

datos_json = st.session_state.datos_json
eventos = datos_json["eventos"]

st.title("🎉 Agenda Inteligente de Almería")
st.caption("Consulta toda la programación y pregunta a nuestra IA.")

if st.session_state.get("error_datos"):
    st.error(st.session_state.error_datos)

fechas = valores_unicos(eventos, "fecha")
lugares = valores_unicos(eventos, "lugar")
categorias = valores_unicos(eventos, "categoria")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Eventos", len(eventos))
m2.metric("Lugares", len(lugares))
m3.metric("Categorías", len(categorias))
m4.metric("Fechas", len(fechas))

tab_programa, tab_chat = st.tabs(["📅 Programa", "🤖 Pregunta a la IA"])

with tab_programa:
    st.subheader(datos_json["documento"]["titulo"])
    if datos_json["documento"]["descripcion"]:
        st.write(datos_json["documento"]["descripcion"])

    if not eventos:
        st.warning("No hay eventos disponibles en el programa.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        texto = f1.text_input("🔎 Buscar", placeholder="Rock, flamenco, niños...")
        fecha = f2.selectbox("📅 Fecha", [TODOS] + fechas)
        lugar = f3.selectbox("📍 Lugar", [TODOS] + lugares)
        categoria = f4.selectbox("🏷️ Categoría", [TODOS] + categorias)

        filtrados = filtrar_eventos(eventos, texto, fecha, lugar, categoria)
        st.write(f"**{len(filtrados)}** evento(s) encontrado(s)")

        if not filtrados:
            st.info("Ningún evento coincide con los filtros.")
        for evento in filtrados:
            mostrar_evento(evento)

    if st.session_state.texto_completo:
        with st.expander("Texto completo del programa"):
            st.text(st.session_state.texto_completo)

with tab_chat:
    if st.session_state.error_embeddings:
        st.warning("Búsqueda semántica no disponible; el chatbot usará todo el programa.")

    if st.button("🗑️ Borrar conversación"):
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input("Pregunta sobre la programación...")

    if pregunta:
        historial = list(st.session_state.mensajes)
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            respuesta = None
            eventos_usados = []
            modo = "vacio"
            with st.spinner("Pensando..."):
                try:
                    contexto, eventos_usados, modo = construir_contexto_chat(
                        pregunta,
                        st.session_state.eventos_indice,
                        st.session_state.embeddings,
                        st.session_state.modelo_embeddings,
                        TOP_K,
                    )
                    if modo == "vacio":
                        respuesta = RESPUESTA_SIN_DATOS
                    else:
                        respuesta = preguntar_chatbot(pregunta, contexto, historial)
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error inesperado en el chatbot: {type(e).__name__}")

            if respuesta:
                st.markdown(respuesta)
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
                if eventos_usados:
                    etiqueta = "Eventos usados (búsqueda semántica)" if modo == "rag" else "Eventos usados (programa completo)"
                    with st.expander(etiqueta):
                        for e in eventos_usados:
                            st.write(f"- {e.get('nombre') or 'Sin nombre'} · {e.get('fecha') or '—'} · {e.get('hora') or '—'} · {e.get('lugar') or '—'}")
