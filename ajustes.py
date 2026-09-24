"""Panel de transparencia para la interfaz de Pedro. No expone secretos."""
import json
from importlib.metadata import version

import streamlit as st

from chatbot import NIM_MODEL, MODELO_EMBEDDINGS
from document_processor import obtener_info_tecnica


def mostrar_ajustes(informe=None):
    info = obtener_info_tecnica()
    st.subheader('La inteligencia, a la vista')
    st.caption('Qué lee tu programa, cómo encuentra los eventos y dónde se procesan tus datos.')
    modelos = [
        ('01 · Lectura del programa', info['modelo_configurado'] or 'Sin configurar',
         'NVIDIA NIM · imagen y texto, con alternativa solo texto'),
        ('02 · Conversación', NIM_MODEL, 'NVIDIA NIM · respuestas basadas en el programa'),
        ('03 · Búsqueda semántica', MODELO_EMBEDDINGS,
         'Embeddings locales · similitud entre tu pregunta y los eventos'),
    ]
    for col, (titulo, modelo, uso) in zip(st.columns(3), modelos):
        with col, st.container(border=True):
            st.caption(titulo)
            st.code(modelo, language=None)
            st.write(uso)
    st.caption('Un modelo configurado no implica una conexión verificada ni una respuesta correcta.')
    if st.session_state.get('error_embeddings'):
        st.warning('La búsqueda semántica no está disponible. El chat recibe el programa completo.')
    elif st.session_state.get('modelo_embeddings') is not None:
        st.success('Búsqueda semántica cargada en esta sesión.')
    if not info['credencial_configurada']:
        st.warning('Falta configurar la credencial de NVIDIA en el servidor.')

    izquierda, derecha = st.columns([3, 2])
    with izquierda, st.container(border=True):
        st.markdown('#### El recorrido de tu documento')
        etapa = st.radio('Explora cada paso', ['PDF', 'OCR', 'IA', 'Agenda'], horizontal=True)
        explicaciones = {
            'PDF': 'PyMuPDF lee el texto original y renderiza cada página como imagen.',
            'OCR': 'Tesseract reconoce el texto de imágenes en el servidor. Se combina con el texto nativo sin repetir líneas iguales.',
            'IA': 'NVIDIA recibe una página cada vez y devuelve eventos estructurados. Si rechaza la imagen, se intenta con texto.',
            'Agenda': 'Se valida el JSON, se combinan las páginas y se eliminan repeticiones exactas con identidad completa.',
        }
        st.write(explicaciones[etapa])
        st.caption('El procesamiento completo no garantiza exactitud: contrasta los datos con el PDF.')
    with derecha, st.container(border=True):
        st.markdown('#### Tus datos')
        st.write('**En el servidor:** PDF, imágenes, OCR y búsqueda semántica.')
        st.write('**En NVIDIA:** imágenes/texto para extracción; pregunta, contexto e historial reciente para el chat.')
        st.caption('Las claves permanecen en el servidor. Esta pantalla no permite verlas ni descargarlas.')

    if informe:
        st.markdown('#### Última extracción')
        a, b, c = st.columns(3)
        a.metric('Páginas procesadas', f"{informe.get('paginas_correctas', 0)} / {informe.get('paginas_totales', 0)}")
        b.metric('Eventos extraídos', informe.get('eventos_finales', 0))
        c.metric('Duración', f"{informe.get('duracion_segundos', 0):.1f} s")
        st.write('Estado:', informe.get('estado', 'desconocido'))
        st.caption('Modelo de esta extracción: ' + str(informe.get('modelo', 'desconocido')))
        with st.expander('Detalle por página'):
            st.dataframe(informe.get('paginas', []), hide_index=True, use_container_width=True)
    else:
        st.info('No hay una extracción registrada para este programa.')

    with st.expander('Tecnología y versiones'):
        tecnologias = info['tecnologias'] + [
            {'nombre': p, 'version': version(p), 'uso': uso}
            for p, uso in [('streamlit', 'Interfaz web'), ('numpy', 'Cálculo de similitudes'),
                           ('sentence-transformers', 'Embeddings multilingües'), ('torch', 'Inferencia local')]]
        st.dataframe(tecnologias, hide_index=True, use_container_width=True)
        st.write('OCR:', info['ocr'])
    diagnostico = {**info, 'modelo_chat': NIM_MODEL, 'modelo_embeddings': MODELO_EMBEDDINGS,
                   'ultima_ejecucion': informe or None}
    st.download_button('Descargar diagnóstico técnico', json.dumps(diagnostico, ensure_ascii=False, indent=2),
                       file_name='diagnostico.json', mime='application/json')
