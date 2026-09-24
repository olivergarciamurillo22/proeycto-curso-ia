"""Pruebas de integración de la web de Pedro, sin red ni credenciales reales."""
import unittest
from unittest.mock import patch, Mock

import numpy as np
from streamlit.testing.v1 import AppTest

import chatbot
from test_processor import resultado, respuesta_http


class IntegracionTests(unittest.TestCase):
    def test_chat_llama_sin_system(self):
        respuesta = respuesta_http()
        respuesta.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'A las 20:00.'}}]}
        with patch.object(chatbot, 'NVIDIA_API_KEY', 'token-de-test'), \
                patch.object(chatbot, 'NIM_MODEL', 'meta/llama-3.2-11b-vision-instruct'), \
                patch.object(chatbot.requests, 'post', return_value=respuesta) as post:
            self.assertIn('20:00', chatbot.preguntar_chatbot('¿Hora?', resultado()['eventos']))
        payload = post.call_args.kwargs['json']
        self.assertTrue(all(m['role'] != 'system' for m in payload['messages']))
        self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_chat_no_expone_cuerpo_error(self):
        respuesta = respuesta_http(401)
        respuesta.text = 'datos privados que no deben mostrarse'
        with patch.object(chatbot, 'NVIDIA_API_KEY', 'token-de-test'), \
                patch.object(chatbot.requests, 'post', return_value=respuesta):
            with self.assertRaises(RuntimeError) as error:
                chatbot.generar_con_nim([{'role': 'user', 'content': 'Hola'}])
        self.assertNotIn('datos privados', str(error.exception))

    def test_rag_y_fallback(self):
        eventos = resultado()['eventos']
        modelo = Mock()
        modelo.encode.return_value = np.array([[1, 0]], dtype=np.float32)
        vectores, _ = chatbot.crear_indice_eventos(eventos, modelo)
        self.assertEqual(chatbot.construir_contexto_chat('hora', eventos, vectores, modelo)[2], 'rag')
        self.assertEqual(chatbot.construir_contexto_chat('hora', eventos)[2], 'completo')
        self.assertEqual(chatbot.construir_contexto_chat('hora', [])[2], 'vacio')

    def test_web_programa_filtros_y_ajustes(self):
        with patch.object(chatbot, 'cargar_modelo_embeddings', side_effect=RuntimeError('offline')), \
                patch.object(chatbot.requests, 'post', side_effect=AssertionError('Red no permitida')):
            app = AppTest.from_file('app.py', default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('⚙️ Ajustes y tecnología', [tab.label for tab in app.tabs])
            self.assertTrue(any('La inteligencia' in item.value for item in app.subheader))
            app.text_input[0].set_value('NO_EXISTE_EVENTO_93842').run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Ningún evento' in item.value for item in app.info))
            app.radio[0].set_value('OCR').run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Tesseract reconoce' in item.value for item in app.markdown))


if __name__ == '__main__':
    unittest.main(verbosity=2)
