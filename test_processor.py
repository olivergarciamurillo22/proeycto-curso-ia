"""Pruebas locales: PDF/OCR reales y HTTP NIM simulado, sin consumir API."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import fitz
import pytesseract
import requests

import document_processor as dp


def resultado(nombre='Concierto de prueba'):
    return {'documento': {'titulo': 'Programa de prueba', 'descripcion': None},
            'eventos': [dict(zip(dp.CAMPOS, (nombre, '24 septiembre', '20:00',
                                           'Plaza Mayor', None, 'Música')))]}


def respuesta_http(status=200, datos=None, finish='stop'):
    response = Mock(status_code=status)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.json.return_value = {'choices': [{'finish_reason': finish, 'message': {
        'content': json.dumps(datos if datos is not None else resultado())}}]}
    return response


def crear_pdf(destino):
    """Una página digital y otra escaneada, ambas con eventos ficticios."""
    doc = fitz.open()
    pagina = doc.new_page()
    pagina.insert_text((60, 80), 'Programa de prueba\nConcierto de prueba\n'
                        '24 septiembre - 20:00\nPlaza Mayor', fontsize=20)
    png = pagina.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes('png')
    doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=png)
    doc.save(destino)
    doc.close()
    return png


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'NVIDIA_API_KEY': 'clave-ficticia-tests',
                                          'NVIDIA_MODEL': 'modelo-ficticio-tests'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_parser_formatos(self):
        raw = json.dumps(resultado(), ensure_ascii=False)
        for text in (raw, f'```json\n{raw}\n```', f'Aquí está:\n{raw}\nFin.'):
            with self.subTest(text=text):
                self.assertEqual(dp.parsear_json_nim(text), resultado())

    def test_parser_rechaza(self):
        for text in ('no json', '{"documento": {}, "eventos": [',
                     '{"documento": {}, "eventos": {}}',
                     '{"documento": {}, "eventos": [{"hora": 20}]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                dp.parsear_json_nim(text)

    def test_normaliza_campos_sin_inventar(self):
        data = dp.parsear_json_nim('{"documento":{},"eventos":[{"nombre":"X"}]}')
        self.assertIsNone(data['eventos'][0]['fecha'])
        self.assertEqual(set(data['eventos'][0]), set(dp.CAMPOS))

    def test_hibrido(self):
        texto = 'Concierto\nPlaza Mayor\n20:00'
        self.assertEqual(dp.combinar_textos(texto, texto), texto)
        self.assertIn('Entrada libre', dp.combinar_textos(texto, texto + '\nEntrada libre'))

    def test_dedupe_conservadora(self):
        self.assertEqual(len(dp._combinar_resultados([resultado(), resultado()])['eventos']), 1)
        distinto = resultado()
        distinto['eventos'][0]['hora'] = '21:00'
        self.assertEqual(len(dp._combinar_resultados([resultado(), distinto])['eventos']), 2)
        incompleto = resultado()
        incompleto['eventos'][0]['fecha'] = None
        self.assertEqual(len(dp._combinar_resultados([incompleto, incompleto])['eventos']), 2)

    @patch('document_processor.requests.post')
    def test_falta_configuracion_no_llama(self, post):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': ''}), self.assertRaises(ValueError):
            dp.analizar_pagina_con_nim(1, None, 'texto', '')
        post.assert_not_called()

    @patch('document_processor.requests.post')
    def test_fallback_multimodal(self, post):
        post.side_effect = [respuesta_http(400), respuesta_http()]
        imagen = dp.Image.new('RGB', (20, 20), 'white')
        with self.assertLogs(dp.logger, level='WARNING') as logs:
            data = dp.analizar_pagina_con_nim(1, imagen, 'Concierto', '')
        self.assertEqual(data, resultado())
        first = post.call_args_list[0].kwargs['json']['messages'][1]['content']
        self.assertTrue(first[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertIsInstance(post.call_args_list[1].kwargs['json']['messages'][1]['content'], str)
        self.assertIn('NIM multimodal falló', ' '.join(logs.output))

    @patch('document_processor.time.sleep')
    @patch('document_processor.requests.post')
    def test_reintento_temporal(self, post, sleep):
        post.side_effect = [requests.Timeout(), respuesta_http(429), respuesta_http()]
        self.assertEqual(dp._llamar_nim('texto'), resultado())
        self.assertEqual(post.call_count, 3)

    @patch('document_processor.requests.post')
    def test_truncado_no_es_exito(self, post):
        post.return_value = respuesta_http(finish='length')
        with self.assertRaises(dp.ErrorNIM):
            dp._llamar_nim('texto')

    @patch('document_processor.pytesseract.get_languages')
    def test_tesseract_ausente(self, languages):
        languages.side_effect = pytesseract.TesseractNotFoundError()
        with self.assertLogs(dp.logger, level='WARNING'):
            self.assertIsNone(dp._idioma_ocr())

    @patch('document_processor.pytesseract.get_languages', return_value=['eng', 'osd'])
    def test_fallback_idioma(self, languages):
        with self.assertLogs(dp.logger, level='WARNING'):
            self.assertEqual(dp._idioma_ocr(), 'eng')

    @patch('document_processor.requests.post')
    def test_pipeline_pdf_y_ocr_reales(self, post):
        if dp._idioma_ocr() is None:
            self.skipTest('Instala Tesseract para verificar OCR real.')
        post.return_value = respuesta_http()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with fitz.open(path) as pdf:
                self.assertEqual(pdf[1].get_text().strip(), '')
            datos, texto = dp.procesar_pdf(path)
        self.assertEqual(datos, resultado())
        self.assertEqual(post.call_count, 2)
        self.assertIn('Concierto', texto.split('--- Página 2 ---')[1])
        second = post.call_args_list[1].kwargs['json']['messages'][1]['content'][0]['text']
        self.assertIn('Concierto', second)

    @patch('document_processor._idioma_ocr', return_value=None)
    @patch('document_processor.analizar_pagina_con_nim')
    def test_fallo_pagina_continua(self, analizar, idioma):
        analizar.side_effect = [dp.ErrorNIM('fallo simulado'), resultado()]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with self.assertLogs(dp.logger, level='WARNING'):
                datos, texto = dp.procesar_pdf(path)
        self.assertEqual(datos, resultado())
        self.assertIn('Página 2', texto)
        self.assertEqual(analizar.call_count, 2)

    def test_pdf_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            dp.procesar_pdf('/no-existe/programa.pdf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
