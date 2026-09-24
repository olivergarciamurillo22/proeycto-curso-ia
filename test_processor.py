"""Pruebas locales: PDF/OCR reales y HTTP NIM simulado, sin consumir API."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pymupdf as fitz
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
        # Cualquier petición no simulada explícitamente es un fallo de la prueba.
        red = patch('document_processor.requests.post', side_effect=AssertionError('Red no permitida en tests'))
        red.start()
        self.addCleanup(red.stop)

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
        self.assertTrue(first[0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
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
        second = post.call_args_list[1].kwargs['json']['messages'][1]['content'][1]['text']
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

    def test_normalizacion_espacios_y_claves_extra(self):
        data = resultado('  José {El músico}  ')
        data['eventos'][0]['descripcion'] = '  '
        data['eventos'][0]['extra'] = 'descartar'
        normal = dp.parsear_json_nim(json.dumps(data))
        self.assertEqual(normal['eventos'][0]['nombre'], 'José {El músico}')
        self.assertIsNone(normal['eventos'][0]['descripcion'])
        self.assertNotIn('extra', normal['eventos'][0])

    def test_parser_texto_con_llaves_antes_y_despues(self):
        raw = json.dumps(resultado('Concierto "Sol" {edición 2}'))
        text = 'Nota {documento} {"documento": "referencia"}\n```json\n' + raw + '\n```\nFin {x}'
        self.assertEqual(dp.parsear_json_nim(text), resultado('Concierto "Sol" {edición 2}'))

    def test_no_eliminar_lineas_por_subcadenas(self):
        text = dp.combinar_textos('Sala 1', 'Sala 10')
        self.assertIn('Sala 1', text.splitlines())
        self.assertIn('Sala 10', text.splitlines())

    def test_prioridad_texto_nativo_suficiente(self):
        native = 'Texto nativo largo y fiable del programa cultural. ' * 3
        self.assertTrue(dp.combinar_textos(native, 'Entrada gratuita').startswith(native.strip()))

    def test_combinacion_metadatos_y_eventos(self):
        first, second = resultado('Uno'), resultado('Dos')
        first['documento'] = {'titulo': None, 'descripcion': 'Descripción primera'}
        second['documento'] = {'titulo': 'Título segunda', 'descripcion': 'Descripción segunda'}
        data = dp._combinar_resultados([first, second])
        self.assertEqual(data['documento'], {'titulo': 'Título segunda', 'descripcion': 'Descripción primera'})
        self.assertEqual([e['nombre'] for e in data['eventos']], ['Uno', 'Dos'])

    def test_no_deduplica_diferente_descripcion(self):
        first, second = resultado(), resultado()
        second['eventos'][0]['descripcion'] = 'Otra actuación'
        self.assertEqual(len(dp._combinar_resultados([first, second])['eventos']), 2)

    def test_no_deduplica_dentro_de_pagina(self):
        data = resultado()
        data['eventos'] *= 2
        self.assertEqual(len(dp._combinar_resultados([data])['eventos']), 2)

    @patch('document_processor.requests.post')
    def test_respuesta_http_malformada(self, post):
        for body in (None, [], {'choices': []}, {'choices': [None]},
                     {'choices': [{'message': {'content': None}}]}):
            post.return_value = respuesta_http()
            post.return_value.json.return_value = body
            with self.subTest(body=body), self.assertRaises(dp.ErrorNIM):
                dp._llamar_nim('texto')

    @patch('document_processor.time.sleep')
    @patch('document_processor.requests.post')
    def test_limite_reintentos_sin_filtrar_secretos(self, post, sleep):
        post.side_effect = requests.ConnectionError('cabecera secreta clave-ficticia-tests')
        with self.assertRaises(dp.ErrorNIM) as error:
            dp._llamar_nim('texto')
        self.assertNotIn('clave-ficticia-tests', str(error.exception))
        self.assertEqual(post.call_count, 3)

    @patch('document_processor.requests.post')
    def test_http_401_no_reintenta(self, post):
        post.return_value = respuesta_http(401)
        with self.assertRaises(dp.ErrorNIM):
            dp._llamar_nim('texto')
        self.assertEqual(post.call_count, 1)

    @patch('document_processor.requests.post')
    def test_peticion_modelo_gemma(self, post):
        post.return_value = respuesta_http()
        with patch.dict(os.environ, {'NVIDIA_MODEL': 'google/gemma-4-31b-it'}):
            dp.analizar_pagina_con_nim(1, dp.Image.new('RGB', (20, 20)), 'Concierto', '')
        payload = post.call_args.kwargs['json']
        self.assertEqual([m['role'] for m in payload['messages']], ['user'])
        self.assertEqual(payload['messages'][0]['content'][0]['type'], 'image_url')
        self.assertIn(dp.SYSTEM_PROMPT, payload['messages'][0]['content'][1]['text'])
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
        self.assertFalse(post.call_args.kwargs['allow_redirects'])

    @patch('document_processor.requests.post')
    def test_texto_sin_imagen(self, post):
        post.return_value = respuesta_http()
        self.assertEqual(dp.analizar_pagina_con_nim(1, None, 'Concierto', ''), resultado())
        self.assertIsInstance(post.call_args.kwargs['json']['messages'][1]['content'], str)

    def test_pagina_sin_contenido(self):
        with self.assertRaises(dp.ErrorNIM):
            dp.analizar_pagina_con_nim(1, None, '', '')

    def test_texto_nativo_renderizado_y_ocr_espanol(self):
        if dp._idioma_ocr() != 'spa':
            self.skipTest('Se requiere Tesseract con spa para verificar OCR español.')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with fitz.open(path) as pdf:
                self.assertIn('Concierto de prueba', pdf[0].get_text('text'))
                pix = pdf[1].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB, alpha=False)
                self.assertEqual((pix.width, pix.height), (1190, 1684))
                with dp.Image.frombytes('RGB', (pix.width, pix.height), pix.samples) as imagen:
                    text = pytesseract.image_to_string(imagen, lang='spa', timeout=60)
                for expected in ('Concierto', '24 septiembre', '20:00', 'Plaza Mayor'):
                    self.assertIn(expected, text)

    @patch('document_processor._idioma_ocr', return_value='spa')
    @patch('document_processor.pytesseract.image_to_string', side_effect=RuntimeError('timeout'))
    @patch('document_processor.analizar_pagina_con_nim', return_value=resultado())
    def test_fallo_ocr_conserva_nativo(self, analizar, ocr, idioma):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with self.assertLogs(dp.logger, level='WARNING'):
                _, texto = dp.procesar_pdf(path)
        self.assertIn('Concierto de prueba', texto)
        self.assertEqual(analizar.call_count, 2)

    @patch('document_processor._idioma_ocr', return_value=None)
    @patch('document_processor.fitz.Page.get_pixmap', side_effect=RuntimeError('render'))
    @patch('document_processor.analizar_pagina_con_nim', return_value=resultado())
    def test_fallo_render_conserva_nativo(self, analizar, pixmap, idioma):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            with fitz.open() as doc:
                doc.new_page().insert_text((50, 50), 'Concierto de prueba')
                doc.save(path)
            with self.assertLogs(dp.logger, level='WARNING'):
                _, texto = dp.procesar_pdf(path)
        self.assertIn('Concierto de prueba', texto)
        self.assertIsNone(analizar.call_args.args[1])

    @patch('document_processor._idioma_ocr', return_value=None)
    @patch('document_processor.analizar_pagina_con_nim', side_effect=dp.ErrorNIM('fallo'))
    def test_fallo_total_nim_explicito(self, analizar, idioma):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with self.assertLogs(dp.logger, level='WARNING') as logs:
                datos, texto = dp.procesar_pdf(path)
        self.assertEqual(datos['eventos'], [])
        self.assertIn('Concierto de prueba', texto)
        self.assertTrue(any('Ninguna página' in line for line in logs.output))

    def test_pdf_corrupto(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corrupto.pdf'
            path.write_bytes(b'%PDF-1.7\nbroken')
            with self.assertRaises(fitz.FileDataError):
                dp.procesar_pdf(path)

    def test_pdf_protegido(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'protegido.pdf'
            with fitz.open() as doc:
                doc.new_page()
                doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw='test', owner_pw='owner')
            with self.assertRaises(ValueError):
                dp.procesar_pdf(path)

    @patch('document_processor._idioma_ocr', return_value=None)
    @patch('document_processor.analizar_pagina_con_nim', return_value=resultado())
    def test_pagina_ilegible_continua(self, analizar, idioma):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            original = fitz.Document.load_page

            def cargar(doc, indice):
                if indice == 0:
                    raise RuntimeError('Página corrupta simulada')
                return original(doc, indice)

            with patch.object(fitz.Document, 'load_page', cargar), self.assertLogs(dp.logger, level='WARNING'):
                datos, texto = dp.procesar_pdf(path)
        self.assertEqual(datos, resultado())
        self.assertIn('[No se pudo leer la página]', texto)
        self.assertIn('Página 2', texto)
        self.assertEqual(analizar.call_count, 1)

    @patch('document_processor._idioma_ocr', return_value='spa')
    @patch('document_processor.fitz.Page.get_text', side_effect=RuntimeError('texto nativo fallido'))
    @patch('document_processor.requests.post')
    def test_fallo_texto_nativo_usa_ocr(self, post, get_text, idioma):
        if 'spa' not in pytesseract.get_languages(config=''):
            self.skipTest('OCR español requerido')
        post.return_value = respuesta_http()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            with self.assertLogs(dp.logger, level='WARNING'):
                _, texto = dp.procesar_pdf(path)
        self.assertEqual(texto.count('Concierto de prueba'), 2)

    @patch('document_processor.requests.post')
    def test_gemma_fallback_solo_texto(self, post):
        post.side_effect = [respuesta_http(422), respuesta_http()]
        with patch.dict(os.environ, {'NVIDIA_MODEL': 'google/gemma-4-31b-it'}), self.assertLogs(dp.logger):
            dp.analizar_pagina_con_nim(1, dp.Image.new('RGB', (20, 20)), 'Concierto', '')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['messages'][0]['role'], 'user')
        self.assertIn(dp.SYSTEM_PROMPT, payload['messages'][0]['content'])
        self.assertIn('Concierto', payload['messages'][0]['content'])

    @patch('document_processor.requests.post')
    def test_cli_exporta_json_y_texto(self, post):
        post.return_value = respuesta_http()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programa.pdf'
            crear_pdf(path)
            out = Path(folder) / 'output'
            with patch.object(sys, 'argv', ['document_processor.py', str(path), '--output', str(out)]):
                dp.main()
            self.assertEqual(json.loads((out / 'eventos.json').read_text(encoding='utf-8')), resultado())
            text = (out / 'texto_extraido.txt').read_text(encoding='utf-8')
            self.assertIn('Página 1', text)
            self.assertIn('Página 2', text)


def demo_sin_credenciales():
    """Ejecuta OCR real y NIM simulado; nunca consulta la red ni modifica .env."""
    out = Path('output')
    out.mkdir(exist_ok=True)
    pdf = out / 'programa_sintetico.pdf'
    crear_pdf(pdf)
    with patch('document_processor._configuracion', return_value=(
            '', 'modelo-simulado', 'https://integrate.api.nvidia.com/v1/chat/completions')), \
            patch('document_processor.requests.post', return_value=respuesta_http()), \
            patch.object(sys, 'argv', ['document_processor.py', str(pdf)]):
        dp.main()
    (out / 'LEEME_SIMULACION.txt').write_text(
        'Estos archivos proceden de un PDF SINTÉTICO y respuestas NVIDIA SIMULADAS.\n'
        'El renderizado y el OCR son reales. No acreditan una llamada real a NVIDIA.\n'
        'La próxima ejecución de document_processor.py reemplaza las salidas.\n', encoding='utf-8')
    print('Demo terminada: OCR real, NIM simulado; output/eventos.json y output/texto_extraido.txt.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--demo']:
        demo_sin_credenciales()
    else:
        unittest.main(verbosity=2)
