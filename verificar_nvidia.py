"""Prueba REAL, optativa y con consumo de API. Nunca se ejecuta con unittest."""
import argparse
import json
from pathlib import Path

import pymupdf as fitz

from document_processor import procesar_pdf_con_informe, obtener_info_tecnica


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', help='PDF real opcional; por defecto crea un documento sintético de prueba.')
    args = parser.parse_args()
    out = Path('output/verificacion_nvidia')
    out.mkdir(parents=True, exist_ok=True)
    ruta = Path(args.pdf) if args.pdf else out / 'programa_sintetico.pdf'
    if not args.pdf:
        with fitz.open() as doc:
            pagina = doc.new_page()
            pagina.insert_text((60, 80), 'Programa de prueba\nConcierto de prueba\n'
                                '24 septiembre - 20:00\nPlaza Mayor', fontsize=20)
            imagen = pagina.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes('png')
            doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=imagen)
            doc.save(ruta)
    datos, texto, informe = procesar_pdf_con_informe(ruta)
    informe['tipo_documento'] = 'aportado' if args.pdf else 'sintetico'
    informe['api'] = 'real'
    tecnologia = obtener_info_tecnica()
    tecnologia['ultima_ejecucion'] = informe
    for nombre, contenido in [('eventos.json', datos), ('tecnologia.json', tecnologia)]:
        (out / nombre).write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'texto_extraido.txt').write_text(texto, encoding='utf-8')
    if informe['estado'] != 'completo':
        raise SystemExit('FAIL: extracción parcial o fallida. Consulta output/verificacion_nvidia/tecnologia.json.')
    if not args.pdf:
        eventos = datos['eventos']
        if not eventos or not any(e['nombre'] == 'Concierto de prueba' and e['hora'] == '20:00'
                                  and e['lugar'] == 'Plaza Mayor' for e in eventos):
            raise SystemExit('FAIL: el modelo no recuperó los datos conocidos del documento de prueba.')
        if not all(p['caracteres_ocr'] > 0 for p in informe['paginas']):
            raise SystemExit('FAIL: OCR no disponible para alguna página de prueba.')
    print(f"PASS: NVIDIA real, {informe['paginas_correctas']} páginas, "
          f"{len(datos['eventos'])} eventos, modelo {informe['modelo']}.")
    print('Salidas: output/verificacion_nvidia/. Documento: ' + informe['tipo_documento'])


if __name__ == '__main__':
    main()
