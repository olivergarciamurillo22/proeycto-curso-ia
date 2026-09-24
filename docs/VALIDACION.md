# Validación de la integración · 24/09/2026

## Resultado

- Suite local: **46 tests PASS, 0 FAIL**, sin llamadas de inferencia reales.
- `pip check`: sin incompatibilidades de dependencias.
- NVIDIA real: dos páginas procesadas con `meta/llama-3.2-11b-vision-instruct`
  en `https://integrate.api.nvidia.com/v1/chat/completions`.
- Ambas páginas utilizaron imagen + texto; no necesitaron fallback.
- OCR español real con Tesseract 5.5.2 y PyMuPDF 1.28.2.
- Chat real: recuperó la hora `20:00` al preguntar por el concierto de prueba.
- Embeddings reales: modelo multilingüe MiniLM, vector de 384 dimensiones y
  recuperación con modo `rag`.
- Streamlit AppTest: arranque, filtros sin resultados y selector interactivo
  de Ajustes probados. No se ha completado una revisión visual en navegador:
  la herramienta de navegador no pudo iniciar su sesión en este entorno.

## Alcance de la prueba de extracción

Se generó un PDF SINTÉTICO con el mismo evento en una página digital y una página
escaneada, para probar extracción nativa, OCR y deduplicación conjuntamente.
Resultado real: un evento «Concierto de prueba», fecha «24 septiembre», hora «20:00»
y lugar «Plaza Mayor». La categoría ausente se mantuvo en null.
Esto valida la conexión y el recorrido de datos; no demuestra todavía precisión
en un programa real, maquetación compleja o documentos de muchas páginas.

La primera prueba de dos páginas con Llama duró 6,935 segundos. Es una observación
puntual, no una promesa de rendimiento. Gemma 4 agotó los tiempos de espera y
Gemma 3 12B respondió HTTP 404; el modelo local se sustituyó por Llama tras probarlo.

## Reproducción

```bash
python -m unittest discover -v
python verificar_nvidia.py
python -m streamlit run app.py
```

El segundo comando consume NVIDIA con la clave del servidor y guarda resultados
en `output/verificacion_nvidia/`, fuera de Git. No se registra ni comparte la clave.
Para completar la validación sobre el programa del proyecto:

```bash
python verificar_nvidia.py --pdf data/programa.pdf
```

El PDF real aún no está en `data/`. La API pública original sigue devolviendo
exactamente `(datos, texto)`. La nueva variante con informe permite a la interfaz
distinguir éxito, resultado parcial y fallo sin inferirlo de una lista de eventos vacía.
