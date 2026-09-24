# Pedro · dirección visual e integración

Oliver pide una web muy cuidada, innovadora y con una sección de ajustes que muestre
los modelos y la tecnología realmente utilizados. Esta es la propuesta concreta
para desarrollar sobre tu `app.py`, conservando filtros, carga de PDF y chat.

## Dirección: Almería, después del sol

Una agenda cultural con personalidad editorial: fondo marfil `#F7F4EC`, tinta
azul noche `#142D3C`, acento terracota `#B7492D` y verde profundo `#176153` para estados.
Titular grande «Tu próximo plan empieza aquí», una tipografía serif para titulares
y sans-serif para controles. Mucho aire, jerarquía clara, bordes sutiles y esquinas
de 16 px. Evitar convertir cada texto en una tarjeta o saturar de emojis.

Cabecera: marca «ALMERÍA / agenda viva», navegación Explorar · Mi agenda · Asistente · Ajustes,
y un indicador discreto del origen del programa: demo, PDF cargado o extracción parcial.
En móvil, una columna, navegación compacta y filtros plegables; ningún control depende de hover.

## Pantalla Explorar

- Hero editorial asimétrico: titular y descripción a la izquierda; a la derecha,
  tarjeta destacada de un evento realmente disponible, sin inventar popularidad.
- Filtros de fecha, categoría y lugar en una franja compacta con «Limpiar filtros».
  Contador de resultados visible y estado vacío con acción para quitar filtros.
- Tarjetas con hora protagonista, categoría discreta, nombre, lugar y descripción.
  Alternar «Tarjetas / Programa por día» preservando los filtros.
- Favoritos en sesión y pestaña «Mi agenda»; explicar si no persisten al cerrar.
  No normalizar ni ordenar fechas inciertas como si fueran fechas completas.

## Innovación útil, con datos verificables

- «Encuentra mi plan»: sugerencias de preguntas construidas con categorías reales
  del programa. Pulsarlas completa o envía una pregunta reconocible al asistente.
- Respuesta del chat con eventos que fundamentan la respuesta, expandibles.
  Mantener el vínculo con las fuentes también al volver a abrir la conversación.
- «Cómo se leyó este programa»: abrir el informe por páginas para distinguir OCR,
  imagen+texto, fallback textual y páginas fallidas. Nunca inventar puntuaciones de confianza.
- «Mi agenda»: guardar eventos y compararlos; advertir coincidencias horarias
  solo si fecha/hora son interpretables. Exportación de calendario solo con fechas completas.
- Microinteracciones de 150–200 ms en filtros y favoritos, con movimiento reducido.
  Estados de carga reales, sin porcentajes ficticios ni etiquetas de conexión inventadas.

## Ajustes y tecnología: base funcional ya integrada

`ajustes.py` implementa el panel y `app.py` lo muestra en la tercera pestaña.
Puedes rediseñarlo manteniendo los datos y comportamiento:

1. Tres tarjetas: modelo de extracción, modelo de conversación y embeddings.
2. Recorrido interactivo PDF → OCR → IA → Agenda: selector con explicación de cada etapa.
3. Informe: páginas procesadas, eventos, duración, modelo utilizado y modo por página.
4. Tecnología: versiones instaladas, motor/idioma OCR y estado real de embeddings.
5. Privacidad: qué se procesa en el servidor y qué se envía a NVIDIA.
6. Descarga de diagnóstico sin claves, tokens ni volcado de `.env`.

La configuración actual NO demuestra que un modelo se haya usado: el modelo de la
última extracción está en `informe['modelo']`. Si no hay informe, mostrar «Sin extracción registrada».
El chat puede usar `NIM_MODEL`; extracción usa `NVIDIA_MODEL`. Los embeddings locales
son `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Contrato de integración

```python
from document_processor import procesar_pdf_con_informe, obtener_info_tecnica

datos, texto, informe = procesar_pdf_con_informe(ruta_pdf)
tecnologia = obtener_info_tecnica()  # sin secretos
```

`procesar_pdf(ruta_pdf)` sigue devolviendo exactamente `(datos, texto)`.
La CLI escribe `output/eventos.json`, `output/texto_extraido.txt` y `output/tecnologia.json`.
El informe tiene estado `completo`, `parcial` o `fallido`; completo describe ejecución,
no exactitud. No reemplazar una agenda anterior cuando falla toda la extracción.
No reconstruir la configuración leyendo `.env` desde la interfaz: usar la función segura.

## Criterios para dar por terminada la interfaz

- Funciona a 390 px y en escritorio, sin desplazamiento horizontal.
- Filtros y favoritos funcionan con valores `null`; no aparecen botones decorativos.
- Se distingue demo/programación real y extracción parcial/completa.
- Chat conserva las fuentes al cambiar de pestaña.
- Ajustes nunca muestran la API key ni afirman «conectado» solo porque existe una clave.
- Teclado, foco visible, contraste legible y movimiento reducido.
- Conservar `python -m unittest discover -v` en verde.
- Antes de publicar, `git pull --rebase`; no subir `.env` ni `output/`.
