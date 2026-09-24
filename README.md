# Programa de eventos: extracción de PDF

Repositorio compartido: https://github.com/olivergarciamurillo22/proeycto-curso-ia

- **Oliver (persona 1):** PDF → imágenes → OCR + texto nativo → NVIDIA NIM → JSON.
- **Pedro (persona 2):** consumo del JSON, web, filtros, chatbot y embeddings/RAG, por separado.

Este módulo implementa exclusivamente la parte de Oliver.

## Instalación

Python 3.10 o superior. En una carpeta cuyo nombre no contenga `:`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

La carpeta local actual termina en `:` y Python no permite crear un venv en esa ruta.
Se ha preparado el entorno en `~/.venvs/agenda-almeria-ai`:

```bash
source ~/.venvs/agenda-almeria-ai/bin/activate
python -m pip install -r requirements.txt
```

OCR real, macOS:

```bash
brew install tesseract tesseract-lang
```

Ubuntu/Colab:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
```

Comprueba `tesseract --list-langs`. Se usa `spa`, después `eng` u otro idioma
disponible. Sin Tesseract se emite un warning y se aprovecha el texto nativo.

## Configuración de NVIDIA

El `.env` local se creó inicialmente con la clave vacía; cualquier valor añadido
posteriormente se conserva y no se utiliza en los tests ni en la demo. En otra máquina, copia
`.env.example` a `.env` **solo si no existe**. Las variables del proceso prevalecen
sobre `.env`.

```dotenv
NVIDIA_API_KEY=
NVIDIA_NIM_URL=https://integrate.api.nvidia.com/v1/chat/completions
NVIDIA_MODEL=google/gemma-4-31b-it
```

Modelo configurado el 24/09/2026: `google/gemma-4-31b-it`. Se ha comprobado su
presencia en `GET https://integrate.api.nvidia.com/v1/models` (HTTP 200, sin clave).
La [referencia oficial del endpoint](https://docs.api.nvidia.com/nim/reference/google-gemma-4-31b-it-infer)
documenta `/v1/chat/completions`, imágenes `image_url` con data URI, texto y los
roles `user`/`assistant`. Para este modelo se incorporan las instrucciones en el
mensaje de usuario y se desactiva el razonamiento. La [ficha oficial del modelo](https://docs.api.nvidia.com/nim/reference/google-gemma-4-31b-it)
recomienda colocar la imagen antes del texto, como hace este módulo.
Esto confirma el contrato documentado y su presencia en el catálogo, **no** una
inferencia real ni los permisos/cuota de tu cuenta. `.env.example` mantiene el
modelo vacío para que siga siendo una plantilla configurable.

No incluyas claves en código, comandos compartidos ni Git. `.env` está ignorado.
La API recibe el contenido de cada página del PDF. Se intenta una petición con
imagen JPEG base64 y texto, y después una petición solo de texto si falla.
El formato sigue la [documentación de entrada multimodal de NVIDIA](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/multimodal-input.html);
la compatibilidad concreta depende del modelo y endpoint configurados.

## Uso e integración con Pedro

```python
from document_processor import procesar_pdf

datos, texto = procesar_pdf('data/programa.pdf')
```

`datos` es un diccionario con exactamente las claves `documento` y `eventos`:

```json
{
  "documento": {"titulo": null, "descripcion": null},
  "eventos": [
    {
      "nombre": "Concierto de ejemplo",
      "fecha": null,
      "hora": null,
      "lugar": null,
      "descripcion": null,
      "categoria": null
    }
  ]
}
```

Los datos ausentes son `None` en Python y `null` al exportar JSON. Todos los demás
valores son strings, con fechas y horas literales; Pedro debe tolerar nulos.
`texto` contiene el texto híbrido completo con separadores `--- Página N ---`.
No incluye respuestas generadas por el modelo.

Para generar los dos archivos de entrega:

```bash
python document_processor.py data/programa.pdf
```

Produce `output/eventos.json` y `output/texto_extraido.txt` (ignorados por Git).
`--output otra_carpeta` permite elegir destino. Una ejecución posterior reemplaza
estos dos archivos de salida.

## Comportamiento y límites

- Cada página se renderiza a escala 2 y pasa por OCR, incluso si tiene texto nativo.
- Se prioriza texto nativo suficiente y se complementa con líneas OCR no repetidas.
- NIM tiene timeout, hasta tres intentos ante problemas de conexión, 429 o 5xx,
  validación estricta y rechazo de respuestas truncadas. El parser tolera Markdown.
- Se eliminan espacios exteriores y los campos vacíos se convierten en `null`.
  Se conservan los textos interiores, acentos y nombres propios; no se convierten
  números u objetos incorrectos en datos inventados.
- Se conserva el primer título/descripción no vacío. Solo se deduplican entre
  páginas eventos idénticos en todos los campos con nombre, fecha, hora y lugar.
- Los fallos de una página se registran como warnings y no detienen las demás.
  **Los resultados pueden ser parciales.** Si falla todo NIM, se devuelve una lista
  vacía con warning; esto no demuestra que el PDF no contenga eventos.
- Clave/modelo ausentes, PDF inexistente, ilegible o protegido causan excepción.
- No se infieren fechas a partir de otras páginas ni se corrigen posibles errores
  del modelo; revisar el resultado contra el programa antes de utilizarlo.

## Verificación

```bash
python -m unittest -v test_processor
```

Las pruebas generan un PDF de dos páginas: una digital y otra con imagen escaneada.
Se ejecutan renderizado y OCR reales; las respuestas HTTP de NIM son simuladas.
También verifican parser, esquema, reintentos, fallback, deduplicación y fallos
aislados. No necesitan clave ni consumen API. La prueba OCR se omite explícitamente
si Tesseract no está instalado.

La validación real contra NVIDIA requiere `.env` y un PDF propio; las pruebas
simuladas no acreditan compatibilidad ni calidad de un modelo concreto.

Para reproducir la demo completa sin credenciales:

```bash
python test_processor.py --demo
```

Genera `output/programa_sintetico.pdf`, procesa sus páginas con OCR real y simula
solo HTTP NVIDIA. Escribe `output/eventos.json`, `output/texto_extraido.txt` y
`output/LEEME_SIMULACION.txt`, que identifica la procedencia ficticia de los datos.
No modifica `.env`, no usa la red y no coloca datos ficticios en `data/programa.pdf`.
La demo reemplaza esas salidas: úsala antes de procesar el PDF real o guarda las
salidas reales en otro directorio. Los tests bloquean por defecto las llamadas HTTP
no simuladas explícitamente.

## Prueba real pendiente

1. Pega la clave en `NVIDIA_API_KEY` dentro de `.env`.
2. Copia el PDF real a `data/programa.pdf` (la carpeta está preparada).
3. Ejecuta:

```bash
source ~/.venvs/agenda-almeria-ai/bin/activate
python document_processor.py data/programa.pdf
```

También puedes omitir el argumento: el PDF predeterminado es `data/programa.pdf`.
Comprueba los warnings y revisa las dos salidas; si la cuenta no permite acceder
al modelo configurado, selecciona otro modelo multimodal habilitado en NVIDIA.
