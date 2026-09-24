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

Copia `.env.example` a `.env` **solo si no existe** y completa la clave y el
identificador exacto de un modelo habilitado en tu cuenta NVIDIA. No se presupone
que un modelo soporte imágenes. Las variables del proceso prevalecen sobre `.env`.

```dotenv
NVIDIA_API_KEY=
NVIDIA_NIM_URL=https://integrate.api.nvidia.com/v1/chat/completions
NVIDIA_MODEL=
```

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

Produce `output/eventos.json` y `output/texto.txt` (ignorados por Git).
`--output otra_carpeta` permite elegir destino. Una ejecución posterior reemplaza
estos dos archivos de salida.

## Comportamiento y límites

- Cada página se renderiza a escala 2 y pasa por OCR, incluso si tiene texto nativo.
- Se prioriza texto nativo suficiente y se complementa con líneas OCR no repetidas.
- NIM tiene timeout, hasta tres intentos ante problemas de conexión, 429 o 5xx,
  validación estricta y rechazo de respuestas truncadas. El parser tolera Markdown.
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
