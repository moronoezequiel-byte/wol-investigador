# WOL Investigador

Proyecto independiente para investigar preguntas en Watchtower Online Library (WOL). Interpreta preguntas naturales, consulta WOL, lee páginas originales y evalúa pertinencia.

## Estructura
- `main.py`: API FastAPI y motor de investigación.
- `static/`: aplicación web adaptable e instalable (PWA) para Android y iPhone.
- `extension/`: extensión Chrome Manifest V3.

## Desarrollo
`pip install -r requirements.txt`
`export OPENAI_API_KEY="tu-clave"`
`uvicorn main:app --reload`

Configurar `OPENAI_API_KEY` como variable secreta en el hosting. No guardar claves en el repositorio. Opcional: `OPENAI_MODEL`.

## Publicación
Build: `pip install -r requirements.txt`
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Instalación
- Android: abrir la URL publicada en Chrome y elegir instalar o agregar a pantalla principal.
- iPhone: abrir la URL en Safari, compartir y elegir Agregar a pantalla de inicio.
- Chrome escritorio: cargar la carpeta `extension` desde `chrome://extensions` con modo desarrollador. Actualizar la URL del popup con la dirección definitiva del hosting.

## Principios
- Fuentes y enlaces solo de wol.jw.org.
- No inventar títulos, artículos ni URLs.
- No generar respuestas doctrinales: mostrar artículos originales y enlaces.
- Leer contenido antes de evaluar pertinencia.
- No considerar coincidencias léxicas como prueba suficiente.

Estado: versión inicial 0.1.0. La extracción HTML y la calidad de búsqueda requieren pruebas reales antes de considerarse validadas.
