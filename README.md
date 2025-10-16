# 🌳 Streamlit — Árbol de Decisión Universal (Clasificación y Regresión)

App en Streamlit para entrenar, evaluar y visualizar árboles de decisión a partir de un CSV o datasets de `sklearn`.
Incluye importancias de variables, matriz de confusión / métricas de regresión, visualización del árbol y descarga del modelo.

## Uso local
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud (desde GitHub)
1. Crea un repositorio en GitHub (p. ej., `streamlit-decision-tree-app`).
2. Sube estos archivos: `app.py`, `requirements.txt`, `README.md`.
3. Ve a https://share.streamlit.io/ e inicia sesión con tu GitHub.
4. Pulsa **New app** → selecciona tu repo, rama (main) y archivo `app.py` → **Deploy**.

## CSV esperado
- Selecciona la variable objetivo (columna a predecir).
- Las categóricas se codifican con one-hot encoding automáticamente.
- Se eliminan filas con NaN.