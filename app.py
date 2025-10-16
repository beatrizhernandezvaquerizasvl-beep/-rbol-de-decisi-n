import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(page_title="Árbol de Decisión — Universal", layout="wide")

st.title("🌳 Árbol de Decisión — Universal (Clasificación y Regresión)")
st.write("Sube un CSV o usa un dataset de ejemplo, elige la variable objetivo y ajusta los hiperparámetros.")

# ----------- Datos -----------
st.sidebar.header("1) Datos")
data_source = st.sidebar.radio("Fuente de datos", ["Subir CSV", "Ejemplos (sklearn)"])

df = None
target = None

if data_source == "Subir CSV":
    file = st.sidebar.file_uploader("Sube un CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=";")
else:
    from sklearn.datasets import load_iris, load_wine, fetch_california_housing
    ex = st.sidebar.selectbox("Dataset de ejemplo", ["Iris (clasificación)", "Wine (clasificación)", "California Housing (regresión)"])
    if ex == "Iris (clasificación)":
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        target = "target"
    elif ex == "Wine (clasificación)":
        wine = load_wine(as_frame=True)
        df = wine.frame.copy()
        target = "target"
    else:
        cal = fetch_california_housing(as_frame=True)
        df = cal.frame.copy()
        df["target"] = cal.target
        target = "target"

if df is not None:
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Detectar columnas numéricas / categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    target = st.sidebar.selectbox("Variable objetivo", options=[c for c in df.columns],
                                  index=(df.columns.tolist().index(target) if target in df.columns else 0))

    # Tipo de tarea según objetivo
    task = st.sidebar.selectbox("Tipo de problema", ["Clasificación", "Regresión"])

    # ----------- Preprocesado simple -----------
    st.sidebar.header("2) Partición")
    test_size = st.sidebar.slider("Tamaño de test (%)", 10, 50, 20, step=5) / 100.0
    random_state = st.sidebar.number_input("random_state", 0, 9999, 42, step=1)

    # Seleccionar variables de entrada
    X_cols = [c for c in df.columns if c != target]
    st.sidebar.write("Columnas usadas como X:", X_cols)

    # One-hot encoding para categóricas
    X = df[X_cols].copy()
    y = df[target].copy()

    if task == "Clasificación":
        y = y.astype(str)

    X = pd.get_dummies(X, drop_first=False)

    # Eliminar filas con NaN simples
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # ----------- Hiperparámetros -----------
    st.sidebar.header("3) Hiperparámetros")
    max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 6, step=1)
    max_depth = None if max_depth == 0 else max_depth
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 2, step=1)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 1, step=1)
    splitter = st.sidebar.selectbox("splitter", ["best", "random"])

    if task == "Clasificación":
        criterion = st.sidebar.selectbox("criterion", ["gini", "entropy", "log_loss"])
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    else:
        criterion_reg = st.sidebar.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
        model = DecisionTreeRegressor(
            criterion=criterion_reg,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    # ----------- Entrenar -----------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task=="Clasificación" else None)

    train_btn = st.sidebar.button("🚀 Entrenar")
    if train_btn:
        model.fit(X_train, y_train)

        st.success("Modelo entrenado.")
        col1, col2 = st.columns(2)

        # ----------- Métricas -----------
        with col1:
            st.subheader("Evaluación")
            y_pred = model.predict(X_test)

            if task == "Clasificación":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                st.write(f"**Accuracy**: {acc:.3f}  \n**Precision (weighted)**: {prec:.3f}  \n**Recall (weighted)**: {rec:.3f}  \n**F1 (weighted)**: {f1:.3f}")
                cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
                st.write("Matriz de confusión (ordenada por etiqueta):")
                st.dataframe(pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())))
                st.text("Classification report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MAE**: {mae:.3f}  \n**RMSE**: {rmse:.3f}  \n**R²**: {r2:.3f}")

        # ----------- Importancias -----------
        with col2:
            st.subheader("Importancia de variables")
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.dataframe(importances.to_frame("importancia"))
            fig = plt.figure(figsize=(6,4))
            importances.plot(kind="bar")
            plt.title("Importancia de variables")
            plt.xlabel("Variable")
            plt.ylabel("Importancia (Gini)")
            plt.tight_layout()
            st.pyplot(fig)

        # ----------- Visualizar árbol -----------
        st.subheader("Visualización del árbol")
        max_depth_plot = st.slider("Profundidad máxima a dibujar (para legibilidad)", 1, 10, 4, step=1)
        fig2 = plt.figure(figsize=(18, 10))
        plot_tree(
            model,
            feature_names=X.columns,
            filled=True,
            rounded=True,
            max_depth=max_depth_plot
        )
        plt.tight_layout()
        st.pyplot(fig2)

        # ----------- Descargar modelo entrenado -----------
        st.subheader("Descargar modelo entrenado")
        buf = io.BytesIO()
        pickle.dump(model, buf)
        st.download_button("💾 Descargar .pkl", data=buf.getvalue(), file_name="decision_tree_model.pkl")

        # ----------- Predicción con entrada manual -----------
        st.subheader("Probar una predicción con entrada manual")
        sample = {}
        for col in X.columns:
            try:
                col_min = float(X[col].min())
                col_max = float(X[col].max())
                default = float(X[col].median())
            except Exception:
                col_min, col_max, default = 0.0, 1.0, 0.0
            val = st.number_input(f"{col}", value=default)
            sample[col] = val

        if st.button("🔮 Predecir muestra"):
            x_new = pd.DataFrame([sample])
            pred = model.predict(x_new)[0]
            st.info(f"Predicción: **{pred}**")

else:
    st.info("⬅️ Sube un CSV o elige un dataset de ejemplo para comenzar.")