"""Aplicación Streamlit: análisis de regresión lineal simple y múltiple.

Ejecutar con:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import utils

st.set_page_config(
    page_title="Análisis de Regresión Lineal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Análisis de Regresión Lineal (Simple y Múltiple)")
st.markdown(
    """
    Sube un archivo **CSV o Excel**, elige la variable dependiente (**Y**) y las
    variables independientes (**X**), y la aplicación construirá un modelo de
    **regresión lineal simple o múltiple** con gráficos, métricas de desempeño y
    un módulo de **predicciones**.
    """
)


# --------------------------------------------------------------------------- #
# 1. Carga de datos
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1️⃣ Datos")
    archivo = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Se detectan automáticamente los separadores ; , o tabulador.",
    )
    usar_ejemplo = st.button("📂 Usar datos de ejemplo")

df = None
if archivo is not None:
    try:
        df = utils.cargar_datos(archivo)
        st.session_state["df"] = df
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el archivo: {exc}")
        st.stop()
elif usar_ejemplo:
    try:
        df = pd.read_csv("sample_data.csv", sep=";", decimal=",")
        st.session_state["df"] = df
    except FileNotFoundError:
        st.error("No se encontró el archivo de ejemplo `sample_data.csv`.")
        st.stop()
elif "df" in st.session_state:
    df = st.session_state["df"]

if df is None:
    st.info("👈 Sube un archivo CSV/Excel o haz clic en **Usar datos de ejemplo** para comenzar.")
    st.stop()


# --------------------------------------------------------------------------- #
# 2. Vista previa y selección de variables
# --------------------------------------------------------------------------- #
st.subheader("📋 Vista previa de los datos")
c1, c2, c3 = st.columns(3)
c1.metric("Filas", f"{df.shape[0]:,}")
c2.metric("Columnas", df.shape[1])
c3.metric("Valores faltantes", int(df.isna().sum().sum()))
st.dataframe(df.head(100))

st.subheader("🎛️ Selección de variables")
columnas = list(df.columns)
candidatas = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]

col_a, col_b = st.columns(2)
with col_a:
    target = st.selectbox("Variable dependiente (Y)", candidatas)
with col_b:
    opciones_x = [c for c in columnas if c != target]
    features = st.multiselect(
        "Variables independientes (X)",
        opciones_x,
        help="Puedes elegir variables numéricas y categóricas (se codifican automáticamente).",
    )

todas = st.checkbox("Seleccionar todas las variables numéricas excepto Y", value=False)
if todas:
    features = [c for c in candidatas if c != target]

if not features:
    st.info("Selecciona al menos una variable independiente (X) para construir el modelo.")
    st.stop()

# Preparación: codificación one-hot y limpieza de valores faltantes
try:
    X, y, nombres = utils.preparar_datos(df, target, features)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

filas_descartadas = len(df) - len(X)
if filas_descartadas > 0:
    st.warning(
        f"Se eliminaron {filas_descartadas} filas con valores faltantes en las columnas seleccionadas."
    )

constantes = [n for n in nombres if X[n].nunique() <= 1]
if constantes:
    st.warning("Variables sin variabilidad (un solo valor) excluidas del modelo: " + ", ".join(constantes))
    X = X.drop(columns=constantes)
    nombres = [n for n in nombres if n not in constantes]

if len(nombres) == 0:
    st.error("No quedaron variables independientes válidas para el modelo.")
    st.stop()

if len(X) < 10:
    st.error("Se necesitan al menos 10 filas completas para entrenar y evaluar el modelo.")
    st.stop()

if any(not pd.api.types.is_numeric_dtype(df[c]) for c in features):
    st.caption("ℹ️ Las variables categóricas seleccionadas se codificaron automáticamente (one-hot).")

es_simple = len(nombres) == 1


# --------------------------------------------------------------------------- #
# 3. Configuración y entrenamiento del modelo
# --------------------------------------------------------------------------- #
st.subheader("⚙️ Configuración del modelo")
col_c, col_d = st.columns(2)
test_size = col_c.slider(
    "Tamaño del conjunto de prueba",
    0.10, 0.50, 0.20, 0.05,
    help="Proporción de datos reservados para evaluar el modelo.",
)
random_state = col_d.number_input("Semilla aleatoria (random_state)", 0, 9999, 42)

resultado = utils.entrenar_modelo(X, y, test_size=test_size, random_state=random_state)
modelo = resultado["modelo"]
y_test = resultado["y_test"]
y_pred_test = modelo.predict(resultado["X_test"])
y_pred_train = modelo.predict(resultado["X_train"])


# --------------------------------------------------------------------------- #
# 4. Métricas de desempeño
# --------------------------------------------------------------------------- #
st.subheader("📊 Métricas de desempeño (conjunto de prueba)")
m_test = utils.calcular_metricas(y_test, y_pred_test, len(nombres))
m_train = utils.calcular_metricas(resultado["y_train"], y_pred_train, len(nombres))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R²", f"{m_test['R²']:.4f}")
c2.metric("R² ajustado", f"{m_test['R² ajustado']:.4f}")
c3.metric("RMSE", f"{m_test['RMSE']:.4f}")
c4.metric("MAE", f"{m_test['MAE']:.4f}")
c5.metric("MSE", f"{m_test['MSE']:.4f}")

with st.expander("Comparación entrenamiento vs. prueba"):
    tabla = pd.DataFrame(
        {
            "Métrica": list(m_test.keys()),
            "Entrenamiento": [f"{m_train[k]:.4f}" for k in m_test],
            "Prueba": [f"{m_test[k]:.4f}" for k in m_test],
        }
    )
    st.dataframe(tabla, hide_index=True)

st.markdown("**Ecuación del modelo:**")
intercepto = modelo.intercept_
eq = f"ŷ = {intercepto:.4f}"
for n, c in zip(nombres, modelo.coef_):
    signo = "+" if c >= 0 else "−"
    eq += f" {signo} {abs(c):.4f} × {n}"
st.code(eq, language="text")


# --------------------------------------------------------------------------- #
# 5. Gráficos del modelo
# --------------------------------------------------------------------------- #
st.subheader("📈 Gráficos del modelo")
tab_modelo, tab_real, tab_resid, tab_coef = st.tabs(
    ["Modelo", "Predichos vs reales", "Residuos", "Coeficientes"]
)

with tab_modelo:
    if es_simple:
        fig = utils.grafico_regresion_simple(X, y, nombres[0])
        st.pyplot(fig)
        st.caption("Regresión simple: dispersión con recta de regresión y banda de confianza del 95 %.")
    else:
        fig = utils.grafico_matriz_dispersion(X, y, nombres)
        st.pyplot(fig)
        st.caption("Regresión múltiple: relación de cada variable independiente con la variable dependiente.")

with tab_real:
    st.pyplot(utils.grafico_actual_vs_predicho(y_test, y_pred_test))
    st.caption("Puntos cercanos a la línea ideal indican un buen ajuste (conjunto de prueba).")

with tab_resid:
    st.pyplot(utils.grafico_residuos(y_test, y_pred_test))
    st.caption("Residuos sin patrones claros y distribuidos alrededor de cero indican un ajuste adecuado.")

with tab_coef:
    st.dataframe(
        pd.DataFrame({"Variable": nombres, "Coeficiente": modelo.coef_}),
        hide_index=True,
    )
    st.pyplot(utils.grafico_coeficientes(modelo, nombres))


# --------------------------------------------------------------------------- #
# 6. Predicciones con el modelo
# --------------------------------------------------------------------------- #
st.subheader("🔮 Predicciones con el modelo")
st.markdown("Ajusta los valores de las variables independientes y presiona **Predecir**.")

if len(nombres) != len(features):
    st.caption("ℹ️ Las columnas generadas por codificación one-hot toman valores 0 o 1.")

with st.form("formulario_prediccion"):
    valores = {}
    cols = st.columns(3)
    for i, nombre in enumerate(nombres):
        col = cols[i % 3]
        vmin = float(X[nombre].min())
        vmax = float(X[nombre].max())
        valor_defecto = float(X[nombre].median())
        paso = max((vmax - vmin) / 50.0, 1e-9)
        valores[nombre] = col.slider(
            nombre, min_value=vmin, max_value=vmax, value=valor_defecto, step=paso
        )
    predecir = st.form_submit_button("🔮 Predecir", type="primary")

if predecir:
    fila = pd.DataFrame([valores])
    pred = float(modelo.predict(fila)[0])
    st.success(f"Valor predicho de **{target}**: **{pred:,.4f}**")

st.markdown("---")
st.caption("Hecho con Streamlit, pandas, scikit-learn, matplotlib y seaborn.")