"""Aplicación Streamlit: análisis de regresión lineal simple y múltiple.

Archivo autocontenido (no requiere módulos adicionales del proyecto).

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Análisis de Regresión Lineal",
    page_icon="📈",
    layout="wide",
)


# =========================================================================== #
# Funciones auxiliares
# =========================================================================== #

def cargar_datos(archivo) -> pd.DataFrame:
    """Carga un archivo CSV o Excel subido por el usuario.

    Para CSV intenta varios separadores (``;``, ``,``, tabulador) y
    codificaciones comunes, de modo que funcionen archivos generados en
    español (por ejemplo con ``;`` y coma decimal).
    """
    nombre = (getattr(archivo, "name", "") or "").lower()

    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)

    for sep in (";", ",", "\t"):
        for encoding in ("utf-8", "latin-1"):
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=sep, decimal=",", encoding=encoding)
                if df.shape[1] > 1:
                    return df
            except Exception:  # noqa: BLE001 - se prueba la siguiente combinación
                continue

    # Último intento con la configuración por defecto de pandas
    archivo.seek(0)
    return pd.read_csv(archivo)


def preparar_datos(df: pd.DataFrame, target: str, features: list[str], dropna: bool = True):
    """Prepara la matriz de predictores X y el vector objetivo y.

    - Codifica automáticamente (one-hot) las variables categóricas elegidas.
    - Elimina filas con valores faltantes en las columnas usadas (si dropna=True).

    Devuelve ``(X, y, nombres)`` donde ``nombres`` son los predictores
    finales (incluye las columnas generadas por la codificación).
    """
    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en los datos.")
    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        raise ValueError("Variables no encontradas: " + ", ".join(faltantes))

    datos = df[[target] + list(features)].copy()

    if dropna:
        datos = datos.dropna()

    y = datos[target]

    categoricas = [c for c in features if not pd.api.types.is_numeric_dtype(datos[c])]
    numericas = [c for c in features if c not in categoricas]

    if numericas:
        X = datos[numericas].copy()
    else:
        X = pd.DataFrame(index=datos.index)

    if categoricas:
        dummies = pd.get_dummies(datos[categoricas], prefix=categoricas)
        X = pd.concat([X, dummies], axis=1)

    return X, y, list(X.columns)


def entrenar_modelo(X, y, test_size: float = 0.2, random_state: int = 42):
    """Divide los datos y ajusta una regresión lineal (scikit-learn)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return {
        "modelo": modelo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def calcular_metricas(y_real, y_pred, n_predictores: int) -> dict:
    """Calcula R², R² ajustado, RMSE, MAE y MSE."""
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    n = len(y_real)
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)

    if n - n_predictores - 1 > 0:
        r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - n_predictores - 1)
    else:
        r2_ajustado = float("nan")

    return {"R²": r2, "R² ajustado": r2_ajustado, "RMSE": rmse, "MAE": mae, "MSE": mse}


def grafico_regresion_simple(X, y, nombre_feature: str):
    """Dispersión con recta de regresión y banda de confianza (regresión simple)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x=X[nombre_feature],
        y=y,
        ax=ax,
        scatter_kws={"s": 40, "alpha": 0.7},
        line_kws={"color": "#d62728", "lw": 2},
        errorbar=("ci", 95),
    )
    ax.set_xlabel(nombre_feature)
    ax.set_ylabel("Variable dependiente (Y)")
    ax.set_title(f"Regresión lineal: Y vs {nombre_feature}")
    fig.tight_layout()
    return fig


def grafico_matriz_dispersion(X, y, nombres: list[str], max_vars: int = 6):
    """Cuadrícula de dispersión de cada predictor contra la variable dependiente."""
    mostrar = nombres[:max_vars]
    n = len(mostrar)
    filas = int(np.ceil(n / 2))
    fig, axes = plt.subplots(filas, 2, figsize=(11, 3.6 * filas), squeeze=False)
    for i, nombre in enumerate(mostrar):
        ax = axes[i // 2][i % 2]
        sns.regplot(
            x=X[nombre],
            y=y,
            ax=ax,
            scatter_kws={"s": 30, "alpha": 0.6},
            line_kws={"color": "#d62728", "lw": 1.8},
            ci=None,
        )
        ax.set_title(f"Y vs {nombre}", fontsize=10)
    for j in range(n, filas * 2):
        axes[j // 2][j % 2].axis("off")
    fig.tight_layout()
    return fig


def grafico_actual_vs_predicho(y_real, y_pred):
    """Dispersión de valores reales contra predichos con la línea ideal."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_real, y_pred, alpha=0.65, s=45, color="#4c72b0", label="Observaciones")
    limite = (min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max()))
    ax.plot(limite, limite, color="#d62728", lw=2, ls="--", label="Línea ideal (y = ŷ)")
    ax.set_xlabel("Valores reales (y)")
    ax.set_ylabel("Valores predichos (ŷ)")
    ax.set_title("Predichos vs. reales")
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_residuos(y_real, y_pred):
    """Residuos vs. predichos e histograma de residuos."""
    residuos = np.asarray(y_real) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, residuos, alpha=0.6, s=35, color="#4c72b0")
    axes[0].axhline(0, color="#d62728", lw=1.5, ls="--")
    axes[0].set_xlabel("Valores predichos (ŷ)")
    axes[0].set_ylabel("Residuos")
    axes[0].set_title("Residuos vs. predichos")
    n_bins = min(20, max(5, int(len(residuos) / 5)))
    axes[1].hist(residuos, bins=n_bins, color="#4c72b0", edgecolor="white")
    axes[1].set_xlabel("Residuos")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Histograma de residuos")
    fig.tight_layout()
    return fig


def grafico_coeficientes(modelo, nombres: list[str]):
    """Barras horizontales con los coeficientes del modelo."""
    coefs = np.asarray(modelo.coef_)
    orden = np.argsort(coefs)
    colores = ["#4c72b0" if c >= 0 else "#d62728" for c in coefs[orden]]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(nombres) + 1.5)))
    ax.barh(np.asarray(nombres)[orden], coefs[orden], color=colores)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Coeficiente")
    ax.set_title("Coeficientes del modelo")
    fig.tight_layout()
    return fig


def generar_reporte_pdf(
    *,
    nombre_datos: str,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    nombres: list[str],
    es_simple: bool,
    X,
    y,
    modelo,
    y_test,
    y_pred_test,
    m_test: dict,
    m_train: dict,
    test_size: float,
    random_state: int,
    prediccion: dict | None = None,
) -> bytes:
    """Genera un informe PDF con el resumen completo del modelo.

    Incluye datos del análisis, ecuación del modelo, métricas de desempeño,
    coeficientes, gráficos y (si existe) la última predicción realizada.
    Los imports de reportlab se hacen aquí dentro para que la app funcione
    incluso si esa librería no está instalada.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    estilos = getSampleStyleSheet()
    estilo_seccion = ParagraphStyle(
        "Seccion",
        parent=estilos["Heading2"],
        fontSize=13,
        spaceBefore=16,
        spaceAfter=6,
        textColor=colors.HexColor("#1f4e79"),
    )
    estilo_codigo = ParagraphStyle(
        "Codigo",
        parent=estilos["BodyText"],
        fontName="Courier",
        fontSize=10,
        backColor=colors.HexColor("#f2f2f2"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=4,
    )
    estilo_centrado = ParagraphStyle(
        "Centrado",
        parent=estilos["Normal"],
        alignment=1,  # centro
        fontSize=10,
        textColor=colors.grey,
    )

    def tabla(datos: list, anchos=None, cabecera: bool = True):
        """Crea una tabla con estilo de la app (cabecera azul)."""
        t = Table(datos, colWidths=anchos, hAlign="LEFT")
        estilo = [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if cabecera:
            estilo += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c72b0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
            ]
        else:
            estilo += [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
            ]
        t.setStyle(TableStyle(estilo))
        return t

    def agregar_grafico(story, fig, ancho: float = 16.5 * cm):
        """Guarda una figura de matplotlib como imagen PNG dentro del PDF."""
        w, h = fig.get_size_inches()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        story.append(Image(buf, width=ancho, height=ancho * h / w))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Informe de Regresión Lineal",
        author="App de Regresión Lineal (Streamlit)",
    )
    story = []

    # Portada del informe
    story.append(Paragraph("Informe del Modelo de Regresión Lineal", estilos["Title"]))
    story.append(Paragraph(f"Generado el {datetime.now():%d/%m/%Y a las %H:%M}", estilo_centrado))
    story.append(Spacer(1, 8))

    # 1. Datos del análisis
    story.append(Paragraph("1. Datos del análisis", estilo_seccion))
    tipo = "Regresión lineal simple (1 variable)" if es_simple else "Regresión lineal múltiple"
    story.append(
        tabla(
            [
                ["Archivo de datos", str(nombre_datos)],
                ["Registros", f"{len(df):,}"],
                ["Columnas", f"{df.shape[1]}"],
                ["Valores faltantes", f"{int(df.isna().sum().sum())}"],
                ["Tipo de modelo", tipo],
                ["Variable dependiente (Y)", str(target)],
                ["Variables independientes (X)", ", ".join(features)],
                ["Predictores del modelo", ", ".join(nombres)],
                ["Partición de prueba", f"{test_size:.0%}"],
                ["Semilla aleatoria", f"{random_state}"],
            ],
            anchos=[5.5 * cm, 11.4 * cm],
            cabecera=False,
        )
    )

    # 2. Ecuación del modelo
    story.append(Paragraph("2. Ecuación del modelo", estilo_seccion))
    intercepto = modelo.intercept_
    eq = f"Y estimada = {intercepto:.4f}"
    for n, c in zip(nombres, modelo.coef_):
        signo = "+" if c >= 0 else "-"
        eq += f" {signo} {abs(c):.4f} * {n}"
    story.append(Paragraph(eq, estilo_codigo))

    # 3. Métricas de desempeño
    story.append(Paragraph("3. Métricas de desempeño", estilo_seccion))
    filas = [["Métrica", "Entrenamiento", "Prueba"]]
    for k in m_test:
        filas.append([k, f"{m_train[k]:.4f}", f"{m_test[k]:.4f}"])
    story.append(tabla(filas))

    # 4. Coeficientes del modelo
    story.append(Paragraph("4. Coeficientes del modelo", estilo_seccion))
    filas = [["Variable", "Coeficiente"]]
    for n, c in zip(nombres, modelo.coef_):
        filas.append([n, f"{c:.6f}"])
    story.append(tabla(filas))

    # 5. Gráficos del modelo
    story.append(Paragraph("5. Gráficos del modelo", estilo_seccion))
    if es_simple:
        fig = grafico_regresion_simple(X, y, nombres[0])
        agregar_grafico(story, fig)
    else:
        fig = grafico_matriz_dispersion(X, y, nombres)
        agregar_grafico(story, fig)
    fig = grafico_actual_vs_predicho(y_test, y_pred_test)
    agregar_grafico(story, fig)
    fig = grafico_residuos(y_test, y_pred_test)
    agregar_grafico(story, fig)
    fig = grafico_coeficientes(modelo, nombres)
    agregar_grafico(story, fig)

    # 6. Predicción realizada (si existe)
    if prediccion:
        story.append(Paragraph("6. Predicción realizada", estilo_seccion))
        filas = [["Variable", "Valor"]]
        for k, v in prediccion["valores"].items():
            filas.append([k, f"{v:,.4f}"])
        filas.append(["Valor predicho (Y)", f"{prediccion['prediccion']:,.4f}"])
        story.append(tabla(filas, anchos=[9 * cm, 8 * cm]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# =========================================================================== #
# Aplicación
# =========================================================================== #

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
        df = cargar_datos(archivo)
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
    X, y, nombres = preparar_datos(df, target, features)
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

resultado = entrenar_modelo(X, y, test_size=test_size, random_state=random_state)
modelo = resultado["modelo"]
y_test = resultado["y_test"]
y_pred_test = modelo.predict(resultado["X_test"])
y_pred_train = modelo.predict(resultado["X_train"])


# --------------------------------------------------------------------------- #
# 4. Métricas de desempeño
# --------------------------------------------------------------------------- #
st.subheader("📊 Métricas de desempeño (conjunto de prueba)")
m_test = calcular_metricas(y_test, y_pred_test, len(nombres))
m_train = calcular_metricas(resultado["y_train"], y_pred_train, len(nombres))

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
        fig = grafico_regresion_simple(X, y, nombres[0])
        st.pyplot(fig)
        st.caption("Regresión simple: dispersión con recta de regresión y banda de confianza del 95 %.")
    else:
        fig = grafico_matriz_dispersion(X, y, nombres)
        st.pyplot(fig)
        st.caption("Regresión múltiple: relación de cada variable independiente con la variable dependiente.")

with tab_real:
    st.pyplot(grafico_actual_vs_predicho(y_test, y_pred_test))
    st.caption("Puntos cercanos a la línea ideal indican un buen ajuste (conjunto de prueba).")

with tab_resid:
    st.pyplot(grafico_residuos(y_test, y_pred_test))
    st.caption("Residuos sin patrones claros y distribuidos alrededor de cero indican un ajuste adecuado.")

with tab_coef:
    st.dataframe(
        pd.DataFrame({"Variable": nombres, "Coeficiente": modelo.coef_}),
        hide_index=True,
    )
    st.pyplot(grafico_coeficientes(modelo, nombres))


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
    st.session_state["ultima_prediccion"] = {"valores": dict(valores), "prediccion": pred}
    st.success(f"Valor predicho de **{target}**: **{pred:,.4f}**")


# --------------------------------------------------------------------------- #
# 7. Informe del modelo en PDF
# --------------------------------------------------------------------------- #
st.subheader("📄 Informe del modelo en PDF")
st.markdown(
    "Genera un informe descargable con los datos del análisis, la ecuación del "
    "modelo, las métricas de desempeño, los coeficientes, los gráficos y, si "
    "realizaste una predicción, el resultado de la misma."
)

if st.button("📄 Generar informe en PDF"):
    with st.spinner("Generando el informe…"):
        nombre_archivo = getattr(archivo, "name", None) or "Datos de ejemplo (sample_data.csv)"
        st.session_state["pdf_reporte"] = generar_reporte_pdf(
            nombre_datos=nombre_archivo,
            df=df,
            target=target,
            features=features,
            nombres=nombres,
            es_simple=es_simple,
            X=X,
            y=y,
            modelo=modelo,
            y_test=y_test,
            y_pred_test=y_pred_test,
            m_test=m_test,
            m_train=m_train,
            test_size=test_size,
            random_state=random_state,
            prediccion=st.session_state.get("ultima_prediccion"),
        )

if "pdf_reporte" in st.session_state:
    st.download_button(
        "⬇️ Descargar informe en PDF",
        data=st.session_state["pdf_reporte"],
        file_name="informe_regresion_lineal.pdf",
        mime="application/pdf",
        type="primary",
    )

st.markdown("---")
st.caption("Hecho con Streamlit, pandas, scikit-learn, matplotlib, seaborn y reportlab.")