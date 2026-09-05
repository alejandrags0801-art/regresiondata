# 📈 Análisis de Regresión Lineal con Streamlit

Aplicación web construida con [Streamlit](https://streamlit.io) que permite cargar un archivo **CSV o Excel**, seleccionar la variable dependiente (**Y**) y las variables independientes (**X**), y construir automáticamente un modelo de **regresión lineal simple o múltiple** con gráficos, métricas de desempeño y un módulo de **predicciones**.

## ✨ Características

- **Carga de datos flexible**: archivos CSV (detecta separadores `;`, `,` y tabulador, con codificación UTF-8 o Latin-1) y Excel (`.xlsx`, `.xls`).
- **Reconocimiento automático de variables**: la aplicación lista las columnas del archivo y muestra una vista previa con dimensiones y valores faltantes.
- **Selección de variables**: eliges la variable dependiente (Y) y una o varias independientes (X).
- **Regresión simple o múltiple automática**: con 1 predictora usa regresión simple; con varias, regresión múltiple.
- **Variables categóricas**: se codifican automáticamente con one-hot encoding.
- **Gráficos del modelo**:
  - Regresión simple: dispersión con recta ajustada y banda de confianza del 95 %.
  - Regresión múltiple: relación de cada variable X con Y.
  - Predichos vs. reales, análisis de residuos y coeficientes del modelo.
- **Métricas de desempeño**: R², R² ajustado, RMSE, MAE y MSE, calculadas sobre el conjunto de prueba, con partición entrenamiento/prueba configurable y comparación entre ambos conjuntos.
- **Ecuación del modelo** con intercepto y coeficientes.
- **Predicciones**: ajustas los valores de las variables X con controles deslizantes y el modelo predice Y.
- **Informe en PDF**: botón para descargar un informe con los datos del análisis, la ecuación, las métricas, los coeficientes, los gráficos del modelo y la última predicción realizada.
- **Datos de ejemplo** incluidos para probar la app sin subir archivos.

## 📁 Estructura del proyecto

| Archivo | Descripción |
| --- | --- |
| `app.py` | Aplicación completa de Streamlit (autocontenida: carga, modelado, gráficos y predicciones) |
| `requirements.txt` | Dependencias de Python |
| `sample_data.csv` | Datos de ejemplo (publicidad y ventas) |
| `.streamlit/config.toml` | Configuración de tema y servidor |
| `.gitignore` | Archivos ignorados por Git |

> ℹ️ `app.py` es **autocontenido**: no depende de otros archivos Python, por lo que la app funciona aunque solo ese archivo esté subido al repositorio.

## 🚀 Ejecutar en local

Requisitos: Python 3.9 o superior.

```bash
git clone https://github.com/TU_USUARIO/regresion-lineal-streamlit.git
cd regresion-lineal-streamlit

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

## ☁️ Desplegar en Streamlit Community Cloud

### 1. Sube el proyecto a GitHub

Crea un repositorio nuevo en <https://github.com/new> (por ejemplo `regresion-lineal-streamlit`) y luego, desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "App de regresión lineal con Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/regresion-lineal-streamlit.git
git push -u origin main
```

### 2. Conecta el repositorio con Streamlit

1. Entra a <https://share.streamlit.io> e **inicia sesión con tu cuenta de GitHub**.
2. Haz clic en **New app** → **Deploy an app**.
3. Selecciona el repositorio `regresion-lineal-streamlit`, la rama `main` y, en **Main file path**, escribe `app.py`.
4. (Opcional) En **Advanced settings** puedes fijar la versión de Python (recomendado: **3.11**).
5. Haz clic en **Deploy**. Streamlit instalará las dependencias de `requirements.txt` y publicará la app en una URL pública como:

   ```
   https://TU_USUARIO-regresion-lineal-streamlit.streamlit.app
   ```

6. Cada vez que hagas `git push` a la rama `main`, la app se **re-despliega automáticamente**.

> 💡 Nota: en el plan gratuito de Streamlit Community Cloud el repositorio debe ser **público** (o vincular tu cuenta de GitHub con Streamlit). También puedes desplegar desde <https://streamlit.io/cloud> arrastrando el repositorio.

## 🛠️ Solución de problemas

### `ModuleNotFoundError: No module named 'utils'`

Este error aparece cuando en el repositorio de GitHub falta algún archivo del proyecto o los archivos quedaron en carpetas diferentes (por ejemplo, `app.py` en la raíz y el resto en una subcarpeta).

- **La app ya es autocontenida**: desde la versión actual, `app.py` incluye toda la lógica, así que ya no depende de `utils.py`.
- Asegúrate de que `app.py`, `requirements.txt` y `sample_data.csv` estén **en la raíz del repositorio** (no dentro de subcarpetas).
- Verifica en GitHub (web) que los archivos estén en el nivel correcto: entra a tu repo → raíz → deben aparecer `app.py`, `requirements.txt`, etc.
- Después de corregirlo, sube los archivos con `git add . && git commit -m "fix: app autocontenida" && git push` y Streamlit se re-desplegará solo. Si no se actualiza, entra a tu app en Streamlit Cloud → menú ⋮ → **Reboot**.

## 🧪 Probar con los datos de ejemplo

`sample_data.csv` contiene datos clásicos de publicidad: inversión en TV, radio y periódico, y las ventas resultantes. Prueba:

- **Y** → `ventas`
- **X** → `tv` (regresión simple) o `tv`, `radio`, `periodico` (regresión múltiple).

## 🛠️ Tecnologías

Streamlit · pandas · NumPy · scikit-learn · matplotlib · seaborn · openpyxl · reportlab

## 📝 Notas

- Las filas con valores faltantes en las columnas seleccionadas se eliminan automáticamente (se muestra un aviso).
- Las variables constantes (un solo valor) se excluyen del modelo.
- Las predicciones se realizan con el modelo entrenado sobre la partición de entrenamiento.