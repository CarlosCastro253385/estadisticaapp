import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="App de Estadística", layout="wide")

st.title("📊 Aplicación de Análisis Estadístico")
st.markdown("""
Esta aplicación permite cargar datos, visualizar distribuciones y 
realizar pruebas de hipótesis con el apoyo de IA.
""")

st.markdown("---")

st.header("1. Carga de Datos")

col1, col2 = st.columns(2)

with col1:
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

with col2:
    st.write("O genera datos de prueba:")
    if st.button("Generar datos sintéticos"):
        data = np.random.normal(loc=50, scale=10, size=100)
        df = pd.DataFrame(data, columns=["Variable_X"])
        st.session_state['df'] = df
        st.success("¡Datos generados!")

if archivo is not None:
    df = pd.read_csv(archivo)
    st.session_state['df'] = df
    st.write("Vista previa de los datos:", df.head())

st.header("2. Visualización de Distribuciones")

if 'df' in st.session_state:
    df = st.session_state['df']
    col_analisis = st.selectbox("Selecciona la variable para graficar", df.columns)
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("Histograma y KDE")
        fig, ax = plt.subplots()
        sns.histplot(df[col_analisis], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

    with col_der:
        st.subheader("Boxplot (Outliers)")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col_analisis], ax=ax, color="salmon")
        st.pyplot(fig)

    st.markdown("### 📝 Análisis de la Distribución")
    col1, col2, col3 = st.columns(3)
    with col1:
        normal = st.radio("¿Parece normal?", ["Sí", "No", "No estoy seguro"])
    with col2:
        sesgo = st.radio("¿Hay sesgo?", ["Derecha", "Izquierda", "Ninguno"])
    with col3:
        outliers = st.radio("¿Hay outliers?", ["Sí", "No"])
else:
    st.warning("Por favor, carga o genera datos primero en la sección 1.")

st.header("3. Prueba de Hipótesis Z")

if 'df' in st.session_state:
    from scipy import stats
    
    col1, col2 = st.columns(2)
    
    with col1:
        mu_0 = st.number_input("Hipótesis Nula (Media H0)", value=50.0)
        alpha = st.number_input("Nivel de significancia (α)", value=0.05, min_value=0.01, max_value=0.10)
    
    with col2:
        tipo_test = st.selectbox("Tipo de prueba", ["Bilateral", "Cola Derecha", "Cola Izquierda"])
        sigma = st.number_input("Desviación Estándar Poblacional (σ)", value=10.0)

    datos = st.session_state['df'][col_analisis]
    n = len(datos)
    media_muestral = datos.mean()
    
    z_stat = (media_muestral - mu_0) / (sigma / np.sqrt(n))
    
    if tipo_test == "Bilateral":
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif tipo_test == "Cola Derecha":
        p_val = 1 - stats.norm.cdf(z_stat)
    else: 
        p_val = stats.norm.cdf(z_stat)

    st.subheader("Resultados de la Prueba")
    st.write(f"Estadístico Z: **{z_stat:.4f}**")
    st.write(f"P-value: **{p_val:.4f}**")

    if p_val < alpha:
        st.error("Resultado: Se rechaza la Hipótesis Nula (H0)")
    else:
        st.success("Resultado: No se rechaza la Hipótesis Nula (H0)")