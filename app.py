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