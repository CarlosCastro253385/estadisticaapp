import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="App de Estadística", layout="wide")

st.title("📊 Aplicación de Análisis Estadístico")
st.markdown("""
Esta aplicación permite cargar datos, visualizar distribuciones y 
realizar pruebas de hipótesis con el apoyo de IA.
""")

---

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