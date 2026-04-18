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
        st.success("¡Datos generados :D!")

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
        sns.boxplot(x=df[col_analisis], ax=ax, color="blueviolet")
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
    st.warning("Por favor, carga o genera datos primero en la sección N1.")

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

    st.subheader("Visualización de la Región Crítica")
    
    x = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(x, 0, 1) 
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label='Distribución Normal Estándar', color='yellow')

    if tipo_test == "Bilateral":
        z_critico = stats.norm.ppf(1 - alpha/2)
        x_cola_izq = np.linspace(-4, -z_critico, 100)
        x_cola_der = np.linspace(z_critico, 4, 100)
        ax.fill_between(x_cola_izq, stats.norm.pdf(x_cola_izq), color='salmon', alpha=0.5, label='Zona de rechazo')
        ax.fill_between(x_cola_der, stats.norm.pdf(x_cola_der), color='salmon', alpha=0.5)
    elif tipo_test == "Cola Derecha":
        z_critico = stats.norm.ppf(1 - alpha)
        x_cola_der = np.linspace(z_critico, 4, 100)
        ax.fill_between(x_cola_der, stats.norm.pdf(x_cola_der), color='salmon', alpha=0.5, label='Zona de rechazo')
    else:
        z_critico = stats.norm.ppf(alpha)
        x_cola_izq = np.linspace(-4, z_critico, 100)
        ax.fill_between(x_cola_izq, stats.norm.pdf(x_cola_izq), color='salmon', alpha=0.5, label='Zona de rechazo')

    ax.axvline(z_stat, color='green', linestyle='--', linewidth=2, label=f'Z calculado ({z_stat:.2f})')
    
    ax.set_title(f"Prueba {tipo_test}")
    ax.legend()
    st.pyplot(fig)
    
st.header("4. Análisis con Inteligencia Artificial")

api_key = st.text_input("Introduce tu Gemini API Key", type="password")

if api_key:
    import google.generativeai as genai
    
    try:
        genai.configure(api_key=api_key)
        
        modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if modelos_disponibles:
            modelo_nombre = modelos_disponibles[0] 
            model = genai.GenerativeModel(modelo_nombre)
            
            if st.button("Consultar a Gemini"):
                prompt = f"""
                Analiza como experto estadístico:
                - Variable: {col_analisis}
                - Z: {z_stat:.4f}, P-value: {p_val:.4f}, Alpha: {alpha}
                - Decisión: {"Rechaza H0" if p_val < alpha else "No rechaza H0"}
                Explica brevemente la conclusión.
                """
                
                with st.spinner(f"Conectando con {modelo_nombre}..."):
                    response = model.generate_content(prompt)
                    st.success("¡Análisis completado!")
                    st.markdown(response.text)
        else:
            st.error("No se encontraron modelos disponibles para esta API Key.")
            
    except Exception as e:
        st.error(f"Error de configuración: {e}")
        st.info("Tip: Asegúrate de que tu API Key sea de 'Google AI Studio' y no de 'Google Cloud'.")
else:
    st.warning("Introduce tu API Key para continuar.")