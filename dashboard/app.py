# APP

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import joblib
import os


#--------------#
#Configuracion de la pagina
# Cargar recursos para el gráfico PCA (Solo se hace una vez)
@st.cache_resource
def load_artifacts():
    # Rutas relativas asumiendo que corres desde la carpeta raiz
    x_path = os.path.join("data", "X_train_processed.csv")
    y_path = os.path.join("data", "y_train.csv")
    pre_path = os.path.join("models", "preprocessor.pkl")
    
    # Cargar datos
    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)
    preprocessor = joblib.load(pre_path)
    
    # Entrenar PCA con los datos históricos
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    
    # Crear DataFrame para plotear el fondo
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['Estado'] = y_train.values # 0 = Sano, 1 = Falla
    
    return pca, preprocessor, df_pca

try:
    pca_model, preprocessor_model, df_background = load_artifacts()
except Exception as e:
    st.error(f"No se pudieron cargar archivos para el gráfico PCA: {e}")
    pca_model = None

# Título
st.title("🏭 Dashboard de Predicción de Fallas")
st.markdown("---")

# Sidebar (Barra lateral) Controles
with st.sidebar:
    st.header("Panel de Control 🎛️")
    
    st.subheader("Seleccionar Modelo")
    model_option = st.selectbox(
        "Modelo:",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        help="Elige qué modelo queres usar para el cálculo."
    )
    
    st.markdown("---")
    st.subheader("Carga de Datos")
    
    with st.form("pipeline_form"):
        material = st.selectbox("Material:", ["Carbon Steel", "Stainless Steel", "PVC", "HDPE", "Fiberglass"])
        condition = st.selectbox("Condición Visual:", ["Normal", "Moderate", "Critical"])
        grade = st.selectbox("Grado:", ["API 5L X52", "API 5L X65", "API 5L X42", "ASTM A106 Grade B"])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            pipe_size = st.number_input("Diámetro (mm):", value=200.0)
            thickness = st.number_input("Espesor (mm):", value=12.0)
            pressure = st.number_input("Presión (psi):", value=600.0)
            temp = st.number_input("Temp (°C):", value=40.0)
        
        with col2:
            time_years = st.number_input("Edad (Años):", value=10, step=1)
            thick_loss = st.number_input("Pérdida Espesor:", value=0.0)
            corr_impact = st.number_input("Imp. Corrosión %:", value=5.0)
            mat_loss = st.number_input("Pérdida Material %:", value=0.0)
            
        submitted = st.form_submit_button("CALCULAR RIESGO", use_container_width=True)

#--------------#
# LÓGICA PRINCIPAL
if submitted:
    # Datos para la API
    input_data = {
        "Pipe_Size_mm": pipe_size, "Thickness_mm": thickness, "Thickness_Loss_mm": thick_loss,
        "Max_Pressure_psi": pressure, "Temperature_C": temp, "Time_Years": int(time_years),
        "Corrosion_Impact_Percent": corr_impact, "Material_Loss_Percent": mat_loss,
        "Condition": condition, "Material": material, "Grade": grade
    }

    # Llamada a la API
    api_url = f"http://127.0.0.1:8000/predict/{model_option}"
    
    try:
        with st.spinner("Analizando integridad estructural..."):
            response = requests.post(api_url, json=input_data)
            
        if response.status_code == 200:
            result = response.json()
            prob = result["failure_probability"] # 0.85
            prob_percent = prob * 100
            
            # SECCIÓN VISUAL
            
            # KPI PRINCIPAL
            col_kpi1, col_kpi2 = st.columns([1, 2])
            
            with col_kpi1:
                st.metric(label="Probabilidad de Falla", value=f"{prob_percent:.1f}%")
                if prob < 0.1: st.info("🔵 ESTADO: NUEVO / OPTIMO")
                elif prob < 0.3: st.success("🟢 ESTADO: SEGURO")
                elif prob < 0.6: st.warning("🟡 ESTADO: PRECAUCIÓN")
                else: st.error("🔴 ESTADO: PELIGRO CRÍTICO")

            with col_kpi2:
                # BARRA DEGRADÉ CUSTOM (HTML/CSS)
                # Creamos una barra con CSS que tenga tu Azul -> Verde -> Amarillo -> Rojo
                st.write(f"**Nivel de Riesgo ({model_option})**")
                
                # Definimos la posición de la flecha
                marker_position = prob_percent 
                if marker_position > 98: marker_position = 98 # Para que no se salga
                
                # HTML
                st.markdown(f"""
                    <div style="width:100%; margin-top:10px;">
                        <div style="
                            height: 30px;
                            width: 100%;
                            background: linear-gradient(to right, #3498db 0%, #2ecc71 25%, #f1c40f 60%, #e74c3c 100%);
                            border-radius: 15px;
                            position: relative;
                            opacity: 0.8;
                        ">
                            <div style="
                                position: absolute;
                                left: {marker_position}%;
                                top: -10px;
                                width: 0; 
                                height: 0; 
                                border-left: 10px solid transparent;
                                border-right: 10px solid transparent;
                                border-top: 15px solid black;
                                transform: translateX(-50%);
                            "></div>
                            <div style="
                                position: absolute; 
                                left: {marker_position}%; 
                                top: 35px; 
                                font-weight: bold; 
                                transform: translateX(-50%);
                            ">{prob_percent:.1f}%</div>
                        </div>
                    </div>
                    <br><br>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # GRÁFICO PCA (MAPA DE UBICACIÓN)
            if pca_model:
                st.subheader("Mapa de Situación (PCA)")
                st.caption("¿Dónde cae este tubo comparado con la historia? (Estrella vs Puntos)")
                
                # Procesar el punto nuevo igual que en el entrenamiento
                # Truco: Creamos un DF con el input y le calculamos lo que falta
                df_new = pd.DataFrame([input_data])
                df_new['Time_Years'] = df_new['Time_Years'].replace(0, 0.1)
                df_new['Corrosion_Rate_mm_y'] = df_new['Thickness_Loss_mm'] / df_new['Time_Years']
                
                # Transformar (Scale + OneHot + PCA)
                X_new_processed = preprocessor_model.transform(df_new)
                X_new_pca = pca_model.transform(X_new_processed)
                
                # Graficar
                # Puntos de fondo
                fig = px.scatter(
                    df_background, x='PC1', y='PC2', 
                    color='Estado', 
                    color_continuous_scale=['blue', 'red'],
                    opacity=0.3, # Transparentes para que no molesten
                    title="Historial de Tuberías"
                )
                
                # La Estrella (El caso actual)
                fig.add_scatter(
                    x=[X_new_pca[0,0]], 
                    y=[X_new_pca[0,1]], 
                    mode='markers',
                    marker=dict(size=25, color='yellow', symbol='star', line=dict(width=2, color='black')),
                    name='TUBO ACTUAL'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Error en conexión con API")
            
    except Exception as e:
        st.error(f"Error: {e}")
# Pies de página
st.markdown("---")
st.caption("Sistema desarrollado por Gerónimo Pautazzo- Ciencia de Datos & ML")

# # Para correr la App: python -m streamlit run dashboard/app.py