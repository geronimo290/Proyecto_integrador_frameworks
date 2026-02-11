# API

import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#----------#
# Configuracion e inicio


# Crear App
app = FastAPI(
    title='API de Prediccion de Fallas',
    version='0.2'
)

# Definicion de la ruta

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #carpeta donde está este archivo (api/)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models') #es la carpeta hermana 'models/' (subimos un nivel con ..)

# Variables para guardar los modelos
models = {}
preprocessor = None
model_columns = None

#----------#
# Carga de modelos 
print('Cargando modelos...')
try:
    # Modelos
    models['Logistic Regression'] = joblib.load(os.path.join(MODELS_DIR, 'logistic_model.pkl'))
    models['Random Forest'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    models['XGBoost'] = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))

    #Preproesador
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))

    print('Sistema cargado exitosamente.')
except Exception as e:
    print(f"ERROR CRÍTICO AL CARGAR MODELOS: {e}")
    # Si falla esto, la API no sirve, así que imprimimos el error fuerte.

#----------#
# Esquema (Filtro de seguridad, si los datos no coiciden, la API rechaza el pediddo)

class PipelineIntput(BaseModel):

    # Datos Fisicos
    Pipe_Size_mm: float             
    Thickness_mm: float             
    Thickness_Loss_mm: float        
    Max_Pressure_psi: float         
    Temperature_C: float            
    
    # Datos de Tiempo y Estado
    Time_Years: int                 
    Corrosion_Impact_Percent: float 
    Material_Loss_Percent: float    
    
    # Datos de Texto (Categorías)
    Condition: str                  
    Material: str                   
    Grade: str                      


    # Ejemplo de la documentacion autoatica
    class Config:
        schema_extra = {
            "example": {
                "Pipe_Size_mm": 200.0,
                "Thickness_mm": 12.0,
                "Thickness_Loss_mm": 2.5,
                "Max_Pressure_psi": 600.0,
                "Temperature_C": 40.0,
                "Time_Years": 10,
                "Corrosion_Impact_Percent": 5.0,
                "Material_Loss_Percent": 20.0,
                "Condition": "Normal",
                "Material": "Carbon Steel",
                "Grade": "ASTM A106 Grade B"
            }
        }

#----------#
# Ruta de prueba

@app.post('/predict/{model_name}')
def predict_pipeline(model_name: str, data: PipelineIntput):
    """
    Recibe los datos del pipeline y el nombre del modelo (XGBoost, Random Forest, Logistic Regresion).
    Devuelve la probabilidad de falla.
    """

    # Validacion de existencia del modelo
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado. Opciones: {list(models.keys())}")
    
    # Convertir JSON a DataFrame
    df_input = pd.DataFrame([data.dict()]) # data.dict() convierte el input de Pydantic a un diccionario de Python

    # Feature Engineering mismo proceso que en el entrenamiento
    df_input['Time_Years'] = df_input['Time_Years'].replace(0, 0.1) # Evitar div por 0
    df_input['Corrosion_Rate_mm_y'] = df_input['Thickness_Loss_mm'] / df_input['Time_Years']
    
    # Preprocesamiento (Escalado + OneHot)
    try:
        X_processed = preprocessor.transform(df_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transformar datos: {str(e)}")

    # Prediccion
    selected_model = models[model_name]

    # Calculamos probabilidad con predict_proba (clase 1 = falla)
    probability = float(selected_model.predict_proba(X_processed)[0, 1])

    # Decicion basada en umbral (50%)
    prediction = "FALLA" if probability > 0.5 else "SEGURO"

    # Respuesta JSON
    return {
        "model_used": model_name,
        "failure_probability": round(probability, 4), # 4 decimales
        "prediction_text": prediction,
        "risk_percentage": f"{probability * 100:.2f}%"
    }
@app.get('/')
def home():
    return {"status": "online", "models_loaded": list(models.keys())}

# Para correr la API: python -m uvicorn api.main:app --reload