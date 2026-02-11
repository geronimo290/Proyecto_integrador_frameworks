# Simulación de bases de datos para hacer consulta SQL

import pandas as pd
import sqlite3
import os

# Configuración de rutas
CSV_PATH = "data/pipeline_data.csv" 
DB_PATH = "data/industria.db"

def crear_base_datos():
    # Chequeo de seguridad
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: No encuentro el archivo {CSV_PATH}")
        print("Asegurate de mover el archivo descargado de Kaggle a la carpeta 'data' y renombrarlo.")
        return

    print("🔄 Leyendo CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Limpieza de nombres de columnas (quitamos espacios)
    df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'Pct') for c in df.columns]
    
    print(f"📊 Columnas detectadas: {list(df.columns)}")

    # Conexión a SQLite
    print("🔌 Conectando a SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Guardar en SQL
    # 'pipelines' será el nombre de la tabla en la base de datos
    print("💾 Guardando datos en la tabla 'pipelines'...")
    df.to_sql("pipelines", conn, if_exists="replace", index=False)
    
    # 4. Verificación
    cursor.execute("SELECT count(*) FROM pipelines")
    filas = cursor.fetchone()[0]
    
    conn.close()
    print(f"✅ ¡ÉXITO! Base de datos creada en '{DB_PATH}' con {filas} registros.")
    print("---------------------------------------------------------")
    print("Ahora tus notebooks y tu API leerán de esta DB, no del CSV.")

if __name__ == "__main__":
    crear_base_datos()