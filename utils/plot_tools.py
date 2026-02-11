#Herramientas EDA
#Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
############################################## EDA ##############################################

#Auditoria de datos Muestra estadisticas descriptivas y variables categoricas
def auditoria_de_datos(df):
    """
    Recibe un DataFrame y devuelve un reporte estadistico con:
    - Informacion del Dataset
    - Shape del Dataset
    - Duplicados
    - Valores Nulos
    - Estadisticas (numericas)
    - Variables Categoricas
    """
    #Info y shape
    print("Informacion del Dataset:")
    df.info()
    print(f"\nShape del Dataset: {df.shape}")
    
    #Duplicados
    duplicados = df.duplicated().sum()
    print(f"Filas Duplicadas: {duplicados}")

    #Valores Nulos
    print("\nNulos por Columnas (:%)")
    nulos = df.isnull().mean() * 100
    print(nulos[nulos > 0]) #Solo columnas con nulos

    #Estadisticas (numericas)
    numericas = df.select_dtypes(include=["number"])

    if not numericas.empty:
        stats = numericas.describe().T

        #Asimetria y Curtosis
        stats["skew"] =  numericas.skew()
        stats["kurtosis"] = numericas.kurt()

        #Mediana (para comparar con la media)
        stats["median"] = numericas.median()

        #Outliers (porcentaje de valores fuera del rango IQR)
        q1 = numericas.quantile(0.25)
        q3 = numericas.quantile(0.75)
        iqr = q3 - q1
        stats["outliers"] = ((numericas < (q1 - 1.5 * iqr)) | (numericas > (q3 + 1.5 * iqr))).mean() * 100

        #Varianza
        stats["variance"] = numericas.var()

        #Variables Categoricas
        categoricas = df.select_dtypes(include=["object", "category"])

        if not categoricas.empty:
            print("\nVariables Categoricas:")
            for col in categoricas.columns:
                print(f"\n{col}:")
                print(categoricas[col].value_counts())
                print(f"\nFrecuencia Relativa (%):")
                print(categoricas[col].value_counts(normalize=True) * 100)

        # Coeficiente de Variación (CV) = (std / mean) * 100
        # Mide la dispersión relativa: > 100% indica datos muy volátiles
        stats["cv"] = (stats["std"] / (stats["mean"].abs() + 0.0001)) * 100

        #Ordenamos las columnas 
        cols_finales = ['mean', 'median', 'std', 'min', 'max', "cv",'skew', 'kurtosis', 'outliers', "variance"]

        print("\nReporte estadistico final")
        return stats[cols_finales]
    else:
        print("\nNo hay columnas numericas para analizar")
        return None

def diagnostico_auditoria(df, stats_numericas):
    """
    Recibe el DF original y el DF de estadísticas numéricas (output de auditoria_de_datos).
    Devuelve un DataFrame con recomendaciones de acción para cada variable.
    """
    reporte_diagnostico = []

    # ANÁLISIS DE VARIABLES NUMÉRICAS
    if stats_numericas is not None:
        for col, row in stats_numericas.iterrows():
            estado = []
            accion = []
            prioridad = "Baja"

            #Criterio de Varianza (Variable Constante)
            if row['std'] == 0:
                estado.append("CONSTANTE")
                accion.append("Eliminar Columna")
                prioridad = "Nula"
            
            else:
                #Criterio de Asimetría (Skewness)
                #Si es mayor a 1 o menor a -1, es fuerte.
                if abs(row['skew']) > 0.9:
                    estado.append("Muy Sesgada")
                    accion.append("Aplicar Log/PowerTransformer")
                    prioridad = "Alta"
                elif abs(row['skew']) > 0.5:
                    estado.append("Leve Sesgo")
                
                #Criterio de Coeficiente de Variación (CV)
                if row['cv'] > 100: 
                    estado.append(f"Muy Volátil (CV={row['cv']:.0f}%)")
                    prioridad = "Revisar"
                elif row['cv'] < 5:
                    estado.append("Muy Estable")

                #Criterio de Outliers
                if row['outliers'] > 5.0: # Más del 5% son outliers
                    estado.append("Muchos Outliers")
                    accion.append("Graficar Boxplot + Investigar")
                    prioridad = "Alta"
                
                #Comparación Media vs Mediana (Divergencia)
                #Si la diferencia es mayor al 10% de la media
                diferencia_pct = abs(row['mean'] - row['median']) / (row['mean'] + 0.0001)
                if diferencia_pct > 0.10:
                    estado.append("Media alejada de Mediana")

                #Acción por defecto si no es constante
                if "Alta" in prioridad:
                    accion.insert(0, "Graficar Histograma")
                else:
                    accion.append("Graficar Histograma (Verificar)")
                    prioridad = "Media"

            reporte_diagnostico.append({
                "Variable": col,
                "Tipo": "Numérica",
                "Estado": ", ".join(estado) if estado else "Normal",
                "Acción Recomendada": " + ".join(list(set(accion))), # Evita duplicados
                "Prioridad": prioridad
            })

    # ANÁLISIS DE VARIABLES CATEGÓRICAS
    categoricas = df.select_dtypes(include=["object", "category"])
    
    for col in categoricas.columns:
        estado = []
        accion = []
        prioridad = "Media"
        
        # Calculamos balance
        conteo = df[col].value_counts(normalize=True)
        top_clase_pct = conteo.iloc[0] # Porcentaje de la clase más frecuente
        cardinalidad = df[col].nunique() # Cantidad de categorías únicas

        #Criterio de Dominancia (Desbalanceo extremo)
        if top_clase_pct > 0.90:
            estado.append(f"Casi Constante ({top_clase_pct:.1%} un valor)")
            accion.append("Posible Eliminar")
            prioridad = "Baja"
        elif top_clase_pct > 0.45:
            estado.append("Desbalanceada")
            accion.append("Graficar Countplot")
            prioridad = "Alta"
        else:
            estado.append("Balanceada")
            accion.append("Graficar Countplot")

        #Criterio de Cardinalidad (Demasiadas categorías)
        if cardinalidad > 50:
            estado.append("Alta Cardinalidad")
            accion.append("Agrupar categorías (Top 10)")
            prioridad = "Revisar"

        reporte_diagnostico.append({
            "Variable": col,
            "Tipo": "Categórica",
            "Estado": ", ".join(estado),
            "Acción Recomendada": " + ".join(accion),
            "Prioridad": prioridad
        })

    #Convertimos a DataFrame
    df_diag = pd.DataFrame(reporte_diagnostico)
    
    #Ordenamos por prioridad 
    mapa_prioridad = {"Alta": 1, "Revisar": 2, "Media": 3, "Baja": 4, "Nula": 5}
    df_diag['Sort_Key'] = df_diag['Prioridad'].map(mapa_prioridad)
    return df_diag.sort_values('Sort_Key').drop(columns=['Sort_Key']).reset_index(drop=True)

def graficar_variables_estrategicas(df, df_diagnostico):
    """
    Recorre el DataFrame de diagnóstico y genera los gráficos correspondientes
    según la columna 'Acción Recomendada'.
    """
    # Estilo visual
    sns.set_style("whitegrid")
    
    # Filtramos las variables que se marcaron para eliminar (Prioridad Nula)
    variables_a_graficar = df_diagnostico[df_diagnostico['Prioridad'] != 'Nula']

    print(f"Generando gráficos para {len(variables_a_graficar)} variables relevantes...\n")

    for index, row in variables_a_graficar.iterrows():
        col = row['Variable']
        accion = row['Acción Recomendada']
        tipo = row['Tipo']
        
        # LÓGICA PARA VARIABLES NUMÉRICAS
        if tipo == "Numérica":
            # Si pide Histograma Y Boxplot 
            if "Histograma" in accion and "Boxplot" in accion:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histograma
                sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
                axes[0].set_title(f'Histograma: {col}')
                
                # Boxplot
                sns.boxplot(x=df[col], ax=axes[1], color='salmon')
                axes[1].set_title(f'Boxplot: {col} (Detectando Outliers)')
                
                plt.suptitle(f"Análisis de {col} - Estado: {row['Estado']}", fontsize=14)
                plt.tight_layout()
                plt.show()

            # Pide Histograma (caso de sesgos leves o normalidad)
            elif "Histograma" in accion:
                plt.figure(figsize=(8, 4))
                sns.histplot(df[col], kde=True, color='skyblue')
                plt.title(f'Distribución de {col} - Estado: {row["Estado"]}')
                plt.show()

        # LÓGICA PARA VARIABLES CATEGÓRICAS
        elif tipo == "Categórica":
            if "Countplot" in accion:
                plt.figure(figsize=(10, 5))
                
                # Ordenamos las barras por cantidad 
                order = df[col].value_counts().index[:15] # Limitamos a top 15 si hay muchas
                
                sns.countplot(y=df[col], order=order, palette="viridis", hue=df[col])
                plt.title(f'Frecuencia de Categorías: {col} - Estado: {row["Estado"]}')
                plt.xlabel("Cantidad")
                plt.show()
    
    print("Generación de gráficos finalizada.")

def transformacion_log(df, col, new_col=None):
    """
    Aplica log1p a una columna, grafica antes/después y muestra métricas.
    
    Parámetros:
    df: DataFrame
    col: str -> nombre de la columna original
    new_col: str -> nombre de la columna transformada (opcional)
    """
    
    if new_col is None:
        new_col = col + "_log"
    
    # Transformación
    df[new_col] = np.log1p(df[col])
    
    # Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[col], kde=True, ax=ax[0], color='red')
    ax[0].set_title(f"Original: {col}")

    sns.histplot(df[new_col], kde=True, ax=ax[1], color='green')
    ax[1].set_title(f"Transformada (Log): {new_col}")

    plt.show()
    
    # Métricas
    skew_orig = df[col].skew()
    skew_log = df[new_col].skew()

    kurt_orig = df[col].kurt()
    kurt_log = df[new_col].kurt()

    cv_orig = df[col].std() / df[col].mean()
    cv_log = df[new_col].std() / df[new_col].mean()

    print(f"Skew Original: {skew_orig:.2f}")
    print(f"Skew Log: {skew_log:.2f}")
    print(f"Kurtosis Original: {kurt_orig:.2f}")
    print(f"Kurtosis Log: {kurt_log:.2f}")
    print(f"CV Original: {cv_orig:.2f}")
    print(f"CV Log: {cv_log:.2f}")


############################################## Preprocesamiento ##############################################

def graficar_control_calidad(y_train, y_test, X_train_df, df_original):


    """
    Esta funcion:
    - Detecta automáticamente todas las columnas transformadas
    - Identifica su columna original
    - Soporta sufijos comunes:
    - _log
    - _wins
    - _wins_log
    - _sqrt
    - _boxcox
    - _yeojohnson
    - Genera un gráfico por cada par original–transformada
    - Además genera los gráficos de split y boxplots

    Args:
        y_train, y_test: Serie con el target del set de entrenamiento y test.
        X_train_df: DataFrame escalado para verificar StandardScaler.
        df_original: DataFrame crudo para comparar distribución.
    """

    print("Generando Gráficos de Control de Calidad...")

    # ============================================================
    # 1) Gráfico de Split Estratificado
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    plt.subplots_adjust(hspace=0.35)

    df_train_dist = y_train.value_counts(normalize=True).reset_index()
    df_train_dist['Set'] = 'Train'
    df_test_dist = y_test.value_counts(normalize=True).reset_index()
    df_test_dist['Set'] = 'Test'
    df_dist = pd.concat([df_train_dist, df_test_dist])

    sns.barplot(
        data=df_dist,
        x=df_train_dist.columns[0],
        y=df_train_dist.columns[1],
        hue='Set',
        ax=axes[0],
        palette='viridis'
    )
    axes[0].set_title("Proporción de Target en Train vs Test")
    axes[0].set_ylabel("Porcentaje")
    axes[0].text(
        0, 0.5, "¡Deben ser iguales!",
        ha='center', color='red', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    # ============================================================
    # 2) Boxplot de Variables Escaladas
    # ============================================================
    cols_to_plot = X_train_df.columns[:10]
    X_melted = X_train_df[cols_to_plot].melt(
        var_name='Variable',
        value_name='Valor Escalado'
    )

    sns.boxplot(
        data=X_melted,
        x='Variable',
        y='Valor Escalado',
        ax=axes[1],
        palette='coolwarm'
    )
    axes[1].set_title("Distribución de Variables Escaladas (Train)")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(-5, 5)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

    # ============================================================
    # 3) Detectar columnas transformadas automáticamente
    # ============================================================
    patrones = ["_log", "_wins", "_wins_log", "_sqrt", "_boxcox", "_yeojohnson"]

    columnas_transformadas = [
        col for col in X_train_df.columns
        if any(col.endswith(p) for p in patrones)
    ]

    # ============================================================
    # 4) Graficar cada transformación original vs transformada
    # ============================================================
    for col_t in columnas_transformadas:
        # Recuperar nombre original removiendo el sufijo
        col_o = re.sub(r"(_log|_wins|_wins_log|_sqrt|_boxcox|_yeojohnson)$", "", col_t)

        if col_o not in df_original.columns:
            print(f"⚠ Advertencia: No se encontró la columna original para {col_t}")
            continue

        plt.figure(figsize=(10, 5))
        sns.kdeplot(X_train_df[col_t], color='red', fill=True, label='Transformada')
        sns.kdeplot(df_original[col_o], color='grey', linestyle='--', label='Original')

        plt.title(f"Transformación: {col_t}")
        plt.legend()
        plt.show()

    print("\nInterpretación:")
    print("1. Barras iguales: El split estratificado funcionó.")
    print("2. Cajas alineadas en 0: El StandardScaler funcionó.")
    print("3. Curvas rojas más suaves: Las transformaciones funcionaron.")