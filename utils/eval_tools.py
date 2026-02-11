import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluar_modelo(model, X_test, y_test, nombre_modelo):
    """
    Calcula métricas, imprime reporte y grafíca la matriz de confusión.
    Retorna un diccionario con las métricas clave.
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilidad de clase 1 (Falla)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"\nEVALUACIÓN: {nombre_modelo}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc_score:.4f}")
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()
    
    return {
        'Modelo': nombre_modelo,
        'Accuracy': acc, 
        'Recall': rec, 
        'F1': f1, 
        'AUC': auc_score,
        'y_prob': y_prob # Guardamos probas para curva ROC comparativa
    }



def evaluar_overfitting(model, X_train, y_train, X_test, y_test, nombre_modelo):
    """
    Calcula métricas en train y test para detectar overfitting o underfitting.
    Retorna un diccionario con todas las métricas.
    """

    # --- Predicciones ---
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob  = model.predict_proba(X_test)[:, 1]

    # --- Métricas ---
    metrics = {
        "Modelo": nombre_modelo,

        "Accuracy_train": accuracy_score(y_train, y_train_pred),
        "Accuracy_test":  accuracy_score(y_test,  y_test_pred),

        "Recall_train": recall_score(y_train, y_train_pred),
        "Recall_test":  recall_score(y_test,  y_test_pred),

        "F1_train": f1_score(y_train, y_train_pred),
        "F1_test":  f1_score(y_test,  y_test_pred),

        "AUC_train": roc_auc_score(y_train, y_train_prob),
        "AUC_test":  roc_auc_score(y_test,  y_test_prob),
    }

    # --- Impresión ---
    print(f"\n Evaluación de Overfitting: {nombre_modelo}")
    print("-" * 50)
    for k, v in metrics.items():
        if k != "Modelo":
            print(f"{k}: {v:.4f}")
    print("-" * 50)

    # --- Diagnóstico automático ---
    print("\n🔍 Diagnóstico:")

    gap_auc = metrics["AUC_train"] - metrics["AUC_test"]
    gap_f1  = metrics["F1_train"] - metrics["F1_test"]

    if gap_auc > 0.10 or gap_f1 > 0.10:
        print(" Overfitting detectado: gran diferencia entre train y test.")
    elif metrics["AUC_train"] < 0.70 and metrics["AUC_test"] < 0.70:
        print(" Underfitting: el modelo no aprende patrones.")
    else:
        print(" Buen equilibrio: el modelo generaliza bien.")

    return metrics