import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

st.title("🤖 Entrenamiento del Modelo")

# 1. Carga de datos
digits = load_digits()
X, y = digits.data, digits.target

# 2. Configuración del Pipeline (Sidebar)
st.sidebar.header("Hiperparámetros")
split_size = st.sidebar.slider("Tamaño del set de prueba (%)", 10, 50, 20)
seed = st.sidebar.number_input("Semilla aleatoria (Seed)", 42)
model_type = st.sidebar.selectbox("Seleccionar Algoritmo", ["Regresión Logística", "Árbol de Decisión"])

# 3. Botón de Entrenar
if st.button("Iniciar Entrenamiento"):
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size/100, random_state=seed
    )
    
    # Selección de Modelo
    if model_type == "Regresión Logística":
        model = LogisticRegression(max_iter=1000)
    else:
        model = DecisionTreeClassifier(random_state=seed)
    
    # Entrenar
    with st.spinner('Entrenando modelo...'):
        model.fit(X_train, y_train)
    
    # Predicción y Métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.success(f"Modelo entrenado con éxito. Accuracy: {acc:.4f}")
    
    # Matriz de Confusión
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    # Guardar el modelo en la carpeta 'models'
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/modelo_digitos.joblib')
    st.info("Modelo guardado en 'models/modelo_digitos.joblib' para usar en inferencia.")