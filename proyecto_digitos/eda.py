import streamlit as st
from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Análisis Exploratorio de Datos (EDA)")

# Cargar datos (Cacheado para eficiencia)
@st.cache_data
def cargar_datos():
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    df['target'] = digits.target
    return df, digits.images

df, images = cargar_datos()

st.write(f"**Dimensiones del dataset:** {df.shape[0]} filas, {df.shape[1]} columnas.")

# Mostrar tabla
if st.checkbox("Mostrar datos crudos (DataFrame)"):
    st.dataframe(df.head())

# Visualización de Clases
st.subheader("Distribución de Clases")
distribucion = df['target'].value_counts().sort_index()
st.bar_chart(distribucion)

# Visualización de Imágenes
st.subheader("Visualización de los píxeles")
st.write("Así es como la computadora 've' los números (matrices 8x8):")

idx = st.slider("Selecciona un índice de imagen para visualizar:", 0, len(df)-1, 0)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.imshow(images[idx], cmap='gray_r')
    ax.set_title(f"Etiqueta Real: {df.iloc[idx]['target']}")
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.write("Matriz numérica (escala de grises 0-16):")
    st.write(images[idx])