import streamlit as st
import joblib
import numpy as np
import cv2  # OpenCV para procesamiento de imágenes
from streamlit_drawable_canvas import st_canvas
import os

st.title("Demo")

# 1. Cargar el Modelo
if not os.path.exists('models/modelo_digitos.joblib'):
    st.error("⚠️ Primero debes entrenar el modelo en la página de Entrenamiento.")
    st.stop()

model = joblib.load('models/modelo_digitos.joblib')
st.success("Modelo cargado. Dibuja un dígito en el recuadro negro.")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### Tu Dibujo")
    # Crear el lienzo para dibujar
    # Usamos fondo negro y trazo blanco porque así suelen venir los datasets de MNIST/Digits
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno (no se usa aquí)
        stroke_width=15,                      # Grosor del pincel (importante para que al reducir no desaparezca)
        stroke_color="#FFFFFF",               # Color del trazo (Blanco)
        background_color="#000000",           # Fondo (Negro)
        height=600, width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    # Obtener la imagen dibujada (viene en formato RGBA de alta resolución)
    img_data = canvas_result.image_data.astype('uint8')
    
    # --- PIPELINE DE PREPROCESAMIENTO (Bloque 1 y 5) ---
    
    # 1. Convertir a escala de grises (eliminamos canales de color y transparencia)
    # OpenCV usa BGR, pero el canvas suele dar RGBA. Convertimos a Grayscale.
    gray_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    
    # 2. Redimensionar a 8x8 píxeles
    # El modelo fue entrenado con imágenes de 8x8. Si no reducimos, fallará.
    # Usamos interpolación AREA que es mejor para reducir imágenes sin perder info.
    resized_image = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA)
    
    # 3. Escalar valores (Normalización)
    # El dibujo original es 0-255. El dataset load_digits de sklearn es 0-16.
    # Hacemos una regla de tres para ajustar la escala.
    processed_image = (resized_image / 255.0) * 16.0
    
    # 4. Aplanar (Flatten)
    # El modelo espera un array de 1 fila y 64 columnas (1, 64)
    final_input = processed_image.reshape(1, -1)

    with col2:
        st.write("### Lo que ve el modelo")
        st.write("Al reducir a 8x8 píxeles:")
        
        # Mostramos la imagen pixelada para entender qué está procesando la IA
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(2,2))
        ax.imshow(processed_image, cmap='gray_r') # gray_r invierte para ver negro sobre blanco
        ax.axis('off')
        st.pyplot(fig)

    # Botón de Predicción
    if st.button("Predecir Dibujo"):
        try:
            prediction = model.predict(final_input)[0]
            st.markdown(f"## 🔮 Predicción: **{prediction}**")
            
            # Opcional: Mostrar probabilidades si el modelo lo permite
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(final_input)[0]
                confidence = proba[prediction] * 100
                st.write(f"Confianza: {confidence:.2f}%")
                
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

st.info("""
**Nota técnica:** El dataset `load_digits` usa imágenes de muy baja resolución (8x8). 
Es normal que le cueste reconocer trazos complejos. Intenta dibujar el número grande y centrado.
""")