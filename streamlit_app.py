import streamlit as st

# Configuración general de la página (se aplica a todas)
st.set_page_config(page_title="TFmicro ejemplos", layout="wide")

# --- DEFINICIÓN DE LAS PÁGINAS ---

# 1. Página de Inicio (Bienvenida)
pg_inicio = st.Page("intro.py", title="Inicio")

# 2. Proyecto 1: Reconocedor de Dígitos
# Apunta a los archivos donde tienes tu código
pg_digitos_eda = st.Page("proyecto_digitos/eda.py", title="1. Análisis Exploratorio")
pg_digitos_train = st.Page("proyecto_digitos/train.py", title="2. Entrenamiento" )
pg_digitos_inf = st.Page("proyecto_digitos/inferencia.py", title="3. Inferencia (Dibujar)")

# 3. Proyecto 2: (Ejemplo futuro)
pg_otro_eda = st.Page("proyecto_extra/ejemplo1.py", title="Vista Ejemplo A")
pg_otro_res = st.Page("proyecto_extra/ejemplo2.py", title="Vista Ejemplo B")


pg = st.navigation(
    {
        "General": [pg_inicio],
        "Proyecto: Dígitos": [pg_digitos_eda, pg_digitos_train, pg_digitos_inf],
        "Proyecto: Otro Ejemplo": [pg_otro_eda, pg_otro_res],
    }
)

# --- EJECUTAR LA NAVEGACIÓN ---
pg.run()