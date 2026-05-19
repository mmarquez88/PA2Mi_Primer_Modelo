import streamlit as st
import pandas as pd
import joblib

# =========================
# TITULO
# =========================
st.title("🍷 Clasificador de Calidad de Vino")

st.write("Ingrese las características del vino para predecir su calidad.")

# =========================
# CARGAR MODELOS
# =========================
logistic_model = joblib.load("model1/logistic_regression_model.pkl")
random_forest_model = joblib.load("model1/Ramdon_Forest_model.pkl")

# =========================
# SELECCIONAR MODELO
# =========================
modelo_seleccionado = st.selectbox(
    "Seleccione el modelo",
    ("Logistic Regression", "Random Forest")
)

# =========================
# INPUTS USUARIO
# =========================
fixed_acidity = st.number_input(
    "Fixed Acidity",
    min_value=0.0,
    step=0.1
)

volatile_acidity = st.number_input(
    "Volatile Acidity",
    min_value=0.0,
    step=0.01
)

citric_acid = st.number_input(
    "Citric Acid",
    min_value=0.0,
    step=0.01
)

residual_sugar = st.number_input(
    "Residual Sugar",
    min_value=0.0,
    step=0.1
)

# =========================
# BOTON PREDICCION
# =========================
if st.button("Obtener Predicción"):

    # Crear dataframe
    datos = pd.DataFrame({
        "fixed_acidity": [fixed_acidity],
        "volatile_acidity": [volatile_acidity],
        "citric_acid": [citric_acid],
        "residual_sugar": [residual_sugar]
    })

    # Elegir modelo
    if modelo_seleccionado == "Logistic Regression":
        prediccion = logistic_model.predict(datos)[0]
    else:
        prediccion = random_forest_model.predict(datos)[0]

    # Mostrar predicción
    st.subheader(f"Calidad Predicha: {prediccion}")

    # Mensaje según calidad
    if prediccion > 7:
        st.success("🍷 Excelente calidad")

    elif prediccion > 4 and prediccion < 8:
        st.warning("🍷 Calidad regular")

    else:
        st.error("🍷 Calidad baja")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("### Elaborado por Marcelo Marquez")
