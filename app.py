__author__ = "Marlon"
__Cop__ = "BrainiaC©"

import numpy as np
import cv2
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

st.title('Detector Numérico')

# Caminho do modelo salvo corretamente
model_path = os.path.join(os.path.dirname(__file__), 'model.keras')
weights_path = "weights.weights.h5"

# Carregar o modelo e os pesos
model = load_model(model_path)
model.load_weights(weights_path)

st.markdown('''
Tente Desenhar um Número!
''')

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=300,
    height=300,
    key='canvas'
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')  # Garante que a imagem seja uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # Converte para escala de cinza
    img = cv2.resize(img, (28, 28))  # Redimensiona para 28x28 pixels
    img = img / 255.0  # Normaliza os valores dos pixels
    img = img.reshape(1, 28, 28, 1)  # Adiciona dimensão extra para o modelo

    if st.button('Predict'):
        val = model.predict(img)
        st.write(f"""# Resultado: {np.argmax(val[0])}""")