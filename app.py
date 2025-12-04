%%writefile app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("My First AI App")
st.write("Upload any image and AI will identify it!")

@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner("AI reasoning loaded"):
    model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        with st.spinner('AI thinking...'):
            img = image_data.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            predictions = model.predict(x)
            results = decode_predictions(predictions, top=3)[0]
            st.success("Prediction Complete")
            for result in results:
                st.write(f"**{result[1]}** : {result[2]*100:.2f}%")