import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
@st.cache_data
def load_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  # Normalize
    return x_test, y_test

x_test, y_test = load_data()

# Streamlit UI
st.title("🖋️ Handwritten Digit Viewer (MNIST)")
st.markdown("Select a digit (0–9) and see a handwritten version of it.")

digit = st.number_input("Choose a digit", min_value=0, max_value=9, value=0, step=1)

if st.button("Generate Image"):
    matches = x_test[y_test == digit]

    if len(matches) == 0:
        st.warning("No images found for that digit.")
    else:
        idx = np.random.randint(0, len(matches))
        img = matches[idx]

        # Display image
        st.image(img, width=150, caption=f"Handwritten digit: {digit}", clamp=True)
