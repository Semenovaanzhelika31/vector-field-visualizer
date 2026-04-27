import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Интерактивная визуализация функции")

function_type = st.selectbox("Выберите функцию", ["sin(x)", "cos(x)", "x^2"])

x = np.linspace(-5, 5, 100)
if function_type == "sin(x)":
    y = np.sin(x)
elif function_type == "cos(x)":
    y = np.cos(x)
else:
    y = x**2

fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid(True)

st.pyplot(fig)
