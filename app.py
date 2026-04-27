import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="Визуализация векторных полей", layout="wide")
st.title("🧲 Научная визуализация векторных полей")
st.markdown("Интерактивное приложение для визуализации двумерных векторных полей")

# Боковая панель с параметрами
st.sidebar.header("Параметры визуализации")

# Выбор типа поля
field_type = st.sidebar.selectbox(
    "Тип векторного поля",
    ["Вихрь", "Источник", "Сток", "Диполь", "Равномерный поток"]
)

# Параметры сетки
grid_size = st.sidebar.slider("Размер сетки (N x N)", 10, 50, 20)
arrow_scale = st.sidebar.slider("Масштаб стрелок", 0.1, 2.0, 1.0)
color = st.sidebar.color_picker("Цвет стрелок", "#1f77b4")
show_grid = st.sidebar.checkbox("Показать сетку", True)

# Создание координатной сетки
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

# Определение компонент векторного поля
if field_type == "Вихрь":
    U = -Y
    V = X
    title = "Вихревое поле (Vortex)"
elif field_type == "Источник":
    r = np.sqrt(X**2 + Y**2) + 0.001
    U = X / r
    V = Y / r
    title = "Поле источника (Source)"
elif field_type == "Сток":
    r = np.sqrt(X**2 + Y**2) + 0.001
    U = -X / r
    V = -Y / r
    title = "Поле стока (Sink)"
elif field_type == "Диполь":
    r2 = X**2 + Y**2 + 0.001
    U = (X**2 - Y**2) / r2
    V = 2 * X * Y / r2
    title = "Дипольное поле (Dipole)"
else:  # Равномерный поток
    U = np.ones_like(X)
    V = np.zeros_like(Y)
    title = "Равномерный поток (Uniform flow)"

# Визуализация
fig, ax = plt.subplots(figsize=(10, 8))
q = ax.quiver(X, Y, U, V, scale=arrow_scale, color=color, alpha=0.8)
ax.set_title(title, fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
if show_grid:
    ax.grid(True, linestyle='--', alpha=0.5)
ax.axis("equal")

# Добавление цветовой шкалы для модуля вектора
magnitude = np.sqrt(U**2 + V**2)
im = ax.scatter(X, Y, c=magnitude, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(im, ax=ax, label='Модуль вектора')

st.pyplot(fig)

# Кнопка сохранения
if st.button("💾 Сохранить изображение как PNG"):
    fig.savefig("vector_field.png", dpi=150)
    st.success("Изображение сохранено как vector_field.png")

# Информация о приложении
with st.expander("ℹ️ О приложении"):
    st.markdown("""
    **Доступные типы полей:**
    - **Вихрь** – вращение вокруг центра
    - **Источник** – векторное поле, расходящееся от центра
    - **Сток** – векторное поле, сходящееся к центру
    - **Диполь** – комбинация источника и стока
    - **Равномерный поток** – постоянное направление
    
    **Управление:** используйте боковую панель для изменения плотности сетки, масштаба стрелок, цвета и отображения сетки.
    """)
