import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Визуализация векторных полей", layout="wide")
st.title("Научная визуализация векторных полей")
st.markdown("Интерактивное приложение для визуализации двумерных векторных полей")

st.sidebar.header("Параметры визуализации")

field_type = st.sidebar.selectbox(
    "Тип векторного поля",
    ["Источник/Сток", "Вихрь", "Вихреисточник", "Диполь", "Равномерный поток"]
)

grid_size = st.sidebar.slider("Размер сетки (N x N)", 10, 50, 20)
arrow_scale = st.sidebar.slider("Множитель длины стрелок", 0.1, 1.5, 0.5)
show_grid = st.sidebar.checkbox("Показать сетку", True)
show_streamlines = st.sidebar.checkbox("Показать линии тока", True)

# Динамические параметры
if field_type in ["Источник/Сток", "Вихреисточник"]:
    Q = st.sidebar.slider("Мощность источника Q (+ источник, - сток)", -2.0, 2.0, 1.0, 0.1)
else:
    Q = 1.0

if field_type in ["Вихрь", "Вихреисточник"]:
    Gamma = st.sidebar.slider("Циркуляция Γ", -2.0, 2.0, 1.0, 0.1)
else:
    Gamma = 1.0

if field_type == "Равномерный поток":
    flow_angle = st.sidebar.slider("Угол равномерного потока (градусы)", 0, 360, 0)
else:
    flow_angle = 0

eps = 1e-6

x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

x_stream = np.linspace(-3, 3, 100)
y_stream = np.linspace(-3, 3, 100)
Xs, Ys = np.meshgrid(x_stream, y_stream)

# Расчёт полей
if field_type == "Вихрь":
    r2 = X**2 + Y**2 + eps
    U = -Gamma * Y / (2*np.pi * r2)
    V =  Gamma * X / (2*np.pi * r2)
    title = f"Вихрь (Γ = {Gamma})"
    r2s = Xs**2 + Ys**2 + eps
    Us = -Gamma * Ys / (2*np.pi * r2s)
    Vs =  Gamma * Xs / (2*np.pi * r2s)

elif field_type == "Источник/Сток":
    r2 = X**2 + Y**2 + eps
    U = Q * X / (2*np.pi * r2)
    V = Q * Y / (2*np.pi * r2)
    type_name = "Источник" if Q > 0 else "Сток"
    title = f"{type_name} (Q = {Q})"
    r2s = Xs**2 + Ys**2 + eps
    Us = Q * Xs / (2*np.pi * r2s)
    Vs = Q * Ys / (2*np.pi * r2s)

elif field_type == "Вихреисточник":
    r2 = X**2 + Y**2 + eps
    U = (Q * X - Gamma * Y) / (2*np.pi * r2)
    V = (Q * Y + Gamma * X) / (2*np.pi * r2)
    title = f"Вихреисточник (Q = {Q}, Γ = {Gamma})"
    r2s = Xs**2 + Ys**2 + eps
    Us = (Q * Xs - Gamma * Ys) / (2*np.pi * r2s)
    Vs = (Q * Ys + Gamma * Xs) / (2*np.pi * r2s)

elif field_type == "Диполь":
    r2 = X**2 + Y**2 + eps
    r4 = r2**2
    U = (X**2 - Y**2) / r4
    V = (2 * X * Y) / r4
    title = "Диполь (ось X)"
    r2s = Xs**2 + Ys**2 + eps
    r4s = r2s**2
    Us = (Xs**2 - Ys**2) / r4s
    Vs = (2 * Xs * Ys) / r4s

else:
    alpha_rad = np.radians(flow_angle)
    U = np.cos(alpha_rad) * np.ones_like(X)
    V = np.sin(alpha_rad) * np.ones_like(Y)
    title = f"Равномерный поток (угол = {flow_angle}°), V₀ = 1"
    Us = np.cos(alpha_rad) * np.ones_like(Xs)
    Vs = np.sin(alpha_rad) * np.ones_like(Ys)

magnitude = np.sqrt(U**2 + V**2)

# Для диполя: ограничиваем максимальную длину стрелок
if field_type == "Диполь":
    max_arrow_length = 2.0  # максимальная длина стрелки в координатах
    # Нормируем слишком длинные векторы
    length = np.sqrt(U**2 + V**2)
    scale_factor = np.where(length > max_arrow_length, max_arrow_length / length, 1.0)
    U_plot = U * scale_factor
    V_plot = V * scale_factor
    mag_plot = magnitude  # цвет оставляем по реальному модулю
else:
    U_plot, V_plot, mag_plot = U, V, magnitude

fig, ax = plt.subplots(figsize=(10, 8))

q = ax.quiver(X, Y, U_plot, V_plot, mag_plot,
              scale=arrow_scale,
              scale_units='xy',
              cmap='viridis',
              alpha=0.8,
              width=0.005)

plt.colorbar(q, ax=ax, label='Модуль скорости')

if show_streamlines:
    ax.streamplot(Xs, Ys, Us, Vs,
                  color='red',
                  linewidth=0.8,
                  density=1.5,
                  arrowsize=0.8)

ax.set_title(title, fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
if show_grid:
    ax.grid(True, linestyle='--', alpha=0.5)
ax.axis("equal")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

st.pyplot(fig)

if st.button("Сохранить изображение как PNG"):
    fig.savefig("vector_field.png", dpi=150)
    st.success("Изображение сохранено как vector_field.png")

with st.expander("О приложении"):
    st.markdown(r"""
    **Доступные типы полей:**
    - **Источник/Сток** – радиальное течение (Q>0 – источник, Q<0 – сток)
    - **Вихрь** – круговое течение, линии тока – окружности
    - **Вихреисточник** – суперпозиция источника и вихря (логарифмические спирали)
    - **Диполь** – поле диполя, замкнутые линии тока
    - **Равномерный поток** – постоянная скорость под заданным углом

    **Особенности:**
    - Длина стрелки пропорциональна модулю скорости
    - Цвет стрелок соответствует модулю скорости
    - Линии тока (красные) строятся численным интегрированием
    - **Для диполя** длина стрелок ограничена (не более 2.0), чтобы избежать огромных стрелок в центре
    """)
