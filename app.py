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
show_grid = st.sidebar.checkbox("Показать сетку", True)
show_streamlines = st.sidebar.checkbox("Показать линии тока", True)

# Масштаб стрелок для разных полей
if field_type == "Диполь":
    st.sidebar.markdown("💡 *Для диполя: уменьшите масштаб (5-15), чтобы стрелки стали длиннее*")
    arrow_scale = st.sidebar.slider("Масштаб стрелок (чем меньше, тем длиннее)", 2, 30, 10, 1)
else:
    arrow_scale = st.sidebar.slider("Множитель длины стрелок", 0.1, 2.0, 0.5, 0.05)

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

# Координатная сетка
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

# Более мелкая сетка для линий тока
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
    # Диполь с моментом M=1
    U = (X**2 - Y**2) / r4
    V = (2 * X * Y) / r4
    title = "Диполь (ось X)"
    r2s = Xs**2 + Ys**2 + eps
    r4s = r2s**2
    Us = (Xs**2 - Ys**2) / r4s
    Vs = (2 * Xs * Ys) / r4s

else:  # Равномерный поток
    alpha_rad = np.radians(flow_angle)
    U = np.cos(alpha_rad) * np.ones_like(X)
    V = np.sin(alpha_rad) * np.ones_like(Y)
    title = f"Равномерный поток (угол = {flow_angle}°), V₀ = 1"
    Us = np.cos(alpha_rad) * np.ones_like(Xs)
    Vs = np.sin(alpha_rad) * np.ones_like(Ys)

magnitude = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(10, 8))

# Отрисовка стрелок
q = ax.quiver(X, Y, U, V, magnitude,
              scale=arrow_scale,
              scale_units='xy',
              cmap='viridis',
              alpha=0.8,
              width=0.005)

plt.colorbar(q, ax=ax, label='Модуль скорости')

# Линии тока
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

with st.expander("📖 О текущем течении"):
    if field_type == "Источник/Сток":
        name = "Источник" if Q > 0 else "Сток"
        st.markdown(f"""
        **{name} (Q = {Q})**
        - **Потенциал скорости:** $\varphi = \frac{{Q}}{{2\pi}}\ln r$
        - **Функция тока:** $\psi = \frac{{Q}}{{2\pi}}\theta$
        - **Поле скоростей:** $u_r = \frac{{Q}}{{2\pi r}}$, $u_\theta = 0$
        - **Анализ:** {name} — радиальное течение. Линии тока — прямые из центра/к центру.
        """)
    elif field_type == "Вихрь":
        st.markdown(f"""
        **Вихрь (Γ = {Gamma})**
        - **Потенциал скорости:** $\varphi = \frac{{\Gamma}}{{2\pi}}\theta$
        - **Функция тока:** $\psi = -\frac{{\Gamma}}{{2\pi}}\ln r$
        - **Поле скоростей:** $u_r = 0$, $u_\theta = \frac{{\Gamma}}{{2\pi r}}$
        - **Анализ:** Круговое течение. Линии тока — концентрические окружности.
        """)
    elif field_type == "Вихреисточник":
        st.markdown(f"""
        **Вихреисточник (Q = {Q}, Γ = {Gamma})**
        - **Потенциал скорости:** $\varphi = \frac{{Q}}{{2\pi}}\ln r + \frac{{\Gamma}}{{2\pi}}\theta$
        - **Функция тока:** $\psi = \frac{{Q}}{{2\pi}}\theta - \frac{{\Gamma}}{{2\pi}}\ln r$
        - **Поле скоростей:** $u_r = \frac{{Q}}{{2\pi r}}$, $u_\theta = \frac{{\Gamma}}{{2\pi r}}$
        - **Анализ:** Спиралевидные линии тока. Суперпозиция источника и вихря.
        """)
    elif field_type == "Диполь":
        st.markdown(f"""
        **Диполь**
        - **Потенциал скорости:** $\varphi = \frac{{M}}{{2\pi}}\frac{{x}}{{x^2+y^2}}$
        - **Функция тока:** $\psi = -\frac{{M}}{{2\pi}}\frac{{y}}{{x^2+y^2}}$
        - **Поле скоростей:** $u_r = \frac{{M\cos\theta}}{{2\pi r^2}}$, $u_\theta = \frac{{M\sin\theta}}{{2\pi r^2}}$
        - **Анализ:** Замкнутые линии тока. В центре — особая точка.
        - **Совет:** Покрутите слайдер "Масштаб стрелок" — уменьшите до 5-10, чтобы стрелки стали длиннее.
        """)
    else:
        st.markdown(f"""
        **Равномерный поток (угол = {flow_angle}°)**
        - **Потенциал скорости:** $\varphi = V_0(x\cos\alpha_0 + y\sin\alpha_0)$
        - **Функция тока:** $\psi = V_0(y\cos\alpha_0 - x\sin\alpha_0)$
        - **Поле скоростей:** $u_x = V_0\cos\alpha_0$, $u_y = V_0\sin\alpha_0$
        - **Анализ:** Параллельные линии тока. Постоянная скорость.
        """)

with st.expander("ℹ️ О приложении"):
    st.markdown("""
    **Настройка диполя:**
    - Масштаб по умолчанию = **10** (стрелки нормальной длины)
    - **Уменьшите** масштаб (5-8) → стрелки станут **длиннее**
    - **Увеличьте** масштаб (15-30) → стрелки станут **короче**
    
    **Все типы полей:**
    - Источник/Сток – радиальное течение
    - Вихрь – круговое течение  
    - Вихреисточник – спирали
    - Диполь – замкнутые линии тока
    - Равномерный поток – параллельные линии
    """)
