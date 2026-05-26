import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Визуализация векторных полей", layout="wide")
st.title("Научная визуализация векторных полей идеальной жидкости")
st.markdown("Интерактивное приложение для визуализации двумерных векторных полей")

st.sidebar.header("Параметры визуализации")

# Типы полей (объединён источник и сток)
field_type = st.sidebar.selectbox(
    "Тип векторного поля",
    ["Источник/Сток", "Вихрь", "Вихреисточник", "Диполь", "Равномерный поток"]
)

grid_size = st.sidebar.slider("Размер сетки (N x N)", 10, 40, 20)
arrow_density = st.sidebar.slider("Густота стрелок (каждая N-я точка)", 1, 3, 1)
show_grid = st.sidebar.checkbox("Показать сетку", True)
show_streamlines = st.sidebar.checkbox("Показать линии тока", True)
separate_plots = st.sidebar.checkbox("Раздельные графики (векторы и линии тока отдельно)", False)

# Параметры течений
Q = st.sidebar.slider("Мощность источника Q (положит. - источник, отриц. - сток)", -2.0, 2.0, 1.0, 0.1)
Gamma = st.sidebar.slider("Циркуляция Γ", -2.0, 2.0, 1.0, 0.1)
flow_angle = st.sidebar.slider("Угол равномерного потока (градусы)", 0, 360, 0)

# Фильтрация — отключаем отображение векторов вблизи центра
exclude_radius = st.sidebar.slider("Радиус исключения векторов (в центре)", 0.0, 1.0, 0.3, 0.05)

eps = 1e-6

# Координатная сетка
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

# Более мелкая сетка для линий тока
x_stream = np.linspace(-3, 3, 150)
y_stream = np.linspace(-3, 3, 150)
Xs, Ys = np.meshgrid(x_stream, y_stream)

# Функции для потенциалов
def compute_potentials(X, Y, field_type, Q, Gamma, flow_angle):
    r = np.sqrt(X**2 + Y**2) + eps
    theta = np.arctan2(Y, X)
    
    if field_type == "Источник/Сток":
        phi = Q / (2*np.pi) * np.log(r)
        psi = Q / (2*np.pi) * theta
    elif field_type == "Вихрь":
        phi = Gamma / (2*np.pi) * theta
        psi = -Gamma / (2*np.pi) * np.log(r)
    elif field_type == "Вихреисточник":
        phi = Q / (2*np.pi) * np.log(r) + Gamma / (2*np.pi) * theta
        psi = Q / (2*np.pi) * theta - Gamma / (2*np.pi) * np.log(r)
    elif field_type == "Диполь":
        M = 1.0
        phi = M / (2*np.pi) * X / (X**2 + Y**2 + eps)
        psi = -M / (2*np.pi) * Y / (X**2 + Y**2 + eps)
    else:  # Равномерный поток
        alpha_rad = np.radians(flow_angle)
        phi = X * np.cos(alpha_rad) + Y * np.sin(alpha_rad)
        psi = Y * np.cos(alpha_rad) - X * np.sin(alpha_rad)
    return phi, psi

# Расчёт полей скоростей
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

else:  # Равномерный поток
    alpha_rad = np.radians(flow_angle)
    U = np.cos(alpha_rad) * np.ones_like(X)
    V = np.sin(alpha_rad) * np.ones_like(Y)
    title = f"Равномерный поток (угол = {flow_angle}°), V₀ = 1"
    Us = np.cos(alpha_rad) * np.ones_like(Xs)
    Vs = np.sin(alpha_rad) * np.ones_like(Ys)

# Вычисляем потенциалы
phi, psi = compute_potentials(X, Y, field_type, Q, Gamma, flow_angle)

# --- ФИЛЬТРАЦИЯ: исключаем векторы вблизи центра (там бесконечность) ---
radius = np.sqrt(X**2 + Y**2)
mask = radius > exclude_radius

# Также ограничиваем максимальную длину вектора для визуализации
magnitude = np.sqrt(U**2 + V**2)
max_magnitude_for_scale = np.percentile(magnitude[mask], 95) if np.any(mask) else 1.0

# Применяем маску
U_display = np.where(mask, U, np.nan)
V_display = np.where(mask, V, np.nan)
mag_display = np.where(mask, magnitude, np.nan)

filtered_count = np.sum(~mask)
if filtered_count > 0 and exclude_radius > 0:
    st.sidebar.info(f"✂️ Исключено {filtered_count} векторов (r < {exclude_radius})")

# Прореживание стрелок для уменьшения загромождения
if arrow_density > 1:
    U_display = U_display[::arrow_density, ::arrow_density]
    V_display = V_display[::arrow_density, ::arrow_density]
    mag_display = mag_display[::arrow_density, ::arrow_density]
    X_plot = X[::arrow_density, ::arrow_density]
    Y_plot = Y[::arrow_density, ::arrow_density]
else:
    X_plot, Y_plot = X, Y

# Функция отрисовки поля
def plot_vector_field(ax, X, Y, U, V, mag, title, show_streamlines=False, Xs=None, Ys=None, Us=None, Vs=None):
    # quiver с правильным масштабированием
    q = ax.quiver(X, Y, U, V, mag,
                  scale=30,  # фиксированный масштаб, а не auto
                  scale_units='xy',
                  cmap='viridis',
                  alpha=0.8,
                  width=0.008,
                  headwidth=3,
                  headlength=4)
    plt.colorbar(q, ax=ax, label='Модуль скорости')
    
    if show_streamlines and Xs is not None:
        ax.streamplot(Xs, Ys, Us, Vs,
                      color='red',
                      linewidth=1.0,
                      density=1.2,
                      arrowsize=0.6)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

# Отрисовка
if separate_plots:
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        plot_vector_field(ax1, X_plot, Y_plot, U_display, V_display, mag_display,
                         title + " (векторы)", show_streamlines=False)
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        mag_stream = np.sqrt(Us**2 + Vs**2)
        plot_vector_field(ax2, Xs, Ys, Us, Vs, mag_stream,
                         title + " (линии тока)", show_streamlines=True,
                         Xs=Xs, Ys=Ys, Us=Us, Vs=Vs)
        st.pyplot(fig2)
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(ax, X_plot, Y_plot, U_display, V_display, mag_display, title,
                     show_streamlines=show_streamlines, Xs=Xs, Ys=Ys, Us=Us, Vs=Vs)
    st.pyplot(fig)

# Визуализация потенциалов
st.subheader("📐 Потенциал скорости φ(x,y) и функция тока ψ(x,y)")

fig_pot, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cf1 = ax1.contourf(X, Y, phi, levels=20, cmap='coolwarm')
ax1.set_title("Потенциал скорости φ")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.colorbar(cf1, ax=ax1)

cf2 = ax2.contourf(X, Y, psi, levels=20, cmap='coolwarm')
ax2.set_title("Функция тока ψ")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.colorbar(cf2, ax=ax2)

st.pyplot(fig_pot)

# Кнопка сохранения
if st.button("Сохранить изображение как PNG"):
    fig.savefig("vector_field.png", dpi=150)
    st.success("Изображение сохранено как vector_field.png")

# Информация о течении
with st.expander("📖 О текущем течении"):
    if field_type == "Источник/Сток":
        sign = "положительная" if Q > 0 else "отрицательная"
        name = "Источник" if Q > 0 else "Сток"
        st.markdown(f"""
        **{name} (Q = {Q})**
        - **Потенциал скорости:** $\\varphi = \\frac{{Q}}{{2\\pi}}\\ln r$
        - **Функция тока:** $\\psi = \\frac{{Q}}{{2\\pi}}\\theta$
        - **Поле скоростей:** $u_r = \\frac{{Q}}{{2\\pi r}}$, $u_\\theta = 0$
        - **Анализ:** {name} — радиальное течение. При Q>0 жидкость вытекает из центра, при Q<0 — втекает.
        - **Визуализация:** векторы в области r < {exclude_radius} исключены (особая точка).
        """)
    elif field_type == "Вихрь":
        st.markdown(f"""
        **Вихрь (Γ = {Gamma})**
        - **Потенциал скорости:** $\\varphi = \\frac{{\\Gamma}}{{2\\pi}}\\theta$
        - **Функция тока:** $\\psi = -\\frac{{\\Gamma}}{{2\\pi}}\\ln r$
        - **Поле скоростей:** $u_r = 0$, $u_\\theta = \\frac{{\\Gamma}}{{2\\pi r}}$
        - **Анализ:** Круговое течение. Линии тока — концентрические окружности.
        """)
    elif field_type == "Вихреисточник":
        st.markdown(f"""
        **Вихреисточник (Q = {Q}, Γ = {Gamma})**
        - **Потенциал скорости:** $\\varphi = \\frac{{Q}}{{2\\pi}}\\ln r + \\frac{{\\Gamma}}{{2\\pi}}\\theta$
        - **Функция тока:** $\\psi = \\frac{{Q}}{{2\\pi}}\\theta - \\frac{{\\Gamma}}{{2\\pi}}\\ln r$
        - **Поле скоростей:** $u_r = \\frac{{Q}}{{2\\pi r}}$, $u_\\theta = \\frac{{\\Gamma}}{{2\\pi r}}$
        - **Анализ:** Спиралевидные линии тока. Суперпозиция источника и вихря.
        """)
    elif field_type == "Диполь":
        st.markdown(f"""
        **Диполь**
        - **Потенциал скорости:** $\\varphi = \\frac{{M}}{{2\\pi}}\\frac{{x}}{{x^2+y^2}} = \\frac{{M\\cos\\theta}}{{2\\pi r}}$
        - **Функция тока:** $\\psi = -\\frac{{M}}{{2\\pi}}\\frac{{y}}{{x^2+y^2}} = -\\frac{{M\\sin\\theta}}{{2\\pi r}}$
        - **Поле скоростей:** $u_r = \\frac{{M\\cos\\theta}}{{2\\pi r^2}}$, $u_\\theta = \\frac{{M\\sin\\theta}}{{2\\pi r^2}}$
        - **Анализ:** Замкнутые линии тока. **Векторы в центре исключены (r < {exclude_radius})**.
        """)
    else:
        st.markdown(f"""
        **Равномерный поток (угол = {flow_angle}°)**
        - **Потенциал скорости:** $\\varphi = V_0(x\\cos\\alpha_0 + y\\sin\\alpha_0)$
        - **Функция тока:** $\\psi = V_0(y\\cos\\alpha_0 - x\\sin\\alpha_0)$
        - **Поле скоростей:** $u_x = V_0\\cos\\alpha_0$, $u_y = V_0\\sin\\alpha_0$
        - **Анализ:** Параллельные линии тока. Постоянная скорость.
        """)

with st.expander("ℹ️ О приложении"):
    st.markdown(f"""
    **Ключевые изменения:**
    - ✅ **Источник и Сток объединены** — регулируется знаком Q
    - ✅ **Исключение векторов вблизи особой точки** (радиус {exclude_radius})
    - ✅ **Фиксированный масштаб стрелок** (scale=30) вместо auto
    - ✅ **Прореживание стрелок** (каждая {arrow_density}-я точка)
    
    **Почему стрелки больше не огромные:**
    1. Векторы внутри радиуса {exclude_radius} не рисуются
    2. Масштаб стрелок фиксированный, а не подстраивается под максимальный вектор
    """)
