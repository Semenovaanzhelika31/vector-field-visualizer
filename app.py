import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Визуализация векторных полей", layout="wide")
st.title("Научная визуализация векторных полей")
st.markdown("Интерактивное приложение для визуализации двумерных векторных полей идеальной жидкости")

st.sidebar.header("Параметры визуализации")

field_type = st.sidebar.selectbox(
    "Тип векторного поля",
    ["Вихрь", "Источник", "Сток", "Вихреисточник", "Диполь", "Равномерный поток"]
)

grid_size = st.sidebar.slider("Размер сетки (N x N)", 10, 50, 20)
arrow_scale = st.sidebar.slider("Множитель длины стрелок", 0.1, 1.5, 0.5)
show_grid = st.sidebar.checkbox("Показать сетку", True)
show_streamlines = st.sidebar.checkbox("Показать линии тока", True)
separate_plots = st.sidebar.checkbox("Раздельные графики (векторы и линии тока отдельно)", False)

# Фильтрация больших векторов
filter_vectors = st.sidebar.checkbox("Фильтровать большие векторы", True)
max_speed = st.sidebar.slider("Максимальный модуль скорости для отображения", 0.5, 10.0, 3.0, 0.5)

Q = st.sidebar.slider("Мощность источника Q", -2.0, 2.0, 1.0, 0.1)
Gamma = st.sidebar.slider("Циркуляция Γ", -2.0, 2.0, 1.0, 0.1)
flow_angle = st.sidebar.slider("Угол равномерного потока (градусы)", 0, 360, 0)

eps = 1e-6

x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

x_stream = np.linspace(-3, 3, 100)
y_stream = np.linspace(-3, 3, 100)
Xs, Ys = np.meshgrid(x_stream, y_stream)

# Функции для потенциалов
def compute_potentials(X, Y, field_type, Q, Gamma, flow_angle):
    r = np.sqrt(X**2 + Y**2) + eps
    theta = np.arctan2(Y, X)
    
    if field_type == "Вихрь":
        phi = Gamma / (2*np.pi) * theta
        psi = -Gamma / (2*np.pi) * np.log(r)
    elif field_type == "Источник":
        phi = Q / (2*np.pi) * np.log(r)
        psi = Q / (2*np.pi) * theta
    elif field_type == "Сток":
        Q_neg = -abs(Q)
        phi = Q_neg / (2*np.pi) * np.log(r)
        psi = Q_neg / (2*np.pi) * theta
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
U, V, Us, Vs = None, None, None, None
title = ""

if field_type == "Вихрь":
    r2 = X**2 + Y**2 + eps
    U = -Gamma * Y / (2*np.pi * r2)
    V =  Gamma * X / (2*np.pi * r2)
    title = f"Вихрь (Γ = {Gamma})"
    r2s = Xs**2 + Ys**2 + eps
    Us = -Gamma * Ys / (2*np.pi * r2s)
    Vs =  Gamma * Xs / (2*np.pi * r2s)

elif field_type == "Источник":
    r2 = X**2 + Y**2 + eps
    U = Q * X / (2*np.pi * r2)
    V = Q * Y / (2*np.pi * r2)
    title = f"Источник (Q = {Q})"
    r2s = Xs**2 + Ys**2 + eps
    Us = Q * Xs / (2*np.pi * r2s)
    Vs = Q * Ys / (2*np.pi * r2s)

elif field_type == "Сток":
    Q_neg = -abs(Q)
    r2 = X**2 + Y**2 + eps
    U = Q_neg * X / (2*np.pi * r2)
    V = Q_neg * Y / (2*np.pi * r2)
    title = f"Сток (Q = {Q_neg})"
    r2s = Xs**2 + Ys**2 + eps
    Us = Q_neg * Xs / (2*np.pi * r2s)
    Vs = Q_neg * Ys / (2*np.pi * r2s)

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

# Фильтрация больших векторов
magnitude = np.sqrt(U**2 + V**2)
if filter_vectors:
    mask = magnitude <= max_speed
    U_filtered = np.where(mask, U, np.nan)
    V_filtered = np.where(mask, V, np.nan)
    magnitude_filtered = np.where(mask, magnitude, np.nan)
    # Для отображения используем отфильтрованные данные
    U_display, V_display, mag_display = U_filtered, V_filtered, magnitude_filtered
    # Считаем сколько векторов отфильтровано
    filtered_count = np.sum(~mask)
    if filtered_count > 0:
        st.sidebar.info(f"✂️ Отфильтровано {filtered_count} векторов (|v| > {max_speed})")
else:
    U_display, V_display, mag_display = U, V, magnitude

# Функция отрисовки поля
def plot_vector_field(ax, X, Y, U, V, mag, title, show_streamlines=False, Xs=None, Ys=None, Us=None, Vs=None):
    q = ax.quiver(X, Y, U, V, mag,
                  scale=arrow_scale,
                  scale_units='xy',
                  cmap='viridis',
                  alpha=0.8,
                  width=0.005)
    plt.colorbar(q, ax=ax, label='Модуль скорости')
    
    if show_streamlines and Xs is not None:
        ax.streamplot(Xs, Ys, Us, Vs,
                      color='red',
                      linewidth=0.8,
                      density=1.5,
                      arrowsize=0.8)
    
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
        plot_vector_field(ax1, X, Y, U_display, V_display, mag_display,
                         title + " (векторы)", show_streamlines=False)
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # Для линий тока используем полные данные (без фильтрации, чтобы линии были непрерывными)
        mag_stream = np.sqrt(Us**2 + Vs**2)
        plot_vector_field(ax2, Xs, Ys, Us, Vs, mag_stream,
                         title + " (линии тока)", show_streamlines=True,
                         Xs=Xs, Ys=Ys, Us=Us, Vs=Vs)
        st.pyplot(fig2)
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(ax, X, Y, U_display, V_display, mag_display, title,
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
    if field_type == "Источник":
        st.markdown(r"""
        **Источник ($Q > 0$)**
        - **Потенциал скорости:** $\varphi = \frac{Q}{2\pi}\ln r$
        - **Функция тока:** $\psi = \frac{Q}{2\pi}\theta$
        - **Поле скоростей:** $u_r = \frac{Q}{2\pi r}$, $u_\theta = 0$
        - **Анализ:** Жидкость равномерно истекает из точечного источника. Линии тока — радиальные прямые.
        """)
    elif field_type == "Сток":
        st.markdown(r"""
        **Сток ($Q < 0$)**
        - **Потенциал скорости:** $\varphi = -\frac{|Q|}{2\pi}\ln r$
        - **Функция тока:** $\psi = -\frac{|Q|}{2\pi}\theta$
        - **Поле скоростей:** $u_r = -\frac{|Q|}{2\pi r}$, $u_\theta = 0$
        - **Анализ:** Жидкость равномерно поглощается в центре. Линии тока — радиальные прямые к центру.
        """)
    elif field_type == "Вихрь":
        st.markdown(r"""
        **Вихрь (потенциальный)**
        - **Потенциал скорости:** $\varphi = \frac{\Gamma}{2\pi}\theta$
        - **Функция тока:** $\psi = -\frac{\Gamma}{2\pi}\ln r$
        - **Поле скоростей:** $u_r = 0$, $u_\theta = \frac{\Gamma}{2\pi r}$
        - **Анализ:** Круговое течение. Линии тока — концентрические окружности.
        """)
    elif field_type == "Вихреисточник":
        st.markdown(r"""
        **Вихреисточник**
        - **Потенциал скорости:** $\varphi = \frac{Q}{2\pi}\ln r + \frac{\Gamma}{2\pi}\theta$
        - **Функция тока:** $\psi = \frac{Q}{2\pi}\theta - \frac{\Gamma}{2\pi}\ln r$
        - **Поле скоростей:** $u_r = \frac{Q}{2\pi r}$, $u_\theta = \frac{\Gamma}{2\pi r}$
        - **Анализ:** Спиралевидные линии тока. Суперпозиция источника и вихря.
        """)
    elif field_type == "Диполь":
        st.markdown(r"""
        **Диполь**
        - **Потенциал скорости:** $\varphi = \frac{M}{2\pi}\frac{x}{x^2+y^2} = \frac{M\cos\theta}{2\pi r}$
        - **Функция тока:** $\psi = -\frac{M}{2\pi}\frac{y}{x^2+y^2} = -\frac{M\sin\theta}{2\pi r}$
        - **Поле скоростей:** $u_r = \frac{M\cos\theta}{2\pi r^2}$, $u_\theta = \frac{M\sin\theta}{2\pi r^2}$
        - **Анализ:** Замкнутые линии тока. В центре — особая точка. **Векторы в центре отфильтрованы**.
        """)
    else:
        st.markdown(r"""
        **Равномерный поток**
        - **Потенциал скорости:** $\varphi = V_0(x\cos\alpha_0 + y\sin\alpha_0)$
        - **Функция тока:** $\psi = V_0(y\cos\alpha_0 - x\sin\alpha_0)$
        - **Поле скоростей:** $u_x = V_0\cos\alpha_0$, $u_y = V_0\sin\alpha_0$
        - **Анализ:** Параллельные линии тока. Постоянная скорость.
        """)

with st.expander("ℹ️ О приложении"):
    st.markdown("""
    **Возможности:**
    - ✅ Фильтрация больших векторов (для диполя, источника, вихря)
    - ✅ Раздельные графики (векторы и линии тока отдельно)
    - ✅ Визуализация потенциала скорости и функции тока
    - ✅ Интерактивная настройка параметров
    
    **Фильтрация:** векторы с модулем скорости > порога не отображаются.
    """)
