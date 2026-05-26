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
separate_plots = st.sidebar.checkbox("Раздельные графики (линии тока и векторы отдельно)", False)

# Фильтрация больших векторов
max_vector_norm = st.sidebar.slider("Максимальная норма вектора для отображения (фильтр)", 0.5, 5.0, 2.0)

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

# Функции для потенциала скорости φ и функции тока ψ
def velocity_potential_source_sink(x, y, Q_val):
    r = np.sqrt(x**2 + y**2) + eps
    return Q_val / (2*np.pi) * np.log(r)

def stream_function_source_sink(x, y, Q_val):
    return Q_val / (2*np.pi) * np.arctan2(y, x)

def velocity_potential_vortex(x, y, Gamma_val):
    return Gamma_val / (2*np.pi) * np.arctan2(y, x)

def stream_function_vortex(x, y, Gamma_val):
    r = np.sqrt(x**2 + y**2) + eps
    return -Gamma_val / (2*np.pi) * np.log(r)

def velocity_potential_dipole(x, y, M=1.0):
    r2 = x**2 + y**2 + eps
    return M / (2*np.pi) * x / r2

def stream_function_dipole(x, y, M=1.0):
    r2 = x**2 + y**2 + eps
    return -M / (2*np.pi) * y / r2

def velocity_potential_uniform(x, y, angle_rad):
    return x * np.cos(angle_rad) + y * np.sin(angle_rad)

def stream_function_uniform(x, y, angle_rad):
    return y * np.cos(angle_rad) - x * np.sin(angle_rad)

# Расчёт полей
U, V, Us, Vs = None, None, None, None
phi, psi = None, None
title = ""

if field_type == "Вихрь":
    r2 = X**2 + Y**2 + eps
    U = -Gamma * Y / (2*np.pi * r2)
    V =  Gamma * X / (2*np.pi * r2)
    title = f"Вихрь (Γ = {Gamma})"
    r2s = Xs**2 + Ys**2 + eps
    Us = -Gamma * Ys / (2*np.pi * r2s)
    Vs =  Gamma * Xs / (2*np.pi * r2s)
    phi = velocity_potential_vortex(X, Y, Gamma)
    psi = stream_function_vortex(X, Y, Gamma)

elif field_type == "Источник":
    r2 = X**2 + Y**2 + eps
    U = Q * X / (2*np.pi * r2)
    V = Q * Y / (2*np.pi * r2)
    title = f"Источник (Q = {Q})"
    r2s = Xs**2 + Ys**2 + eps
    Us = Q * Xs / (2*np.pi * r2s)
    Vs = Q * Ys / (2*np.pi * r2s)
    phi = velocity_potential_source_sink(X, Y, Q)
    psi = stream_function_source_sink(X, Y, Q)

elif field_type == "Сток":
    Q_neg = -abs(Q)
    r2 = X**2 + Y**2 + eps
    U = Q_neg * X / (2*np.pi * r2)
    V = Q_neg * Y / (2*np.pi * r2)
    title = f"Сток (Q = {Q_neg})"
    r2s = Xs**2 + Ys**2 + eps
    Us = Q_neg * Xs / (2*np.pi * r2s)
    Vs = Q_neg * Ys / (2*np.pi * r2s)
    phi = velocity_potential_source_sink(X, Y, Q_neg)
    psi = stream_function_source_sink(X, Y, Q_neg)

elif field_type == "Вихреисточник":
    r2 = X**2 + Y**2 + eps
    U = (Q * X - Gamma * Y) / (2*np.pi * r2)
    V = (Q * Y + Gamma * X) / (2*np.pi * r2)
    title = f"Вихреисточник (Q = {Q}, Γ = {Gamma})"
    r2s = Xs**2 + Ys**2 + eps
    Us = (Q * Xs - Gamma * Ys) / (2*np.pi * r2s)
    Vs = (Q * Ys + Gamma * Xs) / (2*np.pi * r2s)
    phi = velocity_potential_source_sink(X, Y, Q) + velocity_potential_vortex(X, Y, Gamma)
    psi = stream_function_source_sink(X, Y, Q) + stream_function_vortex(X, Y, Gamma)

elif field_type == "Диполь":
    r2 = X**2 + Y**2 + eps
    r4 = r2**2
    M = 1.0
    U = M * (X**2 - Y**2) / r4
    V = M * (2 * X * Y) / r4
    title = "Диполь (ось X)"
    r2s = Xs**2 + Ys**2 + eps
    r4s = r2s**2
    Us = M * (Xs**2 - Ys**2) / r4s
    Vs = M * (2 * Xs * Ys) / r4s
    phi = velocity_potential_dipole(X, Y, M)
    psi = stream_function_dipole(X, Y, M)

else:  # Равномерный поток
    alpha_rad = np.radians(flow_angle)
    U = np.cos(alpha_rad) * np.ones_like(X)
    V = np.sin(alpha_rad) * np.ones_like(Y)
    title = f"Равномерный поток (угол = {flow_angle}°), V₀ = 1"
    Us = np.cos(alpha_rad) * np.ones_like(Xs)
    Vs = np.sin(alpha_rad) * np.ones_like(Ys)
    phi = velocity_potential_uniform(X, Y, alpha_rad)
    psi = stream_function_uniform(X, Y, alpha_rad)

magnitude = np.sqrt(U**2 + V**2)

# Фильтрация векторов
mask = magnitude < max_vector_norm
U_filtered = np.where(mask, U, np.nan)
V_filtered = np.where(mask, V, np.nan)

# Функция отрисовки
def plot_field(ax, X, Y, U, V, magnitude, title, show_streamlines=False, Xs=None, Ys=None, Us=None, Vs=None):
    q = ax.quiver(X, Y, U, V, magnitude,
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

if separate_plots:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_field(ax1, X, Y, U_filtered, V_filtered, magnitude,
               title + " (векторы)", show_streamlines=False)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_field(ax2, Xs, Ys, Us, Vs, np.sqrt(Us**2 + Vs**2),
               title + " (линии тока)", show_streamlines=True,
               Xs=Xs, Ys=Ys, Us=Us, Vs=Vs)
    st.pyplot(fig2)
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_field(ax, X, Y, U_filtered, V_filtered, magnitude, title,
               show_streamlines=show_streamlines, Xs=Xs, Ys=Ys, Us=Us, Vs=Vs)
    st.pyplot(fig)

# Потенциалы
st.subheader("Потенциал скорости φ(x,y) и функция тока ψ(x,y)")
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

if st.button("Сохранить изображение как PNG"):
    fig.savefig("vector_field.png", dpi=150)
    st.success("Изображение сохранено как vector_field.png")

# Таблица с анализом
st.subheader("Анализ течений")
analysis_data = {
    "Течение": ["Источник", "Сток", "Вихрь", "Вихреисточник", "Диполь", "Равномерный поток"],
    "Поле скоростей": [
        r"$u_r = \frac{Q}{2\pi r}$, $u_\theta=0$",
        r"$u_r = -\frac{|Q|}{2\pi r}$, $u_\theta=0$",
        r"$u_r=0$, $u_\theta = \frac{\Gamma}{2\pi r}$",
        r"$u_r = \frac{Q}{2\pi r}$, $u_\theta = \frac{\Gamma}{2\pi r}$",
        r"$u_r = \frac{M\cos\theta}{2\pi r^2}$, $u_\theta = \frac{M\sin\theta}{2\pi r^2}$",
        r"$u_x = V_0\cos\alpha$, $u_y = V_0\sin\alpha$"
    ],
    "Потенциал φ": [
        r"$\frac{Q}{2\pi}\ln r$",
        r"$-\frac{|Q|}{2\pi}\ln r$",
        r"$\frac{\Gamma}{2\pi}\theta$",
        r"$\frac{Q}{2\pi}\ln r + \frac{\Gamma}{2\pi}\theta$",
        r"$\frac{M\cos\theta}{2\pi r}$",
        r"$V_0(x\cos\alpha + y\sin\alpha)$"
    ],
    "Функция тока ψ": [
        r"$\frac{Q}{2\pi}\theta$",
        r"$-\frac{|Q|}{2\pi}\theta$",
        r"$-\frac{\Gamma}{2\pi}\ln r$",
        r"$\frac{Q}{2\pi}\theta - \frac{\Gamma}{2\pi}\ln r$",
        r"$-\frac{M\sin\theta}{2\pi r}$",
        r"$V_0(y\cos\alpha - x\sin\alpha)$"
    ]
}

st.table(analysis_data)

st.markdown("""
### Краткий анализ
- **Источник/Сток** — радиальное течение, линии тока — прямые из центра/к центру.
- **Вихрь** — круговое течение, линии тока — окружности.
- **Вихреисточник** — спиралевидные линии тока, комбинация радиального и кругового движения.
- **Диполь** — замкнутые линии тока, поле диполя (два вихря противоположной циркуляции).
- **Равномерный поток** — параллельные линии тока, постоянная скорость.
""")

with st.expander("О приложении"):
    st.markdown(r"""
    **Особенности:**
    - Фильтрация векторов с модулем > заданного порога.
    - Возможность раздельного отображения векторов и линий тока.
    - Визуализация потенциала скорости и функции тока.
    - Защита от деления на ноль.
    """)
