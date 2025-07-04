import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Seismic Anisotropy Explorer")

# --- Title & Description ---
st.title("🌍 Seismic Anisotropy Explorer")
st.markdown("""
Visualize how Thomsen parameters affect P-wave velocity anisotropy in 2D and 3D.
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛 Control Panel")
    epsilon = st.slider("ε (Epsilon)", 0.0, 0.4, 0.0, 0.01,
                       help="Controls elongation of velocity surface")
    delta = st.slider("δ (Delta)", 0.0, 0.4, 0.0, 0.01,
                     help="Affects curvature near symmetry axes")
    gamma = st.slider("γ (Gamma)", 0.0, 0.4, 0.0, 0.01,
                     help="Shear-wave anisotropy (visualized in 3D)")
    
    st.divider()
    st.markdown("### 🛠 Display Options")
    show_3d = st.checkbox("Show 3D Visualization", True)
    show_physics = st.checkbox("Show Physics Explanation", True)

# --- Physics Calculations ---
def calculate_velocities(epsilon, delta, gamma):
    theta = np.linspace(0, 2*np.pi, 360)
    phi = np.linspace(0, np.pi, 180)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # P-wave velocity equation
    Vp = 3000 * (1 + delta * (np.sin(phi_grid))**2 * (np.cos(phi_grid))**2 
               + epsilon * (np.sin(phi_grid))**4)
    
    # Convert to Cartesian coordinates
    x = Vp * np.sin(phi_grid) * np.cos(theta_grid)
    y = Vp * np.sin(phi_grid) * np.sin(theta_grid)
    z = Vp * np.cos(phi_grid)
    
    return x, y, z, Vp

# --- Plot Generation ---
def create_2d_plot(epsilon, delta):
    theta = np.linspace(0, np.pi/2, 90)
    Vp = 3000 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 
               + epsilon * (np.sin(theta))**4)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(Vp * np.sin(theta), Vp * np.cos(theta), 'b-', linewidth=3)
    ax.set_xlabel('Vpx [m/s]', fontsize=12)
    ax.set_ylabel('Vpy [m/s]', fontsize=12)
    ax.set_title(f"2D Velocity Anisotropy (ε={epsilon}, δ={delta})", pad=20)
    ax.axis('square')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 5000)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def create_3d_plot(x, y, z):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8,
                          rstride=2, cstride=2)
    
    ax.set_xlabel('X [m/s]')
    ax.set_ylabel('Y [m/s]')
    ax.set_zlabel('Z [m/s]')
    ax.set_title("3D Velocity Surface", pad=20)
    fig.colorbar(surf, shrink=0.5, label='Velocity (m/s)')
    ax.set_box_aspect([1,1,1])
    return fig

# --- Export Functionality ---
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return st.download_button(text, buf, file_name=filename, mime="image/png")

# --- Main App ---
x, y, z, Vp = calculate_velocities(epsilon, delta, gamma)

# Create two columns for plots
col1, col2 = st.columns(2)

with col1:
    st.header("2D Visualization")
    fig_2d = create_2d_plot(epsilon, delta)
    st.pyplot(fig_2d)
    get_image_download_link(fig_2d, "2d_anisotropy.png", "⬇ Download 2D Plot")

with col2:
    if show_3d:
        st.header("3D Visualization")
        fig_3d = create_3d_plot(x, y, z)
        st.pyplot(fig_3d)
        get_image_download_link(fig_3d, "3d_anisotropy.png", "⬇ Download 3D Plot")

# --- Physics Explanation ---
if show_physics:
    st.markdown("""
    ## 📚 Physics Explanation
    
    ### Thomsen Parameters
    - **ε (Epsilon)**: 
      - Measures the fractional difference between horizontal and vertical P-wave velocities
      - ε > 0 indicates faster horizontal velocities (typical in sedimentary basins)
      - Controls the 'ellipticity' of the velocity surface
    
    - **δ (Delta)**:
      - Describes the angular dependence of velocities near the vertical axis
      - Affects the curvature of the velocity surface near 0° and 90°
      - Important for seismic amplitude variation with offset (AVO) analysis
    
    - **γ (Gamma)**:
      - Measures shear-wave anisotropy (not shown in 2D plot)
      - γ = (Vsh - Vsv)/Vsv, where Vsh = horizontal shear wave velocity
    
    ### Velocity Equation
    The P-wave velocity in anisotropic media is given by:
    ```
    Vp(θ) = Vp0 * (1 + δ sin²θ cos²θ + ε sin⁴θ)
    ```
    Where:
    - Vp0 = 3000 m/s (vertical velocity)
    - θ = wave propagation angle from vertical
    """)

# --- References ---
st.divider()
st.markdown("""
### 📖 References
1. Thomsen, L. (1986). Weak elastic anisotropy. Geophysics, 51(10), 1954-1966.
2. Tsvankin, I. (2012). Seismic signatures and analysis of reflection data in anisotropic media. SEG.
""")
