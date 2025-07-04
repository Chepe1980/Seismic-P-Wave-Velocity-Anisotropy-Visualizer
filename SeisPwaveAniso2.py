import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Seismic Anisotropy Explorer")

# --- Title & Description ---
st.title("🌍 Seismic Anisotropy Explorer")
st.markdown("""
Visualize how Thomsen parameters affect P-wave velocity anisotropy in 2D and interactive 3D.
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
    theta = np.linspace(0, 2*np.pi, 180)  # Azimuthal angle
    phi = np.linspace(0, np.pi, 90)       # Polar angle
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # P-wave velocity equation (Thomsen's weak anisotropy)
    Vp = 3000 * (1 + delta * (np.sin(phi_grid))**2 * (np.cos(phi_grid))**2 
               + epsilon * (np.sin(phi_grid))**4)
    
    # Convert spherical to Cartesian coordinates
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
    ax.set_title(f"2D Velocity Anisotropy (ε={epsilon:.2f}, δ={delta:.2f})", pad=20)
    ax.axis('square')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 5000)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def create_3d_plot(x, y, z, vp):
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=vp,
            colorscale='Viridis',
            colorbar=dict(title='Velocity (m/s)'),
            opacity=0.9,
            hoverinfo='x+y+z+text',
            text=[f'Vp: {val:.0f} m/s' for val in vp.flatten()]
        )
    ])
    
    fig.update_layout(
        title='3D Velocity Surface (Drag to rotate)',
        scene=dict(
            xaxis_title='X [m/s]',
            yaxis_title='Y [m/s]',
            zaxis_title='Z [m/s]',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    return fig

# --- Export Functionality ---
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return st.download_button(text, buf, file_name=filename, mime="image/png")

# --- Main App ---
x, y, z, Vp = calculate_velocities(epsilon, delta, gamma)

# Create columns for plots
col1, col2 = st.columns(2)

with col1:
    st.header("2D Visualization")
    fig_2d = create_2d_plot(epsilon, delta)
    st.pyplot(fig_2d)
    get_image_download_link(fig_2d, "2d_anisotropy.png", "⬇ Download 2D Plot")

with col2:
    if show_3d:
        st.header("Interactive 3D Visualization")
        fig_3d = create_3d_plot(x, y, z, Vp)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Plotly doesn't support direct image download, so we provide a screenshot tip
        st.markdown("""
        **💡 To export 3D plot:**
        1. Hover over the plot
        2. Click the camera icon in the toolbar
        3. Save the image
        """)

# --- Physics Explanation ---
if show_physics:
    st.markdown("""
    ## 📚 Physics Explanation
    
    ### Thomsen's Weak Anisotropy Parameters
    | Parameter | Physical Meaning | Typical Range |
    |-----------|------------------|---------------|
    | **ε** | P-wave anisotropy along horizontal axis | -0.3 to 0.5 |
    | **δ** | Near-vertical P-wave anisotropy | -0.5 to 0.5 |
    | **γ** | SH-wave anisotropy | -0.5 to 0.5 |

    **Key Relationships:**
    ```math
    V_p(θ) = V_{p0} \left(1 + δ \sin^2θ \cos^2θ + ε \sin^4θ\right)
    ```
    Where θ is the angle from vertical symmetry axis.
    
    **Geological Implications:**
    - ε > δ: Common in shale-rich formations
    - δ > ε: Suggests fracture-dominated anisotropy
    - Negative values: Rare but occur in some carbonate reservoirs
    """)

# --- References ---
st.divider()
st.markdown("""
### 📖 References
1. Thomsen, L. (1986). Weak elastic anisotropy. Geophysics, 51(10), 1954-1966.
2. Tsvankin, I. (2012). Seismic signatures and analysis of reflection data in anisotropic media. SEG.
""")
