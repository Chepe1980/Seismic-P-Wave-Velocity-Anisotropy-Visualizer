import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Title and description
st.title("Seismic P-Wave Velocity Anisotropy Visualizer")
st.markdown("""
Explore how Thomsen parameters (ε, δ) affect P-wave velocity anisotropy.
""")

# Sidebar input boxes for parameters
st.sidebar.header("Anisotropy Parameters")
epsilon = st.sidebar.number_input(
    "ε (Epsilon)", 
    min_value=-0.5, 
    max_value=0.5, 
    value=-0.01, 
    step=0.01,
    help="Controls the 'stretch' of the velocity ellipse"
)
delta = st.sidebar.number_input(
    "δ (Delta)", 
    min_value=-0.5, 
    max_value=0.5, 
    value=-0.01, 
    step=0.01,
    help="Affects curvature near the axes"
)
gamma = st.sidebar.number_input(
    "γ (Gamma)", 
    min_value=-0.5, 
    max_value=0.5, 
    value=-0.01, 
    step=0.01,
    help="Shear-wave anisotropy (unused in this model)"
)

# Main calculation and plotting function
def plot_anisotropy(epsilon, delta):
    theta = np.linspace(0, 90, 90) * np.pi / 180  # Convert to radians
    Vp = 3000 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 
                 + epsilon * (np.sin(theta))**4)
    
    Vpx = Vp * np.sin(theta)
    Vpy = Vp * np.cos(theta)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(Vpx, Vpy, 'b-', linewidth=2, label=f"ε={epsilon:.3f}, δ={delta:.3f}")
    ax.set_xlabel('Vpx [m/s]', fontsize=12)
    ax.set_ylabel('Vpy [m/s]', fontsize=12)
    ax.set_title("P-Wave Velocity Anisotropy", fontsize=14)
    ax.axis('square')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 5000)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig

# Display the plot
st.pyplot(plot_anisotropy(epsilon, delta))
