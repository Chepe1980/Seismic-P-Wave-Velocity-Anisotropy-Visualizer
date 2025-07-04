import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import convolve
import plotly.graph_objects as go
import streamlit as st

# Title and description
st.title("Seismic AVAz Modeling Tool")
st.markdown("""
Visualize Amplitude Variation with Azimuth (AVAz) responses for anisotropic media.
Adjust Thomsen parameters (ε, δ, γ) and rock properties below.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Rock Properties & Parameters")

# Layer 1 (Upper)
st.sidebar.subheader("Layer 1 (Upper)")
vp1 = st.sidebar.number_input("Vp₁ (m/s)", min_value=1000, max_value=8000, value=5500)
vs1 = st.sidebar.number_input("Vs₁ (m/s)", min_value=500, max_value=5000, value=3600)
d1 = st.sidebar.number_input("Density₁ (g/cc)", min_value=1.5, max_value=3.5, value=2.6, step=0.1)
e1 = st.sidebar.number_input("ε₁", min_value=-0.5, max_value=0.5, value=0.1, step=0.01)
g1 = st.sidebar.number_input("γ₁", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)
dlt1 = st.sidebar.number_input("δ₁", min_value=-0.5, max_value=0.5, value=0.0, step=0.01)

# Layer 2 (Target)
st.sidebar.subheader("Layer 2 (Target)")
vp2 = st.sidebar.number_input("Vp₂ (m/s)", min_value=1000, max_value=8000, value=4742)
vs2 = st.sidebar.number_input("Vs₂ (m/s)", min_value=500, max_value=5000, value=3292)
d2 = st.sidebar.number_input("Density₂ (g/cc)", min_value=1.5, max_value=3.5, value=2.4, step=0.1)
e2 = st.sidebar.number_input("ε₂", min_value=-0.5, max_value=0.5, value=-0.01, step=0.01)
g2 = st.sidebar.number_input("γ₂", min_value=-0.5, max_value=0.5, value=-0.05, step=0.01)
dlt2 = st.sidebar.number_input("δ₂", min_value=-0.5, max_value=0.5, value=-0.13, step=0.01)

# Layer 3 (Lower)
st.sidebar.subheader("Layer 3 (Lower)")
vp3 = st.sidebar.number_input("Vp₃ (m/s)", min_value=1000, max_value=8000, value=5500)
vs3 = st.sidebar.number_input("Vs₃ (m/s)", min_value=500, max_value=5000, value=3600)
d3 = st.sidebar.number_input("Density₃ (g/cc)", min_value=1.5, max_value=3.5, value=2.6, step=0.1)
e3 = st.sidebar.number_input("ε₃", min_value=-0.5, max_value=0.5, value=0.2, step=0.01)
g3 = st.sidebar.number_input("γ₃", min_value=-0.5, max_value=0.5, value=0.15, step=0.01)
dlt3 = st.sidebar.number_input("δ₃", min_value=-0.5, max_value=0.5, value=0.1, step=0.01)

# Acquisition parameters
st.sidebar.subheader("Acquisition Parameters")
incidence_angle = st.sidebar.number_input("Incidence Angle (deg)", min_value=0, max_value=90, value=60)
freq = st.sidebar.number_input("Wavelet Frequency (Hz)", min_value=10, max_value=100, value=45)

# --- Core Functions ---
def ricker_wavelet(freq, length, dt):
    """Generate a Ricker wavelet"""
    t = np.arange(-length/2, length/2, dt)
    wavelet = (1 - 2 * (np.pi * freq * t)**2) * np.exp(-(np.pi * freq * t)**2)
    return wavelet

def calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuths):
    """Calculate reflectivity for given parameters"""
    VP2 = (vp[1] + vp[2]) / 2
    VS2 = (vs[1] + vs[2]) / 2
    DEN2 = (d[1] + d[2]) / 2

    A2 = -0.5 * ((vp[2] - vp[1])/VP2 + (d[2] - d[1])/DEN2)  # Negative sign for correct polarity

    ref = np.zeros(len(azimuths))
    for idx, az in enumerate(azimuths):
        az_rad = np.radians(az)

        Biso2 = (0.5*((vp[2]-vp[1])/VP2) - 2*(VS2/VP2)**2*(d[2]-d[1])/DEN2 - 4*(VS2/VP2)**2*(vs[2]-vs[1])/VS2)
        Baniso2 = 0.5*((dlt[2]-dlt[1]) + 2*(2*VS2/VP2)**2*(g[2]-g[1]))
        Caniso2 = 0.5*((vp[2]-vp[1])/VP2 - (e[2]-e[1])*np.cos(az_rad)**4 + (dlt[2]-dlt[1])*np.sin(az_rad)**2*np.cos(az_rad)**2)

        ref[idx] = A2 + (Biso2 + Baniso2*np.cos(az_rad)**2)*np.sin(theta)**2 + Caniso2*np.sin(theta)**2*np.tan(theta)**2

    return ref

def avaz_modeling(params):
    """Main modeling function with 3D plot addition"""
    # Extract parameters
    vp = [params['vp1'], params['vp2'], params['vp3']]
    vs = [params['vs1'], params['vs2'], params['vs3']]
    d = [params['d1'], params['d2'], params['d3']]
    e = [params['e1'], params['e2'], params['e3']]
    g = [params['g1'], params['g2'], params['g3']]
    dlt = [params['dlt1'], params['dlt2'], params['dlt3']]
    theta = np.radians(params['incidence_angle'])
    freq = params['freq']

    # Azimuthal analysis
    azimuths = np.arange(0, 361)
    ref2 = calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuths)

    # Generate synthetic seismogram
    R = np.zeros((150, len(azimuths)))
    R[60, :] = ref2
    wavelet = ricker_wavelet(freq, 0.08, 0.001)
    syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
    center = 60 + len(wavelet)//2
    syn_n = syn[center-75:center+75, :]

    # Create 3D surface plot
    incidence_angles = np.linspace(0, 60, 30)
    azimuths_3d = np.arange(0, 361, 5)
    reflectivity_3d = np.zeros((len(incidence_angles), len(azimuths_3d)))

    for i, ang in enumerate(incidence_angles):
        reflectivity_3d[i,:] = calculate_reflectivity(vp, vs, d, e, g, dlt, np.radians(ang), azimuths_3d)

    fig_3d = go.Figure(data=[go.Surface(
        z=reflectivity_3d,
        x=azimuths_3d,
        y=incidence_angles,
        colorscale='Jet',
        colorbar=dict(title='Reflectivity'),
    )])

    fig_3d.update_layout(
        title='3D AVAz Response: Reflectivity vs Azimuth and Incidence Angle',
        scene=dict(
            xaxis_title='Azimuth (degrees)',
            yaxis_title='Incidence Angle (degrees)',
            zaxis_title='Reflectivity',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        height=800,
        width=1000
    )

    # 2D Plots
    fig_2d, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Reflectivity plot
    sc = ax1.scatter(azimuths, ref2, c=ref2, cmap='jet', norm=Normalize(vmin=-0.4, vmax=0.2))
    ax1.set(title=f'Azimuthal Reflectivity at {params["incidence_angle"]}°', xlabel='Azimuth (degrees)', ylabel='Reflectivity')
    plt.colorbar(sc, ax=ax1).set_label('Reflectivity')

    # Seismogram plot
    vmax = np.abs(syn_n).max()
    img2 = ax2.imshow(syn_n, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=[0, 360, syn_n.shape[0], 0])
    for az in range(0, 361, 10):
        trace = syn_n[:, az]
        ax2.plot(az + syn_n[:,az]/vmax*10, np.arange(syn_n.shape[0]), 'k', linewidth=0.5)
    ax2.axhline(75, color='green', linewidth=2)
    ax2.set(title=f'Synthetic Seismogram ({freq}Hz)', xlabel='Azimuth (degrees)', ylabel='Time (samples)')

    plt.tight_layout()
    return fig_2d, fig_3d

# --- Run Modeling ---
params = {
    'vp1': vp1, 'vp2': vp2, 'vp3': vp3,
    'vs1': vs1, 'vs2': vs2, 'vs3': vs3,
    'd1': d1, 'd2': d2, 'd3': d3,
    'e1': e1, 'e2': e2, 'e3': e3,
    'g1': g1, 'g2': g2, 'g3': g3,
    'dlt1': dlt1, 'dlt2': dlt2, 'dlt3': dlt3,
    'incidence_angle': incidence_angle,
    'freq': freq,
    'depth_target': 2500  # Fixed for simplicity
}

# Generate plots
fig_2d, fig_3d = avaz_modeling(params)

# Display results
st.pyplot(fig_2d)
st.plotly_chart(fig_3d, use_container_width=True)
