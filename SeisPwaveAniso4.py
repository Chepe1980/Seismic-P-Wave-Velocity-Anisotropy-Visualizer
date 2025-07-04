import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import convolve

# ==============================================
# App Configuration
# ==============================================
st.set_page_config(layout="wide")
st.title("Seismic Anisotropy Analysis Toolkit")

# ==============================================
# P-Wave Anisotropy Visualizer (Original Code)
# ==============================================
def pwave_anisotropy_section():
    st.header("P-Wave Velocity Anisotropy Visualizer")
    st.markdown("Explore how Thomsen parameters (ε, δ) affect P-wave velocity anisotropy.")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        epsilon = st.number_input(
            "ε (Epsilon)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=-0.01, 
            step=0.01,
            key="epsilon"
        )
        delta = st.number_input(
            "δ (Delta)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=-0.01, 
            step=0.01,
            key="delta"
        )
        vp0 = st.number_input(
            "Vp₀ (m/s)", 
            min_value=1000, 
            max_value=8000, 
            value=3000,
            key="vp0"
        )

    with col2:
        theta = np.linspace(0, 90, 90) * np.pi / 180
        Vp = vp0 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 
                     + epsilon * (np.sin(theta))**4)
        
        Vpx = Vp * np.sin(theta)
        Vpy = Vp * np.cos(theta)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(Vpx, Vpy, 'b-', linewidth=2, label=f"ε={epsilon:.3f}, δ={delta:.3f}")
        ax.set_xlabel('Vpx [m/s]', fontsize=12)
        ax.set_ylabel('Vpy [m/s]', fontsize=12)
        ax.set_title("P-Wave Velocity Anisotropy", fontsize=14)
        ax.axis('square')
        ax.set_xlim(0, 1.5*vp0)
        ax.set_ylim(0, 1.5*vp0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

# ==============================================
# AVAz Modeling (Your New Code)
# ==============================================
def avaz_section():
    st.header("AVAz Modeling Tool")
    st.markdown("Visualize Amplitude Variation with Azimuth responses for anisotropic media.")

    # Sidebar inputs
    with st.sidebar:
        st.subheader("Rock Properties")
        
        # Layer parameters
        layers = ["Upper (1)", "Target (2)", "Lower (3)"]
        params = {}
        
        for i, layer in enumerate(layers, 1):
            st.markdown(f"**Layer {layer}**")
            params[f'vp{i}'] = st.number_input(f"Vp{i} (m/s)", value=5500 if i!=2 else 4742, key=f"vp{i}")
            params[f'vs{i}'] = st.number_input(f"Vs{i} (m/s)", value=3600 if i!=2 else 3292, key=f"vs{i}")
            params[f'd{i}'] = st.number_input(f"Density{i} (g/cc)", value=2.6 if i!=2 else 2.4, step=0.1, key=f"d{i}")
            params[f'e{i}'] = st.number_input(f"ε{i}", value=0.1 if i==1 else (-0.01 if i==2 else 0.2), step=0.01, key=f"e{i}")
            params[f'g{i}'] = st.number_input(f"γ{i}", value=0.05 if i==1 else (-0.05 if i==2 else 0.15), step=0.01, key=f"g{i}")
            params[f'dlt{i}'] = st.number_input(f"δ{i}", value=0.0 if i==1 else (-0.13 if i==2 else 0.1), step=0.01, key=f"dlt{i}")
            st.markdown("---")

        # Acquisition parameters
        st.subheader("Acquisition")
        params['incidence_angle'] = st.number_input("Incidence Angle (deg)", 0, 90, 60, key="theta")
        params['freq'] = st.number_input("Wavelet Frequency (Hz)", 10, 100, 45, key="freq")

    # Core functions
    def ricker_wavelet(freq, length, dt):
        t = np.arange(-length/2, length/2, dt)
        return (1 - 2*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)

    def calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuths):
        VP2 = (vp[1] + vp[2])/2
        VS2 = (vs[1] + vs[2])/2
        DEN2 = (d[1] + d[2])/2

        A2 = -0.5 * ((vp[2]-vp[1])/VP2 + (d[2]-d[1])/DEN2)
        ref = np.zeros(len(azimuths))
        
        for idx, az in enumerate(azimuths):
            az_rad = np.radians(az)
            Biso2 = 0.5*((vp[2]-vp[1])/VP2) - 2*(VS2/VP2)**2*(d[2]-d[1])/DEN2 - 4*(VS2/VP2)**2*(vs[2]-vs[1])/VS2
            Baniso2 = 0.5*((dlt[2]-dlt[1]) + 2*(2*VS2/VP2)**2*(g[2]-g[1]))
            Caniso2 = 0.5*((vp[2]-vp[1])/VP2 - (e[2]-e[1])*np.cos(az_rad)**4 + (dlt[2]-dlt[1])*np.sin(az_rad)**2*np.cos(az_rad)**2)
            ref[idx] = A2 + (Biso2 + Baniso2*np.cos(az_rad)**2)*np.sin(theta)**2 + Caniso2*np.sin(theta)**2*np.tan(theta)**2
        
        return ref

    # Main computation
    if st.sidebar.button("Run Modeling"):
        with st.spinner("Computing AVAz response..."):
            # Prepare parameters
            vp = [params['vp1'], params['vp2'], params['vp3']]
            vs = [params['vs1'], params['vs2'], params['vs3']]
            d = [params['d1'], params['d2'], params['d3']]
            e = [params['e1'], params['e2'], params['e3']]
            g = [params['g1'], params['g2'], params['g3']]
            dlt = [params['dlt1'], params['dlt2'], params['dlt3']]
            theta = np.radians(params['incidence_angle'])
            
            # Azimuthal analysis
            azimuths = np.arange(0, 361)
            ref = calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuths)

            # Generate synthetic seismogram
            R = np.zeros((150, len(azimuths)))
            R[60, :] = ref
            wavelet = ricker_wavelet(params['freq'], 0.08, 0.001)
            syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
            center = 60 + len(wavelet)//2
            syn_n = syn[center-75:center+75, :]

            # Create plots
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sc = ax1.scatter(azimuths, ref, c=ref, cmap='jet', norm=Normalize(vmin=-0.4, vmax=0.2))
            ax1.set(title=f'Azimuthal Reflectivity at {params["incidence_angle"]}°', 
                   xlabel='Azimuth (degrees)', ylabel='Reflectivity')
            plt.colorbar(sc, ax=ax1).set_label('Reflectivity')

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            vmax = np.abs(syn_n).max()
            ax2.imshow(syn_n, cmap='seismic', aspect='auto', 
                      vmin=-vmax, vmax=vmax, extent=[0, 360, syn_n.shape[0], 0])
            for az in range(0, 361, 30):
                ax2.plot(az + syn_n[:,az]/vmax*10, np.arange(syn_n.shape[0]), 'k', linewidth=0.5)
            ax2.axhline(75, color='green', linewidth=2)
            ax2.set(title=f'Synthetic Seismogram ({params["freq"]}Hz)', 
                   xlabel='Azimuth (degrees)', ylabel='Time (samples)')

            # 3D plot
            incidence_angles = np.linspace(0, 60, 30)
            azimuths_3d = np.arange(0, 361, 5)
            reflectivity_3d = np.zeros((len(incidence_angles), len(azimuths_3d)))
            
            for i, ang in enumerate(incidence_angles):
                reflectivity_3d[i,:] = calculate_reflectivity(vp, vs, d, e, g, dlt, np.radians(ang), azimuths_3d)

            fig3d = go.Figure(data=[go.Surface(
                z=reflectivity_3d,
                x=azimuths_3d,
                y=incidence_angles,
                colorscale='Jet',
                colorbar=dict(title='Reflectivity'),
            )])
            fig3d.update_layout(
                title='3D AVAz Response',
                scene=dict(
                    xaxis_title='Azimuth (degrees)',
                    yaxis_title='Incidence Angle (degrees)',
                    zaxis_title='Reflectivity',
                    camera=dict(eye=dict(x=0.2, y=0.3, z=0.6))
            ))

            # Display results
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.plotly_chart(fig3d, use_container_width=True)

# ==============================================
# App Navigation
# ==============================================
tool = st.sidebar.radio(
    "Select Tool",
    ["P-Wave Anisotropy", "AVAz Modeling"],
    index=0
)

if tool == "P-Wave Anisotropy":
    pwave_anisotropy_section()
else:
    avaz_section()
