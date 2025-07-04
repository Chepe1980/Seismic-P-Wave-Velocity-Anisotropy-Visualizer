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
# Core Functions (Shared by both tools)
# ==============================================
def ricker_wavelet(freq, length, dt):
    t = np.arange(-length/2, length/2, dt)
    return (1 - 2*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)

def calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuth):
    """Calculate reflectivity for a single angle and azimuth"""
    VP2 = (vp[1] + vp[2])/2
    VS2 = (vs[1] + vs[2])/2
    DEN2 = (d[1] + d[2])/2

    A2 = -0.5 * ((vp[2]-vp[1])/VP2 + (d[2]-d[1])/DEN2)
    
    az_rad = np.radians(azimuth)
    Biso2 = 0.5*((vp[2]-vp[1])/VP2) - 2*(VS2/VP2)**2*(d[2]-d[1])/DEN2 - 4*(VS2/VP2)**2*(vs[2]-vs[1])/VS2
    Baniso2 = 0.5*((dlt[2]-dlt[1]) + 2*(2*VS2/VP2)**2*(g[2]-g[1]))
    Caniso2 = 0.5*((vp[2]-vp[1])/VP2 - (e[2]-e[1])*np.cos(az_rad)**4 + (dlt[2]-dlt[1])*np.sin(az_rad)**2*np.cos(az_rad)**2)
    
    return A2 + (Biso2 + Baniso2*np.cos(az_rad)**2)*np.sin(theta)**2 + Caniso2*np.sin(theta)**2*np.tan(theta)**2

# ==============================================
# AVAz Modeling Section (Enhanced)
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
        params['max_angle'] = st.number_input("Maximum Angle (deg)", 1, 90, 60, key="max_angle")
        params['angle_step'] = st.number_input("Angle Step (deg)", 1, 10, 2, key="angle_step")
        params['freq'] = st.number_input("Wavelet Frequency (Hz)", 10, 100, 45, key="freq")
        params['azimuth_step'] = st.number_input("Azimuth Step (deg)", 1, 30, 10, key="azimuth_step")

    if st.sidebar.button("Run Full Modeling"):
        with st.spinner("Computing full angle-azimuth response..."):
            # Prepare parameters
            vp = [params['vp1'], params['vp2'], params['vp3']]
            vs = [params['vs1'], params['vs2'], params['vs3']]
            d = [params['d1'], params['d2'], params['d3']]
            e = [params['e1'], params['e2'], params['e3']]
            g = [params['g1'], params['g2'], params['g3']]
            dlt = [params['dlt1'], params['dlt2'], params['dlt3']]
            
            # Create angle and azimuth ranges
            incidence_angles = np.arange(0, params['max_angle']+1, params['angle_step'])
            azimuths = np.arange(0, 361, params['azimuth_step'])
            
            # Calculate critical angle
            vp1, vp2 = vp[0], vp[1]
            critical_angle = np.degrees(np.arcsin(vp1/vp2)) if vp1 < vp2 else 90
            st.info(f"Critical angle: {critical_angle:.1f}° (computed from Vp1/Vp2 ratio)")

            # ==============================================
            # 1. Full Angle-Azimuth Reflectivity Matrix
            # ==============================================
            reflectivity_matrix = np.zeros((len(incidence_angles), len(azimuths)))
            
            for i, theta_deg in enumerate(incidence_angles):
                theta_rad = np.radians(theta_deg)
                for j, az in enumerate(azimuths):
                    reflectivity_matrix[i,j] = calculate_reflectivity(
                        vp, vs, d, e, g, dlt, theta_rad, az
                    )

            # Plot reflectivity matrix
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            im = ax1.imshow(reflectivity_matrix.T, aspect='auto', 
                          extent=[0, params['max_angle'], 0, 360],
                          cmap='jet', vmin=-0.4, vmax=0.2)
            ax1.set(xlabel='Incidence Angle (degrees)', ylabel='Azimuth (degrees)',
                   title='Reflectivity Matrix (All Angles/Azimuths)')
            plt.colorbar(im, ax=ax1, label='Reflectivity')
            st.pyplot(fig1)

            # ==============================================
            # 2. Synthetic Gathers for Each Angle
            # ==============================================
            st.subheader("Angle Gathers")
            
            # Create time axis
            n_samples = 150
            wavelet = ricker_wavelet(params['freq'], 0.08, 0.001)
            center_sample = n_samples//2 + len(wavelet)//2
            
            # Create figure with subplots for each angle
            n_cols = 3
            n_rows = int(np.ceil(len(incidence_angles)/n_cols))
            fig2, axs = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
            axs = axs.flatten() if n_rows > 1 else [axs]
            
            for idx, theta_deg in enumerate(incidence_angles):
                # Create synthetic traces
                R = np.zeros((n_samples, len(azimuths)))
                R[n_samples//2, :] = reflectivity_matrix[idx, :]
                syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
                syn = syn[center_sample-75:center_sample+75, :]
                
                # Plot gather
                vmax = np.abs(syn).max()
                axs[idx].imshow(syn, cmap='seismic', aspect='auto',
                               vmin=-vmax, vmax=vmax,
                               extent=[0, 360, syn.shape[0], 0])
                axs[idx].set(title=f'{theta_deg}° Incidence',
                            xlabel='Azimuth' if idx >= (n_rows-1)*n_cols else '',
                            ylabel='Time' if idx % n_cols == 0 else '')
            
            plt.tight_layout()
            st.pyplot(fig2)

            # ==============================================
            # 3. 3D Surface Plot (Interactive)
            # ==============================================
            st.subheader("3D AVAz Response")
            fig3d = go.Figure(data=[go.Surface(
                z=reflectivity_matrix,
                x=azimuths,
                y=incidence_angles,
                colorscale='Jet',
                colorbar=dict(title='Reflectivity'),
            )])
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Azimuth (degrees)',
                    yaxis_title='Incidence Angle (degrees)',
                    zaxis_title='Reflectivity',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                ),
                height=800,
                width=1000
            )
            st.plotly_chart(fig3d, use_container_width=True)

# ==============================================
# App Navigation
# ==============================================
tool = st.sidebar.radio(
    "Select Tool",
    ["AVAz Modeling"],
    index=0
)

avaz_section()
