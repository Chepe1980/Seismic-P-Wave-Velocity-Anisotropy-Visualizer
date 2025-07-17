import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import convolve

# ==============================================
# Core Functions
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

def brown_korringa_substitution(Km, Gm, Ks, Kf, phi, alpha):
    """
    Brown-Korringa fluid substitution for anisotropic media
    Km, Gm: Mineral bulk and shear moduli
    Ks: Solid matrix bulk modulus (frame)
    Kf: Fluid bulk modulus
    phi: Porosity
    alpha: Anisotropy parameter (1-3 for typical rocks)
    Returns: New bulk (K_sat) and shear (G_sat) moduli
    """
    beta = 1 - (Ks/Km)
    K_sat = Ks + (beta**2) / ((phi/Kf) + ((beta - phi)/Km) - (alpha*Ks)/(3*Km))
    G_sat = Gm * (1 - (alpha*Ks)/(3*Km))  # Shear modulus affected by anisotropy
    return K_sat, G_sat

def moduli_to_velocity(K, G, density):
    """Convert bulk and shear moduli to Vp and Vs"""
    Vp = np.sqrt((K + 4/3*G)/density)
    Vs = np.sqrt(G/density)
    return Vp, Vs

# ==============================================
# P-Wave Anisotropy Visualizer
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
            key="epsilon_pwave"
        )
        delta = st.number_input(
            "δ (Delta)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=-0.01, 
            step=0.01,
            key="delta_pwave"
        )
        vp0 = st.number_input(
            "Vp₀ (m/s)", 
            min_value=1000, 
            max_value=8000, 
            value=3000,
            key="vp0_pwave"
        )
        st.markdown("---")
        show_all_angles = st.checkbox("Show all incidence angles", value=False, key="show_all_pwave")

    with col2:
        theta = np.linspace(0, 90, 90) * np.pi / 180
        Vp = vp0 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 
                     + epsilon * (np.sin(theta))**4)
        
        Vpx = Vp * np.sin(theta)
        Vpy = Vp * np.cos(theta)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if show_all_angles:
            azimuths = np.linspace(0, 2*np.pi, 36)
            for az in azimuths:
                Vpx_az = Vp * np.sin(theta) * np.cos(az)
                Vpy_az = Vp * np.sin(theta) * np.sin(az)
                ax.plot(Vpx_az, Vpy_az, 'b-', alpha=0.3, linewidth=0.5)
            
            ax.plot(Vpx, np.zeros_like(Vpx), 'r-', label='X-axis (0° azimuth)')
            ax.plot(np.zeros_like(Vpy), Vpy, 'g-', label='Y-axis (90° azimuth)')
        else:
            ax.plot(Vpx, Vpy, 'b-', linewidth=2, label=f"ε={epsilon:.3f}, δ={delta:.3f}")

        ax.set_xlabel('Vpx [m/s]', fontsize=12)
        ax.set_ylabel('Vpy [m/s]', fontsize=12)
        ax.set_title("P-Wave Velocity Anisotropy", fontsize=14)
        ax.axis('square')
        ax.set_xlim(-1.5*vp0, 1.5*vp0)
        ax.set_ylim(-1.5*vp0, 1.5*vp0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

# ==============================================
# AVAz Modeling Section with Fluid Substitution
# ==============================================
def avaz_section():
    st.header("AVAz Modeling Tool with Fluid Substitution")
    st.markdown("Visualize Amplitude Variation with Azimuth responses for anisotropic media.")

    with st.sidebar:
        st.subheader("Rock Properties")
        
        layers = ["Upper (1)", "Target (2)", "Lower (3)"]
        params = {}
        
        for i, layer in enumerate(layers, 1):
            st.markdown(f"**Layer {layer}**")
            params[f'vp{i}'] = st.number_input(f"Vp{i} (m/s)", value=5500 if i!=2 else 4742, key=f"vp{i}_avaz")
            params[f'vs{i}'] = st.number_input(f"Vs{i} (m/s)", value=3600 if i!=2 else 3292, key=f"vs{i}_avaz")
            params[f'd{i}'] = st.number_input(f"Density{i} (g/cc)", value=2.6 if i!=2 else 2.4, step=0.1, key=f"d{i}_avaz")
            params[f'e{i}'] = st.number_input(f"ε{i}", value=0.1 if i==1 else (-0.01 if i==2 else 0.2), step=0.01, key=f"e{i}_avaz")
            params[f'g{i}'] = st.number_input(f"γ{i}", value=0.05 if i==1 else (-0.05 if i==2 else 0.15), step=0.01, key=f"g{i}_avaz")
            params[f'dlt{i}'] = st.number_input(f"δ{i}", value=0.0 if i==1 else (-0.13 if i==2 else 0.1), step=0.01, key=f"dlt{i}_avaz")
            st.markdown("---")

        st.subheader("Acquisition")
        params['max_angle'] = st.number_input("Maximum Angle (deg)", 1, 90, 60, key="max_angle_avaz")
        params['angle_step'] = st.number_input("Angle Step (deg)", 1, 10, 2, key="angle_step_avaz")
        params['freq'] = st.number_input("Wavelet Frequency (Hz)", 10, 100, 45, key="freq_avaz")
        params['azimuth_step'] = st.number_input("Azimuth Step (deg)", 1, 30, 10, key="azimuth_step_avaz")
        
        # Fluid substitution parameters
        st.subheader("Fluid Substitution (Brown-Korringa)")
        enable_fluid_sub = st.checkbox("Enable Fluid Substitution", value=False, key="enable_fluid_sub")
        
        if enable_fluid_sub:
            params['phi'] = st.number_input("Porosity (ϕ)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, key="phi")
            params['Km'] = st.number_input("Mineral Bulk Modulus (GPa)", min_value=10.0, max_value=100.0, value=37.0, step=1.0, key="Km")
            params['Gm'] = st.number_input("Mineral Shear Modulus (GPa)", min_value=10.0, max_value=100.0, value=44.0, step=1.0, key="Gm")
            params['Kf'] = st.number_input("Fluid Bulk Modulus (GPa)", min_value=0.1, max_value=5.0, value=2.2, step=0.1, key="Kf")
            params['alpha'] = st.number_input("Anisotropy Parameter (α)", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="alpha")
            params['new_fluid_density'] = st.number_input("New Fluid Density (g/cc)", min_value=0.1, max_value=1.5, value=1.0, step=0.1, key="new_fluid_density")
        
        # Display mode selection
        display_mode = st.radio(
            "Display Mode",
            ["All Incidence Angles", "Select Specific Angles"],
            index=0,
            key="display_mode"
        )
        
        if display_mode == "Select Specific Angles":
            selected_angles = st.multiselect(
                "Select Incidence Angles to Display",
                options=np.arange(0, params['max_angle']+1, params['angle_step']),
                default=[0, 15, 30, 45],
                key="selected_angles"
            )

    if st.sidebar.button("Run Full AVAz Modeling"):
        with st.spinner("Computing full angle-azimuth response..."):
            # Original rock properties
            vp = [params['vp1'], params['vp2'], params['vp3']]
            vs = [params['vs1'], params['vs2'], params['vs3']]
            d = [params['d1'], params['d2'], params['d3']]
            e = [params['e1'], params['e2'], params['e3']]
            g = [params['g1'], params['g2'], params['g3']]
            dlt = [params['dlt1'], params['dlt2'], params['dlt3']]
            
            # Apply fluid substitution if enabled
            if enable_fluid_sub:
                st.info("Applying Brown-Korringa fluid substitution to target layer...")
                
                # Calculate original moduli for target layer (layer 2)
                density = params['d2']
                Vp_orig = params['vp2']
                Vs_orig = params['vs2']
                K_orig = density * (Vp_orig**2 - 4/3*Vs_orig**2)
                G_orig = density * Vs_orig**2
                
                # Perform fluid substitution
                K_sat, G_sat = brown_korringa_substitution(
                    params['Km']*1e9, params['Gm']*1e9, 
                    K_orig, params['Kf']*1e9, 
                    params['phi'], params['alpha']
                )
                
                # Convert back to velocities
                new_density = params['d2'] + params['phi']*(params['new_fluid_density'] - 1.0)  # Simple density mixing
                Vp_new, Vs_new = moduli_to_velocity(K_sat, G_sat, new_density)
                
                # Update target layer properties
                vp[1] = Vp_new
                vs[1] = Vs_new
                d[1] = new_density
                
                st.success(f"Fluid substitution complete - New Vp: {Vp_new:.1f} m/s, Vs: {Vs_new:.1f} m/s, Density: {new_density:.2f} g/cc")
            
            incidence_angles = np.arange(0, params['max_angle']+1, params['angle_step'])
            azimuths = np.arange(0, 361, params['azimuth_step'])
            
            vp1, vp2 = vp[0], vp[1]
            critical_angle = np.degrees(np.arcsin(vp1/vp2)) if vp1 < vp2 else 90
            st.info(f"Critical angle: {critical_angle:.1f}° (computed from Vp1/Vp2 ratio)")

            # Compute reflectivity matrix
            reflectivity_matrix = np.zeros((len(incidence_angles), len(azimuths)))
            
            for i, theta_deg in enumerate(incidence_angles):
                theta_rad = np.radians(theta_deg)
                for j, az in enumerate(azimuths):
                    reflectivity_matrix[i,j] = calculate_reflectivity(
                        vp, vs, d, e, g, dlt, theta_rad, az
                    )

            # Plot 1: Full AVAz matrix
            st.subheader("Full AVAz Response Matrix")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            im = ax1.imshow(reflectivity_matrix.T, aspect='auto', 
                          extent=[0, params['max_angle'], 0, 360],
                          cmap='jet', vmin=-0.4, vmax=0.2,
                          origin='lower')
            
            if vp1 < vp2:
                ax1.axvline(x=critical_angle, color='white', linestyle='--', 
                           label=f'Critical Angle ({critical_angle:.1f}°)')
            
            ax1.set(xlabel='Incidence Angle (degrees)', ylabel='Azimuth (degrees)',
                   title='Full AVAz Response Matrix (All Angles/Azimuths)')
            plt.colorbar(im, ax=ax1, label='Reflectivity')
            ax1.legend()
            st.pyplot(fig1)

            # Plot 2: Angle gathers based on display mode selection
            st.subheader("Angle Gathers Visualization")
            
            n_samples = 150
            wavelet = ricker_wavelet(params['freq'], 0.08, 0.001)
            center_sample = n_samples//2 + len(wavelet)//2
            
            if display_mode == "All Incidence Angles":
                # Show all angles in a grid
                n_cols = 3
                n_rows = int(np.ceil(len(incidence_angles)/n_cols))
                fig2, axs = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
                axs = axs.flatten() if n_rows > 1 else [axs]
                
                for idx, theta_deg in enumerate(incidence_angles):
                    R = np.zeros((n_samples, len(azimuths)))
                    R[n_samples//2, :] = reflectivity_matrix[idx, :]
                    syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
                    syn = syn[center_sample-75:center_sample+75, :]
                    
                    vmax = np.abs(syn).max()
                    axs[idx].imshow(syn, cmap='seismic', aspect='auto',
                                   vmin=-vmax, vmax=vmax,
                                   extent=[0, 360, syn.shape[0], 0])
                    axs[idx].set(title=f'{theta_deg}° Incidence',
                                xlabel='Azimuth' if idx >= (n_rows-1)*n_cols else '',
                                ylabel='Time' if idx % n_cols == 0 else '')
                
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                # Show selected angles in a single figure
                fig2, ax = plt.subplots(figsize=(12, 6))
                
                for theta_deg in selected_angles:
                    if theta_deg in incidence_angles:
                        idx = np.where(incidence_angles == theta_deg)[0][0]
                        R = np.zeros((n_samples, len(azimuths)))
                        R[n_samples//2, :] = reflectivity_matrix[idx, :]
                        syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
                        syn = syn[center_sample-75:center_sample+75, :]
                        
                        # Offset each gather for better visualization
                        offset = selected_angles.index(theta_deg) * 50
                        for az_idx in range(syn.shape[1]):
                            ax.plot(azimuths[az_idx] + 0.5, syn[:, az_idx] + offset, 'k-', linewidth=0.5)
                        
                        ax.text(360, offset, f'{theta_deg}°', va='center', ha='left')
                
                ax.set_xlabel('Azimuth (degrees)')
                ax.set_ylabel('Amplitude (offset by angle)')
                ax.set_title('Selected Angle Gathers (All Azimuths)')
                ax.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig2)

            # Plot 3: 3D surface
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
st.sidebar.header("Navigation")
tool = st.sidebar.radio(
    "Select Tool",
    ["P-Wave Anisotropy", "AVAz Modeling"],
    index=0
)

if tool == "P-Wave Anisotropy":
    pwave_anisotropy_section()
else:
    avaz_section()
