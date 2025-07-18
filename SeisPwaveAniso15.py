import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import convolve
from plotly.subplots import make_subplots

# ==============================================
# Core Functions
# ==============================================
def ricker_wavelet(freq, length, dt):
    """Generate a Ricker wavelet for seismic modeling"""
    t = np.arange(-length/2, length/2, dt)
    return (1 - 2*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)

def calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuth):
    """Calculate anisotropic reflectivity coefficients"""
    VP2 = (vp[1] + vp[2])/2
    VS2 = (vs[1] + vs[2])/2
    DEN2 = (d[1] + d[2])/2

    A2 = -0.5 * ((vp[2]-vp[1])/VP2 + (d[2]-d[1])/DEN2)
    
    az_rad = np.radians(azimuth)
    Biso2 = 0.5*((vp[2]-vp[1])/VP2) - 2*(VS2/VP2)**2*(d[2]-d[1])/DEN2 - 4*(VS2/VP2)**2*(vs[2]-vs[1])/VS2
    Baniso2 = 0.5*((dlt[2]-dlt[1]) + 2*(2*VS2/VP2)**2*(g[2]-g[1]))
    Caniso2 = 0.5*((vp[2]-vp[1])/VP2 - (e[2]-e[1])*np.cos(az_rad)**4 + (dlt[2]-dlt[1])*np.sin(az_rad)**2*np.cos(az_rad)**2)
    
    return A2 + (Biso2 + Baniso2*np.cos(az_rad)**2)*np.sin(theta)**2 + Caniso2*np.sin(theta)**2*np.tan(theta)**2

def brown_korringa_substitution(Km, Gm, Ks, Gs, Kf, phi, delta, gamma):
    """Brown-Korringa fluid substitution for anisotropic media"""
    beta = 1 - (Ks/Km)
    K_sat = Ks + (beta**2) / ((phi/Kf) + ((beta - phi)/Km) - (delta*Ks)/(3*Km))
    G_sat = Gs * (1 - (gamma*Ks)/(3*Km))
    
    # Update anisotropy parameters
    delta_sat = delta * (K_sat/Ks)
    gamma_sat = gamma * (G_sat/Gs)
    
    return K_sat, G_sat, delta_sat, gamma_sat

def moduli_to_velocity(K, G, density):
    """Convert bulk and shear moduli to Vp and Vs"""
    Vp = np.sqrt((K + 4/3*G)/density)
    Vs = np.sqrt(G/density)
    return Vp, Vs

def velocity_to_moduli(Vp, Vs, density):
    """Convert Vp and Vs to bulk and shear moduli"""
    G = density * Vs**2
    K = density * Vp**2 - (4/3)*G
    return K, G

def create_3d_plot(x, y, z, vp):
    """Create interactive 3D velocity surface plot"""
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

def pwave_anisotropy_section():
    """Visualize P-wave velocity anisotropy based on Thomsen parameters"""
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
        show_3d = st.checkbox("Show 3D Visualization", True)

    with col2:
        theta = np.linspace(0, 90, 90) * np.pi / 180
        phi = np.linspace(0, 2*np.pi, 90)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Calculate velocity for all directions
        Vp = vp0 * (1 + delta * (np.sin(theta_grid))**2 * (np.cos(theta_grid))**2 
                     + epsilon * (np.sin(theta_grid))**4)
        
        # Convert to Cartesian coordinates
        Vpx = Vp * np.sin(theta_grid) * np.cos(phi_grid)
        Vpy = Vp * np.sin(theta_grid) * np.sin(phi_grid)
        Vpz = Vp * np.cos(theta_grid)
        
        # 2D polar plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(Vpx[0,:], Vpy[0,:], 'b-', linewidth=2, label=f"ε={epsilon:.3f}, δ={delta:.3f}")
        ax.set_xlabel('Vpx [m/s]', fontsize=12)
        ax.set_ylabel('Vpy [m/s]', fontsize=12)
        ax.set_title("P-Wave Velocity Anisotropy", fontsize=14)
        ax.axis('square')
        ax.set_xlim(0, 1.5*vp0)
        ax.set_ylim(0, 1.5*vp0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        
        # 3D plot if enabled
        if show_3d:
            st.subheader("3D Velocity Surface")
            fig_3d = create_3d_plot(Vpx, Vpy, Vpz, Vp)
            st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================
# AVAZ Modeling with Fluid Substitution
# ==============================================
def main():
    st.set_page_config(layout="wide", page_title="AVAZ Modeling with Fluid Substitution")
    st.title("AVAZ Modeling with Brown-Korringa Fluid Substitution")
    
    with st.sidebar:
        st.header("Model Parameters")
        
        # Rock properties for three layers
        layers = ["Upper (1)", "Target (2)", "Lower (3)"]
        params = {}
        
        for i, layer in enumerate(layers, 1):
            st.subheader(f"Layer {layer}")
            params[f'vp{i}'] = st.number_input(f"Vp{i} (m/s)", value=5500 if i!=2 else 4742)
            params[f'vs{i}'] = st.number_input(f"Vs{i} (m/s)", value=3600 if i!=2 else 3292)
            params[f'd{i}'] = st.number_input(f"Density{i} (g/cc)", value=2.6 if i!=2 else 2.4, step=0.1)
            params[f'e{i}'] = st.number_input(f"ε{i}", value=0.1 if i==1 else (-0.01 if i==2 else 0.2), step=0.01)
            params[f'g{i}'] = st.number_input(f"γ{i}", value=0.05 if i==1 else (-0.05 if i==2 else 0.15), step=0.01)
            params[f'dlt{i}'] = st.number_input(f"δ{i}", value=0.0 if i==1 else (-0.13 if i==2 else 0.1), step=0.01)
        
        st.subheader("Acquisition Parameters")
        params['max_angle'] = st.slider("Maximum Angle (deg)", 1, 90, 60)
        params['angle_step'] = st.slider("Angle Step (deg)", 1, 10, 2)
        params['freq'] = st.slider("Wavelet Frequency (Hz)", 10, 100, 45)
        params['azimuth_step'] = st.slider("Azimuth Step (deg)", 1, 30, 10)
        
        st.subheader("Fluid Substitution Parameters")
        enable_fluid_sub = st.checkbox("Enable Fluid Substitution", True)
        if enable_fluid_sub:
            params['phi'] = st.slider("Porosity (ϕ)", 0.01, 0.5, 0.2, 0.01)
            params['Km'] = st.number_input("Mineral Bulk Modulus (GPa)", 10.0, 100.0, 37.0, 1.0)
            params['Gm'] = st.number_input("Mineral Shear Modulus (GPa)", 10.0, 100.0, 44.0, 1.0)
            params['Kf'] = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 5.0, 2.2, 0.1)
            params['new_fluid_density'] = st.number_input("New Fluid Density (g/cc)", 0.1, 1.5, 1.0, 0.1)
        
        selected_angle = st.selectbox("Select angle for 2D comparison", 
                                    options=np.arange(0, params['max_angle']+1, params['angle_step']),
                                    index=3)
        
        # Add colormap selection
        st.subheader("Visualization Options")
        seismic_cmap = st.selectbox(
            "Seismic Colormap",
            options=['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma'],
            index=0
        )
        
        # Add button to show P-wave anisotropy section
        show_anisotropy = st.checkbox("Show P-Wave Anisotropy Section", False)

    if show_anisotropy:
        pwave_anisotropy_section()

    if st.sidebar.button("Run Modeling"):
        with st.spinner("Computing models..."):
            # Original properties
            vp_orig = [params['vp1'], params['vp2'], params['vp3']]
            vs_orig = [params['vs1'], params['vs2'], params['vs3']]
            d_orig = [params['d1'], params['d2'], params['d3']]
            e_orig = [params['e1'], params['e2'], params['e3']]
            g_orig = [params['g1'], params['g2'], params['g3']]
            dlt_orig = [params['dlt1'], params['dlt2'], params['dlt3']]
            
            # Fluid substituted properties (initialize as original)
            vp_sub = vp_orig.copy()
            vs_sub = vs_orig.copy()
            d_sub = d_orig.copy()
            e_sub = e_orig.copy()
            g_sub = g_orig.copy()
            dlt_sub = dlt_orig.copy()
            
            if enable_fluid_sub:
                # Calculate original moduli for target layer
                K_orig, G_orig = velocity_to_moduli(params['vp2'], params['vs2'], params['d2'])
                
                # Perform fluid substitution
                K_sat, G_sat, delta_sat, gamma_sat = brown_korringa_substitution(
                    params['Km']*1e9, params['Gm']*1e9, 
                    K_orig, G_orig,
                    params['Kf']*1e9, 
                    params['phi'], 
                    params['dlt2'], params['g2']
                )
                
                # Convert back to velocities
                new_density = params['d2'] + params['phi']*(params['new_fluid_density'] - 1.0)
                Vp_new, Vs_new = moduli_to_velocity(K_sat, G_sat, new_density)
                
                # Update target layer properties
                vp_sub[1] = Vp_new
                vs_sub[1] = Vs_new
                d_sub[1] = new_density
                dlt_sub[1] = delta_sat
                g_sub[1] = gamma_sat
            
            # Common parameters
            incidence_angles = np.arange(0, params['max_angle']+1, params['angle_step'])
            azimuths = np.arange(0, 361, params['azimuth_step'])
            
            # Compute reflectivity matrices
            reflectivity_orig = np.zeros((len(incidence_angles), len(azimuths)))
            reflectivity_sub = np.zeros((len(incidence_angles), len(azimuths)))
            
            for i, theta_deg in enumerate(incidence_angles):
                theta_rad = np.radians(theta_deg)
                for j, az in enumerate(azimuths):
                    reflectivity_orig[i,j] = calculate_reflectivity(
                        vp_orig, vs_orig, d_orig, e_orig, g_orig, dlt_orig, theta_rad, az
                    )
                    reflectivity_sub[i,j] = calculate_reflectivity(
                        vp_sub, vs_sub, d_sub, e_sub, g_sub, dlt_sub, theta_rad, az
                    )
            
            # Generate synthetic seismic
            n_samples = 150
            wavelet = ricker_wavelet(params['freq'], 0.08, 0.001)
            center_sample = n_samples//2 + len(wavelet)//2
            
            seismic_orig = []
            seismic_sub = []
            for theta_deg in incidence_angles:
                # Original
                R = np.zeros((n_samples, len(azimuths)))
                R[n_samples//2, :] = reflectivity_orig[i, :]
                syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
                seismic_orig.append(syn[center_sample-75:center_sample+75, :])
                
                # Substituted
                R = np.zeros((n_samples, len(azimuths)))
                R[n_samples//2, :] = reflectivity_sub[i, :]
                syn = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
                seismic_sub.append(syn[center_sample-75:center_sample+75, :])
            
            # ==============================================
            # Visualizations
            # ==============================================
            
            # 1. 3D Comparison
            st.header("3D AVAZ Response Comparison")
            
            zmin = min(np.min(reflectivity_orig), np.min(reflectivity_sub))
            zmax = max(np.max(reflectivity_orig), np.max(reflectivity_sub))
            
            col1, col2 = st.columns(2)
            with col1:
                fig_orig = go.Figure(data=[go.Surface(
                    z=reflectivity_orig,
                    x=azimuths,
                    y=incidence_angles,
                    colorscale='Jet',
                    cmin=zmin,
                    cmax=zmax
                )])
                fig_orig.update_layout(
                    scene=dict(
                        xaxis_title='Azimuth (deg)',
                        yaxis_title='Incidence Angle (deg)',
                        zaxis_title='Reflectivity',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                    ),
                    title="Original Response",
                    height=500
                )
                st.plotly_chart(fig_orig, use_container_width=True)
            
            with col2:
                fig_sub = go.Figure(data=[go.Surface(
                    z=reflectivity_sub,
                    x=azimuths,
                    y=incidence_angles,
                    colorscale='Jet',
                    cmin=zmin,
                    cmax=zmax
                )])
                fig_sub.update_layout(
                    scene=dict(
                        xaxis_title='Azimuth (deg)',
                        yaxis_title='Incidence Angle (deg)',
                        zaxis_title='Reflectivity',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                    ),
                    title="Fluid-Substituted Response",
                    height=500
                )
                st.plotly_chart(fig_sub, use_container_width=True)
            
            # 2. 2D Comparison at selected angle
            st.header(f"2D Comparison at {selected_angle}° Incidence")
            
            angle_idx = np.where(incidence_angles == selected_angle)[0][0]
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(azimuths, reflectivity_orig[angle_idx, :], 'b-', label='Original')
            ax1.plot(azimuths, reflectivity_sub[angle_idx, :], 'r--', label='Fluid-Substituted')
            ax1.set_xlabel('Azimuth (degrees)')
            ax1.set_ylabel('Reflectivity')
            ax1.set_title(f'AVAZ Reflectivity at {selected_angle}° Incidence')
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)
            
            # 3. Polar View Comparison
            st.header(f"Polar View Comparison at {selected_angle}° Incidence")
            
            fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            theta_rad = np.radians(azimuths)
            ax2.plot(theta_rad, reflectivity_orig[angle_idx, :], 'b-', label='Original')
            ax2.plot(theta_rad, reflectivity_sub[angle_idx, :], 'r--', label='Fluid-Substituted')
            ax2.set_title(f'Polar AVAZ Response at {selected_angle}° Incidence', pad=20)
            ax2.legend()
            st.pyplot(fig2)
            
            # 4. Seismic Gathers Comparison
            st.header("Synthetic Seismic Gathers Comparison")
            
            n_cols = 3
            n_rows = int(np.ceil(len(incidence_angles)/n_cols))
            
            st.subheader("Original Seismic")
            fig3, axs3 = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
            axs3 = axs3.flatten() if n_rows > 1 else [axs3]
            
            for idx, theta_deg in enumerate(incidence_angles):
                vmax = np.abs(seismic_orig[idx]).max()
                axs3[idx].imshow(seismic_orig[idx], 
                               cmap=seismic_cmap, 
                               aspect='auto',
                               vmin=-vmax, 
                               vmax=vmax,
                               extent=[0, 360, seismic_orig[idx].shape[0], 0])
                axs3[idx].set(title=f'{theta_deg}° Incidence',
                             xlabel='Azimuth' if idx >= (n_rows-1)*n_cols else '',
                             ylabel='Time' if idx % n_cols == 0 else '')
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            st.subheader("Fluid-Substituted Seismic")
            fig4, axs4 = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
            axs4 = axs4.flatten() if n_rows > 1 else [axs4]
            
            for idx, theta_deg in enumerate(incidence_angles):
                vmax = np.abs(seismic_sub[idx]).max()
                axs4[idx].imshow(seismic_sub[idx], 
                               cmap=seismic_cmap, 
                               aspect='auto',
                               vmin=-vmax, 
                               vmax=vmax,
                               extent=[0, 360, seismic_sub[idx].shape[0], 0])
                axs4[idx].set(title=f'{theta_deg}° Incidence',
                             xlabel='Azimuth' if idx >= (n_rows-1)*n_cols else '',
                             ylabel='Time' if idx % n_cols == 0 else '')
            
            plt.tight_layout()
            st.pyplot(fig4)
            
            # 5. Difference Analysis
            st.header("Difference Analysis")
            
            reflectivity_diff = reflectivity_sub - reflectivity_orig
            max_diff = np.max(np.abs(reflectivity_diff))
            
            # Difference matrix
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            im = ax5.imshow(reflectivity_diff.T, aspect='auto', 
                          extent=[0, params['max_angle'], 0, 360],
                          cmap='RdBu', vmin=-max_diff, vmax=max_diff,
                          origin='lower')
            ax5.set(xlabel='Incidence Angle (degrees)', 
                   ylabel='Azimuth (degrees)',
                   title='Reflectivity Difference (Fluid-Substituted - Original)')
            plt.colorbar(im, ax=ax5, label='Reflectivity Difference')
            st.pyplot(fig5)
            
            # 3D Difference plot
            fig6 = go.Figure(data=[go.Surface(
                z=reflectivity_diff,
                x=azimuths,
                y=incidence_angles,
                colorscale='RdBu',
                cmin=-max_diff,
                cmax=max_diff
            )])
            fig6.update_layout(
                scene=dict(
                    xaxis_title='Azimuth (deg)',
                    yaxis_title='Incidence Angle (deg)',
                    zaxis_title='Reflectivity Difference',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                ),
                title="3D Reflectivity Difference",
                height=600
            )
            st.plotly_chart(fig6, use_container_width=True)

if __name__ == "__main__":
    main()
