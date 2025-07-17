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

def brown_korringa_substitution(Km, Gm, Ks, Gs, Kf, phi, delta, gamma):
    """
    Brown-Korringa fluid substitution for anisotropic media
    Returns: New bulk (K_sat), shear (G_sat) moduli, and updated anisotropy parameters
    """
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

# ==============================================
# P-Wave Anisotropy Visualizer
# ==============================================
def pwave_anisotropy_section():
    st.header("P-Wave Velocity Anisotropy Visualizer")
    # ... (keep existing pwave_anisotropy_section code unchanged)

# ==============================================
# AVAz Modeling Section with Fluid Substitution Comparison
# ==============================================
def avaz_section():
    st.header("AVAz Modeling with Fluid Substitution Comparison")
    st.markdown("Compare original and fluid-substituted AVAz responses")

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
        enable_fluid_sub = st.checkbox("Enable Fluid Substitution", value=True, key="enable_fluid_sub")
        
        if enable_fluid_sub:
            params['phi'] = st.number_input("Porosity (ϕ)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, key="phi")
            params['Km'] = st.number_input("Mineral Bulk Modulus (GPa)", min_value=10.0, max_value=100.0, value=37.0, step=1.0, key="Km")
            params['Gm'] = st.number_input("Mineral Shear Modulus (GPa)", min_value=10.0, max_value=100.0, value=44.0, step=1.0, key="Gm")
            params['Kf'] = st.number_input("Fluid Bulk Modulus (GPa)", min_value=0.1, max_value=5.0, value=2.2, step=0.1, key="Kf")
            params['new_fluid_density'] = st.number_input("New Fluid Density (g/cc)", min_value=0.1, max_value=1.5, value=1.0, step=0.1, key="new_fluid_density")

    if st.sidebar.button("Run AVAz Modeling"):
        with st.spinner("Computing responses..."):
            # Original rock properties
            vp_orig = [params['vp1'], params['vp2'], params['vp3']]
            vs_orig = [params['vs1'], params['vs2'], params['vs3']]
            d_orig = [params['d1'], params['d2'], params['d3']]
            e_orig = [params['e1'], params['e2'], params['e3']]
            g_orig = [params['g1'], params['g2'], params['g3']]
            dlt_orig = [params['dlt1'], params['dlt2'], params['dlt3']]
            
            # Create copy for fluid-substituted properties
            vp_sub = vp_orig.copy()
            vs_sub = vs_orig.copy()
            d_sub = d_orig.copy()
            e_sub = e_orig.copy()
            g_sub = g_orig.copy()
            dlt_sub = dlt_orig.copy()
            
            # Apply fluid substitution if enabled
            if enable_fluid_sub:
                # Calculate original moduli for target layer (layer 2)
                K_orig, G_orig = velocity_to_moduli(params['vp2'], params['vs2'], params['d2'])
                
                # Perform fluid substitution using the layer's anisotropy parameters
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
                
                # Update target layer properties for substituted case
                vp_sub[1] = Vp_new
                vs_sub[1] = Vs_new
                d_sub[1] = new_density
                dlt_sub[1] = delta_sat
                g_sub[1] = gamma_sat
            
            # Common parameters
            incidence_angles = np.arange(0, params['max_angle']+1, params['angle_step'])
            azimuths = np.arange(0, 361, params['azimuth_step'])
            
            # Compute both reflectivity matrices
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
            
            # Create comparison figure
            st.subheader("3D AVAz Response Comparison")
            
            # Calculate common z-axis limits for fair comparison
            zmin = min(np.min(reflectivity_orig), np.min(reflectivity_sub))
            zmax = max(np.max(reflectivity_orig), np.max(reflectivity_sub))
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                subplot_titles=("Original Response", "Fluid-Substituted Response"),
                horizontal_spacing=0.1
            )
            
            # Original response
            fig.add_trace(
                go.Surface(
                    z=reflectivity_orig,
                    x=azimuths,
                    y=incidence_angles,
                    colorscale='Jet',
                    cmin=zmin,
                    cmax=zmax,
                    colorbar=dict(x=0.45, title='Reflectivity'),
                    name="Original"
                ),
                row=1, col=1
            )
            
            # Fluid-substituted response
            fig.add_trace(
                go.Surface(
                    z=reflectivity_sub,
                    x=azimuths,
                    y=incidence_angles,
                    colorscale='Jet',
                    cmin=zmin,
                    cmax=zmax,
                    colorbar=dict(x=1.0, title='Reflectivity'),
                    name="Substituted"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                scene1=dict(
                    xaxis_title='Azimuth (deg)',
                    yaxis_title='Incidence Angle (deg)',
                    zaxis_title='Reflectivity',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                scene2=dict(
                    xaxis_title='Azimuth (deg)',
                    yaxis_title='Incidence Angle (deg)',
                    zaxis_title='Reflectivity',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                ),
                height=600,
                width=1200,
                margin=dict(l=50, r=50, b=50, t=50),
                title_text="AVAz Response Comparison: Original vs Fluid-Substituted"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show difference plot
            st.subheader("Difference Between Responses")
            
            reflectivity_diff = reflectivity_sub - reflectivity_orig
            max_diff = np.max(np.abs(reflectivity_diff))
            
            fig_diff = go.Figure(data=[go.Surface(
                z=reflectivity_diff,
                x=azimuths,
                y=incidence_angles,
                colorscale='RdBu',
                cmin=-max_diff,
                cmax=max_diff,
                colorbar=dict(title='Reflectivity Difference')
            )])
            
            fig_diff.update_layout(
                scene=dict(
                    xaxis_title='Azimuth (deg)',
                    yaxis_title='Incidence Angle (deg)',
                    zaxis_title='Reflectivity Difference',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                ),
                height=600,
                width=800,
                title_text="Fluid-Substituted minus Original Response"
            )
            
            st.plotly_chart(fig_diff, use_container_width=True)

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
