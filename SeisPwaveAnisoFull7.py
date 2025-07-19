import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import convolve
import pandas as pd
import io

# ==============================================
# Core Functions (unchanged)
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

def create_3d_plot(x, y, z, values, title, colorscale='Viridis'):
    """Create interactive 3D surface plot"""
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=values,
            colorscale=colorscale,
            colorbar=dict(title='Reflectivity'),
            opacity=0.9,
            hoverinfo='x+y+z+text',
            text=[f'R: {val:.4f}' for val in values.flatten()]
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Azimuth (deg)',
            yaxis_title='Angle (deg)',
            zaxis_title='Reflectivity',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    return fig

def pwave_anisotropy_section(epsilon, delta, vp0, title_suffix=""):
    """Visualize P-wave velocity anisotropy based on Thomsen parameters"""
    st.header(f"P-Wave Velocity Anisotropy {title_suffix}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        st.write(f"ε (Epsilon): {epsilon:.3f}")
        st.write(f"δ (Delta): {delta:.3f}")
        st.write(f"Vp₀: {vp0:.0f} m/s")
        show_3d = st.checkbox(f"Show 3D Visualization {title_suffix}", True, key=f"show3d_{title_suffix}")

    with col2:
        # Calculate Vp for 2D plot
        theta = np.linspace(0, 90, 90) * np.pi / 180
        Vp = vp0 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 + epsilon * (np.sin(theta))**4)
        
        # Convert to Cartesian coordinates for 2D plot
        Vpx = Vp * np.sin(theta)
        Vpy = Vp * np.cos(theta)
        
        # 2D polar plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(Vpx, Vpy, 'b-', linewidth=2, label=f"ε={epsilon:.3f}, δ={delta:.3f}")
        ax.set_xlabel('Vpx [m/s]', fontsize=12)
        ax.set_ylabel('Vpy [m/s]', fontsize=12)
        ax.set_title(f"P-Wave Velocity Anisotropy {title_suffix}", fontsize=14)
        ax.axis('square')
        ax.set_xlim(0, 1.5*vp0)
        ax.set_ylim(0, 1.5*vp0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        
        # 3D plot if enabled
        if show_3d:
            st.subheader(f"3D Velocity Surface {title_suffix}")
            # Calculate full 3D velocity field
            theta_3d = np.linspace(0, np.pi, 90)
            phi_3d = np.linspace(0, 2*np.pi, 90)
            theta_grid, phi_grid = np.meshgrid(theta_3d, phi_3d)
            
            Vp_3d = vp0 * (1 + delta * (np.sin(theta_grid))**2 * (np.cos(theta_grid))**2 
                          + epsilon * (np.sin(theta_grid))**4)
            
            # Convert to Cartesian coordinates
            x = Vp_3d * np.sin(theta_grid) * np.cos(phi_grid)
            y = Vp_3d * np.sin(theta_grid) * np.sin(phi_grid)
            z = Vp_3d * np.cos(theta_grid)
            
            fig_3d = go.Figure(data=[
                go.Surface(
                    x=x, y=y, z=z,
                    surfacecolor=Vp_3d,
                    colorscale='Viridis',
                    colorbar=dict(title='Velocity (m/s)'),
                    opacity=0.9
                )
            ])
            fig_3d.update_layout(title=f'3D Velocity {title_suffix}', height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

def process_excel_data(uploaded_file, depth_ranges):
    """Process uploaded Excel file with individual layer depth ranges"""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Ensure required columns exist
        required_cols = ['Depth', 'Vp', 'Vs', 'Density', 'Epsilon', 'Delta', 'Gamma']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Excel file must contain '{col}' column")
                return None
        
        # Get values for each layer based on individual depth ranges
        params = {}
        for i, (min_depth, max_depth) in enumerate(depth_ranges, 1):
            layer_df = df[(df['Depth'] >= min_depth) & (df['Depth'] <= max_depth)]
            if len(layer_df) == 0:
                st.error(f"No data found in Layer {i} depth range ({min_depth}-{max_depth})")
                return None
            
            # Take median values for the layer
            params[f'vp{i}'] = float(layer_df['Vp'].median())
            params[f'vs{i}'] = float(layer_df['Vs'].median())
            params[f'd{i}'] = float(layer_df['Density'].median())
            params[f'e{i}'] = float(layer_df['Epsilon'].median())
            params[f'g{i}'] = float(layer_df['Gamma'].median())
            params[f'dlt{i}'] = float(layer_df['Delta'].median())
        
        return params
    
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None

def plot_depth_ranges(depth_ranges, min_depth, max_depth):
    """Visualize the selected depth ranges"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create a horizontal bar for each layer
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    labels = ['Upper Layer (1)', 'Target Layer (2)', 'Lower Layer (3)']
    
    for i, ((min_d, max_d), color) in enumerate(zip(depth_ranges, colors)):
        ax.barh(i, max_d-min_d, left=min_d, height=0.6, color=color, label=labels[i])
        ax.text((min_d + max_d)/2, i, f'{min_d:.1f}-{max_d:.1f}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlim(min_depth, max_depth)
    ax.set_yticks([])
    ax.set_xlabel('Depth (m)')
    ax.set_title('Selected Depth Ranges')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.3))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

def run_modeling(params, enable_fluid_sub, selected_angle, azimuth_step, freq):
    """Run the modeling for ONLY the selected angle (with all azimuths)"""
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
        
        # Only use the selected angle
        incidence_angles = [selected_angle]
        azimuths = np.arange(0, 361, azimuth_step)
        
        # Compute reflectivity (1D array for single angle)
        reflectivity_orig = np.zeros(len(azimuths))
        reflectivity_sub = np.zeros(len(azimuths))
        
        theta_rad = np.radians(selected_angle)
        for j, az in enumerate(azimuths):
            reflectivity_orig[j] = calculate_reflectivity(
                vp_orig, vs_orig, d_orig, e_orig, g_orig, dlt_orig, theta_rad, az
            )
            reflectivity_sub[j] = calculate_reflectivity(
                vp_sub, vs_sub, d_sub, e_sub, g_sub, dlt_sub, theta_rad, az
            )
        
        # Generate synthetic seismic for single angle
        n_samples = 150
        wavelet = ricker_wavelet(freq, 0.08, 0.001)
        center_sample = n_samples//2 + len(wavelet)//2
        
        R = np.zeros((n_samples, len(azimuths)))
        R[n_samples//2, :] = reflectivity_orig[:]
        seismic_orig = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
        seismic_orig = seismic_orig[center_sample-75:center_sample+75, :]
        
        R = np.zeros((n_samples, len(azimuths)))
        R[n_samples//2, :] = reflectivity_sub[:]
        seismic_sub = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
        seismic_sub = seismic_sub[center_sample-75:center_sample+75, :]
        
        # Prepare 3D reflectivity data (for visualization)
        theta_3d = np.radians(np.linspace(0, selected_angle, 30))
        az_3d = np.radians(azimuths)
        theta_grid, az_grid = np.meshgrid(theta_3d, az_3d)
        
        # Calculate 3D reflectivity surfaces
        r_3d_orig = np.zeros_like(theta_grid)
        r_3d_sub = np.zeros_like(theta_grid)
        for i in range(theta_grid.shape[0]):
            for j in range(theta_grid.shape[1]):
                r_3d_orig[i,j] = calculate_reflectivity(
                    vp_orig, vs_orig, d_orig, e_orig, g_orig, dlt_orig, 
                    theta_grid[i,j], np.degrees(az_grid[i,j])
                )
                r_3d_sub[i,j] = calculate_reflectivity(
                    vp_sub, vs_sub, d_sub, e_sub, g_sub, dlt_sub,
                    theta_grid[i,j], np.degrees(az_grid[i,j])
                )
        
        return {
            'reflectivity_orig': reflectivity_orig,
            'reflectivity_sub': reflectivity_sub,
            'seismic_orig': seismic_orig,
            'seismic_sub': seismic_sub,
            'incidence_angles': incidence_angles,
            'azimuths': azimuths,
            'vp_orig': vp_orig,
            'vp_sub': vp_sub,
            'vs_orig': vs_orig,
            'vs_sub': vs_sub,
            'd_orig': d_orig,
            'd_sub': d_sub,
            'e_orig': e_orig,
            'e_sub': e_sub,
            'g_orig': g_orig,
            'g_sub': g_sub,
            'dlt_orig': dlt_orig,
            'dlt_sub': dlt_sub,
            'r_3d_orig': r_3d_orig,
            'r_3d_sub': r_3d_sub,
            'theta_grid': theta_grid,
            'az_grid': az_grid
        }

def display_results(results, selected_angle):
    """Display modeling results for single angle"""
    tab1, tab2, tab3 = st.tabs([
        "2D Comparisons", 
        "3D Reflectivity", 
        "Anisotropy Analysis"
    ])
    
    with tab1:
        st.header(f"AVAZ Response at {selected_angle}°")
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cartesian plot
        ax1.plot(results['azimuths'], results['reflectivity_orig'], 'b-', label='Original')
        ax1.plot(results['azimuths'], results['reflectivity_sub'], 'r--', label='Fluid-Substituted')
        ax1.set(xlabel='Azimuth (degrees)', ylabel='Reflectivity', 
               title=f'Reflectivity at {selected_angle}° Incidence')
        ax1.grid(True)
        ax1.legend()
        
        # Polar plot
        theta_rad = np.radians(results['azimuths'])
        ax2 = plt.subplot(122, polar=True)
        ax2.plot(theta_rad, results['reflectivity_orig'], 'b-', label='Original')
        ax2.plot(theta_rad, results['reflectivity_sub'], 'r--', label='Fluid-Substituted')
        ax2.set_title(f'Polar View at {selected_angle}°', pad=20)
        ax2.legend()
        
        st.pyplot(fig1)
        
        # Difference plot
        st.subheader("Difference Analysis")
        diff = results['reflectivity_sub'] - results['reflectivity_orig']
        
        fig2, ax = plt.subplots(figsize=(12, 4))
        ax.plot(results['azimuths'], diff, 'g-', label='Fluid Sub - Original')
        ax.set(xlabel='Azimuth (degrees)', ylabel='Reflectivity Difference', 
              title=f'Difference at {selected_angle}°')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig2)
        
        # Seismic comparison
        st.subheader(f"Synthetic Seismic at {selected_angle}°")
        
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original seismic
        vmax = max(np.abs(results['seismic_orig']).max(), np.abs(results['seismic_sub']).max())
        ax1.imshow(results['seismic_orig'], 
                 cmap='seismic', 
                 aspect='auto',
                 vmin=-vmax, 
                 vmax=vmax,
                 extent=[0, 360, results['seismic_orig'].shape[0], 0])
        ax1.set(title=f'Original at {selected_angle}°', xlabel='Azimuth', ylabel='Time')
        
        # Substituted seismic
        ax2.imshow(results['seismic_sub'], 
                 cmap='seismic', 
                 aspect='auto',
                 vmin=-vmax, 
                 vmax=vmax,
                 extent=[0, 360, results['seismic_sub'].shape[0], 0])
        ax2.set(title=f'Fluid-Substituted at {selected_angle}°', xlabel='Azimuth', ylabel='Time')
        
        st.pyplot(fig3)
    
    with tab2:
        st.header("3D Reflectivity Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            # Original 3D reflectivity
            fig_orig = create_3d_plot(
                np.degrees(results['az_grid']),
                np.degrees(results['theta_grid']),
                results['r_3d_orig'],
                "Original Reflectivity",
                colorscale='Jet'
            )
            st.plotly_chart(fig_orig, use_container_width=True)
        
        with col2:
            # Substituted 3D reflectivity
            fig_sub = create_3d_plot(
                np.degrees(results['az_grid']),
                np.degrees(results['theta_grid']),
                results['r_3d_sub'],
                "Fluid-Substituted Reflectivity",
                colorscale='Jet'
            )
            st.plotly_chart(fig_sub, use_container_width=True)
        
        # Difference 3D plot
        st.subheader("3D Reflectivity Difference")
        diff_3d = results['r_3d_sub'] - results['r_3d_orig']
        fig_diff = create_3d_plot(
            np.degrees(results['az_grid']),
            np.degrees(results['theta_grid']),
            diff_3d,
            "Reflectivity Difference (Fluid Sub - Original)",
            colorscale='RdBu'
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    
    with tab3:
        st.header("Anisotropy Parameter Comparison")
        
        # Show P-wave anisotropy for both models
        col1, col2 = st.columns(2)
        with col1:
            pwave_anisotropy_section(
                results['e_orig'][1],
                results['dlt_orig'][1],
                results['vp_orig'][1],
                "(Original)"
            )
        
        with col2:
            pwave_anisotropy_section(
                results['e_sub'][1],
                results['dlt_sub'][1],
                results['vp_sub'][1],
                "(Fluid-Substituted)"
            )
        
        # Parameter comparison table
        st.subheader("Parameter Changes in Target Layer")
        params_df = pd.DataFrame({
            'Parameter': ['Vp (m/s)', 'Vs (m/s)', 'Density (g/cc)', 'Epsilon', 'Delta', 'Gamma'],
            'Original': [
                results['vp_orig'][1], results['vs_orig'][1], results['d_orig'][1],
                results['e_orig'][1], results['dlt_orig'][1], results['g_orig'][1]
            ],
            'Fluid-Substituted': [
                results['vp_sub'][1], results['vs_sub'][1], results['d_sub'][1],
                results['e_sub'][1], results['dlt_sub'][1], results['g_sub'][1]
            ],
            'Change (%)': [
                100*(results['vp_sub'][1]-results['vp_orig'][1])/results['vp_orig'][1],
                100*(results['vs_sub'][1]-results['vs_orig'][1])/results['vs_orig'][1],
                100*(results['d_sub'][1]-results['d_orig'][1])/results['d_orig'][1],
                100*(results['e_sub'][1]-results['e_orig'][1])/results['e_orig'][1],
                100*(results['dlt_sub'][1]-results['dlt_orig'][1])/results['dlt_orig'][1],
                100*(results['g_sub'][1]-results['g_orig'][1])/results['g_orig'][1]
            ]
        })
        
        st.dataframe(params_df.style.format({
            'Original': '{:.2f}',
            'Fluid-Substituted': '{:.2f}',
            'Change (%)': '{:.2f}%'
        }))

def main():
    st.set_page_config(layout="wide", page_title="AVAZ Modeling with Fluid Substitution")
    st.title("AVAZ Modeling with Brown-Korringa Fluid Substitution")
    
    # Initialize session state
    if 'modeling_mode' not in st.session_state:
        st.session_state.modeling_mode = "manual"
    if 'excel_data_processed' not in st.session_state:
        st.session_state.excel_data_processed = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # Modeling mode selection
    modeling_mode = st.sidebar.radio(
        "Modeling Mode",
        ["Manual Input", "Excel Import"],
        index=0 if st.session_state.modeling_mode == "manual" else 1
    )
    
    if modeling_mode == "Manual Input":
        st.session_state.modeling_mode = "manual"
        st.session_state.excel_data_processed = False
        st.session_state.show_results = False
        
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
            selected_angle = st.slider(
                "Angle of Incidence (deg)", 
                1, 90, 30, 1,
                help="Model will be computed ONLY at this angle"
            )
            freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 45)
            azimuth_step = st.slider("Azimuth Step (deg)", 1, 30, 10)
            
            st.subheader("Fluid Substitution Parameters")
            enable_fluid_sub = st.checkbox("Enable Fluid Substitution", True)
            if enable_fluid_sub:
                params['phi'] = st.slider("Porosity (ϕ)", 0.01, 0.5, 0.2, 0.01)
                params['Km'] = st.number_input("Mineral Bulk Modulus (GPa)", 10.0, 100.0, 37.0, 1.0)
                params['Gm'] = st.number_input("Mineral Shear Modulus (GPa)", 10.0, 100.0, 44.0, 1.0)
                params['Kf'] = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 5.0, 2.2, 0.1)
                params['new_fluid_density'] = st.number_input("New Fluid Density (g/cc)", 0.1, 1.5, 1.0, 0.1)
            
            if st.button("Run Modeling"):
                st.session_state.show_results = True
                st.session_state.model_params = params
                st.session_state.enable_fluid_sub = enable_fluid_sub
                st.session_state.selected_angle = selected_angle
                st.session_state.azimuth_step = azimuth_step
                st.session_state.freq = freq
        
        # Main workspace content
        if st.session_state.show_results:
            results = run_modeling(
                st.session_state.model_params,
                st.session_state.enable_fluid_sub,
                st.session_state.selected_angle,
                st.session_state.azimuth_step,
                st.session_state.freq
            )
            display_results(
                results,
                st.session_state.selected_angle
            )
    
    else:  # Excel Import mode
        st.session_state.modeling_mode = "excel"
        st.session_state.show_results = False
        
        with st.sidebar:
            st.header("Excel Import Settings")
            
            uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    # Read Excel to get full depth range
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    min_depth = float(df['Depth'].min())
                    max_depth = float(df['Depth'].max())
                    
                    st.subheader("Layer Depth Ranges")
                    
                    # Calculate default ranges (divide into thirds)
                    range_size = (max_depth - min_depth) / 3
                    default_ranges = [
                        (min_depth, min_depth + range_size),
                        (min_depth + range_size, min_depth + 2*range_size),
                        (min_depth + 2*range_size, max_depth)
                    ]
                    
                    depth_ranges = []
                    layers = ["Upper (1)", "Target (2)", "Lower (3)"]
                    
                    for i, layer in enumerate(layers, 1):
                        st.markdown(f"**{layer}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            min_val = st.number_input(
                                f"Min Depth {i}", 
                                min_value=min_depth, 
                                max_value=max_depth,
                                value=default_ranges[i-1][0],
                                key=f"min_depth_{i}"
                            )
                        with col2:
                            max_val = st.number_input(
                                f"Max Depth {i}", 
                                min_value=min_depth, 
                                max_value=max_depth,
                                value=default_ranges[i-1][1],
                                key=f"max_depth_{i}"
                            )
                        # Ensure valid range
                        if min_val >= max_val:
                            st.error(f"Layer {i}: Min must be less than Max")
                            continue
                        depth_ranges.append((min_val, max_val))
                    
                    # Store depth ranges in session state
                    if len(depth_ranges) == 3:
                        st.session_state.depth_ranges = depth_ranges
                        st.session_state.min_depth = min_depth
                        st.session_state.max_depth = max_depth
                    
                    st.subheader("Acquisition Parameters")
                    selected_angle = st.slider(
                        "Angle of Incidence (deg)", 
                        1, 90, 30, 1,
                        help="Model will be computed ONLY at this angle",
                        key="excel_angle"
                    )
                    freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 45, key="excel_freq")
                    azimuth_step = st.slider("Azimuth Step (deg)", 1, 30, 10, key="excel_azimuth_step")
                    
                    st.subheader("Fluid Substitution Parameters")
                    enable_fluid_sub = st.checkbox("Enable Fluid Substitution", True, key="excel_fluid_sub")
                    if enable_fluid_sub:
                        phi = st.slider("Porosity (ϕ)", 0.01, 0.5, 0.2, 0.01, key="excel_phi")
                        Km = st.number_input("Mineral Bulk Modulus (GPa)", 10.0, 100.0, 37.0, 1.0, key="excel_Km")
                        Gm = st.number_input("Mineral Shear Modulus (GPa)", 10.0, 100.0, 44.0, 1.0, key="excel_Gm")
                        Kf = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 5.0, 2.2, 0.1, key="excel_Kf")
                        new_fluid_density = st.number_input("New Fluid Density (g/cc)", 0.1, 1.5, 1.0, 0.1, key="excel_fluid_density")
                    
                    if st.button("Run Modeling with Excel Data"):
                        st.session_state.excel_data_processed = True
                        st.session_state.uploaded_file = uploaded_file
                        st.session_state.enable_fluid_sub = enable_fluid_sub
                        st.session_state.selected_angle = selected_angle
                        st.session_state.azimuth_step = azimuth_step
                        st.session_state.freq = freq
                        
                        if enable_fluid_sub:
                            st.session_state.phi = phi
                            st.session_state.Km = Km
                            st.session_state.Gm = Gm
                            st.session_state.Kf = Kf
                            st.session_state.new_fluid_density = new_fluid_density
                
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
        
        # Main workspace content for Excel mode
        if uploaded_file is not None and hasattr(st.session_state, 'depth_ranges'):
            st.header("Depth Range Visualization")
            plot_depth_ranges(
                st.session_state.depth_ranges,
                st.session_state.min_depth,
                st.session_state.max_depth
            )
            
            if st.session_state.excel_data_processed:
                try:
                    # Process Excel data with individual layer ranges
                    params = process_excel_data(
                        st.session_state.uploaded_file,
                        st.session_state.depth_ranges
                    )
                    
                    if params is not None:
                        # Add fluid substitution parameters if enabled
                        if st.session_state.enable_fluid_sub:
                            params.update({
                                'phi': st.session_state.phi,
                                'Km': st.session_state.Km,
                                'Gm': st.session_state.Gm,
                                'Kf': st.session_state.Kf,
                                'new_fluid_density': st.session_state.new_fluid_density
                            })
                        
                        # Run modeling
                        results = run_modeling(
                            params,
                            st.session_state.enable_fluid_sub,
                            st.session_state.selected_angle,
                            st.session_state.azimuth_step,
                            st.session_state.freq
                        )
                        display_results(
                            results,
                            st.session_state.selected_angle
                        )
                
                except Exception as e:
                    st.error(f"Modeling error: {str(e)}")

if __name__ == "__main__":
    main()
