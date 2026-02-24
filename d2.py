import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="VTI Model Analysis - Schoenberg Linear Slip Model",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .pressure-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2980b9;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# THEORETICAL FUNCTIONS FROM THE PAPER
# ============================================

def velocity_p_approx(alpha_deg, vp0, delta_N, delta_T, g):
    """
    P-wave velocity from equation 2
    alpha: angle in degrees
    vp0: P-wave velocity at symmetry axis
    delta_N, delta_T: normal and tangential weaknesses
    g: (VS/VP)^2
    """
    alpha_rad = np.radians(alpha_deg)
    sin2 = np.sin(alpha_rad)**2
    
    # First order approximation from equation 2
    vp = vp0 * (1 - delta_N * (1 - 2*g*sin2) - delta_T * g * sin2)
    return vp

def velocity_sv_approx(alpha_deg, vs0, delta_N, delta_T, g):
    """
    SV-wave velocity from equation 2
    """
    alpha_rad = np.radians(alpha_deg)
    sin2 = np.sin(alpha_rad)**2
    cos2 = np.cos(alpha_rad)**2
    
    vsv = vs0 * (1 - delta_T * cos2 - (delta_N - delta_T) * g * sin2)
    return vsv

def velocity_sh_approx(alpha_deg, vs0, delta_T):
    """
    SH-wave velocity from equation 2
    """
    alpha_rad = np.radians(alpha_deg)
    cos_alpha = np.cos(alpha_rad)
    
    vsh = vs0 * (1 - delta_T * (1 - cos_alpha**2))
    return vsh

def attenuation_p_approx(alpha_deg, delta_N_I, delta_T_I, delta_N, delta_T, g):
    """
    P-wave attenuation (1/Q) from equation 3
    """
    alpha_rad = np.radians(alpha_deg)
    sin2 = np.sin(alpha_rad)**2
    cos2 = np.cos(alpha_rad)**2
    
    term1 = delta_N_I * (1 - 2*g*sin2)**2
    term2 = delta_T_I * g * sin2 * (1 + 2*g*cos2)
    
    return term1 + term2

def attenuation_sv_approx(alpha_deg, delta_N_I, delta_T_I, delta_N, delta_T, g):
    """
    SV-wave attenuation (1/Q) from equation 4
    """
    alpha_rad = np.radians(alpha_deg)
    sin2 = np.sin(alpha_rad)**2
    cos2 = np.cos(alpha_rad)**2
    
    term = delta_T_I * cos2 + (delta_N_I - delta_T_I) * g * sin2
    return term

def attenuation_sh_approx(alpha_deg, delta_T_I):
    """
    SH-wave attenuation (1/Q) from equation 5
    """
    alpha_rad = np.radians(alpha_deg)
    cos_alpha = np.cos(alpha_rad)
    
    return delta_T_I * cos_alpha**2

# ============================================
# PRESSURE-DEPENDENT WEAKNESS MODELS
# ============================================

def get_weakness_from_pressure(pressure_mpa, saturation_type='air'):
    """
    Calculate weakness parameters based on pressure using empirical relationships
    Based on typical fracture compliance pressure dependence
    """
    if saturation_type == 'air':
        # Air-filled fractures: higher compliance, stronger pressure dependence
        delta_N_base = 0.65
        delta_T_base = 0.65
        delta_N_I_base = 0.08
        delta_T_I_base = 0.08
    else:  # oil-saturated
        # Oil-saturated fractures: lower compliance, weaker pressure dependence
        delta_N_base = 0.30
        delta_T_base = 0.30
        delta_N_I_base = 0.05
        delta_T_I_base = 0.05
    
    # Pressure dependence (exponential decay with pressure)
    # Higher pressure -> lower weakness (fractures close)
    pressure_factor = np.exp(-0.5 * pressure_mpa)
    
    delta_N = delta_N_base * pressure_factor
    delta_T = delta_T_base * pressure_factor
    delta_N_I = delta_N_I_base * pressure_factor
    delta_T_I = delta_T_I_base * pressure_factor
    
    return delta_N, delta_T, delta_N_I, delta_T_I

# ============================================
# DATA PROCESSING FUNCTIONS
# ============================================

def load_well_data(uploaded_file):
    """
    Load well log data from uploaded CSV file
    """
    df = pd.read_csv(uploaded_file)
    
    # Standardize column names (handle different naming conventions)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'depth' in col_lower:
            column_mapping[col] = 'DEPTH'
        elif 'vp' in col_lower or 'p-wave' in col_lower or 'vp' in col_lower.lower():
            column_mapping[col] = 'VP'
        elif 'vs' in col_lower or 's-wave' in col_lower or 'vs' in col_lower.lower():
            column_mapping[col] = 'VS'
        elif 'rho' in col_lower or 'dens' in col_lower or 'density' in col_lower:
            column_mapping[col] = 'RHO'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    return df

def select_depth_range(df, depth_min, depth_max):
    """
    Select data within specified depth range
    """
    mask = (df['DEPTH'] >= depth_min) & (df['DEPTH'] <= depth_max)
    return df[mask]

def create_vti_model_from_well_data(depth, vp, vs, rho, 
                                    pressure_mpa,
                                    delta_N_air=None, delta_T_air=None,
                                    delta_N_oil=None, delta_T_oil=None,
                                    delta_N_I_air=None, delta_T_I_air=None,
                                    delta_N_I_oil=None, delta_T_I_oil=None,
                                    use_pressure_dependence=True):
    """
    Create VTI model parameters from well log data with pressure dependence
    """
    # Calculate average background properties
    vp0 = np.mean(vp)
    vs0 = np.mean(vs)
    rho0 = np.mean(rho)
    g = (vs0/vp0)**2
    
    # Get weakness parameters based on pressure or use provided values
    if use_pressure_dependence:
        delta_N_air, delta_T_air, delta_N_I_air, delta_T_I_air = get_weakness_from_pressure(pressure_mpa, 'air')
        delta_N_oil, delta_T_oil, delta_N_I_oil, delta_T_I_oil = get_weakness_from_pressure(pressure_mpa, 'oil')
    else:
        # Use provided values (will be set from UI)
        pass
    
    # Create model parameters dictionary
    model_params = {
        'pressure_mpa': pressure_mpa,
        'background': {
            'vp0': vp0,
            'vs0': vs0,
            'rho0': rho0,
            'g': g
        },
        'air_filled': {
            'delta_N': delta_N_air,
            'delta_T': delta_T_air,
            'delta_N_I': delta_N_I_air,
            'delta_T_I': delta_T_I_air,
            'color': 'blue',
            'marker': 'o',
            'label': f'Air-filled ({pressure_mpa} MPa)'
        },
        'oil_saturated': {
            'delta_N': delta_N_oil,
            'delta_T': delta_T_oil,
            'delta_N_I': delta_N_I_oil,
            'delta_T_I': delta_T_I_oil,
            'color': 'red',
            'marker': 's',
            'label': f'Oil-saturated ({pressure_mpa} MPa)'
        }
    }
    
    return model_params

def generate_angular_data_from_well(model_params, n_angles, noise_level_v, noise_level_q, add_missing_sv):
    """
    Generate synthetic angular data points based on well log background
    """
    angles = np.linspace(0, 90, n_angles)
    angles_fine = np.linspace(0, 90, 181)
    
    bg = model_params['background']
    g = bg['g']
    
    data = {
        'angles': angles,
        'angles_fine': angles_fine,
        'air_filled': {'vp': {}, 'vsv': {}, 'vsh': {}, 'qp_inv': {}, 'qsv_inv': {}, 'qsh_inv': {}},
        'oil_saturated': {'vp': {}, 'vsv': {}, 'vsh': {}, 'qp_inv': {}, 'qsv_inv': {}, 'qsh_inv': {}}
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for saturation in ['air_filled', 'oil_saturated']:
        params = model_params[saturation]
        
        # Theoretical curves
        data[saturation]['vp']['theory'] = velocity_p_approx(angles_fine, bg['vp0'], 
                                                              params['delta_N'], params['delta_T'], g)
        data[saturation]['vsv']['theory'] = velocity_sv_approx(angles_fine, bg['vs0'],
                                                                params['delta_N'], params['delta_T'], g)
        data[saturation]['vsh']['theory'] = velocity_sh_approx(angles_fine, bg['vs0'], params['delta_T'])
        
        data[saturation]['qp_inv']['theory'] = attenuation_p_approx(angles_fine,
                                                                     params['delta_N_I'], params['delta_T_I'],
                                                                     params['delta_N'], params['delta_T'], g)
        data[saturation]['qsv_inv']['theory'] = attenuation_sv_approx(angles_fine,
                                                                       params['delta_N_I'], params['delta_T_I'],
                                                                       params['delta_N'], params['delta_T'], g)
        data[saturation]['qsh_inv']['theory'] = attenuation_sh_approx(angles_fine, params['delta_T_I'])
        
        # Data points with noise
        data[saturation]['vp']['data'] = velocity_p_approx(angles, bg['vp0'],
                                                            params['delta_N'], params['delta_T'], g) * \
                                         (1 + noise_level_v * np.random.randn(len(angles)))
        data[saturation]['vsv']['data'] = velocity_sv_approx(angles, bg['vs0'],
                                                              params['delta_N'], params['delta_T'], g) * \
                                         (1 + noise_level_v * np.random.randn(len(angles)))
        data[saturation]['vsh']['data'] = velocity_sh_approx(angles, bg['vs0'], params['delta_T']) * \
                                         (1 + noise_level_v * np.random.randn(len(angles)))
        
        # Attenuation data with noise (ensure positive)
        qp_data = attenuation_p_approx(angles, params['delta_N_I'], params['delta_T_I'],
                                       params['delta_N'], params['delta_T'], g)
        qp_data = qp_data * (1 + noise_level_q * np.random.randn(len(angles)))
        data[saturation]['qp_inv']['data'] = np.maximum(qp_data, 0.001)
        
        qsv_data = attenuation_sv_approx(angles, params['delta_N_I'], params['delta_T_I'],
                                         params['delta_N'], params['delta_T'], g)
        qsv_data = qsv_data * (1 + noise_level_q * np.random.randn(len(angles)))
        data[saturation]['qsv_inv']['data'] = np.maximum(qsv_data, 0.001)
        
        qsh_data = attenuation_sh_approx(angles, params['delta_T_I'])
        qsh_data = qsh_data * (1 + noise_level_q * np.random.randn(len(angles)))
        data[saturation]['qsh_inv']['data'] = np.maximum(qsh_data, 0.001)
        
        # Add missing SV data points if requested
        if saturation == 'air_filled' and add_missing_sv:
            # Make some SV data NaN to simulate missing measurements
            missing_indices = slice(3, 8)
            data[saturation]['qsv_inv']['data'][missing_indices] = np.nan
            data[saturation]['vsv']['data'][missing_indices] = np.nan
    
    return data

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def create_figure2_plot(data, model_params, figsize=(16, 10)):
    """
    Create a plot exactly like Figure 2 from the paper
    """
    bg = model_params['background']
    angles = data['angles']
    angles_fine = data['angles_fine']
    pressure = model_params['pressure_mpa']
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25)
    
    # ===== LEFT COLUMN: VELOCITIES =====
    
    # P-wave velocity (top left)
    ax1 = plt.subplot(gs[0, 0])
    # Air-filled
    ax1.plot(angles_fine, data['air_filled']['vp']['theory'], 
             'b-', linewidth=2, label=f'Air-filled (theory)')
    ax1.scatter(angles, data['air_filled']['vp']['data'], 
                c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax1.plot(angles_fine, data['oil_saturated']['vp']['theory'], 
             'r-', linewidth=2, label=f'Oil-saturated (theory)')
    ax1.scatter(angles, data['oil_saturated']['vp']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax1.set_xlim(0, 90)
    ax1.set_xticks([0, 30, 60, 90])
    ax1.set_ylabel('P-wave velocity (m/s)')
    ax1.set_title('(a) P-wave velocity')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.text(0.02, 0.98, f'Vp₀={bg["vp0"]:.0f} m/s', transform=ax1.transAxes, 
             fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # SV-wave velocity (middle left)
    ax2 = plt.subplot(gs[1, 0])
    # Air-filled
    ax2.plot(angles_fine, data['air_filled']['vsv']['theory'], 
             'b-', linewidth=2, label='Air-filled (theory)')
    valid_air = ~np.isnan(data['air_filled']['vsv']['data'])
    if np.any(valid_air):
        ax2.scatter(angles[valid_air], data['air_filled']['vsv']['data'][valid_air], 
                    c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                    edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax2.plot(angles_fine, data['oil_saturated']['vsv']['theory'], 
             'r-', linewidth=2, label='Oil-saturated (theory)')
    ax2.scatter(angles, data['oil_saturated']['vsv']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax2.set_xlim(0, 90)
    ax2.set_xticks([0, 30, 60, 90])
    ax2.set_ylabel('SV-wave velocity (m/s)')
    ax2.set_title('(c) SV-wave velocity')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.text(0.02, 0.98, f'Vs₀={bg["vs0"]:.0f} m/s', transform=ax2.transAxes, 
             fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # SH-wave velocity (bottom left)
    ax3 = plt.subplot(gs[2, 0])
    # Air-filled
    ax3.plot(angles_fine, data['air_filled']['vsh']['theory'], 
             'b-', linewidth=2, label='Air-filled (theory)')
    ax3.scatter(angles, data['air_filled']['vsh']['data'], 
                c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax3.plot(angles_fine, data['oil_saturated']['vsh']['theory'], 
             'r-', linewidth=2, label='Oil-saturated (theory)')
    ax3.scatter(angles, data['oil_saturated']['vsh']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax3.set_xlim(0, 90)
    ax3.set_xticks([0, 30, 60, 90])
    ax3.set_xlabel('Angle from symmetry axis (degrees)')
    ax3.set_ylabel('SH-wave velocity (m/s)')
    ax3.set_title('(e) SH-wave velocity')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # ===== RIGHT COLUMN: ATTENUATIONS =====
    
    # P-wave attenuation (top right)
    ax4 = plt.subplot(gs[0, 1])
    # Air-filled
    ax4.plot(angles_fine, data['air_filled']['qp_inv']['theory'], 
             'b-', linewidth=2, label='Air-filled (theory)')
    ax4.scatter(angles, data['air_filled']['qp_inv']['data'], 
                c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax4.plot(angles_fine, data['oil_saturated']['qp_inv']['theory'], 
             'r-', linewidth=2, label='Oil-saturated (theory)')
    ax4.scatter(angles, data['oil_saturated']['qp_inv']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax4.set_xlim(0, 90)
    ax4.set_xticks([0, 30, 60, 90])
    ax4.set_ylabel('P-wave attenuation, 1/Q')
    ax4.set_title('(b) P-wave attenuation')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # SV-wave attenuation (middle right)
    ax5 = plt.subplot(gs[1, 1])
    # Air-filled
    ax5.plot(angles_fine, data['air_filled']['qsv_inv']['theory'], 
             'b-', linewidth=2, label='Air-filled (theory)')
    valid_air_qsv = ~np.isnan(data['air_filled']['qsv_inv']['data'])
    if np.any(valid_air_qsv):
        ax5.scatter(angles[valid_air_qsv], data['air_filled']['qsv_inv']['data'][valid_air_qsv], 
                    c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                    edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax5.plot(angles_fine, data['oil_saturated']['qsv_inv']['theory'], 
             'r-', linewidth=2, label='Oil-saturated (theory)')
    ax5.scatter(angles, data['oil_saturated']['qsv_inv']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax5.set_xlim(0, 90)
    ax5.set_xticks([0, 30, 60, 90])
    ax5.set_ylabel('SV-wave attenuation, 1/Q')
    ax5.set_title('(d) SV-wave attenuation')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # SH-wave attenuation (bottom right)
    ax6 = plt.subplot(gs[2, 1])
    # Air-filled
    ax6.plot(angles_fine, data['air_filled']['qsh_inv']['theory'], 
             'b-', linewidth=2, label='Air-filled (theory)')
    ax6.scatter(angles, data['air_filled']['qsh_inv']['data'], 
                c='blue', marker='o', s=60, label=f'Air-filled data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    # Oil-saturated
    ax6.plot(angles_fine, data['oil_saturated']['qsh_inv']['theory'], 
             'r-', linewidth=2, label='Oil-saturated (theory)')
    ax6.scatter(angles, data['oil_saturated']['qsh_inv']['data'], 
                c='red', marker='s', s=60, label=f'Oil-saturated data ({pressure} MPa)',
                edgecolors='black', linewidth=0.5, zorder=5)
    ax6.set_xlim(0, 90)
    ax6.set_xticks([0, 30, 60, 90])
    ax6.set_xlabel('Angle from symmetry axis (degrees)')
    ax6.set_ylabel('SH-wave attenuation, 1/Q')
    ax6.set_title('(f) SH-wave attenuation')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # Add overall title with parameters
    air = model_params['air_filled']
    oil = model_params['oil_saturated']
    fig.suptitle(f'Schoenberg Linear Slip Model - VTI Medium @ {pressure} MPa\n'
                 f'Background: Vp₀={bg["vp0"]:.0f} m/s, Vs₀={bg["vs0"]:.0f} m/s, ρ₀={bg["rho0"]:.0f} kg/m³\n'
                 f'Air-filled: ΔN={air["delta_N"]:.3f}, ΔT={air["delta_T"]:.3f}, ΔNᴵ={air["delta_N_I"]:.4f}, ΔTᴵ={air["delta_T_I"]:.4f}\n'
                 f'Oil-saturated: ΔN={oil["delta_N"]:.3f}, ΔT={oil["delta_T"]:.3f}, ΔNᴵ={oil["delta_N_I"]:.4f}, ΔTᴵ={oil["delta_T_I"]:.4f}',
                 fontsize=11, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN STREAMLIT APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 VTI Model Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Schoenberg Linear Slip Model for Fractured Media</h2>', unsafe_allow_html=True)
    
    # Initialize session state for parameters
    if 'model_params' not in st.session_state:
        st.session_state.model_params = None
    if 'current_pressure' not in st.session_state:
        st.session_state.current_pressure = 2.0
    
    # Sidebar for parameters
    with st.sidebar:
        st.markdown('<h3 style="text-align: center;">⚙️ Model Parameters</h3>', unsafe_allow_html=True)
        
        # Data input method
        data_source = st.radio(
            "Data Input Method",
            ["Upload CSV File", "Use Synthetic Data"]
        )
        
        uploaded_file = None
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        
        # Depth range selection (if file uploaded)
        df_filtered = None
        if data_source == "Upload CSV File" and uploaded_file is not None:
            try:
                df_full = load_well_data(uploaded_file)
                depth_min_full = float(df_full['DEPTH'].min())
                depth_max_full = float(df_full['DEPTH'].max())
                
                st.subheader("📏 Depth Range Selection")
                col1, col2 = st.columns(2)
                with col1:
                    depth_min = st.number_input("Min Depth", 
                                               value=depth_min_full,
                                               min_value=depth_min_full,
                                               max_value=depth_max_full,
                                               step=10.0)
                with col2:
                    depth_max = st.number_input("Max Depth", 
                                               value=depth_max_full,
                                               min_value=depth_min_full,
                                               max_value=depth_max_full,
                                               step=10.0)
                
                if depth_min < depth_max:
                    df_filtered = select_depth_range(df_full, depth_min, depth_max)
                    st.info(f"Selected {len(df_filtered)} data points")
                else:
                    st.error("Min depth must be less than max depth")
                    df_filtered = df_full
                
                # Reset file pointer
                uploaded_file.seek(0)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df_filtered = None
        
        st.markdown("---")
        
        # Pressure selection
        st.markdown('<div class="pressure-box">', unsafe_allow_html=True)
        st.subheader("💧 Pressure Conditions")
        
        pressure_mode = st.radio(
            "Pressure Input Mode",
            ["Single Pressure", "Pressure Scan"]
        )
        
        if pressure_mode == "Single Pressure":
            pressure_mpa = st.slider("Pressure (MPa)", 
                                     min_value=0.1, max_value=50.0, 
                                     value=st.session_state.current_pressure, 
                                     step=0.1,
                                     format="%.1f MPa")
            st.session_state.current_pressure = pressure_mpa
            pressures = [pressure_mpa]
        else:
            col1, col2 = st.columns(2)
            with col1:
                p_min = st.number_input("Min Pressure (MPa)", value=0.1, min_value=0.1, max_value=50.0, step=0.5)
            with col2:
                p_max = st.number_input("Max Pressure (MPa)", value=10.0, min_value=0.1, max_value=50.0, step=0.5)
            n_pressures = st.slider("Number of pressure steps", min_value=2, max_value=10, value=5)
            
            if p_min < p_max:
                pressures = np.linspace(p_min, p_max, n_pressures)
            else:
                st.error("Min pressure must be less than max pressure")
                pressures = [p_min]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Weakness parameters (manual override option)
        st.subheader("🔧 Weakness Parameters")
        
        use_pressure_dependence = st.checkbox("Use pressure-dependent model", value=True)
        
        if not use_pressure_dependence:
            st.info("Manual parameter entry enabled")
            
            st.markdown("**Air-filled Fractures**")
            col1, col2 = st.columns(2)
            with col1:
                delta_N_air = st.number_input("ΔN (air)", value=0.55, min_value=0.0, max_value=1.0, step=0.05, format="%.3f")
                delta_T_air = st.number_input("ΔT (air)", value=0.55, min_value=0.0, max_value=1.0, step=0.05, format="%.3f")
            with col2:
                delta_N_I_air = st.number_input("ΔNᴵ (air)", value=0.05, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
                delta_T_I_air = st.number_input("ΔTᴵ (air)", value=0.05, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
            
            st.markdown("**Oil-saturated Fractures**")
            col3, col4 = st.columns(2)
            with col3:
                delta_N_oil = st.number_input("ΔN (oil)", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.3f")
                delta_T_oil = st.number_input("ΔT (oil)", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.3f")
            with col4:
                delta_N_I_oil = st.number_input("ΔNᴵ (oil)", value=0.03, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
                delta_T_I_oil = st.number_input("ΔTᴵ (oil)", value=0.03, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
        else:
            # Default values (will be overridden by pressure dependence)
            delta_N_air = delta_T_air = delta_N_I_air = delta_T_I_air = None
            delta_N_oil = delta_T_oil = delta_N_I_oil = delta_T_I_oil = None
        
        st.markdown("---")
        
        # Data generation parameters
        st.subheader("📊 Data Generation")
        n_angles = st.slider("Number of angles", min_value=5, max_value=30, value=13)
        noise_level_v = st.slider("Velocity noise level", min_value=0.0, max_value=0.05, value=0.01, step=0.005, format="%.3f")
        noise_level_q = st.slider("Attenuation noise level", min_value=0.0, max_value=0.1, value=0.03, step=0.005, format="%.3f")
        add_missing_sv = st.checkbox("Add missing SV data points", value=True)
        
        st.markdown("---")
        
        # Plot parameters
        st.subheader("🎨 Plot Settings")
        fig_width = st.slider("Figure width", min_value=12, max_value=20, value=16)
        fig_height = st.slider("Figure height", min_value=8, max_value=14, value=10)
        
        # Generate button
        generate_btn = st.button("🚀 Generate Plots", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📥 Download Options")
        st.info("Plots can be downloaded after generation")
    
    # Main content area
    if generate_btn:
        with st.spinner("Generating VTI model analysis..."):
            try:
                # Load or create data for each pressure
                all_figs = []
                
                for i, pressure in enumerate(pressures):
                    if data_source == "Upload CSV File" and df_filtered is not None and len(df_filtered) > 0:
                        depth = df_filtered['DEPTH'].values
                        vp = df_filtered['VP'].values
                        vs = df_filtered['VS'].values
                        rho = df_filtered['RHO'].values
                        
                        # Create model from well data with current pressure
                        model_params = create_vti_model_from_well_data(
                            depth, vp, vs, rho,
                            pressure,
                            delta_N_air, delta_T_air,
                            delta_N_oil, delta_T_oil,
                            delta_N_I_air, delta_T_I_air,
                            delta_N_I_oil, delta_T_I_oil,
                            use_pressure_dependence
                        )
                        
                    elif data_source == "Use Synthetic Data":
                        # Use synthetic data with user-defined background
                        if 'vp0_synth' not in locals():
                            # Default synthetic values if not set
                            vp0_synth = 3000.0
                            vs0_synth = 1500.0
                            rho0_synth = 2200.0
                        
                        depth = np.linspace(0, 1000, 100)
                        vp = vp0_synth * np.ones_like(depth)
                        vs = vs0_synth * np.ones_like(depth)
                        rho = rho0_synth * np.ones_like(depth)
                        
                        model_params = create_vti_model_from_well_data(
                            depth, vp, vs, rho,
                            pressure,
                            delta_N_air, delta_T_air,
                            delta_N_oil, delta_T_oil,
                            delta_N_I_air, delta_T_I_air,
                            delta_N_I_oil, delta_T_I_oil,
                            use_pressure_dependence
                        )
                    else:
                        st.warning("Please upload a valid CSV file first")
                        st.stop()
                    
                    # Generate angular data
                    data = generate_angular_data_from_well(
                        model_params, n_angles, noise_level_v, noise_level_q, add_missing_sv
                    )
                    
                    # Create plot
                    fig = create_figure2_plot(data, model_params, figsize=(fig_width, fig_height))
                    all_figs.append((fig, pressure))
                
                # Display plots
                if pressure_mode == "Pressure Scan":
                    st.markdown(f"### 📈 Pressure Scan Results ({len(pressures)} pressures)")
                    
                    # Create tabs for different pressures
                    tabs = st.tabs([f"{p:.1f} MPa" for p in pressures])
                    
                    for idx, (fig, pressure) in enumerate(all_figs):
                        with tabs[idx]:
                            st.pyplot(fig)
                            
                            # Download buttons for each pressure
                            col1, col2 = st.columns(2)
                            
                            # PNG download
                            buf_png = BytesIO()
                            fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                            buf_png.seek(0)
                            
                            with col1:
                                st.download_button(
                                    label=f"📥 Download PNG ({pressure:.1f} MPa)",
                                    data=buf_png,
                                    file_name=f"vti_model_{pressure:.1f}MPa.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            
                            # PDF download
                            buf_pdf = BytesIO()
                            fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                            buf_pdf.seek(0)
                            
                            with col2:
                                st.download_button(
                                    label=f"📥 Download PDF ({pressure:.1f} MPa)",
                                    data=buf_pdf,
                                    file_name=f"vti_model_{pressure:.1f}MPa.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                else:
                    # Single pressure display
                    fig, pressure = all_figs[0]
                    st.pyplot(fig)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    
                    # PNG download
                    buf_png = BytesIO()
                    fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                    buf_png.seek(0)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download as PNG",
                            data=buf_png,
                            file_name=f"vti_model_{pressure:.1f}MPa.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # PDF download
                    buf_pdf = BytesIO()
                    fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                    buf_pdf.seek(0)
                    
                    with col2:
                        st.download_button(
                            label="📥 Download as PDF",
                            data=buf_pdf,
                            file_name=f"vti_model_{pressure:.1f}MPa.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # Parameters CSV download
                    params_df = pd.DataFrame([
                        {'Parameter': 'Pressure (MPa)', 'Value': pressure},
                        {'Parameter': 'Background Vp0 (m/s)', 'Value': model_params['background']['vp0']},
                        {'Parameter': 'Background Vs0 (m/s)', 'Value': model_params['background']['vs0']},
                        {'Parameter': 'Background Density (kg/m³)', 'Value': model_params['background']['rho0']},
                        {'Parameter': 'g = (Vs/Vp)²', 'Value': model_params['background']['g']},
                        {'Parameter': 'Air-filled ΔN', 'Value': model_params['air_filled']['delta_N']},
                        {'Parameter': 'Air-filled ΔT', 'Value': model_params['air_filled']['delta_T']},
                        {'Parameter': 'Air-filled ΔNᴵ', 'Value': model_params['air_filled']['delta_N_I']},
                        {'Parameter': 'Air-filled ΔTᴵ', 'Value': model_params['air_filled']['delta_T_I']},
                        {'Parameter': 'Oil-saturated ΔN', 'Value': model_params['oil_saturated']['delta_N']},
                        {'Parameter': 'Oil-saturated ΔT', 'Value': model_params['oil_saturated']['delta_T']},
                        {'Parameter': 'Oil-saturated ΔNᴵ', 'Value': model_params['oil_saturated']['delta_N_I']},
                        {'Parameter': 'Oil-saturated ΔTᴵ', 'Value': model_params['oil_saturated']['delta_T_I']}
                    ])
                    
                    csv = params_df.to_csv(index=False)
                    
                    with col3:
                        st.download_button(
                            label="📥 Download Parameters (CSV)",
                            data=csv,
                            file_name=f"vti_model_parameters_{pressure:.1f}MPa.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Display model parameters in a table
                    st.markdown("### 📊 Model Parameters Summary")
                    st.dataframe(params_df, use_container_width=True)
                    
                    # Show pressure-dependent weakness explanation
                    if use_pressure_dependence:
                        st.markdown("### 📈 Pressure-Dependent Weakness Analysis")
                        
                        # Create pressure dependence plot
                        pressures_plot = np.linspace(0.1, 20, 50)
                        delta_N_air_plot = []
                        delta_T_air_plot = []
                        delta_N_oil_plot = []
                        delta_T_oil_plot = []
                        
                        for p in pressures_plot:
                            dN_a, dT_a, _, _ = get_weakness_from_pressure(p, 'air')
                            dN_o, dT_o, _, _ = get_weakness_from_pressure(p, 'oil')
                            delta_N_air_plot.append(dN_a)
                            delta_T_air_plot.append(dT_a)
                            delta_N_oil_plot.append(dN_o)
                            delta_T_oil_plot.append(dT_o)
                        
                        fig_pressure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Delta N plot
                        ax1.plot(pressures_plot, delta_N_air_plot, 'b-', linewidth=2, label='Air-filled ΔN')
                        ax1.plot(pressures_plot, delta_N_oil_plot, 'r-', linewidth=2, label='Oil-saturated ΔN')
                        ax1.axvline(x=pressure, color='k', linestyle='--', alpha=0.5, label=f'Current: {pressure} MPa')
                        ax1.set_xlabel('Pressure (MPa)')
                        ax1.set_ylabel('ΔN')
                        ax1.set_title('Normal Weakness vs Pressure')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Delta T plot
                        ax2.plot(pressures_plot, delta_T_air_plot, 'b-', linewidth=2, label='Air-filled ΔT')
                        ax2.plot(pressures_plot, delta_T_oil_plot, 'r-', linewidth=2, label='Oil-saturated ΔT')
                        ax2.axvline(x=pressure, color='k', linestyle='--', alpha=0.5, label=f'Current: {pressure} MPa')
                        ax2.set_xlabel('Pressure (MPa)')
                        ax2.set_ylabel('ΔT')
                        ax2.set_title('Tangential Weakness vs Pressure')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_pressure)
                        
                        # Close the figure to free memory
                        plt.close(fig_pressure)
                    
                    # Close the figure to free memory
                    plt.close(fig)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    else:
        # Initial state - show instructions
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👈 Welcome to the VTI Model Analysis App!
        
        This application implements the **Schoenberg Linear Slip Model** for fractured VTI media
        with pressure-dependent fracture compliance.
        
        **New Features:**
        - 📏 **Depth Range Selection**: Analyze specific depth intervals
        - 💧 **Pressure-Dependent Models**: Weakness parameters vary with pressure
        - 🔬 **Pressure Scan**: Compare multiple pressures simultaneously
        - 📊 **Pressure Dependence Plots**: Visualize how weaknesses change with pressure
        
        **To get started:**
        1. Upload your well log data (CSV with Depth, VP, VS, RHO columns) OR use synthetic data
        2. Select your desired depth range (for uploaded data)
        3. Choose pressure conditions (single pressure or scan)
        4. Adjust parameters in the sidebar
        5. Click "Generate Plots"
        
        **Pressure-Dependent Model:**
        - Higher pressure → Lower weakness (fractures close)
        - Exponential decay model: ΔN, ΔT ∝ exp(-0.5 × P)
        - Air-filled fractures have higher initial compliance
        - Oil-saturated fractures show weaker pressure dependence
        
        **Required CSV format:**
        - Columns: Depth, VP, VS, RHO (case-insensitive)
        - Any depth units (m or ft)
        - Velocity in m/s
        - Density in kg/m³
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sample data preview if file is uploaded
        if data_source == "Upload CSV File" and uploaded_file is not None:
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.markdown("### 📄 Uploaded Data Preview")
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                # Show depth statistics
                if 'DEPTH' in df_preview.columns or any('depth' in col.lower() for col in df_preview.columns):
                    # Find depth column
                    depth_col = next(col for col in df_preview.columns if 'depth' in col.lower())
                    st.markdown(f"**Depth range:** {df_preview[depth_col].min():.1f} - {df_preview[depth_col].max():.1f}")
                
                # Reset file pointer
                uploaded_file.seek(0)
            except:
                pass

# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
