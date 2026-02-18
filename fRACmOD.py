import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import inv
import os
from io import StringIO
from typing import List, Dict, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Fracture Model Analysis - Thomsen Parameters",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class FractureModelThomsen:
    """
    Calculate Thomsen parameters for various fracture models including:
    - Single fracture set (HTI)
    - Linear slip model (Schoenberg)
    - Two orthogonal fracture sets (Orthorhombic)
    - Two non-orthogonal fracture sets (Monoclinic)
    """
    
    def __init__(self, vp: float, vs: float, rho: float):
        """
        Initialize with background isotropic medium properties
        
        Args:
            vp: P-wave velocity in background medium (m/s)
            vs: S-wave velocity in background medium (m/s)
            rho: Density in background medium (kg/m³)
        """
        self.vp = vp
        self.vs = vs
        self.rho = rho
        
        # Calculate background moduli for isotropic medium
        self.mu = rho * vs**2  # Shear modulus (Pa)
        self.lamb = rho * vp**2 - 2 * self.mu  # Lamé parameter (Pa)
        self.M = self.lamb + 2 * self.mu  # P-wave modulus (Pa)
        
        # Thomsen parameter g = (Vs/Vp)²
        self.gamma_bg = (vs / vp)**2 if vp > 0 else 0
        
        # Background isotropic stiffness tensor (Voigt notation)
        self.C_iso = np.array([
            [self.M, self.lamb, self.lamb, 0, 0, 0],
            [self.lamb, self.M, self.lamb, 0, 0, 0],
            [self.lamb, self.lamb, self.M, 0, 0, 0],
            [0, 0, 0, self.mu, 0, 0],
            [0, 0, 0, 0, self.mu, 0],
            [0, 0, 0, 0, 0, self.mu]
        ])
    
    def _bond_strain_transformation(self, n: np.ndarray) -> np.ndarray:
        """Bond transformation matrix for strain (6x6)"""
        nx, ny, nz = n
        
        M = np.zeros((6, 6))
        
        # Fill the transformation matrix
        M[0, 0] = nx**2
        M[0, 1] = ny**2
        M[0, 2] = nz**2
        M[0, 3] = 2*ny*nz
        M[0, 4] = 2*nx*nz
        M[0, 5] = 2*nx*ny
        
        M[1, 0] = nx**2
        M[1, 1] = ny**2
        M[1, 2] = nz**2
        M[1, 3] = 2*ny*nz
        M[1, 4] = 2*nx*nz
        M[1, 5] = 2*nx*ny
        
        M[2, 0] = nx**2
        M[2, 1] = ny**2
        M[2, 2] = nz**2
        M[2, 3] = 2*ny*nz
        M[2, 4] = 2*nx*nz
        M[2, 5] = 2*nx*ny
        
        M[3, 0] = ny*nz
        M[3, 1] = nx*nz
        M[3, 2] = nx*ny
        M[3, 3] = nx*ny
        M[3, 4] = ny*nz
        M[3, 5] = nx*nz
        
        M[4, 0] = nx*nz
        M[4, 1] = ny*nz
        M[4, 2] = nx*ny
        M[4, 3] = nx*ny
        M[4, 4] = nx*nz
        M[4, 5] = ny*nz
        
        M[5, 0] = nx*ny
        M[5, 1] = nx*nz
        M[5, 2] = ny*nz
        M[5, 3] = ny*nz
        M[5, 4] = nx*ny
        M[5, 5] = nx*nz
        
        return M
    
    def linear_slip_model(self, fracture_sets: List[Dict]) -> Dict:
        """
        Schoenberg's Linear Slip model for multiple fracture sets
        """
        # Background compliance
        S_bg = inv(self.C_iso)
        
        # Initialize total fracture compliance
        S_frac_total = np.zeros((6, 6))
        
        for fracture in fracture_sets:
            ZN = fracture['normal_compliance']
            ZT = fracture['shear_compliance']
            azimuth = np.radians(fracture['azimuth'])
            dip = np.radians(fracture['dip'])
            
            # Fracture normal vector
            nx = np.sin(dip) * np.sin(azimuth)
            ny = np.sin(dip) * np.cos(azimuth)
            nz = np.cos(dip)
            n = np.array([nx, ny, nz])
            
            # Construct fracture compliance in local coordinates
            S_frac_local = np.zeros((6, 6))
            S_frac_local[2, 2] = ZN  # Normal compliance
            S_frac_local[3, 3] = ZT  # Shear compliance
            S_frac_local[4, 4] = ZT  # Shear compliance
            
            # Transform to global coordinates
            M = self._bond_strain_transformation(n)
            S_frac_global = M.T @ S_frac_local @ M
            S_frac_total += S_frac_global
        
        # Total compliance = background + sum of fracture compliances
        S_total = S_bg + S_frac_total
        
        # Effective stiffness
        C_eff = inv(S_total)
        
        # Extract Thomsen parameters
        return self._extract_thomsen_parameters(C_eff, fracture_sets)
    
    def _extract_thomsen_parameters(self, C: np.ndarray, fracture_sets: List[Dict] = None) -> Dict:
        """Extract Thomsen parameters from stiffness tensor"""
        # Get all stiffness components
        C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
        C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
        C12, C13, C23 = C[0, 1], C[0, 2], C[1, 2]
        
        # Determine symmetry
        symmetry = 'HTI'
        if fracture_sets and len(fracture_sets) > 1:
            if len(fracture_sets) == 2:
                azi1, azi2 = fracture_sets[0]['azimuth'], fracture_sets[1]['azimuth']
                dip1, dip2 = fracture_sets[0]['dip'], fracture_sets[1]['dip']
                
                if abs(dip1 - 90) < 1e-6 and abs(dip2 - 90) < 1e-6:
                    angle_diff = abs(azi1 - azi2) % 180
                    symmetry = 'ORTHORHOMBIC' if abs(angle_diff - 90) < 1e-6 else 'MONOCLINIC'
        
        # Calculate parameters based on symmetry
        if symmetry == 'HTI':
            if abs(C11 - C33) > abs(C22 - C33):
                epsilon = (C11 - C33) / (2 * C33) if C33 != 0 else 0
                gamma = (C66 - C55) / (2 * C55) if C55 != 0 else 0
                delta = ((C13 + C55)**2 - (C33 - C55)**2) / (2 * C33 * (C33 - C55)) if abs(C33 - C55) > 1e-10 else 0
            else:
                epsilon = (C22 - C33) / (2 * C33) if C33 != 0 else 0
                gamma = (C66 - C44) / (2 * C44) if C44 != 0 else 0
                delta = ((C23 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44)) if abs(C33 - C44) > 1e-10 else 0
            
            return {'symmetry': 'HTI', 'epsilon': epsilon, 'gamma': gamma, 'delta': delta}
            
        elif symmetry == 'ORTHORHOMBIC':
            epsilon_1 = (C11 - C33) / (2 * C33) if C33 != 0 else 0
            epsilon_2 = (C22 - C33) / (2 * C33) if C33 != 0 else 0
            gamma_1 = (C66 - C55) / (2 * C55) if C55 != 0 else 0
            gamma_2 = (C66 - C44) / (2 * C44) if C44 != 0 else 0
            delta_1 = ((C13 + C55)**2 - (C33 - C55)**2) / (2 * C33 * (C33 - C55)) if abs(C33 - C55) > 1e-10 else 0
            delta_2 = ((C23 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44)) if abs(C33 - C44) > 1e-10 else 0
            delta_3 = ((C12 + C66)**2 - (C11 - C66)**2) / (2 * C11 * (C11 - C66)) if abs(C11 - C66) > 1e-10 else 0
            
            return {'symmetry': 'ORTHORHOMBIC', 
                    'epsilon_1': epsilon_1, 'epsilon_2': epsilon_2,
                    'gamma_1': gamma_1, 'gamma_2': gamma_2, 
                    'delta_1': delta_1, 'delta_2': delta_2, 'delta_3': delta_3}
        
        else:  # MONOCLINIC
            epsilon_x = (C11 - C33) / (2 * C33) if C33 != 0 else 0
            epsilon_y = (C22 - C33) / (2 * C33) if C33 != 0 else 0
            gamma_x = (C66 - C55) / (2 * C55) if C55 != 0 else 0
            gamma_y = (C66 - C44) / (2 * C44) if C44 != 0 else 0
            zeta_1 = C[0, 4] / C33 if abs(C33) > 1e-10 else 0
            zeta_2 = C[1, 3] / C44 if abs(C44) > 1e-10 else 0
            
            return {'symmetry': 'MONOCLINIC', 
                    'epsilon_x': epsilon_x, 'epsilon_y': epsilon_y,
                    'gamma_x': gamma_x, 'gamma_y': gamma_y, 
                    'zeta_1': zeta_1, 'zeta_2': zeta_2}
    
    def hudson_model(self, crack_density: float, aspect_ratio: float = 0.01, 
                    fluid_content: str = 'dry') -> Dict:
        """Hudson's model for penny-shaped cracks"""
        if self.vp <= 0 or self.vs <= 0 or self.rho <= 0:
            return {'symmetry': 'HTI', 'epsilon': 0, 'gamma': 0, 'delta': 0}
        
        # Hudson's crack parameters
        denom1 = 3 * (3 * self.lamb + 4 * self.mu)
        U11 = (16 * (self.lamb + 2 * self.mu)) / denom1 if denom1 != 0 else 0
        denom2 = 3 * (self.lamb + self.mu)
        U33 = (4 * (self.lamb + 2 * self.mu)) / denom2 if denom2 != 0 else 0
        
        # Fluid influence
        if fluid_content == 'dry':
            D = 1.0
            U11_fluid, U33_fluid = U11, U33
        elif fluid_content == 'fluid-saturated':
            denom_fluid = 1 + aspect_ratio * self.lamb / (np.pi * self.mu * self.gamma_bg) if self.mu * self.gamma_bg != 0 else 1
            D = 1 / denom_fluid
            U11_fluid = U11 * D
            U33_fluid = U33 * (1 - D) / (1 - self.gamma_bg) if (1 - self.gamma_bg) != 0 else 0
        else:
            denom_fluid = 1 + aspect_ratio * self.lamb / (np.pi * self.mu) if self.mu != 0 else 1
            D = 1 / denom_fluid
            U11_fluid, U33_fluid = U11 * D, U33
        
        # Isotropic background
        C11_iso = self.lamb + 2 * self.mu
        C33_iso = self.lamb + 2 * self.mu
        C44_iso = self.mu
        C66_iso = self.mu
        C13_iso = self.lamb
        
        # Hudson's corrections
        e = crack_density
        if self.mu != 0:
            C11 = C11_iso - (e / self.mu) * (U33_fluid * self.lamb**2 + 2 * U11_fluid * self.lamb * self.mu + 
                                              4 * U11_fluid * self.mu**2 * (1 - self.gamma_bg))
            C33 = C33_iso - (e / self.mu) * U33_fluid * self.lamb**2
            C13 = C13_iso - (e / self.mu) * U33_fluid * self.lamb * (self.lamb + 2 * self.mu)
            C12 = C13_iso - (e / self.mu) * U33_fluid * self.lamb**2
        else:
            C11, C33, C13, C12 = C11_iso, C33_iso, C13_iso, C13_iso
        
        C44 = C44_iso - e * U11_fluid * self.mu
        C66 = C66_iso - e * U33_fluid * self.mu
        
        # Thomsen parameters
        epsilon = (C11 - C33) / (2 * C33) if C33 != 0 else 0
        gamma = (C66 - C44) / (2 * C44) if C44 != 0 else 0
        delta = ((C13 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44)) if abs(C33 - C44) > 1e-10 else 0
        
        return {'symmetry': 'HTI', 'epsilon': epsilon, 'gamma': gamma, 'delta': delta}
    
    def multiple_fracture_sets_orthorhombic(self, crack_density_set1: float, 
                                           crack_density_set2: float,
                                           azimuth_set2: float = 90.0) -> Dict:
        """
        Two orthogonal vertical fracture sets - Orthorhombic symmetry
        Using Bakulin et al. (2000) formulation
        """
        e1 = crack_density_set1
        e2 = crack_density_set2
        g = self.gamma_bg
        
        # Approximate Thomsen parameters for orthorhombic media
        epsilon1 = 2 * e1 * (g - g**2) + 2 * e2 * (1 - 2*g + g**2)
        epsilon2 = 2 * e2 * (g - g**2) + 2 * e1 * (1 - 2*g + g**2)
        
        gamma1 = e1 * g
        gamma2 = e2 * g
        
        delta1 = 2 * e1 * g * (1 - 2*g) + 2 * e2 * g * (1 - g)
        delta2 = 2 * e2 * g * (1 - 2*g) + 2 * e1 * g * (1 - g)
        delta3 = 2 * (e1 + e2) * g * (1 - g)
        
        return {
            'symmetry': 'ORTHORHOMBIC',
            'epsilon_1': epsilon1,
            'epsilon_2': epsilon2,
            'gamma_1': gamma1,
            'gamma_2': gamma2,
            'delta_1': delta1,
            'delta_2': delta2,
            'delta_3': delta3
        }
    
    def multiple_fracture_sets_monoclinic(self, crack_density_set1: float,
                                         crack_density_set2: float,
                                         azimuth_set2: float = 45.0) -> Dict:
        """
        Two non-orthogonal vertical fracture sets - Monoclinic symmetry
        Using Shuai et al. (2020) formulation
        """
        e1 = crack_density_set1
        e2 = crack_density_set2
        theta = np.radians(azimuth_set2)  # angle between fracture sets
        g = self.gamma_bg
        
        # Thomsen-style parameters for monoclinic media
        epsilon_x = 2 * (e1 + e2 * np.cos(theta)**4) * (g - g**2)
        epsilon_y = 2 * (e2 + e1 * np.cos(theta)**4) * (g - g**2)
        epsilon_z = 2 * (e1 + e2) * np.sin(theta)**2 * np.cos(theta)**2 * (g - g**2)
        
        gamma_x = g * (e1 + e2 * np.cos(theta)**2)
        gamma_y = g * (e2 + e1 * np.cos(theta)**2)
        gamma_z = g * (e1 + e2) * np.sin(theta)**2
        
        # Coupling parameters
        zeta_1 = 2 * (e1 - e2) * np.sin(theta) * np.cos(theta) * (g - g**2)
        zeta_2 = g * (e1 - e2) * np.sin(theta) * np.cos(theta)
        
        return {
            'symmetry': 'MONOCLINIC',
            'epsilon_x': epsilon_x,
            'epsilon_y': epsilon_y,
            'epsilon_z': epsilon_z,
            'gamma_x': gamma_x,
            'gamma_y': gamma_y,
            'gamma_z': gamma_z,
            'zeta_1': zeta_1,
            'zeta_2': zeta_2
        }


def load_well_data(uploaded_file) -> pd.DataFrame:
    """
    Load well log data from uploaded CSV file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        DataFrame with well log data
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.upper()
        
        required_cols = ['DEPTH', 'VP', 'VS', 'RHO']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Please ensure your CSV has columns named: DEPTH, VP, VS, RHO")
            return None
        
        # Remove rows with invalid data
        df = df[(df['VP'] > 0) & (df['VS'] > 0) & (df['RHO'] > 0)]
        df = df.dropna()
        
        if len(df) == 0:
            st.error("No valid data after filtering")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def generate_example_data():
    """Generate synthetic well log data for demonstration"""
    np.random.seed(42)  # For reproducible results
    depth = np.arange(1000, 2000, 2)
    vp = 3000 + 200 * np.sin(depth/200) + 50 * np.random.randn(len(depth))
    vs = vp / 1.8 + 20 * np.random.randn(len(depth))
    rho = 2200 + 100 * np.sin(depth/150) + 20 * np.random.randn(len(depth))
    
    df = pd.DataFrame({
        'DEPTH': depth,
        'VP': vp,
        'VS': vs,
        'RHO': rho
    })
    return df


def analyze_well_with_fracture_models(df: pd.DataFrame, 
                                      crack_density: float = 0.1,
                                      aspect_ratio: float = 0.01,
                                      fracture_azimuth1: float = 0.0,
                                      fracture_azimuth2: float = 90.0,
                                      fracture_azimuth_mono: float = 45.0,
                                      sample_every: int = 1) -> pd.DataFrame:
    """
    Apply fracture models to entire well log
    """
    # Sample data if needed
    if sample_every > 1:
        df_sampled = df.iloc[::sample_every].copy()
    else:
        df_sampled = df.copy()
    
    n_samples = len(df_sampled)
    
    # Initialize result arrays
    results = {
        'DEPTH': df_sampled['DEPTH'].values,
        'VP': df_sampled['VP'].values,
        'VS': df_sampled['VS'].values,
        'RHO': df_sampled['RHO'].values,
        
        # Hudson model
        'HUDSON_EPS': np.zeros(n_samples),
        'HUDSON_GAM': np.zeros(n_samples),
        'HUDSON_DEL': np.zeros(n_samples),
        
        # Linear slip (single set)
        'LS_EPS': np.zeros(n_samples),
        'LS_GAM': np.zeros(n_samples),
        'LS_DEL': np.zeros(n_samples),
        
        # Orthorhombic
        'ORTHO_EPS1': np.zeros(n_samples),
        'ORTHO_EPS2': np.zeros(n_samples),
        'ORTHO_GAM1': np.zeros(n_samples),
        'ORTHO_GAM2': np.zeros(n_samples),
        'ORTHO_DEL1': np.zeros(n_samples),
        'ORTHO_DEL2': np.zeros(n_samples),
        'ORTHO_DEL3': np.zeros(n_samples),
        
        # Monoclinic
        'MONO_EPSX': np.zeros(n_samples),
        'MONO_EPSY': np.zeros(n_samples),
        'MONO_GAMX': np.zeros(n_samples),
        'MONO_GAMY': np.zeros(n_samples),
        'MONO_ZETA1': np.zeros(n_samples),
        'MONO_ZETA2': np.zeros(n_samples)
    }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each sample
    for i in range(n_samples):
        # Update progress
        if i % max(1, n_samples // 100) == 0:
            progress_bar.progress(i / n_samples)
            status_text.text(f"Processing sample {i}/{n_samples}")
        
        vp = df_sampled['VP'].iloc[i]
        vs = df_sampled['VS'].iloc[i]
        rho = df_sampled['RHO'].iloc[i]
        
        # Skip invalid data
        if vp <= 0 or vs <= 0 or rho <= 0:
            continue
        
        # Initialize model for this depth
        model = FractureModelThomsen(vp, vs, rho)
        
        # Hudson model
        hudson = model.hudson_model(crack_density, aspect_ratio, 'dry')
        results['HUDSON_EPS'][i] = hudson['epsilon']
        results['HUDSON_GAM'][i] = hudson['gamma']
        results['HUDSON_DEL'][i] = hudson['delta']
        
        # Linear slip model (single set)
        ZN = crack_density * (1 - model.gamma_bg) / (model.mu * model.gamma_bg) if model.mu * model.gamma_bg != 0 else 0
        ZT = crack_density / model.mu if model.mu != 0 else 0
        
        if ZN > 0 and ZT > 0:
            fracture_sets = [{
                'normal_compliance': ZN,
                'shear_compliance': ZT,
                'azimuth': fracture_azimuth1,
                'dip': 90.0
            }]
            ls = model.linear_slip_model(fracture_sets)
            if ls['symmetry'] == 'HTI':
                results['LS_EPS'][i] = ls['epsilon']
                results['LS_GAM'][i] = ls['gamma']
                results['LS_DEL'][i] = ls['delta']
        
        # Orthorhombic (two orthogonal sets)
        fracture_sets_ortho = [
            {'normal_compliance': ZN, 'shear_compliance': ZT, 
             'azimuth': fracture_azimuth1, 'dip': 90.0},
            {'normal_compliance': ZN/2, 'shear_compliance': ZT/2, 
             'azimuth': fracture_azimuth2, 'dip': 90.0}
        ]
        
        ortho = model.linear_slip_model(fracture_sets_ortho)
        if ortho['symmetry'] == 'ORTHORHOMBIC':
            results['ORTHO_EPS1'][i] = ortho['epsilon_1']
            results['ORTHO_EPS2'][i] = ortho['epsilon_2']
            results['ORTHO_GAM1'][i] = ortho['gamma_1']
            results['ORTHO_GAM2'][i] = ortho['gamma_2']
            results['ORTHO_DEL1'][i] = ortho['delta_1']
            results['ORTHO_DEL2'][i] = ortho['delta_2']
            results['ORTHO_DEL3'][i] = ortho['delta_3']
        
        # Monoclinic (two non-orthogonal sets)
        fracture_sets_mono = [
            {'normal_compliance': ZN, 'shear_compliance': ZT, 
             'azimuth': fracture_azimuth1, 'dip': 90.0},
            {'normal_compliance': ZN/2, 'shear_compliance': ZT/2, 
             'azimuth': fracture_azimuth_mono, 'dip': 90.0}
        ]
        
        mono = model.linear_slip_model(fracture_sets_mono)
        if mono['symmetry'] == 'MONOCLINIC':
            results['MONO_EPSX'][i] = mono['epsilon_x']
            results['MONO_EPSY'][i] = mono['epsilon_y']
            results['MONO_GAMX'][i] = mono['gamma_x']
            results['MONO_GAMY'][i] = mono['gamma_y']
            results['MONO_ZETA1'][i] = mono['zeta_1']
            results['MONO_ZETA2'][i] = mono['zeta_2']
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return pd.DataFrame(results)


def plot_well_results_plotly(results_df: pd.DataFrame):
    """
    Plot well log results with Thomsen parameters using Plotly
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=('Original Logs', 'Single Set (ε)', 'Orthorhombic', 'Monoclinic', 'Additional Parameters'),
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    depth = results_df['DEPTH']
    
    # Plot 1: Original logs
    fig.add_trace(
        go.Scatter(x=results_df['VP'], y=depth, mode='lines', name='VP', 
                  line=dict(color='blue', width=1), legendgroup='group1'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results_df['VS'], y=depth, mode='lines', name='VS', 
                  line=dict(color='red', width=1), legendgroup='group1'),
        row=1, col=1
    )
    
    # Plot 2: Hudson vs Linear Slip (ε)
    fig.add_trace(
        go.Scatter(x=results_df['HUDSON_EPS'], y=depth, mode='lines', name='Hudson ε',
                  line=dict(color='blue', width=1.5), legendgroup='group2'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results_df['LS_EPS'], y=depth, mode='lines', name='Linear Slip ε',
                  line=dict(color='red', width=1.5, dash='dash'), legendgroup='group2'),
        row=1, col=2
    )
    
    # Plot 3: Orthorhombic parameters
    fig.add_trace(
        go.Scatter(x=results_df['ORTHO_EPS1'], y=depth, mode='lines', name='ε₁',
                  line=dict(color='green', width=1.5), legendgroup='group3'),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=results_df['ORTHO_EPS2'], y=depth, mode='lines', name='ε₂',
                  line=dict(color='green', width=1.5, dash='dash'), legendgroup='group3'),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=results_df['ORTHO_GAM1'], y=depth, mode='lines', name='γ₁',
                  line=dict(color='magenta', width=1.5), legendgroup='group3'),
        row=1, col=3
    )
    
    # Plot 4: Monoclinic parameters
    fig.add_trace(
        go.Scatter(x=results_df['MONO_EPSX'], y=depth, mode='lines', name='εₓ',
                  line=dict(color='cyan', width=1.5), legendgroup='group4'),
        row=1, col=4
    )
    fig.add_trace(
        go.Scatter(x=results_df['MONO_EPSY'], y=depth, mode='lines', name='εᵧ',
                  line=dict(color='cyan', width=1.5, dash='dash'), legendgroup='group4'),
        row=1, col=4
    )
    fig.add_trace(
        go.Scatter(x=results_df['MONO_GAMX'], y=depth, mode='lines', name='γₓ',
                  line=dict(color='orange', width=1.5), legendgroup='group4'),
        row=1, col=4
    )
    
    # Plot 5: Delta and coupling parameters
    fig.add_trace(
        go.Scatter(x=results_df['HUDSON_DEL'], y=depth, mode='lines', name='Hudson δ',
                  line=dict(color='blue', width=1.5), legendgroup='group5'),
        row=1, col=5
    )
    fig.add_trace(
        go.Scatter(x=results_df['LS_DEL'], y=depth, mode='lines', name='LS δ',
                  line=dict(color='red', width=1.5, dash='dash'), legendgroup='group5'),
        row=1, col=5
    )
    fig.add_trace(
        go.Scatter(x=results_df['ORTHO_DEL1'], y=depth, mode='lines', name='δ₁',
                  line=dict(color='green', width=1), legendgroup='group5'),
        row=1, col=5
    )
    fig.add_trace(
        go.Scatter(x=results_df['MONO_ZETA1'], y=depth, mode='lines', name='ζ₁',
                  line=dict(color='magenta', width=1), legendgroup='group5'),
        row=1, col=5
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="Well Log Fracture Model Analysis",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=10)
        ),
        hovermode='y unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="ε", row=1, col=2)
    fig.update_xaxes(title_text="Orthorhombic Parameters", row=1, col=3)
    fig.update_xaxes(title_text="Monoclinic Parameters", row=1, col=4)
    fig.update_xaxes(title_text="Delta / Zeta", row=1, col=5)
    
    fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_model_comparison_plotly(model, crack_density_range: np.ndarray):
    """
    Plot comparison of different fracture models using Plotly
    """
    results = {
        'crack_density': crack_density_range,
        'hudson_epsilon': [],
        'hudson_gamma': [],
        'hudson_delta': [],
        'linear_slip_epsilon': [],
        'linear_slip_gamma': [],
        'linear_slip_delta': [],
        'orthorhombic_epsilon1': [],
        'orthorhombic_epsilon2': [],
        'orthorhombic_gamma1': [],
        'orthorhombic_gamma2': [],
        'monoclinic_epsilon_x': [],
        'monoclinic_epsilon_y': []
    }
    
    for e in crack_density_range:
        # Hudson model
        h = model.hudson_model(e, aspect_ratio=0.01, fluid_content='dry')
        results['hudson_epsilon'].append(h['epsilon'])
        results['hudson_gamma'].append(h['gamma'])
        results['hudson_delta'].append(h['delta'])
        
        # Linear slip model (single fracture set)
        ZN = e * (1 - model.gamma_bg) / (model.mu * model.gamma_bg) if model.mu * model.gamma_bg != 0 else 0
        ZT = e / model.mu if model.mu != 0 else 0
        
        fracture_sets = [{
            'normal_compliance': ZN,
            'shear_compliance': ZT,
            'azimuth': 0.0,
            'dip': 90.0
        }]
        
        ls = model.linear_slip_model(fracture_sets)
        if ls['symmetry'] == 'HTI':
            results['linear_slip_epsilon'].append(ls['epsilon'])
            results['linear_slip_gamma'].append(ls['gamma'])
            results['linear_slip_delta'].append(ls['delta'])
        
        # Orthorhombic (two orthogonal sets)
        ortho = model.multiple_fracture_sets_orthorhombic(e, e/2)
        results['orthorhombic_epsilon1'].append(ortho['epsilon_1'])
        results['orthorhombic_epsilon2'].append(ortho['epsilon_2'])
        results['orthorhombic_gamma1'].append(ortho['gamma_1'])
        results['orthorhombic_gamma2'].append(ortho['gamma_2'])
        
        # Monoclinic (two non-orthogonal sets at 45°)
        mono = model.multiple_fracture_sets_monoclinic(e, e/2, azimuth_set2=45)
        results['monoclinic_epsilon_x'].append(mono['epsilon_x'])
        results['monoclinic_epsilon_y'].append(mono['epsilon_y'])
    
    # Create subplots (2x2)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P-Wave Anisotropy Parameters (ε)', 
                       'Shear-Wave Anisotropy Parameters (γ)',
                       'NMO-Related Parameters (δ) - Single Set Only',
                       'Anisotropy Ratios for Multiple Fracture Sets'),
        horizontal_spacing=0.15,
        vertical_spacing=0.12
    )
    
    # Plot epsilon parameters (row 1, col 1)
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['hudson_epsilon'], 
                  mode='lines', name='Hudson (ε)', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['linear_slip_epsilon'], 
                  mode='lines', name='Linear Slip (ε)', line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['orthorhombic_epsilon1'], 
                  mode='lines', name='Orthorhombic ε₁', line=dict(color='green', width=2, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['orthorhombic_epsilon2'], 
                  mode='lines', name='Orthorhombic ε₂', line=dict(color='green', width=2, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['monoclinic_epsilon_x'], 
                  mode='lines', name='Monoclinic εₓ', line=dict(color='magenta', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['monoclinic_epsilon_y'], 
                  mode='lines', name='Monoclinic εᵧ', line=dict(color='magenta', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Plot gamma parameters (row 1, col 2)
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['hudson_gamma'], 
                  mode='lines', name='Hudson (γ)', line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['linear_slip_gamma'], 
                  mode='lines', name='Linear Slip (γ)', line=dict(color='red', width=2, dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['orthorhombic_gamma1'], 
                  mode='lines', name='Orthorhombic γ₁', line=dict(color='green', width=2, dash='dot')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['orthorhombic_gamma2'], 
                  mode='lines', name='Orthorhombic γ₂', line=dict(color='green', width=2, dash='dot')),
        row=1, col=2
    )
    
    # Plot delta parameters (row 2, col 1)
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['hudson_delta'], 
                  mode='lines', name='Hudson (δ)', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=results['linear_slip_delta'], 
                  mode='lines', name='Linear Slip (δ)', line=dict(color='red', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Plot anisotropy ratios (row 2, col 2)
    ortho_ratio = np.array(results['orthorhombic_epsilon1']) / np.array(results['orthorhombic_epsilon2'])
    mono_ratio = np.array(results['monoclinic_epsilon_x']) / np.array(results['monoclinic_epsilon_y'])
    
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=ortho_ratio, 
                  mode='lines', name='ε₁/ε₂ (Orthorhombic)', line=dict(color='green', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=crack_density_range, y=mono_ratio, 
                  mode='lines', name='εₓ/εᵧ (Monoclinic)', line=dict(color='magenta', width=2)),
        row=2, col=2
    )
    
    # Add horizontal line at y=1
    fig.add_hline(y=1.0, line=dict(color='black', width=1, dash='dot'), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Fracture Model Comparison - Thomsen Parameters",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=10)
        ),
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Crack Density", row=1, col=1)
    fig.update_xaxes(title_text="Crack Density", row=1, col=2)
    fig.update_xaxes(title_text="Crack Density", row=2, col=1)
    fig.update_xaxes(title_text="Crack Density", row=2, col=2)
    
    fig.update_yaxes(title_text="ε", row=1, col=1)
    fig.update_yaxes(title_text="γ", row=1, col=2)
    fig.update_yaxes(title_text="δ", row=2, col=1)
    fig.update_yaxes(title_text="Parameter Ratio", row=2, col=2)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def extract_thomsen_values(results_df: pd.DataFrame, depth: float, tolerance: float = 1.0):
    """
    Extract Thomsen parameters at a specific depth
    """
    # Find closest depth
    idx = (np.abs(results_df['DEPTH'] - depth)).idxmin()
    actual_depth = results_df.loc[idx, 'DEPTH']
    
    if abs(actual_depth - depth) > tolerance:
        st.warning(f"Closest depth is {actual_depth:.1f} m (requested {depth:.1f} m)")
    
    values = {
        'depth': actual_depth,
        'hudson': {
            'epsilon': results_df.loc[idx, 'HUDSON_EPS'],
            'gamma': results_df.loc[idx, 'HUDSON_GAM'],
            'delta': results_df.loc[idx, 'HUDSON_DEL']
        },
        'linear_slip': {
            'epsilon': results_df.loc[idx, 'LS_EPS'],
            'gamma': results_df.loc[idx, 'LS_GAM'],
            'delta': results_df.loc[idx, 'LS_DEL']
        },
        'orthorhombic': {
            'epsilon_1': results_df.loc[idx, 'ORTHO_EPS1'],
            'epsilon_2': results_df.loc[idx, 'ORTHO_EPS2'],
            'gamma_1': results_df.loc[idx, 'ORTHO_GAM1'],
            'gamma_2': results_df.loc[idx, 'ORTHO_GAM2'],
            'delta_1': results_df.loc[idx, 'ORTHO_DEL1'],
            'delta_2': results_df.loc[idx, 'ORTHO_DEL2'],
            'delta_3': results_df.loc[idx, 'ORTHO_DEL3']
        },
        'monoclinic': {
            'epsilon_x': results_df.loc[idx, 'MONO_EPSX'],
            'epsilon_y': results_df.loc[idx, 'MONO_EPSY'],
            'gamma_x': results_df.loc[idx, 'MONO_GAMX'],
            'gamma_y': results_df.loc[idx, 'MONO_GAMY'],
            'zeta_1': results_df.loc[idx, 'MONO_ZETA1'],
            'zeta_2': results_df.loc[idx, 'MONO_ZETA2']
        }
    }
    
    return values


# Main Streamlit app
def main():
    st.markdown('<h1 class="main-header">📊 Fracture Model Analysis - Thomsen Parameters</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data source
        data_source = st.radio(
            "Data Source",
            ["Upload CSV", "Use Example Data"]
        )
        
        # Fracture parameters
        st.markdown("### Fracture Parameters")
        crack_density = st.slider("Crack Density", 0.01, 0.2, 0.08, 0.01)
        aspect_ratio = st.slider("Aspect Ratio", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        
        st.markdown("### Fracture Orientations")
        fracture_azimuth1 = st.slider("Set 1 Azimuth (°)", 0, 180, 0, 5)
        fracture_azimuth2 = st.slider("Set 2 Azimuth (Orthorhombic) (°)", 0, 180, 90, 5)
        fracture_azimuth_mono = st.slider("Set 2 Azimuth (Monoclinic) (°)", 0, 180, 45, 5)
        
        st.markdown("### Processing Options")
        sample_every = st.slider("Sample every N points", 1, 10, 1, 1,
                                 help="Use 1 for all samples, higher values for faster processing")
        
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        st.info("Results will be available for download after processing")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        This app calculates Thomsen anisotropy parameters for different fracture models:
        - **Hudson's Model**: Penny-shaped cracks
        - **Linear Slip Model**: Schoenberg's formulation
        - **Orthorhombic**: Two orthogonal fracture sets
        - **Monoclinic**: Two non-orthogonal fracture sets
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Required CSV Format")
        st.code("""
DEPTH,VP,VS,RHO
1000,3200,1650,2400
1005,3250,1680,2420
...
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Load data
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_well_data(uploaded_file)
    else:
        df = generate_example_data()
        st.info("Using example synthetic data for demonstration")
    
    if df is not None:
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", len(df))
        with col2:
            st.metric("Depth Range", f"{df['DEPTH'].min():.0f} - {df['DEPTH'].max():.0f} m")
        with col3:
            st.metric("VP Range", f"{df['VP'].min():.0f} - {df['VP'].max():.0f} m/s")
        with col4:
            st.metric("RHO Range", f"{df['RHO'].min():.0f} - {df['RHO'].max():.0f} kg/m³")
        
        # Depth range selection
        st.markdown('<h2 class="sub-header">📏 Select Depth Range</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            depth_min = float(df['DEPTH'].min())
            depth_max = float(df['DEPTH'].max())
            start_depth = st.number_input("Start Depth (m)", depth_min, depth_max, depth_min)
        with col2:
            end_depth = st.number_input("End Depth (m)", depth_min, depth_max, depth_max)
        
        if start_depth < end_depth:
            df_selected = df[(df['DEPTH'] >= start_depth) & (df['DEPTH'] <= end_depth)].copy()
            st.success(f"Selected {len(df_selected)} samples from {start_depth:.1f} to {end_depth:.1f} m")
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Processing data..."):
                    # Run analysis
                    results = analyze_well_with_fracture_models(
                        df_selected,
                        crack_density=crack_density,
                        aspect_ratio=aspect_ratio,
                        fracture_azimuth1=fracture_azimuth1,
                        fracture_azimuth2=fracture_azimuth2,
                        fracture_azimuth_mono=fracture_azimuth_mono,
                        sample_every=sample_every
                    )
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    
                    # Display results
                    st.markdown('<h2 class="sub-header">📈 Well Log Analysis Results</h2>', 
                               unsafe_allow_html=True)
                    
                    # Plot well results
                    fig1 = plot_well_results_plotly(results)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Model comparison using average properties
                    st.markdown('<h2 class="sub-header">🔄 Model Comparison</h2>', 
                               unsafe_allow_html=True)
                    
                    avg_vp = results['VP'].mean()
                    avg_vs = results['VS'].mean()
                    avg_rho = results['RHO'].mean()
                    
                    st.info(f"Using average properties: VP={avg_vp:.0f} m/s, VS={avg_vs:.0f} m/s, RHO={avg_rho:.0f} kg/m³")
                    
                    model_avg = FractureModelThomsen(avg_vp, avg_vs, avg_rho)
                    crack_density_range = np.linspace(0.01, 0.15, 20)
                    fig2 = plot_model_comparison_plotly(model_avg, crack_density_range)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Extract values at specific depths
                    st.markdown('<h2 class="sub-header">🎯 Extract Values at Depth</h2>', 
                               unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        extract_depth = st.number_input(
                            "Enter depth to extract values (m)",
                            float(results['DEPTH'].min()),
                            float(results['DEPTH'].max()),
                            float(results['DEPTH'].mean())
                        )
                        
                        if st.button("Extract Values"):
                            values = extract_thomsen_values(results, extract_depth)
                            
                            with col2:
                                st.markdown(f"### Thomsen Parameters at {values['depth']:.1f} m")
                                
                                tab1, tab2, tab3, tab4 = st.tabs(["Hudson", "Linear Slip", "Orthorhombic", "Monoclinic"])
                                
                                with tab1:
                                    st.write(f"**ε** = {values['hudson']['epsilon']:.6f}")
                                    st.write(f"**γ** = {values['hudson']['gamma']:.6f}")
                                    st.write(f"**δ** = {values['hudson']['delta']:.6f}")
                                
                                with tab2:
                                    st.write(f"**ε** = {values['linear_slip']['epsilon']:.6f}")
                                    st.write(f"**γ** = {values['linear_slip']['gamma']:.6f}")
                                    st.write(f"**δ** = {values['linear_slip']['delta']:.6f}")
                                
                                with tab3:
                                    st.write(f"**ε₁** = {values['orthorhombic']['epsilon_1']:.6f}")
                                    st.write(f"**ε₂** = {values['orthorhombic']['epsilon_2']:.6f}")
                                    st.write(f"**γ₁** = {values['orthorhombic']['gamma_1']:.6f}")
                                    st.write(f"**γ₂** = {values['orthorhombic']['gamma_2']:.6f}")
                                
                                with tab4:
                                    st.write(f"**εₓ** = {values['monoclinic']['epsilon_x']:.6f}")
                                    st.write(f"**εᵧ** = {values['monoclinic']['epsilon_y']:.6f}")
                                    st.write(f"**γₓ** = {values['monoclinic']['gamma_x']:.6f}")
                                    st.write(f"**γᵧ** = {values['monoclinic']['gamma_y']:.6f}")
                    
                    # Download results
                    st.markdown('<h2 class="sub-header">💾 Download Results</h2>', 
                               unsafe_allow_html=True)
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fracture_model_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    # Summary statistics
                    with st.expander("📊 View Summary Statistics"):
                        stats_cols = ['HUDSON_EPS', 'LS_EPS', 'ORTHO_EPS1', 'ORTHO_EPS2', 
                                     'MONO_EPSX', 'MONO_EPSY', 'HUDSON_GAM', 'LS_GAM']
                        stats = results[stats_cols].describe()
                        st.dataframe(stats)
        else:
            st.error("Start depth must be less than end depth")
    
    # Display previous results if they exist in session state
    if 'results' in st.session_state:
        st.markdown('<h2 class="sub-header">📈 Previous Analysis Results</h2>', 
                   unsafe_allow_html=True)
        fig1 = plot_well_results_plotly(st.session_state['results'])
        st.plotly_chart(fig1, use_container_width=True)


if __name__ == "__main__":
    main()
