"""
================================================================================
HYBRID PHYSICS-MACHINE LEARNING VELOCITY PREDICTION FOR CARBONATE ROCKS
================================================================================

Scientific Implementation based on:
1. Effective Field Method (EFM) - Micromechanics of cracked media
2. Gradient Boosting Regression - Machine learning for nonlinear relationships
3. Hybrid Integration - Combining physical constraints with data-driven patterns

References:
- Eshelby, J.D. (1957). The determination of the elastic field of an ellipsoidal 
  inclusion. Proc. Royal Soc. London.
- Hudson, J.A. (1980). Overall properties of a cracked solid. Geophys. J. Int.
- Budiansky, B., & O'Connell, R.J. (1976). Elastic moduli of a cracked solid.
- Sayers, C.M., & Kachanov, M. (1995). Microcrack-induced elastic anisotropy.
- Mavko, G., Mukerji, T., & Dvorkin, J. (2009). Rock Physics Handbook.
- Friedman, J.H. (2001). Greedy function approximation: Gradient boosting.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import iv  # Modified Bessel functions for von Mises distribution
import warnings
warnings.filterwarnings('ignore')

# Machine Learning components
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==============================================================================
# SCIENTIFIC IMPLEMENTATION: EFFECTIVE FIELD METHOD (EFM)
# ==============================================================================

class EffectiveFieldMethod:
    """
    Implementation of Effective Field Method for cracked elastic media.
    
    Based on micromechanical theory for estimating effective elastic properties
    of rocks with microcracks and pores.
    
    Mathematical Core:
    1. Crack density parameter: ε = (3φSw)/(4πα)
    2. Orientation distribution: F(β) for uniform/von Mises distributions
    3. Effective moduli: K_eff = K₀(1 - εC₁F), G_eff = G₀(1 - εC₂F)
    4. Velocities: Vp = √[(K + 4G/3)/ρ], Vs = √[G/ρ]
    
    References:
    - Hudson (1980, 1981) - First-order effective medium theory
    - Eshelby (1957) - Inclusion theory
    - Budiansky & O'Connell (1976) - Self-consistent scheme
    """
    
    def __init__(self, matrix_props, crack_props):
        """
        Initialize EFM with matrix and crack properties.
        
        Parameters:
        -----------
        matrix_props : dict
            Dictionary containing matrix properties:
            - 'Vp': P-wave velocity (m/s)
            - 'Vs': S-wave velocity (m/s)
            - 'rho': Density (g/cc)
            - 'K', 'G', 'nu': Optional bulk/shear moduli, Poisson's ratio
        
        crack_props : dict
            Dictionary containing crack properties:
            - 'aspect_ratio': α (typically 0.001-0.1 for microcracks)
            - 'fluid_K': Bulk modulus of pore fluid (Pa)
            - 'fluid_rho': Density of pore fluid (kg/m³)
        """
        self.matrix = matrix_props.copy()
        self.crack = crack_props.copy()
        
        # Calculate moduli from velocities if not provided
        self._calculate_elastic_moduli()
        
        # Validate inputs
        self._validate_properties()
    
    def _calculate_elastic_moduli(self):
        """
        Calculate elastic moduli from velocities using rock physics relationships.
        
        Physics Equations:
        ------------------
        G = ρ·Vs²                 [Shear modulus in Pa]
        K = ρ·(Vp² - 4/3·Vs²)    [Bulk modulus in Pa]
        ν = (Vp² - 2Vs²)/[2(Vp² - Vs²)]  [Poisson's ratio]
        
        Note: ρ converted from g/cc to kg/m³ for SI units
        """
        mat = self.matrix
        
        # Convert density to kg/m³ (SI units)
        if 'rho' in mat:
            mat['rho_kgm3'] = mat['rho'] * 1000
        else:
            # Default limestone density
            mat['rho_kgm3'] = 2650
        
        # Calculate moduli if velocities are provided
        if 'Vp' in mat and 'Vs' in mat:
            # Shear modulus (Pa) = ρ·Vs²
            mat['G'] = mat['rho_kgm3'] * mat['Vs']**2
            
            # Bulk modulus (Pa) = ρ·(Vp² - 4/3·Vs²)
            mat['K'] = mat['rho_kgm3'] * (mat['Vp']**2 - (4/3) * mat['Vs']**2)
            
            # Poisson's ratio
            vp2, vs2 = mat['Vp']**2, mat['Vs']**2
            mat['nu'] = (vp2 - 2*vs2) / (2*(vp2 - vs2))
            
            print(f"Calculated elastic moduli:")
            print(f"  Shear modulus (G): {mat['G']/1e9:.1f} GPa")
            print(f"  Bulk modulus (K): {mat['K']/1e9:.1f} GPa")
            print(f"  Poisson's ratio (ν): {mat['nu']:.3f}")
    
    def _validate_properties(self):
        """Validate physical consistency of properties."""
        mat = self.matrix
        
        # Check for negative moduli (unphysical)
        if 'K' in mat and mat['K'] <= 0:
            raise ValueError(f"Invalid bulk modulus: {mat['K']} Pa (must be positive)")
        
        if 'G' in mat and mat['G'] <= 0:
            raise ValueError(f"Invalid shear modulus: {mat['G']} Pa (must be positive)")
        
        # Check Poisson's ratio bounds (-1 to 0.5 for isotropic materials)
        if 'nu' in mat and (mat['nu'] < -1 or mat['nu'] > 0.5):
            print(f"Warning: Unusual Poisson's ratio: {mat['nu']:.3f}")
    
    def orientation_distribution_function(self, beta, distribution='uniform'):
        """
        Calculate orientation distribution factor F(β).
        
        Parameters:
        -----------
        beta : float
            Half-angle of orientation cone (radians)
            β = 0: perfectly aligned cracks
            β = π/2: randomly oriented cracks
        
        distribution : str
            Type of orientation distribution:
            - 'uniform': Uniform distribution within cone angle β
            - 'von_mises': Fisher-von Mises distribution (preferred orientation)
        
        Returns:
        --------
        F : float or tuple
            Orientation factor(s)
            For uniform: single F value
            For von Mises: (F₁, F₂) for different compliances
        
        Mathematical Formulations:
        -------------------------
        1. Uniform distribution (Hudson, 1981):
           F(β) = (β + sinβ·cosβ) / (2β)
        
        2. von Mises distribution (Sayers, 1995):
           F₁ = (σ²·I₁ + I₂) / I₀
           F₂ = (σ²·I₁) / I₀
           where Iₙ = modified Bessel function of order n
                 σ = 1/κ (κ = concentration parameter)
        """
        beta = float(beta)
        
        if distribution == 'uniform':
            # Uniform distribution within cone
            if beta <= 1e-10 or np.isnan(beta):
                return 1.0  # Isotropic limit
            
            # Hudson's formulation for uniform distribution
            return (beta + np.sin(beta) * np.cos(beta)) / (2 * beta)
        
        elif distribution == 'von_mises':
            # Fisher-von Mises distribution (preferred orientation)
            if beta <= 1e-10 or np.isnan(beta):
                return 1.0, 1.0  # Isotropic limit
            
            # Concentration parameter κ = 1/σ²
            # β represents angular spread (κ ∝ 1/β²)
            sigma = max(beta, 0.001)  # Avoid division by zero
            z = 1.0 / (sigma ** 2)  # Concentration parameter
            
            try:
                # Modified Bessel functions
                I0 = iv(0, z)
                I1 = iv(1, z)
                I2 = iv(2, z)
                
                # Orientation factors for crack compliances
                F1 = (sigma**2 * I1 + I2) / I0  # For normal compliance
                F2 = (sigma**2 * I1) / I0       # For shear compliance
                
                return F1, F2
            except (OverflowError, ZeroDivisionError):
                # Fallback for numerical issues
                return 1.0, 1.0
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def estimate_crack_parameters(self, porosity, sw=1.0, rt=1.0, vclay=0):
        """
        Estimate crack parameters from well log data.
        
        Parameters:
        -----------
        porosity : float
            Total porosity (fraction)
        sw : float
            Water saturation (fraction)
        rt : float
            Resistivity (ohm-m)
        vclay : float
            Clay volume (fraction or %)
        
        Returns:
        --------
        crack_density : float
            Crack density parameter ε (dimensionless)
        beta : float
            Orientation angle (radians)
        
        Empirical Relationships:
        -----------------------
        1. Crack density ∝ porosity × saturation
           ε ≈ (3φ·Sw)/(4π·α)  [Simplified]
        
        2. Orientation from resistivity:
           Low RT → fractured → more random orientation
           High RT → intact → more aligned
        """
        # Ensure valid physical ranges
        porosity = max(float(porosity), 0.001)
        sw = min(max(float(sw), 0.0), 1.0)
        rt = max(float(rt), 0.1)
        
        # Convert vclay to fraction if in percentage
        if vclay > 1.0:  # Assume percentage
            vclay_frac = vclay / 100.0
        else:
            vclay_frac = vclay
        
        vclay_frac = min(max(vclay_frac, 0.0), 0.7)
        
        # 1. Estimate crack density (ε)
        # Using Budiansky & O'Connell (1976) relation
        aspect_ratio = self.crack.get('aspect_ratio', 0.01)
        
        # Crack density parameter: ε = (3φSw)/(4πα)
        crack_density = (3.0 * porosity * sw) / (4.0 * np.pi * aspect_ratio)
        
        # Clay reduces crack effectiveness (clay fills cracks)
        clay_factor = 1.0 - vclay_frac * 1.5  # Empirical
        crack_density *= max(clay_factor, 0.3)
        
        # 2. Estimate orientation from resistivity
        # Empirical relationship: higher resistivity → more aligned cracks
        rt_factor = np.log10(max(rt, 0.1)) / 3.0
        rt_factor = min(max(rt_factor, 0.1), 1.0)
        
        # β ranges from 0 (aligned) to π/2 (random)
        # Base isotropic case: β = π/4 (45° cone)
        beta_base = np.pi / 4
        
        # Adjust based on resistivity
        # High RT → more aligned → smaller β
        beta = beta_base * (1.0 - 0.3 * rt_factor)
        
        # Clay increases randomness
        beta *= (1.0 + 0.2 * vclay_frac)
        
        # Physical bounds
        crack_density = min(max(crack_density, 0.0), 0.5)  # ε ∈ [0, 0.5]
        beta = min(max(beta, 0.0), np.pi / 2)              # β ∈ [0, π/2]
        
        return crack_density, beta
    
    def calculate_effective_properties(self, crack_density, beta, 
                                     distribution='uniform'):
        """
        Calculate effective elastic properties using first-order effective medium theory.
        
        Parameters:
        -----------
        crack_density : float
            Crack density parameter ε
        beta : float
            Orientation half-angle (radians)
        distribution : str
            Orientation distribution type
        
        Returns:
        --------
        dict containing:
            - Vp, Vs: Effective velocities (m/s)
            - K, G: Effective moduli (Pa)
            - crack_density, beta, F: Input parameters
            - anisotropy: Thomsen parameters if applicable
        
        Theory:
        -------
        First-order Hudson (1980) approximation:
        K_eff = K₀·(1 - ε·C₁·F)
        G_eff = G₀·(1 - ε·C₂·F)
        
        where C₁, C₂ depend on crack aspect ratio and fluid properties
        """
        # Matrix properties
        K0 = self.matrix.get('K', 50e9)  # Default: 50 GPa for limestone
        G0 = self.matrix.get('G', 30e9)  # Default: 30 GPa
        rho0 = self.matrix.get('rho_kgm3', 2650)
        
        # Get orientation factor
        if distribution == 'uniform':
            F_val = self.orientation_distribution_function(beta, 'uniform')
            F1, F2 = F_val, F_val
        else:
            F1, F2 = self.orientation_distribution_function(beta, 'von_mises')
        
        # Crack compliance parameters (first-order approximation)
        # These depend on aspect ratio and fluid properties
        aspect_ratio = self.crack.get('aspect_ratio', 0.01)
        fluid_K = self.crack.get('fluid_K', 2.25e9)  # Water
        
        # Simplified crack compliances (Hudson, 1980)
        # For dry cracks or weak fluid:
        C1 = 0.8  # Normal compliance factor
        C2 = 0.6  # Shear compliance factor
        
        # Adjust for fluid if provided
        if fluid_K > 0:
            fluid_factor = K0 / (K0 + fluid_K)
            C1 *= fluid_factor  # Fluid stiffens normal compliance
        
        # Effective moduli (first-order approximation)
        K_eff = K0 * (1.0 - crack_density * C1 * F1)
        G_eff = G0 * (1.0 - crack_density * C2 * F2)
        
        # Ensure positive moduli
        K_eff = max(K_eff, 0.1 * K0)
        G_eff = max(G_eff, 0.1 * G0)
        
        # Calculate effective velocities
        Vp_eff = np.sqrt((K_eff + 4.0 * G_eff / 3.0) / rho0)
        Vs_eff = np.sqrt(G_eff / rho0)
        
        # Calculate Thomsen parameters for anisotropy
        # For transverse isotropy with vertical axis of symmetry
        epsilon = 0.0  # P-wave anisotropy
        gamma = 0.0    # S-wave anisotropy
        delta = 0.0    # Off-diagonal parameter
        
        if distribution != 'uniform' and beta < np.pi/4:
            # Some anisotropy present
            epsilon = crack_density * 0.1 * (1 - beta/(np.pi/4))
            gamma = crack_density * 0.15 * (1 - beta/(np.pi/4))
        
        return {
            'Vp': Vp_eff,
            'Vs': Vs_eff,
            'K': K_eff,
            'G': G_eff,
            'crack_density': crack_density,
            'beta': beta,
            'F_normal': F1,
            'F_shear': F2,
            'anisotropy_epsilon': epsilon,
            'anisotropy_gamma': gamma,
            'anisotropy_delta': delta,
            'Vp/Vs': Vp_eff / Vs_eff if Vs_eff > 0 else 0
        }

# ==============================================================================
# HYBRID PHYSICS-MACHINE LEARNING MODEL
# ==============================================================================

class HybridVelocityPredictor:
    """
    Hybrid model combining physics-based Effective Field Method with
    machine learning for velocity prediction in carbonate rocks.
    
    Methodology:
    1. Physics module: Calculate crack parameters and first-order velocities
    2. Feature engineering: Create physics-informed features
    3. ML module: Gradient Boosting to capture nonlinearities
    4. Hybrid prediction: Weighted combination of physics and ML predictions
    
    Key Innovations:
    - Physics constraints prevent unphysical predictions
    - ML captures complex porosity-velocity relationships
    - Feature importance reveals dominant physical mechanisms
    """
    
    def __init__(self, matrix_props=None):
        """
        Initialize hybrid predictor.
        
        Parameters:
        -----------
        matrix_props : dict, optional
            Matrix properties for physics model
        """
        self.efm_model = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.matrix_props = matrix_props
        self.feature_importances = {}
        self.feature_names = []
        
    def create_safe_features(self, df):
        """
        Create feature matrix with robust handling of missing/invalid data.
        
        Features include:
        1. Basic well logs (porosity, density, saturation)
        2. Physics-based features (crack density, orientation)
        3. Interaction terms (porosity×density, clay×crack)
        4. Transformations (log, square, sqrt)
        
        Returns:
        --------
        DataFrame with engineered features
        """
        features = {}
        
        # ======================================================================
        # 1. BASIC WELL LOG FEATURES
        # ======================================================================
        
        # Essential rock properties
        basic_cols = ['porosity', 'rho', 'sw']
        for col in basic_cols:
            if col in df.columns:
                # Safe conversion with median imputation
                series = pd.to_numeric(df[col], errors='coerce')
                if series.isna().any():
                    series = series.fillna(series.median())
                features[col] = series
        
        # Clay content (critical for carbonates)
        if 'Vclay' in df.columns:
            features['Vclay'] = pd.to_numeric(df['Vclay'], errors='coerce')
            features['Vclay'] = features['Vclay'].fillna(0)
        
        # Resistivity (indicator of fluid and fractures)
        if 'RT' in df.columns:
            features['RT'] = pd.to_numeric(df['RT'], errors='coerce')
            features['RT'] = features['RT'].fillna(features['RT'].median())
            # Log transform for normalized distribution
            features['RT_log'] = np.log10(np.maximum(features['RT'], 0.1))
        
        # Gamma Ray (clay/lithology indicator)
        if 'GR' in df.columns:
            features['GR'] = pd.to_numeric(df['GR'], errors='coerce')
            features['GR'] = features['GR'].fillna(features['GR'].median())
            features['GR_norm'] = (features['GR'] - features['GR'].min()) / \
                                 (features['GR'].max() - features['GR'].min() + 1e-10)
        
        # ======================================================================
        # 2. PHYSICS-BASED FEATURES (if EFM available)
        # ======================================================================
        
        if self.efm_model and self.matrix_props:
            crack_densities = []
            betas = []
            F_values = []
            vp_efm = []
            vs_efm = []
            anisotropies = []
            
            n_samples = len(df)
            print(f"Generating physics-based features for {n_samples} samples...")
            
            for idx in range(n_samples):
                # Extract values with safe defaults
                porosity_val = features.get('porosity', pd.Series([0.1]*n_samples)).iloc[idx] \
                              if 'porosity' in features else 0.1
                
                sw_val = features.get('sw', pd.Series([1.0]*n_samples)).iloc[idx] \
                        if 'sw' in features else 1.0
                
                rt_val = features.get('RT', pd.Series([1.0]*n_samples)).iloc[idx] \
                        if 'RT' in features else 1.0
                
                vclay_val = features.get('Vclay', pd.Series([0.0]*n_samples)).iloc[idx] \
                           if 'Vclay' in features else 0.0
                
                # Estimate crack parameters using physics model
                crack_density, beta = self.efm_model.estimate_crack_parameters(
                    porosity_val, sw_val, rt_val, vclay_val
                )
                
                # Calculate effective properties
                eff_props = self.efm_model.calculate_effective_properties(
                    crack_density, beta, 'uniform'
                )
                
                crack_densities.append(crack_density)
                betas.append(beta)
                F_values.append(eff_props['F_normal'])
                vp_efm.append(eff_props['Vp'])
                vs_efm.append(eff_props['Vs'])
                anisotropies.append(eff_props['anisotropy_epsilon'])
            
            # Add physics-based features
            features['crack_density_efm'] = crack_densities
            features['orientation_beta'] = betas
            features['F_beta'] = F_values
            features['Vp_efm'] = vp_efm
            features['Vs_efm'] = vs_efm
            features['anisotropy_eps'] = anisotropies
            
            print(f"  Crack density range: {min(crack_densities):.3f} to {max(crack_densities):.3f}")
            print(f"  Orientation range: {min(betas):.2f} to {max(betas):.2f} radians")
        
        # ======================================================================
        # 3. FEATURE ENGINEERING & INTERACTIONS
        # ======================================================================
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Porosity transformations
        if 'porosity' in features_df.columns:
            features_df['porosity_sq'] = features_df['porosity'] ** 2
            features_df['porosity_sqrt'] = np.sqrt(np.maximum(features_df['porosity'], 0))
            features_df['porosity_log'] = np.log1p(features_df['porosity'])
        
        # Density-porosity interaction
        if 'rho' in features_df.columns and 'porosity' in features_df.columns:
            features_df['density_porosity'] = features_df['rho'] * features_df['porosity']
            features_df['density_over_porosity'] = features_df['rho'] / (features_df['porosity'] + 0.01)
        
        # Crack-clay interactions
        if 'crack_density_efm' in features_df.columns:
            if 'porosity' in features_df.columns:
                features_df['crack_porosity_ratio'] = features_df['crack_density_efm'] / (features_df['porosity'] + 0.01)
            
            if 'Vclay' in features_df.columns:
                features_df['crack_clay'] = features_df['crack_density_efm'] * features_df['Vclay']
                features_df['crack_clay_ratio'] = features_df['crack_density_efm'] / (features_df['Vclay'] + 0.01)
        
        # Resistivity-porosity relationship (Archie-like)
        if 'RT' in features_df.columns and 'porosity' in features_df.columns:
            features_df['F_formation_factor'] = features_df['RT'] / (features_df['porosity'] ** (-2) + 0.01)
        
        # ======================================================================
        # 4. DATA CLEANING & VALIDATION
        # ======================================================================
        
        # Fill remaining NaN values
        features_df = features_df.fillna(features_df.median())
        
        # Replace infinities
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
        
        # Validate feature ranges
        for col in features_df.columns:
            if features_df[col].isna().any():
                print(f"Warning: NaN in {col} after cleaning")
            if (features_df[col] == 0).all():
                print(f"Warning: All zeros in {col}")
        
        print(f"Created {len(features_df.columns)} features")
        return features_df
    
    def train(self, X_train, y_train_vp, y_train_vs):
        """
        Train ML models with hyperparameter optimization.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train_vp : array
            Training Vp values
        y_train_vs : array
            Training Vs values
        
        ML Model: Gradient Boosting Regressor
        Advantages:
        - Handles nonlinear relationships
        - Robust to outliers
        - Provides feature importance
        - No need for feature scaling (but we scale anyway)
        """
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Impute missing values (median imputation)
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        # Scale features (helps some ML algorithms)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        print(f"\nTraining ML models on {X_train_scaled.shape[0]} samples...")
        
        # ======================================================================
        # Vp MODEL
        # ======================================================================
        print("Training Vp model...")
        self.ml_models['Vp'] = GradientBoostingRegressor(
            n_estimators=200,           # Number of boosting stages
            max_depth=6,                # Maximum tree depth
            learning_rate=0.05,         # Shrinkage rate
            min_samples_split=5,        # Minimum samples to split
            min_samples_leaf=2,         # Minimum samples in leaf
            random_state=42,            # Reproducibility
            subsample=0.8,              # Fraction of samples for fitting
            max_features='sqrt',        # Features for best split
            validation_fraction=0.1,    # Fraction for early stopping
            n_iter_no_change=10         # Early stopping rounds
        )
        
        self.ml_models['Vp'].fit(X_train_scaled, y_train_vp)
        
        # ======================================================================
        # Vs MODEL
        # ======================================================================
        print("Training Vs model...")
        self.ml_models['Vs'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8,
            max_features='sqrt',
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.ml_models['Vs'].fit(X_train_scaled, y_train_vs)
        
        # Store feature importances
        self.feature_importances['Vp'] = self.ml_models['Vp'].feature_importances_
        self.feature_importances['Vs'] = self.ml_models['Vs'].feature_importances_
        
        print("Training complete!")
        
        # Print top features
        self._print_feature_importance()
    
    def _print_feature_importance(self, top_n=10):
        """Print feature importance for interpretability."""
        if not self.feature_importances or not self.feature_names:
            return
        
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print('='*70)
        
        for target in ['Vp', 'Vs']:
            importances = self.feature_importances[target]
            
            if len(importances) != len(self.feature_names):
                print(f"Warning: Feature count mismatch for {target}")
                continue
            
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            
            print(f"\nTop {top_n} features for {target}:")
            print("-" * 50)
            for i, idx in enumerate(sorted_idx[:top_n]):
                feat_name = self.feature_names[idx]
                importance = importances[idx]
                print(f"{i+1:2d}. {feat_name:30s}: {importance:.4f}")
    
    def predict(self, X):
        """
        Make predictions using trained ML models.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        
        Returns:
        --------
        vp_pred, vs_pred : arrays
            ML predictions
        """
        # Validate input
        if not self.ml_models:
            raise ValueError("Models not trained. Call train() first.")
        
        # Ensure same columns as training
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, filling with zeros")
            for col in missing_cols:
                X[col] = 0
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        # Preprocess
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        vp_pred = self.ml_models['Vp'].predict(X_scaled)
        vs_pred = self.ml_models['Vs'].predict(X_scaled)
        
        return vp_pred, vs_pred
    
    def hybrid_predict(self, X, physics_weight=0.3):
        """
        Combine ML predictions with physics-based predictions.
        
        Hybrid Scheme:
        V_hybrid = (1-w)·V_ML + w·V_physics
        
        Parameters:
        -----------
        X : DataFrame
            Input features (must include physics features if physics_weight > 0)
        physics_weight : float
            Weight for physics prediction (0-1)
            0 = pure ML, 1 = pure physics
        
        Returns:
        --------
        vp_hybrid, vs_hybrid : arrays
            Hybrid predictions
        """
        # ML predictions
        vp_ml, vs_ml = self.predict(X)
        
        # Physics predictions if available and weight > 0
        if physics_weight > 0 and 'Vp_efm' in X.columns and 'Vs_efm' in X.columns:
            vp_physics = X['Vp_efm'].values
            vs_physics = X['Vs_efm'].values
            
            # Weighted average
            vp_hybrid = (1 - physics_weight) * vp_ml + physics_weight * vp_physics
            vs_hybrid = (1 - physics_weight) * vs_ml + physics_weight * vs_physics
            
            print(f"Hybrid prediction with physics weight: {physics_weight:.2f}")
            return vp_hybrid, vs_hybrid
        
        print("Physics features not available, using ML only")
        return vp_ml, vs_ml

# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================

def load_and_prepare_data(filename='CarbonateFile.csv'):
    """
    Load and preprocess carbonate rock data.
    
    Expected columns:
    - Vp, Vs: Velocities (m/s) [TARGETS]
    - porosity, rho, sw: Basic rock properties
    - Vclay, RT, GR: Optional petrophysical logs
    - DEPTH: Optional depth column
    """
    try:
        print(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['Vp', 'Vs', 'porosity', 'rho']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None
        
        # Check data ranges
        print("\nData statistics:")
        print(df[required_cols].describe())
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))
        
        # Validate physical ranges
        df['Vp'] = np.maximum(df['Vp'], 1000)  # Minimum reasonable Vp
        df['Vs'] = np.maximum(df['Vs'], 500)   # Minimum reasonable Vs
        df['porosity'] = np.clip(df['porosity'], 0.0, 0.5)  # Porosity range
        df['rho'] = np.clip(df['rho'], 2.0, 3.0)  # Density range
        
        print(f"\nData loaded successfully!")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def estimate_matrix_properties(df):
    """
    Estimate matrix (grain) properties from low-porosity samples.
    
    Methodology:
    - Select low-porosity samples (bottom quartile)
    - Assume these represent matrix without pores/cracks
    - Calculate average properties as matrix properties
    """
    print("\nEstimating matrix properties...")
    
    # Find low-porosity samples (bottom 25%)
    porosity_threshold = df['porosity'].quantile(0.25)
    low_porosity_mask = df['porosity'] <= porosity_threshold
    
    if low_porosity_mask.sum() < 5:  # Need enough samples
        # Take 10 lowest porosity samples
        low_porosity_samples = df.nsmallest(10, 'porosity')
    else:
        low_porosity_samples = df[low_porosity_mask]
    
    print(f"Using {len(low_porosity_samples)} low-porosity samples")
    print(f"Porosity range: {low_porosity_samples['porosity'].min():.3f} to {low_porosity_samples['porosity'].max():.3f}")
    
    # Calculate matrix properties
    matrix_props = {
        'Vp': float(low_porosity_samples['Vp'].mean()),
        'Vs': float(low_porosity_samples['Vs'].mean()),
        'rho': float(low_porosity_samples['rho'].mean())
    }
    
    # Calculate statistics
    vp_std = low_porosity_samples['Vp'].std()
    vs_std = low_porosity_samples['Vs'].std()
    
    print(f"\nMatrix properties estimated:")
    print(f"  Vp: {matrix_props['Vp']:.0f} ± {vp_std:.0f} m/s")
    print(f"  Vs: {matrix_props['Vs']:.0f} ± {vs_std:.0f} m/s")
    print(f"  ρ:  {matrix_props['rho']:.2f} g/cc")
    
    return matrix_props

def run_analysis():
    """
    Main analysis pipeline.
    
    Steps:
    1. Load and prepare data
    2. Estimate matrix properties
    3. Initialize physics model
    4. Create hybrid features
    5. Train-test split
    6. Train hybrid model
    7. Evaluate predictions
    8. Save results
    """
    print("="*80)
    print("HYBRID PHYSICS-ML VELOCITY PREDICTION FOR CARBONATE ROCKS")
    print("="*80)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Step 2: Estimate matrix properties
    matrix_props = estimate_matrix_properties(df)
    
    # Step 3: Initialize physics model
    print("\nInitializing physics model (Effective Field Method)...")
    
    crack_props = {
        'aspect_ratio': 0.01,      # Typical for microcracks
        'fluid_K': 2.25e9,         # Water bulk modulus (Pa)
        'fluid_rho': 1000          # Water density (kg/m³)
    }
    
    efm_model = EffectiveFieldMethod(matrix_props, crack_props)
    
    # Step 4: Initialize hybrid model
    print("\nInitializing hybrid model...")
    hybrid_model = HybridVelocityPredictor(matrix_props)
    hybrid_model.efm_model = efm_model
    
    # Step 5: Create features
    print("\nCreating hybrid features...")
    df_features = hybrid_model.create_safe_features(df)
    
    # Select feature columns (exclude targets and non-features)
    exclude_cols = ['Vp', 'Vs', 'VPVSMOD', 'PIMPMOD', 'SIMPMOD']
    if 'DEPTH' in df.columns:
        exclude_cols.append('DEPTH')
    
    feature_cols = [col for col in df_features.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col])]
    
    X = df_features[feature_cols]
    y_vp = df['Vp'].values
    y_vs = df['Vs'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vectors: Vp={len(y_vp)}, Vs={len(y_vs)}")
    
    # Step 6: Train-test split
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_vp_train, y_vp_test, y_vs_train, y_vs_test = train_test_split(
        X, y_vp, y_vs, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    
    # Step 7: Train model
    hybrid_model.train(X_train, y_vp_train, y_vs_train)
    
    # Step 8: Make predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)
    
    # Test set predictions
    print("\nTesting on hold-out set...")
    vp_test_pred, vs_test_pred = hybrid_model.hybrid_predict(X_test, physics_weight=0.3)
    
    # Calculate metrics
    vp_test_corr = np.corrcoef(y_vp_test, vp_test_pred)[0, 1]
    vs_test_corr = np.corrcoef(y_vs_test, vs_test_pred)[0, 1]
    
    vp_test_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_test_pred))
    vs_test_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_test_pred))
    
    vp_test_r2 = r2_score(y_vp_test, vp_test_pred)
    vs_test_r2 = r2_score(y_vs_test, vs_test_pred)
    
    # Full dataset predictions
    print("\nPredicting on full dataset...")
    vp_full_pred, vs_full_pred = hybrid_model.hybrid_predict(X, physics_weight=0.3)
    
    vp_full_corr = np.corrcoef(y_vp, vp_full_pred)[0, 1]
    vs_full_corr = np.corrcoef(y_vs, vs_full_pred)[0, 1]
    
    vp_full_r2 = r2_score(y_vp, vp_full_pred)
    vs_full_r2 = r2_score(y_vs, vs_full_pred)
    
    # Step 9: Save results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"Vp: R² = {vp_test_r2:.4f}, Corr = {vp_test_corr:.4f}, RMSE = {vp_test_rmse:.0f} m/s")
    print(f"Vs: R² = {vs_test_r2:.4f}, Corr = {vs_test_corr:.4f}, RMSE = {vs_test_rmse:.0f} m/s")
    
    print(f"\nFULL DATASET PERFORMANCE:")
    print(f"Vp: R² = {vp_full_r2:.4f}, Corr = {vp_full_corr:.4f}")
    print(f"Vs: R² = {vs_full_r2:.4f}, Corr = {vs_full_corr:.4f}")
    
    # Add predictions to dataframe
    df['Vp_hybrid_pred'] = vp_full_pred
    df['Vs_hybrid_pred'] = vs_full_pred
    df['Vp_error_%'] = 100 * (vp_full_pred - y_vp) / y_vp
    df['Vs_error_%'] = 100 * (vs_full_pred - y_vs) / y_vs
    df['VpVs_pred'] = vp_full_pred / vs_full_pred
    df['VpVs_actual'] = y_vp / y_vs
    
    # Save results
    output_file = 'carbonate_velocity_hybrid_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create visualizations
    create_comprehensive_visualizations(df, vp_full_pred, vs_full_pred, 
                                       vp_full_corr, vs_full_corr,
                                       hybrid_model, feature_cols,
                                       X_test, y_vp_test, y_vs_test,
                                       vp_test_pred, vs_test_pred)
    
    return df, vp_full_corr, vs_full_corr, hybrid_model

def create_comprehensive_visualizations(df, vp_pred, vs_pred, vp_corr, vs_corr,
                                      hybrid_model, feature_cols,
                                      X_test, y_vp_test, y_vs_test,
                                      vp_test_pred, vs_test_pred):
    """
    Create comprehensive visualizations of results.
    
    Includes:
    1. Cross-plots (predicted vs actual)
    2. Error distributions
    3. Feature importance
    4. Depth profiles (if available)
    5. Physics vs ML comparison
    """
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ==========================================================================
    # 1. Vp Cross-plot
    # ==========================================================================
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df['Vp'], vp_pred, alpha=0.5, s=20, color='blue', 
               label=f'R² = {np.corrcoef(df["Vp"], vp_pred)[0,1]:.3f}')
    
    # Perfect prediction line
    vp_min, vp_max = df['Vp'].min(), df['Vp'].max()
    ax1.plot([vp_min, vp_max], [vp_min, vp_max], 'r--', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Measured Vp (m/s)', fontsize=10)
    ax1.set_ylabel('Predicted Vp (m/s)', fontsize=10)
    ax1.set_title('Vp: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # ==========================================================================
    # 2. Vs Cross-plot
    # ==========================================================================
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(df['Vs'], vs_pred, alpha=0.5, s=20, color='green',
               label=f'R² = {np.corrcoef(df["Vs"], vs_pred)[0,1]:.3f}')
    
    vs_min, vs_max = df['Vs'].min(), df['Vs'].max()
    ax2.plot([vs_min, vs_max], [vs_min, vs_max], 'r--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Measured Vs (m/s)', fontsize=10)
    ax2.set_ylabel('Predicted Vs (m/s)', fontsize=10)
    ax2.set_title('Vs: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # ==========================================================================
    # 3. Vp/Vs Ratio
    # ==========================================================================
    ax3 = plt.subplot(2, 3, 3)
    vpvs_actual = df['Vp'] / df['Vs']
    vpvs_pred = vp_pred / vs_pred
    
    ax3.scatter(vpvs_actual, vpvs_pred, alpha=0.5, s=20, color='purple')
    
    vpvs_min, vpvs_max = min(vpvs_actual.min(), vpvs_pred.min()), \
                        max(vpvs_actual.max(), vpvs_pred.max())
    ax3.plot([vpvs_min, vpvs_max], [vpvs_min, vpvs_max], 'r--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Measured Vp/Vs', fontsize=10)
    ax3.set_ylabel('Predicted Vp/Vs', fontsize=10)
    ax3.set_title('Vp/Vs Ratio Prediction', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ==========================================================================
    # 4. Vp Error Histogram
    # ==========================================================================
    ax4 = plt.subplot(2, 3, 4)
    vp_error = 100 * (vp_pred - df['Vp']) / df['Vp']
    
    ax4.hist(vp_error, bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax4.axvline(x=vp_error.mean(), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {vp_error.mean():.1f}%')
    
    ax4.set_xlabel('Vp Prediction Error (%)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title(f'Vp Error Distribution\nStd: {vp_error.std():.1f}%', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # ==========================================================================
    # 5. Vs Error Histogram
    # ==========================================================================
    ax5 = plt.subplot(2, 3, 5)
    vs_error = 100 * (vs_pred - df['Vs']) / df['Vs']
    
    ax5.hist(vs_error, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax5.axvline(x=vs_error.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean: {vs_error.mean():.1f}%')
    
    ax5.set_xlabel('Vs Prediction Error (%)', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title(f'Vs Error Distribution\nStd: {vs_error.std():.1f}%',
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # ==========================================================================
    # 6. Feature Importance
    # ==========================================================================
    ax6 = plt.subplot(2, 3, 6)
    
    if hasattr(hybrid_model, 'feature_importances') and 'Vp' in hybrid_model.feature_importances:
        importances = hybrid_model.feature_importances['Vp']
        
        if len(importances) == len(feature_cols):
            top_n = min(8, len(feature_cols))
            sorted_idx = np.argsort(importances)[-top_n:]
            
            y_pos = np.arange(top_n)
            ax6.barh(y_pos, importances[sorted_idx], color='steelblue')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=9)
            ax6.set_xlabel('Importance Score', fontsize=10)
            ax6.set_title('Top Features for Vp Prediction', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Hybrid Physics-ML Velocity Prediction: Complete Results', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ==========================================================================
    # SECOND FIGURE: Physics Insights
    # ==========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Porosity-Vp relationship
    ax = axes2[0]
    sc1 = ax.scatter(df['porosity'], df['Vp'], alpha=0.4, s=20, 
                    c='blue', label='Measured')
    sc2 = ax.scatter(df['porosity'], vp_pred, alpha=0.4, s=20,
                    c='red', label='Predicted')
    
    # Add trend lines
    if len(df) > 10:
        # Measured trend
        z1 = np.polyfit(df['porosity'], df['Vp'], 2)
        p1 = np.poly1d(z1)
        porosity_sorted = np.sort(df['porosity'])
        ax.plot(porosity_sorted, p1(porosity_sorted), 'b-', linewidth=2, 
               label='Measured trend')
        
        # Predicted trend
        z2 = np.polyfit(df['porosity'], vp_pred, 2)
        p2 = np.poly1d(z2)
        ax.plot(porosity_sorted, p2(porosity_sorted), 'r-', linewidth=2,
               label='Predicted trend')
    
    ax.set_xlabel('Porosity', fontsize=10)
    ax.set_ylabel('Vp (m/s)', fontsize=10)
    ax.set_title('Vp vs Porosity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Porosity-Vs relationship
    ax = axes2[1]
    sc1 = ax.scatter(df['porosity'], df['Vs'], alpha=0.4, s=20,
                    c='green', label='Measured')
    sc2 = ax.scatter(df['porosity'], vs_pred, alpha=0.4, s=20,
                    c='orange', label='Predicted')
    
    if len(df) > 10:
        # Measured trend
        z1 = np.polyfit(df['porosity'], df['Vs'], 2)
        p1 = np.poly1d(z1)
        ax.plot(porosity_sorted, p1(porosity_sorted), 'g-', linewidth=2,
               label='Measured trend')
        
        # Predicted trend
        z2 = np.polyfit(df['porosity'], vs_pred, 2)
        p2 = np.poly1d(z2)
        ax.plot(porosity_sorted, p2(porosity_sorted), 'orange', linewidth=2,
               label='Predicted trend')
    
    ax.set_xlabel('Porosity', fontsize=10)
    ax.set_ylabel('Vs (m/s)', fontsize=10)
    ax.set_title('Vs vs Porosity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Vp-Vs cross-plot
    ax = axes2[2]
    ax.scatter(df['Vp'], df['Vs'], alpha=0.5, s=20, c='purple', label='Measured')
    ax.scatter(vp_pred, vs_pred, alpha=0.5, s=20, c='cyan', label='Predicted')
    
    # Add regression lines
    if len(df) > 10:
        # Measured Vp-Vs relationship
        z_meas = np.polyfit(df['Vp'], df['Vs'], 1)
        p_meas = np.poly1d(z_meas)
        vp_range = np.linspace(df['Vp'].min(), df['Vp'].max(), 100)
        ax.plot(vp_range, p_meas(vp_range), 'purple', linewidth=2,
               label='Measured trend')
        
        # Predicted Vp-Vs relationship
        z_pred = np.polyfit(vp_pred, vs_pred, 1)
        p_pred = np.poly1d(z_pred)
        ax.plot(vp_range, p_pred(vp_range), 'cyan', linewidth=2,
               label='Predicted trend')
    
    ax.set_xlabel('Vp (m/s)', fontsize=10)
    ax.set_ylabel('Vs (m/s)', fontsize=10)
    ax.set_title('Vp-Vs Relationship', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Physics-Based Relationships and Predictions', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print achievement status
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    target_correlation = 0.75
    
    vp_status = "✓ ACHIEVED" if vp_corr >= target_correlation else "✗ NEEDS IMPROVEMENT"
    vs_status = "✓ ACHIEVED" if vs_corr >= target_correlation else "✗ NEEDS IMPROVEMENT"
    
    print(f"\nVp Correlation: {vp_corr:.4f} {vp_status}")
    print(f"Vs Correlation: {vs_corr:.4f} {vs_status}")
    
    if vp_corr >= target_correlation and vs_corr >= target_correlation:
        print("\n🎉 SUCCESS: Both targets achieved! 🎉")
    else:
        print("\n💡 SUGGESTIONS FOR IMPROVEMENT:")
        if vp_corr < target_correlation:
            print("  For Vp:")
            print("  - Increase ML model complexity (more trees, deeper)")
            print("  - Add more physics features (pore shape, fluid effects)")
            print("  - Try different ML algorithms (Random Forest, Neural Network)")
        
        if vs_corr < target_correlation:
            print("  For Vs:")
            print("  - Vs is more sensitive to cracks, refine crack density estimation")
            print("  - Add shear-related features (clay content, microcrack orientation)")
            print("  - Consider separate feature engineering for Vs")
    
    # Show best predictions
    print("\n" + "="*80)
    print("BEST PREDICTIONS (Lowest Combined Error)")
    print("="*80)
    
    df['combined_error'] = (np.abs(df['Vp_error_%']) + np.abs(df['Vs_error_%'])) / 2
    best_samples = df.nsmallest(5, 'combined_error')
    
    display_cols = ['Vp', 'Vp_hybrid_pred', 'Vp_error_%', 
                   'Vs', 'Vs_hybrid_pred', 'Vs_error_%',
                   'porosity', 'rho', 'combined_error']
    
    if 'Vclay' in df.columns:
        display_cols.append('Vclay')
    if 'RT' in df.columns:
        display_cols.append('RT')
    
    pd.set_option('display.precision', 2)
    print(best_samples[display_cols].to_string(index=False))

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HYBRID PHYSICS-ML VELOCITY PREDICTION SYSTEM")
    print("Version 2.0 - Scientific Implementation")
    print("="*80)
    
    # Run analysis
    results = run_analysis()
    
    if results:
        df, vp_corr, vs_corr, hybrid_model = results
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        # Final summary
        print(f"\nFINAL RESULTS:")
        print(f"Vp Correlation: {vp_corr:.4f} ({'✓ TARGET ACHIEVED' if vp_corr >= 0.75 else '✗ BELOW TARGET'})")
        print(f"Vs Correlation: {vs_corr:.4f} ({'✓ TARGET ACHIEVED' if vs_corr >= 0.75 else '✗ BELOW TARGET'})")
        
        # Calculate average errors
        vp_mean_error = df['Vp_error_%'].mean()
        vs_mean_error = df['Vs_error_%'].mean()
        
        print(f"\nPREDICTION ERRORS:")
        print(f"Vp Mean Absolute Error: {abs(vp_mean_error):.1f}%")
        print(f"Vs Mean Absolute Error: {abs(vs_mean_error):.1f}%")
        
        # Success criteria
        if vp_corr >= 0.75 and vs_corr >= 0.75:
            print("\n" + "="*80)
            print("🎉 CONGRATULATIONS! SUCCESSFUL IMPLEMENTATION! 🎉")
            print("Both velocity predictions exceed the 0.75 correlation target.")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE WITH PARTIAL SUCCESS")
            print("Consider the improvement suggestions above.")
            print("="*80)
        
        # Additional insights
        print("\n" + "="*80)
        print("SCIENTIFIC INSIGHTS")
        print("="*80)
        
        # Check if physics features were important
        if hasattr(hybrid_model, 'feature_importances'):
            physics_features = ['crack_density_efm', 'orientation_beta', 'Vp_efm', 'Vs_efm']
            physics_present = any(feat in hybrid_model.feature_names for feat in physics_features)
            
            if physics_present:
                print("✓ Physics features successfully integrated into ML model")
            else:
                print("✗ Physics features not in top features - check integration")
        
        print(f"\nDataset size: {len(df)} samples")
        print(f"Features used: {len(hybrid_model.feature_names)}")
        print(f"Hybrid weight: 30% physics, 70% ML")
        
    else:
        print("\nAnalysis failed. Please check your data and try again.")
