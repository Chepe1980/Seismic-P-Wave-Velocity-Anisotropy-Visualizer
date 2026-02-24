import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import convolve
from scipy.linalg import inv
import pandas as pd
import io
from typing import List, Dict, Tuple, Optional
import matplotlib.gridspec as gridspec
from scipy import stats
import os
from io import BytesIO
import base64

# ==============================================
# Page configuration
# ==============================================
st.set_page_config(
    page_title="Integrated AVAZ & Fracture Modeling",
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

# ==============================================
# APP 1: AVAZ Modeling Functions
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

def create_3d_plot(x, y, z, vp, colormap='Viridis'):
    """Create interactive 3D velocity surface plot"""
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=vp,
            colorscale=colormap,
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

def morlet_wavelet(t, s=1.0, w=5.0):
    """Morlet wavelet function"""
    return np.pi**(-0.25) * np.exp(1j * w * t) * np.exp(-0.5 * t**2)

def cwt_analysis(signal_data, scales):
    """Perform Continuous Wavelet Transform on signal data"""
    n_samples = len(signal_data)
    n_scales = len(scales)
    
    # Create empty array for CWT results
    cwt_matrix = np.zeros((n_scales, n_samples))
    
    # Perform CWT for each scale
    for i, scale in enumerate(scales):
        # Ensure scale is at least 1
        scale = max(1, scale)
        
        # Create wavelet with proper length
        wavelet_length = min(10 * scale, n_samples)
        if wavelet_length % 2 == 0:
            wavelet_length += 1  # Make it odd for symmetry
        
        t = np.linspace(-scale*3, scale*3, wavelet_length)
        wavelet = morlet_wavelet(t/scale)
        wavelet = wavelet.real  # Take real part for analysis
        
        # Normalize wavelet
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        
        # Convolution with signal (mode='same' returns same length)
        conv_result = np.convolve(signal_data, wavelet, mode='same')
        
        # Ensure we have the right length
        if len(conv_result) == n_samples:
            cwt_matrix[i, :] = conv_result
        else:
            # Trim or pad to match expected length
            if len(conv_result) > n_samples:
                cwt_matrix[i, :] = conv_result[:n_samples]
            else:
                cwt_matrix[i, :n_samples] = conv_result
    
    return cwt_matrix  # Return magnitude

def plot_horizontal_angles_seismic_plotly(results, seismic_cmap, diff_cmap):
    """Plot seismic gathers with incidence angles arranged horizontally using Plotly"""
    st.header("Seismic Gathers - Horizontal Angle Arrangement")
    st.markdown("All incidence angles (0-50°) arranged horizontally, each showing azimuth 0-360°")
    
    # Get dimensions
    n_angles = len(results['incidence_angles'])
    time_samples = results['seismic_orig'][0].shape[0]
    n_azimuths = results['seismic_orig'][0].shape[1]
    
    # Original Seismic - Horizontal arrangement with reflectivity plots below
    st.subheader("Original Seismic - Horizontal Angle Arrangement with Reflectivity vs Azimuth")
    
    # Create subplot figure with 2 rows for each angle (heatmap on top, reflectivity below)
    fig_orig = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling in heatmaps
    vmax_global = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.abs(results['seismic_orig'][angle_idx]).max()
        vmax_global = max(vmax_global, vmax_angle)
    
    # Add heatmaps in top row and reflectivity plots in bottom row
    for angle_idx in range(n_angles):
        seismic_data = results['seismic_orig'][angle_idx]
        reflectivity_data = results['reflectivity_orig'][angle_idx, :]
        
        # Add heatmap in top row
        fig_orig.add_trace(
            go.Heatmap(
                z=seismic_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=seismic_cmap,
                zmin=-vmax_global,
                zmax=vmax_global,
                showscale=angle_idx == n_angles-1,  # Only show colorbar for last plot
                colorbar=dict(title="Amplitude", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity vs azimuth plot in bottom row
        fig_orig.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_data,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_orig.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_orig.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_orig.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity plots)
        fig_orig.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_orig.update_yaxes(title_text="Reflectivity", row=2, col=1)
        else:
            fig_orig.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity plots
        fig_orig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_orig.update_layout(
        height=900,
        title_text="Original Seismic - Heatmaps (top) with Reflectivity vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_orig, use_container_width=True)
    
    # Fluid-Substituted Seismic - Horizontal arrangement with reflectivity plots below
    st.subheader("Fluid-Substituted Seismic - Horizontal Angle Arrangement with Reflectivity vs Azimuth")
    
    fig_sub = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_sub = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.abs(results['seismic_sub'][angle_idx]).max()
        vmax_global_sub = max(vmax_global_sub, vmax_angle)
    
    # Add heatmaps in top row and reflectivity plots in bottom row
    for angle_idx in range(n_angles):
        seismic_data = results['seismic_sub'][angle_idx]
        reflectivity_data = results['reflectivity_sub'][angle_idx, :]
        
        # Add heatmap in top row
        fig_sub.add_trace(
            go.Heatmap(
                z=seismic_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=seismic_cmap,
                zmin=-vmax_global_sub,
                zmax=vmax_global_sub,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="Amplitude", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity vs azimuth plot in bottom row
        fig_sub.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_data,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_sub.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_sub.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_sub.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity plots)
        fig_sub.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_sub.update_yaxes(title_text="Reflectivity", row=2, col=1)
        else:
            fig_sub.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity plots
        fig_sub.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_sub.update_layout(
        height=900,
        title_text="Fluid-Substituted Seismic - Heatmaps (top) with Reflectivity vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_sub, use_container_width=True)
    
    # Difference plots - Horizontal arrangement with reflectivity difference below
    st.subheader("Seismic Difference - Horizontal Angle Arrangement with Reflectivity Difference")
    
    fig_diff = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity Diff at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_diff = 0
    diff_data_all = []
    for angle_idx in range(n_angles):
        diff_data = results['seismic_sub'][angle_idx] - results['seismic_orig'][angle_idx]
        diff_data_all.append(diff_data)
        vmax_angle = np.abs(diff_data).max()
        vmax_global_diff = max(vmax_global_diff, vmax_angle)
    
    # Calculate reflectivity difference
    reflectivity_diff = results['reflectivity_sub'] - results['reflectivity_orig']
    max_reflectivity_diff = np.max(np.abs(reflectivity_diff))
    
    # Add heatmaps in top row and reflectivity difference plots in bottom row
    for angle_idx in range(n_angles):
        diff_data = diff_data_all[angle_idx]
        reflectivity_diff_data = reflectivity_diff[angle_idx, :]
        
        # Add heatmap in top row
        fig_diff.add_trace(
            go.Heatmap(
                z=diff_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=diff_cmap,
                zmin=-vmax_global_diff,
                zmax=vmax_global_diff,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="Amplitude Diff", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity difference vs azimuth plot in bottom row
        fig_diff.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_diff_data,
                mode='lines+markers',
                line=dict(color='purple', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity Diff {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_diff.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_diff.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_diff.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity difference plots)
        fig_diff.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_diff.update_yaxes(title_text="Reflectivity Diff", row=2, col=1)
        else:
            fig_diff.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity difference plots
        fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_diff.update_layout(
        height=900,
        title_text="Seismic Difference - Heatmaps (top) with Reflectivity Difference vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_diff, use_container_width=True)

def plot_horizontal_angles_cwt_plotly(results, cwt_scale, cwt_cmap, diff_cmap):
    """Plot CWT analysis with incidence angles arranged horizontally using Plotly"""
    st.header("CWT Analysis - Horizontal Angle Arrangement")
    st.markdown(f"Continuous Wavelet Transform at Scale {cwt_scale} - All incidence angles arranged horizontally")
    
    # Get dimensions
    n_angles = len(results['incidence_angles'])
    time_samples = results['seismic_orig'][0].shape[0]
    n_azimuths = results['seismic_orig'][0].shape[1]
    
    # Perform CWT for all angles and azimuths
    with st.spinner(f"Performing CWT analysis at scale {cwt_scale}..."):
        # Initialize arrays for CWT results
        cwt_orig = []
        cwt_sub = []
        
        # Process each angle
        for angle_idx in range(n_angles):
            # Get seismic data for this angle
            seismic_orig_angle = results['seismic_orig'][angle_idx]
            seismic_sub_angle = results['seismic_sub'][angle_idx]
            
            # Initialize arrays for CWT results at selected scale
            cwt_orig_at_scale = np.zeros_like(seismic_orig_angle)
            cwt_sub_at_scale = np.zeros_like(seismic_sub_angle)
            
            # Perform CWT for each azimuth trace
            for az_idx in range(n_azimuths):
                # Original seismic trace
                trace_orig = seismic_orig_angle[:, az_idx]
                if len(trace_orig) > 0:
                    cwt_result_orig = cwt_analysis(trace_orig, [cwt_scale])
                    cwt_orig_at_scale[:, az_idx] = cwt_result_orig[0, :]
                
                # Fluid-substituted seismic trace
                trace_sub = seismic_sub_angle[:, az_idx]
                if len(trace_sub) > 0:
                    cwt_result_sub = cwt_analysis(trace_sub, [cwt_scale])
                    cwt_sub_at_scale[:, az_idx] = cwt_result_sub[0, :]
            
            # Store results for this angle
            cwt_orig.append(cwt_orig_at_scale)
            cwt_sub.append(cwt_sub_at_scale)
    
    # Original CWT - Horizontal arrangement
    st.subheader(f"Original CWT at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_orig = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Set global vmax for consistent color scaling
    vmax_global = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.max(cwt_orig[angle_idx])
        vmax_global = max(vmax_global, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        cwt_data = cwt_orig[angle_idx]
        
        fig_cwt_orig.add_trace(
            go.Heatmap(
                z=cwt_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=cwt_cmap,
                zmin=0,
                zmax=vmax_global,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_orig.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_orig.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_orig.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_orig.update_layout(
        height=500,
        title_text=f"Original CWT at Scale {cwt_scale} - All Incidence Angles (0-50°)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_orig, use_container_width=True)
    
    # Fluid-Substituted CWT - Horizontal arrangement
    st.subheader(f"Fluid-Substituted CWT at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_sub = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_sub = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.max(cwt_sub[angle_idx])
        vmax_global_sub = max(vmax_global_sub, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        cwt_data = cwt_sub[angle_idx]
        
        fig_cwt_sub.add_trace(
            go.Heatmap(
                z=cwt_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=cwt_cmap,
                zmin=0,
                zmax=vmax_global_sub,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_sub.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_sub.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_sub.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_sub.update_layout(
        height=500,
        title_text=f"Fluid-Substituted CWT at Scale {cwt_scale} - All Incidence Angles (0-50°)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_sub, use_container_width=True)
    
    # CWT Difference plots - Horizontal arrangement
    st.subheader(f"CWT Difference at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_diff = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Calculate differences and set global vmax for consistent color scaling
    vmax_global_diff = 0
    cwt_diff = []
    
    for angle_idx in range(n_angles):
        diff_data = cwt_sub[angle_idx] - cwt_orig[angle_idx]
        cwt_diff.append(diff_data)
        vmax_angle = np.max(np.abs(diff_data))
        vmax_global_diff = max(vmax_global_diff, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        diff_data = cwt_diff[angle_idx]
        
        fig_cwt_diff.add_trace(
            go.Heatmap(
                z=diff_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=diff_cmap,
                zmin=-vmax_global_diff,
                zmax=vmax_global_diff,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Diff", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_diff.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_diff.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_diff.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_diff.update_layout(
        height=500,
        title_text=f"CWT Difference at Scale {cwt_scale} (Fluid-Substituted - Original)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_diff, use_container_width=True)
    
    # Return CWT results for further analysis
    return {
        'cwt_orig': cwt_orig,
        'cwt_sub': cwt_sub,
        'cwt_diff': cwt_diff
    }

def pwave_anisotropy_section_plotly(epsilon, delta, vp0, plot_cmap):
    """P-Wave Velocity Anisotropy Visualizer using Plotly"""
    st.header("P-Wave Velocity Anisotropy Visualizer")
    st.markdown("Explore how Thomsen parameters (ε, δ) affect P-wave velocity anisotropy.")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        epsilon = st.number_input(
            "ε (Epsilon)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=float(epsilon),
            step=0.01,
            key="epsilon_ani"
        )
        delta = st.number_input(
            "δ (Delta)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=float(delta),
            step=0.01,
            key="delta_ani"
        )
        vp0 = st.number_input(
            "Vp₀ (m/s)", 
            min_value=1000.0,
            max_value=8000.0,
            value=float(vp0),
            key="vp0_ani"
        )
        show_3d = st.checkbox("Show 3D Visualization", True, key="show3d_ani")

    with col2:
        # Calculate Vp for 2D plot
        theta = np.linspace(0, 90, 90) * np.pi / 180
        Vp = vp0 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 + epsilon * (np.sin(theta))**4)
        
        # Convert to Cartesian coordinates for 2D plot
        Vpx = Vp * np.sin(theta)
        Vpy = Vp * np.cos(theta)
        
        # 2D polar plot using Plotly
        fig_2d = go.Figure()
        
        # Add trace
        fig_2d.add_trace(go.Scatter(
            x=Vpx, y=Vpy,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f"ε={epsilon:.3f}, δ={delta:.3f}"
        ))
        
        # Update layout
        fig_2d.update_layout(
            title="P-Wave Velocity Anisotropy",
            xaxis_title="Vpx [m/s]",
            yaxis_title="Vpy [m/s]",
            height=600,
            width=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(range=[0, 1.5*vp0]),
            yaxis=dict(range=[0, 1.5*vp0]),
            hovermode='closest'
        )
        
        # Add grid
        fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_2d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 3D plot if enabled
        if show_3d:
            st.subheader("3D Velocity Surface")
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
            
            fig_3d = create_3d_plot(x, y, z, Vp_3d)
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
        
        # Log the depth ranges being used
        st.info("📊 **Depth Ranges Selected for Modeling:**")
        for i, (min_depth, max_depth) in enumerate(depth_ranges, 1):
            layer_name = ["Upper", "Target", "Lower"][i-1]
            st.write(f"  - **Layer {i} ({layer_name})**: {min_depth:.1f} m to {max_depth:.1f} m")
            
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
            
            # Show the extracted values for each layer
            with st.expander(f"📋 Layer {i} ({layer_name}) Extracted Values"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Vp:** {params[f'vp{i}']:.1f} m/s")
                    st.write(f"**Vs:** {params[f'vs{i}']:.1f} m/s")
                    st.write(f"**Density:** {params[f'd{i}']:.3f} g/cc")
                with col2:
                    st.write(f"**ε (Epsilon):** {params[f'e{i}']:.4f}")
                    st.write(f"**δ (Delta):** {params[f'dlt{i}']:.4f}")
                    st.write(f"**γ (Gamma):** {params[f'g{i}']:.4f}")
        
        return params
    
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None

def plot_depth_ranges_plotly(depth_ranges, min_depth, max_depth):
    """Visualize the selected depth ranges using Plotly"""
    fig = go.Figure()
    
    # Colors for each layer
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    labels = ['Upper Layer (1)', 'Target Layer (2)', 'Lower Layer (3)']
    
    # Create horizontal bars for each layer
    for i, ((min_d, max_d), color) in enumerate(zip(depth_ranges, colors)):
        fig.add_trace(go.Bar(
            y=[labels[i]],
            x=[max_d - min_d],
            base=[min_d],
            orientation='h',
            marker=dict(color=color),
            name=labels[i],
            text=[f'{min_d:.1f}-{max_d:.1f}'],
            textposition='inside',
            textfont=dict(color='white', size=12, family="Arial Black"),
            hoverinfo='text',
            hovertext=f'{labels[i]}<br>Depth: {min_d:.1f} - {max_d:.1f} m'
        ))
    
    fig.update_layout(
        title='Selected Depth Ranges',
        xaxis_title='Depth (m)',
        yaxis_title='Layer',
        height=300,
        barmode='overlay',
        showlegend=True,
        legend=dict(x=1.02, y=1),
        xaxis=dict(range=[min_depth, max_depth]),
        yaxis=dict(showticklabels=False),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_modeling(params, enable_fluid_sub, seismic_cmap, selected_angle, azimuth_step, freq):
    """Run the modeling for ALL angles (0-50°) and azimuths (0-360°)"""
    with st.spinner("Computing models..."):
        # Original properties
        vp_orig = [params['vp1'], params['vp2'], params['vp3']]
        vs_orig = [params['vs1'], params['vs2'], params['vs3']]
        d_orig = [params['d1'], params['d2'], params['d3']]
        e_orig = [params['e1'], params['e2'], params['e3']]
        g_orig = [params['g1'], params['g2'], params['g3']]
        dlt_orig = [params['dlt1'], params['dlt2'], params['dlt3']]
        
        # Fluid substituted properties
        vp_sub = vp_orig.copy()
        vs_sub = vs_orig.copy()
        d_sub = d_orig.copy()
        e_sub = e_orig.copy()
        g_sub = g_orig.copy()
        dlt_sub = dlt_orig.copy()
        
        if enable_fluid_sub:
            K_orig, G_orig = velocity_to_moduli(params['vp2'], params['vs2'], params['d2'])
            K_sat, G_sat, delta_sat, gamma_sat = brown_korringa_substitution(
                params['Km']*1e9, params['Gm']*1e9, 
                K_orig, G_orig,
                params['Kf']*1e9, 
                params['phi'], 
                params['dlt2'], params['g2']
            )
            new_density = params['d2'] + params['phi']*(params['new_fluid_density'] - 1.0)
            Vp_new, Vs_new = moduli_to_velocity(K_sat, G_sat, new_density)
            
            vp_sub[1] = Vp_new
            vs_sub[1] = Vs_new
            d_sub[1] = new_density
            dlt_sub[1] = delta_sat
            g_sub[1] = gamma_sat
        
        # Compute for ALL angles (0-50°) and azimuths (0-360°)
        incidence_angles = np.linspace(0, 50, 11)  # 11 steps from 0-50°
        azimuths = np.arange(0, 361, azimuth_step)
        
        # Compute reflectivity (2D array: angles × azimuths)
        reflectivity_orig = np.zeros((len(incidence_angles), len(azimuths)))
        reflectivity_sub = np.zeros((len(incidence_angles), len(azimuths)))
        
        for i, theta in enumerate(incidence_angles):
            theta_rad = np.radians(theta)
            for j, az in enumerate(azimuths):
                reflectivity_orig[i,j] = calculate_reflectivity(
                    vp_orig, vs_orig, d_orig, e_orig, g_orig, dlt_orig, theta_rad, az
                )
                reflectivity_sub[i,j] = calculate_reflectivity(
                    vp_sub, vs_sub, d_sub, e_sub, g_sub, dlt_sub, theta_rad, az
                )
        
        # Generate synthetic seismic for all angles
        n_samples = 150
        wavelet = ricker_wavelet(freq, 0.08, 0.001)
        center_sample = n_samples//2 + len(wavelet)//2
        
        seismic_orig = []
        seismic_sub = []
        
        for i in range(len(incidence_angles)):
            R = np.zeros((n_samples, len(azimuths)))
            R[n_samples//2, :] = reflectivity_orig[i,:]
            seismic = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
            seismic_orig.append(seismic[center_sample-75:center_sample+75, :])
            
            R = np.zeros((n_samples, len(azimuths)))
            R[n_samples//2, :] = reflectivity_sub[i,:]
            seismic = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
            seismic_sub.append(seismic[center_sample-75:center_sample+75, :])
        
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
            'dlt_sub': dlt_sub
        }

def display_results_plotly(results, seismic_cmap, diff_cmap, cwt_cmap, selected_angle):
    """Display modeling results with 0-50° 3D AVAZ comparison using Plotly"""
    # Find nearest angle index within 0-50° range
    angle_idx = np.argmin(np.abs(results['incidence_angles'] - min(selected_angle, 50)))
    actual_angle = results['incidence_angles'][angle_idx]
    
    st.header("3D AVAZ Response Comparison (0-50° Incidence)")
    col1, col2 = st.columns(2)
    
    # Get min/max for consistent color scaling
    zmin = min(np.min(results['reflectivity_orig']), np.min(results['reflectivity_sub']))
    zmax = max(np.max(results['reflectivity_orig']), np.max(results['reflectivity_sub']))
    
    with col1:
        fig_orig = go.Figure(data=[go.Surface(
            z=results['reflectivity_orig'],
            x=results['azimuths'],
            y=results['incidence_angles'],
            colorscale=seismic_cmap,
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
            title="Original Response (0-50°)",
            height=500
        )
        st.plotly_chart(fig_orig, use_container_width=True)
    
    with col2:
        fig_sub = go.Figure(data=[go.Surface(
            z=results['reflectivity_sub'],
            x=results['azimuths'],
            y=results['incidence_angles'],
            colorscale=seismic_cmap,
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
            title="Fluid-Substituted Response (0-50°)",
            height=500
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    
    # 2. 2D Comparison at nearest angle using Plotly
    st.header(f"2D Comparison at {actual_angle:.1f}° Incidence (Closest to Selected {selected_angle}°)")
    
    fig_2d = go.Figure()
    
    # Add original reflectivity trace
    fig_2d.add_trace(go.Scatter(
        x=results['azimuths'],
        y=results['reflectivity_orig'][angle_idx, :],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=2)
    ))
    
    # Add fluid-substituted reflectivity trace
    fig_2d.add_trace(go.Scatter(
        x=results['azimuths'],
        y=results['reflectivity_sub'][angle_idx, :],
        mode='lines',
        name='Fluid-Substituted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_2d.update_layout(
        title=f'AVAZ Reflectivity at {actual_angle:.1f}° Incidence',
        xaxis_title='Azimuth (degrees)',
        yaxis_title='Reflectivity',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    # Add grid
    fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_2d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig_2d, use_container_width=True)
    
    # 3. Polar View Comparison using Plotly
    st.header(f"Polar View Comparison at {actual_angle:.1f}° Incidence")
    
    fig_polar = go.Figure()
    
    # Add original reflectivity trace
    fig_polar.add_trace(go.Scatterpolar(
        r=results['reflectivity_orig'][angle_idx, :],
        theta=results['azimuths'],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=2)
    ))
    
    # Add fluid-substituted reflectivity trace
    fig_polar.add_trace(go.Scatterpolar(
        r=results['reflectivity_sub'][angle_idx, :],
        theta=results['azimuths'],
        mode='lines',
        name='Fluid-Substituted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_polar.update_layout(
        title=f'Polar AVAZ Response at {actual_angle:.1f}° Incidence',
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction="clockwise")
        ),
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_polar, use_container_width=True)
    
    # 4. Horizontal Arrangement of Seismic Gathers using Plotly
    plot_horizontal_angles_seismic_plotly(results, seismic_cmap, diff_cmap)
    
    # 5. Horizontal Arrangement of CWT Analysis using Plotly
    # Add CWT scale selection in the main display
    st.header("CWT Analysis Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        cwt_scale = st.slider(
            "CWT Scale (Higher = Lower Frequency)", 
            min_value=1, 
            max_value=20, 
            value=10,
            help="Scale parameter for Continuous Wavelet Transform"
        )
    with col2:
        st.markdown("**Scale Information:**")
        st.markdown("- Scale 1: Highest frequency details")
        st.markdown("- Scale 10: Mid-frequency features")
        st.markdown("- Scale 20: Lowest frequency trends")
    with col3:
        st.markdown("**Current Colormaps:**")
        st.markdown(f"- Seismic/CWT: {seismic_cmap}")
        st.markdown(f"- Difference: {diff_cmap}")
    
    # Call the new horizontal CWT plotting function
    cwt_results = plot_horizontal_angles_cwt_plotly(results, cwt_scale, cwt_cmap, diff_cmap)
    
    # 6. Additional CWT Visualizations using Plotly
    st.header("Additional CWT Visualizations")
    
    # Frequency-Scale Analysis at middle angle
    st.subheader("Frequency-Scale Analysis")
    
    # Analyze at middle azimuth and middle angle
    mid_azimuth_idx = len(results['azimuths']) // 2
    mid_angle_idx = len(results['incidence_angles']) // 2
    mid_angle = results['incidence_angles'][mid_angle_idx]
    
    # Get traces for analysis
    orig_trace = results['seismic_orig'][mid_angle_idx][:, mid_azimuth_idx]
    sub_trace = results['seismic_sub'][mid_angle_idx][:, mid_azimuth_idx]
    
    # Perform full CWT with all scales
    scales_full = np.arange(1, 21, 1)
    cwt_orig_full = cwt_analysis(orig_trace, scales_full)
    cwt_sub_full = cwt_analysis(sub_trace, scales_full)
    
    # Create subplots for scale-time analysis
    fig_scale_time = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Original: CWT Scale-Time at {mid_angle}°', 
                       f'Fluid-Substituted: CWT Scale-Time at {mid_angle}°'],
        horizontal_spacing=0.1
    )
    
    # Original CWT scale-time
    fig_scale_time.add_trace(
        go.Heatmap(
            z=cwt_orig_full,
            x=np.arange(len(orig_trace)),
            y=scales_full,
            colorscale=cwt_cmap,
            showscale=True,
            colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle", x=0.46)
        ),
        row=1, col=1
    )
    
    # Fluid-substituted CWT scale-time
    fig_scale_time.add_trace(
        go.Heatmap(
            z=cwt_sub_full,
            x=np.arange(len(sub_trace)),
            y=scales_full,
            colorscale=cwt_cmap,
            showscale=True,
            colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle", x=1.02)
        ),
        row=1, col=2
    )
    
    # Update axes
    fig_scale_time.update_xaxes(title_text="Time Sample", row=1, col=1)
    fig_scale_time.update_xaxes(title_text="Time Sample", row=1, col=2)
    fig_scale_time.update_yaxes(
        title_text="Scale (Lower = Higher Frequency)", 
        autorange='reversed',
        row=1, col=1
    )
    fig_scale_time.update_yaxes(
        title_text="Scale (Lower = Higher Frequency)", 
        autorange='reversed',
        row=1, col=2
    )
    
    fig_scale_time.update_layout(
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_scale_time, use_container_width=True)
    
    # 7. Difference Analysis using Plotly
    st.header("Difference Analysis")
    
    reflectivity_diff = results['reflectivity_sub'] - results['reflectivity_orig']
    max_diff = np.max(np.abs(reflectivity_diff))
    
    # Difference matrix using Plotly
    fig_diff_matrix = go.Figure(data=[
        go.Heatmap(
            z=reflectivity_diff.T,
            x=results['incidence_angles'],
            y=results['azimuths'],
            colorscale=diff_cmap,
            zmin=-max_diff,
            zmax=max_diff,
            colorbar=dict(title="Reflectivity Difference", len=0.8, y=0.5, yanchor="middle")
        )
    ])
    
    fig_diff_matrix.update_layout(
        title='Reflectivity Difference (Fluid-Substituted - Original)',
        xaxis_title='Incidence Angle (degrees)',
        yaxis_title='Azimuth (degrees)',
        height=500
    )
    
    st.plotly_chart(fig_diff_matrix, use_container_width=True)
    
    # 3D Difference plot
    fig_diff_3d = go.Figure(data=[go.Surface(
        z=reflectivity_diff,
        x=results['azimuths'],
        y=results['incidence_angles'],
        colorscale=diff_cmap,
        cmin=-max_diff,
        cmax=max_diff
    )])
    fig_diff_3d.update_layout(
        scene=dict(
            xaxis_title='Azimuth (deg)',
            yaxis_title='Incidence Angle (deg)',
            zaxis_title='Reflectivity Difference',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        title="3D Reflectivity Difference",
        height=600
    )
    st.plotly_chart(fig_diff_3d, use_container_width=True)

def run_avaz_modeling_app():
    """Main function for AVAZ Modeling App (App 1)"""
    st.title("AVAZ Modeling with Brown-Korringa Fluid Substitution")
    
    # Initialize session state
    if 'modeling_mode' not in st.session_state:
        st.session_state.modeling_mode = "manual"
    if 'excel_data_processed' not in st.session_state:
        st.session_state.excel_data_processed = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'show_anisotropy_section' not in st.session_state:
        st.session_state.show_anisotropy_section = False
    
    # Modeling mode selection
    modeling_mode = st.sidebar.radio(
        "Modeling Mode",
        ["Manual Input", "Excel Import"],
        index=0 if st.session_state.modeling_mode == "manual" else 1
    )
    
    # Colormap selection in sidebar for all plots
    st.sidebar.header("Colormap Settings")
    
    # Main colormap for seismic and CWT plots - Set RdBu as default
    main_cmap = st.sidebar.selectbox(
        "Main Colormap (Seismic & CWT)",
        options=['RdBu', 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'rainbow', 
                 'turbo', 'portland', 'blackbody', 'electric', 'bluered'],
        index=0,  # RdBu as default
        help="Colormap for seismic amplitude and CWT magnitude plots - RdBu set as default"
    )
    
    # Difference colormap
    diff_cmap = st.sidebar.selectbox(
        "Difference Colormap",
        options=['RdBu', 'RdYlBu', 'RdYlGn', 'Picnic', 'Portland', 'Earth', 
                 'Electric', 'Viridis', 'Cividis', 'balance', 'delta', 'curl'],
        index=0,
        help="Colormap for difference plots"
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
                1, 70, 30, 1,
                help="Model will show results for this angle in 2D views"
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
            
            # Add button to show P-wave anisotropy section
            show_anisotropy = st.checkbox("Show P-Wave Anisotropy Section", False)
            
            if st.button("Run Modeling"):
                st.session_state.show_results = True
                st.session_state.show_anisotropy_section = show_anisotropy
                st.session_state.model_params = params
                st.session_state.enable_fluid_sub = enable_fluid_sub
                st.session_state.main_cmap = main_cmap
                st.session_state.diff_cmap = diff_cmap
                st.session_state.selected_angle = selected_angle
                st.session_state.azimuth_step = azimuth_step
                st.session_state.freq = freq
        
        # Main workspace content
        if st.session_state.show_results:
            # Show P-wave anisotropy section if checkbox was checked
            if st.session_state.show_anisotropy_section:
                st.markdown("---")
                pwave_anisotropy_section_plotly(
                    st.session_state.model_params['e2'],
                    st.session_state.model_params['dlt2'],
                    st.session_state.model_params['vp2'],
                    st.session_state.main_cmap
                )
                st.markdown("---")
            
            # Run the main modeling
            results = run_modeling(
                st.session_state.model_params,
                st.session_state.enable_fluid_sub,
                st.session_state.main_cmap,
                st.session_state.selected_angle,
                st.session_state.azimuth_step,
                st.session_state.freq
            )
            display_results_plotly(
                results,
                st.session_state.main_cmap,
                st.session_state.diff_cmap,
                st.session_state.main_cmap,
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
                        1, 70, 30, 1,
                        help="Model will show results for this angle in 2D views",
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
                    
                    show_anisotropy = st.checkbox("Show P-Wave Anisotropy Section", False, key="excel_show_anisotropy")
                    
                    if st.button("Run Modeling with Excel Data"):
                        st.session_state.excel_data_processed = True
                        st.session_state.show_anisotropy_section = show_anisotropy
                        st.session_state.uploaded_file = uploaded_file
                        st.session_state.enable_fluid_sub = enable_fluid_sub
                        st.session_state.main_cmap = main_cmap
                        st.session_state.diff_cmap = diff_cmap
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
            plot_depth_ranges_plotly(
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
                        
                        # Show P-wave anisotropy section if checkbox was checked
                        if st.session_state.show_anisotropy_section:
                            st.markdown("---")
                            pwave_anisotropy_section_plotly(params['e2'], params['dlt2'], params['vp2'], st.session_state.main_cmap)
                            st.markdown("---")
                        
                        # Run modeling
                        results = run_modeling(
                            params,
                            st.session_state.enable_fluid_sub,
                            st.session_state.main_cmap,
                            st.session_state.selected_angle,
                            st.session_state.azimuth_step,
                            st.session_state.freq
                        )
                        display_results_plotly(
                            results,
                            st.session_state.main_cmap,
                            st.session_state.diff_cmap,
                            st.session_state.main_cmap,
                            st.session_state.selected_angle
                        )
                
                except Exception as e:
                    st.error(f"Modeling error: {str(e)}")


# ==============================================
# APP 2: Fracture Model Analysis Functions
# ==============================================

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

def run_fracture_modeling_app():
    """Main function for Fracture Model Analysis App (App 2)"""
    st.markdown('<h1 class="main-header">📊 Fracture Model Analysis - Thomsen Parameters</h1>', unsafe_allow_html=True)
    
    # Initialize session state for results
    if 'fracture_results' not in st.session_state:
        st.session_state.fracture_results = None
    
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
                    st.session_state.fracture_results = results
                    
                    # Force a rerun to display results
                    st.rerun()
        else:
            st.error("Start depth must be less than end depth")
    
    # Display results if they exist in session state
    if st.session_state.fracture_results is not None:
        results = st.session_state.fracture_results
        
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
                float(results['DEPTH'].mean()),
                key="extract_depth_input"
            )
            
            if st.button("Extract Values", key="extract_button"):
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
        with st.expander("📊 View Summary Statistics"):
            stats_cols = ['HUDSON_EPS', 'LS_EPS', 'ORTHO_EPS1', 'ORTHO_EPS2', 
                         'MONO_EPSX', 'MONO_EPSY', 'HUDSON_GAM', 'LS_GAM']
            stats = results[stats_cols].describe()
            st.dataframe(stats)


# ============================================
# APP 3: VTI Model Analysis (Schoenberg Linear Slip Model)
# ============================================

# Theoretical functions from the paper
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

# Pressure-dependent weakness models
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

def select_depth_range_vti(df, depth_min, depth_max):
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

def run_vti_modeling_app():
    """Main function for Anisotropic Model Analysis App (App 3)"""
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
                    df_filtered = select_depth_range_vti(df_full, depth_min, depth_max)
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


# ==============================================
# Main App with Tabs
# ==============================================

def main():
    """Main function to run the integrated app with tabs"""
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 AVAZ Modeling with Fluid Substitution", 
        "🔬 Fracture Model Analysis (Thomsen Parameters)",
        "🌍 Anisotropic Model Analysis (Schoenberg Linear Slip)"
    ])
    
    with tab1:
        run_avaz_modeling_app()
    
    with tab2:
        run_fracture_modeling_app()
    
    with tab3:
        run_vti_modeling_app()


if __name__ == "__main__":
    main()
