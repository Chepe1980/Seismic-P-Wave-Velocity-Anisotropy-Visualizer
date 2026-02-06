"""
AZIMUTH AND INCIDENCE DATA CONCATENATION
Creates a single image from azimuth and incidence angle measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate
from PIL import Image
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE DATA PROCESSING CLASS
# ============================================================================

class AzimuthIncidenceConcatenator:
    """
    Main class for concatenating azimuth and incidence angle data into a single image.
    """
    
    def __init__(self):
        # Original data from your table
        self.original_incidence = [0, 15, 30, 45]  # Adjust these based on your actual data
        self.original_azimuth = np.linspace(0, 360, 11, endpoint=False)
        self.original_data = None
        self.processed_data = None
        self.azimuth_full = None
        self.incidence_full = None
        
    def load_original_data(self):
        """
        Load the data from your table format.
        Returns the data matrix (11 azimuths × 4 incidence angles)
        """
        # Based on your table structure:
        # Rows: Azimuth 1-11 (0° to 327.27°)
        # Columns: 4 incidence angles (0°, 15°, 30°, 45° in this example)
        
        data_matrix = np.array([
            # Format: [incidence_0°, incidence_15°, incidence_30°, incidence_45°]
            [100, 100, 100, 100],  # Azimuth 1 (0°)
            [100, 100, 100, 100],  # Azimuth 2 (32.73°)
            [100, 100, 100, 100],  # Azimuth 3 (65.45°)
            [100, 100, 102, 104],  # Azimuth 4 (98.18°)
            [105, 105, 105, 105],  # Azimuth 5 (130.91°)
            [105, 105, 105, 105],  # Azimuth 6 (163.64°)
            [105, 105, 105, 105],  # Azimuth 7 (196.36°)
            [105, 105, 105, 105],  # Azimuth 8 (229.09°)
            [100, 102, 103, 104],  # Azimuth 9 (261.82°)
            [100, 102, 103, 104],  # Azimuth 10 (294.55°)
            [100, 102, 103, 104],  # Azimuth 11 (327.27°)
        ])
        
        self.original_data = data_matrix
        return data_matrix
    
    def interpolate_azimuth_360(self, n_azimuth_points=360):
        """
        Interpolate azimuth data from 11 points to full 360° coverage.
        
        Args:
            n_azimuth_points: Number of azimuth points in output (default: 360)
            
        Returns:
            Interpolated data matrix
        """
        if self.original_data is None:
            self.load_original_data()
        
        n_incidence = self.original_data.shape[1]
        interpolated_data = np.zeros((n_azimuth_points, n_incidence))
        
        # Convert to radians for circular interpolation
        azimuth_rad = np.radians(self.original_azimuth)
        azimuth_full_rad = np.radians(np.linspace(0, 360, n_azimuth_points, endpoint=False))
        
        for inc_idx in range(n_incidence):
            # Get values for this incidence angle
            values = self.original_data[:, inc_idx]
            
            # Create circular interpolation (wrap around 360°)
            values_extended = np.concatenate([values, [values[0]]])
            azimuth_extended = np.concatenate([azimuth_rad, [azimuth_rad[0] + 2 * np.pi]])
            
            # Create interpolation function
            f = interpolate.interp1d(azimuth_extended, values_extended, 
                                     kind='cubic', fill_value='extrapolate')
            
            # Interpolate to full resolution
            interpolated_data[:, inc_idx] = f(azimuth_full_rad)
        
        self.azimuth_full = np.linspace(0, 360, n_azimuth_points, endpoint=False)
        return interpolated_data
    
    def interpolate_incidence_50(self, n_incidence_points=50, max_incidence=50):
        """
        Interpolate incidence data to cover 0-50° range.
        
        Args:
            n_incidence_points: Number of incidence points in output
            max_incidence: Maximum incidence angle (default: 50°)
            
        Returns:
            Tuple of (incidence_angles, interpolated_data)
        """
        # First interpolate azimuth to 360 points
        data_azimuth_interp = self.interpolate_azimuth_360()
        
        n_azimuth = data_azimuth_interp.shape[0]
        incidence_full = np.linspace(0, max_incidence, n_incidence_points)
        interpolated_data = np.zeros((n_azimuth, n_incidence_points))
        
        for az_idx in range(n_azimuth):
            # Get values for this azimuth position
            values = data_azimuth_interp[az_idx, :]
            
            # Create interpolation function
            f = interpolate.interp1d(self.original_incidence, values, 
                                     kind='quadratic', fill_value='extrapolate',
                                     bounds_error=False)
            
            # Interpolate to full resolution
            interpolated_data[az_idx, :] = f(incidence_full)
        
        self.incidence_full = incidence_full
        self.processed_data = interpolated_data
        return incidence_full, interpolated_data
    
    def process_full_data(self):
        """
        Complete data processing pipeline.
        Returns the fully interpolated data matrix.
        """
        print("Processing azimuth and incidence data...")
        print(f"Original data shape: {self.original_data.shape}")
        
        # Process data
        incidence_full, data_full = self.interpolate_incidence_50()
        
        print(f"Processed data shape: {data_full.shape}")
        print(f"Azimuth range: 0-360° ({len(self.azimuth_full)} points)")
        print(f"Incidence range: 0-50° ({len(incidence_full)} points)")
        
        return data_full

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class DataVisualizer:
    """
    Class for creating various visualizations of the concatenated data.
    """
    
    @staticmethod
    def create_polar_plot(azimuth_data, incidence_data, data_matrix, 
                         title="Azimuth vs Incidence Angle (Polar View)",
                         output_path="polar_visualization.png",
                         figsize=(10, 8)):
        """
        Create polar coordinate visualization.
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'radar_cmap', 
            ['darkblue', 'blue', 'cyan', 'white', 'yellow', 'orange', 'red', 'darkred']
        )
        
        # Convert to meshgrid for polar plot
        theta = np.radians(azimuth_data)
        r = np.array(incidence_data)
        R, Theta = np.meshgrid(r, theta)
        
        # Create plot
        img = ax.pcolormesh(Theta, R, data_matrix, cmap=cmap, shading='auto', 
                           vmin=np.min(data_matrix), vmax=np.max(data_matrix))
        
        # Configure polar plot
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)       # Clockwise
        ax.set_ylim(0, 50)               # Incidence angle limit
        
        # Title and labels
        ax.set_title(title, fontsize=14, pad=20)
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.08, shrink=0.8)
        cbar.set_label('Measurement Value', fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', color='white')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Polar plot saved to: {output_path}")
        return fig
    
    @staticmethod
    def create_rectangular_plot(azimuth_data, incidence_data, data_matrix,
                               title="Concatenated Data: Azimuth (0-360°) vs Incidence (0-50°)",
                               output_path="rectangular_visualization.png",
                               figsize=(14, 6)):
        """
        Create rectangular (Cartesian) visualization.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'data_cmap',
            ['blue', 'cyan', 'lime', 'yellow', 'orange', 'red']
        )
        
        # Display as image
        img = ax.imshow(data_matrix.T, aspect='auto', cmap=cmap,
                       extent=[0, 360, incidence_data[-1], incidence_data[0]],
                       interpolation='bilinear',
                       vmin=np.min(data_matrix), vmax=np.max(data_matrix))
        
        # Labels and title
        ax.set_xlabel('Azimuth Angle (°)', fontsize=12)
        ax.set_ylabel('Incidence Angle (°)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        
        # Set ticks
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_xticklabels([f'{i}°' for i in range(0, 361, 45)])
        
        # Add grid
        ax.grid(True, alpha=0.2, linestyle='-', color='white')
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax, fraction=0.023, pad=0.04)
        cbar.set_label('Value', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Rectangular plot saved to: {output_path}")
        return fig
    
    @staticmethod
    def create_3d_surface_plot(azimuth_data, incidence_data, data_matrix,
                              title="3D Surface Plot of Data",
                              output_path="3d_surface_plot.png",
                              figsize=(12, 8)):
        """
        Create 3D surface plot of the data.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(azimuth_data, incidence_data)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, data_matrix.T, cmap='viridis',
                              alpha=0.8, linewidth=0.5, antialiased=True)
        
        # Labels
        ax.set_xlabel('Azimuth Angle (°)', fontsize=11, labelpad=10)
        ax.set_ylabel('Incidence Angle (°)', fontsize=11, labelpad=10)
        ax.set_zlabel('Value', fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"3D surface plot saved to: {output_path}")
        return fig

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

class ImageGenerator:
    """
    Class for generating image files from the data.
    """
    
    @staticmethod
    def create_grayscale_image(data_matrix, output_path="concatenated_image_grayscale.png",
                              scale_factor=10):
        """
        Create a grayscale image from the data matrix.
        
        Args:
            data_matrix: 2D numpy array with data values
            output_path: Path to save the image
            scale_factor: Factor to scale the image for better visibility
            
        Returns:
            PIL Image object
        """
        # Normalize data to 0-255
        data_min = np.min(data_matrix)
        data_max = np.max(data_matrix)
        
        if data_max > data_min:
            normalized_data = ((data_matrix - data_min) / (data_max - data_min) * 255)
        else:
            normalized_data = np.full_like(data_matrix, 128)
        
        # Convert to uint8
        normalized_data = normalized_data.astype(np.uint8)
        
        # Create image (transpose to have incidence as rows)
        img_array = normalized_data.T
        
        # Create PIL Image
        img_pil = Image.fromarray(img_array, mode='L')
        
        # Scale up for better visibility
        original_width, original_height = img_pil.size
        new_width = original_width * scale_factor
        new_height = original_height * scale_factor
        
        img_pil = img_pil.resize((new_width, new_height), Image.NEAREST)
        
        # Save image
        img_pil.save(output_path)
        
        print(f"Grayscale image saved to: {output_path}")
        print(f"Original size: {original_width} × {original_height}")
        print(f"Scaled size: {new_width} × {new_height}")
        
        return img_pil
    
    @staticmethod
    def create_color_image(data_matrix, output_path="concatenated_image_color.png",
                          scale_factor=10, colormap='viridis'):
        """
        Create a color image from the data matrix.
        
        Args:
            data_matrix: 2D numpy array with data values
            output_path: Path to save the image
            scale_factor: Factor to scale the image
            colormap: Matplotlib colormap name
            
        Returns:
            PIL Image object
        """
        # Normalize data to 0-1
        data_min = np.min(data_matrix)
        data_max = np.max(data_matrix)
        
        if data_max > data_min:
            normalized_data = (data_matrix - data_min) / (data_max - data_min)
        else:
            normalized_data = np.full_like(data_matrix, 0.5)
        
        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        colored_data = cmap(normalized_data)
        
        # Convert to uint8 (0-255)
        colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
        
        # Create image (transpose to have incidence as rows)
        img_array = colored_data.transpose(1, 0, 2)
        
        # Create PIL Image
        img_pil = Image.fromarray(img_array, mode='RGB')
        
        # Scale up
        original_width, original_height = img_pil.size
        new_width = original_width * scale_factor
        new_height = original_height * scale_factor
        
        img_pil = img_pil.resize((new_width, new_height), Image.NEAREST)
        
        # Save image
        img_pil.save(output_path)
        
        print(f"Color image saved to: {output_path}")
        print(f"Colormap used: {colormap}")
        
        return img_pil

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class DataExporter:
    """
    Class for exporting data in various formats.
    """
    
    @staticmethod
    def export_to_csv(azimuth_data, incidence_data, data_matrix, 
                     output_path="concatenated_data.csv"):
        """
        Export data to CSV file.
        """
        # Create DataFrame
        df = pd.DataFrame(data_matrix.T, 
                         index=[f"Inc_{inc:.1f}°" for inc in incidence_data],
                         columns=[f"Az_{az:.1f}°" for az in azimuth_data])
        
        # Add summary statistics
        df['Mean'] = df.mean(axis=1)
        df['Std'] = df.std(axis=1)
        df['Min'] = df.min(axis=1)
        df['Max'] = df.max(axis=1)
        
        # Save to CSV
        df.to_csv(output_path)
        
        print(f"Data exported to CSV: {output_path}")
        print(f"Data shape: {df.shape}")
        
        return df
    
    @staticmethod
    def export_to_numpy(azimuth_data, incidence_data, data_matrix,
                       output_path="concatenated_data.npz"):
        """
        Export data to NumPy compressed format.
        """
        np.savez_compressed(output_path,
                           azimuth_data=azimuth_data,
                           incidence_data=incidence_data,
                           data_matrix=data_matrix,
                           metadata={
                               'description': 'Concatenated azimuth and incidence data',
                               'azimuth_range': f"0-360° ({len(azimuth_data)} points)",
                               'incidence_range': f"0-{incidence_data[-1]:.1f}° ({len(incidence_data)} points)",
                               'data_shape': data_matrix.shape
                           })
        
        print(f"Data exported to NumPy format: {output_path}")
        
    @staticmethod
    def create_statistics_report(data_matrix, output_path="data_statistics.txt"):
        """
        Create a text report with data statistics.
        """
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA STATISTICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data Shape: {data_matrix.shape}\n")
            f.write(f"Total Data Points: {data_matrix.size}\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"  Minimum Value: {np.min(data_matrix):.2f}\n")
            f.write(f"  Maximum Value: {np.max(data_matrix):.2f}\n")
            f.write(f"  Mean Value: {np.mean(data_matrix):.2f}\n")
            f.write(f"  Standard Deviation: {np.std(data_matrix):.2f}\n")
            f.write(f"  Median Value: {np.median(data_matrix):.2f}\n\n")
            
            f.write("Value Distribution:\n")
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                f.write(f"  {p}th Percentile: {np.percentile(data_matrix, p):.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Statistics report saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to execute the entire concatenation process.
    """
    print("=" * 70)
    print("AZIMUTH AND INCIDENCE DATA CONCATENATION SYSTEM")
    print("=" * 70)
    print()
    
    # Step 1: Initialize and process data
    print("STEP 1: DATA PROCESSING")
    print("-" * 40)
    
    processor = AzimuthIncidenceConcatenator()
    processor.load_original_data()
    data_full = processor.process_full_data()
    
    print(f"✓ Data processing complete")
    print()
    
    # Step 2: Create visualizations
    print("STEP 2: CREATING VISUALIZATIONS")
    print("-" * 40)
    
    visualizer = DataVisualizer()
    
    # Create polar plot
    polar_fig = visualizer.create_polar_plot(
        processor.azimuth_full, 
        processor.incidence_full, 
        data_full,
        title="Azimuth vs Incidence Data Distribution",
        output_path="output/azimuth_polar.png"
    )
    print(f"✓ Polar plot created")
    
    # Create rectangular plot
    rect_fig = visualizer.create_rectangular_plot(
        processor.azimuth_full,
        processor.incidence_full,
        data_full,
        title="Concatenated Azimuth and Incidence Data",
        output_path="output/azimuth_rectangular.png"
    )
    print(f"✓ Rectangular plot created")
    
    # Create 3D plot
    try:
        _3d_fig = visualizer.create_3d_surface_plot(
            processor.azimuth_full,
            processor.incidence_full,
            data_full,
            title="3D Surface Visualization",
            output_path="output/azimuth_3d.png"
        )
        print(f"✓ 3D surface plot created")
    except:
        print("⚠ 3D plot skipped (optional dependency)")
    
    print()
    
    # Step 3: Generate image files
    print("STEP 3: GENERATING IMAGE FILES")
    print("-" * 40)
    
    img_generator = ImageGenerator()
    
    # Create grayscale image
    gray_img = img_generator.create_grayscale_image(
        data_full,
        output_path="output/concatenated_grayscale.png",
        scale_factor=10
    )
    print(f"✓ Grayscale image generated")
    
    # Create color image
    color_img = img_generator.create_color_image(
        data_full,
        output_path="output/concatenated_color.png",
        scale_factor=10,
        colormap='plasma'
    )
    print(f"✓ Color image generated")
    
    print()
    
    # Step 4: Export data
    print("STEP 4: EXPORTING DATA")
    print("-" * 40)
    
    exporter = DataExporter()
    
    # Export to CSV
    csv_df = exporter.export_to_csv(
        processor.azimuth_full,
        processor.incidence_full,
        data_full,
        output_path="output/concatenated_data.csv"
    )
    print(f"✓ CSV export complete")
    
    # Export to NumPy
    exporter.export_to_numpy(
        processor.azimuth_full,
        processor.incidence_full,
        data_full,
        output_path="output/concatenated_data.npz"
    )
    print(f"✓ NumPy export complete")
    
    # Create statistics report
    exporter.create_statistics_report(
        data_full,
        output_path="output/data_statistics.txt"
    )
    print(f"✓ Statistics report generated")
    
    print()
    
    # Step 5: Summary
    print("STEP 5: PROCESSING COMPLETE")
    print("-" * 40)
    
    print("✅ All processing steps completed successfully!")
    print()
    print("📁 OUTPUT FILES GENERATED:")
    print("=" * 40)
    print("1. output/azimuth_polar.png        - Polar coordinate visualization")
    print("2. output/azimuth_rectangular.png  - Rectangular visualization")
    print("3. output/azimuth_3d.png           - 3D surface plot")
    print("4. output/concatenated_grayscale.png - Grayscale image")
    print("5. output/concatenated_color.png   - Color image")
    print("6. output/concatenated_data.csv    - Data in CSV format")
    print("7. output/concatenated_data.npz    - Data in NumPy format")
    print("8. output/data_statistics.txt      - Statistical analysis")
    print()
    print("📊 DATA SUMMARY:")
    print("=" * 40)
    print(f"   Azimuth range:    0-360° ({len(processor.azimuth_full)} points)")
    print(f"   Incidence range:  0-50° ({len(processor.incidence_full)} points)")
    print(f"   Data resolution:  {data_full.shape[0]} × {data_full.shape[1]}")
    print(f"   Value range:      {np.min(data_full):.2f} to {np.max(data_full):.2f}")
    print("=" * 70)

# ============================================================================
# CONFIGURATION AND EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    import os
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created 'output' directory for results")
    
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if all required packages are installed:")
        print("   pip install numpy matplotlib scipy pillow pandas")
        print("2. Make sure the output directory is writable")
        print("3. Check the data format in the load_original_data() method")
