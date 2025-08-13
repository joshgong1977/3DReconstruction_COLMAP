#!/usr/bin/env python3
"""
Simple COLMAP runner for your existing images
Just put your images in the same folder as this script and run it!
"""

import os
from pathlib import Path

# Import the COLMAP pipeline (save the previous artifact as colmap_pipeline.py)
try:
    from colmap_pipeline import COLMAPPipeline
except ImportError:
    print(" Please save the COLMAP pipeline code as 'colmap_pipeline.py' in the same folder")
    exit(1)

def main():
    # Configuration
    current_dir = Path(__file__).parent
    image_folder = current_dir  # Look for images in the same folder as this script
    output_folder = current_dir / "colmap_output"
    
    # Your camera parameters (from your original code)
    camera_params = {
        'fx': 2905.88,
        'fy': 2905.88,
        # cx and cy will be automatically calculated as width/2 and height/2
    }
    
    # Create pipeline first to use the fixed image detection
    pipeline = COLMAPPipeline(
        image_folder=str(image_folder),
        output_folder=str(output_folder),
        camera_model="PINHOLE"
    )
    
    # Use the pipeline's fixed method to get unique image files
    image_files = pipeline.get_image_files()
    
    print(f"Found {len(image_files)} images:")
    for img in sorted(image_files):
        print(f"   - {img.name}")
    
    if len(image_files) < 2:
        print("Need at least 2 images for reconstruction")
        print("Please put your images (100_7101.JPG, 100_7102.JPG, etc.) in the same folder as this script")
        return
    
    # Run the pipeline
    print(f"\n Starting COLMAP reconstruction...")
    print("This may take several minutes depending on image size and number...")
    success = pipeline.run_full_pipeline(camera_params=camera_params)
    
    if success:
        print("\n Reconstruction completed successfully!")
        
        # Get reconstruction info
        info = pipeline.get_reconstruction_info()
        print(f"\n Results:")
        print(f"   - Sparse points: {info['num_sparse_points']:,}")
        print(f"   - Dense points: {info['num_dense_points']:,}")
        print(f"   - Files created in: {output_folder}")
        
        # List output files
        print(f"\n Output files:")
        output_files = [
            "sparse_points.ply",
            "dense/fused.ply",
            "dense/meshed-poisson.ply"
        ]
        
        for file_path in output_files:
            full_path = output_folder / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"   {file_path} ({size_mb:.1f} MB)")
            else:
                print(f"   {file_path} (not created)")
        
        # Ask if user wants to visualize
        try:
            response = input("\n Open 3D visualization? (y/n): ").strip().lower()
            if response in ['y', 'yes', '']:
                print("Opening visualization (close the window when done)...")
                pipeline.visualize_results()
        except KeyboardInterrupt:
            print("\nSkipping visualization.")
        
    else:
        print("\n Reconstruction failed!")
        print("Check the error messages above for details.")
        print("\nCommon issues:")
        print("- Make sure COLMAP is installed: sudo apt install colmap")
        print("- Check that images have sufficient overlap")
        print("- Verify image quality and lighting conditions")

if __name__ == "__main__":
    main()
