import os
import shutil
import subprocess
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import json

class COLMAPPipeline:
    def __init__(self, image_folder: str, output_folder: str, camera_model: str = "PINHOLE"):
        """
        Initialize COLMAP pipeline
        
        Args:
            image_folder: Path to folder containing input images
            output_folder: Path to output folder for results
            camera_model: Camera model (PINHOLE, SIMPLE_PINHOLE, RADIAL, OPENCV, etc.)
        """
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.camera_model = camera_model
        
        # Create output directories
        self.database_path = self.output_folder / "database.db"
        self.sparse_folder = self.output_folder / "sparse"
        self.dense_folder = self.output_folder / "dense"
        
        # Create all necessary directories
        self.output_folder.mkdir(exist_ok=True)
        self.sparse_folder.mkdir(exist_ok=True)
        self.dense_folder.mkdir(exist_ok=True)
        
        # Check if COLMAP is available
        self._check_colmap_installation()
    
    def _check_colmap_installation(self):
        """Check if COLMAP is installed and accessible"""
        try:
            result = subprocess.run(['colmap', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("COLMAP not found or not working properly")
            print(" COLMAP installation verified")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            print(" COLMAP not found!")
            print("\nTo install COLMAP:")
            print("1. Ubuntu/Debian: sudo apt-get install colmap")
            print("2. macOS: brew install colmap")
            print("3. Windows: Download from https://colmap.github.io/")
            print("4. Or build from source: https://colmap.github.io/install.html")
            raise RuntimeError(f"COLMAP installation required: {e}")
    
    def _run_command(self, command: List[str], description: str = "") -> bool:
        """Run a command and handle errors"""
        try:
            print(f"Running: {description}")
            print(f"Command: {' '.join(command)}")
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f" Error in {description}:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            else:
                print(f" {description} completed successfully")
                return True
                
        except subprocess.TimeoutExpired:
            print(f"Timeout in {description}")
            return False
        except Exception as e:
            print(f" Exception in {description}: {e}")
            return False
    
    def feature_extraction(self, camera_params: Optional[dict] = None) -> bool:
        """
        Extract features from images
        
        Args:
            camera_params: Optional camera parameters dict with keys:
                          fx, fy, cx, cy for PINHOLE model
        """
        print("\n=== FEATURE EXTRACTION ===")
        
        command = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.image_folder),
            '--ImageReader.camera_model', self.camera_model,
            '--SiftExtraction.use_gpu', '1' if self._check_gpu() else '0',
            '--SiftExtraction.max_image_size', '3200',
            '--SiftExtraction.max_num_features', '8192'
        ]
        
        # Add camera parameters if provided
        if camera_params and self.camera_model == "PINHOLE":
            fx = camera_params.get('fx', 2905.88)
            fy = camera_params.get('fy', 2905.88)
            cx = camera_params.get('cx')
            cy = camera_params.get('cy')
            
            # Get image dimensions to set principal point if not provided
            if cx is None or cy is None:
                sample_image = self._find_sample_image()
                if sample_image:
                    img = cv2.imread(str(sample_image))
                    height, width = img.shape[:2]
                    cx = cx or width / 2
                    cy = cy or height / 2
            
            command.extend([
                '--ImageReader.camera_params', f"{fx},{fy},{cx},{cy}"
            ])
        
        return self._run_command(command, "Feature extraction")
    
    def _find_sample_image(self):
        """Find a sample image to get dimensions"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for file in self.image_folder.iterdir():
            if file.suffix.lower() in [ext.lower() for ext in image_extensions]:
                return file
        return None
    
    def feature_matching(self) -> bool:
        """Match features between images"""
        print("\n=== FEATURE MATCHING ===")
        
        # Updated command without deprecated parameters
        command = [
            'colmap', 'exhaustive_matcher',
            '--database_path', str(self.database_path),
            '--SiftMatching.use_gpu', '1' if self._check_gpu() else '0',
            '--SiftMatching.guided_matching', '1',
            '--SiftMatching.max_ratio', '0.8',
            '--SiftMatching.max_distance', '0.7'
        ]
        
        return self._run_command(command, "Feature matching")
    
    def sparse_reconstruction(self) -> bool:
        """Perform sparse 3D reconstruction (SfM)"""
        print("\n=== SPARSE RECONSTRUCTION ===")
        
        command = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.image_folder),
            '--output_path', str(self.sparse_folder),
            '--Mapper.ba_refine_focal_length', '1',
            '--Mapper.ba_refine_principal_point', '1',
            '--Mapper.ba_refine_extra_params', '1',
            '--Mapper.min_model_size', '3',
            '--Mapper.abs_pose_min_num_inliers', '30',
            '--Mapper.abs_pose_min_inlier_ratio', '0.25',
            '--Mapper.ba_local_max_num_iterations', '25',
            '--Mapper.ba_global_max_num_iterations', '50'
        ]
        
        success = self._run_command(command, "Sparse reconstruction")
        
        if success:
            # Find the reconstruction folder (usually named '0')
            reconstruction_folders = [f for f in self.sparse_folder.iterdir() if f.is_dir()]
            if reconstruction_folders:
                print(f" Sparse reconstruction completed with {len(reconstruction_folders)} model(s)")
                return True
            else:
                print(" No reconstruction models found")
                return False
        
        return False
    
    def dense_reconstruction(self, model_folder: str = "0") -> bool:
        """
        Perform dense 3D reconstruction (MVS)
        
        Args:
            model_folder: Name of the sparse reconstruction folder to use
        """
        print("\n=== DENSE RECONSTRUCTION ===")
        
        sparse_model_path = self.sparse_folder / model_folder
        if not sparse_model_path.exists():
            print(f" Sparse model folder {sparse_model_path} does not exist")
            return False
        
        # Step 1: Image undistortion
        print("Step 1: Image undistortion")
        command_undistort = [
            'colmap', 'image_undistorter',
            '--image_path', str(self.image_folder),
            '--input_path', str(sparse_model_path),
            '--output_path', str(self.dense_folder),
            '--output_type', 'COLMAP'
        ]
        
        if not self._run_command(command_undistort, "Image undistortion"):
            return False
        
        # Step 2: Patch match stereo
        print("Step 2: Patch match stereo")
        command_stereo = [
            'colmap', 'patch_match_stereo',
            '--workspace_path', str(self.dense_folder),
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', '1',
            '--PatchMatchStereo.max_image_size', '2000',
            '--PatchMatchStereo.window_radius', '5',
            '--PatchMatchStereo.window_step', '1',
            '--PatchMatchStereo.num_samples', '15',
            '--PatchMatchStereo.num_iterations', '5',
            '--PatchMatchStereo.geom_consistency_regularizer', '0.3',
            '--PatchMatchStereo.geom_consistency_max_cost', '3.0'
        ]
        
        if not self._run_command(command_stereo, "Patch match stereo"):
            return False
        
        # Step 3: Stereo fusion
        print("Step 3: Stereo fusion")
        command_fusion = [
            'colmap', 'stereo_fusion',
            '--workspace_path', str(self.dense_folder),
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', str(self.dense_folder / 'fused.ply'),
            '--StereoFusion.max_image_size', '2000',
            '--StereoFusion.min_num_pixels', '5',
            '--StereoFusion.max_num_pixels', '10000',
            '--StereoFusion.max_traversal_depth', '100',
            '--StereoFusion.max_reproj_error', '2.0',
            '--StereoFusion.max_depth_error', '0.01',
            '--StereoFusion.max_normal_error', '10',
            '--StereoFusion.check_num_images', '50'
        ]
        
        if not self._run_command(command_fusion, "Stereo fusion"):
            return False
        
        # Step 4: Poisson meshing (optional)
        print("Step 4: Poisson meshing")
        command_poisson = [
            'colmap', 'poisson_mesher',
            '--input_path', str(self.dense_folder / 'fused.ply'),
            '--output_path', str(self.dense_folder / 'meshed-poisson.ply'),
            '--PoissonMeshing.point_weight', '1.0',
            '--PoissonMeshing.depth', '13',
            '--PoissonMeshing.color', '32',
            '--PoissonMeshing.trim', '7',
            '--PoissonMeshing.num_threads', '-1'
        ]
        
        self._run_command(command_poisson, "Poisson meshing")  # Optional, don't fail if it doesn't work
        
        return True
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for COLMAP"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def convert_to_ply(self, model_folder: str = "0") -> bool:
        """Convert sparse reconstruction to PLY format"""
        sparse_model_path = self.sparse_folder / model_folder
        
        if not sparse_model_path.exists():
            print(f" Sparse model folder {sparse_model_path} does not exist")
            return False
        
        output_ply = self.output_folder / "sparse_points.ply"
        
        command = [
            'colmap', 'model_converter',
            '--input_path', str(sparse_model_path),
            '--output_path', str(output_ply),
            '--output_type', 'PLY'
        ]
        
        return self._run_command(command, f"Converting sparse model to PLY: {output_ply}")
    
    def visualize_results(self, show_sparse: bool = True, show_dense: bool = True):
        """Visualize reconstruction results using Open3D"""
        print("\n=== VISUALIZATION ===")
        
        point_clouds = []
        
        # Load sparse point cloud
        if show_sparse:
            sparse_ply = self.output_folder / "sparse_points.ply"
            if sparse_ply.exists():
                try:
                    pcd_sparse = o3d.io.read_point_cloud(str(sparse_ply))
                    if len(pcd_sparse.points) > 0:
                        # Color sparse points differently
                        pcd_sparse.paint_uniform_color([1, 0, 0])  # Red for sparse
                        point_clouds.append(pcd_sparse)
                        print(f" Loaded sparse point cloud: {len(pcd_sparse.points)} points")
                    else:
                        print(" Sparse point cloud is empty")
                except Exception as e:
                    print(f" Failed to load sparse point cloud: {e}")
            else:
                print(" Sparse point cloud file not found")
        
        # Load dense point cloud
        if show_dense:
            dense_ply = self.dense_folder / "fused.ply"
            if dense_ply.exists():
                try:
                    pcd_dense = o3d.io.read_point_cloud(str(dense_ply))
                    if len(pcd_dense.points) > 0:
                        point_clouds.append(pcd_dense)
                        print(f" Loaded dense point cloud: {len(pcd_dense.points)} points")
                    else:
                        print(" Dense point cloud is empty")
                except Exception as e:
                    print(f" Failed to load dense point cloud: {e}")
            else:
                print(" Dense point cloud file not found")
        
        # Load mesh if available
        mesh_ply = self.dense_folder / "meshed-poisson.ply"
        if mesh_ply.exists():
            try:
                mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
                if len(mesh.vertices) > 0:
                    print(f" Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                    # Uncomment to show mesh instead of point cloud
                    # point_clouds = [mesh]
            except Exception as e:
                print(f" Failed to load mesh: {e}")
        
        # Visualize
        if point_clouds:
            print("Opening visualization...")
            o3d.visualization.draw_geometries(point_clouds, 
                                            window_name="COLMAP Reconstruction Results",
                                            width=1200, height=800)
        else:
            print(" No point clouds to visualize")
    
    def get_image_files(self) -> List[Path]:
        """Get unique list of image files, avoiding duplicates"""
        image_files = set()  # Use set to avoid duplicates
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        for file in self.image_folder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.add(file)
        
        return sorted(list(image_files))
    
    def get_reconstruction_info(self, model_folder: str = "0") -> dict:
        """Get information about the reconstruction"""
        sparse_model_path = self.sparse_folder / model_folder
        
        info = {
            'sparse_model_exists': sparse_model_path.exists(),
            'dense_model_exists': (self.dense_folder / "fused.ply").exists(),
            'mesh_exists': (self.dense_folder / "meshed-poisson.ply").exists(),
            'num_images': 0,
            'num_sparse_points': 0,
            'num_dense_points': 0
        }
        
        # Count images using the fixed method
        image_files = self.get_image_files()
        info['num_images'] = len(image_files)
        
        # Count sparse points
        sparse_ply = self.output_folder / "sparse_points.ply"
        if sparse_ply.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(sparse_ply))
                info['num_sparse_points'] = len(pcd.points)
            except:
                pass
        
        # Count dense points
        dense_ply = self.dense_folder / "fused.ply"
        if dense_ply.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(dense_ply))
                info['num_dense_points'] = len(pcd.points)
            except:
                pass
        
        return info
    
    def run_full_pipeline(self, camera_params: Optional[dict] = None, 
                         skip_dense: bool = False) -> bool:
        """
        Run the complete COLMAP pipeline
        
        Args:
            camera_params: Optional camera parameters
            skip_dense: If True, only run sparse reconstruction
        """
        print(" Starting COLMAP Pipeline")
        print(f"Input folder: {self.image_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Camera model: {self.camera_model}")
        
        # Step 1: Feature extraction
        if not self.feature_extraction(camera_params):
            print("Pipeline failed at feature extraction")
            return False
        
        # Step 2: Feature matching
        if not self.feature_matching():
            print("Pipeline failed at feature matching")
            return False
        
        # Step 3: Sparse reconstruction
        if not self.sparse_reconstruction():
            print("Pipeline failed at sparse reconstruction")
            return False
        
        # Convert sparse model to PLY
        if not self.convert_to_ply():
            print("Warning: Failed to convert sparse model to PLY")
        
        # Step 4: Dense reconstruction (optional)
        if not skip_dense:
            if not self.dense_reconstruction():
                print("Pipeline failed at dense reconstruction")
                return False
        
        # Print summary
        info = self.get_reconstruction_info()
        print(f"\n COLMAP Pipeline completed successfully!")
        print(f" Reconstruction Summary:")
        print(f"   - Images processed: {info['num_images']}")
        print(f"   - Sparse points: {info['num_sparse_points']:,}")
        print(f"   - Dense points: {info['num_dense_points']:,}")
        print(f"   - Mesh generated: {'Yes' if info['mesh_exists'] else 'No'}")
        
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="COLMAP-based SfM and MVS Pipeline")
    parser.add_argument("--images", required=True, help="Path to folder containing images")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--camera_model", default="PINHOLE", 
                       choices=["PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE"],
                       help="Camera model to use")
    parser.add_argument("--fx", type=float, help="Focal length x (for PINHOLE model)")
    parser.add_argument("--fy", type=float, help="Focal length y (for PINHOLE model)")
    parser.add_argument("--cx", type=float, help="Principal point x")
    parser.add_argument("--cy", type=float, help="Principal point y")
    parser.add_argument("--skip_dense", action="store_true", help="Skip dense reconstruction")
    parser.add_argument("--visualize", action="store_true", help="Visualize results after reconstruction")
    
    args = parser.parse_args()
    
    # Prepare camera parameters
    camera_params = None
    if args.fx or args.fy or args.cx or args.cy:
        camera_params = {}
        if args.fx: camera_params['fx'] = args.fx
        if args.fy: camera_params['fy'] = args.fy
        if args.cx: camera_params['cx'] = args.cx
        if args.cy: camera_params['cy'] = args.cy
    
    # Create and run pipeline
    pipeline = COLMAPPipeline(args.images, args.output, args.camera_model)
    
    success = pipeline.run_full_pipeline(camera_params, args.skip_dense)
    
    if success and args.visualize:
        pipeline.visualize_results()
    
    return 0 if success else 1


if __name__ == "__main__":
    # Example usage when running as script
    import sys
    
    if len(sys.argv) == 1:
        # Demo mode - provide example usage
        print("COLMAP SfM/MVS Pipeline")
        print("=" * 50)
        print("\nExample usage:")
        print("python colmap_pipeline.py --images ./images --output ./output --visualize")
        print("\nWith camera parameters:")
        print("python colmap_pipeline.py --images ./images --output ./output --fx 2905.88 --fy 2905.88 --visualize")
        print("\nFor interactive usage:")
        
        # Interactive mode
        image_folder = input("Enter path to images folder: ").strip()
        output_folder = input("Enter path to output folder: ").strip()
        
        if not image_folder or not output_folder:
            print(" Both image and output folders are required")
            sys.exit(1)
        
        pipeline = COLMAPPipeline(image_folder, output_folder)
        success = pipeline.run_full_pipeline()
        
        if success:
            visualize = input("Visualize results? (y/n): ").strip().lower()
            if visualize in ['y', 'yes']:
                pipeline.visualize_results()
    else:
        sys.exit(main())
