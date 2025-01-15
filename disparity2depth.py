import numpy as np
import cv2
import open3d as o3d
import os

def disparity_to_depth(disparity_map, focal_length, baseline):
    """Convert disparity map to depth map."""
    depth_map = (focal_length * baseline) / (disparity_map + 1e-6)  # Avoid division by zero
    depth_map[disparity_map == 0] = 0  # Mask invalid disparity
    return depth_map

def generate_point_cloud(disparity_map, left_image, calibration_data):
    """Generate a 3D point cloud from disparity map."""
    h, w = disparity_map.shape
    focal_length = calibration_data['focal_length']
    baseline = calibration_data['baseline']
    cx, cy = calibration_data['cx'], calibration_data['cy']
    
    # Convert disparity to depth
    depth_map = disparity_to_depth(disparity_map, focal_length, baseline)
    
    # Create a 3D point cloud
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            depth = depth_map[v, u]
            if depth > 0:  # Ignore invalid depths
                x = (u - cx) * depth / focal_length
                y = (v - cy) * depth / focal_length
                z = depth
                points.append((x, y, z))
                colors.append(left_image[v, u] / 255.0)  # Normalize color to [0, 1]
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.repeat(colors[:, None], 3, axis=1))  # Grayscale to RGB
    return point_cloud

def process_kitti_sequence(disparity_map_dir, left_image_dir, calibration_data):
    """Process the KITTI sequence of disparity maps and images."""
    point_clouds = []  # List to store point clouds from each frame
    
    # List all disparity maps and left images (assuming files are named in the same way)
    disparity_maps = sorted(os.listdir(disparity_map_dir))
    left_images = sorted(os.listdir(left_image_dir))
    
    for i in range(len(disparity_maps)):
        # Load disparity map and left image for the current frame
        disparity_map = cv2.imread(os.path.join(disparity_map_dir, disparity_maps[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        left_image = cv2.imread(os.path.join(left_image_dir, left_images[i]), cv2.IMREAD_GRAYSCALE)

        # Generate point cloud for the current frame
        point_cloud = generate_point_cloud(disparity_map, left_image, calibration_data)
        
        # Append the point cloud to the list
        point_clouds.append(point_cloud)
        
        print(f"Processed frame {i+1}/{len(disparity_maps)}")

    # Combine all point clouds into one
    combined_point_cloud = point_clouds[0]
    for point_cloud in point_clouds[1:]:
        combined_point_cloud += point_cloud
    
    return combined_point_cloud

# Calibration data (from KITTI dataset)
calibration_data = {
    'focal_length': 721.5377,
    'baseline': 0.537,         # Example baseline in meters
    'cx': 609.5593,           # Principal point x-coordinate
    'cy': 172.854             # Principal point y-coordinate
}

# Directories containing the KITTI dataset's disparity maps and left images
disparity_map_dir = 'pm_disparity/'  # Directory with disparity maps
left_image_dir = 'left_images/'        # Directory with left images

# Process the KITTI sequence and get the combined 3D point cloud
combined_point_cloud = process_kitti_sequence(disparity_map_dir, left_image_dir, calibration_data)

# Save and visualize the combined 3D point cloud
o3d.io.write_point_cloud("combined_point_cloud.ply", combined_point_cloud)
o3d.visualization.draw_geometries([combined_point_cloud])
