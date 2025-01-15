import numpy as np
import cv2

def compute_average_error(ground_truth_disp, generated_disp, invalid_disp_value=0):
    """
    Compute the average error between ground truth and generated disparity maps.

    Parameters:
    - ground_truth_disp (numpy.ndarray): Ground truth disparity map.
    - generated_disp (numpy.ndarray): Generated disparity map.
    - invalid_disp_value (int or float): Value in the ground truth representing invalid pixels.

    Returns:
    - average_error (float): Average error between ground truth and generated disparity maps.
    """
    # Ensure inputs are float for accurate calculations
    ground_truth_disp = ground_truth_disp.astype(np.float32)
    generated_disp = generated_disp.astype(np.float32)

    # Mask invalid pixels
    valid_mask = ground_truth_disp != invalid_disp_value

    # Calculate absolute error where valid
    error = np.abs(ground_truth_disp[valid_mask] - generated_disp[valid_mask])

    # Compute average error
    average_error = np.mean(error) if error.size > 0 else None

    return average_error

# Example usage
# Load ground truth and generated disparity maps
ground_truth_disp = cv2.imread('sgm_disparity_gt/000007_10.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
generated_disp = cv2.imread('pm_disparity/000007_10.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Define the invalid disparity value in the ground truth (commonly 0 or -1 in KITTI)
invalid_disp_value = 0

# Compute the average error
average_error = compute_average_error(ground_truth_disp, generated_disp, invalid_disp_value)
print(f"Average Disparity Error: {average_error:.4f} pixels")