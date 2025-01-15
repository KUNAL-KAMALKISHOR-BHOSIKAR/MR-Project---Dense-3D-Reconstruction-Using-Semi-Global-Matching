import numpy as np
import cv2

def compute_matching_cost(left_image, right_image, x, y, disparity, patch_size=3):
    """Compute matching cost for a patch centered at (x, y) with a given disparity."""
    height, width = left_image.shape
    half_patch = patch_size // 2
    cost = 0

    for dy in range(-half_patch, half_patch + 1):
        for dx in range(-half_patch, half_patch + 1):
            lx, ly = x + dx, y + dy
            rx, ry = x + dx - disparity, y + dy

            if 0 <= lx < width and 0 <= ly < height and 0 <= rx < width and 0 <= ry < height:
                cost += abs(int(left_image[ly, lx]) - int(right_image[ry, rx]))
    
    return cost

def initialize_disparity(height, width, max_disparity):
    """Randomly initialize disparity map."""
    return np.random.randint(0, max_disparity, size=(height, width), dtype=np.int32)

def spatial_propagation(disparity, x, y, width, height, max_disparity, left_image, right_image, patch_size):
    """Perform spatial propagation step."""
    min_cost = compute_matching_cost(left_image, right_image, x, y, disparity[y, x], patch_size)
    best_disparity = disparity[y, x]

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbor_disparity = disparity[ny, nx]
            cost = compute_matching_cost(left_image, right_image, x, y, neighbor_disparity, patch_size)
            if cost < min_cost:
                min_cost = cost
                best_disparity = neighbor_disparity
    
    return best_disparity

def random_search(disparity, x, y, width, height, max_disparity, left_image, right_image, patch_size, scale=2):
    """Perform random search step."""
    current_disparity = disparity[y, x]
    min_cost = compute_matching_cost(left_image, right_image, x, y, current_disparity, patch_size)
    best_disparity = current_disparity

    while scale > 0:
        random_disparity = np.clip(current_disparity + np.random.randint(-scale, scale + 1), 0, max_disparity - 1)
        cost = compute_matching_cost(left_image, right_image, x, y, random_disparity, patch_size)
        if cost < min_cost:
            min_cost = cost
            best_disparity = random_disparity
        scale //= 2

    return best_disparity

def patch_match(left_image, right_image, max_disparity, num_iterations=3, patch_size=3):
    """Perform PatchMatch Stereo Matching."""
    height, width = left_image.shape
    disparity = initialize_disparity(height, width, max_disparity)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        for y in range(height):
            for x in range(width):
                # Spatial Propagation
                disparity[y, x] = spatial_propagation(disparity, x, y, width, height, max_disparity, left_image, right_image, patch_size)

                # Random Search
                disparity[y, x] = random_search(disparity, x, y, width, height, max_disparity, left_image, right_image, patch_size)
    
    return disparity

# Load rectified stereo images
left_image = cv2.imread('left_images/000012_11.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right_images/000012_11.png', cv2.IMREAD_GRAYSCALE)

# Parameters
max_disparity = 64
num_iterations = 5
patch_size = 3

# Run PatchMatch Stereo
disparity_map = patch_match(left_image, right_image, max_disparity, num_iterations, patch_size)

# Display Disparity Map
disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imshow('Disparity Map', disparity_map_normalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('pm_disparity/000012_11.png', disparity_map)