import numpy as np
import cv2

def census_transform(image, kernel_size=5):
    """Apply census transform to an image."""
    height, width = image.shape
    offset = kernel_size // 2
    census = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            center_pixel = image[y, x]
            binary = 0
            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    if dy == 0 and dx == 0:
                        continue
                    binary <<= 1
                    binary |= image[y + dy, x + dx] < center_pixel
            census[y, x] = binary
    return census

def compute_cost(left_census, right_census, max_disparity):
    """Compute matching costs."""
    height, width = left_census.shape
    cost_volume = np.zeros((height, width, max_disparity), dtype=np.uint8)

    for d in range(max_disparity):
        shifted_right = np.roll(right_census, shift=-d, axis=1)
        shifted_right[:, :d] = 0  # Mask invalid disparities
        cost_volume[:, :, d] = np.bitwise_xor(left_census, shifted_right)
    
    return cost_volume

def aggregate_costs(cost_volume, penalty1=10, penalty2=150):
    """Aggregate costs in multiple directions."""
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1), # 4 main directions
        (1, 1), (-1, -1), (1, -1), (-1, 1) # Diagonals
    ]
    height, width, max_disparity = cost_volume.shape
    aggregated_cost = np.zeros_like(cost_volume, dtype=np.int32)

    for direction in directions:
        dy, dx = direction
        direction_cost = np.zeros_like(cost_volume, dtype=np.int32)
        for y in range(height):
            for x in range(width):
                for d in range(max_disparity):
                    current_cost = cost_volume[y, x, d]
                    if 0 <= y - dy < height and 0 <= x - dx < width:
                        prev_costs = direction_cost[y - dy, x - dx, :]
                        min_prev = prev_costs.min()
                        cost1 = prev_costs[d]
                        cost2 = min_prev + penalty1
                        cost3 = min_prev + penalty2
                        direction_cost[y, x, d] = current_cost + min(cost1, cost2, cost3)
                    else:
                        direction_cost[y, x, d] = current_cost
        aggregated_cost += direction_cost
    
    return aggregated_cost

def compute_disparity(aggregated_cost):
    """Compute disparity from aggregated costs."""
    disparity_map = np.argmin(aggregated_cost, axis=2).astype(np.uint8)
    return disparity_map

# Load rectified stereo images
left_image = cv2.imread('left_images/000012_11.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right_images/000012_11.png', cv2.IMREAD_GRAYSCALE)

# Parameters
max_disparity = 64
penalty1 = 10
penalty2 = 150

# Step 1: Census Transform
left_census = census_transform(left_image)
right_census = census_transform(right_image)

# Step 2: Compute Cost Volume
cost_volume = compute_cost(left_census, right_census, max_disparity)

# Step 3: Cost Aggregation
aggregated_cost = aggregate_costs(cost_volume, penalty1, penalty2)

# Step 4: Disparity Computation
disparity_map = compute_disparity(aggregated_cost)

# Display Disparity Map
# cv2.imshow('Disparity Map', disparity_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save Disparity Map
cv2.imwrite('sgm_disparity/000012_11.png', disparity_map)