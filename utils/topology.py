import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure


def compute_betti_numbers(mask):
    """
    Compute Betti numbers using connected components analysis
    
    Note: This is a simplified 3D implementation. Full persistent homology
    requires GUDHI library which has complex dependencies.
    
    Args:
        mask: Binary segmentation mask (H, W, D) numpy array
    Returns:
        beta_0: Number of connected components
        beta_1: Number of tunnels/loops (approximated)
        beta_2: Number of cavities (approximated)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(bool)
    
    # Beta_0: Connected components
    labeled, beta_0 = label(mask)
    
    # Beta_1 (loops): Euler characteristic approach
    # This is approximation - true computation requires homology
    euler = beta_0
    
    # Count voxels, edges, faces (simplified)
    voxels = mask.sum()
    
    # Approximate beta_1 using Euler characteristic
    # χ = β_0 - β_1 + β_2
    # For solid objects, typically β_2 = 0, so β_1 ≈ β_0 - χ
    beta_1 = 0  # Simplified - detecting loops requires complex topology
    
    # Beta_2 (voids): Detect enclosed cavities
    filled = binary_fill_holes(mask)
    beta_2 = (filled & ~mask).sum() > 0
    
    return int(beta_0), int(beta_1), int(beta_2)


def binary_fill_holes(mask):
    """Fill holes in 3D binary mask"""
    from scipy.ndimage import binary_fill_holes as fill_3d
    return fill_3d(mask)


def compute_persistence_diagram_simplified(mask, distance_map=None):
    """
    Simplified persistence computation using distance transform
    
    Note: This is NOT full persistent homology. For production use,
    integrate GUDHI library. This provides basic approximation for
    development and testing.
    
    Args:
        mask: Binary mask or probability map
        distance_map: Optional precomputed distance transform
    Returns:
        persistence_pairs: List of (birth, death, persistence) tuples
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if distance_map is None:
        # Compute distance transform
        distance_map = distance_transform_edt(mask)
    
    # Find local maxima as birth points
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance_map, size=3)
    peaks = (distance_map == local_max) & (distance_map > 0)
    
    # Extract peak values
    peak_values = distance_map[peaks]
    
    # Create simple persistence pairs
    # (birth at 0, death at distance value)
    persistence_pairs = []
    for val in peak_values:
        persistence_pairs.append((0.0, float(val), float(val)))
    
    # Sort by persistence
    persistence_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return persistence_pairs[:10]  # Top 10 features


def extract_boundaries(mask, threshold=0.5):
    """
    Extract organ boundaries using gradient
    
    Args:
        mask: Segmentation mask (can be soft or hard)
    Returns:
        boundary_mask: Binary boundary map
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Threshold if soft mask
    if mask_np.max() <= 1.0 and mask_np.min() >= 0.0:
        binary_mask = (mask_np > threshold).astype(float)
    else:
        binary_mask = (mask_np > 0).astype(float)
    
    # Compute gradient
    grad_x = np.abs(np.diff(binary_mask, axis=0, prepend=0))
    grad_y = np.abs(np.diff(binary_mask, axis=1, prepend=0))
    grad_z = np.abs(np.diff(binary_mask, axis=2, prepend=0))
    
    boundary = (grad_x + grad_y + grad_z) > 0
    
    return boundary.astype(np.float32)


def detect_junctions(mask, num_classes):
    """
    Detect multi-organ junction points
    
    Args:
        mask: Multi-class segmentation (H, W, D)
        num_classes: Number of classes
    Returns:
        junction_map: Map with junction point counts
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    junctions = np.zeros_like(mask, dtype=np.float32)
    
    # Simplified: Check 3x3x3 neighborhoods
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            for k in range(1, mask.shape[2] - 1):
                # Get 3x3x3 neighborhood
                neighborhood = mask[i-1:i+2, j-1:j+2, k-1:k+2]
                unique_classes = len(np.unique(neighborhood))
                
                # Junction if 3+ organs meet
                if unique_classes >= 3:
                    junctions[i, j, k] = unique_classes - 2
    
    return junctions


def compute_topological_complexity(mask, w_b=1.0, w_j=2.0, w_a=3.0):
    """
    Compute topological complexity score: C = w_b*B + w_j*J + w_a*A
    
    Args:
        mask: Segmentation mask
        w_b, w_j, w_a: Weights for boundaries, junctions, anomalies
    Returns:
        complexity_map: Topological complexity score per voxel
    """
    if isinstance(mask, torch.Tensor):
        device = mask.device
        mask_np = mask.cpu().numpy()
    else:
        device = None
        mask_np = mask
    
    # Get number of classes
    num_classes = int(mask_np.max()) + 1
    
    # Compute components
    boundaries = extract_boundaries(mask_np)
    junctions = detect_junctions(mask_np, num_classes)
    
    # Anomalies: simplified using persistence
    # High persistence = well-defined structure (low anomaly)
    # Low persistence = potential anomaly
    distance_map = distance_transform_edt(mask_np > 0)
    anomalies = 1.0 / (distance_map + 1.0)  # Inverse distance as proxy
    
    # Combine
    complexity = w_b * boundaries + w_j * junctions + w_a * anomalies
    
    # Normalize to [0, 1]
    if complexity.max() > 0:
        complexity = complexity / complexity.max()
    
    if device is not None:
        complexity = torch.from_numpy(complexity).float().to(device)
    
    return complexity


def compute_critical_points(distance_map):
    """
    Extract critical points from distance transform
    
    Args:
        distance_map: Distance transform of segmentation
    Returns:
        critical_coords: List of (x, y, z) coordinates
    """
    if isinstance(distance_map, torch.Tensor):
        distance_map = distance_map.cpu().numpy()
    
    # Find local maxima (peaks)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance_map, size=3)
    peaks = (distance_map == local_max) & (distance_map > 0)
    
    # Get coordinates
    critical_coords = np.argwhere(peaks)
    
    return critical_coords