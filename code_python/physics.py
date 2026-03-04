import torch as th
from torch import Tensor
from state import SimulationState
from constants import *


def move(
    state: SimulationState,
    distance_x: Tensor,
    distance_y: Tensor,
    distance_sq: Tensor
) -> None:
    """
    Updates the positions and velocities of all individuals.
    
    Key changes from original:
    - Only individuals WITHOUT territory can move (STATUS_CHILD and STATUS_JUVENILE_NO_TERR)
    - Individuals WITH territory are completely immobile (STATUS_ADULT and STATUS_JUVENILE_TERR)
    - Territory holders still create repulsion forces (others can bounce off them)
    - No attraction to territory center since territory holders don't move
    
    Args:
        state: SimulationState object to modify
        distance_x: Pre-calculated [N, N] matrix of x-direction differences
        distance_y: Pre-calculated [N, N] matrix of y-direction differences
        distance_sq: Pre-calculated [N, N] matrix of squared distances
    """
    n = state.pop_size
    if n == 0:
        return

    # ==================== IDENTIFY MOBILE INDIVIDUALS ====================
    # Mobile: only individuals WITHOUT territory (children and juveniles without territory)
    mobile_mask = (state.status[:n] == STATUS_CHILD) | \
                  (state.status[:n] == STATUS_JUVENILE_NO_TERR)
    
    if not mobile_mask.any():
        return  # No one to move
    
    mobile_indices = th.nonzero(mobile_mask, as_tuple=True)[0]
    # ==================== END IDENTIFY MOBILE INDIVIDUALS ====================

    # ==================== STEP 1: REPULSION FROM ALL INDIVIDUALS ====================
    # Calculate forces from ALL individuals (including immobile territory holders)
    mask = distance_sq < MAX_RADIUS_SQ
    distance_cube = distance_sq ** 1.5 + distance_sq.eq(0)
    new_speed_x = (distance_x * mask / distance_cube).sum(1) * REPULSION
    new_speed_y = (distance_y * mask / distance_cube).sum(1) * REPULSION

    # Apply forces ONLY to mobile individuals
    state.speed_x[mobile_indices] += new_speed_x[mobile_indices]
    state.speed_y[mobile_indices] += new_speed_y[mobile_indices]
    # ==================== END STEP 1 ====================

    # ==================== STEP 2: REPULSION FROM WALLS ====================
    # Wall repulsion only for mobile individuals
    x_mobile = state.x[mobile_indices]
    y_mobile = state.y[mobile_indices]
    
    state.speed_x[mobile_indices] += REPULSION / (x_mobile ** 2 + 1) - \
                                     REPULSION / ((MAP_X_SIZE - x_mobile) ** 2 + 1)
    state.speed_y[mobile_indices] += REPULSION / (y_mobile ** 2 + 1) - \
                                     REPULSION / ((MAP_Y_SIZE - y_mobile) ** 2 + 1)
    # ==================== END STEP 2 ====================

    # ==================== STEP 3: ATTRACTION TO TERRITORY CENTER ====================
    # COMPLETELY REMOVED - territory holders don't move, so no attraction needed
    # Mobile individuals (no territory) don't have a territory center to be attracted to
    # ==================== END STEP 3 ====================

    # ==================== STEP 4: CLAMP SPEED AND UPDATE POSITION ====================
    # Clamp speed for mobile individuals
    speed_magnitude = th.sqrt(state.speed_x[mobile_indices] ** 2 + state.speed_y[mobile_indices] ** 2)
    speed_scale = th.clamp(MAX_SPEED / (speed_magnitude + 1e-6), max=1.0)
    state.speed_x[mobile_indices] *= speed_scale
    state.speed_y[mobile_indices] *= speed_scale

    # Update positions ONLY for mobile individuals
    state.x[mobile_indices] += state.speed_x[mobile_indices]
    state.y[mobile_indices] += state.speed_y[mobile_indices]
    # ==================== END STEP 4 ====================

    # ==================== STEP 5: ENFORCE MAP BOUNDARIES ====================
    # Boundaries only for mobile individuals
    state.x[mobile_indices].clamp_(0, MAP_X_SIZE)
    state.y[mobile_indices].clamp_(0, MAP_Y_SIZE)
    # ==================== END STEP 5 ====================

    # ==================== STEP 6: RESET SPEED FOR NEXT STEP ====================
    # Reset speed only for mobile individuals
    state.speed_x[mobile_indices] = 0.0
    state.speed_y[mobile_indices] = 0.0
    
    # Territory holders (immobile) already have speed = 0.0
    # ==================== END STEP 6 ====================




def calculate_pairwise_distances(
    x1: Tensor, y1: Tensor,
    x2: Tensor, y2: Tensor
) -> Tensor:
    """
    Calculate squared pairwise distances between two sets of points.

    Args:
        x1, y1: Tensors of shape [N] (first set of coordinates)
        x2, y2: Tensors of shape [M] (second set of coordinates)

    Returns:
        Tensor of shape [N, M] with squared distances between all pairs.
    """
    a = th.stack([x1, y1], dim=1)
    b = th.stack([x2, y2], dim=1)
    return th.cdist(a, b, p=2).pow(2)        
        
def calculate_full_distance_matrix(
    state: SimulationState
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Calculate full pairwise distance matrix for all individuals.
    
    This function should be called ONCE per timestep and the results
    reused for repulsion, fitness calculation, reproduction, etc.
    
    Args:
        state: SimulationState object
        
    Returns:
        Tuple of (distance_x, distance_y, distance_sq):
            - distance_x: [N, N] matrix of x-direction differences
            - distance_y: [N, N] matrix of y-direction differences
            - distance_sq: [N, N] matrix of squared distances
    """
    n = state.pop_size
    if n == 0:
        empty = th.empty((0, 0), device=DEVICE)
        return empty, empty, empty
    
    distance_x = state.x[:n].unsqueeze(1) - state.x[:n].unsqueeze(0)
    distance_y = state.y[:n].unsqueeze(1) - state.y[:n].unsqueeze(0)
    distance_sq = distance_x ** 2 + distance_y ** 2
    
    return distance_x, distance_y, distance_sq

def calculate_area_and_fitness(
    state: SimulationState,
    distance_sq: Tensor,
    range_sq: float = RANGE_SQ,
) -> None:
    """
    Calculate territory area and fitness for each individual.
    Modifies state.fitness in-place.

    Args:
        state: SimulationState object
        distance_sq: Pre-calculated [N, N] squared distance matrix
        range_sq: Squared range for neighbor detection (default: RANGE_SQ)
    """
    n = state.pop_size
    if n == 0:
        return

    # Clone to avoid modifying the shared distance matrix
    dist_sq = distance_sq.clone()

    # Exclude self-distance by setting diagonal to infinity
    dist_sq.fill_diagonal_(float('inf'))

    # Only consider neighbors within RANGE_SQ
    nearby_mask = dist_sq < range_sq
    nearby_count = nearby_mask.sum(dim=1).float()

    # Avoid division by zero
    nearby_count = th.clamp(nearby_count, min=1.0)

    # Fitness: inversely proportional to number of nearby neighbors
    state.fitness[:n] = 1.0 / nearby_count
    
    
def calculate_fitness_for_candidates(
    candidate_x: Tensor,
    candidate_y: Tensor,
    competitor_x: Tensor,
    competitor_y: Tensor,
    range_sq: float = RANGE_SQ,
    max_area: float = MAX_AREA
) -> Tuple[Tensor, Tensor]:
    """
    Calculate area and fitness for candidate individuals against competitors.
    Used for territory acquisition checks.

    Args:
        candidate_x: X coordinates of candidates seeking territory [num_candidates]
        candidate_y: Y coordinates of candidates seeking territory [num_candidates]
        competitor_x: X coordinates of existing territory holders [num_competitors]
        competitor_y: Y coordinates of existing territory holders [num_competitors]
        range_sq: Squared range for territory overlap (default: RANGE_SQ)
        max_area: Maximum possible territory area (default: MAX_AREA)

    Returns:
        Tuple of (area, fitness) tensors for each candidate [num_candidates]
    """
    if candidate_x.numel() == 0:
        return th.tensor([], device=candidate_x.device), th.tensor([], device=candidate_x.device)

    if competitor_x.numel() == 0:
        # No competitors, full fitness
        area = th.full((candidate_x.numel(),), max_area, device=candidate_x.device)
        fitness = th.full((candidate_x.numel(),), 100.0, device=candidate_x.device)
        return area, fitness

    # Calculate pairwise squared distances
    dist_sq = calculate_pairwise_distances(candidate_x, candidate_y, competitor_x, competitor_y)

    # Count neighbors within range
    overlap_count = (dist_sq < range_sq).sum(dim=1).float()

    # Calculate area and fitness using sigmoid formula from devils_with_kids
    # Area = (pi - overlap / 2) * RangeSq
    area = (math.pi - overlap_count / 2) * range_sq
    area = th.clamp(area, min=0.0)

    # Fitness = 100 / (1 + e^(5 - 10 * Area / MaxArea))
    fitness = 100.0 / (1.0 + th.exp(5.0 - 10.0 * area / max_area))

    return area, fitness