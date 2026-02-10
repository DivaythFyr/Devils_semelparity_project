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
    Updates the positions and velocities of all individuals in the simulation for one time step.

    Implements the following mechanics, step by step:

    1. **Repulsion from All Individuals:**  
       Calculates pairwise repulsion forces between ALL individuals within a specified radius.

    2. **Wall Repulsion:**  
       Applies repulsion forces to individuals near the map boundaries (walls).

    3. **Territory Attraction (Residents Only):**  
       Applies an attraction force to residents only, pulling them toward the center of their territory.

    4. **Movement:**  
       Updates positions of all individuals based on their velocities (with speed clamping).

    5. **Border Correction:**  
       Reflects individuals off the map boundaries and clamps positions within the map.

    6. **Territory Boundary Enforcement (Residents Only):**  
       Ensures that residents remain within the radius of their assigned territory.

    Modifies the state in-place. Does not return a value.
    
    Args:
        state: SimulationState object to modify
        distance_x: Pre-calculated [N, N] matrix of x-direction differences
        distance_y: Pre-calculated [N, N] matrix of y-direction differences
        distance_sq: Pre-calculated [N, N] matrix of squared distances
    """
    n = state.pop_size
    if n == 0:
        return

    resident_mask = (state.status[:n] == STATUS_ADULT) | (state.status[:n] == STATUS_JUVENILE_TERR)

    # ==================== STEP 1: REPULSION FROM ALL INDIVIDUALS ====================
    # Uses pre-calculated distance matrices (no recalculation needed)
    mask = distance_sq < MAX_RADIUS_SQ
    distance_cube = distance_sq ** 1.5 + distance_sq.eq(0)
    new_speed_x = (distance_x * mask / distance_cube).sum(1) * REPULSION
    new_speed_y = (distance_y * mask / distance_cube).sum(1) * REPULSION

    state.speed_x[:n] += new_speed_x
    state.speed_y[:n] += new_speed_y

    # ==================== STEP 2: REPULSION FROM WALLS ====================
    # OPTIMIZED: Reduced from 4 intermediate tensors to 2 local references
    x = state.x[:n]
    y = state.y[:n]
    state.speed_x[:n] += REPULSION / (x ** 2 + 1) - REPULSION / ((MAP_X_SIZE - x) ** 2 + 1)
    state.speed_y[:n] += REPULSION / (y ** 2 + 1) - REPULSION / ((MAP_Y_SIZE - y) ** 2 + 1)

    # ==================== STEP 3: ATTRACTION TO TERRITORY CENTER ====================
    has_territory = resident_mask & (state.territory_center_x[:n] >= 0)
    
    if has_territory.any():
        terr_indices = th.nonzero(has_territory, as_tuple=True)[0]
        
        dist_to_center_x = state.territory_center_x[terr_indices] - state.x[terr_indices]
        dist_to_center_y = state.territory_center_y[terr_indices] - state.y[terr_indices]
        dist_to_center_sq = dist_to_center_x ** 2 + dist_to_center_y ** 2
        
        outside_range = dist_to_center_sq > RANGE_SQ
        
        if outside_range.any():
            outside_indices = terr_indices[outside_range]
            # OPTIMIZED: Reuse already-calculated distances instead of recalculating
            dist_x_outside = dist_to_center_x[outside_range]
            dist_y_outside = dist_to_center_y[outside_range]
            dist_outside = th.sqrt(dist_to_center_sq[outside_range]) + 1e-6
            
            attraction_strength = REPULSION * 0.1
            state.speed_x[outside_indices] += attraction_strength * dist_x_outside / dist_outside
            state.speed_y[outside_indices] += attraction_strength * dist_y_outside / dist_outside

    # ==================== STEP 4: CLAMP SPEED AND UPDATE POSITION (IMPLEMENT MOVEMENT) ====================
    speed_magnitude = th.sqrt(state.speed_x[:n] ** 2 + state.speed_y[:n] ** 2)
    speed_scale = th.clamp(MAX_SPEED / (speed_magnitude + 1e-6), max=1.0)
    state.speed_x[:n] *= speed_scale
    state.speed_y[:n] *= speed_scale

    state.x[:n] += state.speed_x[:n]
    state.y[:n] += state.speed_y[:n]

    # ==================== STEP 5: ENFORCE MAP BOUNDARIES ====================
    # OPTIMIZED: Use in-place clamp to avoid tensor allocation
    state.x[:n].clamp_(0, MAP_X_SIZE)
    state.y[:n].clamp_(0, MAP_Y_SIZE)

    # ==================== STEP 6: RESET SPEED FOR NEXT STEP ====================
    state.speed_x[:n] = 0.0
    state.speed_y[:n] = 0.0



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