# file for functions that operate on SimulationState objects,
# not specialized like, for instance, infection.py

from constants import *
from state import SimulationState, SimulationStats
import torch as th
from torch import Tensor
from physics import *


def initialize_population(
    state: SimulationState,
    pop_size: int = INIT_POP_SIZE,
    min_age: int = AGE_JUVENILE_TO_ADULT,
    max_age: int = MAX_AGE,
    initial_fitness: float = INITIAL_FITNESS,
    split_genotype_by_x: bool = True,
    map_x_size: float = MAP_X_SIZE,
    map_y_size: float = MAP_Y_SIZE,
    max_speed: float = MAX_SPEED,
    device: th.device = DEVICE
) -> None:
    """
    Initialize population with adults distributed across the map.
    Doesn't return; updates state in-place.
    
    Modifies state in-place. Does not return.
    
    Args:
        state: SimulationState object to populate
        pop_size: Number of individuals to create (default: INIT_POP_SIZE)
        min_age: Minimum age in days (default: TIME_OF_MATURITY = 220)
        max_age: Maximum age in days (default: 721)
        initial_fitness: Starting fitness value (default: 100.0)
        split_genotype_by_x: If True, left half = [0,0], right half = [1,1] (default: True)
        map_x_size: Map width for coordinate distribution (default: MAP_X_SIZE)
        map_y_size: Map height for coordinate distribution (default: MAP_Y_SIZE)
        max_speed: Maximum speed magnitude per axis (default: MAX_SPEED)
        device: Torch device for tensors (default: DEVICE)
    
    Raises:
        ValueError: If pop_size exceeds MAX_POP_SIZE
    """
    if pop_size > MAX_POP_SIZE:
        raise ValueError(f"pop_size {pop_size} exceeds MAX_POP_SIZE {MAX_POP_SIZE}")
    
    # Update population count
    state.pop_size = pop_size
    n = pop_size
    
    # 1. COORDINATES — uniform random distribution
    state.x[:n] = th.rand(n, device=device) * map_x_size
    state.y[:n] = th.rand(n, device=device) * map_y_size
    
    # 2. TERRITORY CENTERS — same as initial coordinates
    state.territory_center_x[:n] = state.x[:n].clone()
    state.territory_center_y[:n] = state.y[:n].clone()
    
    # 3. AGE — random uniform in [min_age, max_age]
    state.age[:n] = th.randint(min_age, max_age + 1, (n,), device=device, dtype=state.age.dtype)
    
    # 4. FITNESS — all start at maximum
    state.fitness[:n] = initial_fitness
    
    # 5. SEX — 50/50 female (False) / male (True)
    state.sex[:n] = th.rand(n, device=device) >= INITIAL_SEX_RATIO
    
    # 6. SEX CHROMOSOMES — [X, X] for females, [X, Y] for males
    # Column 0: always X (0) from mother
    # Column 1: X (0) for females, Y (1) for males
    state.chrom_sex[:n, 0] = 0
    state.chrom_sex[:n, 1] = state.sex[:n].float()
    
    # 7. REPRODUCTION GENOTYPE — split by map position
    if split_genotype_by_x:
        mid_x = map_x_size / 2.0
        left_mask = state.x[:n] <= mid_x
        right_mask = ~left_mask
        
        # Left side: iteroparous [0, 0]
        state.chrom_a[:n][left_mask] = 0.0
        # Right side: semelparous [1, 1]
        state.chrom_a[:n][right_mask] = 1.0
    else:
        # All iteroparous by default
        state.chrom_a[:n] = 0.0
    
    # 8. SPEED — random in [-max_speed, +max_speed]
    state.speed_x[:n] = (th.rand(n, device=device) * 2.0 - 1.0) * max_speed
    state.speed_y[:n] = (th.rand(n, device=device) * 2.0 - 1.0) * max_speed
    
    # 9. STATUS — all adults
    state.status[:n] = STATUS_ADULT
    
    # 10. INFECTION — all healthy
    state.infection_stage[:n] = 0
    state.age_of_disease[:n] = 0
    
    # 11. MATING FLAG — none have mated
    state.mated_male[:n] = False
    
    # 12. UNIQUE IDs
    state.animal_ids[:n] = th.arange(0, n, dtype=th.int64, device=device)
    state.next_animal_id = n
    
    # 13. COUNTERS
    state.current_time = 0
    state.num_infection = 0
    state.dead = 0
    

def add_animal(
    state: SimulationState,
    x: Tensor,
    y: Tensor,
    age: Tensor,
    sex: Tensor,
    chrom_sex: Tensor,
    chrom_a: Tensor,
    fitness: Tensor,
    status: Tensor,
    infection_stage: Tensor,  
    age_of_disease: Tensor,
    mated_male: Tensor,
    device: th.device = DEVICE
) -> None:
    """
    Add one or more new individuals to the population in a vectorized way.
    If adding all would exceed MAX_POP_SIZE, only add as many as possible.
    All input tensors must have the same length (number of new individuals).
    Modifies state in-place.

    Args:
        state: SimulationState object to update
        x, y: Coordinates of new individuals [num_new]
        age: Ages of new individuals [num_new]
        sex: Sex of new individuals [num_new] (bool: False=female, True=male)
        chrom_sex: Sex chromosomes [num_new, 2]
        chrom_a: Reproductive alleles [num_new, 2]
        fitness: Fitness values [num_new]
        status: Status codes [num_new]
        infection_stage: Infection stage [num_new] (0=healthy, 1=latent, 2=infectious, 3=terminal)
        age_of_disease: Age of disease [num_new]
        mated_male: Mated flag [num_new]
        device: Torch device (default: DEVICE)
    """
    num_new = x.shape[0]
    start = state.pop_size
    end = min(start + num_new, MAX_POP_SIZE)
    actual_new = end - start
    if actual_new <= 0:
        return  # No space to add any new individuals

    # Assign new values (only up to available space)
    state.x[start:end] = x[:actual_new]
    state.y[start:end] = y[:actual_new]
    state.age[start:end] = age[:actual_new]
    state.sex[start:end] = sex[:actual_new]
    state.chrom_sex[start:end] = chrom_sex[:actual_new]
    state.chrom_a[start:end] = chrom_a[:actual_new]
    state.fitness[start:end] = fitness[:actual_new]
    state.status[start:end] = status[:actual_new]
    state.infection_stage[start:end] = infection_stage[:actual_new]  # CHANGED
    state.age_of_disease[start:end] = age_of_disease[:actual_new]
    state.mated_male[start:end] = mated_male[:actual_new]
    
    # NEW FIELDS: initialize with zeros (will be set upon infection)
    state.stage1_duration[start:end] = 0.0
    state.stage2_duration[start:end] = 0.0
    state.stage3_duration[start:end] = 0.0

    # Territory centers: set to initial position
    state.territory_center_x[start:end] = x[:actual_new]
    state.territory_center_y[start:end] = y[:actual_new]

    # Speed: start at zero
    state.speed_x[start:end] = 0.0
    state.speed_y[start:end] = 0.0

    # Unique IDs
    state.animal_ids[start:end] = th.arange(state.next_animal_id, state.next_animal_id + actual_new, device=device)
    state.next_animal_id += actual_new

    # Update population size
    state.pop_size += actual_new
    
 

# In simulation_core.py, modifying delete_animal:

def delete_animal(
    state: SimulationState,
    indices_to_remove: Tensor,
    day_in_year: int = -1,
    cleanup_memory: bool = False
) -> None:
    """
    Remove individuals with optional memory cleanup.
    
    Args:
        state: SimulationState object to update
        indices_to_remove: 1D tensor of indices to remove
        day_in_year: Current day in year cycle
        cleanup_memory: If True, zeros out data of the deceased (frees memory)
    """
    n_remove = indices_to_remove.shape[0]
    if n_remove == 0:
        return

    n = state.pop_size

    # ==================== FILTER PENDING OFFSPRING ====================
    if 0 <= day_in_year < TIME_OF_DISPOSAL:
        if state.pending_offspring_count > 0:
            dead_ids = state.animal_ids[indices_to_remove]
            pending_mother_ids = state.pending_offspring_mother_id[:state.pending_offspring_count]
            mother_alive_mask = ~th.isin(pending_mother_ids, dead_ids)
            
            if not mother_alive_mask.all():
                keep_indices = th.nonzero(mother_alive_mask, as_tuple=True)[0]
                new_count = keep_indices.shape[0]
                
                if new_count > 0:
                    state.pending_offspring_x[:new_count] = state.pending_offspring_x[keep_indices]
                    state.pending_offspring_y[:new_count] = state.pending_offspring_y[keep_indices]
                    state.pending_offspring_sex[:new_count] = state.pending_offspring_sex[keep_indices]
                    state.pending_offspring_chrom_a[:new_count] = state.pending_offspring_chrom_a[keep_indices]
                    state.pending_offspring_chrom_sex[:new_count] = state.pending_offspring_chrom_sex[keep_indices]
                    state.pending_offspring_mother_id[:new_count] = state.pending_offspring_mother_id[keep_indices]
                
                state.pending_offspring_count = new_count
    # ==================== END FILTER PENDING OFFSPRING ====================

    # ==================== MEMORY CLEANUP FOR THE DECEASED ====================
    if cleanup_memory and n_remove > 0:
        # ZERO OUT data of deceased individuals
        state.x[indices_to_remove] = 0.0
        state.y[indices_to_remove] = 0.0
        state.speed_x[indices_to_remove] = 0.0
        state.speed_y[indices_to_remove] = 0.0
        state.territory_center_x[indices_to_remove] = -1.0
        state.territory_center_y[indices_to_remove] = -1.0
        state.age[indices_to_remove] = 0
        state.sex[indices_to_remove] = False
        state.fitness[indices_to_remove] = 0.0
        state.chrom_sex[indices_to_remove] = 0
        state.chrom_a[indices_to_remove] = 0
        state.infection_stage[indices_to_remove] = 0  # CHANGED
        state.age_of_disease[indices_to_remove] = 0
        state.stage1_duration[indices_to_remove] = 0.0  # CHANGED: phase1 → stage1
        state.stage2_duration[indices_to_remove] = 0.0  # CHANGED: phase2 → stage2
        state.stage3_duration[indices_to_remove] = 0.0  # NEW
        state.mated_male[indices_to_remove] = False
        state.mated_female[indices_to_remove] = False
    # ==================== END MEMORY CLEANUP ====================

    # ==================== REMOVE ANIMALS ====================
    keep_mask = th.ones(n, dtype=th.bool, device=indices_to_remove.device)
    keep_mask[indices_to_remove] = False
    keep_indices = th.nonzero(keep_mask, as_tuple=True)[0]

    new_n = n - n_remove
    
    # Copy survivors to the beginning of arrays
    state.x[:new_n] = state.x[keep_indices]
    state.y[:new_n] = state.y[keep_indices]
    state.age[:new_n] = state.age[keep_indices]
    state.sex[:new_n] = state.sex[keep_indices]
    state.chrom_sex[:new_n] = state.chrom_sex[keep_indices]
    state.chrom_a[:new_n] = state.chrom_a[keep_indices]
    state.fitness[:new_n] = state.fitness[keep_indices]
    state.status[:new_n] = state.status[keep_indices]
    state.infection_stage[:new_n] = state.infection_stage[keep_indices]  # CHANGED
    state.age_of_disease[:new_n] = state.age_of_disease[keep_indices]
    state.stage1_duration[:new_n] = state.stage1_duration[keep_indices]  # CHANGED
    state.stage2_duration[:new_n] = state.stage2_duration[keep_indices]  # CHANGED
    state.stage3_duration[:new_n] = state.stage3_duration[keep_indices]  # NEW
    state.mated_male[:new_n] = state.mated_male[keep_indices]
    state.mated_female[:new_n] = state.mated_female[keep_indices]
    state.territory_center_x[:new_n] = state.territory_center_x[keep_indices]
    state.territory_center_y[:new_n] = state.territory_center_y[keep_indices]
    state.speed_x[:new_n] = state.speed_x[keep_indices]
    state.speed_y[:new_n] = state.speed_y[keep_indices]
    state.animal_ids[:new_n] = state.animal_ids[keep_indices]

    state.pop_size = new_n
    # ==================== END REMOVE ANIMALS ====================


    
def disperse_offspring(
    state: SimulationState,
    day_in_year: int,
    time_of_disposal: int = TIME_OF_DISPOSAL,
    age_child_after_disposal: int = AGE_CHILD_AFTER_DISPOSAL,
    status: int = STATUS_CHILD
) -> None:
    """
    Disperse all children who have reached the dispersal age.
    Sets their status to 'juvenile without territory', resets their territory center,
    and (optionally) randomizes their position on the map.

    Args:
        state: SimulationState object to update in-place.
        day_in_year: Current day in the annual cycle.
        time_of_disposal: Day of the year at which the function work and children go from mothers.
        age_child_after_disposal: Age to set for dispersed children (default: AGE_CHILD_AFTER_DISPOSAL).
        status: Status code to assign to dispersed offspring (default: STATUS_CHILD).
    """
    if day_in_year == time_of_disposal:
        # Find indices of children at dispersal age and with child status
        n = state.pop_size
        # Only those with status == child and age in [min_age, max_age)
        to_disperse: th.Tensor = th.nonzero(
            (state.status[:n] == status),
            as_tuple=True
        )[0]

        if to_disperse.numel() == 0:
            return

        # Set status.
        state.status[to_disperse] = status
        
        # Set age to age_child_after_disposal for dispersed children
        state.age[to_disperse] = age_child_after_disposal

        # Reset territory center (no territory yet)
        state.territory_center_x[to_disperse] = 0.0
        state.territory_center_y[to_disperse] = 0.0

        # Optionally: reset speed
        state.speed_x[to_disperse] = 0.0
        state.speed_y[to_disperse] = 0.0
        
def add_pending_offspring(
    state: SimulationState,
    x: Tensor,
    y: Tensor,
    sex: Tensor,
    chrom_a: Tensor,
    chrom_sex: Tensor,
    mother_id: Tensor
) -> None:
    """
    Add offspring to pending list. They will be born on day 100 (TIME_OF_DISPOSAL).
    Called during reproduction (days 0-10).
    
    Args:
        state: SimulationState object to update in-place.
        x, y: Coordinates (same as mother's position).
        sex: Sex of offspring [num_new] (bool).
        chrom_a: Reproductive alleles [num_new, 2].
        chrom_sex: Sex chromosomes [num_new, 2].
        mother_id: ID of mother for each offspring [num_new].
    """
    num_new = x.shape[0]
    if num_new == 0:
        return
    
    start = state.pending_offspring_count
    max_pending = state.pending_offspring_x.shape[0]
    end = min(start + num_new, max_pending)
    actual_new = end - start
    
    if actual_new <= 0:
        return  # No space
    
    state.pending_offspring_x[start:end] = x[:actual_new]
    state.pending_offspring_y[start:end] = y[:actual_new]
    state.pending_offspring_sex[start:end] = sex[:actual_new]
    state.pending_offspring_chrom_a[start:end] = chrom_a[:actual_new]
    state.pending_offspring_chrom_sex[start:end] = chrom_sex[:actual_new]
    state.pending_offspring_mother_id[start:end] = mother_id[:actual_new]
    
    state.pending_offspring_count += actual_new


def birth_pending_offspring(
    state: SimulationState,
    day_in_year: int,
    birth_day: int = TIME_OF_DISPOSAL
) -> None:
    """
    On day 100, convert all pending offspring to actual children.
    
    Args:
        state: SimulationState object to update in-place.
        day_in_year: Current day in the annual cycle.
        birth_day: Day when pending offspring are born (default: TIME_OF_DISPOSAL = 100).
    """
    if day_in_year != birth_day:
        return
    
    count = state.pending_offspring_count
    if count == 0:
        return
    
    device = state.x.device
    
    add_animal(
        state,
        x=state.pending_offspring_x[:count].clone(),
        y=state.pending_offspring_y[:count].clone(),
        age=th.zeros(count, dtype=state.age.dtype, device=device),
        sex=state.pending_offspring_sex[:count].clone(),
        chrom_sex=state.pending_offspring_chrom_sex[:count].clone(),
        chrom_a=state.pending_offspring_chrom_a[:count].clone(),
        fitness=th.full((count,), INITIAL_FITNESS, device=device),
        status=th.full((count,), STATUS_CHILD, dtype=state.status.dtype, device=device),
        infection_stage=th.zeros(count, dtype=state.infection_stage.dtype, device=device),
        age_of_disease=th.zeros(count, dtype=state.age_of_disease.dtype, device=device),
        mated_male=th.zeros(count, dtype=th.bool, device=device),
        device=device
    )
    
    # Clear pending offspring
    state.pending_offspring_count = 0

    
def transition_child_to_juvenile_no_terr(
    state: SimulationState,
    transition_age: int = AGE_CHILD_TO_JUVENILE
) -> None:
    """
    Transition all children who have reached the specified age to juveniles without territory.
    Sets their status and resets territory center.

    Args:
        state: SimulationState object to update in-place.
        transition_age: Age at which children become juveniles without territory (default: AGE_CHILD_TO_JUVENILE).
    """
    n = state.pop_size
    mask = (
        (state.status[:n] == STATUS_CHILD) &
        (state.age[:n] >= transition_age)
    )
    indices = th.nonzero(mask, as_tuple=True)[0]
    if indices.numel() == 0:
        return

    state.status[indices] = STATUS_JUVENILE_NO_TERR
    state.territory_center_x[indices] = 0.0
    state.territory_center_y[indices] = 0.0
    state.speed_x[indices] = 0.0
    state.speed_y[indices] = 0.0
    
    
def transition_juvenile_no_terr_to_terr(
    state: SimulationState,
    fitness_threshold: float = MIN_FITNESS_JUVENILE_FOR_TERR
) -> None:
    """
    Transition eligible juveniles without territory to juveniles with territory.
    A juvenile can acquire territory only if their Fitness > fitness_threshold.
    
    Uses fully vectorized approach: calculates fitness for all candidates at once
    against current territory holders, without sequential recalculation.
    This matches the logic in devils_with_kids.py.

    Args:
        state: SimulationState object to update in-place.
        fitness_threshold: Minimum fitness required to acquire territory.
    """
    n = state.pop_size
    if n == 0:
        return

    # Find juveniles without territory
    juv_no_terr_mask = state.status[:n] == STATUS_JUVENILE_NO_TERR
    juv_no_terr_indices = th.nonzero(juv_no_terr_mask, as_tuple=True)[0]

    if juv_no_terr_indices.numel() == 0:
        return

    # Get current territory holders (adults + juveniles with territory)
    has_territory_mask = (state.status[:n] == STATUS_ADULT) | (state.status[:n] == STATUS_JUVENILE_TERR)

    if has_territory_mask.sum() == 0:
        # No competition, all juveniles can get territory
        successful_indices = juv_no_terr_indices
    else:
        # Positions of territory holders
        terr_x = state.x[:n][has_territory_mask]
        terr_y = state.y[:n][has_territory_mask]

        # Positions of juveniles seeking territory
        juv_x = state.x[juv_no_terr_indices]
        juv_y = state.y[juv_no_terr_indices]

        # Calculate fitness for all candidates at once (vectorized)
        _, fitness = calculate_fitness_for_candidates(juv_x, juv_y, terr_x, terr_y)

        # Find juveniles with fitness > threshold
        can_get_territory = fitness > fitness_threshold
        successful_indices = juv_no_terr_indices[can_get_territory]

    if successful_indices.numel() == 0:
        return

    # Update status
    state.status[successful_indices] = STATUS_JUVENILE_TERR

    # Set territory center to current position
    state.territory_center_x[successful_indices] = state.x[successful_indices].clone()
    state.territory_center_y[successful_indices] = state.y[successful_indices].clone()
    
        
def transition_juvenile_terr_to_adult(
    state: SimulationState,
    maturity_age: int = AGE_JUVENILE_TO_ADULT
) -> None:
    """
    Transition all juveniles with territory who have reached maturity age to adults.
    Updates their status accordingly.

    Args:
        state: SimulationState object to update in-place.
        maturity_age: Age at which juveniles with territory become adults (default: AGE_JUVENILE_TO_ADULT).
    """
    n = state.pop_size
    mask = (
        (state.status[:n] == STATUS_JUVENILE_TERR) &
        (state.age[:n] >= maturity_age)
    )
    indices = th.nonzero(mask, as_tuple=True)[0]
    if indices.numel() == 0:
        return

    state.status[indices] = STATUS_ADULT
        
    

def process_all_deaths(
    state: SimulationState,
    day_in_year: int,
    base_mortality: float = MORTALITY,
    disease_mortality_factor_stage1: float = DISEASE_MORTALITY_FACTOR_STAGE1,
    disease_mortality_factor_stage2: float = DISEASE_MORTALITY_FACTOR_STAGE2,
    disease_mortality_factor_stage3: float = DISEASE_MORTALITY_FACTOR_STAGE3,
    dispersal_deadline: int = DISPERSAL_DEADLINE,
    maturity_age: int = AGE_JUVENILE_TO_ADULT,
    semelparous_death_day: int = 10,
    device: th.device = DEVICE
) -> None:
    """
    Обработка смертей с реалистичной логикой болезни:
    - Стадия 1 (латентная): до 2% в день, большинство выживает
    - Стадия 2 (инфекционная): до 10% в день, значительная смертность
    - Стадия 3 (терминальная): до 20% в день + 100% смертность в конце
    """
    n: int = state.pop_size
    if n == 0:
        return
    
    # ==================== INITIALIZE DEATH MASK ====================
    death_mask: Tensor = th.zeros(n, dtype=th.bool, device=device)
    
    # ==================== 1. DEATH BY AGE ====================
    age_death_mask: Tensor = state.age[:n] > MAX_AGE
    death_mask |= age_death_mask
    
    # ==================== 2. DEATH BY BASE MORTALITY ====================
    rand_mortality: Tensor = th.rand(n, device=device)
    base_mortality_mask: Tensor = rand_mortality < base_mortality
    death_mask |= base_mortality_mask
    
    # ==================== 3. СМЕРТНОСТЬ ОТ БОЛЕЗНИ ====================
    # ----- Стадия 1 (латентная) - низкая смертность -----
    stage1_mask = state.infection_stage[:n] == INFECTION_STAGE_LATENT
    
    if stage1_mask.any():
        stage1_indices = th.nonzero(stage1_mask, as_tuple=True)[0]
        age_of_disease = state.age_of_disease[stage1_indices].float()
        stage1_duration = state.stage1_duration[stage1_indices]
        
        # Формула: factor / (1 + exp(10 * (duration/2 + duration/3 - age) / duration))
        valid_mask = stage1_duration > 0
        if valid_mask.any():
            valid_indices = stage1_indices[valid_mask]
            valid_age = age_of_disease[valid_mask]
            valid_duration = stage1_duration[valid_mask]
            
            sigmoid_arg = 10.0 * ((valid_duration/2.0) + (valid_duration/3.0) - valid_age) / valid_duration
            disease_death_rate = disease_mortality_factor_stage1 / (1.0 + th.exp(sigmoid_arg))
            
            rand_disease = th.rand(valid_indices.shape[0], device=device)
            disease_death = valid_indices[rand_disease < disease_death_rate]
            
            if disease_death.numel() > 0:
                death_mask[disease_death] = True
    
    # ----- Стадия 2 (инфекционная) - средняя смертность -----
    stage2_mask = state.infection_stage[:n] == INFECTION_STAGE_INFECTIOUS
    
    if stage2_mask.any():
        stage2_indices = th.nonzero(stage2_mask, as_tuple=True)[0]
        age_of_disease = state.age_of_disease[stage2_indices].float()
        stage1_duration = state.stage1_duration[stage2_indices]
        stage2_duration = state.stage2_duration[stage2_indices]
        
        # Возраст в текущей стадии = общий возраст болезни - длительность стадии 1
        age_in_stage2 = age_of_disease - stage1_duration
        
        valid_mask = stage2_duration > 0
        if valid_mask.any():
            valid_indices = stage2_indices[valid_mask]
            valid_age = age_in_stage2[valid_mask]
            valid_duration = stage2_duration[valid_mask]
            
            # Отсекаем отрицательный возраст (если age_of_disease < stage1_duration)
            positive_age_mask = valid_age >= 0
            if positive_age_mask.any():
                final_indices = valid_indices[positive_age_mask]
                final_age = valid_age[positive_age_mask]
                final_duration = valid_duration[positive_age_mask]
                
                sigmoid_arg = 10.0 * ((final_duration/2.0) + (final_duration/3.0) - final_age) / final_duration
                disease_death_rate = disease_mortality_factor_stage2 / (1.0 + th.exp(sigmoid_arg))
                
                rand_disease = th.rand(final_indices.shape[0], device=device)
                disease_death = final_indices[rand_disease < disease_death_rate]
                
                if disease_death.numel() > 0:
                    death_mask[disease_death] = True
    
    # ----- Стадия 3 (терминальная) - высокая смертность -----
    stage3_mask = state.infection_stage[:n] == INFECTION_STAGE_TERMINAL
    
    if stage3_mask.any():
        stage3_indices = th.nonzero(stage3_mask, as_tuple=True)[0]
        age_of_disease = state.age_of_disease[stage3_indices].float()
        stage1_duration = state.stage1_duration[stage3_indices]
        stage2_duration = state.stage2_duration[stage3_indices]
        stage3_duration = state.stage3_duration[stage3_indices]
        
        # 1. Ежедневная повышенная смертность (до 20%)
        valid_mask = stage3_duration > 0
        if valid_mask.any():
            valid_indices = stage3_indices[valid_mask]
            valid_age_of_disease = age_of_disease[valid_mask]
            valid_stage3_duration = stage3_duration[valid_mask]
            
            # Возраст в стадии 3 = общий возраст - (стадия1 + стадия2)
            age_in_stage3 = valid_age_of_disease - (stage1_duration[valid_mask] + stage2_duration[valid_mask])
            
            positive_age_mask = age_in_stage3 >= 0
            if positive_age_mask.any():
                final_indices = valid_indices[positive_age_mask]
                final_age = age_in_stage3[positive_age_mask]
                final_duration = valid_stage3_duration[positive_age_mask]
                
                sigmoid_arg = 10.0 * ((final_duration/2.0) + (final_duration/3.0) - final_age) / final_duration
                disease_death_rate = disease_mortality_factor_stage3 / (1.0 + th.exp(sigmoid_arg))
                
                rand_disease = th.rand(final_indices.shape[0], device=device)
                disease_death = final_indices[rand_disease < disease_death_rate]
                
                if disease_death.numel() > 0:
                    death_mask[disease_death] = True
        
        # 2. 100% смертность в конце терминальной стадии
        total_duration = stage1_duration + stage2_duration + stage3_duration
        death_at_end = age_of_disease >= total_duration
        
        if death_at_end.any():
            dying_indices = stage3_indices[death_at_end]
            death_mask[dying_indices] = True
    
    # ==================== 4. DEATH BY NO TERRITORY ====================
    juv_no_terr_mask: Tensor = state.status[:n] == STATUS_JUVENILE_NO_TERR
    
    if juv_no_terr_mask.any():
        ages: Tensor = state.age[:n]
        fitness: Tensor = state.fitness[:n]
        
        # Mandatory death at maturity age
        mandatory_death_mask: Tensor = juv_no_terr_mask & (ages >= maturity_age)
        death_mask |= mandatory_death_mask
        
        # Probabilistic death between dispersal_deadline and maturity_age
        prob_death_candidates: Tensor = juv_no_terr_mask & (ages > dispersal_deadline) & (ages < maturity_age)
        
        if prob_death_candidates.any():
            # fitness_factor: lower fitness = higher death chance (0 to 1)
            fitness_factor: Tensor = th.clamp(1.0 - (fitness / 100.0), min=0.0, max=1.0)
            
            # age_over_deadline: normalized time past deadline (0 to 1)
            age_range: float = float(maturity_age - dispersal_deadline)
            age_over_deadline: Tensor = (ages.float() - dispersal_deadline) / age_range
            age_over_deadline = th.clamp(age_over_deadline, min=0.0, max=1.0)
            
            # Death probability = fitness_factor * age_over_deadline
            death_prob: Tensor = fitness_factor * age_over_deadline
            
            rand_terr: Tensor = th.rand(n, device=device)
            prob_death_mask: Tensor = prob_death_candidates & (rand_terr < death_prob)
            death_mask |= prob_death_mask
    
    # ==================== 5. DEATH BY REPRODUCTION SEMELPARITY ====================
    if day_in_year == semelparous_death_day:
        # Semelparous: both alleles are 1 (sum == 2)
        is_semelparous: Tensor = state.chrom_a[:n].sum(dim=1) == 2
        
        # Males: sex == True (1)
        is_male: Tensor = state.sex[:n] == True
        
        # Mated this breeding season
        has_mated: Tensor = state.mated_male[:n] == True
        
        semelparous_death_mask: Tensor = is_semelparous & is_male & has_mated
        death_mask |= semelparous_death_mask
    
    # ==================== 6. APPLY ALL DEATHS ====================
    if death_mask.any():
        indices_to_remove = th.nonzero(death_mask, as_tuple=True)[0]
        delete_animal(state, indices_to_remove, day_in_year=day_in_year)



def check_simulation_stop(
    state: SimulationState,
    current_time: int,
    max_time: int = 42000,
    max_age: int = MAX_AGE
) -> bool:
    """
    Check whether the simulation should stop.

    Conditions:
        1. Population extinct → stop immediately (reason: "extinction").
        2. 100% semelparous fixation → stop after MAX_AGE * 2 additional timesteps (reason: "semelparous").
        3. 100% iteroparous fixation → stop after MAX_AGE * 2 additional timesteps (reason: "iteroparous").
        4. Reached maximum simulation time → stop (reason: "out_of_time").
        5. Otherwise → continue.

    Args:
        state: SimulationState object.
        current_time: Current simulation timestep.
        max_time: Maximum number of simulation timesteps (default: TIMEPOINTS).
        max_age: Maximum age constant used to calculate the waiting period.

    Returns:
        should_stop (bool): True if simulation should end.
    """
    pop_size: int = state.pop_size
    waiting_period: int = max_age * 2

    # --- Condition 1: Extinction ---
    if pop_size == 0:
        state.stoppage_reason = "extinction"
        return True

    # --- Condition 4: Out of time ---
    if current_time >= max_time - 1:
        state.stoppage_reason = "outOfTime"
        return True

    # --- Calculate genotype fractions ---
    chrom_sum: th.Tensor = state.chrom_a[:pop_size].sum(dim=1)
    semel_count: int = int((chrom_sum == 2).sum().item())
    iterop_count: int = int((chrom_sum == 0).sum().item())

    semel_frac: float = semel_count / pop_size
    iterop_frac: float = iterop_count / pop_size

    is_fixed: bool = (semel_frac >= SEMELPAROUS_WIN_PROPORTION) or (iterop_frac >= ITEROPAROUS_WIN_PROPORTION)

    if is_fixed:
        # First time fixation is detected — record the timestep
        if state.fixation_time is None:
            state.fixation_time = current_time

        # --- Conditions 2 & 3: Fixation + waited long enough ---
        elapsed: int = current_time - state.fixation_time
        if elapsed >= waiting_period:
            fixation_type: str = "semelparous" if semel_frac >= SEMELPAROUS_WIN_PROPORTION else "iteroparous"
            state.stoppage_reason = fixation_type
            return True
    else:
        # Fixation was lost (e.g. mutation or offspring changed ratio) — reset
        state.fixation_time = None

    # --- Condition 5: Continue ---
    return False