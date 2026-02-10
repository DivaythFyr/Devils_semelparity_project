from dataclasses import dataclass
from typing import Optional
from torch import Tensor
import torch as th
from constants import (
    DEVICE, MAX_POP_SIZE, INIT_POP_SIZE,
    STATUS_CHILD, REPLICATION_MATRIX_INIT, NUM_OF_PROGENY,
    STATUS_JUVENILE_NO_TERR, STATUS_JUVENILE_TERR, STATUS_ADULT,
    INFECTIVITY1, INFECTIVITY2
)

@dataclass
class SimulationState:
    """
    container with mutable variables for all simulation state.
    All tensors live on DEVICE. Replaced globally after each operation.
    """
    
    # ---- POPULATION SIZE ----
    pop_size: int  # Current number of individuals
    
    # ---- SPATIAL STATE [pop_size] ----
    x: Tensor  # X coordinates in [0, MAP_X_SIZE] (move function)
    y: Tensor  # Y coordinates in [0, MAP_Y_SIZE] (move function)
    speed_x: Tensor  # X velocity component (move function, repulsion)
    speed_y: Tensor  # Y velocity component (move function, repulsion)
    
    # ---- TERRITORY STATE [pop_size] ----
    territory_center_x: Tensor  # X center of territory, -1.0 if no territory (residents only)
    territory_center_y: Tensor  # Y center of territory, -1.0 if no territory (residents only)
    
    # ---- DEMOGRAPHICS [pop_size] ----
    age: Tensor  # Age in days (TimeRunAndDeath increments, status transitions)
    sex: Tensor  # Biological sex: 0=female, 1=male (Replication, infection transmission)
    status: Tensor  # Life stage: STATUS_CHILD/JUVENILE_NO_TERR/JUVENILE_TERR/ADULT (status updates)
    fitness: Tensor  # Territory quality score 0-100 (CalculateAreaAndFitness, juvenile territory acquisition)
    
    # ---- GENETICS [pop_size, 2] ----
    chrom_sex: Tensor  # Sex chromosomes [[X1, X2], ...], 0=X, 1=Y (chrom_cal, inheritance)
    chrom_a: Tensor  # Reproductive alleles [[a1, a2], ...], 0=iteroparous, 1=semelparous (chrom_cal, death logic). Heterozygous are iteroparous [0,1] or [1,0]
    
    # ---- INFECTION STATE [pop_size] ----
    # ---- INFECTION STATE ----
    infection_stage: Tensor  # 0=healthy, 1=latent, 2=infectious, 3=terminal
    age_of_disease: Tensor  # Days since disease onset
    
    # Individual stage durations
    stage1_duration: Tensor  # Duration of latent stage (180-360 days)
    stage2_duration: Tensor  # Duration of infectious stage (90-360 days)
    stage3_duration: Tensor  # Duration of terminal stage (180-360 days)
    
    
    # ---- REPRODUCTION STATE ----
    mated_male: Tensor  # [pop_size] bool, tracks semelparous males post-mating (death day 10)
    mated_female: Tensor  # [pop_size] bool, tracks females that have mated this season
    replication_matrix: Optional[Tensor]  # [females, males] or [females] after day 10, breeding pairs (Replication, DisperseJuvenile)
    
    # ---- OFFSPRING BUFFERS ----
    # Accumulated during Replication (days 0-10), born on day 100
    new_x: Tensor  # [num_offspring] X coords of newborns (disperse offspring → add_animal)
    new_y: Tensor  # [num_offspring] Y coords of newborns (disperse offspring → add_animal)
    new_sex: Tensor  # [num_females, num_progeny, 2] sex chromosomes (chrom_cal → add_animal)
    new_a: Tensor  # [num_females, num_progeny, 2] reproductive alleles (chrom_cal → add_animal)
    
    # ---- UNIQUE IDs [pop_size] ----
    animal_ids: Tensor  # Unique int64 ID per individual (tracking, debugging, never reused)
    next_animal_id: int  # Counter for next ID assignment (AddAnimal increments)
    
    # ---- COUNTERS ----
    current_time: int  # Simulation day (main loop, yearly cycles 0-119)
    num_infection: int  # Total infected count (SeedPathogen, MovementAndInfection updates)
    dead: int  # Deaths this year, resets day 0 (TimeRunAndDeath, statistics)
    
    # ---- OPTIONAL TRACKING ----
    time_near_edge: Optional[int]  # Day when edge event occurred (legacy monitoring)
    
    # ==================== PENDING OFFSPRING ====================
    # Offspring stored during reproduction (days 0-10), born on day 100
    pending_offspring_x: Tensor              # [MAX_PENDING]
    pending_offspring_y: Tensor              # [MAX_PENDING]
    pending_offspring_sex: Tensor            # [MAX_PENDING] bool
    pending_offspring_chrom_a: Tensor        # [MAX_PENDING, 2]
    pending_offspring_chrom_sex: Tensor      # [MAX_PENDING, 2]
    pending_offspring_mother_id: Tensor      # [MAX_PENDING] int64
    pending_offspring_count: int = 0
    # ==================== END PENDING OFFSPRING ====================
    
    stoppage_reason: Optional[str] = None  # Added field for stoppage reason (e.g. "semelparous fixation", "iteroparous fixation", "max age reached")
    fixation_time: Optional[int] = None  # Added field to track when fixation was first detected


def create_initial_state(
    max_pop_size: int = MAX_POP_SIZE,
    device: th.device = DEVICE
) -> SimulationState:
    """
    Factory function: creates empty preallocated state.
    
    Args:
        max_pop_size: Maximum population size for preallocated tensors.
        device: Torch device (CPU/CUDA).
    
    Returns:
        SimulationState with all tensors preallocated and pop_size=0.
    """
    # Pending offspring storage
    max_pending = max_pop_size * NUM_OF_PROGENY
    
    return SimulationState(
        pop_size=0,
        
        # Preallocate spatial arrays to MAX_POP_SIZE, slice later
        x=th.zeros(max_pop_size, device=device),
        y=th.zeros(max_pop_size, device=device),
        speed_x=th.zeros(max_pop_size, device=device),
        speed_y=th.zeros(max_pop_size, device=device),
        
        # Territories default to -1 (no territory)
        territory_center_x=th.full((max_pop_size,), -1.0, device=device),
        territory_center_y=th.full((max_pop_size,), -1.0, device=device),
        
        # Demographics
        age=th.zeros(max_pop_size, device=device),
        sex=th.zeros(max_pop_size, dtype=th.bool, device=device),
        status=th.full((max_pop_size,), STATUS_CHILD, dtype=th.long, device=device),
        fitness=th.zeros(max_pop_size, device=device),
        
        # Genetics [pop, 2]
        chrom_sex=th.zeros((max_pop_size, 2), device=device),
        chrom_a=th.zeros((max_pop_size, 2), device=device),
        
        # Infection - NAME CORRECTED!
        infection_stage=th.zeros(max_pop_size, device=device),  
        age_of_disease=th.zeros(max_pop_size, device=device),
        
        # NEW FIELDS FOR STAGE DURATIONS
        stage1_duration=th.zeros(max_pop_size, dtype=th.float32, device=device),
        stage2_duration=th.zeros(max_pop_size, dtype=th.float32, device=device),
        stage3_duration=th.zeros(max_pop_size, dtype=th.float32, device=device),
        
        # Reproduction
        mated_male=th.zeros(max_pop_size, dtype=th.bool, device=device),
        mated_female=th.zeros(max_pop_size, dtype=th.bool, device=device),
        
        replication_matrix=REPLICATION_MATRIX_INIT,
        
        # Offspring buffers (empty until Replication)
        new_x=th.empty(0, device=device),
        new_y=th.empty(0, device=device),
        new_sex=th.empty((0, 3, 2), device=device),  # [0, NUM_OF_PROGENY, 2]
        new_a=th.empty((0, 3, 2), device=device),
        
        # IDs
        animal_ids=th.zeros(max_pop_size, dtype=th.int64, device=device),
        next_animal_id=0,
        
        # Counters
        current_time=0,
        num_infection=0,
        dead=0,
        
        # Optional
        time_near_edge=None,
        
        # Pending offspring
        pending_offspring_x=th.zeros(max_pending, device=device),
        pending_offspring_y=th.zeros(max_pending, device=device),
        pending_offspring_sex=th.zeros(max_pending, dtype=th.bool, device=device),
        pending_offspring_chrom_a=th.zeros((max_pending, 2), device=device),
        pending_offspring_chrom_sex=th.zeros((max_pending, 2), device=device),
        pending_offspring_mother_id=th.zeros(max_pending, dtype=th.int64, device=device),
        pending_offspring_count=0,
    )
    
    


@dataclass
class SimulationStats:
    time: int
    total_population: int
    infected: int
    iteroparous: int
    semelparous: int
    infection_rate: float
    adults: int
    juveniles: int
    children: int
    juveniles_no_terr: int
    juveniles_terr: int
    residents: int
    males: int
    females: int
    males_over_2_years: int
    females_over_2_years: int
    infectivity1: float
    infectivity2: float
    run_id: int
    
    

def collect_statistics(
    state: SimulationState,
    run_num: int
) -> SimulationStats:
    """
    Collects all relevant statistics from the simulation state.
    Returns a SimulationStats dataclass instance.
    """
    n: int = state.pop_size
    if n == 0:
        return SimulationStats(
            time=state.current_time,
            total_population=0,
            infected=0,
            iteroparous=0,
            semelparous=0,
            infection_rate=0.0,
            adults=0,
            juveniles=0,
            children=0,
            juveniles_no_terr=0,
            juveniles_terr=0,
            residents=0,
            males=0,
            females=0,
            males_over_2_years=0,
            females_over_2_years=0,
            infectivity1=INFECTIVITY1,
            infectivity2=INFECTIVITY2,
            run_id=run_num
        )

    infected: int = int((state.infection_stage[:n] > 0).sum().item())
    semel_mask = (state.chrom_a[:n].sum(1) == 2)
    semel_count: int = int(semel_mask.sum().item())
    itero_count: int = n - semel_count

    children_count: int = int((state.status[:n] == STATUS_CHILD).sum().item())
    juv_no_terr_count: int = int((state.status[:n] == STATUS_JUVENILE_NO_TERR).sum().item())
    juv_terr_count: int = int((state.status[:n] == STATUS_JUVENILE_TERR).sum().item())
    adult_count: int = int((state.status[:n] == STATUS_ADULT).sum().item())
    residents_count: int = juv_terr_count + adult_count

    males: int = int((state.sex[:n] == 1).sum().item())
    females: int = int((state.sex[:n] == 0).sum().item())
    males_over_2_years: int = int(((state.sex[:n] == 1) & (state.age[:n] >= 240)).sum().item())
    females_over_2_years: int = int(((state.sex[:n] == 0) & (state.age[:n] >= 240)).sum().item())
    infection_rate: float = (infected / n * 100) if n > 0 else 0.0

    return SimulationStats(
        time=state.current_time,
        total_population=n,
        infected=infected,
        iteroparous=itero_count,
        semelparous=semel_count,
        infection_rate=float(infection_rate),
        adults=adult_count,
        juveniles=juv_no_terr_count + juv_terr_count,
        children=children_count,
        juveniles_no_terr=juv_no_terr_count,
        juveniles_terr=juv_terr_count,
        residents=residents_count,
        males=males,
        females=females,
        males_over_2_years=males_over_2_years,
        females_over_2_years=females_over_2_years,
        infectivity1=INFECTIVITY1,
        infectivity2=INFECTIVITY2,
        run_id=run_num
    )
    
    
    



    