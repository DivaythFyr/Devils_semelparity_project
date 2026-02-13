import torch as th
from state import SimulationState
from constants import *
from torch import Tensor
from physics import *
from simulation_core import *


def calculate_chrom(
    state: SimulationState,
    female_indices: Tensor,
    male_indices: Tensor,
    num_progeny: int = NUM_OF_PROGENY,
    device: th.device = DEVICE
) -> tuple[Tensor, Tensor]:
    """
    Calculate offspring sex chromosomes and reproductive alleles for each mating pair.
    Each child inherits chromosomes/alleles randomly and independently.

    Args:
        state: SimulationState object containing parent genetics
        female_indices: Tensor of indices for mated females
        male_indices: Tensor of indices for selected males (same length as female_indices)
        num_progeny: Number of offspring per pair (default: NUM_OF_PROGENY)
        device: Torch device (default: DEVICE)

    Returns:
        Tuple of:
            - sex_chroms: Tensor [num_females, num_progeny, 2] (offspring sex chromosomes)
            - repro_alleles: Tensor [num_females, num_progeny, 2] (offspring reproductive alleles). Meaning alleles that influence reproduction: iteroparous, semelparous.
    """

    # Get parent chromosomes
    female_sex_chrom = state.chrom_sex[female_indices]      # [n_females, 2]
    male_sex_chrom = state.chrom_sex[male_indices]          # [n_females, 2]
    female_repro_allele = state.chrom_a[female_indices]     # [n_females, 2]
    male_repro_allele = state.chrom_a[male_indices]         # [n_females, 2]

    n_females = female_sex_chrom.shape[0]

    # Generate random indices for each progeny (0 or 1)
    mother_sex_idx = th.randint(0, 2, (n_females, num_progeny), device=device)
    father_sex_idx = th.randint(0, 2, (n_females, num_progeny), device=device)
    mother_allele_idx = th.randint(0, 2, (n_females, num_progeny), device=device)
    father_allele_idx = th.randint(0, 2, (n_females, num_progeny), device=device)

    # Use gather to select chromosomes/alleles for each progeny
    mother_sex = female_sex_chrom.gather(1, mother_sex_idx)
    father_sex = male_sex_chrom.gather(1, father_sex_idx)
    sex_chroms = th.stack([mother_sex, father_sex], dim=2)  # [n_females, num_progeny, 2]

    mother_alleles = female_repro_allele.gather(1, mother_allele_idx)
    father_alleles = male_repro_allele.gather(1, father_allele_idx)
    repro_alleles = th.stack([mother_alleles, father_alleles], dim=2)  # [n_females, num_progeny, 2]

    return sex_chroms, repro_alleles


def replication(
    state: SimulationState,
    distance_sq: Tensor,
    day_in_year: int,
    breeding_days: int = BREEDING_DAYS,
    num_progeny: int = NUM_OF_PROGENY,
    device: th.device = DEVICE
) -> None:
    """
    Handles sexual reproduction for the breeding season.

    - Resets mated status at start of breeding season (day 0).
    - Finds eligible adult females (unmated) and males.
    - Uses the pre-calculated distance matrix to determine which pairs can mate.
    - Randomly assigns partners for each female among available males (vectorized).
    - Calls calculate_chrom to generate offspring genotypes.
    - Stores offspring as PENDING (born on day 100 via birth_pending_offspring).
    - Marks mated males for semelparous death logic.
    - Marks mated females to prevent multiple matings per season.

    Args:
        state: SimulationState object (from state.py)
        distance_sq: Pre-calculated [N, N] squared distance matrix
        day_in_year: Current day in annual cycle
        breeding_days: Number of days for breeding season (default: BREEDING_DAYS)
        num_progeny: Number of offspring per pair (default: NUM_OF_PROGENY)
        device: Torch device (default: DEVICE)
    """
    n = state.pop_size
    if n == 0:
        return

    # ==================== RESET MATED STATUS AT START OF BREEDING SEASON ====================
    if day_in_year == 0:
        state.mated_female[:n] = False
        state.mated_male[:n] = False
    # ==================== END RESET ====================

    if day_in_year >= breeding_days:
        return  # Not breeding season

    # ==================== 1. IDENTIFY ELIGIBLE ADULTS ====================
    # Only unmated females can reproduce (1 mating per female per season)
    female_mask = (state.sex[:n] == False) & \
                  (state.status[:n] == STATUS_ADULT) & \
                  (state.mated_female[:n] == False)
    male_mask = (state.sex[:n] == True) & (state.status[:n] == STATUS_ADULT)
    female_indices = th.nonzero(female_mask, as_tuple=True)[0]
    male_indices = th.nonzero(male_mask, as_tuple=True)[0]

    if female_indices.numel() == 0 or male_indices.numel() == 0:
        return  # No eligible pairs
    # ==================== END IDENTIFY ELIGIBLE ADULTS ====================

    # ==================== 2. FIND MATING PAIRS ====================
    # Calculate interaction matrix from distance_sq for eligible pairs only
    # [num_females, num_males]
    dist_sq_subset = distance_sq[female_indices][:, male_indices]
    interaction_matrix = dist_sq_subset < MAX_RADIUS_SQ
    
    partner_mask = interaction_matrix  # [num_females, num_males]
    partner_mask_f = partner_mask.float()
    has_partner = partner_mask_f.sum(dim=1) > 0  # [num_females]

    # Only keep females with at least one partner
    valid_female_indices = female_indices[has_partner]
    partner_mask_f = partner_mask_f[has_partner]
    num_valid = valid_female_indices.shape[0]

    if num_valid == 0:
        return  # No matings

    # Normalize partner mask to get probabilities
    partner_probs = partner_mask_f / partner_mask_f.sum(dim=1, keepdim=True)
    # Use multinomial to select one partner per female
    selected_idx = th.multinomial(partner_probs, 1).squeeze(1)  # [num_valid]
    selected_males = male_indices[selected_idx]  # [num_valid]
    # ==================== END FIND MATING PAIRS ====================

    # ==================== 3. MARK MATED INDIVIDUALS ====================
    # Mark males for semelparous death logic on day 10
    state.mated_male[selected_males] = True
    # Mark females to prevent multiple matings this season
    state.mated_female[valid_female_indices] = True
    # ==================== END MARK MATED INDIVIDUALS ====================

    # ==================== 4. GENERATE OFFSPRING GENOTYPES ====================
    sex_chroms, repro_alleles = calculate_chrom(
        state,
        valid_female_indices,
        selected_males,
        num_progeny=num_progeny,
        device=device
    )
    # ==================== END GENERATE OFFSPRING GENOTYPES ====================

    # ==================== 5. PREPARE OFFSPRING DATA ====================
    # Offspring positions (same as mother)
    mothers_x = state.x[valid_female_indices]
    mothers_y = state.y[valid_female_indices]
    
    # Flatten for all offspring: [num_valid * num_progeny]
    offspring_x = mothers_x.unsqueeze(1).expand(-1, num_progeny).reshape(-1)
    offspring_y = mothers_y.unsqueeze(1).expand(-1, num_progeny).reshape(-1)
    
    # Flatten sex chromosomes and alleles: [num_valid * num_progeny, 2]
    offspring_chrom_sex = sex_chroms.reshape(-1, 2)
    offspring_chrom_a = repro_alleles.reshape(-1, 2)
    
    # Determine offspring sex from sex chromosomes
    # Sex is determined by second chromosome: 0=X (female), 1=Y (male)
    offspring_sex = offspring_chrom_sex[:, 1] == 1  # True = male, False = female
    
    # Get mother IDs for each offspring
    mother_ids = state.animal_ids[valid_female_indices]
    mother_ids_expanded = mother_ids.unsqueeze(1).expand(-1, num_progeny).reshape(-1)
    # ==================== END PREPARE OFFSPRING DATA ====================

    # ==================== 6. STORE AS PENDING OFFSPRING ====================
    # Offspring will be born on annual 100th day (TIME_OF_DISPOSAL)
    # If mother dies before annual 100th day, her pending offspring are removed
    add_pending_offspring(
        state,
        x=offspring_x,
        y=offspring_y,
        sex=offspring_sex,
        chrom_a=offspring_chrom_a,
        chrom_sex=offspring_chrom_sex,
        mother_id=mother_ids_expanded
    )
    # ==================== END STORE AS PENDING OFFSPRING ====================