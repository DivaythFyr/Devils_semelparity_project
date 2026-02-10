import math
from typing import Dict, List, Tuple, Optional
import torch as th

# ---- DEVICE SELECTION ----
DEVICE: th.device = th.device("cuda" if th.cuda.is_available() else "cpu")  
# Target device for all tensors. Used in: create_initial_state(), seed_pathogen(), 
# infection_spread(), replication(), calculate_chrom(), move(), all tensor operations.

TIME_OF_PATHOGEN: int = 0         
# Day when pathogen is first introduced. Used in: main() for conditional seeding.
# NOTE: Currently main.py seeds when no one is infected; consider using this constant.

DAYS_PER_YEAR: int = 120       
# Days in annual cycle for seasonal effects (breeding, disposal). 
# Used in: main() (day_in_year calculation), infection_spread(), replication(), 
# birth_pending_offspring(), disperse_offspring(), death functions.

MAX_AGE: int = 720               
# Maximum age before forced removal by aging. Used in: death_by_age().

# ---- POPULATION LIMITS ----
INIT_POP_SIZE: int = 2000         
# Default initial population size. Used in: initialize_population().

MAX_POP_SIZE: int = 3000       
# Safety cap for tensor preallocation. Used in: create_initial_state().

INITIAL_FITNESS: float = 100.0  
# Default starting fitness for new individuals. Used in: initialize_population().

MIN_FITNESS_JUVENILE_FOR_TERR: float = 95.0  
# Minimum fitness required for a juvenile to acquire territory. 
# Used in: transition_juvenile_no_terr_to_terr().

# ---- SPACE & MOVEMENT ----
MAP_X_SIZE: float = 35000.0       
# Map width (X coordinate boundary). 
# Used in: initialize_population(), move() (wall repulsion, boundary correction), 
# draw_snapshot() (plot limits).

MAP_Y_SIZE: float = 4000.0        
# Map height (Y coordinate boundary). 
# Used in: initialize_population(), move() (wall repulsion, boundary correction), 
# draw_snapshot() (plot limits).

MAX_SPEED: float = 20.0           
# Maximum absolute speed per axis. Used in: move() (speed clamping).

MAX_RADIUS: float = 1000.0        
# Maximum individual influence radius for repulsion and infection. 
# Used in: move() (pairwise repulsion), infection_spread() (distance filtering).

MAX_RADIUS_SQ: float = MAX_RADIUS ** 2  
# Precomputed square of MAX_RADIUS for efficient distance checks. 
# Used in: move() (repulsion mask), infection_spread() (distance mask).

RANGE: float = 150.0              
# Interaction / territory radius for mating, area calculation, and territory clamping. 
# Used in: move() (territory boundary enforcement), calculate_area_and_fitness(), 
# calculate_fitness_for_candidates(), replication() (interaction matrix).

RANGE_SQ: float = RANGE ** 2      
# Precomputed square of RANGE for efficient distance filters. 
# Used in: calculate_area_and_fitness(), calculate_fitness_for_candidates(), 
# main() (interaction matrix calculation).

MAX_AREA: float = math.pi * RANGE_SQ  
# Maximum circular interaction area for fitness normalization. 
# Used in: calculate_area_and_fitness(), calculate_fitness_for_candidates().

REPULSION: float = 550000.0       
# Repulsion strength to prevent individual overlap. 
# Used in: move() (pairwise repulsion force, wall repulsion force).
 
ATTRACTION_STRENGTH: float = 0.05 ### Attraction strength towards territory center.  
# Strength of attraction force pulling residents towards their territory center.`

DISPERSAL: float = 2.0            
# Multiplier for speed during juvenile dispersal phase. 
# Used in: move() (speed clamping), disperse_offspring().

# ---- DEMOGRAPHY ----
MORTALITY: float = 0.002967       
# Baseline daily mortality probability. Used in: death_by_base_mortality().

TIME_OF_DISPOSAL: int = 100       
# Day in year when offspring disposal/birth phase starts. 
# Used in: birth_pending_offspring(), disperse_offspring().

TIME_OF_DISPERSAL: int = 60       
# Length of dispersal window in days. 
# Used in: disperse_offspring() (dispersal deadline calculation).

DISPERSAL_DEADLINE: int = TIME_OF_DISPOSAL + TIME_OF_DISPERSAL  
# Last day for juveniles without territory to secure one before death. 
# Used in: death_by_no_territory(), disperse_offspring().


INCUBATION: int = 225  # Average 225 days (180-270) instead of 60             
# Days in incubation phase before infection becomes active (phase 1 duration). 
# Used in: main() (disease progression, phase 1 to phase 2 transition).

LATENCY: int = 225     # Average 225 days (90-360) instead of 120
              
# Days in latent phase before symptoms appear (total time before phase 2). 
# Used in: main() (disease progression, phase 1 to phase 2 transition).

AGE_CHILD_AFTER_DISPOSAL: int = 85  
# Age assigned to children after disposal phase. 
# Used in: birth_pending_offspring() (initial age of newborns).

STOPPAGE_SEMELPARITY_PROPORTION_CONDITION: float = 0.999  
# Proportion of semelparous adults triggering simulation stoppage. 
# Used in: main() (stoppage logic).

STOPPAGE_ITEROPAROUS_PROPORTION_CONDITION: float = 0.999  
# Proportion of iteroparous adults triggering simulation stoppage. 
# Used in: main() (stoppage logic).

# ---- INFECTION ----
INFECTIVITY1: float = 0.5
# Sexual transmission probability parameter (during breeding season, adults only, phase 1). 
# Used in: infection_spread().

INFECTIVITY2: float = 0.5
# Nonsexual (contact) transmission probability parameter (all residents, all phases). 
# Used in: infection_spread().

# ---- INFECTION STAGES ----

INFECTION_STAGE_HEALTHY: int = 0        # Healthy
INFECTION_STAGE_LATENT: int = 1         # Latent (0.5-1.0 years, no transmission)
INFECTION_STAGE_INFECTIOUS: int = 2     # Infectious (0.25-1.0 years, transmission)
INFECTION_STAGE_TERMINAL: int = 3       # Terminal (0.5-1.0 years, 100% mortality)

# ---- STAGE DURATIONS (in days) ----
STAGE1_DURATION_MIN: int = 60    # 0.5 years = 60 days
STAGE1_DURATION_MAX: int = 120    # 1.0 years = 120 days
STAGE2_DURATION_MIN: int = 30     # 0.25 years = 30 days  
STAGE2_DURATION_MAX: int = 120    # 1.0 years = 120 days
STAGE3_DURATION_MIN: int = 60    # 0.5 years = 60 days
STAGE3_DURATION_MAX: int = 120    # 1.0 years = 120 days

# ---- STAGE-SPECIFIC TRANSMISSION MULTIPLIERS ----
STAGE1_TRANSMISSION_MULTIPLIER: float = 0.1    # Latent stage (very low)
STAGE2_TRANSMISSION_MULTIPLIER: float = 5.0    # Infectious stage  
STAGE3_TRANSMISSION_MULTIPLIER: float = 10.0   # Terminal stage

# ---- TRANSMISSION CONSTRAINTS ----
STAGE1_CAN_TRANSMIT_SEXUAL: bool = False       # Latent does NOT transmit sexually
STAGE1_CAN_TRANSMIT_CONTACT: bool = True       # Latent transmits by contact (with very low probability)
STAGE2_CAN_TRANSMIT_SEXUAL: bool = True        # Infectious transmits by all means
STAGE2_CAN_TRANSMIT_CONTACT: bool = True
STAGE3_CAN_TRANSMIT_SEXUAL: bool = True        # Terminal transmits by all means
STAGE3_CAN_TRANSMIT_CONTACT: bool = True

# ---- MORTALITY ----
DISEASE_MORTALITY_FACTOR: float = 0.1     # Maximum 10% in sigmoid formula

# ---- DISEASE MORTALITY ----
DISEASE_MORTALITY_FACTOR_STAGE1: float = 0.02    # 2% maximum for latent
DISEASE_MORTALITY_FACTOR_STAGE2: float = 0.10    # 10% maximum for infectious
DISEASE_MORTALITY_FACTOR_STAGE3: float = 0.20    # 20% maximum for terminal + 100% at the end


# ---- REPRODUCTION ----
NUM_OF_PROGENY: int = 3           
# Number of offspring per successful mating event. 
# Used in: replication(), calculate_chrom(), create_initial_state() (pending offspring allocation).

REPLICATION_MATRIX_INIT: Optional[th.Tensor] = None    
# Placeholder for mating interaction matrix (initialized at runtime). 
# Used in: replication() (partner selection).

BREEDING_DAYS: int = 11 
# Number of days in the breeding season (day 0 to BREEDING_DAYS-1). 
# Used in: replication() (breeding eligibility), infection_spread() (sexual transmission).

# ---- STATUS CODES ----
STATUS_CHILD: int = 0             
# Status code for children (age < AGE_CHILD_TO_JUVENILE). 
# Used in: initialize_population(), transition_child_to_juvenile_no_terr(), 
# collect_statistics(), draw_snapshot(), birth_pending_offspring().

STATUS_JUVENILE_NO_TERR: int = 1  
# Status code for juveniles without territory. 
# Used in: transition_child_to_juvenile_no_terr(), transition_juvenile_no_terr_to_terr(), 
# death_by_no_territory(), collect_statistics(), draw_snapshot().

STATUS_JUVENILE_TERR: int = 2     
# Status code for juveniles with territory (residents). 
# Used in: transition_juvenile_no_terr_to_terr(), transition_juvenile_terr_to_adult(), 
# move() (resident mask), infection_spread() (resident mask), collect_statistics(), draw_snapshot().

STATUS_ADULT: int = 3             
# Status code for adults (residents, breeding eligible). 
# Used in: transition_juvenile_terr_to_adult(), move() (resident mask), 
# infection_spread() (resident mask, sexual transmission), replication() (adult eligibility), 
# collect_statistics(), draw_snapshot(), main() (stoppage logic).


INITIAL_SEX_RATIO: float = 0.5 # How many female (False) / male (True) there will be from [0,1] random. 
# Used in: initialize_population()

# ---- Transition Status Ages ----
AGE_CHILD_TO_JUVENILE: int = 160  
# Age threshold for transition from STATUS_CHILD to STATUS_JUVENILE_NO_TERR. 
# Used in: transition_child_to_juvenile_no_terr().

AGE_JUVENILE_TO_ADULT: int = 220  
# Age threshold for transition from STATUS_JUVENILE_TERR to STATUS_ADULT. 
# Used in: transition_juvenile_terr_to_adult().
 
SEMELPAROUS_DEATH_DAY: int = 10 # Days after breeding when semelparous adults die.  
# Used in: process_all_deaths().


## Proportions for stoppage logic
SEMELPAROUS_WIN_PROPORTION: float = 0.99
ITEROPAROUS_WIN_PROPORTION: float = 0.99

