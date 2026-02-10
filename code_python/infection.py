from constants import *
from state import SimulationState
import torch as th
from torch import Tensor
import math

def calculate_transmission_probability(
    state: SimulationState,
    transmitter_indices: Tensor,
    susceptible_indices: Tensor,
    dist_sq_matrix: Tensor,
    day_in_year: int,
    infectivity1: float = INFECTIVITY1,
    infectivity2: float = INFECTIVITY2,
    stage1_multiplier: float = STAGE1_TRANSMISSION_MULTIPLIER,
    stage2_multiplier: float = STAGE2_TRANSMISSION_MULTIPLIER,
    stage3_multiplier: float = STAGE3_TRANSMISSION_MULTIPLIER,
    breeding_days: int = BREEDING_DAYS,
    device: th.device = DEVICE
) -> Tensor:
    """
    Calculates the transmission probability matrix [num_susceptibles, num_transmitters]
    using the formula: StageMultiplier × DistanceFactor × (SexualComponent + NonsexualComponent)
    """
    num_susceptibles = susceptible_indices.shape[0]
    num_transmitters = transmitter_indices.shape[0]
    
    if num_susceptibles == 0 or num_transmitters == 0:
        return th.empty((0, 0), device=device)
    
    # ==================== 1. DISTANCE FACTOR ====================
    distance_factor = th.clamp((MAX_RADIUS_SQ - dist_sq_matrix) / MAX_RADIUS_SQ, 0, 1)
    
    # ==================== 2. STAGE MULTIPLIERS ====================
    transmitter_stages = state.infection_stage[transmitter_indices]
    
    stage_multipliers = th.zeros(num_transmitters, device=device)
    stage1_mask = transmitter_stages == INFECTION_STAGE_LATENT
    stage2_mask = transmitter_stages == INFECTION_STAGE_INFECTIOUS
    stage3_mask = transmitter_stages == INFECTION_STAGE_TERMINAL
    
    stage_multipliers[stage1_mask] = stage1_multiplier
    stage_multipliers[stage2_mask] = stage2_multiplier
    stage_multipliers[stage3_mask] = stage3_multiplier
    
    stage_multiplier_matrix = stage_multipliers.unsqueeze(0).expand(num_susceptibles, -1)
    
    # ==================== 3. DETERMINE WHO CAN TRANSMIT SEXUALLY/CONTACT ====================
    # Define masks for possible transmission routes for EACH transmitter
    transmitter_sexual_allowed_mask = th.zeros(num_transmitters, dtype=th.bool, device=device)
    transmitter_contact_allowed_mask = th.zeros(num_transmitters, dtype=th.bool, device=device)
    
    # Стадия 1
    if STAGE1_CAN_TRANSMIT_SEXUAL:
        transmitter_sexual_allowed_mask[stage1_mask] = True
    if STAGE1_CAN_TRANSMIT_CONTACT:
        transmitter_contact_allowed_mask[stage1_mask] = True
    
    # Стадия 2
    if STAGE2_CAN_TRANSMIT_SEXUAL:
        transmitter_sexual_allowed_mask[stage2_mask] = True
    if STAGE2_CAN_TRANSMIT_CONTACT:
        transmitter_contact_allowed_mask[stage2_mask] = True
    
    # Стадия 3
    if STAGE3_CAN_TRANSMIT_SEXUAL:
        transmitter_sexual_allowed_mask[stage3_mask] = True
    if STAGE3_CAN_TRANSMIT_CONTACT:
        transmitter_contact_allowed_mask[stage3_mask] = True
    
    # ==================== 4. SEXUAL COMPONENT ====================
    sexual_component = th.zeros((num_susceptibles, num_transmitters), device=device)
    
    if day_in_year < breeding_days:
        # Определяем, кто взрослый
        transmitter_adult_mask = state.status[transmitter_indices] == STATUS_ADULT
        susceptible_adult_mask = state.status[susceptible_indices] == STATUS_ADULT
        
        # Создаем маску для пар, где ОБА взрослые И передатчик может передавать половым путем
        adult_pair_mask = transmitter_adult_mask.unsqueeze(0) & \
                         susceptible_adult_mask.unsqueeze(1) & \
                         transmitter_sexual_allowed_mask.unsqueeze(0)
        
        sexual_component[adult_pair_mask] = infectivity1
    
    # ==================== 5. NONSEXUAL COMPONENT ====================
    # Контактная передача всегда доступна (но с разными множителями по стадиям)
    nonsexual_component = th.zeros((num_susceptibles, num_transmitters), device=device)
    
    # Создаем матричную маску для пар, где передатчик может передавать контактным путем
    contact_allowed_matrix = transmitter_contact_allowed_mask.unsqueeze(0).expand(num_susceptibles, -1)
    
    # Применяем базовую вероятность контактной передачи
    nonsexual_component[contact_allowed_matrix] = infectivity2
    
    # ==================== 6. COMBINE ALL COMPONENTS ====================
    total_prob = stage_multiplier_matrix * distance_factor * (sexual_component + nonsexual_component)
    total_prob = th.clamp(total_prob, 0, 1)
    
    return total_prob

def infection_spread(
    state: SimulationState,
    distance_sq: Tensor,
    day_in_year: int,
    infectivity1: float = INFECTIVITY1,
    infectivity2: float = INFECTIVITY2,
    stage1_multiplier: float = STAGE1_TRANSMISSION_MULTIPLIER,
    stage2_multiplier: float = STAGE2_TRANSMISSION_MULTIPLIER,
    stage3_multiplier: float = STAGE3_TRANSMISSION_MULTIPLIER,
    breeding_days: int = BREEDING_DAYS,
    device: th.device = DEVICE
) -> None:
    """
    Распространение инфекции по новой формуле:
    
    Вероятность = StageMultiplier × DistanceFactor × (SexualComponent + NonsexualComponent)
    
    Где:
    - StageMultiplier: 0.1 (стадия 1), 5.0 (стадия 2), 10.0 (стадия 3)
    - DistanceFactor: (MAX_RADIUS_SQ - dist^2) / MAX_RADIUS_SQ, от 0 до 1
    - SexualComponent: infectivity1 (только в сезон размножения, только взрослые)
    - NonsexualComponent: infectivity2 (всегда, но зависит от стадии)
    
    Args:
        state: SimulationState object to modify
        distance_sq: Pre-calculated [N, N] squared distance matrix
        day_in_year: Current day in annual cycle
        infectivity1: Базовая вероятность половой передачи
        infectivity2: Базовая вероятность неполовой передачи
        stage1_multiplier: Множитель для латентной стадии
        stage2_multiplier: Множитель для инфекционной стадии
        stage3_multiplier: Множитель для терминальной стадии
        breeding_days: Number of days for breeding season
        device: Torch device
    """
    n = state.pop_size
    if n == 0:
        return

    # ==================== 1. ИНИЦИАЛИЗАЦИЯ ДЛИТЕЛЬНОСТЕЙ ДЛЯ НОВЫХ ЗАРАЖЕННЫХ ====================
    newly_infected_mask = (state.infection_stage[:n] == INFECTION_STAGE_LATENT) & \
                          (state.stage1_duration[:n] == 0)
    if newly_infected_mask.any():
        newly_infected_indices = th.nonzero(newly_infected_mask, as_tuple=True)[0]
        initialize_disease_durations(state, newly_infected_indices, device)

    # ==================== 2. ПРОГРЕССИЯ БОЛЕЗНИ ====================
    # Увеличиваем возраст болезни у всех инфицированных
    infected_mask = state.infection_stage[:n] > 0
    state.age_of_disease[:n] += infected_mask.to(state.age_of_disease.dtype)
    
    # Переход между стадиями
    # Стадия 1 → Стадия 2
    stage1_mask = state.infection_stage[:n] == INFECTION_STAGE_LATENT
    transition_to_stage2 = stage1_mask & (state.age_of_disease[:n] >= state.stage1_duration[:n])
    
    if transition_to_stage2.any():
        state.infection_stage[:n] = th.where(
            transition_to_stage2,
            INFECTION_STAGE_INFECTIOUS,
            state.infection_stage[:n]
        )
    
    # Стадия 2 → Стадия 3
    stage2_mask = state.infection_stage[:n] == INFECTION_STAGE_INFECTIOUS
    transition_to_stage3 = stage2_mask & \
                          (state.age_of_disease[:n] >= (state.stage1_duration[:n] + state.stage2_duration[:n]))
    
    if transition_to_stage3.any():
        state.infection_stage[:n] = th.where(
            transition_to_stage3,
            INFECTION_STAGE_TERMINAL,
            state.infection_stage[:n]
        )

    # ==================== 3. ОПРЕДЕЛЕНИЕ КТО МОЖЕТ ПЕРЕДАВАТЬ/ПОЛУЧАТЬ ====================
    # Только резиденты могут передавать/получать инфекцию
    resident_mask = (state.status[:n] == STATUS_ADULT) | (state.status[:n] == STATUS_JUVENILE_TERR)
    
    # КТО МОЖЕТ ПЕРЕДАВАТЬ: резиденты на стадиях 1, 2 или 3
    can_transmit_mask = resident_mask & (state.infection_stage[:n] > 0)
    
    # КТО МОЖЕТ ЗАРАЖАТЬСЯ: здоровые резиденты
    can_be_infected_mask = resident_mask & (state.infection_stage[:n] == INFECTION_STAGE_HEALTHY)
    
    transmitters = th.nonzero(can_transmit_mask, as_tuple=True)[0]
    susceptibles = th.nonzero(can_be_infected_mask, as_tuple=True)[0]
    
    if transmitters.numel() == 0 or susceptibles.numel() == 0:
        return
    
    # ==================== 4. ВЫЧИСЛЕНИЕ МАТРИЦЫ ВЕРОЯТНОСТЕЙ ====================
    # Извлекаем подматрицу расстояний [susceptibles, transmitters]
    dist_sq_subset = distance_sq[susceptibles][:, transmitters]
    
    # Рассчитываем матрицу вероятностей
    prob_matrix = calculate_transmission_probability(
        state=state,
        transmitter_indices=transmitters,
        susceptible_indices=susceptibles,
        dist_sq_matrix=dist_sq_subset,
        day_in_year=day_in_year,
        infectivity1=infectivity1,
        infectivity2=infectivity2,
        stage1_multiplier=stage1_multiplier,
        stage2_multiplier=stage2_multiplier,
        stage3_multiplier=stage3_multiplier,
        breeding_days=breeding_days,
        device=device
    )
    
    # ==================== 5. ПРИМЕНЕНИЕ ВЕРОЯТНОСТЕЙ ДЛЯ ЗАРАЖЕНИЯ ====================
    # Для каждого восприимчивого берем максимальную вероятность от всех передатчиков
    if prob_matrix.numel() > 0:
        max_probs, _ = prob_matrix.max(dim=1)  # [num_susceptibles]
        
        # Случайное испытание для каждого восприимчивого
        random_draws = th.rand(max_probs.shape[0], device=device)
        newly_infected_mask = (random_draws < max_probs) & (max_probs > 0)
        
        if newly_infected_mask.any():
            new_indices = susceptibles[newly_infected_mask]
            state.infection_stage[new_indices] = INFECTION_STAGE_LATENT
            state.age_of_disease[new_indices] = 0
            initialize_disease_durations(state, new_indices, device)
            state.num_infection += new_indices.numel()


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (остаются без изменений) ====================

def initialize_disease_durations(
    state: SimulationState,
    infected_indices: Tensor,
    device: th.device = DEVICE
) -> None:
    """
    Инициализирует случайные длительности стадий болезни для новых заражённых.
    """
    num_new = infected_indices.shape[0]
    if num_new == 0:
        return
    
    # Стадия 1: Латентная (0.5-1.0 года)
    stage1_durations = th.randint(
        low=STAGE1_DURATION_MIN,
        high=STAGE1_DURATION_MAX + 1,
        size=(num_new,),
        device=device,
        dtype=th.float32
    )
    
    # Стадия 2: Инфекционная (0.25-1.0 года)
    stage2_durations = th.randint(
        low=STAGE2_DURATION_MIN,
        high=STAGE2_DURATION_MAX + 1,
        size=(num_new,),
        device=device,
        dtype=th.float32
    )
    
    # Стадия 3: Терминальная (0.5-1.0 года)
    stage3_durations = th.randint(
        low=STAGE3_DURATION_MIN,
        high=STAGE3_DURATION_MAX + 1,
        size=(num_new,),
        device=device,
        dtype=th.float32
    )
    
    # Устанавливаем длительности
    state.stage1_duration[infected_indices] = stage1_durations
    state.stage2_duration[infected_indices] = stage2_durations
    state.stage3_duration[infected_indices] = stage3_durations


def seed_pathogen(
    state: SimulationState,
    num_infected: int = 10,
    device: th.device = DEVICE
) -> None:
    """
    Infect initial residents with RANDOM disease stages for realism.
    First infected individuals start at different points in disease progression.
    
    Modifies state in-place. Does not return.

    Args:
        state: SimulationState object to modify
        num_infected: Number of residents to infect (default: 10)
        device: Torch device for tensors (default: DEVICE)
    """
    n = state.pop_size
    if n == 0:
        return

    # Identify residents (adults and territorial juveniles)
    resident_mask = (state.status[:n] == STATUS_ADULT) | (state.status[:n] == STATUS_JUVENILE_TERR)
    resident_indices = th.nonzero(resident_mask, as_tuple=True)[0]

    if resident_indices.numel() == 0:
        return

    # Limit number to available residents
    num_to_infect = min(num_infected, resident_indices.numel())

    # Randomly select unique residents to infect
    selected_indices = resident_indices[th.randperm(resident_indices.numel(), device=device)[:num_to_infect]]

    # ==================== РАСПРЕДЕЛЕНИЕ ПО СТАДИЯМ ====================
    # Случайно распределяем стадии: ~40% латентные, ~40% инфекционные, ~20% терминальные
    num_latent = int(num_to_infect * 0.4)  # ~4 особи: стадия 1
    num_infectious = int(num_to_infect * 0.4)  # ~4 особи: стадия 2
    num_terminal = num_to_infect - num_latent - num_infectious  # ~2 особи: стадия 3
    
    # Перемешиваем индексы для случайного распределения
    permuted_indices = selected_indices[th.randperm(num_to_infect, device=device)]
    
    # Присваиваем стадии
    if num_latent > 0:
        latent_indices = permuted_indices[:num_latent]
        state.infection_stage[latent_indices] = INFECTION_STAGE_LATENT
        state.age_of_disease[latent_indices] = 0  # Начинают с 0 дней
        initialize_disease_durations(state, latent_indices, device)
    
    if num_infectious > 0:
        infectious_start = num_latent
        infectious_end = num_latent + num_infectious
        infectious_indices = permuted_indices[infectious_start:infectious_end]
        state.infection_stage[infectious_indices] = INFECTION_STAGE_INFECTIOUS
        
        # Для инфекционных ставим случайный возраст болезни (уже в середине стадии 1 или начале стадии 2)
        for idx in infectious_indices:
            # Случайная длительность стадии 1 (180-360 дней)
            stage1_duration = th.randint(
                STAGE1_DURATION_MIN,
                STAGE1_DURATION_MAX + 1,
                (1,), device=device, dtype=th.float32
            )
            state.stage1_duration[idx] = stage1_duration
            
            # Возраст болезни: где-то в стадии 2 (прошли всю стадию 1 + часть стадии 2)
            age_in_stage1 = stage1_duration * th.rand((1,), device=device) * 0.2 + stage1_duration * 0.8
            state.age_of_disease[idx] = age_in_stage1
    
    if num_terminal > 0:
        terminal_start = num_latent + num_infectious
        terminal_indices = permuted_indices[terminal_start:]
        state.infection_stage[terminal_indices] = INFECTION_STAGE_TERMINAL
        
        # Для терминальных ставим случайный возраст (уже в стадии 3)
        for idx in terminal_indices:
            # Случайные длительности всех стадий
            stage1_duration = th.randint(
                STAGE1_DURATION_MIN,
                STAGE1_DURATION_MAX + 1,
                (1,), device=device, dtype=th.float32
            )
            stage2_duration = th.randint(
                STAGE2_DURATION_MIN,
                STAGE2_DURATION_MAX + 1,
                (1,), device=device, dtype=th.float32
            )
            stage3_duration = th.randint(
                STAGE3_DURATION_MIN,
                STAGE3_DURATION_MAX + 1,
                (1,), device=device, dtype=th.float32
            )
            
            state.stage1_duration[idx] = stage1_duration
            state.stage2_duration[idx] = stage2_duration
            state.stage3_duration[idx] = stage3_duration
            
            # Возраст болезни: прошли стадии 1 и 2 + часть стадии 3
            total_stage12 = stage1_duration + stage2_duration
            age_in_stage3 = stage3_duration * th.rand((1,), device=device) * 0.4 + stage3_duration * 0.3
            state.age_of_disease[idx] = total_stage12 + age_in_stage3

    # Update infection count
    state.num_infection += num_to_infect
    
    print(f"🎲 Начальное заражение: {num_latent} латентных, {num_infectious} инфекционных, {num_terminal} терминальных")