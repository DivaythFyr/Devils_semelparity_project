import math
import random
import sys
import os
import torch as th
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import csv
import imageio
import time
from datetime import datetime
import os
import shutil
import glob


# Global variables for configuration
if th.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = th.device(dev)
print(f"Device: {device}")

# Monte Carlo configuration
MONTE_CARLO_RUNS = 2  # Number of simulations
PARAM_RANGES = {
    'I1': (0, 7),     # Infectivity1 from 0 to 7
    'I2': (0, 0.05)   # Infectivity2 from 0 to 0.05
}

# Other parameters 
time_draw = [10, 6000, 12000, 18000, 24000, 30000, 36000, 42000]
Timepoints = 42001
CurrentTime = 0
TimeOfPathogen = 0
PopSize = 500 * 2
MaxPopSize = 50000 * 2
MapXSize = 35000
MapYSize = 4000
MaxSpeed = 20.0
MaxRadius = 1000.0
MaxRadiusSq = MaxRadius ** 2.0
Range = 150
RangeSq = Range ** 2.0
MaxArea = math.pi * RangeSq
Repulsion = 550000
dispersal = 2
Mortality = 0.002967
timeOfDisposal = 100
timeOfDispersal = 60
dispersalDeadline = timeOfDisposal + timeOfDispersal
timeOfMaturity = 220
incubation = 60
Infectivity1, Infectivity2 = 0, 0.05
latency = 120
numofinfection = 0
numofprogeny = 3
dead = 0
time_near_edge = None
stats_file = None
csv_writer = None
run_id = 0
# In the global variables section add:
f_a = None
m_a = None
f_sex = None
m_sex = None
# In the global variables section add:
reproducing_female_indices = None
reproducing_female_coords = None
reproducing_female_genos = None
# Add to the beginning of the file:
female_genotypes_memory = None
female_coords_memory = None
chrom_cal_female_indices = None
reproduction_pairs_log = None
reproduction_pairs = []  

animal_ids = None
next_animal_id = 0  # Counter for unique IDs
id_to_index = {}    # Dictionary for fast index lookup by ID (CPU only!)


# Add to global variables
female_ids_in_matrix = None  # Female IDs in reproduction matrix
male_ids_in_matrix = None    # Male IDs in reproduction matrix

# Create global constants for paths
RESULTS_DIR = "monte_carlo_results"
STATS_DIR = "monte_carlo_stats"
GIFS_DIR = "simulation_gifs"
SNAPSHOTS_DIR = "simulation_snapshots"


# In the global variables section add after other variables:
territory_center_x = None  # Territory center by X
territory_center_y = None  # Territory center by Y
territory_radius = Range   # Territory radius (use existing Range)

# ==================== CONSTANTS FOR NEW STATUS SYSTEM ====================
STATUS_CHILD = 0           # 0-159 days
STATUS_JUVENILE_NO_TERR = 1 # 160-219 days, without territory (dies!)
STATUS_JUVENILE_TERR = 2    # 160-219 days, with territory (resident)
STATUS_ADULT = 3           # 220+ days (resident)


def generate_random_parameters():
    """Генерация случайных параметров в заданных диапазонах"""
    return {
        'I1': random.uniform(*PARAM_RANGES['I1']),
        'I2': random.uniform(*PARAM_RANGES['I2'])
    }


def reset_simulation_state():
    """Полный сброс всех переменных состояния между симуляциями"""
    global status
    global Age, AgeOfDisease, Sex, chrom_sex, chrom_a
    global Fitness, X, Y, speedX, speedY, new_X, new_Y
    global PopSize, InfectionStatus, juvenile, mated_male
    global CurrentTime, numofinfection, dead, time_near_edge
    global Infectivity1, Infectivity2, replication_matrix
    global new_sex, new_a, stats_file, csv_writer
    global f_a, m_a, f_sex, m_sex 
    global animal_ids, next_animal_id, id_to_index
    global territory_center_x, territory_center_y  # ← ДОБАВЛЕНО

    # Сбрасываем скалярные переменные
    PopSize = 0
    CurrentTime = 0
    TimeOfPathogen = 0
    numofinfection = 0
    dead = 0
    time_near_edge = None
    next_animal_id = 0
    
    # Сбрасываем тензоры (освобождаем память)
    status = None
    Age = None
    AgeOfDisease = None
    Sex = None
    chrom_sex = None
    chrom_a = None
    Fitness = None
    X = None
    Y = None
    speedX = None
    speedY = None
    new_X = None
    new_Y = None
    InfectionStatus = None
    juvenile = None
    mated_male = None
    replication_matrix = None
    new_sex = None
    new_a = None
    f_a = None
    m_a = None
    f_sex = None
    m_sex = None
    animal_ids = None
    
    # НОВЫЕ ТЕНЗОРЫ ТЕРРИТОРИЙ ← ДОБАВЛЕНО
    territory_center_x = None
    territory_center_y = None
    
    # Очищаем словарь id_to_index
    id_to_index.clear()
    
    # Закрываем файл статистики если открыт
    if stats_file:
        try:
            stats_file.close()
        except:
            pass
    stats_file = None
    csv_writer = None
    
    # Очищаем память CUDA
    if th.cuda.is_available():
        th.cuda.empty_cache()
    
    print("✅ Состояние симуляции полностью сброшено")


def _update_id_to_index():
    """
    Обновляет словарь id_to_index после изменений в популяции.
    Вызывается после добавления или удаления особей.
    """
    global id_to_index
    
    # Очищаем старый словарь
    id_to_index.clear()
    
    # Заполняем новый словарь (только если есть особи)
    if PopSize > 0 and animal_ids is not None:
        ids_cpu = animal_ids.cpu().numpy()
        for i, animal_id in enumerate(ids_cpu):
            id_to_index[animal_id] = i
    
    print(f"📊 id_to_index обновлен: {len(id_to_index)} записей")


def get_index_by_id(animal_id):
    """
    Быстрый поиск индекса особи по её ID.
    Возвращает индекс или None если особь не найдена.
    """
    return id_to_index.get(animal_id.item() if isinstance(animal_id, th.Tensor) else animal_id)

def Start(initial_pop_size=2000):
    """Инициализация новой симуляции с УСИЛЕННОЙ проверкой"""
    global Age, AgeOfDisease, Sex, chrom_sex, chrom_a
    global Fitness, X, Y, speedX, speedY, new_X, new_Y
    global PopSize, InfectionStatus, status, mated_male
    global animal_ids, next_animal_id
    global territory_center_x, territory_center_y
    
    # Устанавливаем начальный размер популяции
    PopSize = initial_pop_size
    
    print(f"🚀 Инициализация новой симуляции с {PopSize} особями...")
    
    # 1. ВОЗРАСТ - случайный от 220 до 721 дней (взрослые)
    Age = th.randint(220, 721, (PopSize,), device=device)
    
    # 2. ФИТНЕС - все начинают с максимальным фитнесом
    Fitness = th.zeros(PopSize, device=device) + 100
    
    # 3. КООРДИНАТЫ - равномерное распределение
    X = th.rand(PopSize, device=device) * MapXSize
    Y = th.rand(PopSize, device=device) * MapYSize
    
    # 3.5. ЦЕНТРЫ ТЕРРИТОРИИ
    territory_center_x = X.clone()
    territory_center_y = Y.clone()
    
    # 4. СКОРОСТЬ - случайное направление
    speedX = MaxSpeed - th.rand(PopSize, device=device) * MaxSpeed * 2
    speedY = MaxSpeed - th.rand(PopSize, device=device) * MaxSpeed * 2
    
    # 5. ПОЛ - 50/50
    Sex = th.rand(PopSize, device=device) > 0.5
    
    # 6. ГЕНОТИПЫ - КРИТИЧЕСКИ ВАЖНЫЙ РАЗДЕЛ
    chrom_sex = th.cat((th.zeros(PopSize, device=device), Sex)).reshape(2, PopSize).transpose(0, 1)
    chrom_a = th.zeros(PopSize * 2, device=device).reshape((PopSize, 2))
    
    # Разделяем по сторонам карты
    mid_x = MapXSize / 2
    mask_left = X <= mid_x
    mask_right = X > mid_x
    
    print(f"🔍 Начальное распределение по сторонам:")
    print(f"  Левая сторона (X ≤ {mid_x:.0f}): {mask_left.sum().item()} особей")
    print(f"  Правая сторона (X > {mid_x:.0f}): {mask_right.sum().item()} особей")
    
    # ==================== КРИТИЧЕСКОЕ: ПРОВЕРКА ПЕРЕД ПРИСВАИВАНИЕМ ====================
    print(f"  Проверка chrom_a перед инициализацией:")
    print(f"    Размер chrom_a: {chrom_a.shape}")
    print(f"    Первые 3 строки chrom_a: {chrom_a[:3]}")
    
    # НАЗНАЧАЕМ ГЕНОТИПЫ ЧЕТКО И ЯВНО
    # Левая сторона: ВСЕ итеропарные [0, 0]
    if mask_left.any():
        # Явно создаем тензор [0, 0] для левой стороны
        left_genotypes = th.zeros((mask_left.sum().item(), 2), device=device)
        print(f"  Создано left_genotypes: {left_genotypes.shape}, первые 3: {left_genotypes[:3]}")
        
        # ПРОВЕРКА: правильно ли присваиваем?
        indices_left = mask_left.nonzero(as_tuple=True)[0]
        print(f"  Индексы левой стороны (первые 5): {indices_left[:5].tolist()}")
        
        chrom_a[mask_left] = left_genotypes
    
    # Правая сторона: ВСЕ семельпарные [1, 1]
    if mask_right.any():
        # Явно создаем тензор [1, 1] для правой стороны
        right_genotypes = th.ones((mask_right.sum().item(), 2), device=device)
        print(f"  Создано right_genotypes: {right_genotypes.shape}, первые 3: {right_genotypes[:3]}")
        
        # ПРОВЕРКА: правильно ли присваиваем?
        indices_right = mask_right.nonzero(as_tuple=True)[0]
        print(f"  Индексы правой стороны (первые 5): {indices_right[:5].tolist()}")
        
        chrom_a[mask_right] = right_genotypes
    
    # ==================== ПРОВЕРКА ПОСЛЕ ИНИЦИАЛИЗАЦИИ ====================
    print(f"  🔍 Проверка после инициализации:")
    
    # Проверяем генотипы на левой стороне
    if mask_left.any():
        left_genotypes_check = chrom_a[mask_left]
        left_semel = (left_genotypes_check.sum(1) == 2).sum().item()
        left_itero = (left_genotypes_check.sum(1) == 0).sum().item()
        print(f"    Левая сторона: [1,1]={left_semel} (должно быть 0!), [0,0]={left_itero} (должно быть все!)")
        
        if left_semel > 0:
            print(f"    🚨 ОШИБКА: {left_semel} семельпарных на левой стороне!")
    
    # Проверяем генотипы на правой стороне
    if mask_right.any():
        right_genotypes_check = chrom_a[mask_right]
        right_semel = (right_genotypes_check.sum(1) == 2).sum().item()
        right_itero = (right_genotypes_check.sum(1) == 0).sum().item()
        print(f"    Правая сторона: [1,1]={right_semel} (должно быть все!), [0,0]={right_itero} (должно быть 0!)")
        
        if right_itero > 0:
            print(f"    🚨 ОШИБКА: {right_itero} итеропарных на правой стороне!")
            
            # Находим конкретные ошибки
            wrong_mask = right_genotypes_check.sum(1) == 0
            wrong_indices = mask_right.nonzero(as_tuple=True)[0][wrong_mask]
            print(f"    Индексы ошибочных особей на правой стороне: {wrong_indices[:10].tolist()}")
    
    # ==================== ПРОВЕРКА ПО СЕКСУ ====================
    print(f"  🔍 Распределение по полу:")
    females_left = (Sex[mask_left] == 0).sum().item()
    females_right = (Sex[mask_right] == 0).sum().item()
    males_left = (Sex[mask_left] == 1).sum().item()
    males_right = (Sex[mask_right] == 1).sum().item()
    
    print(f"    Самки: слева={females_left}, справа={females_right}")
    print(f"    Самцы: слева={males_left}, справа={males_right}")
    
    # 7. УНИКАЛЬНЫЕ ID
    animal_ids = th.arange(next_animal_id, next_animal_id + PopSize, 
                          dtype=th.int64, device=device)
    next_animal_id += PopSize
    
    # 8. СТАТУСЫ - все взрослые
    status = th.full((PopSize,), STATUS_ADULT, dtype=th.long, device=device)
    
    # 9. ИНФЕКЦИЯ - все здоровы
    InfectionStatus = th.zeros(PopSize, device=device)
    AgeOfDisease = th.zeros(PopSize, device=device)
    
    # 10. РАЗМНОЖЕНИЕ - флаг для самцов семельпарных
    mated_male = th.zeros(PopSize, dtype=th.bool, device=device)
    
    # 11. ВСПОМОГАТЕЛЬНЫЕ ПЕРЕМЕННЫЕ
    new_X = th.empty(0, device=device)
    new_Y = th.empty(0, device=device)
    new_sex = th.empty((0, 2), device=device)
    new_a = th.empty((0, 2), device=device)
    
    # 12. МАТРИЦА РАЗМНОЖЕНИЯ
    global replication_matrix
    replication_matrix = None
    
    # 13. ОБНОВЛЯЕМ СЛОВАРЬ id_to_index
    _update_id_to_index()
    
    print(f"✅ Начальная популяция инициализирована")
    print(f"   С центрами территории для всех {PopSize} взрослых особей")
    return





def _update_statuses():
    """
    ОБНОВЛЯЕТ СТАТУСЫ ВСЕХ ОСОБЕЙ на основе возраста.
    """
    global status
    
    if PopSize == 0:
        return
    
    print(f"\n🔍 _update_statuses(): день {CurrentTime}, PopSize={PopSize}")
    
    # Статистика до обновления
    children_before = (status == STATUS_CHILD).sum().item()
    juv_no_terr_before = (status == STATUS_JUVENILE_NO_TERR).sum().item()
    juv_terr_before = (status == STATUS_JUVENILE_TERR).sum().item()
    adults_before = (status == STATUS_ADULT).sum().item()
    
    print(f"  До обновления: дети={children_before}, "
          f"ювенилы_без_территории={juv_no_terr_before}, "
          f"ювенилы_с_территорией={juv_terr_before}, взрослые={adults_before}")
    
    # ДИАГНОСТИКА: возраст ювенилов
    if juv_terr_before > 0:
        juv_terr_ages = Age[status == STATUS_JUVENILE_TERR]
        print(f"  Ювенилы с территорией: возраст min={juv_terr_ages.min().item()}, "
              f"max={juv_terr_ages.max().item()}, среднее={juv_terr_ages.mean().item():.1f}")
        # Сколько из них достигли 220 дней?
        becoming_adult_count = (juv_terr_ages >= 220).sum().item()
        exactly_220 = (juv_terr_ages == 220).sum().item()
        print(f"  Достигли 220+ дней: {becoming_adult_count}, точно 220 дней: {exactly_220}")
    
    # 1. Кто становится ювенилом? (достиг 160 дней, был ребенком)
    becoming_juvenile = (Age == 160) & (status == STATUS_CHILD)
    if becoming_juvenile.any():
        count = becoming_juvenile.sum().item()
        status[becoming_juvenile] = STATUS_JUVENILE_NO_TERR
        print(f"🎉 {count} детей стали ювенилами (достигли 160 дней)")
    
    # 2. Кто становится взрослым? (достиг 220 дней, был ювенилом)
    becoming_adult = (Age == 220) & ((status == STATUS_JUVENILE_NO_TERR) | (status == STATUS_JUVENILE_TERR))
    if becoming_adult.any():
        count = becoming_adult.sum().item()
        # Проверяем, сколько из них были с территорией
        was_with_terr = (status[becoming_adult] == STATUS_JUVENILE_TERR).sum().item()
        was_without_terr = count - was_with_terr
        
        status[becoming_adult] = STATUS_ADULT
        print(f"🎉 {count} ювенилов стали взрослыми (достигли 220 дней)")
        print(f"   Из них: с территорией={was_with_terr}, без территории={was_without_terr}")
    
    # Статистика после обновления
    children_after = (status == STATUS_CHILD).sum().item()
    juv_no_terr_after = (status == STATUS_JUVENILE_NO_TERR).sum().item()
    juv_terr_after = (status == STATUS_JUVENILE_TERR).sum().item()
    adults_after = (status == STATUS_ADULT).sum().item()
    
    print(f"  После обновления: дети={children_after}, "
          f"ювенилы_без_территории={juv_no_terr_after}, "
          f"ювенилы_с_территорией={juv_terr_after}, взрослые={adults_after}")


def SeedPathogen(AmountOfPathogens, types, init=False):
    ''' Заражение начальной инфекцией - ТОЛЬКО РЕЗИДЕНТОВ '''
    print(f"🦠 SeedPathogen(): первичное заражение {AmountOfPathogens} особей (день {CurrentTime})")
    global InfectionStatus
    
    # Находим резидентов (взрослые + ювенилы с территорией)
    residents_mask = (status == STATUS_ADULT) | (status == STATUS_JUVENILE_TERR)
    residents_indices = residents_mask.nonzero(as_tuple=True)[0]
    
    if len(residents_indices) == 0:
        print("⚠️ SeedPathogen(): нет резидентов для заражения")
        return
    
    # Выбираем случайных резидентов для заражения
    available = min(AmountOfPathogens, len(residents_indices))
    index = th.randperm(residents_indices.shape[0], device=device)[:available]
    selected_indices = residents_indices[index]
    
    # Заражаем выбранных резидентов
    InfectionStatus[selected_indices] = 1  # Начинают с латентной фазы
    AgeOfDisease[selected_indices] = 0
    
    # Статистика
    infected_adults = (status[selected_indices] == STATUS_ADULT).sum().item()
    infected_juv_terr = (status[selected_indices] == STATUS_JUVENILE_TERR).sum().item()
    
    print(f"✅ Заражено {available} резидентов:")
    print(f"   - Взрослые: {infected_adults}")
    print(f"   - Ювенилы с территорией: {infected_juv_terr}")
    
    # Обновляем счетчик инфекции
    global numofinfection
    numofinfection = (InfectionStatus > 0).sum().item()
    print(f"   - Всего зараженных в популяции: {numofinfection}")
    
    return


def collect_detailed_statistics():
    """Собирает детальную статистику по генотипам"""
    if PopSize == 0:
        return {}
    
    # Подсчет генотипов
    genotype_counts = {
        '00': ((chrom_a[:, 0] == 0) & (chrom_a[:, 1] == 0)).sum().item(),
        '01': ((chrom_a[:, 0] == 0) & (chrom_a[:, 1] == 1)).sum().item(),
        '10': ((chrom_a[:, 0] == 1) & (chrom_a[:, 1] == 0)).sum().item(),
        '11': ((chrom_a[:, 0] == 1) & (chrom_a[:, 1] == 1)).sum().item(),
    }
    
    # Частоты аллелей
    total_alleles = PopSize * 2
    allele_1_freq = chrom_a.sum().item() / total_alleles
    allele_0_freq = 1 - allele_1_freq
    
    # Средняя приспособленность по генотипам
    if PopSize > 0:
        fitness_by_genotype = {
            '00': Fitness[(chrom_a[:, 0] == 0) & (chrom_a[:, 1] == 0)].mean().item() if genotype_counts['00'] > 0 else 0,
            '01': Fitness[((chrom_a[:, 0] == 0) & (chrom_a[:, 1] == 1)) | 
                         ((chrom_a[:, 0] == 1) & (chrom_a[:, 1] == 0))].mean().item() if (genotype_counts['01'] + genotype_counts['10']) > 0 else 0,
            '11': Fitness[(chrom_a[:, 0] == 1) & (chrom_a[:, 1] == 1)].mean().item() if genotype_counts['11'] > 0 else 0,
        }
    else:
        fitness_by_genotype = {'00': 0, '01': 0, '11': 0}
    
    return {
        'time': CurrentTime,
        'genotype_00': genotype_counts['00'],
        'genotype_01': genotype_counts['01'] + genotype_counts['10'],
        'genotype_11': genotype_counts['11'],
        'allele_0_freq': allele_0_freq,
        'allele_1_freq': allele_1_freq,
        'fitness_00': fitness_by_genotype['00'],
        'fitness_01': fitness_by_genotype['01'],
        'fitness_11': fitness_by_genotype['11'],
    }



def distance_calculator(mat_x: th.Tensor, size: int, mapsize: float) -> th.Tensor:
    """
    Расчёт разницы координат между всеми парами точек без тороидальности.
    diff[i, j] = mat_x[i] - mat_x[j]
    """
    # Используем broadcasting для создания матрицы разниц
    diff = mat_x.unsqueeze(1) - mat_x.unsqueeze(0)   # размер (N, N)
    
    # Возвращаем просто разницу, без обёртки через края карты
    return diff





# ==================== ФУНКЦИЯ ПОИСКА ТЕРРИТОРИИ ДЛЯ ЮВЕНИЛОВ ====================

def _find_territory_for_juveniles():
    """
    Ищет территорию для ювенилов без территории (STATUS_JUVENILE_NO_TERR).
    Условия:
    1. Возраст 160-219 дней
    2. Fitness > 95
    3. Нет текущей территории
    4. Устанавливает центр территории = текущие координаты
    """
    global status, territory_center_x, territory_center_y
    
    if PopSize == 0:
        return 0
    
    # 1. Находим ТОЛЬКО ювенилов без территории
    juv_no_terr_mask = (status == STATUS_JUVENILE_NO_TERR)
    juv_no_terr_indices = juv_no_terr_mask.nonzero(as_tuple=True)[0]
    
    if len(juv_no_terr_indices) == 0:
        return 0
    
    # 2. Проверяем условие Fitness > 95
    can_get_territory = (Fitness[juv_no_terr_indices] > 95)
    successful_indices = juv_no_terr_indices[can_get_territory]
    
    if successful_indices.shape[0] == 0:
        return 0
    
    # 3. Устанавливаем центры территории для этих ювенилов
    territory_center_x[successful_indices] = X[successful_indices].clone()
    territory_center_y[successful_indices] = Y[successful_indices].clone()
    
    # 4. Обновляем статус
    status[successful_indices] = STATUS_JUVENILE_TERR
    
    successful_count = successful_indices.shape[0]
    
    # 5. ДЕТАЛЬНАЯ СТАТИСТИКА
    if successful_count > 0 and CurrentTime % 10 == 0:  # Реже выводим
        print(f"🏡 _find_territory_for_juveniles(): {successful_count} ювенилов получили территорию")
        
        # Покажем только первые 2 для экономии логов
        for i in range(min(2, successful_count)):
            idx = successful_indices[i]
            x_center = territory_center_x[idx].item()
            y_center = territory_center_y[idx].item()
            current_x = X[idx].item()
            current_y = Y[idx].item()
            fitness = Fitness[idx].item()
            
            print(f"   #{i}: ID={animal_ids[idx].item()}, "
                  f"Центр=({x_center:.0f}, {y_center:.0f}), "
                  f"Текущая позиция=({current_x:.0f}, {current_y:.0f}), "
                  f"Fitness={fitness:.1f}")
    
    return successful_count


# ==================== ОБНОВЛЯЕМ ФУНКЦИЮ CalculateAreaAndFitness() ====================


def CalculateAreaAndFitness(summ):
    """
    Рассчитывает Area и Fitness для всех особей.
    Теперь используется для определения, могут ли ювенилы получить территорию.
    """
    global Fitness
    
    # 1. Рассчитываем Area (старая логика)
    Area = (math.pi - summ / 2) * RangeSq
    Area = th.nn.functional.relu(Area)
    
    if summ.isnan().sum() > 0:
        print("angle error")
    elif Area.isnan().sum() > 0:
        print("area error")
    
    # 2. Рассчитываем Fitness (старая логика)
    Fitness = 100 / (1 + math.e ** (5 - 10 * Area / MaxArea))
    
    # 3. УБИРАЕМ СТАРУЮ ЛОГИКУ С juvenile
    # Было: juv = (Fitness > 95) * juvenile
    #       juvenile -= juv
    # Теперь: статус территории обновляется в _find_territory_for_juveniles()
    
    # 4. Вывод статистики по фитнесу
    if PopSize > 0:
        avg_fitness = Fitness.mean().item()
        juv_count = (status == STATUS_JUVENILE_NO_TERR).sum().item()
        juv_with_high_fitness = ((status == STATUS_JUVENILE_NO_TERR) & (Fitness > 95)).sum().item()
        
        if juv_count > 0:
            print(f"📊 CalculateAreaAndFitness(): средний фитнес={avg_fitness:.1f}, "
                  f"ювенилов={juv_count}, из них с Fitness>95={juv_with_high_fitness}")
            






def MovementAndInfection(replication=False):
    global X, Y, speedX, speedY, InfectionStatus, new_Y, new_X
    global numofinfection, CurrentTime, status
    global territory_center_x, territory_center_y
    
    # 0. ПЕРВИЧНОЕ ЗАРАЖЕНИЕ (если нужно)
    if CurrentTime > TimeOfPathogen and numofinfection == 0:
        SeedPathogen(10, 1)

    # ==================== ДИАГНОСТИКА: КОГДА ВЫЗЫВАЕТСЯ Replication? ====================
    day_in_year = CurrentTime % 120
    print(f"\n🔍 MovementAndInfection(): день {CurrentTime}, день_в_году={day_in_year}, replication={replication}")
    
    # 1. РАСЧЕТ РАССТОЯНИЙ
    distance_x = distance_calculator(X, PopSize, MapXSize)
    distance_y = distance_calculator(Y, PopSize, MapYSize)
    distance_sq = distance_x**2 + distance_y**2
    
    # 2. ВЗАИМОДЕЙСТВИЯ - ДЛЯ РАЗНЫХ ЦЕЛЕЙ
    
    # 2.1. Резиденты (для статистики, инфекции, территорий)
    resident_mask = (status == STATUS_ADULT) | (status == STATUS_JUVENILE_TERR)
    
    # 2.2. Взрослые (для размножения) - как в старой версии
    adult_mask = (status == STATUS_ADULT)
    
    # 2.3. Маска взаимодействий по расстоянию
    interaction_by_distance = (distance_sq < MaxRadiusSq)
    
    # 2.4. Маска для размножения: взрослый-взрослый (как в старой версии)
    adult_matrix = adult_mask.unsqueeze(1) & adult_mask.unsqueeze(0)
    interaction_for_replication = interaction_by_distance & adult_matrix
    
    # 3. РАЗМНОЖЕНИЕ (если период размножения)
    if replication:
        print(f"  ✅ ВЫЗЫВАЕМ Replication()...")
        # ПРОСТОЙ вызов как в старой версии
        Replication(interaction_for_replication)
    else:
        print(f"  ⏭️ Пропуск Replication (не период размножения)")
        

    # 4. РАСЧЕТ ПЕРЕКРЫТИЯ ТЕРРИТОРИЙ для фитнеса
    X_copy = (distance_sq < 4 * RangeSq) * 1 - distance_sq.eq(0) * 1
    AngleAlpha = 2 * X_copy * th.acos(X_copy * th.sqrt(distance_sq / RangeSq / 4))
    summ = (AngleAlpha - th.sin(AngleAlpha)).sum(dim=1)
    
    # 5. ФИЗИКА ДВИЖЕНИЯ
    # 5.1. ОТТАЛКИВАНИЕ ОТ ОСОБЕЙ - ДЛЯ ВСЕХ!
    mask = (distance_sq < MaxRadiusSq)
    distanceCube = (distance_sq)**1.5 + distance_sq.eq(0)
    
    NewSpeedX = (distance_x * mask / distanceCube).sum(1) * Repulsion
    NewSpeedY = (distance_y * mask / distanceCube).sum(1) * Repulsion
    
    speedX = speedX + NewSpeedX
    speedY = speedY + NewSpeedY
    
    # 5.2. ОТТАЛКИВАНИЕ ОТ СТЕН - ДЛЯ ВСЕХ!
    wall_threshold = 100
    wall_repulsion = Repulsion * 1
    
    near_left = X < wall_threshold
    if near_left.any():
        force = (wall_threshold - X[near_left]) / wall_threshold * wall_repulsion
        speedX[near_left] += force
    
    near_right = X > (MapXSize - wall_threshold)
    if near_right.any():
        force = (X[near_right] - (MapXSize - wall_threshold)) / wall_threshold * -wall_repulsion
        speedX[near_right] += force
    
    near_bottom = Y < wall_threshold
    if near_bottom.any():
        force = (wall_threshold - Y[near_bottom]) / wall_threshold * wall_repulsion
        speedY[near_bottom] += force
    
    near_top = Y > (MapYSize - wall_threshold)
    if near_top.any():
        force = (Y[near_top] - (MapYSize - wall_threshold)) / wall_threshold * -wall_repulsion
        speedY[near_top] += force
    
    # 5.3. СИЛА ПРИВЯЗКИ К ЦЕНТРУ ТЕРРИТОРИИ - ТОЛЬКО ДЛЯ РЕЗИДЕНТОВ!
    if resident_mask.any():
        # Находим резидентов, у которых есть территория (значение != -1)
        has_territory = (territory_center_x != -1) & resident_mask
        
        if has_territory.any():
            # Рассчитываем расстояние до центра территории
            dist_to_center_x = X[has_territory] - territory_center_x[has_territory]
            dist_to_center_y = Y[has_territory] - territory_center_y[has_territory]
            dist_to_center_sq = dist_to_center_x**2 + dist_to_center_y**2
            
            # Сила привязки: тем сильнее, чем дальше от центра
            territory_attraction = 0.05  # Сила возврата к центру
            
            # Применяем силу привязки
            speedX[has_territory] -= dist_to_center_x * territory_attraction
            speedY[has_territory] -= dist_to_center_y * territory_attraction
            
            # СТАТИСТИКА: сколько резидентов далеко от центра территории
            if CurrentTime % 30 == 0:  # Только раз в 30 дней для экономии логов
                far_from_center = dist_to_center_sq > (RangeSq * 0.25)  # Более чем в половине радиуса
                if far_from_center.any():
                    print(f"📍 {far_from_center.sum().item()} резидентов далеко от центра территории")
    
    # 5.4. ДВИЖЕНИЕ ВСЕХ ОСОБЕЙ
    speedX = th.clamp(speedX, -MaxSpeed*dispersal, MaxSpeed*dispersal)
    speedY = th.clamp(speedY, -MaxSpeed*dispersal, MaxSpeed*dispersal)
    
    X = X + speedX
    Y = Y + speedY
    
    # 5.5. КОРРЕКТИРОВКА ГРАНИЦ - ДЛЯ ВСЕХ
    mask_left = X < 0
    mask_right = X > MapXSize
    mask_bottom = Y < 0
    mask_top = Y > MapYSize
    
    speedX[mask_left | mask_right] *= -1
    speedY[mask_bottom | mask_top] *= -1
    
    X = th.clamp(X, 0, MapXSize)
    Y = th.clamp(Y, 0, MapYSize)
    
    # 5.6. ОГРАНИЧЕНИЕ РЕЗИДЕНТОВ ПРЕДЕЛАМИ ТЕРРИТОРИИ
    if resident_mask.any():
        # Находим резидентов, у которых есть территория
        has_territory = (territory_center_x != -1) & resident_mask
        
        if has_territory.any():
            # Рассчитываем расстояние до центра территории после движения
            dist_to_center_x = X[has_territory] - territory_center_x[has_territory]
            dist_to_center_y = Y[has_territory] - territory_center_y[has_territory]
            dist_to_center = th.sqrt(dist_to_center_x**2 + dist_to_center_y**2)
            
            # Те, кто вышел за пределы территории
            outside_territory = dist_to_center > Range
            
            if outside_territory.any():
                # Возвращаем их к границе территории
                correction_factor = Range / dist_to_center[outside_territory]
                
                # Корректируем позиции
                X_correction = territory_center_x[has_territory][outside_territory] + \
                              dist_to_center_x[outside_territory] * correction_factor
                Y_correction = territory_center_y[has_territory][outside_territory] + \
                              dist_to_center_y[outside_territory] * correction_factor
                
                X[has_territory.nonzero(as_tuple=True)[0][outside_territory]] = X_correction
                Y[has_territory.nonzero(as_tuple=True)[0][outside_territory]] = Y_correction
                
                # Обнуляем скорость в направлении от центра
                correction_indices = has_territory.nonzero(as_tuple=True)[0][outside_territory]
                for idx in correction_indices:
                    # Проекция скорости на направление от центра
                    dx = X[idx] - territory_center_x[idx]
                    dy = Y[idx] - territory_center_y[idx]
                    if dx != 0 or dy != 0:
                        dir_magnitude = th.sqrt(dx**2 + dy**2)
                        dir_x = dx / dir_magnitude
                        dir_y = dy / dir_magnitude
                        
                        # Скалярное произведение скорости на направление
                        speed_proj = speedX[idx] * dir_x + speedY[idx] * dir_y
                        
                        # Если скорость направлена от центра, уменьшаем ее
                        if speed_proj > 0:
                            speedX[idx] -= dir_x * speed_proj * 0.5
                            speedY[idx] -= dir_y * speed_proj * 0.5
                
                # Статистика (только если есть изменения и не слишком часто)
                if CurrentTime % 30 == 0:
                    num_outside = outside_territory.sum().item()
                    resident_count = resident_mask.sum().item()
                    print(f"📍 {num_outside}/{resident_count} резидентов вышли за пределы территории и были возвращены")
    
    # 6. ПЕРЕДАЧА ИНФЕКЦИИ
    
    # 6.0. ИНИЦИАЛИЗАЦИЯ
    new_infections = th.zeros(PopSize, dtype=th.bool, device=device)
    
    # 6.1. Кто может передавать инфекцию? 
    can_transmit_mask = (InfectionStatus > 0) & resident_mask
    
    # 6.2. Кто может получать инфекцию?
    can_be_infected = (InfectionStatus == 0) & resident_mask
    
    if can_transmit_mask.any() and can_be_infected.any():
        # ИНДЕКСЫ ЗАРАЗНЫХ И ВОСПРИИМЧИВЫХ ОСОБЕЙ
        transmitter_indices = can_transmit_mask.nonzero(as_tuple=True)[0]
        susceptible_indices = can_be_infected.nonzero(as_tuple=True)[0]
        
        # МАТРИЦА РАССТОЯНИЙ МЕЖДУ ЗАРАЗНЫМИ И ВОСПРИИМЧИВЫМИ
        if len(transmitter_indices) > 0 and len(susceptible_indices) > 0:
            # Координаты заразных
            X_t = X[transmitter_indices]
            Y_t = Y[transmitter_indices]
            
            # Координаты восприимчивых
            X_s = X[susceptible_indices]
            Y_s = Y[susceptible_indices]
            
            # Матрица расстояний^2
            dist_x_matrix = X_t.unsqueeze(1) - X_s.unsqueeze(0)
            dist_y_matrix = Y_t.unsqueeze(1) - Y_s.unsqueeze(0)
            dist_sq_matrix = dist_x_matrix**2 + dist_y_matrix**2
            
            # ФАКТОР РАССТОЯНИЯ
            distance_factor = th.clamp((MaxRadiusSq - dist_sq_matrix) / MaxRadiusSq, 0, 1)
            
            # МАСКА ВЗАИМОДЕЙСТВИЙ (в пределах MaxRadius)
            interaction_matrix = dist_sq_matrix < MaxRadiusSq
            
            # ОСНОВНАЯ МАТРИЦА ПЕРЕДАЧИ
            transmission_matrix = interaction_matrix.float() * distance_factor
            
            # ПОЛОВАЯ И НЕПОЛОВАЯ ПЕРЕДАЧА
            if replication and (CurrentTime % 120 <= 10):
                # ДНИ 0-10: Infectivity1 + Infectivity2
                
                # 6.3. ПОЛОВАЯ ПЕРЕДАЧА (Infectivity1): только между взрослыми
                transmitter_adult_mask = adult_mask[transmitter_indices]
                susceptible_adult_mask = adult_mask[susceptible_indices]
                
                # Только те, кто в фазе 1 (латентной) передают половым путем
                phase1_transmitters = (InfectionStatus[transmitter_indices] == 1)
                
                # Маска для половой передачи
                sexual_mask = transmitter_adult_mask.unsqueeze(1) & susceptible_adult_mask.unsqueeze(0) & phase1_transmitters.unsqueeze(1)
                
                # Вероятность половой передачи
                sexual_transmission = transmission_matrix * sexual_mask.float() * Infectivity1
                sexual_transmission = th.clamp(sexual_transmission, 0, 1)
                
                # 6.4. НЕПОЛОВАЯ ПЕРЕДАЧА (Infectivity2): между всеми резидентами
                nonsexual_transmission = transmission_matrix * Infectivity2
                nonsexual_transmission = th.clamp(nonsexual_transmission, 0, 1)
                
                # ОБЩАЯ ВЕРОЯТНОСТЬ ПЕРЕДАЧИ
                total_transmission = sexual_transmission + nonsexual_transmission
                total_transmission = th.clamp(total_transmission, 0, 1)
                
            else:
                # ДНИ 11-119: только Infectivity2
                total_transmission = transmission_matrix * Infectivity2
                total_transmission = th.clamp(total_transmission, 0, 1)
            
            # 6.5. ПРИМЕНЯЕМ ВЕРОЯТНОСТЬ ЗАРАЖЕНИЯ
            # Для каждого восприимчивого: вероятность заразиться от ЛЮБОГО заразного
            max_prob_per_susceptible = total_transmission.max(dim=0)[0]
            
            # Случайные числа для каждого восприимчивого
            rand_nums = th.rand(len(susceptible_indices), device=device)
            
            # Кто заразился?
            infected_susceptible_mask = rand_nums < max_prob_per_susceptible
            
            if infected_susceptible_mask.any():
                # Индексы заразившихся в общей популяции
                new_infected_global_indices = susceptible_indices[infected_susceptible_mask]
                
                # Заражаем
                InfectionStatus[new_infected_global_indices] = 1
                AgeOfDisease[new_infected_global_indices] = 0
                new_infections[new_infected_global_indices] = True
                
                # Статистика
                num_new = infected_susceptible_mask.sum().item()
                
                # Уменьшаем частоту вывода для экономии логов
                if num_new > 5 and CurrentTime % 10 == 0:
                    adults_new = adult_mask[new_infected_global_indices].sum().item()
                    juv_terr_new = (~adult_mask[new_infected_global_indices] & resident_mask[new_infected_global_indices]).sum().item()
                    
                    print(f"🦠 Новые заражения: {num_new} резидентов")
                    if replication and (CurrentTime % 120 <= 10):
                        print(f"   Тип передачи: половая+неполовая (дни 0-10)")
                    else:
                        print(f"   Тип передачи: только неполовая (дни 11-119)")
    
    # 6.6. Обновляем счетчик инфекции
    numofinfection = (InfectionStatus > 0).sum().item()
    
    # 7. РАСЧЕТ ФИТНЕСА
    CalculateAreaAndFitness(summ)
    
    # 8. СТАТИСТИКА
    day_in_year = CurrentTime % 120
    resident_count = resident_mask.sum().item()
    infected_residents = (resident_mask & (InfectionStatus > 0)).sum().item()
    infected_phase1 = (resident_mask & (InfectionStatus == 1)).sum().item()
    infected_phase2 = (resident_mask & (InfectionStatus == 2)).sum().item()
    
    # СТАТИСТИКА ПО ТЕРРИТОРИЯМ
    if resident_mask.any() and CurrentTime % 30 == 0:  # Только раз в 30 дней
        has_territory = (territory_center_x != -1) & resident_mask
        if has_territory.any():
            # Среднее расстояние до центра территории
            dist_x = X[has_territory] - territory_center_x[has_territory]
            dist_y = Y[has_territory] - territory_center_y[has_territory]
            avg_dist = th.sqrt(dist_x**2 + dist_y**2).mean().item()
            
            # Процент внутри территории
            inside_territory = th.sqrt(dist_x**2 + dist_y**2) <= Range
            inside_pct = (inside_territory.sum().item() / has_territory.sum().item() * 100)
            
            print(f"📍 Территории: {has_territory.sum().item()} резидентов с территорией")
            print(f"   Среднее расстояние до центра: {avg_dist:.1f} (макс {Range})")
            print(f"   Внутри территории: {inside_pct:.1f}%")
    
    # Уменьшаем частоту вывода основного сообщения
    if CurrentTime % 30 == 0:
        print(f"🏃 MovementAndInfection(): день {day_in_year} (всего {CurrentTime})")
        print(f"   Резиденты: {resident_count}, зараженные резидентов: {infected_residents}")
        print(f"   Фазы: латентная={infected_phase1}, больная={infected_phase2}")



def AddAnimal():
    """
    Добавление новорожденных с правильным определением пола по аллелям.
    """
    global Fitness, X, Y, AgeOfDisease, new_sex, new_a
    global speedX, speedY, PopSize, Age, InfectionStatus, new_X, new_Y
    global chrom_sex, chrom_a, Sex, status
    global animal_ids, next_animal_id
    global territory_center_x, territory_center_y
    
    print(f"\n👶 AddAnimal(): начало, PopSize={PopSize}")
    
    # 0. ПРОВЕРКА ВХОДНЫХ ДАННЫХ
    if new_X is None or new_Y is None or len(new_X) == 0:
        print("⚠️ AddAnimal(): пустые координаты")
        return
    
    if new_sex is None or new_a is None or new_sex.numel() == 0:
        print("⚠️ AddAnimal(): нет генетической информации")
        return
    
    print(f"  new_X shape: {new_X.shape}, new_sex shape: {new_sex.shape}, new_a shape: {new_a.shape}")
    
    # 1. Проверяем размерности
    total_children_expected = new_X.shape[0]
    
    # Преобразуем new_sex и new_a к правильной форме
    if len(new_sex.shape) == 3:
        # [самки, потомки, 2] -> [все_потомки, 2]
        num_females = new_sex.shape[0]
        num_progeny = new_sex.shape[1]
        total_children = num_females * num_progeny
        
        if total_children != total_children_expected:
            print(f"⚠️ Несоответствие: ожидалось {total_children_expected} координат, генетика для {total_children} детей")
            # Берем минимум
            total_children_to_use = min(total_children, total_children_expected)
        else:
            total_children_to_use = total_children_expected
        
        # Преобразуем в плоский вид
        new_sex_flat = new_sex.reshape(-1, 2)[:total_children_to_use]
        new_a_flat = new_a.reshape(-1, 2)[:total_children_to_use]
        
    elif len(new_sex.shape) == 2:
        # Уже плоский [потомки, 2]
        new_sex_flat = new_sex[:total_children_expected]
        new_a_flat = new_a[:total_children_expected]
        total_children_to_use = min(new_sex_flat.shape[0], total_children_expected)
    else:
        print(f"⚠️ Неверная размерность new_sex: {new_sex.shape}")
        return
    
    # Обрезаем координаты до количества детей с генетикой
    new_X = new_X[:total_children_to_use]
    new_Y = new_Y[:total_children_to_use]
    
    num_new = len(new_X)
    
    if num_new == 0:
        print("⚠️ Нет детей для добавления")
        return
    
    print(f"  Добавляем {num_new} детей")
    
    # 2. ОГРАНИЧЕНИЕ ПО МАКСИМАЛЬНОЙ ПОПУЛЯЦИИ
    available_space = MaxPopSize - PopSize
    
    if available_space <= 0:
        print(f"⚠️ Достигнут максимальный размер популяции {MaxPopSize}")
        return
    
    if num_new > available_space:
        print(f"⚠️ Доступно место только для {available_space} детей из {num_new}")
        num_new = available_space
        new_X = new_X[:num_new]
        new_Y = new_Y[:num_new]
        new_sex_flat = new_sex_flat[:num_new]
        new_a_flat = new_a_flat[:num_new]
    
    # 3. СОЗДАЕМ НОВЫЕ ОСОБИ
    zero_list = th.zeros(num_new, device=device)
    
    # Уникальные ID
    new_ids = th.arange(next_animal_id, next_animal_id + num_new, 
                       dtype=th.int64, device=device)
    next_animal_id += num_new
    
    # Фитнес - максимальный
    Fitness = th.cat((Fitness, zero_list + 100))
    
    # ==================== ОПРЕДЕЛЕНИЕ ПОЛА ПО АЛЛЕЛЯМ ====================
    # new_sex_flat имеет форму [потомки, 2] где:
    # [аллель от матери для пола, аллель от отца для пола]
    # 0 = X, 1 = Y
    # Пол = сумма аллелей: 0+0=0 (самка), 0+1=1 (самец)
    
    if new_sex_flat.shape[1] == 2:
        # Вычисляем пол: 0 = самка, 1 = самец
        sex_of_children = new_sex_flat.sum(dim=1)  # Суммируем аллели
        
        # Проверяем, что все значения 0 или 1
        if (sex_of_children > 1).any():
            print(f"⚠️ Ошибка: некорректные аллели пола: {sex_of_children.unique()}")
            # Исправляем: все что >1 делаем 1 (самец)
            sex_of_children = th.clamp(sex_of_children, 0, 1)
        
        females = (sex_of_children == 0).sum().item()
        males = (sex_of_children == 1).sum().item()
        print(f"  Пол потомков: самки={females} ({females/num_new*100:.1f}%), "
              f"самцы={males} ({males/num_new*100:.1f}%)")
        
        # Добавляем пол в тензор Sex
        Sex = th.cat((Sex, sex_of_children))
    else:
        print(f"⚠️ Неправильная форма new_sex_flat: {new_sex_flat.shape}")
        # Запасной вариант: случайный пол 50/50
        sex_of_children = th.randint(0, 2, (num_new,), device=device)
        Sex = th.cat((Sex, sex_of_children))
    
    # Координаты
    X = th.cat((X, new_X))
    Y = th.cat((Y, new_Y))
    
    # Генотипы
    chrom_sex = th.cat((chrom_sex, new_sex_flat))
    chrom_a = th.cat((chrom_a, new_a_flat))
    
    # Центры территории - дети не имеют территории
    if territory_center_x is None:
        territory_center_x = th.full((num_new,), -1.0, device=device)
        territory_center_y = th.full((num_new,), -1.0, device=device)
    else:
        territory_center_x = th.cat((territory_center_x, th.full((num_new,), -1.0, device=device)))
        territory_center_y = th.cat((territory_center_y, th.full((num_new,), -1.0, device=device)))
    
    # Скорость
    speedX = th.cat((speedX, MaxSpeed - th.rand(num_new, device=device) * MaxSpeed * 2))
    speedY = th.cat((speedY, MaxSpeed - th.rand(num_new, device=device) * MaxSpeed * 2))
    
    # Возраст - 85 дней (timeOfDisposal - 15)
    Age = th.cat((Age, zero_list + (timeOfDisposal - 15)))
    
    # Статус - ДЕТИ
    new_status = th.full((num_new,), STATUS_CHILD, dtype=th.long, device=device)
    status = th.cat((status, new_status))
    
    # Инфекция - все здоровы
    InfectionStatus = th.cat((InfectionStatus, zero_list))
    AgeOfDisease = th.cat((AgeOfDisease, zero_list))
    
    # ID
    animal_ids = th.cat((animal_ids, new_ids))
    
    # 4. ОБНОВЛЕНИЕ РАЗМЕРА
    old_pop_size = PopSize
    PopSize += num_new
    
    # 5. ОБНОВЛЯЕМ СЛОВАРЬ
    _update_id_to_index()
    
    # 6. ДИАГНОСТИКА ГЕНОТИПОВ
    if num_new > 0:
        # Генотипы по размножению
        semel_children = (new_a_flat.sum(1) == 2).sum().item()
        itero_children = (new_a_flat.sum(1) == 0).sum().item()
        hetero_children = num_new - semel_children - itero_children
        
        print(f"✅ Добавлено {num_new} детей")
        print(f"   Возраст: {timeOfDisposal - 15} дней")
        print(f"   Статус: дети (STATUS_CHILD)")
        print(f"   Территория: нет (-1)")
        print(f"   Генотипы: [1,1]={semel_children}, [0,0]={itero_children}, гетерозиготы={hetero_children}")
        print(f"   Популяция: {old_pop_size} → {PopSize}")
    
    # 7. ОЧИСТКА
    new_X = th.empty(0, device=device)
    new_Y = th.empty(0, device=device)
    new_sex = th.empty((0, numofprogeny, 2), device=device)
    new_a = th.empty((0, numofprogeny, 2), device=device)




def DeleteAnimal(selection_mask):
    global AgeOfDisease, Fitness, X, Y, Sex, replication_matrix
    global speedX, speedY, PopSize, Age, InfectionStatus, dead, CurrentTime
    global chrom_sex, chrom_a, new_X, new_Y, new_a, new_sex
    global status, animal_ids, territory_center_x, territory_center_y
    
    print(f"\n💀 DeleteAnimal(): день {CurrentTime}")
    
    if PopSize == 0:
        return
    
    old_pop_size = PopSize
    
    # 1. Сохраняем маски ДО удаления
    male_mask_before = (Sex == 1)
    female_mask_before = (Sex == 0)
    adult_female_mask_before = (Sex == 0) & (status == STATUS_ADULT)
    
    # Индексы взрослых самок до удаления
    adult_female_indices_before = adult_female_mask_before.nonzero(as_tuple=True)[0]
    
    # 2. УДАЛЕНИЕ ИЗ ОСНОВНЫХ ТЕНЗОРОВ
    Age = Age.masked_select(selection_mask)
    Fitness = Fitness.masked_select(selection_mask)
    X = X.masked_select(selection_mask)
    Y = Y.masked_select(selection_mask)
    speedX = speedX.masked_select(selection_mask)
    speedY = speedY.masked_select(selection_mask)
    AgeOfDisease = AgeOfDisease.masked_select(selection_mask)
    InfectionStatus = InfectionStatus.masked_select(selection_mask)
    
    # 3. ПОДСЧЕТ СМЕРТЕЙ
    deaths = selection_mask.eq(0).sum().item()
    dead += deaths
    
    print(f"  Смертей: {deaths}")
    
    # 4. КРИТИЧЕСКОЕ: фильтрация new_a и new_sex при смерти матерей
    day_in_year = CurrentTime % 120
    
    # Если смерть произошла ДО дня 100 (до рождения потомства) И есть new_a
    if day_in_year < timeOfDisposal and new_a is not None and new_a.numel() > 0:
        print(f"  🔍 Фильтрация new_a при смерти матерей (день {day_in_year} < {timeOfDisposal})")
        
        # НОВАЯ ЛОГИКА: Фильтруем ТОЛЬКО взрослых самок, которые участвуют в размножении
        
        # Создаем маску выживших взрослых самок
        # 1. Получаем текущую маску взрослых самок (после удаления, но до обновления других тензоров)
        adult_female_mask_after = th.zeros(old_pop_size, dtype=th.bool, device=device)
        # Копируем маску взрослых самок
        adult_female_mask_after[adult_female_indices_before] = True
        # Применяем маску выживания
        adult_female_mask_after = adult_female_mask_after & selection_mask
        
        # 2. Находим индексы выживших взрослых самок в исходном порядке
        survived_adult_female_indices = adult_female_indices_before[selection_mask[adult_female_indices_before]]
        
        # 3. Проверяем соответствие размеров
        num_adult_females_before = len(adult_female_indices_before)
        num_adult_females_after = len(survived_adult_female_indices)
        
        print(f"    Взрослых самок было: {num_adult_females_before}, осталось: {num_adult_females_after}")
        print(f"    Размер new_a до фильтрации: {new_a.shape}")
        
        # 4. Если new_a содержит данные для всех взрослых самок
        if new_a.shape[0] == num_adult_females_before:
            # Создаем маску выживших для new_a
            survival_mask_for_new_a = selection_mask[adult_female_indices_before]
            
            # Фильтруем new_a и new_sex
            new_a = new_a[survival_mask_for_new_a]
            new_sex = new_sex[survival_mask_for_new_a]
            
            print(f"    Размер new_a после фильтрации: {new_a.shape}")
            
            # Проверяем генотипы оставшихся потомков
            if new_a.numel() > 0:
                all_children = new_a.reshape(-1, 2)
                semel_children = (all_children.sum(1) == 2).sum().item()
                itero_children = (all_children.sum(1) == 0).sum().item()
                
                print(f"    Осталось потомков: {all_children.shape[0]}")
                print(f"      [1,1] семельпарные: {semel_children}")
                print(f"      [0,0] итеропарные: {itero_children}")
        else:
            print(f"    ⚠️ Размеры не совпадают: new_a={new_a.shape[0]}, взрослых самок={num_adult_females_before}")
            print(f"    🧹 Очищаем new_a и new_sex")
            new_sex = th.empty((0, numofprogeny, 2), device=device)
            new_a = th.empty((0, numofprogeny, 2), device=device)
    
    # 5. УДАЛЕНИЕ ИЗ ОСТАЛЬНЫХ ТЕНЗОРОВ
    Sex = Sex.masked_select(selection_mask)
    
    if chrom_sex is not None and len(chrom_sex) == old_pop_size:
        chrom_sex = chrom_sex.masked_select(selection_mask.unsqueeze(1).expand(-1, 2)).reshape(-1, 2)
    
    if chrom_a is not None and len(chrom_a) == old_pop_size:
        chrom_a = chrom_a.masked_select(selection_mask.unsqueeze(1).expand(-1, 2)).reshape(-1, 2)
    
    # 6. УДАЛЕНИЕ ID И СТАТУСОВ
    if animal_ids is not None and len(animal_ids) == old_pop_size:
        animal_ids = animal_ids.masked_select(selection_mask)
    
    if status is not None and len(status) == old_pop_size:
        status = status.masked_select(selection_mask)
    
    # 7. УДАЛЕНИЕ ТЕРРИТОРИЙ
    if territory_center_x is not None and len(territory_center_x) == old_pop_size:
        territory_center_x = territory_center_x.masked_select(selection_mask)
        territory_center_y = territory_center_y.masked_select(selection_mask)
    
    # 8. ОБНОВЛЕНИЕ РАЗМЕРА ПОПУЛЯЦИИ
    PopSize -= deaths
    
    # 9. ОБНОВЛЕНИЕ СЛОВАРЯ ID
    _update_id_to_index()
    
    # 10. СБРОС СЧЕТЧИКА dead В НАЧАЛЕ ГОДА
    if day_in_year == 0:
        dead = 0
    
    # 11. ФИНАЛЬНАЯ ПРОВЕРКА
    print(f"  ✅ DeleteAnimal(): удалено {deaths} особей, осталось {PopSize}")




def chrom_cal(rep, adult_female_indices=None, adult_male_indices=None):
    """
    Вычисляет генетику потомства для матрицы взаимодействий.
    
    ГЕНЕТИКА ПОЛА:
    - Мать передает X-аллель (всегда X)
    - Отец передает X или Y-аллель (случайно)
    - Потомок: XX = самка (0+0), XY = самец (0+1)
    """
    global chrom_sex, chrom_a, Sex
    
    print(f"\n🎯 chrom_cal() день {CurrentTime}")
    
    # Если индексы не переданы - используем всех
    if adult_female_indices is None:
        adult_female_indices = (Sex == 0).nonzero(as_tuple=True)[0]
    if adult_male_indices is None:
        adult_male_indices = (Sex == 1).nonzero(as_tuple=True)[0]
    
    # Берем генетику ТОЛЬКО взрослых самок и самцов
    f_sex = chrom_sex[adult_female_indices]  # [самка, аллель] - у самок всегда [X,X]?
    m_sex = chrom_sex[adult_male_indices]    # [самец, аллель] - у самцов [X,Y]
    f_a = chrom_a[adult_female_indices]      # генотип по размножению
    m_a = chrom_a[adult_male_indices]        # генотип по размножению
    
    female_number = len(adult_female_indices)
    male_number = len(adult_male_indices)
    
    print(f"  Взрослых самок: {female_number}, взрослых самцов: {male_number}")
    print(f"  Матрица rep: {rep.shape}")
    
    # Проверка размеров
    if rep.shape[0] != female_number:
        print(f"  ⚠️ Обрезаем матрицу rep: {rep.shape[0]} → {female_number}")
        if rep.shape[0] > female_number:
            rep = rep[:female_number, :]
        else:
            new_rep = th.zeros((female_number, rep.shape[1]), device=device)
            new_rep[:rep.shape[0], :] = rep
            rep = new_rep
    
    if rep.shape[1] != male_number:
        print(f"  ⚠️ Обрезаем матрицу rep: {rep.shape[1]} → {male_number}")
        if rep.shape[1] > male_number:
            rep = rep[:, :male_number]
        else:
            new_rep = th.zeros((rep.shape[0], male_number), device=device)
            new_rep[:, :rep.shape[1]] = rep
            rep = new_rep
    
    # ==================== СОЗДАНИЕ ГАМЕТ МАТЕРЕЙ ====================
    female_a = th.zeros((female_number, numofprogeny), dtype=th.long, device=device)
    
    for f_idx in range(female_number):
        mother_geno = f_a[f_idx]  # [аллель1, аллель2]
        allele1, allele2 = mother_geno[0].item(), mother_geno[1].item()
        
        for child_idx in range(numofprogeny):
            # Мать ВСЕГДА передает один из своих аллелей
            if allele1 == 0 and allele2 == 0:  # [0,0] → всегда 0
                female_a[f_idx, child_idx] = 0
            elif allele1 == 1 and allele2 == 1:  # [1,1] → всегда 1  
                female_a[f_idx, child_idx] = 1
            else:  # [0,1] или [1,0] → случайный аллель
                female_a[f_idx, child_idx] = th.randint(0, 2, (1,), device=device).item()
    
    # ==================== ВЫБОР ПАРТНЕРОВ И ГАМЕТЫ ОТЦОВ ====================
    rep2 = th.zeros_like(rep)
    male_a = th.zeros((female_number, numofprogeny), dtype=th.long, device=device)
    
    # Для пола: нужны два тензора - аллель от матери и аллель от отца
    mother_allele_for_sex = th.zeros((female_number, numofprogeny), dtype=th.long, device=device)
    father_allele_for_sex = th.zeros((female_number, numofprogeny), dtype=th.long, device=device)
    
    for f_idx in range(female_number):
        # Доступные самцы для этой самки
        available_males = (rep[f_idx] == 1).nonzero(as_tuple=True)[0]
        
        if len(available_males) > 0:
            # Выбираем случайного партнера
            rand_idx = th.randint(0, len(available_males), (1,), device=device)
            chosen_male_idx = available_males[rand_idx].item()
            
            if chosen_male_idx < male_number:
                rep2[f_idx, chosen_male_idx] = 1
                
                # ==================== СОЗДАНИЕ ГАМЕТ ОТЦОВ ====================
                father_geno = m_a[chosen_male_idx]  # Генотип отца
                allele1, allele2 = father_geno[0].item(), father_geno[1].item()
                
                for child_idx in range(numofprogeny):
                    if allele1 == 0 and allele2 == 0:  # [0,0] → всегда 0
                        male_a[f_idx, child_idx] = 0
                    elif allele1 == 1 and allele2 == 1:  # [1,1] → всегда 1
                        male_a[f_idx, child_idx] = 1
                    else:  # [0,1] или [1,0] → случайный аллель
                        male_a[f_idx, child_idx] = th.randint(0, 2, (1,), device=device).item()
                
                # ==================== ОПРЕДЕЛЕНИЕ ПОЛА ====================
                # У матери: ВСЕГДА X-аллель (0)
                mother_allele_for_sex[f_idx, :] = 0  # X от матери
                
                # У отца: случайно X (0) или Y (1)
                # Отец может передать X или Y с вероятностью 50/50
                for child_idx in range(numofprogeny):
                    father_allele_for_sex[f_idx, child_idx] = th.randint(0, 2, (1,), device=device).item()
    
    # ==================== ФОРМИРУЕМ ПОТОМСТВО ====================
    # Генотип потомков: [аллель от матери, аллель от отца]
    c_a = th.stack((female_a, male_a), 2)  # [самки, потомки, 2]
    
    # Генетика пола: [аллель от матери для пола, аллель от отца для пола]
    c_sex = th.stack((mother_allele_for_sex, father_allele_for_sex), 2)  # [самки, потомки, 2]
    
    # ==================== ДИАГНОСТИКА ====================
    mated_females = (rep2.sum(1) > 0).sum().item()
    print(f"  Нашли партнеров: {mated_females}/{female_number} взрослых самок")
    
    # Статистика потомков
    if c_a.numel() > 0:
        all_children_geno = c_a.reshape(-1, 2)
        total_children = all_children_geno.shape[0]
        
        if total_children > 0:
            semel_children = (all_children_geno.sum(1) == 2).sum().item()
            itero_children = (all_children_geno.sum(1) == 0).sum().item()
            hetero_children = total_children - semel_children - itero_children
            
            print(f"  Генетика размножения:")
            print(f"    [1,1] семельпарные: {semel_children} ({semel_children/total_children*100:.1f}%)")
            print(f"    [0,0] итеропарные: {itero_children} ({itero_children/total_children*100:.1f}%)")
            print(f"    Гетерозиготы: {hetero_children} ({hetero_children/total_children*100:.1f}%)")
    
    # Статистика по полу
    if c_sex.numel() > 0:
        all_children_sex_alleles = c_sex.reshape(-1, 2)
        # Пол = сумма аллелей: 0+0=0 (самка), 0+1=1 (самец)
        children_sex = all_children_sex_alleles.sum(1)
        females = (children_sex == 0).sum().item()
        males = (children_sex == 1).sum().item()
        
        if total_children > 0:
            print(f"  Пол потомков:")
            print(f"    Самки (XX): {females} ({females/total_children*100:.1f}%)")
            print(f"    Самцы (XY): {males} ({males/total_children*100:.1f}%)")
    
    print(f"  ✅ Размерности: c_sex={c_sex.shape}, c_a={c_a.shape}")
    
    return rep2, c_sex, c_a



def Replication(interaction):
    global X, Y, replication_matrix, CurrentTime
    global new_sex, new_a, Sex, status
    
    day_in_year = CurrentTime % 120
    
    # Только в дни 0-10
    if day_in_year > 10:
        return
    
    print(f"\n🔄 Replication(): день {CurrentTime}, день_в_году={day_in_year}")
    
    # Взрослые самки и самцы
    adult_females_mask = (Sex == 0) & (status == STATUS_ADULT)
    adult_males_mask = (Sex == 1) & (status == STATUS_ADULT)
    
    adult_female_indices = adult_females_mask.nonzero(as_tuple=True)[0]
    adult_male_indices = adult_males_mask.nonzero(as_tuple=True)[0]
    
    if len(adult_female_indices) == 0 or len(adult_male_indices) == 0:
        print(f"  ⚠️ Нет взрослых самок ({len(adult_female_indices)}) или самцов ({len(adult_male_indices)})")
        replication_matrix = None
        new_sex = th.empty((0, numofprogeny, 2), device=device)
        new_a = th.empty((0, numofprogeny, 2), device=device)
        return
    
    # Матрица взаимодействий между взрослыми
    rep_all = interaction[adult_females_mask][:, adult_male_indices]
    
    print(f"  Взрослых самок: {len(adult_female_indices)}, взрослых самцов: {len(adult_male_indices)}")
    print(f"  Матрица взаимодействий: {rep_all.shape}, возможных пар: {rep_all.sum().item()}")
    
    # ==================== ПРОСТАЯ КУМУЛЯТИВНАЯ ЛОГИКА ====================
    if day_in_year == 0:
        # ДЕНЬ 0: создаем матрицу с нуля
        print(f"  🎉 ДЕНЬ 0: создаем матрицу размножения")
        rep_new, ns, na = chrom_cal(rep_all, adult_female_indices, adult_male_indices)
        replication_matrix = rep_new
        new_sex = ns
        new_a = na
    else:
        # ДНИ 1-10: простая логика - всегда пересчитываем
        print(f"  📅 День {day_in_year}: пересчитываем матрицу размножения")
        rep_new, ns, na = chrom_cal(rep_all, adult_female_indices, adult_male_indices)
        replication_matrix = rep_new
        new_sex = ns
        new_a = na
    
    # Статистика
    if replication_matrix is not None:
        mated_count = (replication_matrix.sum(1) > 0).sum().item()
        total_females = len(adult_female_indices)
        print(f"  📊 Размножающихся самок: {mated_count}/{total_females} ({mated_count/total_females*100:.1f}%)")
        print(f"  📏 Размеры: матрица={replication_matrix.shape}, "
              f"new_sex={new_sex.shape}, new_a={new_a.shape}")
        



def DisperseJuvenile():
    """
    Рождение потомства на день 100.
    """
    global replication_matrix, new_sex, new_a, new_X, new_Y
    global X, Y, Sex, status
    
    print(f"\n🎉 DisperseJuvenile(): день {CurrentTime}")
    
    if replication_matrix is None:
        print("⚠️ Нет матрицы размножения")
        return
    
    print(f"📊 Матрица размножения: {replication_matrix.shape}")
    
    # Находим взрослых самок СЕЙЧАС (день 100)
    adult_females_mask = (Sex == 0) & (status == STATUS_ADULT)
    adult_female_indices = adult_females_mask.nonzero(as_tuple=True)[0]
    
    if len(adult_female_indices) == 0:
        print("⚠️ Нет взрослых самок")
        return
    
    # ВАЖНО: Матрица была создана в дни 0-10, когда самок могло быть больше
    # Нужно сопоставить индексы
    
    print(f"  Сейчас взрослых самок: {len(adult_female_indices)}")
    print(f"  Матрица была создана для: {replication_matrix.shape[0]} самок")
    
    # Если матрица одномерная (преобразована после дня 10)
    if len(replication_matrix.shape) == 1:
        # Матрица [самки] - булева маска, какие самки размножались
        print(f"  Одномерная матрица (маска размножавшихся самок)")
        
        # Проблема: матрица может быть больше, чем текущих самок
        if replication_matrix.shape[0] > len(adult_female_indices):
            print(f"  🔧 Обрезаем матрицу: {replication_matrix.shape[0]} → {len(adult_female_indices)}")
            replication_matrix = replication_matrix[:len(adult_female_indices)]
        
        # Самки, которые размножались (по старой матрице)
        reproducing_females_mask = replication_matrix > 0
        
    else:
        # Двумерная матрица [самки × самцы]
        print(f"  Двумерная матрица")
        
        if replication_matrix.shape[0] > len(adult_female_indices):
            print(f"  🔧 Обрезаем матрицу по строкам: {replication_matrix.shape[0]} → {len(adult_female_indices)}")
            replication_matrix = replication_matrix[:len(adult_female_indices), :]
        
        reproducing_females_mask = replication_matrix.sum(1) > 0
    
    # Проверяем, сколько самок действительно размножалось
    reproducing_indices = adult_female_indices[reproducing_females_mask[:len(adult_female_indices)]]
    
    if len(reproducing_indices) == 0:
        print("⚠️ Ни одна самка не размножалась (после обрезки матрицы)")
        replication_matrix = None
        new_sex = th.empty((0, numofprogeny, 2), device=device)
        new_a = th.empty((0, numofprogeny, 2), device=device)
        return
    
    print(f"✅ Найдено {len(reproducing_indices)} размножавшихся самок (после обрезки)")
    
    # Координаты матерей
    new_X = X[reproducing_indices].repeat_interleave(numofprogeny)
    new_Y = Y[reproducing_indices].repeat_interleave(numofprogeny)
    
    # Случайное смещение
    deltaX = th.randint(-50, 50, new_X.shape, device=device)
    deltaY = th.randint(-50, 50, new_Y.shape, device=device)
    new_X = new_X + deltaX
    new_Y = new_Y + deltaY
    
    # Коррекция границ
    new_X = th.where(new_X < 0, -new_X, new_X)
    new_X = th.where(new_X > MapXSize, 2 * MapXSize - new_X, new_X)
    new_Y = th.where(new_Y < 0, -new_Y, new_Y)
    new_Y = th.where(new_Y > MapYSize, 2 * MapYSize - new_Y, new_Y)
    
    # Генетика потомства
    if new_sex is not None and new_a is not None and new_sex.numel() > 0 and new_a.numel() > 0:
        print(f"  🔍 Размер new_sex: {new_sex.shape}, new_a: {new_a.shape}")
        
        # Исправляем размерность new_sex если нужно
        if len(new_sex.shape) == 4:
            print(f"  🔧 Исправляем размерность new_sex: {new_sex.shape} → ", end="")
            if new_sex.shape[3] == 2:
                new_sex = new_sex[:, :, :, 0]
                print(f"{new_sex.shape}")
            else:
                new_sex = new_sex.squeeze(3)
                print(f"{new_sex.shape}")
        
        # ВАЖНО: new_sex и new_a тоже создавались для большего числа самок
        # Обрезаем их до текущего количества самок
        if new_sex.shape[0] > len(adult_female_indices):
            print(f"  🔧 Обрезаем new_sex: {new_sex.shape[0]} → {len(adult_female_indices)}")
            new_sex = new_sex[:len(adult_female_indices)]
            new_a = new_a[:len(adult_female_indices)]
        
        # Фильтруем по маске размножавшихся самок
        mask_to_use = reproducing_females_mask[:new_sex.shape[0]]
        new_sex = new_sex[mask_to_use]
        new_a = new_a[mask_to_use]
        
        print(f"  ✅ После фильтрации: new_sex={new_sex.shape}, new_a={new_a.shape}")
        
        # Диагностика
        all_children = new_a.reshape(-1, 2)
        total_children = all_children.shape[0]
        
        if total_children > 0:
            semel_children = (all_children.sum(1) == 2).sum().item()
            itero_children = (all_children.sum(1) == 0).sum().item()
            hetero_children = total_children - semel_children - itero_children
            
            print(f"  🔍 Генетика потомства: {total_children} потомков")
            print(f"    [1,1] семельпарные: {semel_children} ({semel_children/total_children*100:.1f}%)")
            print(f"    [0,0] итеропарные: {itero_children} ({itero_children/total_children*100:.1f}%)")
            print(f"    Гетерозиготы: {hetero_children} ({hetero_children/total_children*100:.1f}%)")
    else:
        print("⚠️ Нет генетической информации")
        new_sex = th.empty((0, numofprogeny, 2), device=device)
        new_a = th.empty((0, numofprogeny, 2), device=device)
        return
    
    # Добавляем потомство
    old_pop_size = PopSize
    print(f"  🚀 Вызываем AddAnimal() с {len(new_X)} координатами")
    AddAnimal()
    
    expected_children = len(reproducing_indices) * numofprogeny
    print(f"✅ Ожидалось {expected_children} потомков")
    print(f"   Популяция была: {old_pop_size}, стала: {PopSize}")
    print(f"   Разница: {PopSize - old_pop_size} (должно быть {expected_children})")
    
    # Сброс
    replication_matrix = None
    new_sex = th.empty((0, numofprogeny, 2), device=device)
    new_a = th.empty((0, numofprogeny, 2), device=device)
    new_X = th.empty(0, device=device)
    new_Y = th.empty(0, device=device)





def TimeRunAndDeath():
    global InfectionStatus, AgeOfDisease, Age, Sex, dead, status, replication_matrix
    
    print(f"\n🔍 TimeRunAndDeath(): день {CurrentTime}")
    
    # 1. УВЕЛИЧЕНИЕ ВОЗРАСТА
    Age = Age + 1
    
    # 2. Поиск территории и обновление статусов
    _find_territory_for_juveniles()
    _update_statuses()
    
    # 3. ОБНОВЛЕНИЕ ИНФЕКЦИИ
    mask = (AgeOfDisease <= incubation) * InfectionStatus.eq(1)
    InfectionStatus[mask] = 2
    mask = (AgeOfDisease > incubation) * InfectionStatus.eq(2)
    InfectionStatus[mask] = 1
    AgeOfDisease = InfectionStatus.gt(0) + AgeOfDisease
    
    # 4. ВЫЧИСЛЯЕМ ДЕНЬ В ГОДУ
    day_in_year = CurrentTime % 120
    
    # 5. СМЕРТНОСТЬ
    death_mask = th.zeros(PopSize, dtype=th.bool, device=device)
    
    if CurrentTime > 0:  # Не в день 0
        # 5.1. Естественная смертность + смерть от болезни
        natural_death = Mortality + InfectionStatus.eq(1) * 0.1 / (
                    1 + th.exp(10 * ((latency / 2) + incubation - AgeOfDisease) / latency))
        death_mask |= (th.rand(PopSize, device=device) < natural_death)
        
        # 5.2. Смерть от старости
        death_mask |= (Age > 720)
        
        # 5.3. КРИТИЧЕСКОЕ: Смерть семельпарных самцов (день 10)
        if day_in_year == 10:
            print(f"💀 ДЕНЬ 10: проверка смерти семельпарных самцов")
            
            # Находим семельпарных самцов ([1,1])
            semel_males = (chrom_a.sum(1) == 2) & (Sex == 1)
            
            # Проверяем, участвовали ли они в размножении
            if replication_matrix is not None and semel_males.any():
                # Находим индексы самцов в текущей популяции
                male_indices = (Sex == 1).nonzero(as_tuple=True)[0]
                
                if len(male_indices) > 0:
                    # Берем только тех самцов, которые есть в матрице
                    males_in_matrix = min(len(male_indices), replication_matrix.shape[1])
                    
                    # Какие самцы участвовали в размножении
                    male_reproduced = replication_matrix.sum(0)[:males_in_matrix] > 0
                    
                    # Создаем маску для всей популяции
                    reproduced_full = th.zeros(PopSize, dtype=th.bool, device=device)
                    reproduced_full[male_indices[:males_in_matrix]] = male_reproduced
                    
                    # Умирают семельпарные самцы, которые размножались
                    dying_males = semel_males & reproduced_full
                    death_mask |= dying_males
                    
                    if dying_males.sum().item() > 0:
                        print(f"💀 Смерть семельпарных самцов: {dying_males.sum().item()} особей")
            
            # После дня 10 преобразуем матрицу в одномерную
            if replication_matrix is not None and len(replication_matrix.shape) == 2:
                replication_matrix = replication_matrix.sum(1) > 0
                replication_matrix = replication_matrix.float()
                print(f"🔄 Матрица преобразована в одномерную: {replication_matrix.shape}")
        
        # 5.4. Смерть ювенилов без территории
        juv_no_terr_mask = (status == STATUS_JUVENILE_NO_TERR)
        
        if juv_no_terr_mask.any():
            juv_fitness = Fitness[juv_no_terr_mask]
            fitness_factor = th.nn.functional.relu(80 - juv_fitness) / 80
            
            age_over_deadline = (Age[juv_no_terr_mask] > dispersalDeadline)
            death_prob = fitness_factor * age_over_deadline.float()
            rand_vals = th.rand(juv_no_terr_mask.sum(), device=device)
            actually_dying = rand_vals < death_prob
            
            if actually_dying.any():
                dying_full_mask = th.zeros(PopSize, dtype=th.bool, device=device)
                juv_indices = juv_no_terr_mask.nonzero(as_tuple=True)[0][actually_dying]
                dying_full_mask[juv_indices] = True
                death_mask |= dying_full_mask
        
        # 5.5. Смерть ювенилов без территории в последний день
        last_juv_day = (status == STATUS_JUVENILE_NO_TERR) & (Age == 219)
        death_mask |= last_juv_day
    
    # 6. УДАЛЕНИЕ УМЕРШИХ
    survival_mask = ~death_mask
    
    if death_mask.any():
        print(f"💀 День {CurrentTime}: смертей {death_mask.sum().item()}/{PopSize}")
        
        if len(survival_mask) == PopSize:
            DeleteAnimal(survival_mask)
        else:
            print(f"🚨 Размер маски не совпадает с PopSize")
    else:
        if CurrentTime % 30 == 0:
            print(f"✅ День {CurrentTime}: смертей нет")
    
    # 7. СБРОС СЧЕТЧИКА dead В НАЧАЛЕ ГОДА
    if day_in_year == 0:
        dead = 0
    
    if PopSize == 0:
        print(f"🛑 День {CurrentTime}: ПОПУЛЯЦИЯ ВЫМЕРЛА")


def dispersal_cal(x1, x2, y1, y2):
    """
    Евклидово расстояние между точками (x1, y1) и (x2, y2) без тороидальности.
    """
    dx = x1 - x2
    dy = y1 - y2
    return th.sqrt(dx ** 2 + dy ** 2)




def collect_statistics(params, run_num):
    """Собирает ВСЕ статистики по популяции с учетом НОВЫХ статусов."""
    if PopSize == 0:
        return {
            'time': CurrentTime,
            'total_population': 0,
            'infected': 0,
            'iteroparous': 0,
            'semelparous': 0,
            'infection_rate': 0,
            'adults': 0,
            'juveniles': 0,
            'children': 0,
            'juveniles_no_terr': 0,
            'juveniles_terr': 0,
            'residents': 0,
            'males': 0,
            'females': 0,
            'males_over_2_years': 0,
            'females_over_2_years': 0,
            'infectivity1': params['I1'],
            'infectivity2': params['I2'],
            'run_id': run_num
        }
    
    total = int(PopSize)
    infected = int((InfectionStatus > 0).sum().item())
    
    # Генетика
    semel_mask = chrom_a.sum(1) == 2
    semel_count = int(semel_mask.sum().item())
    itero_count = total - semel_count
    
    # НОВАЯ СТАТИСТИКА ПО СТАТУСАМ
    children_count = int((status == STATUS_CHILD).sum().item())
    juv_no_terr_count = int((status == STATUS_JUVENILE_NO_TERR).sum().item())
    juv_terr_count = int((status == STATUS_JUVENILE_TERR).sum().item())
    adult_count = int((status == STATUS_ADULT).sum().item())
    
    # Резиденты (те, кто имеет территорию)
    residents_count = juv_terr_count + adult_count
    
    # Пол
    males = int((Sex == 1).sum().item())
    females = int((Sex == 0).sum().item())
    
    # Возрастная статистика (дополнительно)
    males_over_2_years = int(((Sex == 1) & (Age >= 240)).sum().item())
    females_over_2_years = int(((Sex == 0) & (Age >= 240)).sum().item())
    
    infection_rate = (infected / total * 100) if total > 0 else 0
    
    return {
        'time': CurrentTime,
        'total_population': total,
        'infected': infected,
        'iteroparous': itero_count,
        'semelparous': semel_count,
        'infection_rate': float(infection_rate),
        'adults': adult_count,
        'juveniles': juv_no_terr_count + juv_terr_count,  # всего ювенилов
        'children': children_count,
        'juveniles_no_terr': juv_no_terr_count,  # НОВОЕ
        'juveniles_terr': juv_terr_count,        # НОВОЕ
        'residents': residents_count,            # НОВОЕ
        'males': males,
        'females': females,
        'males_over_2_years': males_over_2_years,
        'females_over_2_years': females_over_2_years,
        'infectivity1': params['I1'],
        'infectivity2': params['I2'],
        'run_id': run_num
    }

def save_statistics_to_file(stats, filename):
    """Сохраняет статистику в CSV файл"""
    if filename is None:
        return
    
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writerow(stats)
    except Exception as e:
        print(f"⚠️ Ошибка записи в файл {filename}: {e}")


def print_statistics_to_terminal(stats, run_num):
    """Вывод статистики в терминал"""
    print(f"Run {run_num:03d} | Время: {stats['time']:6d} | "
          f"Поп: {stats['total_population']:6d} | "
          f"Зараж: {stats['infected']:4d}({stats['infection_rate']:5.1f}%) | "
          f"Itero: {stats['iteroparous']:6d} | Semel: {stats['semelparous']:6d}")


def debug_territories():
    """
    Функция для отладки системы территорий.
    Вызывается периодически для проверки целостности.
    """
    if PopSize == 0:
        return
    
    print(f"\n🔍 ДЕБАГ ТЕРРИТОРИЙ (день {CurrentTime}):")
    
    # 1. Статистика по статусам
    children = (status == STATUS_CHILD).sum().item()
    juv_no_terr = (status == STATUS_JUVENILE_NO_TERR).sum().item()
    juv_terr = (status == STATUS_JUVENILE_TERR).sum().item()
    adults = (status == STATUS_ADULT).sum().item()
    
    print(f"  Статусы: дети={children}, ювенилы_без_территории={juv_no_terr}, "
          f"ювенилы_с_территорией={juv_terr}, взрослые={adults}")
    
    # 2. Проверка территорий
    residents = (status == STATUS_ADULT) | (status == STATUS_JUVENILE_TERR)
    non_residents = ~residents
    
    if residents.any():
        # У резидентов должна быть территория (!= -1)
        residents_without_territory = residents & ((territory_center_x == -1) | (territory_center_y == -1))
        residents_with_territory = residents & ((territory_center_x != -1) & (territory_center_y != -1))
        
        print(f"  Резиденты: {residents.sum().item()} всего")
        print(f"    С территорией: {residents_with_territory.sum().item()}")
        print(f"    Без территории: {residents_without_territory.sum().item()} (ОШИБКА!)")
        
        if residents_with_territory.any():
            # Среднее расстояние до центра
            dist_x = X[residents_with_territory] - territory_center_x[residents_with_territory]
            dist_y = Y[residents_with_territory] - territory_center_y[residents_with_territory]
            avg_dist = th.sqrt(dist_x**2 + dist_y**2).mean().item()
            max_dist = th.sqrt(dist_x**2 + dist_y**2).max().item()
            
            inside = th.sqrt(dist_x**2 + dist_y**2) <= Range
            inside_pct = inside.sum().item() / residents_with_territory.sum().item() * 100
            
            print(f"    Среднее расстояние до центра: {avg_dist:.1f}")
            print(f"    Максимальное расстояние: {max_dist:.1f} (радиус={Range})")
            print(f"    Внутри территории: {inside_pct:.1f}%")
    
    if non_residents.any():
        # У не-резидентов не должно быть территории (== -1)
        non_residents_with_territory = non_residents & ((territory_center_x != -1) | (territory_center_y != -1))
        
        print(f"  Не-резиденты (дети+ювенилы без территории): {non_residents.sum().item()}")
        print(f"    Без территории (корректно): {(non_residents.sum().item() - non_residents_with_territory.sum().item())}")
        print(f"    С территорией (ОШИБКА): {non_residents_with_territory.sum().item()}")
    
    # 3. Примеры нескольких особей
    print(f"\n  Примеры особей (первые 3 резидента):")
    resident_indices = residents.nonzero(as_tuple=True)[0]
    for i in range(min(3, len(resident_indices))):
        idx = resident_indices[i]
        stat = "ADULT" if status[idx] == STATUS_ADULT else "JUV_TERR"
        has_terr = "ДА" if territory_center_x[idx] != -1 else "НЕТ"
        x_center = territory_center_x[idx].item()
        y_center = territory_center_y[idx].item()
        x_current = X[idx].item()
        y_current = Y[idx].item()
        dist = math.sqrt((x_current - x_center)**2 + (y_current - y_center)**2)
        
        print(f"    #{i}: ID={animal_ids[idx].item()}, статус={stat}, территория={has_terr}")
        print(f"       Центр=({x_center:.0f}, {y_center:.0f}), "
              f"Текущ=({x_current:.0f}, {y_current:.0f}), "
              f"Расстояние={dist:.0f}/{Range}")




def on_running(ix, params, run_num, stats_file):
    """
    ПРАВИЛЬНЫЙ ПОРЯДОК по вашему описанию:
    1. Движение и инфекция (с Replication в дни 0-10)
    2. Дисперсия (день 100)
    3. Время и смерть
    """
    global CurrentTime, numofinfection, chrom_a, Infectivity1, Infectivity2
    
    CurrentTime = ix
    
    # 0. УСТАНОВКА ПАРАМЕТРОВ
    Infectivity1 = params['I1']
    Infectivity2 = params['I2']
    
    # 1. ДЕНЬ В ГОДУ
    day_in_year = CurrentTime % 120
    
    # 2. ПЕРВИЧНОЕ ЗАРАЖЕНИЕ (только в день 0)
    if CurrentTime == 0 and numofinfection == 0:
        residents = ((status == STATUS_ADULT) | (status == STATUS_JUVENILE_TERR))
        if residents.any():
            SeedPathogen(10, 1)
    
    # 3. ПОВТОРНОЕ ЗАРАЖЕНИЕ
    elif ix > 100 and numofinfection == 0:
        residents = ((status == STATUS_ADULT) | (status == STATUS_JUVENILE_TERR))
        if residents.any():
            SeedPathogen(10, 1)
    
    # 4. ДВИЖЕНИЕ И ИНФЕКЦИЯ
    if day_in_year <= 10:
        # Дни 0-10: с размножением
        MovementAndInfection(replication=True)
    else:
        # Дни 11-119: без размножения
        MovementAndInfection(replication=False)
    
    # 5. ДИСПЕРСИЯ (день 100)
    if day_in_year == timeOfDisposal:
        DisperseJuvenile()
    
    # 6. ВРЕМЯ И СМЕРТЬ
    TimeRunAndDeath()
    
    # 7. СТАТИСТИКА
    if CurrentTime % 10 == 0:
        stats = collect_statistics(params, run_num)
        save_statistics_to_file(stats, stats_file)
        
        if CurrentTime % 1000 == 0:
            print_statistics_to_terminal(stats, run_num)
    
    return PopSize






def setup_directories():
    """Создает все необходимые директории"""
    directories = [RESULTS_DIR, STATS_DIR, GIFS_DIR, SNAPSHOTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Создана/проверена директория: {directory}/")
    return directories


def setup_stats_file(param_combination_id, run_num, params):
    """Создает файл для записи статистики с НОВЫМИ полями."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STATS_DIR}/run_{run_num:03d}_stats_I1_{params['I1']:.2f}_I2_{params['I2']:.4f}.csv"
        
        # ОБНОВЛЕННЫЕ ЗАГОЛОВКИ
        headers = [
            'time', 'total_population', 'infected', 'iteroparous', 'semelparous',
            'infection_rate', 'adults', 'juveniles', 'children',
            'juveniles_no_terr', 'juveniles_terr', 'residents',  # НОВЫЕ ПОЛЯ
            'males', 'females', 'males_over_2_years', 'females_over_2_years',
            'infectivity1', 'infectivity2', 'run_id'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        print(f"📁 Файл статистики создан: {filename}")
        return filename
    except Exception as e:
        print(f"⚠️ Ошибка создания файла статистики: {e}")
        return None
    


def cleanup_old_snapshots():
    """
    Очищает старые папки со снимками перед началом новых симуляций
    """
    print("\n🧹 Очистка старых папок со снимками...")
    
    cleaned_count = 0
    
    # Очищаем SNAPSHOTS_DIR
    if os.path.exists(SNAPSHOTS_DIR):
        for item in os.listdir(SNAPSHOTS_DIR):
            item_path = os.path.join(SNAPSHOTS_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Удалена папка: {item_path}")
                    cleaned_count += 1
                elif item.endswith('.png'):
                    os.remove(item_path)
                    print(f"  Удален файл: {item_path}")
                    cleaned_count += 1
            except Exception as e:
                print(f"  ⚠️ Не удалось удалить {item_path}: {e}")
    
    # Также очищаем корневую директорию на всякий случай
    patterns = ["snapshots_run_*", "month_*.png"]
    for pattern in patterns:
        for item in glob.glob(pattern):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                cleaned_count += 1
            except:
                pass
    
    print(f"✅ Очищено {cleaned_count} элементов")
    return cleaned_count


def draw_func(time_step, iteroparous, semelparous, tpop, folder="snapshots", run_num=0):
    """
    Рисует и сохраняет три графика с ФИКСИРОВАННЫМИ размерами.
    Кадр сохраняется каждый шаг.
    """
    # Создаем папку если не существует
    os.makedirs(folder, exist_ok=True)

    if PopSize <= 0:
        return

    # Фиксируем размер фигуры для всех кадров
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False,
                           tight_layout=True, gridspec_kw={'height_ratios': [1.5, 1, 1]})
    fig.set_size_inches(12, 7)
    
    # ФИКСИРУЕМ ГРАНИЦЫ (важно для одинаковых размеров кадров!)
    ax[0].set_xlim(0, MapXSize)
    ax[0].set_ylim(0, MapYSize)
    ax[0].set_aspect('equal')
    
    # ==================== ОБНОВЛЕННАЯ ЦВЕТОВАЯ СХЕМА ====================
    # 1. Основные цвета как в вашем примере:
    #    - Итеропарные взрослые: 'y' (желтый)
    #    - Семельпарные взрослые: 'dodgerblue' (голубой)
    #    - Итеропарные ювенилы: 'khaki' (светло-оранжевый)
    #    - Семельпарные ювенилы: 'lightblue' (светло-голубой)
    
    # 2. ДЕТИ (0-159 дней) - новые цвета:
    #    - Итеропарные дети: 'limegreen' (ярко-зеленый)
    #    - Семельпарные дети: 'aquamarine' (бирюзовый)
    
    # 3. Зараженные: красная обводка поверх основного цвета
    
    # Определяем статусы
    is_child = Age < 160  # 0-159 дней
    is_juvenile = (Age >= 160) & (Age < 220)  # 160-219 дней
    is_adult = Age >= 220  # 220+ дней
    
    
    is_infected = InfectionStatus > 0
    is_semelparous = chrom_a.sum(1) == 2  # Только чистые семельпарные [1,1]



    # Инициализируем массивы
    colorr = [None] * PopSize
    edgecolor = [None] * PopSize
    markersize = [None] * PopSize
    
    for i in range(PopSize):
        # 1. ОСНОВНОЙ ЦВЕТ ПО ВОЗРАСТУ И ГЕНОТИПУ
        if is_child[i]:
            # ДЕТИ
            if is_semelparous[i]:
                colorr[i] = 'aquamarine'  # Семельпарные дети - бирюзовый
            else:
                colorr[i] = 'limegreen'   # Итеропарные дети - ярко-зеленый
                
        elif is_adult[i]:
            # ВЗРОСЛЫЕ
            if is_semelparous[i]:
                colorr[i] = 'dodgerblue'  # Семельпарные взрослые - голубой
            else:
                colorr[i] = 'y'           # Итеропарные взрослые - желтый
                
        else:
            # ЮВЕНИЛЫ (160-219 дней)
            if is_semelparous[i]:
                colorr[i] = 'lightblue'   # Семельпарные ювенилы - светло-голубой
            else:
                colorr[i] = 'khaki'       # Итеропарные ювенилы - светло-оранжевый
        
        # 2. ОБВОДКА - по умолчанию нет
        edgecolor[i] = "none"
        
        # 3. РАЗМЕР ТОЧКИ
        if is_child[i]:
            markersize[i] = 30   # Дети - маленькие
        elif is_adult[i]:
            markersize[i] = 50   # Взрослые - большие
        else:
            markersize[i] = 40   # Ювенилы - средние
        
        # 4. ЗАРАЖЕННЫЕ ОСОБИ - КРАСНАЯ ОБВОДКА поверх основного цвета
        if is_infected[i]:
            edgecolor[i] = 'red'  # Красная обводка
            markersize[i] += 10   # Зараженные немного больше
    
    # ==================== 1. ПРОСТРАНСТВЕННОЕ РАСПРЕДЕЛЕНИЕ ====================
    scatter = ax[0].scatter(X.cpu().numpy(), Y.cpu().numpy(), 
                           c=colorr, edgecolors=edgecolor, 
                           s=markersize, alpha=0.8, linewidth=1.5)
    ax[0].set_title(f"Spatial distribution - Day {time_step} (Run {run_num})")
    ax[0].set_xlabel("X coordinate")
    ax[0].set_ylabel("Y coordinate")
    
    # Легенда для пространственного графика
    legend_elements = [
        # Дети
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                  markersize=8, label='Child Iteroparous'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='aquamarine', 
                  markersize=8, label='Child Semelparous'),
        # Ювенилы
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='khaki', 
                  markersize=8, label='Juvenile Iteroparous'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=8, label='Juvenile Semelparous'),
        # Взрослые
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='y', 
                  markersize=8, label='Adult Iteroparous'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', 
                  markersize=8, label='Adult Semelparous'),
        # Зараженные (красная обводка)
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                  markersize=8, label='Infected', markeredgecolor='red', 
                  markeredgewidth=1.5)
    ]
    ax[0].legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=2)
    
    # ==================== 2. ГИСТОГРАММА ВОЗРАСТА ====================
    # ФИКСИРУЕМ ГРАНИЦЫ гистограммы
    if PopSize > 0:
        # Возраст в годах
        age_years = (Age // 120).cpu().numpy()
        max_age = max(age_years.max(), 1)  # минимум 1 год
        
        # Разделяем по генотипам
        itero_mask = chrom_a.sum(1).eq(0)
        semel_mask = chrom_a.sum(1).gt(0)
        
        # Гистограмма для итеропарных
        if itero_mask.any():
            itero_ages = age_years[itero_mask.cpu().numpy()]
            ax[1].hist(itero_ages, color='y', alpha=0.7, label='Iteroparous', 
                      bins=min(20, int(max_age) + 1), range=(0, max_age))
        
        # Гистограмма для семельпарных
        if semel_mask.any():
            semel_ages = age_years[semel_mask.cpu().numpy()]
            ax[1].hist(semel_ages, color='dodgerblue', alpha=0.7, label='Semelparous',
                      bins=min(20, int(max_age) + 1), range=(0, max_age))
        
        ax[1].set_title(f"Age distribution (Run {run_num})")
        ax[1].set_xlabel("Age (years)")
        ax[1].set_ylabel("Count")
        ax[1].set_xlim(0, max_age)
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "No population data", 
                  ha='center', va='center', transform=ax[1].transAxes)
        ax[1].set_xlim(0, 10)  # Фиксируем пустую гистограмму
    
    # ==================== 3. ДИНАМИКА ЧИСЛЕННОСТИ ====================
    # ФИКСИРУЕМ ГРАНИЦЫ графика динамики
    if len(iteroparous) > 1:
        time_axis = np.arange(len(iteroparous))
        
        # Находим максимальное значение для оси Y
        max_pop = max(max(iteroparous), max(semelparous), max(tpop), 1)
        
        ax[2].plot(time_axis, semelparous, c="dodgerblue", alpha=0.8, linewidth=2.5, label="Semelparous")
        ax[2].plot(time_axis, iteroparous, c="y", alpha=0.8, linewidth=2.5, label="Iteroparous")
        ax[2].plot(time_axis, tpop, c="black", linewidth=2, label="Total Pop")
        
        # Заливка под кривыми
        ax[2].fill_between(time_axis, 0, semelparous, color="dodgerblue", alpha=0.15)
        ax[2].fill_between(time_axis, 0, iteroparous, color="y", alpha=0.15)
        
        ax[2].set_title(f"Population dynamics (Run {run_num})")
        ax[2].set_xlabel("Time (days)")
        ax[2].set_ylabel("Population size")
        ax[2].set_xlim(0, len(iteroparous)-1)
        ax[2].set_ylim(0, max_pop * 1.1)  # +10% от максимума
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
    else:
        ax[2].text(0.5, 0.5, "Insufficient data for dynamics",
                   ha='center', va='center', transform=ax[2].transAxes)
        ax[2].set_xlim(0, 100)
        ax[2].set_ylim(0, 100)
    
    # ==================== 4. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ====================
    # Рассчитываем статистику
    semel_count = is_semelparous.sum().item()
    itero_count = PopSize - semel_count
    infected_count = is_infected.sum().item()
    
    child_count = is_child.sum().item()
    juvenile_count = is_juvenile.sum().item()
    adult_count = is_adult.sum().item()
    
    # Процент семельпарии
    semel_ratio = semel_count / PopSize if PopSize > 0 else 0
    
    info_text = f"Day: {time_step}\n"
    info_text += f"Total: {PopSize}\n"
    info_text += f"Children: {child_count}\n"
    info_text += f"Juveniles: {juvenile_count}\n"
    info_text += f"Adults: {adult_count}\n"
    info_text += f"Infected: {infected_count}\n"
    info_text += f"Iteroparous: {itero_count}\n"
    info_text += f"Semelparous: {semel_count}"
    
    fig.text(0.02, 0.02, info_text, fontsize=7, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Заголовок графика
    year = time_step // 120
    day_in_year = time_step % 120
    plt.suptitle(f"Year {year}, Day {day_in_year} | Semelparous: {semel_ratio:.3f}", 
                 fontsize=10, y=0.98)
    
    # ==================== 5. СОХРАНЕНИЕ КАДРА ====================
    # Убедимся, что все элементы на месте
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Оставляем место для суперзаголовка
    
    filename = os.path.join(folder, f"day_{time_step:06d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Явно закрываем фигуру
    
    print(f"📸 Кадр сохранен: {filename}")
    
    # Очистка памяти
    import gc
    gc.collect()





def create_gif_from_snapshots(snapshot_folder, gif_name, cleanup=True):
    """
    Создает GIF из папки со снимками и очищает папку
    """
    try:
        # Используем imageio.v2 для совместимости
        import imageio.v2 as imageio
        
        # Получаем все PNG файлы
        png_files = sorted([f for f in os.listdir(snapshot_folder) 
                           if f.endswith(".png")])
        
        if len(png_files) < 2:
            print(f"⚠️ Недостаточно кадров для GIF: {len(png_files)} файлов")
            return False
        
        files = [os.path.join(snapshot_folder, f) for f in png_files]
        
        # Создаем GIF
        images = []
        for f in files:
            try:
                images.append(imageio.imread(f))
            except Exception as e:
                print(f"⚠️ Ошибка чтения файла {f}: {e}")
        
        if len(images) > 0:
            # Проверяем размеры всех изображений
            first_shape = images[0].shape
            for i, img in enumerate(images):
                if img.shape != first_shape:
                    print(f"⚠️ Размер изображения {i} не совпадает: {img.shape} != {first_shape}")
                    # Обрезаем/изменяем размер до первого
                    import cv2
                    img_resized = cv2.resize(img, (first_shape[1], first_shape[0]))
                    images[i] = img_resized
            
            imageio.mimsave(gif_name, images, duration=0.2)
            print(f"✅ GIF создан: {gif_name} ({len(images)} кадров)")
            
            # Очистка папки со снимками
            if cleanup:
                try:
                    shutil.rmtree(snapshot_folder)
                    print(f"🧹 Папка снимков очищена: {snapshot_folder}")
                except Exception as e:
                    print(f"⚠️ Ошибка при очистке папки: {e}")
            
            return True
        else:
            print(f"⚠️ Не удалось загрузить изображения для GIF")
            return False
            
    except Exception as e:
        print(f"⚠️ Ошибка создания GIF: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def run_single_simulation(params, run_num, max_timepoints=42000, create_gif=True):
    """Запуск одной симуляции с НОВОЙ логикой статусов."""
    global Infectivity1, Infectivity2, CurrentTime
    
    # 1. ПОЛНЫЙ СБРОС
    reset_simulation_state()
    
    print(f"\n{'='*80}")
    print(f"🚀 ЗАПУСК СИМУЛЯЦИИ #{run_num} с НОВОЙ СИСТЕМОЙ СТАТУСОВ")
    print(f"Параметры: Infectivity1={params['I1']:.3f}, Infectivity2={params['I2']:.5f}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 2. УСТАНОВКА ПАРАМЕТРОВ
    Infectivity1 = params['I1']
    Infectivity2 = params['I2']
    
    # 3. ИНИЦИАЛИЗАЦИЯ НОВОЙ СИМУЛЯЦИИ
    Start(initial_pop_size=2000)
    
    # 4. ФАЙЛ СТАТИСТИКИ
    stats_file_path = setup_stats_file(0, run_num, params)
    
    # 5. ПАПКА ДЛЯ СНИМКОВ
    if create_gif:
        snapshot_folder = f"{SNAPSHOTS_DIR}/snapshots_run_{run_num:03d}"
        os.makedirs(snapshot_folder, exist_ok=True)
        print(f"📁 Папка для снимков: {snapshot_folder}")
    
    stop_time, stop_reason = None, None
    
    # Данные для графиков
    iteroparous_list, semelparous_list, tpop_list = [], [], []
    
    # 6. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for i in range(max_timepoints):
        CurrentTime = i
        
        # Основной цикл симуляции
        on_running(i, params, run_num, stats_file_path)
        
        # Проверка вымирания
        total_pop = int(PopSize) if PopSize > 0 else 0
        
        # Условия ранней остановки
        if total_pop == 0:
            stop_time, stop_reason = i, "вымирание"
            break
        
        # Расчет генотипов для динамики
        semel_mask = chrom_a.sum(1) == 2
        semel_count = semel_mask.sum().item()
        itero_count = total_pop - semel_count
        
        # Сохраняем данные для графиков
        iteroparous_list.append(itero_count)
        semelparous_list.append(semel_count)
        tpop_list.append(total_pop)
        
        # Создаем снимок для GIF
        if create_gif and i % 3 == 0:
            try:
                draw_func(i, iteroparous_list, semelparous_list, tpop_list, snapshot_folder, run_num)
            except Exception as e:
                print(f"⚠️ Ошибка при создании кадра {i}: {e}")
        
        # Проверка достижения фиксации
        if total_pop > 0:
            semel_frac = semel_count / total_pop
            itero_frac = itero_count / total_pop
    
    # НЕ проверять фиксацию в день 0!
        if i > 0:  # ← ДОБАВИТЬ ЭТУ ПРОВЕРКУ
            if semel_frac >= 0.999:
                stop_time, stop_reason = i, "100% семельпария"
                break
                
            if itero_frac >= 0.999:
                stop_time, stop_reason = i, "100% итеропария"
                break
                
            if semel_count == 0:
                stop_time, stop_reason = i, "исчезновение семельпарии"
                break
        
        # Периодический вывод прогресса
        if i % 5000 == 0 and i > 0:
            elapsed_so_far = time.time() - start_time
            infected = (InfectionStatus > 0).sum().item()
            infection_rate = (infected / total_pop * 100) if total_pop > 0 else 0
            
            # НОВАЯ СТАТИСТИКА ПО СТАТУСАМ
            children = (status == STATUS_CHILD).sum().item()
            juv_no_terr = (status == STATUS_JUVENILE_NO_TERR).sum().item()
            juv_terr = (status == STATUS_JUVENILE_TERR).sum().item()
            adults = (status == STATUS_ADULT).sum().item()
            
            print(f"Run {run_num}: шаг {i}/{max_timepoints} ({i/max_timepoints*100:.1f}%)")
            print(f"  Время: {elapsed_so_far:.0f}с, Поп: {total_pop}, Зараж: {infected}({infection_rate:.1f}%)")
            print(f"  Генотипы: Itero={itero_count}({itero_frac*100:.1f}%), Semel={semel_count}({semel_frac*100:.1f}%)")
            print(f"  Статусы: дети={children}, ювенилы_без_территории={juv_no_terr}, "
                  f"ювенилы_с_территорией={juv_terr}, взрослые={adults}")
    
    # 7. ЗАВЕРШЕНИЕ СИМУЛЯЦИИ
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ СИМУЛЯЦИЯ #{run_num} ЗАВЕРШЕНА")
    print(f"⏱ Время: {elapsed:.1f} сек")
    print(f"📈 Финальная популяция: {PopSize}")
    
    if stop_reason:
        print(f"🛑 Причина остановки: {stop_reason} на шаге {stop_time}")
    else:
        print(f"🏁 Симуляция завершена полностью ({max_timepoints} шагов)")
    
    # 8. ФИНАЛЬНАЯ СТАТИСТИКА
    if PopSize > 0:
        children = (status == STATUS_CHILD).sum().item()
        juv_no_terr = (status == STATUS_JUVENILE_NO_TERR).sum().item()
        juv_terr = (status == STATUS_JUVENILE_TERR).sum().item()
        adults = (status == STATUS_ADULT).sum().item()
        
        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА ПО СТАТУСАМ:")
        print(f"  Дети: {children} ({children/PopSize*100:.1f}%)")
        print(f"  Ювенилы без территории: {juv_no_terr} ({juv_no_terr/PopSize*100:.1f}%)")
        print(f"  Ювенилы с территорией: {juv_terr} ({juv_terr/PopSize*100:.1f}%)")
        print(f"  Взрослые: {adults} ({adults/PopSize*100:.1f}%)")
        
        # Статистика по поиску территории
        if juv_no_terr > 0:
            juv_fitness = Fitness[status == STATUS_JUVENILE_NO_TERR]
            can_get_territory = (juv_fitness > 95).sum().item()
            print(f"  Ювенилы без территории: {can_get_territory}/{juv_no_terr} "
                  f"({can_get_territory/juv_no_terr*100:.1f}%) могут получить территорию")
    
    # 9. СОЗДАНИЕ GIF
    if create_gif:
        gif_name = f"simulation_run_{run_num:03d}_I1_{params['I1']:.2f}_I2_{params['I2']:.4f}.gif"
        gif_path = os.path.join(GIFS_DIR, gif_name)
        
        print(f"\n🎬 Создание GIF для симуляции #{run_num}...")
        
        if os.path.exists(snapshot_folder) and len(os.listdir(snapshot_folder)) > 0:
            success = create_gif_from_snapshots(snapshot_folder, gif_path, cleanup=True)
            if success:
                print(f"✅ GIF создан: {gif_path}")
            else:
                print(f"⚠️ Не удалось создать GIF")
        else:
            print(f"⚠️ Нет снимков для GIF")
    
    # 10. СВОДКА РЕЗУЛЬТАТОВ
    final_semel_count = int((chrom_a.sum(1) == 2).sum().item())
    final_itero_count = int(PopSize - final_semel_count) if PopSize > 0 else 0
    
    summary = {
        'run_id': run_num,
        'infectivity1': float(params['I1']),
        'infectivity2': float(params['I2']),
        'final_population': int(PopSize),
        'final_semel': final_semel_count,
        'final_itero': final_itero_count,
        'final_children': int((status == STATUS_CHILD).sum().item()) if PopSize > 0 else 0,
        'final_juv_no_terr': int((status == STATUS_JUVENILE_NO_TERR).sum().item()) if PopSize > 0 else 0,
        'final_juv_terr': int((status == STATUS_JUVENILE_TERR).sum().item()) if PopSize > 0 else 0,
        'final_adults': int((status == STATUS_ADULT).sum().item()) if PopSize > 0 else 0,
        'stop_time': stop_time if stop_reason else max_timepoints,
        'stop_reason': stop_reason if stop_reason else "полное время",
        'execution_time': float(elapsed),
    }
    
    if PopSize > 0:
        final_semel_pct = float(final_semel_count / PopSize * 100)
        final_itero_pct = float(final_itero_count / PopSize * 100)
        summary['final_semel_pct'] = final_semel_pct
        summary['final_itero_pct'] = final_itero_pct
    
    print(f"\n📊 ИТОГОВАЯ СВОДКА:")
    print(f"  Параметры: I1={params['I1']:.3f}, I2={params['I2']:.5f}")
    print(f"  Финальная популяция: {summary['final_population']}")
    print(f"  Семельпарные: {summary['final_semel']} ({summary.get('final_semel_pct', 0):.1f}%)")
    print(f"  Итеропарные: {summary['final_itero']} ({summary.get('final_itero_pct', 0):.1f}%)")
    print(f"  Причина остановки: {summary['stop_reason']}")
    
    return summary




def run_monte_carlo_simulation(num_runs=MONTE_CARLO_RUNS, create_gifs=True):
    """Основная функция для запуска Monte Carlo симуляций с НОВОЙ системой статусов."""
    print(f"\n{'#'*80}")
    print(f"🎲 ЗАПУСК MONTE CARLO СИМУЛЯЦИЙ С НОВОЙ СИСТЕМОЙ СТАТУСОВ")
    print(f"Количество симуляций: {num_runs}")
    print(f"Статусы: дети, ювенилы_без_территории, ювенилы_с_территорией, взрослые")
    print(f"{'#'*80}")
    
    # Создаем все директории
    setup_directories()
    
    # Очищаем старые снимки перед началом
    if create_gifs:
        cleanup_old_snapshots()
    
    all_summaries = []
    
    for run_num in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"🏃 Запуск симуляции {run_num}/{num_runs} с НОВОЙ СИСТЕМОЙ СТАТУСОВ")
        print(f"{'='*80}")
        
        # Генерация случайных параметров
        params = generate_random_parameters()
        print(f"📊 Параметры для симуляции #{run_num}:")
        print(f"  I1 (Infectivity1): {params['I1']:.4f}")
        print(f"  I2 (Infectivity2): {params['I2']:.6f}")
        
        # Запуск симуляции
        summary = run_single_simulation(params, run_num, create_gif=create_gifs)
        all_summaries.append(summary)
        
        # Сохраняем промежуточные результаты
        df_summary = pd.DataFrame(all_summaries)
        
        # Убедимся, что все значения преобразованы в простые типы
        for col in df_summary.columns:
            if df_summary[col].dtype == object:
                try:
                    df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
                except:
                    pass
        
        interim_filename = f"{RESULTS_DIR}/monte_carlo_summary_interim_run_{run_num}.csv"
        df_summary.to_csv(interim_filename, index=False)
        print(f"📄 Промежуточный отчет сохранен: {interim_filename}")
        
        # Вывод промежуточной статистики по статусам
        if len(all_summaries) > 0:
            print(f"\n📈 ПРОМЕЖУТОЧНАЯ СТАТИСТИКА ПО {len(all_summaries)} СИМУЛЯЦИЯМ:")
            
            # Средние по всем симуляциям
            avg_pop = df_summary['final_population'].mean()
            avg_semel_pct = df_summary['final_semel_pct'].mean() if 'final_semel_pct' in df_summary.columns else 0
            avg_itero_pct = df_summary['final_itero_pct'].mean() if 'final_itero_pct' in df_summary.columns else 0
            
            # Статистика по статусам (если есть)
            status_cols = ['final_children', 'final_juv_no_terr', 'final_juv_terr', 'final_adults']
            status_cols_present = [col for col in status_cols if col in df_summary.columns]
            
            if status_cols_present:
                for col in status_cols_present:
                    avg = df_summary[col].mean()
                    print(f"  {col}: {avg:.1f} особей в среднем")
            
            print(f"  Средняя популяция: {avg_pop:.0f}")
            print(f"  Средний % семельпарии: {avg_semel_pct:.1f}%")
            print(f"  Средний % итеропарии: {avg_itero_pct:.1f}%")
    
    # Создаем финальный отчет
    print(f"\n{'#'*80}")
    print(f"📊 ФИНАЛЬНЫЙ ОТЧЕТ MONTE CARLO ({len(all_summaries)} симуляций)")
    print(f"{'#'*80}")
    
    final_df = pd.DataFrame(all_summaries)
    
    # Убедимся, что все числовые колонки правильно обработаны
    numeric_cols = ['infectivity1', 'infectivity2', 'final_population', 
                    'final_semel', 'final_itero', 'stop_time', 'execution_time', 
                    'final_semel_pct', 'final_itero_pct']
    
    # Добавляем колонки статусов если они есть
    status_cols = ['final_children', 'final_juv_no_terr', 'final_juv_terr', 'final_adults']
    for col in status_cols:
        if col in final_df.columns:
            numeric_cols.append(col)
    
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # Сохраняем финальный отчет
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{RESULTS_DIR}/final_monte_carlo_summary_{timestamp}.csv"
    final_df.to_csv(final_filename, index=False)
    
    # Также сохраняем упрощенную версию
    simple_filename = f"{RESULTS_DIR}/final_monte_carlo_summary_latest.csv"
    final_df.to_csv(simple_filename, index=False)
    
    print(f"\n📁 РЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
    print(f"  📂 {RESULTS_DIR}/ - финальные отчеты")
    print(f"     ✓ {final_filename}")
    print(f"     ✓ {simple_filename}")
    
    if os.path.exists(STATS_DIR):
        stats_files = len([f for f in os.listdir(STATS_DIR) if f.endswith('.csv')])
        print(f"  📂 {STATS_DIR}/ - детальная статистика ({stats_files} файлов)")
    
    if create_gifs and os.path.exists(GIFS_DIR):
        gif_files = len([f for f in os.listdir(GIFS_DIR) if f.endswith('.gif')])
        print(f"  📂 {GIFS_DIR}/ - анимации симуляций ({gif_files} GIF файлов)")
    
    # Выводим статистику по причинам остановки
    print(f"\n📊 СТАТИСТИКА ОСТАНОВОК:")
    if 'stop_reason' in final_df.columns:
        stop_reasons = final_df['stop_reason'].value_counts()
        for reason, count in stop_reasons.items():
            percentage = count / len(final_df) * 100
            print(f"  {reason}: {count} симуляций ({percentage:.1f}%)")
    
    # Показываем сводку по параметрам и статусам
    print(f"\n📈 СВОДКА ПО ПАРАМЕТРАМ И СТАТУСАМ:")
    print(f"  Средний Infectivity1: {final_df['infectivity1'].mean():.3f}")
    print(f"  Средний Infectivity2: {final_df['infectivity2'].mean():.5f}")
    print(f"  Средняя финальная популяция: {final_df['final_population'].mean():.0f}")
    
    if 'final_semel_pct' in final_df.columns:
        print(f"  Средний % семельпарии: {final_df['final_semel_pct'].mean():.1f}%")
    
    # Статистика по статусам
    status_summary_cols = ['final_children', 'final_juv_no_terr', 'final_juv_terr', 'final_adults']
    for col in status_summary_cols:
        if col in final_df.columns:
            avg = final_df[col].mean()
            if 'final_population' in final_df.columns and final_df['final_population'].mean() > 0:
                pct = avg / final_df['final_population'].mean() * 100
                print(f"  {col}: {avg:.1f} особей ({pct:.1f}%)")
            else:
                print(f"  {col}: {avg:.1f} особей")
    
    # Корреляция между параметрами и результатами
    print(f"\n🔗 КОРРЕЛЯЦИИ (если достаточно данных):")
    if len(final_df) >= 5:
        try:
            corr_cols = ['infectivity1', 'infectivity2', 'final_semel_pct', 'final_population']
            corr_cols_present = [col for col in corr_cols if col in final_df.columns]
            
            if len(corr_cols_present) >= 2:
                corr_matrix = final_df[corr_cols_present].corr()
                print("  Матрица корреляций:")
                for i, col1 in enumerate(corr_cols_present):
                    for j, col2 in enumerate(corr_cols_present):
                        if i < j:  # Выводим только верхний треугольник
                            corr_value = corr_matrix.iloc[i, j]
                            if abs(corr_value) > 0.3:  # Только значимые корреляции
                                print(f"    {col1} ↔ {col2}: {corr_value:.3f}")
        except:
            print("  Не удалось рассчитать корреляции")
    
    print(f"\n🎉 ВСЕ {len(all_summaries)} СИМУЛЯЦИЙ ЗАВЕРШЕНЫ!")
    print(f"💾 Все данные сохранены в организованных директориях")
    
    return final_df






# Сначала запустите ОДНУ симуляцию для отладки:
if __name__ == "__main__":
    # Тестовая симуляция с фиксированными параметрами
    test_params = {'I1': 3.5, 'I2': 0.03}
    test_result = run_single_simulation(test_params, run_num=1, max_timepoints=900, create_gif=True)
    print("\n✅ ТЕСТОВАЯ СИМУЛЯЦИЯ ЗАВЕРШЕНА!")



# ====================== Запуск Monte Carlo ======================
#if __name__ == "__main__":
    # Запускаем Monte Carlo симуляции с созданием GIF
    # Для тестирования можно поставить меньше симуляций, например num_runs=5
    #results = run_monte_carlo_simulation(num_runs=5, create_gifs=True)
    
    #print(f"\n{'='*80}")
    #print("🎉 ВСЕ СИМУЛЯЦИИ ЗАВЕРШЕНЫ!")
    #print(f"{'='*80}")
    
    # Показываем структуру созданных директорий
    #print(f"\n📁 ФИНАЛЬНАЯ СТРУКТУРА ДИРЕКТОРИЙ:")
    #for root, dirs, files in os.walk('.'):
        # Показываем только наши директории
        #if any(dir_name in root for dir_name in [RESULTS_DIR, STATS_DIR, GIFS_DIR, SNAPSHOTS_DIR]):
            #level = root.replace('.', '').count(os.sep)
            #indent = ' ' * 2 * level
            #print(f"{indent}📂 {os.path.basename(root)}/")
            #subindent = ' ' * 2 * (level + 1)
            #for file in files[:5]:  # Показываем первые 5 файлов
                #if file.endswith(('.csv', '.gif', '.png')):
                    #print(f"{subindent}📄 {file}")
            #if len(files) > 5:
                #print(f"{subindent}... и еще {len(files) - 5} файлов")