from constants import *
from state import *
from simulation_core import *
from infection import *
from physics import *
from genetics import *
import time
import pandas as pd
from visualisation import *
import argparse

# ---- TIME CONFIG ----
TIMEPOINTS: int = 42000     
# Total simulated days (iterations in main loop). Used in: main().


def main(
    infectivity1: float = INFECTIVITY1,
    infectivity2: float = INFECTIVITY2,
    base_output_folder: str = "../output",
    gif_name: str = "simulation_run_000.gif",
    stats_name: str = "simulation_stats.csv"
) -> None:
    # Override constants with provided values
    global INFECTIVITY1, INFECTIVITY2
    INFECTIVITY1 = infectivity1
    INFECTIVITY2 = infectivity2

    simulation_state: SimulationState = create_initial_state()
    initialize_population(simulation_state)
    
    run_num: int = 0
    folders: dict[str, Path] = create_output_folders(base_folder=base_output_folder, run_num=run_num)
    stats_list: list[SimulationStats] = []
    draw_times: set[int] = {t for t in range(0, TIMEPOINTS, 100)}
    
    PRINT_INTERVAL: int = 1 # Print every n timesteps
    STATS_INTERVAL: int = 30 # Collect stats every n timesteps

    # --- Benchmarking setup ---
    section_names = [
        "pathogen_seeding", "movement", "infection_spread", "reproduction",
        "offspring_lifecycle", "aging", "status_transitions", 
        "death_processing", "stoppage_logic", "statistics_collection",
        "visualization", "progress_reporting"
    ]
    timings = {name: 0.0 for name in section_names}
    total_loop_time = 0.0
    # --- End benchmarking setup ---

    for t in range(TIMEPOINTS):
        loop_start = time.perf_counter()

        day_in_year: int = simulation_state.current_time % DAYS_PER_YEAR
        n = simulation_state.pop_size
        
        # Calculate distances ONCE per timestep. Used in infection spread, calculate fitness, reproduction, movement
        distance_x, distance_y, distance_sq = calculate_full_distance_matrix(simulation_state)

        # --- Pathogen seeding ---
        t0 = time.perf_counter()
        num_infected: int = int((simulation_state.infection_stage[:n] > 0).sum().item())  
        if num_infected == 0: # only if there are no infected
            seed_pathogen(simulation_state)
            print(f"🦠 Pathogen seeded at t={t} (infected count was {num_infected})")
        timings["pathogen_seeding"] += time.perf_counter() - t0

        # --- Infection spread ---
        t0 = time.perf_counter()
        # In main.py, in the "Infection spread" section:
        infection_spread(
            simulation_state,
            distance_sq=distance_sq,
            day_in_year=day_in_year,
            infectivity1=INFECTIVITY1,
            infectivity2=INFECTIVITY2,
            stage1_multiplier=STAGE1_TRANSMISSION_MULTIPLIER,
            stage2_multiplier=STAGE2_TRANSMISSION_MULTIPLIER,
            stage3_multiplier=STAGE3_TRANSMISSION_MULTIPLIER,
            breeding_days=BREEDING_DAYS,
            device=DEVICE
        )
        timings["infection_spread"] += time.perf_counter() - t0
        
        # --- Reproduction ---
        t0 = time.perf_counter()
        if day_in_year < BREEDING_DAYS:
            replication(
                simulation_state,
                distance_sq=distance_sq,
                day_in_year=day_in_year,
                breeding_days=BREEDING_DAYS,
                num_progeny=NUM_OF_PROGENY,
                device=DEVICE
            )
        timings["reproduction"] += time.perf_counter() - t0

        # --- Movement ---
        t0 = time.perf_counter()
        move(simulation_state, distance_x=distance_x, distance_y=distance_y, distance_sq=distance_sq)
        timings["movement"] += time.perf_counter() - t0
        
        # --- Offspring lifecycle ---
        t0 = time.perf_counter()
        birth_pending_offspring(simulation_state, day_in_year=day_in_year)
        disperse_offspring(
            simulation_state,
            day_in_year=day_in_year,
            age_child_after_disposal=AGE_CHILD_AFTER_DISPOSAL,
            time_of_disposal=TIME_OF_DISPOSAL,
            status=STATUS_CHILD
        )
        timings["offspring_lifecycle"] += time.perf_counter() - t0

        n = simulation_state.pop_size

        # --- Aging ---
        t0 = time.perf_counter()
        simulation_state.age[:n] += 1
        timings["aging"] += time.perf_counter() - t0

        # --- Status transitions ---
        t0 = time.perf_counter()
        transition_child_to_juvenile_no_terr(simulation_state)
        transition_juvenile_no_terr_to_terr(simulation_state)
        transition_juvenile_terr_to_adult(simulation_state)
        timings["status_transitions"] += time.perf_counter() - t0

        # --- Death processing ---
        t0 = time.perf_counter()
        process_all_deaths(
            state=simulation_state,
            day_in_year=day_in_year,
            base_mortality=MORTALITY,
            disease_mortality_factor_stage1=DISEASE_MORTALITY_FACTOR_STAGE1,
            disease_mortality_factor_stage2=DISEASE_MORTALITY_FACTOR_STAGE2,
            disease_mortality_factor_stage3=DISEASE_MORTALITY_FACTOR_STAGE3,
            dispersal_deadline=DISPERSAL_DEADLINE,
            maturity_age=AGE_JUVENILE_TO_ADULT,
            semelparous_death_day=SEMELPAROUS_DEATH_DAY,
            device=DEVICE
        )
        timings["death_processing"] += time.perf_counter() - t0

        # --- Stoppage logic ---
        t0 = time.perf_counter()
        if check_simulation_stop(
            state=simulation_state,
            current_time=simulation_state.current_time,
            max_time=TIMEPOINTS,
        ):
            print(f"⏹️ Simulation stopped at t={simulation_state.current_time} (reason: {simulation_state.stoppage_reason})")
            break
        
        timings["stoppage_logic"] += time.perf_counter() - t0

        # --- Statistics collection ---
        t0 = time.perf_counter()
        if t % STATS_INTERVAL == 0:
            stats: SimulationStats = collect_statistics(simulation_state, run_num=run_num)
            stats_list.append(stats)
        timings["statistics_collection"] += time.perf_counter() - t0

        # --- Visualization ---
        t0 = time.perf_counter()
        # if t in draw_times:
        #     draw_snapshot(
        #         state=simulation_state,
        #         output_folder=folders["snapshots"],
        #         run_num=run_num
        #     )
        timings["visualization"] += time.perf_counter() - t0

        # --- Progress reporting ---
        t0 = time.perf_counter()
        if t % PRINT_INTERVAL == 0:
            print(f"t={t:6d} | Pop: {simulation_state.pop_size:5d}")
        timings["progress_reporting"] += time.perf_counter() - t0

        simulation_state.current_time += 1
        total_loop_time += time.perf_counter() - loop_start

    # --- POST-SIMULATION OUTPUT ---
    df: pd.DataFrame = pd.DataFrame([s.__dict__ for s in stats_list])
    csv_path = Path(base_output_folder) / stats_name  # Use custom stats filename
    
    df['result'] = simulation_state.stoppage_reason  # Add stoppage reason to the DataFrame
    df.to_csv(csv_path, index=False)
    print(f"📊 Statistics saved: {csv_path}")

    # create_gif_from_snapshots(
    #     snapshot_folder=folders["snapshots"],
    #     output_folder=Path(base_output_folder),  # Use custom base folder
    #     gif_name=gif_name,  # Use custom GIF filename
    #     duration=0.5
    # )
    
    # Print stoppage reason in a parseable format for run_multiple_experiments.py
    print(f"STOPPAGE_REASON:{simulation_state.stoppage_reason}")

    # --- BENCHMARK SUMMARY ---
    print("\n=== BENCHMARK SUMMARY ===")
    total = sum(timings.values())
    for name in section_names:
        sec = timings[name]
        print(f"{name:22s}: {sec:8.3f} s ({sec/total*100:5.1f}%)")
    print(f"{'Total loop time':22s}: {total_loop_time:8.3f} s (100.0%)")
    print("=========================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Tasmanian Devil simulation.")
    parser.add_argument('--INFECTIVITY1', type=float, default=INFECTIVITY1, help='Sexual transmission probability parameter.')
    parser.add_argument('--INFECTIVITY2', type=float, default=INFECTIVITY2, help='Nonsexual transmission probability parameter.')
    parser.add_argument('--output_folder', type=str, default="../output", help='Base output folder for results.')
    # parser.add_argument('--output_gif', type=str, default="simulation_run_000.gif", help='Filename for the output GIF.')
    parser.add_argument('--output_stats', type=str, default="simulation_stats.csv", help='Filename for the output CSV statistics.')
    
    args = parser.parse_args()
    
    start: float = time.time()
    main(
        infectivity1=args.INFECTIVITY1,
        infectivity2=args.INFECTIVITY2,
        base_output_folder=args.output_folder,
        # gif_name=args.output_gif,
        stats_name=args.output_stats
    )
    end: float = time.time()
    print(f"Simulation runtime: {end - start:.2f} seconds ({(end - start) / 60:.2f} minutes)")