from constants import *
from state import *
from simulation_core import *
from infection import *
from physics import *
from genetics import *
import random
import numpy as np
import torch
import itertools
import sys
import time
from pathlib import Path
import pandas as pd
from visualisation import *


# ---- TIME CONFIG ----
TIMEPOINTS: int = 42000
# Total simulated days (iterations in main loop).

NUM_OF_TECH_SAMPLES: int = 1
# Number of technical replicates per parameter set.


def main(
    base_output_folder: str = "../output",
    stats_name: str = "simulation_stats.csv",
) -> None:
    simulation_state: SimulationState = create_initial_state()
    initialize_population(simulation_state)

    # Build output paths:
    # One dedicated folder per run, based on CSV stem.
    # CSV is stored in the same folder as snapshot images.
    base_path: Path = Path(base_output_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    csv_stem: str = Path(stats_name).stem
    snapshots_folder: Path = base_path / "snapshots" / csv_stem
    snapshots_folder.mkdir(parents=True, exist_ok=True)

    csv_path: Path = snapshots_folder / stats_name

    stats_list: list[SimulationStats] = []
    draw_times: set[int] = {t for t in range(0, TIMEPOINTS, 25)}

    PRINT_INTERVAL: int = 1
    STATS_INTERVAL: int = 1

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

    # Clean only this run's snapshots folder
    delete_sub_snapshots(str(snapshots_folder))

    for t in range(TIMEPOINTS):
        loop_start = time.perf_counter()

        day_in_year: int = simulation_state.current_time % DAYS_PER_YEAR
        n = simulation_state.pop_size

        # Calculate distances ONCE per timestep. Used in infection spread, fitness, reproduction, movement.
        distance_x, distance_y, distance_sq = calculate_full_distance_matrix(simulation_state)

        # --- Pathogen seeding ---
        t0 = time.perf_counter()
        num_infected: int = int((simulation_state.infection_stage[:n] > 0).sum().item())
        if num_infected == 0 and n > 200: 
            seed_pathogen(simulation_state)
            print(f"🦠 Pathogen seeded at t={t} (infected count was {num_infected})")
        timings["pathogen_seeding"] += time.perf_counter() - t0

        # --- Infection spread ---
        t0 = time.perf_counter()
        infection_spread(
            simulation_state,
            distance_sq=distance_sq,
            day_in_year=day_in_year,
            infectivity1=INFECTIVITY1,
            infectivity2=INFECTIVITY2,
            stage1_multiplier=STAGE1_TRANSMISSION_MULTIPLIER,
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
            stats: SimulationStats = collect_statistics(simulation_state, run_num=0)
            stats_list.append(stats)
        timings["statistics_collection"] += time.perf_counter() - t0

        # --- Visualization (snapshots only) ---
        t0 = time.perf_counter()
        if t in draw_times:
            draw_snapshot(
                state=simulation_state,
                output_folder=snapshots_folder,
                run_num=0
            )
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
    df["result"] = simulation_state.stoppage_reason
    df.to_csv(csv_path, index=False)
    print(f"📊 Statistics saved: {csv_path}")

    # Rename snapshots so each has CSV stem in filename
    # Example: sample_000__INFECTIVITY1_0.01__INFECTIVITY2_0.05_000120.png
    for png_file in sorted(snapshots_folder.glob("snapshot_*.png")):
        time_part: str = png_file.stem.split("_")[-1]
        new_name: Path = snapshots_folder / f"{csv_stem}_{time_part}.png"
        if new_name != png_file:
            png_file.rename(new_name)

    print(f"🖼️ Snapshots saved: {snapshots_folder}")

    # Print stoppage reason in parseable format
    print(f"STOPPAGE_REASON:{simulation_state.stoppage_reason}")

    # --- BENCHMARK SUMMARY ---
    print("\n=== BENCHMARK SUMMARY ===")
    total = sum(timings.values())
    for name in section_names:
        sec = timings[name]
        pct = (sec / total * 100.0) if total > 0 else 0.0
        print(f"{name:22s}: {sec:8.3f} s ({pct:5.1f}%)")
    print(f"{'Total loop time':22s}: {total_loop_time:8.3f} s (100.0%)")
    print("=========================\n")
    
    
if __name__ == "__main__":

    NUM_MONTE_CARLO_RUNS: int = 60
    BASE_SEED: int = 12345

    run_counter: int = 0

    for mc_run in range(NUM_MONTE_CARLO_RUNS):
        for tech_sample in range(NUM_OF_TECH_SAMPLES):
            run_counter += 1

            # Unique seed per Monte Carlo run + technical sample
            seed = BASE_SEED + mc_run * NUM_OF_TECH_SAMPLES + tech_sample
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            rng = np.random.default_rng(seed)

            # Randomly sampled parameters (Monte Carlo)
            step = 0.005  # allowed values: 0.000, 0.005, 0.010, ..., 0.500
            current_params: dict[str, float] = {
                "INFECTIVITY1": float(rng.integers(0, int(0.5 / step) + 1) * step),
                "INFECTIVITY2": float(rng.integers(0, int(0.5 / step) + 1) * step),
            }

            # Apply sampled values before each main() run
            for key, value in current_params.items():
                if key in globals():
                    globals()[key] = value

                for module_name in ("constants", "infection", "physics", "simulation_core", "genetics"):
                    module = sys.modules.get(module_name)
                    if module is not None and hasattr(module, key):
                        setattr(module, key, value)

            # Run name includes MC id + seed + sampled params
            param_suffix = "__".join(
                f"{k}_{v:.5f}" if isinstance(v, float) else f"{k}_{v}"
                for k, v in current_params.items()
            )
            output_stem = f"mc_{mc_run:05d}__sample_{tech_sample:03d}__seed_{seed}__{param_suffix}"
            stats_name = f"{output_stem}_statistics.csv"

            print(f"\nRun #{run_counter}")
            print(f"  mc_run={mc_run}")
            print(f"  tech_sample={tech_sample}")
            print(f"  seed={seed}")
            print(f"  params={current_params}")

            start: float = time.time()
            main(
                base_output_folder="../experiments_outputs",
                stats_name=stats_name,
            )
            end: float = time.time()
            print(f"Simulation runtime: {end - start:.2f} seconds ({(end - start) / 60:.2f} minutes)")


# if __name__ == "__main__":
#     parameter_sets: dict[str, list] = {
#         "INFECTIVITY1": [0.01],
#         "INFECTIVITY2": [0.01],
#         # Add more parameters if needed, e.g.:
#         # "MORTALITY": [0.002, 0.003, 0.004],
#     }

#     param_names: list[str] = list(parameter_sets.keys())
#     param_value_lists: list[list] = [parameter_sets[name] for name in param_names]

#     run_counter: int = 0

#     # Loop over all parameter combinations
#     for combo in itertools.product(*param_value_lists):
#         current_params: dict[str, float] = dict(zip(param_names, combo))

#         # Technical samples loop
#         for tech_sample in range(NUM_OF_TECH_SAMPLES):
#             run_counter += 1

#             # Assign looped values before each main() run
#             for key, value in current_params.items():
#                 # Update globals in this file
#                 if key in globals():
#                     globals()[key] = value

#                 # Also update same-name globals in imported modules
#                 for module_name in ("constants", "infection", "physics", "simulation_core", "genetics"):
#                     module = sys.modules.get(module_name)
#                     if module is not None and hasattr(module, key):
#                         setattr(module, key, value)

#             # Build run name from sample + looped params
#             param_suffix = "__".join(
#                 f"{k}_{v:g}" if isinstance(v, float) else f"{k}_{v}"
#                 for k, v in current_params.items()
#             )
#             output_stem = f"sample_{tech_sample:03d}__{param_suffix}"

#             stats_name = f"{output_stem}_statistics.csv"

#             print(f"\nRun #{run_counter}")
#             print(f"  tech_sample={tech_sample}")
#             print(f"  params={current_params}")

#             start: float = time.time()
#             main(
#                 base_output_folder="../experiments_outputs",
#                 stats_name=stats_name,
#             )
#             end: float = time.time()
#             print(f"Simulation runtime: {end - start:.2f} seconds ({(end - start) / 60:.2f} minutes)")