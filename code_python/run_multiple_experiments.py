import subprocess
import os
import itertools
from constants import *  # Import parameters from constants.py
import sys



# Define the ranges for INFECTIVITY1 and INFECTIVITY2
INFECTIVITY1_values: list[float] = [0.0, 0.005, 0.01, 0.05, 0.1,  0.5,   1.0, 2.0, 3.0, 4.0, 5.0]
INFECTIVITY2_values: list[float] = [ 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5]

# Generate all combinations of parameter values
param_sets: list[dict[str, float]] = [
    {'INFECTIVITY1': inf1, 'INFECTIVITY2': inf2}
    for inf1, inf2 in itertools.product(INFECTIVITY1_values, INFECTIVITY2_values)
]

# Number of technical samples (e.g., 3)
num_technical_samples: int = 3

# Directory to save outputs
output_dir: str = '../experiments_outputs'
os.makedirs(output_dir, exist_ok=True)

for i, params in enumerate(param_sets):
    for sample in range(num_technical_samples):
        # Construct temporary names (without stoppage reason)
        # temp_gif_name: str = (
        #     f'INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}'
        #     f'_sample_{sample}_animation.gif'
        # )
        temp_stats_name: str = (
            f'INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}'
            f'_sample_{sample}_statistics.csv'
        )

        # Run main.py with parameters and capture output
        cmd: list[str] = [
            sys.executable, 'main.py',
            '--INFECTIVITY1', str(params['INFECTIVITY1']),
            '--INFECTIVITY2', str(params['INFECTIVITY2']),
            '--output_folder', output_dir,
            # '--output_gif', temp_gif_name,
            '--output_stats', temp_stats_name
        ]
        result: subprocess.CompletedProcess = subprocess.run(
            cmd, capture_output=True, text=False  # Changed to text=False to capture bytes
        )

        print('Completed run for ')
        print(f'INFECTIVITY1={params["INFECTIVITY1"]}, INFECTIVITY2={params["INFECTIVITY2"]}, tech sample={sample}')