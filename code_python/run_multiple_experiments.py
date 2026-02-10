import subprocess
import os
import itertools
from constants import *  # Import parameters from constants.py
import sys

# Define the ranges for INFECTIVITY1 and INFECTIVITY2
INFECTIVITY1_values: list[float] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
INFECTIVITY2_values: list[float] = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

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
        # Construct descriptive names for GIF and statistics files
        gif_name: str = f'experiment_{i}_INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}_sample_{sample}_animation.gif'
        stats_name: str = f'experiment_{i}_INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}_sample_{sample}_statistics.txt'
        
        # Run main.py with parameters (assuming main.py accepts command-line args)
        cmd: list[str] = [
            sys.executable, 'main.py',
            '--INFECTIVITY1', str(params['INFECTIVITY1']),
            '--INFECTIVITY2', str(params['INFECTIVITY2']),
            '--output_folder', output_dir,  # Add this line to set the base output folder
            '--output_gif', os.path.join(output_dir, gif_name),  # This is now relative to output_dir
            '--output_stats', os.path.join(output_dir, stats_name)  # This is now relative to output_dir
        ]
        subprocess.run(cmd)