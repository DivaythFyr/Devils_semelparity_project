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


# ...existing code...

def extract_stoppage_reason(output: str) -> str:
    """
    Parse the stdout of main.py to extract the stoppage reason.
    Searches from the end of the output to find the last occurrence.

    Args:
        output: Full stdout string from the subprocess.

    Returns:
        The stoppage reason string, or 'unknown' if not found.
    """
    for line in reversed(output.splitlines()):
        if line.startswith("STOPPAGE_REASON:"):
            return line.split(":", 1)[1].strip()
    return "unknown"

for i, params in enumerate(param_sets):
    for sample in range(num_technical_samples):
        # Construct temporary names (without stoppage reason)
        temp_gif_name: str = (
            f'INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}'
            f'_sample_{sample}_animation.gif'
        )
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
            '--output_gif', temp_gif_name,
            '--output_stats', temp_stats_name
        ]
        result: subprocess.CompletedProcess = subprocess.run(
            cmd, capture_output=True, text=True
        )

        # Print subprocess output for monitoring
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Extract stoppage reason and rename files
        reason: str = extract_stoppage_reason(result.stdout)

        final_gif_name: str = (
            f'INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}'
            f'_sample_{sample}_result_{reason}_animation.gif'
        )
        final_stats_name: str = (
            f'INF1_{params["INFECTIVITY1"]}_INF2_{params["INFECTIVITY2"]}'
            f'_sample_{sample}_result_{reason}_statistics.csv'
        )

        # Rename GIF
        temp_gif_path: str = os.path.join(output_dir, temp_gif_name)
        final_gif_path: str = os.path.join(output_dir, final_gif_name)
        if os.path.exists(temp_gif_path):
            os.rename(temp_gif_path, final_gif_path)
            print(f"✅ GIF renamed: {final_gif_name}")

        # Rename statistics CSV
        temp_stats_path: str = os.path.join(output_dir, temp_stats_name)
        final_stats_path: str = os.path.join(output_dir, final_stats_name)
        if os.path.exists(temp_stats_path):
            os.rename(temp_stats_path, final_stats_path)
            print(f"✅ Stats renamed: {final_stats_name}")