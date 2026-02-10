import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio.v2 as imageio

from state import SimulationState
from constants import (
    MAP_X_SIZE, MAP_Y_SIZE, STATUS_CHILD, STATUS_JUVENILE_NO_TERR,
    STATUS_JUVENILE_TERR, STATUS_ADULT
)

def create_output_folders(
    base_folder: str = "../output",
    run_num: int = 0
) -> dict[str, Path]:
    base_path = Path(base_folder)
    snapshots_path = base_path / f"snapshots_run_{run_num:03d}"
    base_path.mkdir(parents=True, exist_ok=True)
    snapshots_path.mkdir(parents=True, exist_ok=True)
    return {"base": base_path, "snapshots": snapshots_path}

def draw_snapshot(
    state,
    output_folder,
    run_num=0,
    figsize=(16, 4)
):
    n = state.pop_size
    time_step = state.current_time
    if n == 0:
        return

    x = state.x[:n].cpu().numpy()
    y = state.y[:n].cpu().numpy()
    status = state.status[:n].cpu().numpy()
    infection_stage = state.infection_stage[:n].cpu().numpy()
    chrom_a = state.chrom_a[:n].cpu().numpy()
    age = state.age[:n].cpu().numpy() if hasattr(state, "age") else None

    # Class logic
    is_child = status == 0
    is_juvenile_no_terr = status == 1
    is_juvenile_terr = status == 2
    is_adult = status == 3
    is_semel = chrom_a.sum(axis=1) == 2
    is_itero = chrom_a.sum(axis=1) == 0
    is_heterozygous = chrom_a.sum(axis=1) == 1
    is_infected = infection_stage > 0

    # Color and size arrays
    colors = np.full(n, "gray", dtype=object)
    sizes = np.full(n, 40, dtype=float)
    edgecolors = np.array(["none"] * n, dtype=object)  # Ensure 1D array of strings

    # Assign colors by class (same as devils_with_kids)
    colors[is_child & is_itero] = "limegreen"      # Child Iteroparous
    colors[is_child & is_semel] = "aquamarine"     # Child Semelparous
    colors[is_child & is_heterozygous] = "purple"     # Child Heterozygous
    
    colors[is_juvenile_no_terr & is_itero] = "khaki"       # Juvenile Iteroparous
    colors[is_juvenile_no_terr & is_semel] = "lightblue"   # Juvenile No Territory Semelparous
    colors[is_juvenile_no_terr & is_heterozygous] = "orange"   # Juvenile No Territory Heterozygous
    
    colors[is_juvenile_terr & is_itero] = "#330019"       # Juvenile Territory Iteroparous
    colors[is_juvenile_terr & is_semel] = "#99004C"   # Juvenile Territory Semelparous
    colors[is_juvenile_terr & is_heterozygous] = "pink"   # Juvenile Territory Heterozygous
    
    colors[is_adult & is_itero] = "y"              # Adult Iteroparous
    colors[is_adult & is_semel] = "dodgerblue"     # Adult Semelparous
    colors[is_adult & is_heterozygous] = "brown"     # Adult Heterozygous

    # Sizes
    sizes[is_child] = 30
    sizes[is_juvenile_no_terr] = 40
    sizes[is_juvenile_terr] = 40
    sizes[is_adult] = 50

    # Infected: red edge, slightly larger
    edgecolors[is_infected] = "red"
    sizes[is_infected] += 10

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=colors, s=sizes, edgecolors=edgecolors.tolist(), alpha=0.8, linewidth=1.5)
    ax.set_xlim(0, state.map_x_size if hasattr(state, "map_x_size") else MAP_X_SIZE)
    ax.set_ylim(0, state.map_y_size if hasattr(state, "map_y_size") else MAP_Y_SIZE)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spatial Distribution (Time: {time_step}, Pop: {n})")
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    filename = output_folder / f"snapshot_{run_num:03d}_{time_step:06d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_gif_from_snapshots(
    snapshot_folder: Path,
    output_folder: Path,
    gif_name: str = "simulation.gif",
    duration: float = 0.1,
    cleanup: bool = True  # Default to True for automatic cleanup
) -> Optional[Path]:
    from PIL import Image
    png_files = sorted(snapshot_folder.glob("snapshot_*.png"))
    if not png_files:
        print(f"⚠️ No PNG files found in {snapshot_folder}")
        return None
    images = [Image.open(png_file) for png_file in png_files]
    gif_path = output_folder / gif_name
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),
        loop=0
    )
    if cleanup:
        for png_file in png_files:
            png_file.unlink()
        print(f"🗑️ Cleaned up {len(png_files)} PNG files")
    return gif_path

def should_draw_snapshot(
    time_step: int,
    draw_times: Optional[List[int]] = None,
    draw_interval: Optional[int] = None
) -> bool:
    if draw_times is not None:
        return time_step in draw_times
    if draw_interval is not None:
        return time_step % draw_interval == 0
    return time_step == 0 or time_step % 1000 == 0