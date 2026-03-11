import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio.v2 as imageio
import shutil

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
    figsize=(16, 6)
):
    from matplotlib.patches import Patch

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
    edgecolors = np.array(["none"] * n, dtype=object)

    # Assign colors by class
    colors[is_child & is_itero] = "limegreen"
    colors[is_child & is_semel] = "aquamarine"
    colors[is_child & is_heterozygous] = "purple"

    colors[is_juvenile_no_terr & is_itero] = "khaki"
    colors[is_juvenile_no_terr & is_semel] = "lightblue"
    colors[is_juvenile_no_terr & is_heterozygous] = "orange"

    colors[is_juvenile_terr & is_itero] = "#330019"
    colors[is_juvenile_terr & is_semel] = "#99004C"
    colors[is_juvenile_terr & is_heterozygous] = "pink"

    colors[is_adult & is_itero] = "y"
    colors[is_adult & is_semel] = "dodgerblue"
    colors[is_adult & is_heterozygous] = "brown"

    # Sizes
    sizes[is_child] = 30
    sizes[is_juvenile_no_terr] = 40
    sizes[is_juvenile_terr] = 40
    sizes[is_adult] = 50

    # Infected: red edge, slightly larger
    edgecolors[is_infected] = "red"
    sizes[is_infected] += 10

    # --- Count animals in each category ---
    count_child_itero = int((is_child & is_itero).sum())
    count_child_semel = int((is_child & is_semel).sum())
    count_child_hetero = int((is_child & is_heterozygous).sum())
    count_juv_nt_itero = int((is_juvenile_no_terr & is_itero).sum())
    count_juv_nt_semel = int((is_juvenile_no_terr & is_semel).sum())
    count_juv_nt_hetero = int((is_juvenile_no_terr & is_heterozygous).sum())
    count_juv_t_itero = int((is_juvenile_terr & is_itero).sum())
    count_juv_t_semel = int((is_juvenile_terr & is_semel).sum())
    count_juv_t_hetero = int((is_juvenile_terr & is_heterozygous).sum())
    count_adult_itero = int((is_adult & is_itero).sum())
    count_adult_semel = int((is_adult & is_semel).sum())
    count_adult_hetero = int((is_adult & is_heterozygous).sum())
    count_infected = int(is_infected.sum())

    # --- Legend entries with counts ---
    legend_items = [
        Patch(facecolor="limegreen", edgecolor="black", label=f"Child Itero ({count_child_itero})"),
        Patch(facecolor="aquamarine", edgecolor="black", label=f"Child Semel ({count_child_semel})"),
        Patch(facecolor="purple", edgecolor="black", label=f"Child Hetero ({count_child_hetero})"),
        Patch(facecolor="khaki", edgecolor="black", label=f"Juv NoTerr Itero ({count_juv_nt_itero})"),
        Patch(facecolor="lightblue", edgecolor="black", label=f"Juv NoTerr Semel ({count_juv_nt_semel})"),
        Patch(facecolor="orange", edgecolor="black", label=f"Juv NoTerr Hetero ({count_juv_nt_hetero})"),
        Patch(facecolor="#330019", edgecolor="black", label=f"Juv Terr Itero ({count_juv_t_itero})"),
        Patch(facecolor="#99004C", edgecolor="black", label=f"Juv Terr Semel ({count_juv_t_semel})"),
        Patch(facecolor="pink", edgecolor="black", label=f"Juv Terr Hetero ({count_juv_t_hetero})"),
        Patch(facecolor="y", edgecolor="black", label=f"Adult Itero ({count_adult_itero})"),
        Patch(facecolor="dodgerblue", edgecolor="black", label=f"Adult Semel ({count_adult_semel})"),
        Patch(facecolor="brown", edgecolor="black", label=f"Adult Hetero ({count_adult_hetero})"),
        Patch(facecolor="white", edgecolor="red", linewidth=2, label=f"Infected ({count_infected})"),
    ]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=colors, s=sizes, edgecolors=edgecolors.tolist(), alpha=0.3, linewidth=1.5)
    ax.set_xlim(0, state.map_x_size if hasattr(state, "map_x_size") else MAP_X_SIZE)
    ax.set_ylim(0, state.map_y_size if hasattr(state, "map_y_size") else MAP_Y_SIZE)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spatial Distribution (Time: {time_step}, Pop: {n})")
    ax.set_aspect('equal', adjustable='box')

    # Place legend above the map
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=7,
        frameon=True,
        fancybox=True,
        shadow=False,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.0,
    )

    # Make room at the top for the legend
    fig.subplots_adjust(top=0.78)

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
    # if cleanup:
    #     for png_file in png_files:
    #         png_file.unlink()
    #     print(f"🗑️ Cleaned up {len(png_files)} PNG files")
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

def delete_sub_snapshots(image_folder: str) -> None:
    """
    Deletes the contents of the folder containing sub-snapshot images, but keeps the folder.
    
    Args:
        image_folder: Path to the folder whose contents to delete.
    """
    folder_path = Path(image_folder)
    if folder_path.exists() and folder_path.is_dir():
        for file_path in folder_path.glob("*.png"):  # Adjust pattern if needed (e.g., "*" for all files)
            file_path.unlink()
        print(f"Deleted contents of folder: {image_folder}")
    else:
        print(f"Folder not found: {image_folder}")