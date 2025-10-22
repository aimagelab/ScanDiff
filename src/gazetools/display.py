import matplotlib.pyplot as plt
from PIL.Image import Image
from matplotlib.axes import Axes
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns

from .utils import compute_saliency_map

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

sns.set(color_codes=True)

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLORS = {
    "butter": ["#fce94f", "#edd400", "#c4a000"],
    "orange": ["#fcaf3e", "#f57900", "#ce5c00"],
    "chocolate": ["#e9b96e", "#c17d11", "#8f5902"],
    "chameleon": ["#8ae234", "#73d216", "#4e9a06"],
    "skyblue": ["#729fcf", "#3465a4", "#204a87"],
    "plum": ["#ad7fa8", "#75507b", "#5c3566"],
    "scarletred": ["#ef2929", "#cc0000", "#a40000"],
    "aluminium": ["#eeeeec", "#d3d7cf", "#babdb6", "#888a85", "#555753", "#2e3436"],
}


def draw_scanpath(
    ax: Axes,
    fix_x: List[float],
    fix_y: List[float],
    fix_d: List[float],
    alpha=1,
    invert_y=False,
    ydim=None,
):
    if invert_y:
        if ydim is None:
            raise RuntimeError("ydim must be provided")
        fix_y = ydim - 1 - fix_y

    if len(fix_d) == 0:
        fix_d = [200] * len(fix_x)  # default duration for visualization

    for i in range(1, len(fix_x)):
        ax.arrow(
            fix_x[i - 1],
            fix_y[i - 1],
            fix_x[i] - fix_x[i - 1],
            fix_y[i] - fix_y[i - 1],
            alpha=alpha,
            fc=COLORS["orange"][0],
            ec=COLORS["orange"][0],
            fill=True,
            shape="full",
            width=3,
            head_width=0,
            head_starts_at_zero=False,
            overhang=0,
        )

    if len(fix_d) == 0:
        fix_d = [200] * len(fix_x) # default duration for visualization
    
    for i in range(len(fix_x)):
        if i == 0:
            ax.plot(
                fix_x[i],
                fix_y[i],
                marker="o",
                ms=fix_d[i] / 7,
                mfc=COLORS["skyblue"][0],
                mec="black",
                alpha=0.7,
            )
        elif i == len(fix_x) - 1:
            ax.plot(
                fix_x[i],
                fix_y[i],
                marker="o",
                ms=fix_d[i] / 7,
                mfc=COLORS["scarletred"][0],
                mec="black",
                alpha=0.7,
            )
        else:
            ax.plot(
                fix_x[i],
                fix_y[i],
                marker="o",
                ms=fix_d[i] / 7,
                mfc=COLORS["aluminium"][0],
                mec="black",
                alpha=0.7,
            )

    for i in range(len(fix_x)):
        ax.text(
            fix_x[i] - 4,
            fix_y[i] + 1,
            str(i + 1),
            color="black",
            ha="left",
            va="center",
            multialignment="center",
            alpha=alpha,
            fontsize=14,
        )


def save_image_scanpaths(
    image: Image, x: List[float], y: List[float], t: List[float], save_path: str
):
    plt.grid(False)
    _, axs = plt.subplots()
    axs.grid(False)
    axs.imshow(image, interpolation="none")
    draw_scanpath(axs, x, y, t)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_image_heatmaps(
    x: List[float], y: List[float], size: Tuple[int, int], save_path: str
):
    filter = compute_saliency_map(np.column_stack([x, y]), size)

    plt.grid(False)
    _, axs = plt.subplots()
    axs.grid(False)
    axs.imshow(filter.T, interpolation="none")
    plt.title("Fixation Map")
    plt.savefig(save_path)
    plt.close()

def save_overlaid_saliency_map(img, sal_map, output_file):
    #process sal_map
    sal_map = np.expand_dims(sal_map, axis=2)
    mask = np.zeros(np.array(img).shape, dtype=np.uint8)
    sal_map = (sal_map - np.min(sal_map)) / (np.max(sal_map) - np.min(sal_map))
    sal_map *= 255
    sal_map = sal_map.astype(np.uint8)
    
    over = np.dstack((mask, 255 - sal_map))
    
    plt.grid(False)
    _, axs = plt.subplots()
    axs.grid(False)
    axs.imshow(img, interpolation="none")
    axs.imshow(over, cmap='gray', alpha=1)
    plt.axis("off")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()