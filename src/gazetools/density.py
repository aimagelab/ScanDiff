import matplotlib.patches as mpatches

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from tqdm import tqdm

def compute_density_chart_core(
    data: List[Tuple[str, List[float], str]],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    degree=False,
):
    fig = plt.figure()
    for d in data:
        dat = np.array(d[1])
        dat = dat[dat < np.percentile(dat, q=99)]
        sns.kdeplot(dat, fill=True, color=d[2])
    plt.legend(
        title=title,
        loc="upper right",
        labels=list(map(lambda d: d[0], data)),
    )
    if degree:
        plt.xticks(np.arange(-180, 181, 90))
        plt.xlim(-200.0, 200.0)
    plt.gca().axes.yaxis.set_ticklabels([])  # type: ignore
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    fig.savefig(save_path)


def compute_bivariate_density_chart_core(
    data: List[Tuple[str, List[float], List[float], str]],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
):
    fig = plt.figure()
    for d in data:
        sns.kdeplot(x=d[1], y=d[2], fill=len(data) == 1, color=d[3])
    plt.legend(
        title=title,
        loc="upper right",
        handles=list(map(lambda d: mpatches.Patch(color=d[3], label=d[0]), data)),
    )
    plt.ylim(bottom=-100)
    plt.xlim(left=-100)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    fig.savefig(save_path)

def compute_lengths_density_chart(
    originals: List[float],
    generated: List[float],
    save_path: str,
):
    data = [
        ("Human", originals, "b"),
        ("Generated", generated, "r"),
    ]
    compute_density_chart_core(
        data,
        "Scanpath Length Density",
        "Density",
        "Scanpath Length (num_fixations)",
        save_path,
    )

def compute_durations_density_chart(
    originals: List[float],
    generated: List[float],
    save_path: str,
):
    data = [
        ("Human", originals, "b"),
        ("Generated", generated, "r"),
    ]
    compute_density_chart_core(
        data,
        "Inter-times Density",
        "Density",
        "Inter-times (ms)",
        save_path,
    )


def compute_marks_density_chart(
    originals: Tuple[List[float], List[float]],
    generated: Tuple[
        List[float], List[float]
    ],  # | List[Tuple[str, List[float], List[float]]]
    save_path: str,
):
    data = [
        ("Human", originals[0], originals[1], "b"),
        ("Generated", generated[0], generated[1], "r"),
    ]

    compute_bivariate_density_chart_core(
        data,
        "Marks Density",
        "Y Marks",
        "X Marks",
        save_path,
    )


def compute_saccades_direction_chart(
    originals: List[Tuple[List[float], List[float]]],
    generated: List[Tuple[List[float], List[float]]],
    save_path: str,
):

    def calculate_direction(scanpath: List[Tuple[float, float]]):
        l: List[float] = []
        for f in range(len(scanpath) - 1):
            curr_fix2 = scanpath[f + 1]
            curr_fix1 = scanpath[f]
            l.append(
                np.arctan2(curr_fix2[1] - curr_fix1[1], curr_fix2[0] - curr_fix1[0])
                + np.pi
            )
        return l

    originals_directions: List[float] = []

    for x, y in tqdm(originals, desc="Calculating Direction for Reals"):  # type: ignore
        scanpath = list(zip(x, y))
        originals_directions = originals_directions + calculate_direction(scanpath)

    generated_directions: List[float] = []

    for x, y in tqdm(generated, desc="Calculating Direction for Generated"):  # type: ignore
        scanpath = list(zip(x, y))
        generated_directions = generated_directions + calculate_direction(scanpath)

    data = [
        ("Human", np.rad2deg(np.array(originals_directions)) - 180, "b"),
        ("Generated", np.rad2deg(np.array(generated_directions)) - 180, "r"),
    ]

    compute_density_chart_core(
        data,
        "Saccades Direction Density",
        "Density",
        "Direction (deg)",
        save_path,
        degree=True,
    )


def compute_saccades_amplitude_chart(
    originals: List[Tuple[List[float], List[float]]],
    generated: List[Tuple[List[float], List[float]]],
    save_path: str,
):

    def calculate_amplitude(scanpath: List[Tuple[float, float]]):
        l: List[float] = []
        for f in range(len(scanpath) - 1):
            curr_fix2 = scanpath[f + 1]
            curr_fix1 = scanpath[f]
            l.append(np.linalg.norm(np.array(curr_fix2) - np.array(curr_fix1)))  # type: ignore
        return l

    originals_amplitudes: List[float] = []

    for x, y in tqdm(originals, desc="Calculating Amplitudes for Reals"):  # type: ignore
        scanpath = list(zip(x, y))
        originals_amplitudes = originals_amplitudes + calculate_amplitude(scanpath)

    generated_amplitudes: List[float] = []

    for x, y in tqdm(generated, desc="Calculating Amplitudes for Generated"):  # type: ignore
        scanpath = list(zip(x, y))
        generated_amplitudes = generated_amplitudes + calculate_amplitude(scanpath)

    data = [
        ("Human", originals_amplitudes, "b"),
        ("Generated", generated_amplitudes, "r"),
    ]

    compute_density_chart_core(
        data,
        "Saccades Amplitude Density",
        "Density",
        "Amplitude (pix)",
        save_path,
    )
