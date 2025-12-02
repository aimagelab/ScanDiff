from typing import List
from matplotlib import pyplot as plt
from scipy.stats import entropy

import numpy as np


def perform_kld(real: List[float], generated: List[float], nbins=100) -> float:
    bins = np.linspace(
        np.min([np.min(real), np.min(generated)]),
        np.max([np.max(real), np.max(generated)]),
        nbins,
    )
    h1 = np.histogram(real, bins=bins)[0]
    h1 = h1 / np.sum(h1)
    h2 = np.histogram(generated, bins=bins)[0]
    h2 = h2 / np.sum(h2)

    return entropy(h1, h2 + 1e-10)  # type: ignore
