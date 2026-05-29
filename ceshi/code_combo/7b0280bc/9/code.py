# Combo variant of 7b0280bc (index 9)
# combo_spec: D11/2+D16/0
# Name: 干扰色环境下断裂轨道寻路

from __future__ import annotations

# Design analysis
# ---------------
# `D16/0` already changes edge validity: visually near-complete rails can be broken and must
# be excluded from the graph. `D11/2` adds a second, orthogonal burden: ignore isolated
# nonfunctional color noise before reconstructing the graph.
# These two behaviors compose cleanly if the distractor color is restricted to isolated
# background singletons. Then the combo solver can do:
#   1. remove singleton-color noise,
#   2. run the original broken-rail solver on the cleaned grid.
# The generator therefore starts from a valid `D16/0` sample and injects only isolated pixels
# of one extra color, keeping them away from existing foreground so they never mutate graph
# connectivity.

import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
ROOT = Path("/home/lixinyu/snap/new")
for p in (PARENT, ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from arc_utils import colors_present, components_mask, get_bg
from _shared import load_variant


_D16 = load_variant("D16", 0)
DISTRACTOR_COUNT_RANGE = (12, 22)


def _remove_singleton_noise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(g))
    for col in colors_present(g, exclude={bg}):
        comps = components_mask(g == int(col), connectivity=8)
        if comps and max(int(comp.shape[0]) for comp in comps) == 1:
            g[g == int(col)] = int(bg)
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignore isolated distractor pixels, then solve the broken-rail shortest-path task.

    Input structure:
    - Same graph vocabulary as D16/0: node squares plus track rails, where some rails have
      tiny internal breaks and are unusable.
    - One additional color appears only as isolated singleton noise pixels in the background.

    Rule:
    - Filter out the singleton distractor color.
    - Reconstruct the graph from the cleaned grid.
    - Broken rails remain unusable exactly as in D16/0.

    Key insight: The combo adds a perception stage before the topological stage: first ignore
    what is not part of the graph at all, then reason about which remaining rails are intact.
    """
    cleaned = _remove_singleton_noise(np.asarray(grid, dtype=int))
    return _D16.main(cleaned)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        g = np.asarray(_D16.generate_input(), dtype=int).copy()
        bg = int(get_bg(g))
        used = set(map(int, np.unique(g)))
        candidates = [c for c in range(10) if c not in used and c != bg]
        if not candidates:
            continue
        distractor = int(rng.choice(candidates))

        coords = [(int(r), int(c)) for r, c in np.argwhere(g == bg)]
        rng.shuffle(coords)
        added = 0
        target = rng.randint(*DISTRACTOR_COUNT_RANGE)
        for r, c in coords:
            # Keep distractors visually isolated from every non-background pixel.
            r0 = max(0, r - 1)
            r1 = min(g.shape[0], r + 2)
            c0 = max(0, c - 1)
            c1 = min(g.shape[1], c + 2)
            patch = g[r0:r1, c0:c1]
            if np.any(patch != bg):
                continue
            g[r, c] = distractor
            added += 1
            if added >= target:
                break
        if added < 8:
            continue

        try:
            _ = main(g)
        except Exception:
            continue
        return g
    raise RuntimeError("generate_input: failed to inject valid distractor noise")
