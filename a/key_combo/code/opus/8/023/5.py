from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def _add_arc_utils_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            return
    fallback = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if fallback.exists():
        sys.path.insert(0, str(fallback.parent))
        return
    raise ImportError("could not locate arc_utils.py")


_add_arc_utils_to_path()

from arc_utils import Color, bbox, get_bg  # noqa: E402


BACKGROUND = int(Color.BLACK)
FRAGMENT_COLOR = int(Color.BLUE)
LEGEND_COLOR = int(Color.ORANGE)
SEED_SIZE = 3
LEGEND_SIZE = SEED_SIZE + 1
MAX_GENERATION_ATTEMPTS = 300

SEED_LIBRARY: dict[str, np.ndarray] = {
    "ring": np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "cross": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}

CORNER_SLICES = (
    (slice(0, 2), slice(0, 2)),
    (slice(0, 2), slice(1, 3)),
    (slice(1, 3), slice(0, 2)),
    (slice(1, 3), slice(1, 3)),
)

FRAGMENT_PLANS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (0, 3),
    (1, 2),
)


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    bg = int(get_bg(grid))
    box = bbox(grid != bg)
    assert box is not None, "expected a non-empty top-left legend"
    r0, c0, r1, c1 = box
    assert (r0, c0) == (0, 0), "legend must be anchored at the top-left corner"
    assert (r1, c1) == (LEGEND_SIZE - 1, LEGEND_SIZE - 1), "legend must be a 4x4 corner box"
    return r0, c0, r1, c1


def _extract_observed_seed(grid: np.ndarray) -> np.ndarray:
    _legend_bbox(grid)
    legend = grid[:LEGEND_SIZE, :LEGEND_SIZE]
    assert np.all(legend[:SEED_SIZE, SEED_SIZE] == LEGEND_COLOR), "right legend edge must be solid orange"
    assert np.all(legend[SEED_SIZE, :LEGEND_SIZE] == LEGEND_COLOR), "bottom legend edge must be solid orange"
    seed = legend[:SEED_SIZE, :SEED_SIZE]
    assert np.all((seed == BACKGROUND) | (seed == FRAGMENT_COLOR)), "seed area may contain only black and blue"
    return seed == FRAGMENT_COLOR


def _matching_seed_candidates(observed_mask: np.ndarray) -> list[np.ndarray]:
    matches: list[np.ndarray] = []
    for candidate in SEED_LIBRARY.values():
        fragments = []
        for rs, cs in CORNER_SLICES:
            frag = np.zeros_like(candidate, dtype=bool)
            frag[rs, cs] = candidate[rs, cs]
            assert frag.sum() == 3, "each corner fragment must be a 3-cell L shape"
            fragments.append(frag)

        found = False
        for count in (1, 2):
            for ids in combinations(range(len(fragments)), count):
                merged = np.zeros_like(candidate, dtype=bool)
                for idx in ids:
                    merged |= fragments[idx]
                if np.array_equal(merged, observed_mask):
                    found = True
                    break
            if found:
                matches.append(candidate)
                break
    return matches


def _infer_seed_mask(observed_mask: np.ndarray) -> np.ndarray:
    matches = _matching_seed_candidates(observed_mask)
    assert len(matches) == 1, "observed fragments must determine a unique four-way symmetric 3x3 seed"
    return matches[0]


def _expand_seed(seed_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    inner = seed_mask[rows % SEED_SIZE, cols % SEED_SIZE]
    outer = seed_mask[(rows // SEED_SIZE) % SEED_SIZE, (cols // SEED_SIZE) % SEED_SIZE]

    out = np.full((h, w), BACKGROUND, dtype=int)
    out[inner & outer] = FRAGMENT_COLOR
    out[inner & ~outer] = LEGEND_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a four-way symmetric 3x3 seed from corner fragments, then use it to render a recursive tiled expansion.

    Input structure:
    - A black background contains one top-left 4x4 legend box.
    - The legend box has a solid orange L border on its right column and bottom row.
    - The enclosed 3x3 seed area contains one or two blue corner-L fragments from an unknown four-way symmetric seed.

    Rule:
    - Remove the orange border and treat the enclosed 3x3 area as a partially observed seed.
    - Infer the unique four-way symmetric 3x3 blue mask whose visible corner fragments match the observation.
    - Expand that seed one recursive level: every blue seed cell becomes a blue copy of the seed mask, every black seed cell
      becomes an orange copy of the same seed mask, and all other cells stay black; tile this 9x9 result from the top-left and crop to the input size.

    Key insight:
    - The input never shows the full seed directly; the blue pixels are only corner evidence, so the recursive output is correct only after symmetry-based seed reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BACKGROUND, "background must be black"
    observed_seed = _extract_observed_seed(g)
    seed_mask = _infer_seed_mask(observed_seed)
    return _expand_seed(seed_mask, g.shape)


def _observed_mask_from_plan(seed_mask: np.ndarray, corners: tuple[int, ...]) -> np.ndarray:
    observed = np.zeros_like(seed_mask, dtype=bool)
    for corner_id in corners:
        rs, cs = CORNER_SLICES[corner_id]
        observed[rs, cs] |= seed_mask[rs, cs]
    return observed


def _plan_for(shape: tuple[int, int], seed_name: str) -> tuple[int, ...]:
    h, w = map(int, shape)
    seed_index = 0 if seed_name == "ring" else 1
    idx = (h * 37 + w * 19 + seed_index * 11) % len(FRAGMENT_PLANS)
    return FRAGMENT_PLANS[idx]


def _render_input(shape: tuple[int, int], seed_name: str) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[:SEED_SIZE, SEED_SIZE] = LEGEND_COLOR
    grid[SEED_SIZE, :LEGEND_SIZE] = LEGEND_COLOR

    seed_mask = SEED_LIBRARY[seed_name]
    corners = _plan_for(shape, seed_name)
    observed = _observed_mask_from_plan(seed_mask, corners)
    grid[:SEED_SIZE, :SEED_SIZE][observed] = FRAGMENT_COLOR
    return grid


def _validate_case(grid: np.ndarray, seed_name: str) -> bool:
    seed_mask = SEED_LIBRARY[seed_name]
    observed_seed = _extract_observed_seed(grid)
    if not np.array_equal(_infer_seed_mask(observed_seed), seed_mask):
        return False

    expected = _expand_seed(seed_mask, grid.shape)
    if not np.array_equal(main(grid), expected):
        return False

    other_name = "cross" if seed_name == "ring" else "ring"
    other_output = _expand_seed(SEED_LIBRARY[other_name], grid.shape)
    if np.array_equal(expected, other_output):
        return False

    # Reject the shortcut that skips symmetry completion and expands the partial observation directly.
    shortcut_output = _expand_seed(observed_seed, grid.shape)
    if np.array_equal(expected, shortcut_output):
        return False

    visible_corners = 0
    for rs, cs in CORNER_SLICES:
        fragment = np.zeros_like(seed_mask, dtype=bool)
        fragment[rs, cs] = seed_mask[rs, cs]
        if np.array_equal(observed_seed & fragment, fragment):
            visible_corners += 1
    if visible_corners not in (1, 2):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    seed_name = random.choice(tuple(SEED_LIBRARY))
    grid = _render_input((h, w), seed_name)
    if not _validate_case(grid, seed_name):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
