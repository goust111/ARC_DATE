from __future__ import annotations

from collections import Counter
from importlib import util as importlib_util

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"

try:
    _ARC_UTILS_SPEC = importlib_util.spec_from_file_location("_arc_utils_013_8", _ARC_UTILS_PATH)
    if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
        raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
    _ARC_UTILS = importlib_util.module_from_spec(_ARC_UTILS_SPEC)
    _ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
    Color = _ARC_UTILS.Color
    get_bg = _ARC_UTILS.get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


BG_COLOR = Color.GREEN
SEPARATOR_COLOR = Color.GREY
SEED_COLOR = Color.ORANGE
GROWTH_COLOR = Color.PINK
TOP_PALETTE = np.array([Color.BLUE, Color.RED, Color.YELLOW, Color.TEAL, Color.MAROON], dtype=int)

_RNG = np.random.default_rng()
_GEN_COUNTER = 0
_BRANCHES = ("deep", "single", "seed_only", "empty")
_GEN_ATTEMPTS = 300


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h = grid.shape[0]
    strong = []
    weak = []
    for r in range(2, h - 2):
        row = grid[r]
        if not np.all(row == row[0]):
            continue
        if int(row[0]) == bg:
            continue
        weak.append(r)
        if np.all(grid[r + 1 :] == bg) and np.any(grid[r - 2 : r] != bg):
            strong.append(r)
    candidates = strong if strong else weak
    assert len(candidates) == 1, "expected exactly one horizontal separator line"
    return int(candidates[0])


def _seed_mask_from_top_rows(grid: np.ndarray, separator_row: int, bg: int) -> np.ndarray:
    top_a = grid[separator_row - 2]
    top_b = grid[separator_row - 1]
    return (top_a != bg) & (top_b != bg) & (top_a == top_b)


def _next_generation(mask: np.ndarray) -> np.ndarray:
    nxt = np.zeros_like(mask, dtype=bool)
    for c in range(mask.shape[0] - 2):
        if bool(mask[c]) and (not bool(mask[c + 1])) and bool(mask[c + 2]):
            nxt[c + 1] = True
    return nxt


def _generation_masks(seed_mask: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    current = seed_mask.astype(bool).copy()
    while current.any():
        masks.append(current.copy())
        current = _next_generation(current)
    return masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two rows above the separator, then grow the recovered seed row into alternating orange-pink triangles.

    Input structure:
    - One full-width horizontal separator line splits the grid.
    - The two rows immediately above that line contain colored cells aligned by column,
      with some columns left as background.
    - Everything below the separator is background in the input, and pink/orange do not appear there.

    Rule:
    - Let bg be the dominant background color and find the separator row.
    - For each column in the two rows above the separator, if both cells are non-background
      and have the same color, place an orange seed two rows below the separator in that column.
    - Repeatedly inspect only the latest painted row: every adjacent triple of columns matching
      colored-background-colored creates one child pixel two rows lower in the middle column.
    - Alternate generated row colors orange, pink, orange, ... while keeping the separator
      and everything above it unchanged.

    Key insight:
    - The visible lower pattern is not drawn in the input; its first row must be decoded from
      column-wise equality of the two top rows, then the triangle continues by gap-filling every other row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    separator_row = _find_separator_row(g, bg)
    seed_row = separator_row + 2
    assert seed_row < g.shape[0], "seed row must be within bounds"

    out = g.copy()
    out[separator_row + 1 :] = bg

    current = _seed_mask_from_top_rows(g, separator_row, bg)
    row = seed_row
    color = SEED_COLOR
    while row < g.shape[0] and current.any():
        out[row, current] = color
        current = _next_generation(current)
        row += 2
        color = GROWTH_COLOR if color == SEED_COLOR else SEED_COLOR

    return out


def _random_top_color() -> int:
    return int(_RNG.choice(TOP_PALETTE))


def _random_mismatch_pair() -> tuple[int, int]:
    if _RNG.random() < 0.34:
        return BG_COLOR, _random_top_color()
    if _RNG.random() < 0.5:
        return _random_top_color(), BG_COLOR
    colors = _RNG.choice(TOP_PALETTE, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _make_seed_only_pattern(width: int) -> np.ndarray:
    target_count = int(_RNG.integers(2, min(6, width) + 1))
    chosen: list[int] = []
    for c in _RNG.permutation(width):
        if all(abs(int(c) - old) != 2 for old in chosen):
            chosen.append(int(c))
            if len(chosen) == target_count:
                break
    if not chosen:
        chosen.append(int(_RNG.integers(0, width)))

    seed = np.zeros(width, dtype=bool)
    seed[chosen] = True
    assert len(_generation_masks(seed)) == 1
    return seed


def _make_single_generation_pattern(width: int) -> np.ndarray:
    seed = np.zeros(width, dtype=bool)
    start = int(_RNG.integers(0, width - 2))
    seed[start] = True
    seed[start + 2] = True

    if width >= 10 and _RNG.random() < 0.5:
        possible = [c for c in range(0, width - 2) if abs(c - start) >= 5]
        if possible:
            other = int(_RNG.choice(possible))
            seed[other] = True
            seed[other + 2] = True

    assert len(_generation_masks(seed)) == 2
    return seed


def _make_deep_pattern(width: int, levels: int) -> np.ndarray:
    span = 2 * levels - 1
    start = int(_RNG.integers(0, width - span + 1))
    seed = np.zeros(width, dtype=bool)
    seed[start : start + span : 2] = True
    assert len(_generation_masks(seed)) == levels
    return seed


def _encode_top_rows(grid: np.ndarray, separator_row: int, seed_mask: np.ndarray) -> None:
    top_a = np.full(seed_mask.shape[0], BG_COLOR, dtype=int)
    top_b = np.full(seed_mask.shape[0], BG_COLOR, dtype=int)

    for c, is_seed in enumerate(seed_mask):
        if bool(is_seed):
            color = _random_top_color()
            top_a[c] = color
            top_b[c] = color
        else:
            top_a[c], top_b[c] = _random_mismatch_pair()

    if not np.any(top_a != BG_COLOR):
        top_a[int(_RNG.integers(0, seed_mask.shape[0]))] = _random_top_color()
    if not np.any(top_b != BG_COLOR):
        top_b[int(_RNG.integers(0, seed_mask.shape[0]))] = _random_top_color()

    grid[separator_row - 2] = top_a
    grid[separator_row - 1] = top_b


def _gen_attempt(branch: str) -> np.ndarray | None:
    top_pad = int(_RNG.integers(0, 3))
    separator_row = top_pad + 2

    if branch == "deep":
        levels = int(_RNG.integers(3, 7))
        min_h = separator_row + 3 + 2 * (levels - 1)
        height = int(_RNG.integers(min_h, min(30, min_h + 6) + 1))
        span = 2 * levels - 1
        width = int(_RNG.integers(max(8, span + 1), min(30, span + 9) + 1))
        seed_mask = _make_deep_pattern(width, levels)
    else:
        height = int(_RNG.integers(10, 25))
        width = int(_RNG.integers(8, 25))
        max_levels = 1 + (height - 1 - (separator_row + 2)) // 2
        if max_levels < 2:
            return None
        if branch == "single":
            seed_mask = _make_single_generation_pattern(width)
        elif branch == "seed_only":
            seed_mask = _make_seed_only_pattern(width)
        elif branch == "empty":
            seed_mask = np.zeros(width, dtype=bool)
        else:
            raise ValueError(f"unknown branch: {branch}")

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[separator_row] = SEPARATOR_COLOR
    _encode_top_rows(grid, separator_row, seed_mask)

    derived_seed = _seed_mask_from_top_rows(grid, separator_row, BG_COLOR)
    if not np.array_equal(seed_mask, derived_seed):
        return None

    masks = _generation_masks(seed_mask)
    if branch == "deep" and len(masks) < 3:
        return None
    if branch == "single" and len(masks) != 2:
        return None
    if branch == "seed_only" and len(masks) != 1:
        return None
    if branch == "empty" and masks:
        return None

    if separator_row + 2 * max(1, len(masks)) >= height and len(masks) > 1:
        return None

    transformed = main(grid)
    assert transformed.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    branch = _BRANCHES[_GEN_COUNTER % len(_BRANCHES)]
    _GEN_COUNTER += 1

    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate branch {branch!r} after {_GEN_ATTEMPTS} attempts")
