from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


BG = 0
BAR_COLORS = [2, 3, 4, 5, 6]
GRID_H_RANGE = (9, 13)
GRID_W_RANGE = (18, 24)
LONGEST_RANGE = (5, 7)
BAR_COUNT_RANGE = (3, 5)
GEN_RETRIES = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES_BY_SIZE = {
    2: [
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1], [1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    ],
}


def _is_straight_bar(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.unique(rows).size == 1:
        ordered = np.sort(cols)
        return bool(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)))
    if np.unique(cols).size == 1:
        ordered = np.sort(rows)
        return bool(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)))
    return False


def _square_mask(side: int, outer: int) -> np.ndarray:
    top = (outer - side) // 2
    left = (outer - side) // 2
    mask = np.zeros((outer, outer), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def _layer_masks(lengths: list[int]) -> list[np.ndarray]:
    outer = int(lengths[0])
    squares = [_square_mask(int(length), outer) for length in lengths]
    masks: list[np.ndarray] = []
    for i, sq in enumerate(squares):
        mask = sq.copy()
        if i + 1 < len(squares):
            mask &= ~squares[i + 1]
        masks.append(mask)
    return masks


def _render_layers(lengths: list[int], colors: list[int], cue_rot: dict[int, int]) -> np.ndarray:
    side = int(lengths[0])
    out = np.full((side, side), BG, dtype=int)
    masks = _layer_masks(lengths)
    for color, mask in reversed(list(zip(colors, masks))):
        rot = int(cue_rot.get(int(color), 0)) % 4
        rotated = np.rot90(mask.astype(int), -rot).astype(bool)
        out[rotated] = int(color)
    return out


def _rank_only_board(lengths: list[int], colors: list[int]) -> np.ndarray:
    side = int(lengths[0])
    out = np.full((side, side), BG, dtype=int)
    k = len(colors)
    for layer, color in enumerate(reversed(colors)):
        lo = k - 1 - layer
        hi = side - lo
        out[lo:hi, lo:hi] = int(color)
    return out


def _left_bars_and_right_cues(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    mid = W // 2

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    bars: list[tuple[int, int]] = []
    cues: dict[int, int] = {}

    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        if c1 < mid:
            assert _is_straight_bar(coords), "left-half foreground must be straight bars"
            bars.append((int(comp["area"]), color))
        elif c0 >= mid:
            assert color not in cues, "cue colors must be unique"
            cues[color] = int(comp["area"])
        else:
            raise AssertionError("foreground component crosses the left/right split")

    assert 3 <= len(bars) <= 5, "expected 3-5 bars"
    lengths = [length for length, _ in bars]
    assert len(lengths) == len(set(lengths)), "bar lengths must be distinct"
    return sorted(bars, key=lambda item: item[0], reverse=True), cues


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the left-half bars by length, turn them into nested centered squares, and rotate each color layer by its same-color right-half cue size.

    Input structure:
    - The background is black.
    - The left half contains 3-5 monochrome 4-connected straight bars with distinct colors and distinct lengths.
    - The right half contains optional monochrome 4-connected cue sprites, at most one per bar color.

    Rule:
    - Sort the left-half bars by decreasing length and let the longest length be N.
    - For each sorted bar of length L, create a centered LxL square inside an N x N board; each layer is that square minus the next smaller square, with the smallest square kept whole.
    - Match each layer color to a same-color cue on the right; if found, rotate that entire layer clockwise by (cue_size mod 4) quarter-turns inside the N x N board, then paint layers from inner to outer so later outer layers overwrite earlier inner ones.

    Key insight:
    - The bars supply both the nesting order and the exact side lengths of the square layers, while the cue sprites only contribute per-color rotation counts through their pixel totals.
    """
    bars, cues = _left_bars_and_right_cues(np.asarray(grid, dtype=int))
    lengths = [length for length, _color in bars]
    colors = [color for _length, color in bars]
    cue_rot = {int(color): int(size) % 4 for color, size in cues.items()}
    return _render_layers(lengths, colors, cue_rot)


def _bar_patch(length: int, color: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        return np.full((1, int(length)), int(color), dtype=int)
    return np.full((int(length), 1), int(color), dtype=int)


def _cue_patch(color: int, size: int) -> np.ndarray:
    shape = random.choice(SHAPES_BY_SIZE[int(size)])
    return (shape * int(color)).astype(int)


def _choose_lengths(n: int, k: int) -> list[int] | None:
    pool = list(range(1, int(n)))
    random.shuffle(pool)
    for _ in range(80):
        picked = sorted(random.sample(pool, k - 1) + [int(n)], reverse=True)
        if len(set(picked)) != k:
            continue
        masks = _layer_masks(picked)
        if any(any(not np.array_equal(mask, np.rot90(mask, turn)) for turn in (1, 2, 3)) for mask in masks):
            return picked
    return None


def _place_bars(height: int, width: int, lengths: list[int], colors: list[int]) -> np.ndarray | None:
    canvas = np.full((height, width), BG, dtype=int)
    horizontals = [bool(random.getrandbits(1)) for _ in lengths]
    if len(lengths) >= 2 and any(length > 1 for length in lengths):
        i, j = random.sample(range(len(lengths)), 2)
        horizontals[i] = True
        horizontals[j] = False

    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
    for idx in order:
        patch = _bar_patch(lengths[idx], colors[idx], horizontals[idx])
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=BG)
    return canvas


def _candidate_driver_indices(lengths: list[int]) -> list[int]:
    masks = _layer_masks(lengths)
    indices = []
    for idx, mask in enumerate(masks):
        if any(not np.array_equal(mask, np.rot90(mask, turn)) for turn in (1, 2, 3)):
            indices.append(idx)
    return indices


def _cue_plan(lengths: list[int], colors: list[int]) -> dict[int, int] | None:
    driver_indices = _candidate_driver_indices(lengths)
    if not driver_indices:
        return None

    driver = random.choice(driver_indices)
    driver_mask = _layer_masks(lengths)[driver]
    valid_driver_rots = [rot for rot in (1, 2, 3) if not np.array_equal(driver_mask, np.rot90(driver_mask, -rot))]
    if not valid_driver_rots:
        return None

    cue_rot: dict[int, int] = {int(colors[driver]): random.choice(valid_driver_rots)}
    remaining = [color for i, color in enumerate(colors) if i != driver]
    random.shuffle(remaining)

    for color in remaining:
        roll = random.random()
        if roll < 0.35:
            continue
        if roll < 0.55:
            cue_rot[int(color)] = 0
        else:
            cue_rot[int(color)] = random.choice([1, 2, 3])

    return cue_rot


def _place_cues(height: int, width: int, cue_rot: dict[int, int]) -> np.ndarray | None:
    canvas = np.full((height, width), BG, dtype=int)
    placements = list(cue_rot.items())
    random.shuffle(placements)
    for color, rot in placements:
        size = {0: 4, 1: 5, 2: 2, 3: 3}[int(rot) % 4]
        patch = _cue_patch(int(color), int(size))
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=BG)
    return canvas


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    mid = width // 2
    right_width = width - mid - 1
    if mid < 8 or right_width < 6:
        return None

    longest = random.randint(*LONGEST_RANGE)
    max_bars = min(BAR_COUNT_RANGE[1], longest)
    if max_bars < BAR_COUNT_RANGE[0]:
        return None
    bar_count = random.randint(BAR_COUNT_RANGE[0], max_bars)
    lengths = _choose_lengths(longest, bar_count)
    if lengths is None:
        return None

    colors = random.sample(BAR_COLORS, bar_count)
    cue_rot = _cue_plan(lengths, colors)
    if cue_rot is None:
        return None

    left = _place_bars(height, mid, lengths, colors)
    right = _place_cues(height, right_width, cue_rot)
    if left is None or right is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    paste(grid, left, 0, 0, transparent=BG)
    paste(grid, right, 0, mid + 1, transparent=BG)

    expected = _render_layers(lengths, colors, cue_rot)
    baseline = _render_layers(lengths, colors, {})
    rank_only = _rank_only_board(lengths, colors)
    if np.array_equal(expected, baseline):
        return None
    if np.array_equal(expected, rank_only):
        return None
    if np.count_nonzero(expected == BG) == expected.size:
        return None

    bars, cues = _left_bars_and_right_cues(grid)
    parsed_lengths = [length for length, _color in bars]
    parsed_colors = [color for _length, color in bars]
    parsed_rot = {color: size % 4 for color, size in cues.items()}
    if parsed_lengths != lengths:
        return None
    if parsed_colors != colors:
        return None
    if parsed_rot != cue_rot:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    # Extra semantic checks: exact lengths and cues must both matter on accepted samples.
    if np.array_equal(actual, _render_layers([lengths[0] - i for i in range(len(lengths))], colors, cue_rot)):
        return None
    if len(cue_rot) == len(colors) and all(rot == 0 for rot in cue_rot.values()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
