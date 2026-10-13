from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
YELLOW = Color.YELLOW
TEAL = Color.TEAL

_MAX_ATTEMPTS = 300
_MIN_PANEL_W = 7
_MAX_PANEL_W = 9
_MIN_EXTRA_ROWS = 1
_MAX_EXTRA_ROWS = 4
_GATE_PAIRS = ((BLUE, BLUE), (BLUE, RED), (RED, BLUE), (RED, RED))
_RNG = random.Random()


def _apply_gate_rows(a: np.ndarray, b: np.ndarray, gate_colors: np.ndarray) -> np.ndarray:
    use_union = np.asarray(gate_colors, dtype=int) == BLUE
    return np.where(use_union[:, None], np.logical_or(a, b), np.logical_xor(a, b))


def _fold_left_to_right(
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    div1: np.ndarray,
    div2: np.ndarray,
) -> np.ndarray:
    mid_fold = _apply_gate_rows(left, mid, div1)
    return _apply_gate_rows(mid_fold, right, div2)


def _fold_right_to_left(
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    div1: np.ndarray,
    div2: np.ndarray,
) -> np.ndarray:
    mid_fold = _apply_gate_rows(mid, right, div2)
    return _apply_gate_rows(left, mid_fold, div1)


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    allowed = {BLACK, BLUE, RED, YELLOW}
    assert set(colors_present(g)) <= allowed, "input must use only black/yellow panels and blue/red dividers"

    divider_cols: list[int] = []
    for c in range(g.shape[1]):
        col_colors = set(map(int, g[:, c]))
        if col_colors and col_colors <= {BLUE, RED}:
            divider_cols.append(int(c))

    assert len(divider_cols) == 2, "expected exactly two full-height blue/red divider columns"
    d1, d2 = divider_cols
    panel_w = d1
    assert panel_w > 0, "left panel must be non-empty"
    assert d2 - d1 - 1 == panel_w, "middle panel width must match left panel width"
    assert g.shape[1] - d2 - 1 == panel_w, "right panel width must match left panel width"

    left = g[:, :panel_w]
    mid = g[:, d1 + 1 : d2]
    right = g[:, d2 + 1 :]
    for panel in (left, mid, right):
        assert set(colors_present(panel)) <= {BLACK, YELLOW}, "panels must be binary black/yellow masks"

    div1 = g[:, d1]
    div2 = g[:, d2]
    assert set(map(int, div1)) <= {BLUE, RED}
    assert set(map(int, div2)) <= {BLUE, RED}
    return left == YELLOW, mid == YELLOW, right == YELLOW, div1.astype(int), div2.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold three equal binary panels left-to-right through two per-row red/blue operation gates.

    Input structure:
    - The grid is arranged as panel, divider column, panel, divider column, panel.
    - The three panels have equal size and use only black and yellow.
    - Each divider is a single full-height column of blue/red cells that may vary by row.

    Rule:
    - Work row by row and return one panel-sized output grid.
    - First combine the left and middle rows using the first divider row: blue means union, red means symmetric difference.
    - Then combine that intermediate row with the right row using the second divider row with the same color-to-operation mapping.
    - Paint true cells teal and false cells black in the compact output.

    Key insight:
    - The divider columns are two serial row-wise instruction gates, so the solution depends on discovering the left-to-right two-step fold rather than a single three-layer overlay.
    """
    left, mid, right, div1, div2 = _parse_grid(grid)
    out_mask = _fold_left_to_right(left, mid, right, div1, div2)
    return np.where(out_mask, TEAL, BLACK).astype(int)


def _mask_from_segments(width: int, segments: list[tuple[int, int]]) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    for start, length in segments:
        row[start : start + length] = True
    return row


def _sample_segments(width: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count == 0:
        return []
    for _ in range(200):
        blocked = np.zeros(width, dtype=bool)
        segments: list[tuple[int, int]] = []
        lengths = [rng.randint(1, 2) for _ in range(count)]
        lengths.sort(reverse=True)
        ok = True
        for length in lengths:
            candidates = [
                start
                for start in range(width - length + 1)
                if not blocked[start : start + length].any()
            ]
            if not candidates:
                ok = False
                break
            start = rng.choice(candidates)
            segments.append((int(start), int(length)))
            blocked[max(0, start - 1) : min(width, start + length + 1)] = True
        if ok:
            segments.sort()
            return segments
    raise RuntimeError("failed to sample disjoint row segments")


def _sample_mask(width: int, rng: random.Random, count: int) -> np.ndarray:
    return _mask_from_segments(width, _sample_segments(width, count, rng))


def _random_triplet(width: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool_count = rng.randint(2, 3)
    pool_masks = [_mask_from_segments(width, [seg]) for seg in _sample_segments(width, pool_count, rng)]
    panels: list[np.ndarray] = []
    for _ in range(3):
        row = np.zeros(width, dtype=bool)
        for mask in pool_masks:
            if rng.random() < 0.45:
                row |= mask
        if not row.any() and rng.random() < 0.75:
            row |= rng.choice(pool_masks)
        panels.append(row)
    if not any(panel.any() for panel in panels):
        panels[rng.randrange(3)] = rng.choice(pool_masks).copy()
    if np.array_equal(panels[0], panels[1]) and np.array_equal(panels[1], panels[2]):
        panels[2] = np.logical_xor(panels[2], rng.choice(pool_masks))
    return panels[0], panels[1], panels[2]


def _special_blocks(width: int, rng: random.Random) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]]:
    empty = np.zeros(width, dtype=bool)

    gate1_a = _sample_mask(width, rng, 1)
    gate1_b = _sample_mask(width, rng, 1)
    while np.any(gate1_a & gate1_b):
        gate1_b = _sample_mask(width, rng, 1)
    gate1_left = gate1_a
    gate1_mid = gate1_a | gate1_b
    gate1_right = empty.copy()
    gate1_block = [
        (gate1_left.copy(), gate1_mid.copy(), gate1_right.copy(), BLUE, BLUE),
        (gate1_left.copy(), gate1_mid.copy(), gate1_right.copy(), RED, BLUE),
    ]

    gate2_a = _sample_mask(width, rng, 1)
    gate2_b = _sample_mask(width, rng, 1)
    while np.any(gate2_a & gate2_b):
        gate2_b = _sample_mask(width, rng, 1)
    gate2_left = empty.copy()
    gate2_mid = gate2_a | gate2_b
    gate2_right = gate2_a
    gate2_block = [
        (gate2_left.copy(), gate2_mid.copy(), gate2_right.copy(), BLUE, BLUE),
        (gate2_left.copy(), gate2_mid.copy(), gate2_right.copy(), BLUE, RED),
    ]

    order_a = _sample_mask(width, rng, 1)
    order_block = [
        (order_a.copy(), order_a.copy(), order_a.copy(), RED, BLUE),
    ]

    redred_a = _sample_mask(width, rng, 1)
    redred_b = _sample_mask(width, rng, 1)
    while np.any(redred_a & redred_b):
        redred_b = _sample_mask(width, rng, 1)
    redred_block = [
        (redred_a.copy(), (redred_a | redred_b).copy(), redred_b.copy(), RED, RED),
    ]

    return [gate1_block, gate2_block, order_block, redred_block]


def _build_grid(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]],
    width: int,
) -> np.ndarray:
    height = len(rows)
    left = np.vstack([row[0] for row in rows]).astype(bool)
    mid = np.vstack([row[1] for row in rows]).astype(bool)
    right = np.vstack([row[2] for row in rows]).astype(bool)
    div1 = np.array([row[3] for row in rows], dtype=int)
    div2 = np.array([row[4] for row in rows], dtype=int)

    grid = np.full((height, 3 * width + 2), BLACK, dtype=int)
    grid[:, :width] = np.where(left, YELLOW, BLACK)
    grid[:, width] = div1
    grid[:, width + 1 : 2 * width + 1] = np.where(mid, YELLOW, BLACK)
    grid[:, 2 * width + 1] = div2
    grid[:, 2 * width + 2 :] = np.where(right, YELLOW, BLACK)
    return grid


def _has_gate1_contrast(
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    div1: np.ndarray,
    div2: np.ndarray,
    out_mask: np.ndarray,
) -> bool:
    for i in range(left.shape[0]):
        for j in range(i + 1, left.shape[0]):
            same_panels = (
                np.array_equal(left[i], left[j])
                and np.array_equal(mid[i], mid[j])
                and np.array_equal(right[i], right[j])
            )
            if same_panels and int(div2[i]) == int(div2[j]) and int(div1[i]) != int(div1[j]):
                if not np.array_equal(out_mask[i], out_mask[j]):
                    return True
    return False


def _has_gate2_contrast(
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    div1: np.ndarray,
    div2: np.ndarray,
    out_mask: np.ndarray,
) -> bool:
    for i in range(left.shape[0]):
        for j in range(i + 1, left.shape[0]):
            same_panels = (
                np.array_equal(left[i], left[j])
                and np.array_equal(mid[i], mid[j])
                and np.array_equal(right[i], right[j])
            )
            if same_panels and int(div1[i]) == int(div1[j]) and int(div2[i]) != int(div2[j]):
                if not np.array_equal(out_mask[i], out_mask[j]):
                    return True
    return False


def _validate_candidate(grid: np.ndarray) -> bool:
    left, mid, right, div1, div2 = _parse_grid(grid)
    out_mask = _fold_left_to_right(left, mid, right, div1, div2)

    if not np.array_equal(main(grid), np.where(out_mask, TEAL, BLACK).astype(int)):
        return False
    if set(map(int, div1)) != {BLUE, RED}:
        return False
    if set(map(int, div2)) != {BLUE, RED}:
        return False
    if set(zip(div1.tolist(), div2.tolist())) != set(_GATE_PAIRS):
        return False
    if not _has_gate1_contrast(left, mid, right, div1, div2, out_mask):
        return False
    if not _has_gate2_contrast(left, mid, right, div1, div2, out_mask):
        return False
    if not np.any(out_mask != np.logical_or(np.logical_or(left, mid), right)):
        return False
    if not np.any(out_mask != np.logical_xor(np.logical_xor(left, mid), right)):
        return False
    if not np.any(out_mask != _fold_right_to_left(left, mid, right, div1, div2)):
        return False
    if not np.any(out_mask != _fold_left_to_right(np.zeros_like(left), mid, right, div1, div2)):
        return False
    if not np.any(out_mask != _fold_left_to_right(left, np.zeros_like(mid), right, div1, div2)):
        return False
    if not np.any(out_mask != _fold_left_to_right(left, mid, np.zeros_like(right), div1, div2)):
        return False
    if not any(int(div1[r]) == RED and np.any(left[r] & mid[r]) for r in range(left.shape[0])):
        return False
    intermediate = _apply_gate_rows(left, mid, div1)
    if not any(int(div2[r]) == RED and np.any(intermediate[r] & right[r]) for r in range(left.shape[0])):
        return False
    if not out_mask.any() or np.all(out_mask):
        return False
    for panel in (left, mid, right):
        density = float(panel.mean())
        if not (0.08 <= density <= 0.70):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(_MIN_PANEL_W, _MAX_PANEL_W)
    blocks = _special_blocks(width, rng)
    extra_rows = rng.randint(_MIN_EXTRA_ROWS, _MAX_EXTRA_ROWS)
    for _ in range(extra_rows):
        left, mid, right = _random_triplet(width, rng)
        gate1, gate2 = rng.choice(_GATE_PAIRS)
        blocks.append([(left, mid, right, gate1, gate2)])
    rng.shuffle(blocks)
    rows = [row for block in blocks for row in block]
    grid = _build_grid(rows, width)
    return grid if _validate_candidate(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-gate fold input")
