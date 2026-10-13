from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
import random

import numpy as np


GRID_CELLS = 5
CELL_INNER = 2
CELL_STEP = CELL_INNER + 1
GRID_SIZE = GRID_CELLS * CELL_STEP + 1
MAX_ATTEMPTS = 1200
VISIBLE_COUNT_OPTIONS = ((2, 3), (2, 4), (3, 4))
COLOR_POOL = tuple(range(1, 10))


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.ravel())


ALL_MASKS_2X2 = tuple(
    np.array(
        [[(bits >> 0) & 1, (bits >> 1) & 1], [(bits >> 2) & 1, (bits >> 3) & 1]],
        dtype=bool,
    )
    for bits in range(1, 16)
)
MASKS_WITH_3_OR_4 = tuple(mask for mask in ALL_MASKS_2X2 if 3 <= int(mask.sum()) <= 4)
MAJORITY_MASKS = tuple(mask for mask in ALL_MASKS_2X2 if 1 <= int(mask.sum()) <= 3)


def _infer_cell_grid_params(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    line = int(grid[0, 0])
    row_divs = [r for r in range(h) if np.all(grid[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(grid[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2
    assert row_divs[0] == 0 and col_divs[0] == 0
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    step = reduce(gcd, drs + dcs)
    assert step > 1
    assert (h - 1) % step == 0 and (w - 1) % step == 0
    return int(line), int(step)


def _cell_origin(r: int, c: int, step: int) -> tuple[int, int]:
    return r * step + 1, c * step + 1


def _extract_cells(grid: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    n_rows = (grid.shape[0] - 1) // step
    n_cols = (grid.shape[1] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_keys: list[list[tuple[int, ...]]] = []
    for r in range(n_rows):
        row_keys: list[tuple[int, ...]] = []
        for c in range(n_cols):
            rr, cc = _cell_origin(r, c, step)
            cell = grid[rr : rr + inner, cc : cc + inner].copy()
            key = tuple(int(x) for x in cell.ravel())
            patterns.setdefault(key, cell)
            row_keys.append(key)
        cell_keys.append(row_keys)
    return patterns, cell_keys


def _positions_of(cell_keys: list[list[tuple[int, ...]]], target: tuple[int, ...]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, row in enumerate(cell_keys):
        for c, key in enumerate(row):
            if key == target:
                out.add((r, c))
    return out


def _placed_active_positions(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        out.add((top + int(r), left + int(c)))
    return out


def _candidate_offsets(
    mask: np.ndarray, visible_positions: set[tuple[int, int]], n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n_rows - mh + 1):
        for left in range(n_cols - mw + 1):
            active = _placed_active_positions(mask, top, left)
            if visible_positions <= active:
                candidates.append((top, left))
    return candidates


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    line, step = _infer_cell_grid_params(g)
    n_rows = (g.shape[0] - 1) // step
    n_cols = (g.shape[1] - 1) // step
    assert n_rows == n_cols == GRID_CELLS
    assert step - 1 == CELL_INNER

    patterns, cell_keys = _extract_cells(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_keys:
        for key in row:
            counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 3, "expected majority plus two minority patterns"

    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    a_key, a_count = ordered[0]
    b_key, b_count = ordered[1]
    m_key, m_count = ordered[2]
    assert a_count < b_count < m_count

    a_pattern = patterns[a_key]
    b_pattern = patterns[b_key]
    m_pattern = patterns[m_key]
    a_mask = a_pattern != 0
    b_mask = b_pattern != 0
    assert np.any(a_mask) and np.any(b_mask)

    a_visible = _positions_of(cell_keys, a_key)
    b_visible = _positions_of(cell_keys, b_key)
    assert len(a_visible) == a_count and len(b_visible) == b_count

    a_candidates = _candidate_offsets(a_mask, a_visible, n_rows, n_cols)
    b_candidates = _candidate_offsets(b_mask, b_visible, n_rows, n_cols)
    assert len(a_candidates) == 1, "A blueprint placement must be unique"
    assert len(b_candidates) == 1, "B blueprint placement must be unique"
    a_offset = a_candidates[0]
    b_offset = b_candidates[0]
    a_full = _placed_active_positions(a_mask, *a_offset)
    b_full = _placed_active_positions(b_mask, *b_offset)

    return {
        "grid": g,
        "line": line,
        "step": step,
        "patterns": patterns,
        "cell_keys": cell_keys,
        "counts": counts,
        "a_key": a_key,
        "b_key": b_key,
        "m_key": m_key,
        "a_count": a_count,
        "b_count": b_count,
        "m_count": m_count,
        "a_pattern": a_pattern,
        "b_pattern": b_pattern,
        "m_pattern": m_pattern,
        "a_mask": a_mask,
        "b_mask": b_mask,
        "a_visible": a_visible,
        "b_visible": b_visible,
        "a_full": a_full,
        "b_full": b_full,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale two minority cell blueprints to the full cell grid and resolve their overlap by rarity.

    Input structure:
    - A separator color fills full divider rows and columns, splitting the grid into a 5x5 array of cells.
    - Each cell has a 2x2 interior pattern, with one majority pattern and two distinct minority patterns.
    - The rarer minority pattern appears fewer times than the other minority pattern.

    Rule:
    - Identify the majority pattern and the two minority patterns by cell-frequency.
    - For each minority pattern, treat its 2x2 interior as a binary blueprint where non-zero pixels are active.
    - Place each blueprint on the 5x5 cell array at its unique offset so that all visible cells of that minority type land on active positions.
    - Fill every cell marked only by A with the rarer minority pattern, every cell marked only by B with the other minority pattern, overlapping cells with the rarer pattern, and all remaining cells with the majority pattern.

    Key insight:
    - Each minority cell is a miniature occupancy map for a larger cell-grid blueprint, and the overlapped blueprint cells are resolved by comparing which minority pattern is globally rarer in the input.
    """
    info = _solve_details(grid)
    out = info["grid"].copy()
    step = info["step"]

    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            if (r, c) in info["a_full"]:
                pattern = info["a_pattern"]
            elif (r, c) in info["b_full"]:
                pattern = info["b_pattern"]
            else:
                pattern = info["m_pattern"]
            rr, cc = _cell_origin(r, c, step)
            out[rr : rr + CELL_INNER, cc : cc + CELL_INNER] = pattern
    return out


def _pattern_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, 0).astype(int)


def _render_grid(
    separator: int,
    majority: np.ndarray,
    a_pattern: np.ndarray,
    b_pattern: np.ndarray,
    a_positions: set[tuple[int, int]],
    b_positions: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            pattern = majority
            if (r, c) in b_positions:
                pattern = b_pattern
            if (r, c) in a_positions:
                pattern = a_pattern
            rr, cc = _cell_origin(r, c, CELL_STEP)
            grid[rr : rr + CELL_INNER, cc : cc + CELL_INNER] = pattern
    return grid


def _sample_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_mask = random.choice(MASKS_WITH_3_OR_4).copy()
    b_mask = random.choice(MASKS_WITH_3_OR_4).copy()
    majority_choices = [mask for mask in MAJORITY_MASKS if _mask_key(mask) not in {_mask_key(a_mask), _mask_key(b_mask)}]
    majority_mask = random.choice(majority_choices).copy()
    return majority_mask, a_mask, b_mask


def _sample_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in coord) for coord in np.argwhere(mask)]


def _shuffle(seq):
    seq = list(seq)
    random.shuffle(seq)
    return seq


def _choose_visible_sets(
    a_full: set[tuple[int, int]], b_full: set[tuple[int, int]], target_counts: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    overlap = a_full & b_full
    a_only = a_full - b_full
    b_only = b_full - a_full
    if not overlap or not a_only or not b_only:
        return None

    configs_by_count: dict[tuple[int, int], list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]] = {}
    hidden_non_overlap_options = _shuffle(a_only | b_only)
    count_options = [target_counts]
    conflict_modes = _shuffle(("b_visible", "hidden"))
    overlap_cells = _shuffle(overlap)

    for hidden_extra in hidden_non_overlap_options:
        for a_vis_count, b_vis_count in count_options:
            if not (2 <= a_vis_count < b_vis_count <= 4):
                continue
            if len(a_full) <= a_vis_count or len(b_full) < b_vis_count:
                continue
            for conflict_cell in overlap_cells:
                for mode in conflict_modes:
                    a_pool = a_full - {conflict_cell}
                    if len(a_pool) < a_vis_count:
                        continue
                    for a_combo in combinations(sorted(a_pool), a_vis_count):
                        a_visible = set(a_combo)
                        if hidden_extra in a_full and hidden_extra in a_visible:
                            continue

                        b_base = {conflict_cell} if mode == "b_visible" else set()
                        excluded = {conflict_cell} if mode == "hidden" else set()
                        b_pool = b_full - a_visible - excluded - b_base
                        need = b_vis_count - len(b_base)
                        if need < 0 or len(b_pool) < need:
                            continue
                        for b_combo in combinations(sorted(b_pool), need):
                            b_visible = b_base | set(b_combo)
                            if a_visible & b_visible:
                                continue
                            if hidden_extra in b_full and hidden_extra in b_visible:
                                continue
                            if not ((a_full - a_visible) | (b_full - b_visible)):
                                continue
                            if conflict_cell in a_visible:
                                continue
                            if mode == "hidden" and conflict_cell in b_visible:
                                continue
                            key = (len(a_visible), len(b_visible))
                            configs_by_count.setdefault(key, []).append((a_visible, b_visible))
    if not configs_by_count:
        return None
    return random.choice(configs_by_count[target_counts])


def _semantic_instance_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False
    try:
        info = _solve_details(inp)
    except Exception:
        return False

    overlap = info["a_full"] & info["b_full"]
    a_only = info["a_full"] - info["b_full"]
    b_only = info["b_full"] - info["a_full"]
    if not overlap or not a_only or not b_only:
        return False

    changed_overlap = [
        pos
        for pos in overlap
        if info["cell_keys"][pos[0]][pos[1]] != info["a_key"]
    ]
    if not changed_overlap:
        return False

    changed_non_overlap = [
        pos
        for pos in (a_only | b_only)
        if info["cell_keys"][pos[0]][pos[1]] == info["m_key"]
    ]
    if not changed_non_overlap:
        return False

    return np.array_equal(main(inp), expected)


def _generate_attempt() -> np.ndarray | None:
    target_counts = random.choice(VISIBLE_COUNT_OPTIONS)
    separator, maj_color, a_color, b_color = random.sample(COLOR_POOL, 4)
    majority_mask, a_mask, b_mask = _sample_masks()
    majority = _pattern_from_mask(majority_mask, maj_color)
    a_pattern = _pattern_from_mask(a_mask, a_color)
    b_pattern = _pattern_from_mask(b_mask, b_color)

    offsets = [(r, c) for r in range(GRID_CELLS - CELL_INNER + 1) for c in range(GRID_CELLS - CELL_INNER + 1)]
    random.shuffle(offsets)
    for a_top, a_left in offsets:
        a_full = _placed_active_positions(a_mask, a_top, a_left)
        for b_top, b_left in _shuffle(offsets):
            b_full = _placed_active_positions(b_mask, b_top, b_left)
            chosen = _choose_visible_sets(a_full, b_full, target_counts)
            if chosen is None:
                continue
            a_visible, b_visible = chosen

            inp = _render_grid(separator, majority, a_pattern, b_pattern, a_visible, b_visible)
            expected = _render_grid(separator, majority, a_pattern, b_pattern, a_full, b_full)
            if not _semantic_instance_ok(inp, expected):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
