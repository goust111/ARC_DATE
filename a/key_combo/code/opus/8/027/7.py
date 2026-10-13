from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7)

PANEL_H_RANGE = (6, 10)
PANEL_W_RANGE = (6, 10)
EXTRA_OVERLAP_RANGE = (1, 5)
EXTRA_LEFT_ONLY_RANGE = (1, 4)
EXTRA_RIGHT_ONLY_RANGE = (1, 4)
GEN_ATTEMPTS = 400


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in pairs:
        if current == int(src):
            current = int(dst)
    return current


def _parse_legend_pairs(row: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(v) for v in row]
    pairs: list[tuple[int, int]] = []
    c = 0
    width = len(vals)
    while c < width:
        if vals[c] == BG:
            c += 1
            continue
        assert c + 1 < width and vals[c + 1] != BG, "legend pair must be two adjacent non-background cells"
        pairs.append((vals[c], vals[c + 1]))
        c += 2
        if c < width and vals[c] != BG:
            raise AssertionError("legend pairs must be separated by one background cell")
        c += 1
    assert 2 <= len(pairs) <= 3, "expected 2-3 legend pairs"
    return pairs


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[2:, c] == SEPARATOR)]
    assert len(candidates) == 1, "expected exactly one vertical separator column"
    return int(candidates[0])


def _solve_panels(left: np.ndarray, right: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    assert left.shape == right.shape
    out = np.zeros_like(left)
    h, w = left.shape
    for r in range(h):
        for c in range(w):
            lv = int(left[r, c])
            rv = int(right[r, c])
            if lv == BG and rv == BG:
                continue
            if lv != BG and rv != BG:
                out[r, c] = _apply_chain(lv, pairs)
            else:
                out[r, c] = lv if lv != BG else rv
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two aligned side panels and recolor only their intersections by an ordered legend chain.

    Input structure:
    - Row 0 is a horizontal legend made of 2-3 adjacent color pairs, with one black gap between pairs.
    - Row 1 is a full grey separator row.
    - Rows 2 onward contain two equal-size panels split by one full grey separator column.

    Rule:
    - Remove the legend row, the grey separator row, and the grey separator column.
    - Align the left and right panels by matching relative coordinates.
    - Output background for double-background cells, keep single-panel foreground unchanged, and for overlap cells apply the ordered chain substitution to the left-panel color only.

    Key insight:
    - The legend does not recolor everything: it matters only where both panels occupy the same aligned position, and the legend row order determines the final color there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 9
    assert np.all(g[1, :] == SEPARATOR), "expected a full grey separator row"

    pairs = _parse_legend_pairs(g[0])
    sep_col = _find_separator_col(g)

    left = g[2:, :sep_col]
    right = g[2:, sep_col + 1 :]
    assert left.shape == right.shape and left.size > 0, "panels must have equal non-empty shapes"
    assert not np.any(left == SEPARATOR) and not np.any(right == SEPARATOR), "separator color must not appear inside panels"
    return _solve_panels(left, right, pairs)


def _build_pairs() -> tuple[list[tuple[int, int]], dict[str, object]]:
    colors = random.sample(PALETTE, 5)
    a, b, c, x, y = colors
    chain_pairs = [(a, b), (b, c)]
    indep_pair = (x, y)
    if random.random() < 0.5:
        pairs = chain_pairs + [indep_pair]
        chain_indices = (0, 1)
    else:
        pairs = [indep_pair] + chain_pairs
        chain_indices = (1, 2)
    meta = {
        "chain_start": a,
        "chain_mid": b,
        "chain_final": c,
        "indep_src": x,
        "indep_dst": y,
        "sources": (a, b, x),
        "chain_indices": chain_indices,
    }
    return pairs, meta


def _sample_free_coord(h: int, w: int, used: set[tuple[int, int]]) -> tuple[int, int]:
    choices = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used]
    if not choices:
        raise ValueError("panel is full")
    return random.choice(choices)


def _place_required_cells(
    left: np.ndarray,
    right: np.ndarray,
    meta: dict[str, object],
) -> set[tuple[int, int]]:
    a = int(meta["chain_start"])
    b = int(meta["chain_mid"])
    x = int(meta["indep_src"])

    used: set[tuple[int, int]] = set()

    p = _sample_free_coord(*left.shape, used)
    used.add(p)
    left[p] = a
    right[p] = x

    p = _sample_free_coord(*left.shape, used)
    used.add(p)
    left[p] = b
    right[p] = a

    p = _sample_free_coord(*left.shape, used)
    used.add(p)
    left[p] = x
    right[p] = a

    p = _sample_free_coord(*left.shape, used)
    used.add(p)
    left[p] = a

    p = _sample_free_coord(*left.shape, used)
    used.add(p)
    right[p] = x

    return used


def _add_random_cells(
    left: np.ndarray,
    right: np.ndarray,
    used: set[tuple[int, int]],
    sources: tuple[int, int, int],
) -> None:
    extra_overlap = random.randint(*EXTRA_OVERLAP_RANGE)
    extra_left_only = random.randint(*EXTRA_LEFT_ONLY_RANGE)
    extra_right_only = random.randint(*EXTRA_RIGHT_ONLY_RANGE)

    for _ in range(extra_overlap):
        p = _sample_free_coord(*left.shape, used)
        used.add(p)
        left[p] = random.choice(sources)
        right[p] = random.choice(sources)

    for _ in range(extra_left_only):
        p = _sample_free_coord(*left.shape, used)
        used.add(p)
        left[p] = random.choice(sources)

    for _ in range(extra_right_only):
        p = _sample_free_coord(*left.shape, used)
        used.add(p)
        right[p] = random.choice(sources)


def _build_grid(
    pairs: list[tuple[int, int]],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    panel_h, panel_w = left.shape
    grid = np.zeros((panel_h + 2, panel_w * 2 + 1), dtype=int)
    legend_width = 2 * len(pairs) + (len(pairs) - 1)
    start = random.randint(0, grid.shape[1] - legend_width)
    cursor = start
    for i, (src, dst) in enumerate(pairs):
        grid[0, cursor] = int(src)
        grid[0, cursor + 1] = int(dst)
        cursor += 2
        if i != len(pairs) - 1:
            cursor += 1
    grid[1, :] = SEPARATOR
    grid[2:, :panel_w] = left
    grid[2:, panel_w] = SEPARATOR
    grid[2:, panel_w + 1 :] = right
    return grid


def _swap_chain_order(pairs: list[tuple[int, int]], chain_indices: tuple[int, int]) -> list[tuple[int, int]]:
    swapped = list(pairs)
    i, j = map(int, chain_indices)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped


def _validate_semantics(
    pairs: list[tuple[int, int]],
    left: np.ndarray,
    right: np.ndarray,
    meta: dict[str, object],
) -> bool:
    overlap = (left != BG) & (right != BG)
    left_only = (left != BG) & (right == BG)
    right_only = (left == BG) & (right != BG)
    if int(overlap.sum()) < 3 or not np.any(left_only) or not np.any(right_only):
        return False

    out = _solve_panels(left, right, pairs)

    if not np.any(overlap & (out != left)):
        return False

    left_only_changed = False
    for r, c in np.argwhere(left_only):
        if _apply_chain(int(left[r, c]), pairs) != int(left[r, c]):
            left_only_changed = True
            break
    if not left_only_changed:
        return False

    alt_right_source = np.zeros_like(out)
    h, w = left.shape
    for r in range(h):
        for c in range(w):
            lv = int(left[r, c])
            rv = int(right[r, c])
            if lv == BG and rv == BG:
                continue
            if lv != BG and rv != BG:
                alt_right_source[r, c] = _apply_chain(rv, pairs)
            else:
                alt_right_source[r, c] = lv if lv != BG else rv
    if np.array_equal(out, alt_right_source):
        return False

    swapped = _swap_chain_order(pairs, meta["chain_indices"])
    if np.array_equal(out, _solve_panels(left, right, swapped)):
        return False

    probe_cells = np.argwhere(overlap & (out != left))
    if probe_cells.size == 0:
        return False
    pr, pc = map(int, random.choice(probe_cells.tolist()))
    altered_right = right.copy()
    altered_right[pr, pc] = BG
    if np.array_equal(out, _solve_panels(left, altered_right, pairs)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)

    pairs, meta = _build_pairs()
    left = np.zeros((panel_h, panel_w), dtype=int)
    right = np.zeros((panel_h, panel_w), dtype=int)

    try:
        used = _place_required_cells(left, right, meta)
        _add_random_cells(left, right, used, meta["sources"])
    except ValueError:
        return None

    if not _validate_semantics(pairs, left, right, meta):
        return None

    grid = _build_grid(pairs, left, right)
    if not np.array_equal(main(grid), _solve_panels(left, right, pairs)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
