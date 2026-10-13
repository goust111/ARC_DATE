from __future__ import annotations

import random
from itertools import combinations, permutations

import numpy as np


BG = 0
GRAY = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7)
LINE_KINDS = ("H", "V", "D", "U")
MAX_GEN_ATTEMPTS = 2000


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    h, w = shape
    if kind == "H":
        return frozenset((value, c) for c in range(w))
    if kind == "V":
        return frozenset((r, value) for r in range(h))
    if kind == "D":
        return frozenset((r, r - value) for r in range(h) if 0 <= r - value < w)
    if kind == "U":
        return frozenset((r, value - r) for r in range(h) if 0 <= value - r < w)
    raise AssertionError(f"unknown line kind: {kind}")


def _infer_signature(coords: np.ndarray, shape: tuple[int, int]) -> tuple[str, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[tuple[str, int]] = []
    if np.all(rows == rows[0]):
        candidates.append(("H", int(rows[0])))
    if np.all(cols == cols[0]):
        candidates.append(("V", int(cols[0])))
    down = rows - cols
    if np.all(down == down[0]):
        candidates.append(("D", int(down[0])))
    up = rows + cols
    if np.all(up == up[0]):
        candidates.append(("U", int(up[0])))
    assert len(candidates) == 1
    signature = candidates[0]
    assert set(map(tuple, coords)).issubset(_line_cells(shape, *signature))
    return signature


def _find_gray_square(grid: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(grid == GRAY)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    assert side_r == side_c
    assert coords.shape[0] == side_r * side_c
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY)
    return r0, c0, side_r


def _recover_lines(grid: np.ndarray) -> dict[int, dict[str, object]]:
    lines: dict[int, dict[str, object]] = {}
    signatures: set[tuple[str, int]] = set()
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, GRAY)):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 2
        kind, value = _infer_signature(coords, grid.shape)
        assert (kind, value) not in signatures
        signatures.add((kind, value))
        lines[color] = {
            "kind": kind,
            "value": value,
            "cells": _line_cells(grid.shape, kind, value),
        }
    assert 3 <= len(lines) <= 6
    return lines


def _intersection_cell(
    shape: tuple[int, int], sig_a: tuple[str, int], sig_b: tuple[str, int]
) -> tuple[int, int] | None:
    cells = _line_cells(shape, *sig_a) & _line_cells(shape, *sig_b)
    if not cells:
        return None
    assert len(cells) == 1
    return next(iter(cells))


def _valid_orders_from_lines(
    grid: np.ndarray, lines: dict[int, dict[str, object]]
) -> tuple[list[tuple[int, ...]], dict[int, set[int]], set[tuple[int, int]]]:
    colors = list(lines)
    intersects: dict[int, set[int]] = {color: set() for color in colors}
    below_edges: set[tuple[int, int]] = set()

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            covering = [color for color, info in lines.items() if (r, c) in info["cells"]]
            visible = int(grid[r, c])
            if visible == GRAY:
                assert not covering
                continue
            if not covering:
                assert visible == BG
                continue
            assert visible in covering
            if len(covering) >= 2:
                for a, b in combinations(covering, 2):
                    intersects[a].add(b)
                    intersects[b].add(a)
                for color in covering:
                    if color != visible:
                        below_edges.add((color, visible))

    def is_valid(order: tuple[int, ...]) -> bool:
        pos = {color: idx for idx, color in enumerate(order)}
        if any(pos[below] >= pos[above] for below, above in below_edges):
            return False
        return all(order[i + 1] in intersects[order[i]] for i in range(len(order) - 1))

    return [order for order in permutations(colors) if is_valid(order)], intersects, below_edges


def _solve_details(grid: np.ndarray) -> tuple[int, tuple[int, ...], bool, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _sq_r0, _sq_c0, side = _find_gray_square(g)
    lines = _recover_lines(g)
    assert side == len(lines)
    valid_orders, _intersects, _below_edges = _valid_orders_from_lines(g, lines)
    assert len(valid_orders) == 1
    order = valid_orders[0]
    horizontal_count = sum(1 for info in lines.values() if info["kind"] == "H")
    vertical_count = sum(1 for info in lines.values() if info["kind"] == "V")
    fill_rows = horizontal_count >= vertical_count
    return side, order, fill_rows, lines


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occlusion order of full straight lines and stripe-fill the gray square with it.

    Input structure:
    - A black background contains 3-6 distinct colored one-pixel straight lines.
    - Each line is a full horizontal, vertical, or 45-degree diagonal line whose crossing pixels may be hidden by higher layers.
    - A separate solid gray square has side length equal to the number of lines and is not crossed by any line.

    Rule:
    - Reconstruct each full line from its visible collinear pixels.
    - At every crossing cell, the visible color is above every hidden line through that cell; use all such constraints to infer the unique bottom-to-top order whose consecutive lines geometrically cross.
    - Count horizontal versus vertical lines, ignoring diagonals. If horizontals are at least as many, fill the gray square by rows from top to bottom using the bottom-to-top line colors; otherwise fill it by columns from left to right.

    Key insight:
    - The output needs two independent inferences: crossing visibility determines the stripe color order, while the horizontal-versus-vertical vote determines whether those colors become row stripes or column stripes.
    """
    side, order, fill_rows, _lines = _solve_details(np.asarray(grid, dtype=int))
    out = np.zeros((side, side), dtype=int)
    for idx, color in enumerate(order):
        if fill_rows:
            out[idx, :] = int(color)
        else:
            out[:, idx] = int(color)
    return out


def _diag_down_length(size: int, value: int) -> int:
    return size - abs(value)


def _diag_up_length(size: int, value: int) -> int:
    return min(value + 1, 2 * size - 1 - value)


def _candidate_values(
    size: int, kind: str, min_len: int, square_cells: set[tuple[int, int]]
) -> list[int]:
    values: list[int] = []
    if kind == "H":
        pool = range(size)
    elif kind == "V":
        pool = range(size)
    elif kind == "D":
        pool = range(-(size - min_len), size - min_len + 1)
    elif kind == "U":
        pool = range(min_len - 1, 2 * size - min_len)
    else:
        raise AssertionError(kind)

    for value in pool:
        cells = _line_cells((size, size), kind, value)
        if kind == "D" and _diag_down_length(size, value) < min_len:
            continue
        if kind == "U" and _diag_up_length(size, value) < min_len:
            continue
        if cells & square_cells:
            continue
        values.append(int(value))
    return values


def _choose_kind_counts(n: int, fill_rows: bool) -> tuple[int, int, int, int] | None:
    options: list[tuple[int, int, int, int]] = []
    for h_count in range(1, n):
        for v_count in range(1, n - h_count + 1):
            if fill_rows and h_count < v_count:
                continue
            if not fill_rows and h_count >= v_count:
                continue
            remain = n - h_count - v_count
            for d_count in range(remain + 1):
                u_count = remain - d_count
                if max(h_count, v_count, d_count, u_count) > (n + 1) // 2:
                    continue
                options.append((h_count, v_count, d_count, u_count))
    if not options:
        return None
    options.sort(key=lambda item: (abs(item[0] - item[1]), -(item[2] + item[3])))
    best_band = [item for item in options if abs(item[0] - item[1]) <= 1]
    return random.choice(best_band or options)


def _shuffle_kinds(counts: tuple[int, int, int, int]) -> list[str] | None:
    kinds = (
        ["H"] * counts[0]
        + ["V"] * counts[1]
        + ["D"] * counts[2]
        + ["U"] * counts[3]
    )
    for _ in range(400):
        random.shuffle(kinds)
        if all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)):
            return kinds[:]
    return None


def _choose_square(size: int, side: int) -> tuple[int, int, set[tuple[int, int]]]:
    margin = 1 if size - side >= 3 else 0
    r_choices = list(range(margin, size - side - margin + 1))
    c_choices = list(range(margin, size - side - margin + 1))
    r0 = random.choice(r_choices)
    c0 = random.choice(c_choices)
    square_cells = {
        (r, c) for r in range(r0, r0 + side) for c in range(c0, c0 + side)
    }
    return r0, c0, square_cells


def _choose_signatures(
    size: int, kind_sequence: list[str], square_cells: set[tuple[int, int]], min_len: int
) -> list[tuple[str, int]] | None:
    pools = {
        kind: _candidate_values(size, kind, min_len, square_cells) for kind in LINE_KINDS
    }
    if any(not pools[kind] for kind in set(kind_sequence)):
        return None

    cell_cache = {
        (kind, value): _line_cells((size, size), kind, value)
        for kind in LINE_KINDS
        for value in pools[kind]
    }
    chosen: list[tuple[str, int]] = []
    used_values: dict[str, set[int]] = {kind: set() for kind in LINE_KINDS}
    used_cross_cells: set[tuple[int, int]] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(kind_sequence):
            return True
        kind = kind_sequence[idx]
        candidates: list[tuple[int, float, int, list[tuple[int, int]]]] = []
        for value in pools[kind]:
            if value in used_values[kind]:
                continue
            sig = (kind, value)
            cells = cell_cache[sig]
            new_cross_cells: list[tuple[int, int]] = []
            crosses_prev = idx == 0
            valid = True
            extra_crosses = 0
            for prev_idx, prev_sig in enumerate(chosen):
                inter = cell_cache[prev_sig] & cells
                if not inter:
                    continue
                assert len(inter) == 1
                cross_cell = next(iter(inter))
                if cross_cell in used_cross_cells or cross_cell in new_cross_cells:
                    valid = False
                    break
                new_cross_cells.append(cross_cell)
                if prev_idx == idx - 1:
                    crosses_prev = True
                else:
                    extra_crosses += 1
            if not crosses_prev or not valid:
                continue
            candidates.append((extra_crosses, random.random(), value, new_cross_cells))

        candidates.sort()
        for _extra, _tie, value, new_cross_cells in candidates:
            sig = (kind, value)
            chosen.append(sig)
            used_values[kind].add(value)
            used_cross_cells.update(new_cross_cells)
            if backtrack(idx + 1):
                return True
            for cell in new_cross_cells:
                used_cross_cells.remove(cell)
            used_values[kind].remove(value)
            chosen.pop()
        return False

    if not backtrack(0):
        return None
    return chosen


def _render_lines(
    size: int, signatures_in_order: list[tuple[str, int]], colors_in_order: list[int]
) -> np.ndarray:
    grid = np.full((size, size), BG, dtype=int)
    for signature, color in zip(signatures_in_order, colors_in_order):
        for r, c in _line_cells((size, size), *signature):
            grid[r, c] = int(color)
    return grid


def _expected_output(order: tuple[int, ...], fill_rows: bool) -> np.ndarray:
    side = len(order)
    out = np.zeros((side, side), dtype=int)
    for idx, color in enumerate(order):
        if fill_rows:
            out[idx, :] = int(color)
        else:
            out[:, idx] = int(color)
    return out


def _semantic_checks(
    grid: np.ndarray, intended_order: list[int], intended_fill_rows: bool
) -> bool:
    try:
        side, recovered_order, fill_rows, lines = _solve_details(grid)
    except Exception:
        return False

    if side != len(intended_order):
        return False
    if list(recovered_order) != intended_order:
        return False
    if fill_rows != intended_fill_rows:
        return False
    if not np.array_equal(main(grid), _expected_output(recovered_order, fill_rows)):
        return False
    if any(int(np.sum(grid == color)) < 2 for color in intended_order):
        return False

    valid_orders, intersects, below_edges = _valid_orders_from_lines(grid, lines)
    if len(valid_orders) != 1:
        return False
    if len(below_edges) < side - 1:
        return False
    if not all(recovered_order[i + 1] in intersects[recovered_order[i]] for i in range(side - 1)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    line_count = random.randint(3, 6)
    fill_rows = random.choice([True, False])
    counts = _choose_kind_counts(line_count, fill_rows)
    if counts is None:
        return None
    kind_sequence = _shuffle_kinds(counts)
    if kind_sequence is None:
        return None

    size_min = max(line_count + 8, 12)
    size_max = 18
    if size_min > size_max:
        return None
    size = random.randint(size_min, size_max)
    square_r0, square_c0, square_cells = _choose_square(size, line_count)
    min_len = max(line_count + 2, 6)

    signatures = _choose_signatures(size, kind_sequence, square_cells, min_len)
    if signatures is None:
        return None

    colors_in_order = random.sample(list(LINE_COLORS), line_count)
    grid = _render_lines(size, signatures, colors_in_order)
    grid[square_r0 : square_r0 + line_count, square_c0 : square_c0 + line_count] = GRAY

    if not _semantic_checks(grid, colors_in_order, fill_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/007/1")
