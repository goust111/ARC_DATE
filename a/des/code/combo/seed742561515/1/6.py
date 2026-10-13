from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_counts = np.sum(grid == YELLOW, axis=1)
    col_counts = np.sum(grid == YELLOW, axis=0)
    row_div = int(np.argmax(row_counts))
    col_div = int(np.argmax(col_counts))
    assert row_counts[row_div] >= w - 1
    assert col_counts[col_div] >= h - 1
    return row_div, col_div


def _quadrant_bounds(h: int, w: int, row_div: int, col_div: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, row_div, 0, col_div),
        (0, 1): (0, row_div, col_div + 1, w),
        (1, 0): (row_div + 1, h, 0, col_div),
        (1, 1): (row_div + 1, h, col_div + 1, w),
    }


def _extract_object(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> dict[str, object] | None:
    r0, r1, c0, c1 = bounds
    sub = grid[r0:r1, c0:c1]
    mask = sub != BG
    if not mask.any():
        return None
    colors = {int(v) for v in np.unique(sub[mask])}
    assert YELLOW not in colors
    assert len(colors) == 1
    comps = _components(mask)
    assert len(comps) == 1
    local = [(int(r), int(c)) for r, c in comps[0]]
    full = [(r0 + r, c0 + c) for r, c in local]
    return {"color": int(next(iter(colors))), "local": local, "full": full}


def _marker_color_on_row(grid: np.ndarray, row_div: int) -> int:
    non_yellow = [int(v) for v in grid[row_div, :] if int(v) != YELLOW]
    assert len(non_yellow) == 1
    return non_yellow[0]


def _marker_color_on_col(grid: np.ndarray, col_div: int) -> int:
    non_yellow = [int(v) for v in grid[:, col_div] if int(v) != YELLOW]
    assert len(non_yellow) == 1
    return non_yellow[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique empty quadrant by mirroring the two adjacent quadrant objects.

    The quadrant sharing the horizontal divider contributes a reflection across that
    divider and is recolored by the horizontal divider marker. The quadrant sharing
    the vertical divider contributes a reflection across that divider and is
    recolored by the vertical divider marker. Overlaps take the diagonal quadrant's
    object color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_div, col_div = _find_dividers(g)
    bounds = _quadrant_bounds(h, w, row_div, col_div)
    objects = {key: _extract_object(g, qb) for key, qb in bounds.items()}

    empty = [key for key, obj in objects.items() if obj is None]
    assert len(empty) == 1
    er, ec = empty[0]
    row_neighbor = (1 - er, ec)
    col_neighbor = (er, 1 - ec)
    diagonal = (1 - er, 1 - ec)

    row_obj = objects[row_neighbor]
    col_obj = objects[col_neighbor]
    diag_obj = objects[diagonal]
    assert row_obj is not None and col_obj is not None and diag_obj is not None

    target_r0, target_r1, target_c0, target_c1 = bounds[(er, ec)]
    h_color = _marker_color_on_row(g, row_div)
    v_color = _marker_color_on_col(g, col_div)
    overlap_color = int(diag_obj["color"])

    row_proj: set[tuple[int, int]] = set()
    for r, c in row_obj["full"]:
        nr, nc = 2 * row_div - int(r), int(c)
        if target_r0 <= nr < target_r1 and target_c0 <= nc < target_c1:
            row_proj.add((nr, nc))

    col_proj: set[tuple[int, int]] = set()
    for r, c in col_obj["full"]:
        nr, nc = int(r), 2 * col_div - int(c)
        if target_r0 <= nr < target_r1 and target_c0 <= nc < target_c1:
            col_proj.add((nr, nc))

    out = g.copy()
    for r, c in row_proj:
        out[r, c] = h_color
    for r, c in col_proj:
        out[r, c] = v_color
    for r, c in row_proj & col_proj:
        out[r, c] = overlap_color
    return out


def _grow_shape(
    h: int,
    w: int,
    target_size: int,
    seed: set[tuple[int, int]] | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    forbidden = forbidden or set()
    allowed = {(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden}
    if len(allowed) < target_size:
        return None
    if seed is None:
        if not allowed:
            return None
        filled = {random.choice(tuple(allowed))}
    else:
        filled = set(seed)
        if not filled or any(cell not in allowed for cell in filled):
            return None

    frontier = {
        (nr, nc)
        for r, c in filled
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) in allowed and (nr, nc) not in filled
    }
    while len(filled) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        filled.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in allowed and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _dilate4(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _sample_target_masks(qh: int, qw: int, force_overlap: bool) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    area = qh * qw
    if force_overlap:
        max_size = min(max(5, area // 2), area)
        for _ in range(120):
            shared_size = random.randint(1, min(3, max_size - 2))
            shared = _grow_shape(qh, qw, shared_size)
            if shared is None:
                continue
            a_size = random.randint(shared_size + 1, max_size)
            b_size = random.randint(shared_size + 1, max_size)
            a = _grow_shape(qh, qw, a_size, seed=shared)
            b = _grow_shape(qh, qw, b_size, seed=shared)
            if a is None or b is None:
                continue
            inter = a & b
            if inter and len(inter) < min(len(a), len(b)):
                return a, b
        return None

    max_size = min(max(4, area // 3), area)
    for _ in range(120):
        a_size = random.randint(3, max_size)
        a = _grow_shape(qh, qw, a_size)
        if a is None:
            continue
        forbidden = _dilate4(a, qh, qw) if random.random() < 0.65 else set(a)
        remaining = area - len(forbidden)
        if remaining < 3:
            forbidden = set(a)
        b_size = random.randint(3, max_size)
        b = _grow_shape(qh, qw, b_size, forbidden=forbidden)
        if b is not None and not (a & b):
            return a, b
    return None


def _to_full(bounds: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, _, c0, _ = bounds
    return {(r0 + r, c0 + c) for r, c in cells}


def _mirror_across_row(row_div: int, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(2 * row_div - r, c) for r, c in cells}


def _mirror_across_col(col_div: int, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, 2 * col_div - c) for r, c in cells}


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _expected_output(
    inp: np.ndarray,
    row_target: set[tuple[int, int]],
    col_target: set[tuple[int, int]],
    h_marker: int,
    v_marker: int,
    overlap_color: int,
) -> np.ndarray:
    out = inp.copy()
    _paint(out, row_target, h_marker)
    _paint(out, col_target, v_marker)
    _paint(out, row_target & col_target, overlap_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(4, 8)
    qw = random.randint(4, 8)
    h = 2 * qh + 1
    w = 2 * qw + 1
    row_div = qh
    col_div = qw
    bounds = _quadrant_bounds(h, w, row_div, col_div)

    er, ec = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    row_neighbor = (1 - er, ec)
    col_neighbor = (er, 1 - ec)
    diagonal = (1 - er, 1 - ec)
    force_overlap = random.choice([False, True])

    masks = _sample_target_masks(qh, qw, force_overlap)
    if masks is None:
        return None
    row_local, col_local = masks
    diag_size = random.randint(2, min(max(2, qh * qw // 3), max(2, qh * qw - 1)))
    diag_local = _grow_shape(qh, qw, diag_size)
    if diag_local is None:
        return None

    colors = random.sample(PALETTE, 5)
    row_obj_color, col_obj_color, diag_obj_color, h_marker, v_marker = colors

    empty_bounds = bounds[(er, ec)]
    row_target = _to_full(empty_bounds, row_local)
    col_target = _to_full(empty_bounds, col_local)

    row_source = _mirror_across_row(row_div, row_target)
    col_source = _mirror_across_col(col_div, col_target)
    diag_source = _to_full(bounds[diagonal], diag_local)

    inp = np.zeros((h, w), dtype=int)
    inp[row_div, :] = YELLOW
    inp[:, col_div] = YELLOW

    h_marker_c = random.choice([c for c in range(w) if c != col_div])
    v_marker_r = random.choice([r for r in range(h) if r != row_div])
    inp[row_div, h_marker_c] = h_marker
    inp[v_marker_r, col_div] = v_marker

    _paint(inp, row_source, row_obj_color)
    _paint(inp, col_source, col_obj_color)
    _paint(inp, diag_source, diag_obj_color)

    expected = _expected_output(inp, row_target, col_target, h_marker, v_marker, diag_obj_color)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
