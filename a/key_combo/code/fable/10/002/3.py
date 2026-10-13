from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE = list(range(10))
NON_GREY = [c for c in PALETTE if c != GREY]
COLOR_DOTS = [c for c in PALETTE if c not in (GREY,)]

GEN_RETRIES = 500
GEN_SEMANTIC_CHECKS = 12


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one divider color"
    return candidates[0]


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = 0
    for d in dividers:
        if start < d:
            segs.append((start, d))
        start = d + 1
    if start < length:
        segs.append((start, length))
    assert segs
    return segs


def _divider_grid_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = _get_bg(grid)
    divider, divider_rows, divider_cols = _find_dividers(grid, bg)
    row_segs = _segments(grid.shape[0], divider_rows)
    col_segs = _segments(grid.shape[1], divider_cols)
    row_sizes = {r1 - r0 for r0, r1 in row_segs}
    col_sizes = {c1 - c0 for c0, c1 in col_segs}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "cells must have equal size"
    return bg, divider, row_segs, col_segs


def _locate_template(
    grid: np.ndarray, row_segs: list[tuple[int, int]], col_segs: list[tuple[int, int]]
) -> tuple[tuple[int, int], np.ndarray, int]:
    template_info = None
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = grid[r0:r1, c0:c1]
            grey_mask = cell == GREY
            if not grey_mask.any():
                continue
            comps = _components(grey_mask)
            large = [comp for comp in comps if len(comp) > 1]
            singles = [comp for comp in comps if len(comp) == 1]
            assert len(large) == 1, "template cell must contain one multi-pixel grey component"
            assert template_info is None, "template cell must be unique"
            template_mask = np.zeros_like(cell, dtype=bool)
            for rr, cc in large[0]:
                template_mask[rr, cc] = True
            template_info = ((ri, ci), template_mask, len(singles))
    assert template_info is not None, "missing template cell"
    return template_info


def _stamp_template(
    grid: np.ndarray,
    row_segs: list[tuple[int, int]],
    col_segs: list[tuple[int, int]],
    template_rc: tuple[int, int],
    template_mask: np.ndarray,
    baseline: int,
) -> np.ndarray:
    out = grid.copy()
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = out[r0:r1, c0:c1]
            if (ri, ci) == template_rc:
                cell[:, :] = _get_bg(grid)
                cell[template_mask] = GREY
                continue

            colors = {int(v) for v in np.unique(cell)}
            non_bg = [c for c in colors if c not in (_get_bg(grid),)]
            if not non_bg:
                cell[:, :] = _get_bg(grid)
                continue
            assert GREY not in non_bg, "only template cell may contain grey"
            assert len(non_bg) == 1, "scatter dots in a cell must share one color"
            dot_color = non_bg[0]
            dot_count = int(np.count_nonzero(cell == dot_color))
            assert 1 <= dot_count <= 4, "colored dot count must be 1..4"
            rot = (dot_count - baseline) % 4
            rotated = np.rot90(template_mask, k=rot)
            cell[:, :] = _get_bg(grid)
            cell[rotated] = dot_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each cell's colored dot count and color into a rotated recolored copy of the grey template.

    Input structure:
    - One color forms full horizontal and vertical divider bars that split the canvas into equal-sized cells.
    - Exactly one cell contains a grey asymmetric template shape together with several isolated grey baseline dots.
    - Every other non-empty cell contains 1 to 4 isolated dots of a single non-grey color, or is empty.

    Rule:
    - Read the grey multi-pixel component in the template cell as the template mask and count that cell's isolated grey dots as the baseline value `b`.
    - For each colored-dot cell, erase its dots, rotate the template counterclockwise by `((dot_count - b) mod 4) * 90` degrees, recolor it to the dots' color, and stamp it back in the same relative cell position.
    - Empty cells stay empty, the template cell keeps only the grey template, and divider bars remain unchanged.

    Key insight:
    - The dots encode two independent channels: count gives a rotation relative to the hidden baseline inside the template cell, while color gives the stamped template color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg, divider, row_segs, col_segs = _divider_grid_structure(g)
    assert divider != GREY and bg != GREY
    template_rc, template_mask, baseline = _locate_template(g, row_segs, col_segs)
    assert baseline >= 1, "template cell must contain at least one baseline dot"
    assert not np.array_equal(template_mask, np.rot90(template_mask, 1))
    assert not np.array_equal(template_mask, np.rot90(template_mask, 2))
    return _stamp_template(g, row_segs, col_segs, template_rc, template_mask, baseline)


def _random_template(cell_size: int) -> np.ndarray:
    target_area = random.randint(4, min(7, cell_size * cell_size - 6))
    for _ in range(200):
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        start = (random.randrange(1, cell_size - 1), random.randrange(1, cell_size - 1))
        mask[start] = True
        cells = [start]
        while len(cells) < target_area:
            rr, cc = random.choice(cells)
            nbrs = []
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < cell_size and 0 <= nc < cell_size and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            choice = random.choice(nbrs)
            mask[choice] = True
            cells.append(choice)
        if np.count_nonzero(mask) != target_area:
            continue
        if any(np.all(mask[r, :]) for r in range(cell_size)):
            continue
        if any(np.all(mask[:, c]) for c in range(cell_size)):
            continue
        rots = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rots) != 4:
            continue
        if np.count_nonzero(mask[0]) + np.count_nonzero(mask[-1]) + np.count_nonzero(mask[:, 0]) + np.count_nonzero(mask[:, -1]) == 0:
            continue
        return mask
    raise RuntimeError("failed to build asymmetric template")


def _scatter_positions(cell_size: int, blocked: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(cell_size) for c in range(cell_size) if (r, c) not in blocked]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        ok = True
        for rr, cc in blocked:
            if abs(rr - r) + abs(cc - c) <= 1:
                ok = False
                break
        if not ok:
            continue
        for rr, cc in chosen:
            if abs(rr - r) + abs(cc - c) <= 1:
                ok = False
                break
        if ok:
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _empty_cell_count(rows: int, cols: int) -> int:
    total = rows * cols
    max_empty = max(1, total // 3)
    return random.randint(1, max_empty)


def _wrong_absolute_rotation(
    grid: np.ndarray,
    row_segs: list[tuple[int, int]],
    col_segs: list[tuple[int, int]],
    template_rc: tuple[int, int],
    template_mask: np.ndarray,
) -> np.ndarray:
    bg = _get_bg(grid)
    out = grid.copy()
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = out[r0:r1, c0:c1]
            if (ri, ci) == template_rc:
                cell[:, :] = bg
                cell[template_mask] = GREY
                continue
            vals = [int(v) for v in np.unique(cell) if int(v) != bg]
            if not vals:
                cell[:, :] = bg
                continue
            color = vals[0]
            count = int(np.count_nonzero(cell == color))
            cell[:, :] = bg
            cell[np.rot90(template_mask, k=(count - 1) % 4)] = color
    return out


def _wrong_no_rotation(
    grid: np.ndarray,
    row_segs: list[tuple[int, int]],
    col_segs: list[tuple[int, int]],
    template_rc: tuple[int, int],
    template_mask: np.ndarray,
) -> np.ndarray:
    bg = _get_bg(grid)
    out = grid.copy()
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = out[r0:r1, c0:c1]
            if (ri, ci) == template_rc:
                cell[:, :] = bg
                cell[template_mask] = GREY
                continue
            vals = [int(v) for v in np.unique(cell) if int(v) != bg]
            if not vals:
                cell[:, :] = bg
                continue
            color = vals[0]
            cell[:, :] = bg
            cell[template_mask] = color
    return out


def _wrong_ignore_color(
    grid: np.ndarray,
    row_segs: list[tuple[int, int]],
    col_segs: list[tuple[int, int]],
    template_rc: tuple[int, int],
    template_mask: np.ndarray,
    baseline: int,
) -> np.ndarray:
    bg = _get_bg(grid)
    out = grid.copy()
    fixed_color = RED if RED != GREY else BLUE
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = out[r0:r1, c0:c1]
            if (ri, ci) == template_rc:
                cell[:, :] = bg
                cell[template_mask] = GREY
                continue
            vals = [int(v) for v in np.unique(cell) if int(v) != bg]
            if not vals:
                cell[:, :] = bg
                continue
            count = int(np.count_nonzero(cell == vals[0]))
            cell[:, :] = bg
            cell[np.rot90(template_mask, k=(count - baseline) % 4)] = fixed_color
    return out


def _build_instance() -> np.ndarray | None:
    rows = random.randint(3, 4)
    cols = random.randint(3, 4)
    cell_size = random.randint(5, 6)
    full_h = rows * cell_size + (rows - 1)
    full_w = cols * cell_size + (cols - 1)
    if max(full_h, full_w) > 30:
        return None

    bg = random.choice([c for c in NON_GREY if c != GREY])
    divider = random.choice([c for c in NON_GREY if c not in (bg, GREY)])
    dot_palette = [c for c in COLOR_DOTS if c not in (bg, divider, GREY)]
    if len(dot_palette) < 3:
        return None

    grid = np.full((full_h, full_w), bg, dtype=int)
    for r in range(cell_size, full_h, cell_size + 1):
        grid[r, :] = divider
    for c in range(cell_size, full_w, cell_size + 1):
        grid[:, c] = divider

    template_mask = _random_template(cell_size)
    template_rc = (random.randrange(rows), random.randrange(cols))
    baseline = random.randint(2, 4)

    row_segs = [(r * (cell_size + 1), r * (cell_size + 1) + cell_size) for r in range(rows)]
    col_segs = [(c * (cell_size + 1), c * (cell_size + 1) + cell_size) for c in range(cols)]

    tmpl_r0, tmpl_r1 = row_segs[template_rc[0]]
    tmpl_c0, tmpl_c1 = col_segs[template_rc[1]]
    blocked = set(map(tuple, np.argwhere(template_mask)))
    baseline_pos = _scatter_positions(cell_size, blocked, baseline)
    if baseline_pos is None:
        return None
    template_cell = grid[tmpl_r0:tmpl_r1, tmpl_c0:tmpl_c1]
    template_cell[template_mask] = GREY
    for r, c in baseline_pos:
        template_cell[r, c] = GREY

    non_template_cells = [(ri, ci) for ri in range(rows) for ci in range(cols) if (ri, ci) != template_rc]
    random.shuffle(non_template_cells)
    if len(non_template_cells) < 5:
        return None

    empty_needed = _empty_cell_count(rows, cols)
    empty_cells = set(non_template_cells[:empty_needed])
    active_cells = [rc for rc in non_template_cells if rc not in empty_cells]
    if len(active_cells) < 4:
        return None

    required_counts = [1, 2, 3, 4]
    colors = random.sample(dot_palette, k=random.randint(3, min(5, len(dot_palette))))
    per_cell_spec: dict[tuple[int, int], tuple[int, int]] = {}

    for rc, count in zip(active_cells[:4], required_counts):
        color = random.choice(colors)
        per_cell_spec[rc] = (count, color)

    for rc in active_cells[4:]:
        count = random.randint(1, 4)
        color = random.choice(colors)
        per_cell_spec[rc] = (count, color)

    used_rotations = {(count - baseline) % 4 for count, _color in per_cell_spec.values()}
    if used_rotations != {0, 1, 2, 3}:
        return None
    if len({color for _count, color in per_cell_spec.values()}) < 2:
        return None
    if not any(count == baseline for count, _color in per_cell_spec.values()):
        return None

    for (ri, ci), (count, color) in per_cell_spec.items():
        r0, r1 = row_segs[ri]
        c0, c1 = col_segs[ci]
        cell = grid[r0:r1, c0:c1]
        pos = _scatter_positions(cell_size, set(), count)
        if pos is None:
            return None
        for r, c in pos:
            cell[r, c] = color

    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bg, divider, row_segs, col_segs = _divider_grid_structure(grid)
        template_rc, template_mask, baseline = _locate_template(grid, row_segs, col_segs)
        out = main(grid)
    except AssertionError:
        return False

    if divider in (bg, GREY):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False

    grey_cells = 0
    empty_cells = 0
    colored_cells = 0
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            cell = grid[r0:r1, c0:c1]
            colors = {int(v) for v in np.unique(cell) if int(v) != bg}
            if not colors:
                empty_cells += 1
                continue
            if colors == {GREY}:
                grey_cells += 1
                comps = _components(cell == GREY)
                if len([comp for comp in comps if len(comp) > 1]) != 1:
                    return False
                if (ri, ci) != template_rc:
                    return False
            else:
                colored_cells += 1
                if GREY in colors or len(colors) != 1:
                    return False
                color = next(iter(colors))
                count = int(np.count_nonzero(cell == color))
                if not (1 <= count <= 4):
                    return False
    if grey_cells != 1 or empty_cells < 1 or colored_cells < 4:
        return False

    abs_wrong = _wrong_absolute_rotation(grid, row_segs, col_segs, template_rc, template_mask)
    no_rot_wrong = _wrong_no_rotation(grid, row_segs, col_segs, template_rc, template_mask)
    color_wrong = _wrong_ignore_color(grid, row_segs, col_segs, template_rc, template_mask, baseline)
    if np.array_equal(out, abs_wrong):
        return False
    if np.array_equal(out, no_rot_wrong):
        return False
    if np.array_equal(out, color_wrong):
        return False

    painted_colors = sorted(int(v) for v in np.unique(out) if int(v) not in (bg, divider, GREY))
    if len(painted_colors) < 2:
        return False
    if int(np.count_nonzero(out == GREY)) != int(np.count_nonzero(template_mask)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance()
        if grid is None:
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid zero-point recoding puzzle")
