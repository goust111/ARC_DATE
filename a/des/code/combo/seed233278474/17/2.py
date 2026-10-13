from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _decode_l_component(comp: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int] | None:
    comp_set = set(comp)
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    for jr, jc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        if (jr, jc) not in comp_set:
            continue

        row_cells = sorted(c for r, c in comp if r == jr)
        col_cells = sorted(r for r, c in comp if c == jc)
        if len(row_cells) < 2 or len(col_cells) < 2:
            continue

        if row_cells[0] == jc and row_cells[-1] > jc:
            hdir = 1
            h_end = row_cells[-1]
        elif row_cells[-1] == jc and row_cells[0] < jc:
            hdir = -1
            h_end = row_cells[0]
        else:
            continue

        if col_cells[0] == jr and col_cells[-1] > jr:
            vdir = 1
            v_end = col_cells[-1]
        elif col_cells[-1] == jr and col_cells[0] < jr:
            vdir = -1
            v_end = col_cells[0]
        else:
            continue

        expected = {(jr, c) for c in range(jc, h_end + hdir, hdir)}
        expected |= {(r, jc) for r in range(jr, v_end + vdir, vdir)}
        if comp_set == expected:
            return comp_set, hdir, vdir

    return None


def _find_legend(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, set[tuple[int, int]], int, int]] = []
    for comp in _components4(grid != bg):
        if len(comp) <= 1:
            continue
        decoded = _decode_l_component(comp)
        if decoded is None:
            continue
        cells, hdir, vdir = decoded
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        border_hits = int(min(rows) == 0) + int(max(rows) == h - 1) + int(min(cols) == 0) + int(max(cols) == w - 1)
        candidates.append((border_hits, len(comp), cells, hdir, vdir))

    assert candidates, "expected one non-background L-shaped legend"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _score, _size, cells, hdir, vdir = candidates[-1]
    return cells, hdir, vdir


def _rank_colors(
    grid: np.ndarray,
    bg: int,
    legend_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    first_pos: dict[int, tuple[int, int]] = {}

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if (r, c) in legend_cells:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            counts[color] = counts.get(color, 0) + 1
            first_pos.setdefault(color, (r, c))

    assert counts, "expected at least one colored fragment group"
    return sorted(counts.items(), key=lambda item: (-item[1], first_pos[item[0]]))


def _render_output(ranked: list[tuple[int, int]], bg: int, hdir: int, vdir: int) -> np.ndarray:
    odd_lengths = [count for idx, (_color, count) in enumerate(ranked) if idx % 2 == 0]
    even_lengths = [count for idx, (_color, count) in enumerate(ranked) if idx % 2 == 1]

    height = max(max(even_lengths, default=0), len(odd_lengths))
    width = max(max(odd_lengths, default=0), len(even_lengths))
    out = np.full((height, width), bg, dtype=int)

    anchor_r = height - 1 if vdir < 0 else 0
    anchor_c = 0 if hdir > 0 else width - 1

    odd_index = 0
    even_index = 0
    for rank, (color, count) in enumerate(ranked, start=1):
        if rank % 2 == 1:
            row = anchor_r + odd_index * vdir
            for step in range(count):
                col = anchor_c + step * hdir
                out[row, col] = color
            odd_index += 1
        else:
            col = anchor_c + even_index * hdir
            for step in range(count):
                row = anchor_r + step * vdir
                out[row, col] = color
            even_index += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered single-pixel colors by count and lay them out as ranked lines.

    The unique non-background multi-pixel L component is treated as the direction
    legend and removed. Remaining non-background pixels are counted per color.
    Colors are ranked by descending count. Rank 1, 3, 5... are painted as
    horizontal lines; rank 2, 4, 6... are painted as vertical lines. The legend's
    horizontal and vertical arms determine the output anchor corner and the
    directions in which those lines extend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_cells, hdir, vdir = _find_legend(g, bg)
    ranked = _rank_colors(g, bg, legend_cells)
    return _render_output(ranked, bg, hdir, vdir)


# --------------- generator ---------------

LEGEND_COLOR = 9
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (14, 22)
GROUP_COUNT_RANGE = (2, 6)
COUNT_VALUES = tuple(range(2, 8))
LEGEND_ARM_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300


def _legend_cells_for_grid(height: int, arm_h: int, arm_v: int) -> set[tuple[int, int]]:
    cells = {(height - 1, c) for c in range(arm_h)}
    cells |= {(height - 1 - r, 0) for r in range(arm_v)}
    return cells


def _far_from_all(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in others)


def _pick_fragment_positions(
    height: int,
    width: int,
    total: int,
    legend_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if _far_from_all((r, c), legend_cells)
    ]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    for cell in candidates:
        if _far_from_all(cell, chosen_set):
            chosen.append(cell)
            chosen_set.add(cell)
            if len(chosen) == total:
                return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.randint(0, 8)

    group_count = random.randint(*GROUP_COUNT_RANGE)
    counts = random.sample(COUNT_VALUES, k=group_count)
    colors = random.sample([c for c in range(10) if c not in {bg, LEGEND_COLOR}], k=group_count)

    arm_h = random.randint(*LEGEND_ARM_RANGE)
    arm_v = random.randint(*LEGEND_ARM_RANGE)
    legend_cells = _legend_cells_for_grid(height, arm_h, arm_v)

    total_pixels = sum(counts)
    positions = _pick_fragment_positions(height, width, total_pixels, legend_cells)
    if positions is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for r, c in legend_cells:
        grid[r, c] = LEGEND_COLOR

    pixel_colors: list[int] = []
    ranked = sorted(zip(colors, counts), key=lambda item: -item[1])
    for color, count in zip(colors, counts):
        pixel_colors.extend([color] * count)
    random.shuffle(pixel_colors)
    for (r, c), color in zip(positions, pixel_colors):
        grid[r, c] = color

    expected = _render_output(ranked, bg, hdir=1, vdir=-1)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
