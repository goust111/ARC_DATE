from __future__ import annotations

from collections import Counter
import random
import sys
from typing import Iterable

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import bbox, get_bg, label_mask


GRID_CELLS = 3
OUTPUT_SIZE = GRID_CELLS * GRID_CELLS
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    if r > 0:
        yield r - 1, c
    if r + 1 < GRID_CELLS:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < GRID_CELLS:
        yield r, c + 1


def _cell_components(mask: np.ndarray, value: bool) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(GRID_CELLS):
        for c in range(GRID_CELLS):
            if bool(mask[r, c]) != bool(value) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                comp.append(cell)
                for nxt in _neighbors(cell):
                    if nxt not in seen and bool(mask[nxt]) == bool(value):
                        seen.add(nxt)
                        stack.append(nxt)
            comps.append(comp)
    return comps


def _valid_mask(mask: np.ndarray) -> bool:
    ones = int(mask.sum())
    if not (2 <= ones <= 7):
        return False
    zeros = GRID_CELLS * GRID_CELLS - ones
    if zeros == 0:
        return False
    zero_components = _cell_components(mask, False)
    return all(len(comp) >= 2 for comp in zero_components)


VALID_MASKS: list[np.ndarray] = []
for bits in range(1 << (GRID_CELLS * GRID_CELLS)):
    mask = np.array([(bits >> i) & 1 for i in range(GRID_CELLS * GRID_CELLS)], dtype=bool).reshape(GRID_CELLS, GRID_CELLS)
    if _valid_mask(mask):
        VALID_MASKS.append(mask)


def _detect_lattice_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    foreground = [(count, color) for color, count in counts.items() if color != bg]
    assert foreground, "expected a lattice color"
    foreground.sort(reverse=True)
    assert len(foreground) == 1 or foreground[0][0] > foreground[1][0], "expected a unique lattice color"
    return int(foreground[0][1])


def _interior_bounds(frame_bbox: tuple[int, int, int, int], cell_h: int, cell_w: int, i: int, j: int) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    rs = r0 + 1 + i * (cell_h + 1)
    cs = c0 + 1 + j * (cell_w + 1)
    return rs, cs, rs + cell_h - 1, cs + cell_w - 1


def _parse_input(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    lattice = _detect_lattice_color(g, bg)

    frame_bbox = bbox(g == lattice)
    assert frame_bbox is not None, "expected a lattice frame"
    r0, c0, r1, c1 = frame_bbox
    frame_h = r1 - r0 + 1
    frame_w = c1 - c0 + 1
    assert (frame_h - 4) % GRID_CELLS == 0 and (frame_w - 4) % GRID_CELLS == 0, "invalid 3x3 lattice dimensions"
    cell_h = (frame_h - 4) // GRID_CELLS
    cell_w = (frame_w - 4) // GRID_CELLS
    assert cell_h >= 1 and cell_w >= 1, "cells must be nonempty"

    open_mask = np.zeros_like(g, dtype=bool)
    open_mask[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] != lattice
    labels, n_comp = label_mask(open_mask, connectivity=4)
    assert n_comp > 0, "expected non-lattice interior regions"

    comp_area = np.bincount(labels.ravel(), minlength=n_comp + 1)
    comp_bbox: dict[int, tuple[int, int, int, int]] = {}
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labels == comp_id)
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0)
        comp_bbox[comp_id] = (int(rr0), int(cc0), int(rr1), int(cc1))

    intact_mask = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
    intact_colors = np.full((GRID_CELLS, GRID_CELLS), bg, dtype=int)
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            rs, cs, re, ce = _interior_bounds(frame_bbox, cell_h, cell_w, i, j)
            cell_open = open_mask[rs : re + 1, cs : ce + 1]
            if not bool(np.all(cell_open)):
                continue

            ids = np.unique(labels[rs : re + 1, cs : ce + 1])
            ids = ids[ids > 0]
            if ids.size != 1:
                continue

            comp_id = int(ids[0])
            if int(comp_area[comp_id]) != cell_h * cell_w:
                continue
            if comp_bbox[comp_id] != (rs, cs, re, ce):
                continue

            color_pixels = np.argwhere((g[rs : re + 1, cs : ce + 1] != bg) & (g[rs : re + 1, cs : ce + 1] != lattice))
            assert color_pixels.shape[0] == 1, "each intact cell must contain exactly one singleton pixel"
            dr, dc = map(int, color_pixels[0])
            intact_mask[i, j] = True
            intact_colors[i, j] = int(g[rs + dr, cs + dc])

    assert intact_mask.any(), "expected at least one intact cell"
    return bg, intact_mask, intact_colors


def _render_output(bg: int, intact_mask: np.ndarray, intact_colors: np.ndarray) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), int(bg), dtype=int)
    block_shape = intact_mask.astype(bool)
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            if not bool(intact_mask[i, j]):
                continue
            color = int(intact_colors[i, j])
            block = np.full((GRID_CELLS, GRID_CELLS), int(bg), dtype=int)
            block[block_shape] = color
            r0 = i * GRID_CELLS
            c0 = j * GRID_CELLS
            out[r0 : r0 + GRID_CELLS, c0 : c0 + GRID_CELLS] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read which 3x3 lattice cells remain topologically intact, then stamp that hidden mask into itself with per-cell singleton colors.

    Input structure:
    - One frequent color forms a rectangular 3x3 lattice frame with intact outer border and possibly broken internal dividers.
    - Every cell interior contains non-lattice space; some cells are isolated intact rectangles, while others connect through divider gaps into larger open regions.
    - Each intact cell contains exactly one colored singleton; broken merged regions may also contain distractor singletons.

    Rule:
    - Treat every non-lattice pixel inside the frame, including singleton pixels, as open space and compute its 4-connected components.
    - A cell is "intact" iff its whole interior is one component whose area and bounding box match that cell interior exactly.
    - Build a 9x9 output: for every intact mask position, fill the corresponding 3x3 output block with another copy of the full 3x3 intact mask, using the intact cell's singleton color; leave broken positions as background.

    Key insight:
    - The visible lattice damage is only a code for a hidden binary mask, and the singleton in each sealed cell supplies the color for that cell's entire nested copy of the mask.
    """
    bg, intact_mask, intact_colors = _parse_input(grid)
    return _render_output(bg, intact_mask, intact_colors)


def _mask_edges(cells: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cell_set = set(cells)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell in cells:
        for nxt in _neighbors(cell):
            if nxt in cell_set and cell < nxt:
                edges.append((cell, nxt))
    return edges


def _open_wall(
    grid: np.ndarray,
    top: int,
    left: int,
    cell_h: int,
    cell_w: int,
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
    rng: random.Random,
) -> None:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        row0 = top + 1 + r1 * (cell_h + 1)
        wall_c = left + (min(c1, c2) + 1) * (cell_w + 1)
        gap_len = rng.randint(1, max(1, min(3, cell_h)))
        offset = rng.randint(0, cell_h - gap_len)
        grid[row0 + offset : row0 + offset + gap_len, wall_c] = bg
    else:
        col0 = left + 1 + c1 * (cell_w + 1)
        wall_r = top + (min(r1, r2) + 1) * (cell_h + 1)
        gap_len = rng.randint(1, max(1, min(3, cell_w)))
        offset = rng.randint(0, cell_w - gap_len)
        grid[wall_r, col0 + offset : col0 + offset + gap_len] = bg


def _draw_lattice(grid: np.ndarray, top: int, left: int, cell_h: int, cell_w: int, lattice: int) -> tuple[int, int, int, int]:
    height = GRID_CELLS * cell_h + GRID_CELLS + 1
    width = GRID_CELLS * cell_w + GRID_CELLS + 1
    bottom = top + height - 1
    right = left + width - 1

    row_lines = [top + k * (cell_h + 1) for k in range(GRID_CELLS + 1)]
    col_lines = [left + k * (cell_w + 1) for k in range(GRID_CELLS + 1)]
    for rr in row_lines:
        grid[rr, left : right + 1] = lattice
    for cc in col_lines:
        grid[top : bottom + 1, cc] = lattice
    return top, left, bottom, right


def _place_singletons(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    cell_h: int,
    cell_w: int,
    intact_mask: np.ndarray,
    colors: np.ndarray,
    rng: random.Random,
) -> None:
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            rs, cs, re, ce = _interior_bounds(frame_bbox, cell_h, cell_w, i, j)
            cells = [(r, c) for r in range(rs, re + 1) for c in range(cs, ce + 1)]
            r, c = rng.choice(cells)
            if bool(intact_mask[i, j]):
                grid[r, c] = int(colors[i, j])


def _add_broken_singletons(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    cell_h: int,
    cell_w: int,
    intact_mask: np.ndarray,
    bg: int,
    lattice: int,
    intact_colors: np.ndarray,
    rng: random.Random,
) -> None:
    broken_cells = [(i, j) for i in range(GRID_CELLS) for j in range(GRID_CELLS) if not bool(intact_mask[i, j])]
    assert broken_cells, "generator requires at least one broken cell"

    broken_count = rng.randint(1, min(len(broken_cells), 4))
    selected = rng.sample(broken_cells, broken_count)

    available = [color for color in PALETTE if color not in {bg, lattice}]
    intact_palette = [int(intact_colors[i, j]) for i in range(GRID_CELLS) for j in range(GRID_CELLS) if bool(intact_mask[i, j])]
    for i, j in selected:
        rs, cs, re, ce = _interior_bounds(frame_bbox, cell_h, cell_w, i, j)
        cells = [(r, c) for r in range(rs, re + 1) for c in range(cs, ce + 1)]
        r, c = rng.choice(cells)
        if rng.random() < 0.35 and intact_palette:
            color = rng.choice(intact_palette)
        else:
            color = rng.choice(available)
        grid[r, c] = int(color)


def _expected_from_spec(bg: int, intact_mask: np.ndarray, intact_colors: np.ndarray) -> np.ndarray:
    return _render_output(bg, intact_mask, intact_colors)


def _make_instance(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = random.Random() if rng is None else rng

    intact_mask = np.array(rng.choice(VALID_MASKS), copy=True)
    intact_count = int(intact_mask.sum())

    cell_h = rng.randint(2, 5)
    cell_w = rng.randint(2, 5)
    frame_h = GRID_CELLS * cell_h + GRID_CELLS + 1
    frame_w = GRID_CELLS * cell_w + GRID_CELLS + 1

    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    height = frame_h + top + bottom_margin
    width = frame_w + left + right_margin

    bg, lattice = rng.sample(PALETTE, 2)
    remaining = [color for color in PALETTE if color not in {bg, lattice}]
    intact_palette = rng.sample(remaining, intact_count)

    intact_colors = np.full((GRID_CELLS, GRID_CELLS), bg, dtype=int)
    color_iter = iter(intact_palette)
    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            if bool(intact_mask[i, j]):
                intact_colors[i, j] = next(color_iter)

    grid = np.full((height, width), bg, dtype=int)
    frame_bbox = _draw_lattice(grid, top, left, cell_h, cell_w, lattice)

    broken_components = _cell_components(intact_mask, False)
    for comp in broken_components:
        root = comp[0]
        seen = {root}
        queue = [root]
        tree_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        while queue:
            cell = queue.pop(0)
            for nxt in _neighbors(cell):
                if nxt in comp and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
                    tree_edges.append((cell, nxt))
        for edge in tree_edges:
            _open_wall(grid, top, left, cell_h, cell_w, edge[0], edge[1], bg, rng)

        extra_edges = [edge for edge in _mask_edges(comp) if edge not in tree_edges]
        rng.shuffle(extra_edges)
        for edge in extra_edges[: rng.randint(0, len(extra_edges))]:
            if rng.random() < 0.45:
                _open_wall(grid, top, left, cell_h, cell_w, edge[0], edge[1], bg, rng)

    _place_singletons(grid, frame_bbox, cell_h, cell_w, intact_mask, intact_colors, rng)
    _add_broken_singletons(grid, frame_bbox, cell_h, cell_w, intact_mask, bg, lattice, intact_colors, rng)

    assert get_bg(grid) == bg, "generator must keep the background dominant"
    expected = _expected_from_spec(bg, intact_mask, intact_colors)
    return grid, expected, intact_mask.astype(bool), int(bg)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected, intact_mask, _bg = _make_instance(random.Random())
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected) and 2 <= int(intact_mask.sum()) <= 7:
            return grid
    raise RuntimeError("failed to generate a valid broken-lattice nesting input")
