from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400
BAND_H = 6
MIN_SPACING = 4

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize_shape(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _parse_scene(
    grid: np.ndarray,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[int, int],
    np.ndarray,
]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    object_comps = _components((g != BG) & (g != GREY))
    seeds: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    coord_to_seed: dict[tuple[int, int], int] = {}
    coord_to_node: dict[tuple[int, int], int] = {}
    seed_mask = np.zeros_like(g, dtype=bool)

    for cells in object_comps:
        colors = {int(g[r, c]) for r, c in cells}
        assert len(colors) == 1, "every non-grey component must be monochrome"
        color = int(next(iter(colors)))
        if len(cells) == 1:
            r, c = cells[0]
            node_id = len(nodes)
            nodes.append({"color": color, "pos": (r, c)})
            coord_to_node[(r, c)] = node_id
            continue

        seed_id = len(seeds)
        shape = _normalize_shape(cells)
        seeds.append({"color": color, "shape": shape, "cells": tuple(cells)})
        for r, c in cells:
            coord_to_seed[(r, c)] = seed_id
            seed_mask[r, c] = True

    assert seeds, "expected at least one seed"
    assert nodes, "expected at least one relay node"

    node_to_seed: dict[int, int] = {}
    grey_comps = _components(g == GREY)
    assert grey_comps, "expected grey connection lines"
    for cells in grey_comps:
        adjacent_seeds: set[int] = set()
        adjacent_nodes: set[int] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in coord_to_seed:
                    adjacent_seeds.add(coord_to_seed[(nr, nc)])
                elif (nr, nc) in coord_to_node:
                    adjacent_nodes.add(coord_to_node[(nr, nc)])
        assert len(adjacent_seeds) == 1, "each grey line must belong to exactly one seed"
        assert adjacent_nodes, "each grey line must reach at least one relay node"
        seed_id = next(iter(adjacent_seeds))
        for node_id in adjacent_nodes:
            prev = node_to_seed.get(node_id)
            assert prev is None or prev == seed_id
            node_to_seed[node_id] = seed_id

    assert len(node_to_seed) == len(nodes), "every node must attach to one grey line"
    return seeds, nodes, node_to_seed, seed_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each relay node with the shape of its connected seed, using the relay's own color.

    Input structure:
    - Background is Black.
    - Several small monochrome seed blocks (within 3x3) appear, with unique colors
      and distinct connected shapes.
    - Single-pixel relay nodes of other colors sit at the ends of Grey one-pixel-wide
      connection lines.
    - Each Grey connected component touches exactly one seed and one or more relay nodes.

    Rule:
    - Trace each Grey connected component to determine which relay nodes belong to which seed.
    - Use the connected seed's relative shape as a stamp template.
    - For every relay node, place that template with the node as the template's top-left
      corner and paint it in the node's own color, clipping at the grid boundary.
    - Seed pixels stay unchanged, and only the Grey line cells covered by a stamp are replaced.

    Key insight:
    - Shape and color come from different places: the Grey path identifies the seed shape,
      while the relay node itself supplies the paint color.
    """
    g = np.asarray(grid, dtype=int)
    seeds, nodes, node_to_seed, seed_mask = _parse_scene(g)
    out = g.copy()
    h, w = out.shape

    for node_id, node in enumerate(nodes):
        r0, c0 = node["pos"]
        color = int(node["color"])
        shape = seeds[node_to_seed[node_id]]["shape"]
        for dr, dc in shape:
            r = r0 + dr
            c = c0 + dc
            if 0 <= r < h and 0 <= c < w and not seed_mask[r, c]:
                out[r, c] = color
    return out


def _split_slack(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _sample_spaced_columns(lo: int, hi: int, count: int, spacing: int) -> list[int] | None:
    if count == 0:
        return []
    cols: list[int] = []
    current = lo
    remaining = count
    while remaining:
        max_pos = hi - (remaining - 1) * spacing
        if current > max_pos:
            return None
        col = random.randint(current, max_pos)
        cols.append(col)
        current = col + spacing
        remaining -= 1
    return cols


def _right_boundary_rows(shape: tuple[tuple[int, int], ...]) -> list[int]:
    _h, w = _shape_dims(shape)
    return sorted({r for r, c in shape if c == w - 1})


def _band_tops(height: int, n_bands: int) -> list[int]:
    slack = height - n_bands * BAND_H
    gaps = _split_slack(slack, n_bands + 1)
    tops: list[int] = []
    row = gaps[0]
    for i in range(n_bands):
        tops.append(row)
        row += BAND_H + gaps[i + 1]
    return tops


def _place_seed(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    band_top: int,
    band_bottom: int,
) -> tuple[int, int, int, int]:
    h, w = _shape_dims(shape)
    candidates: list[tuple[int, int, int]] = []
    for right_row in _right_boundary_rows(shape):
        for trunk_rel in (1, 2):
            seed_top = band_top + trunk_rel - right_row
            if band_top <= seed_top and seed_top + h - 1 <= band_bottom:
                candidates.append((seed_top, right_row, band_top + trunk_rel))
    assert candidates
    seed_top, right_row, trunk_row = random.choice(candidates)
    seed_left = random.randint(1, 3)
    for dr, dc in shape:
        grid[seed_top + dr, seed_left + dc] = color
    return seed_top, seed_left, right_row, trunk_row


def _coverage_types(shape: tuple[tuple[int, int], ...], x: int, width: int) -> list[str]:
    coverage: list[str] = []
    shape_set = set(shape)
    if (1, 0) in shape_set:
        coverage.append("above_down")
    if (0, 1) in shape_set and x + 1 < width:
        coverage.extend(["above_right", "below_right"])
    return coverage


def _draw_band_component(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    seed_color: int,
    node_colors: list[int],
    band_top: int,
    band_bottom: int,
) -> bool:
    height, width = grid.shape
    shape_h, shape_w = _shape_dims(shape)
    seed_top, seed_left, _right_row, trunk_row = _place_seed(grid, shape, seed_color, band_top, band_bottom)
    trunk_start = seed_left + shape_w
    n_nodes = len(node_colors)
    clip_node = shape_w > 1 and random.random() < 0.45
    clip_col = None
    if clip_node:
        clip_amount = random.randint(1, min(shape_w - 1, 2))
        clip_col = width - clip_amount

    safe_hi = width - shape_w - 1
    lo = trunk_start + 3
    normal_count = n_nodes - int(clip_node)
    normal_hi = safe_hi if clip_col is None else min(safe_hi, clip_col - MIN_SPACING)
    cols = _sample_spaced_columns(lo, normal_hi, normal_count, MIN_SPACING)
    if cols is None:
        return False
    if clip_col is not None:
        if cols and clip_col - cols[-1] < MIN_SPACING:
            return False
        cols.append(clip_col)
    if not cols:
        return False

    specs: list[dict[str, object]] = []
    want_coverage = random.random() < 0.7
    used_coverage = False
    for color, x in zip(node_colors, cols, strict=True):
        valid_types = ["above_down", "below_up"]
        if x + 1 < width:
            valid_types.extend(["above_right", "below_right"])
        coverage_types = _coverage_types(shape, x, width)
        if want_coverage and not used_coverage and coverage_types:
            node_type = random.choice(coverage_types)
            used_coverage = True
        else:
            node_type = random.choice(valid_types)

        if node_type == "above_down":
            node_pos = (trunk_row - 1, x)
            branch_cells: list[tuple[int, int]] = []
            trunk_tip = x
        elif node_type == "below_up":
            node_pos = (trunk_row + 1, x)
            branch_cells = []
            trunk_tip = x
        elif node_type == "above_right":
            node_pos = (trunk_row - 1, x)
            branch_cells = [(trunk_row - 1, x + 1)]
            trunk_tip = x + 1
        else:
            node_pos = (trunk_row + 1, x)
            branch_cells = [(trunk_row + 1, x + 1)]
            trunk_tip = x + 1
        specs.append(
            {
                "color": color,
                "type": node_type,
                "node_pos": node_pos,
                "branch_cells": branch_cells,
                "trunk_tip": trunk_tip,
            }
        )

    trunk_end = max(int(spec["trunk_tip"]) for spec in specs)
    if trunk_end >= width:
        return False

    for c in range(trunk_start, trunk_end + 1):
        if grid[trunk_row, c] != BG:
            return False
    for c in range(trunk_start, trunk_end + 1):
        grid[trunk_row, c] = GREY

    for spec in specs:
        for r, c in spec["branch_cells"]:
            if not (0 <= r < height and 0 <= c < width) or grid[r, c] != BG:
                return False
            grid[r, c] = GREY

    for spec in specs:
        r, c = spec["node_pos"]
        if grid[r, c] != BG:
            return False
        grid[r, c] = int(spec["color"])

    seed_cells = sum(1 for _ in shape)
    if seed_cells <= 1:
        return False
    if not used_coverage and want_coverage and not clip_node:
        return False
    _ = seed_top
    return True


def _generate_horizontal() -> np.ndarray:
    n_bands = random.randint(2, 4)
    min_height = n_bands * BAND_H
    height = random.randint(min_height, min(30, min_height + 4))
    width = random.randint(20, 30)
    band_tops = _band_tops(height, n_bands)
    grid = np.full((height, width), BG, dtype=int)

    seed_shapes = random.sample(list(SHAPES), n_bands)
    seed_colors = random.sample(list(COLOR_POOL), n_bands)
    node_palette = [color for color in COLOR_POOL if color not in seed_colors]

    for shape, seed_color, band_top in zip(seed_shapes, seed_colors, band_tops, strict=True):
        band_bottom = band_top + BAND_H - 1
        node_count = random.randint(1, 3)
        node_colors = [random.choice(node_palette) for _ in range(node_count)]
        if not _draw_band_component(grid, shape, seed_color, node_colors, band_top, band_bottom):
            raise ValueError("band construction failed")

    return grid


def _valid_generated(inp: np.ndarray) -> bool:
    try:
        seeds, nodes, node_to_seed, _seed_mask = _parse_scene(inp)
        if len(seeds) < 2 or len(nodes) < len(seeds):
            return False
        out = main(inp)
        if out.shape != inp.shape or not np.any(out != inp):
            return False
        if max(inp.shape) > 30 or max(out.shape) > 30:
            return False
        if len(node_to_seed) != len(nodes):
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp = _generate_horizontal()
        except ValueError:
            continue
        if random.random() < 0.5:
            inp = inp.T.copy()
        if _valid_generated(inp):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
