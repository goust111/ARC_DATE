from __future__ import annotations

import random
from collections import Counter

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_BG = 0
_PALETTE = list(range(1, 10))
_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in _DIRS4:
        yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        direction = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if prev_dir is not None and direction != prev_dir:
            turns += 1
        prev_dir = direction
    return turns


def _traverse_open_path(cells: list[tuple[int, int]], seed: tuple[int, int]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    degrees = {
        cell: sum((nbr in cell_set) for nbr in _neighbors4(cell[0], cell[1]))
        for cell in cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert seed in endpoints, "seed color must mark one endpoint"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be one-cell-wide and non-branching"

    ordered = [seed]
    prev = None
    cur = seed
    while True:
        next_cells = [nbr for nbr in _neighbors4(cur[0], cur[1]) if nbr in cell_set and nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unambiguous"
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(cells), "path traversal must visit every path cell exactly once"
    return ordered


def _classify_component(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    coords = np.array(cells, dtype=int)
    rows = coords[:, 0]
    cols = coords[:, 1]
    colors = [int(grid[r, c]) for r, c in cells]
    color_counts = Counter(colors)

    if len(color_counts) == 1:
        col = int(cols[0])
        assert np.all(cols == col), "bar must occupy one column"
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert len(cells) == r1 - r0 + 1, "bar must be vertically contiguous"
        return {
            "kind": "bar",
            "color": colors[0],
            "height": len(cells),
            "cells": sorted(cells),
            "col": col,
        }

    assert len(cells) >= 3, "path must be at least three cells long"
    assert len(color_counts) == 2, "path must use exactly one body color and one seed color"
    singleton_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(singleton_colors) == 1, "exactly one endpoint pixel must use the seed color"
    seed_color = singleton_colors[0]
    body_color = [color for color in color_counts if color != seed_color][0]
    seed = next(cell for cell in cells if int(grid[cell]) == seed_color)
    ordered = _traverse_open_path(cells, seed)
    return {
        "kind": "path",
        "seed_color": seed_color,
        "body_color": body_color,
        "seed": seed,
        "ordered_cells": ordered,
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected bars and paths on the background"

    bars: dict[int, dict] = {}
    paths: list[dict] = []
    heights = []
    for cells in components:
        info = _classify_component(g, cells)
        if info["kind"] == "bar":
            assert info["color"] not in bars, "bar colors must be unique"
            bars[info["color"]] = info
            heights.append(info["height"])
        else:
            paths.append(info)

    assert paths, "expected at least one path"
    assert bars, "expected at least one bar"
    assert len(paths) == len(bars), "bars and paths must pair one-to-one"
    assert len(set(heights)) == len(heights), "bar heights must be distinct"

    seed_colors = [path["seed_color"] for path in paths]
    assert len(set(seed_colors)) == len(seed_colors), "seed colors must be unique across paths"
    for path in paths:
        assert path["seed_color"] in bars, "each path seed color must match one bar color"
    return {"bg": bg, "bars": bars, "paths": paths}


def _expected_output(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for path in scene["paths"]:
        beat = scene["bars"][path["seed_color"]]["height"]
        for idx, cell in enumerate(path["ordered_cells"]):
            out[cell] = path["seed_color"] if idx % beat == 0 else path["body_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each snake path on beats whose period is the height of its same-colored vertical bar.

    Input structure:
    - The grid has a dominant background plus several disconnected foreground components.
    - Some components are solid one-column vertical bars of distinct heights.
    - The other components are one-cell-wide open 4-connected snake paths whose body has one color and whose single seed endpoint has a different color.

    Rule:
    - For each path, find the vertical bar whose color matches the path's seed endpoint color and let its height be k.
    - Starting from the seed endpoint as index 0, walk along the path cell by cell in path order.
    - Recolor exactly the cells whose path index is a multiple of k to the seed color; keep all other path cells in the body color, and leave every bar unchanged.

    Key insight:
    - The seed endpoint serves two roles at once: it identifies which bar belongs to the path, and it also fixes the counting start for the periodic recoloring.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _expected_output(grid, scene)


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    pieces = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _make_polyline_path(length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(200):
        segment_count = rng.randint(2, min(5, length - 1))
        segment_lengths = _random_composition(length - 1, segment_count, rng)
        directions = [rng.choice(_DIRS4)]
        for _seg in range(1, segment_count):
            dr, dc = directions[-1]
            if dr != 0:
                directions.append(rng.choice([(0, -1), (0, 1)]))
            else:
                directions.append(rng.choice([(-1, 0), (1, 0)]))

        path = [(0, 0)]
        visited = {(0, 0)}
        r = c = 0
        ok = True
        for seg_len, (dr, dc) in zip(segment_lengths, directions):
            for _step in range(seg_len):
                r += dr
                c += dc
                if (r, c) in visited:
                    ok = False
                    break
                visited.add((r, c))
                path.append((r, c))
            if not ok:
                break
        if not ok or len(path) != length or _turn_count(path) == 0:
            continue

        min_r = min(rc[0] for rc in path)
        min_c = min(rc[1] for rc in path)
        normalized = [(rr - min_r, cc - min_c) for rr, cc in path]
        span_h = 1 + max(rr for rr, _cc in normalized)
        span_w = 1 + max(cc for _rr, cc in normalized)
        if max(span_h, span_w) > 9:
            continue
        return normalized
    return None


def _mark_forbidden(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr, dc in _DIRS8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _place_coords(
    forbidden: np.ndarray,
    local_coords: list[tuple[int, int]],
    border: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = forbidden.shape
    obj_h = 1 + max(r for r, _c in local_coords)
    obj_w = 1 + max(c for _r, c in local_coords)
    if h < obj_h + 2 * border or w < obj_w + 2 * border:
        return None

    offsets = [
        (r0, c0)
        for r0 in range(border, h - obj_h - border + 1)
        for c0 in range(border, w - obj_w - border + 1)
    ]
    rng.shuffle(offsets)
    for r0, c0 in offsets:
        cells = [(r0 + r, c0 + c) for r, c in local_coords]
        if any(forbidden[r, c] for r, c in cells):
            continue
        _mark_forbidden(forbidden, cells)
        return cells
    return None


def _sample_length(height: int, category: str, rng: random.Random) -> int:
    if category == "sparse":
        return rng.randint(height + 1, min(2 * height, 14))
    if category == "dense":
        return rng.randint(2 * height + 1, min(3 * height + 1, 16))
    return rng.randint(2 * height + 1, min(2 * height + 3, 16))


def _make_case(rng: random.Random) -> np.ndarray | None:
    path_count = rng.choice([2, 3])
    colors = rng.sample(_PALETTE, 2 * path_count)
    seed_colors = colors[:path_count]
    body_colors = colors[path_count:]
    heights = rng.sample([2, 3, 4, 5, 6], path_count)

    categories = ["dense", "sparse"] if path_count == 2 else ["dense", "sparse", "medium"]
    rng.shuffle(categories)

    paths = []
    for seed_color, body_color, height, category in zip(seed_colors, body_colors, heights, categories):
        length = _sample_length(height, category, rng)
        local = _make_polyline_path(length, rng)
        if local is None:
            return None
        if rng.random() < 0.5:
            local = list(reversed(local))
            min_r = min(r for r, _c in local)
            min_c = min(c for _r, c in local)
            local = [(r - min_r, c - min_c) for r, c in local]
        paths.append(
            {
                "seed_color": seed_color,
                "body_color": body_color,
                "height": height,
                "local_coords": local,
            }
        )

    bars = [
        {
            "color": seed_color,
            "height": height,
            "local_coords": [(r, 0) for r in range(height)],
        }
        for seed_color, height in zip(seed_colors, heights)
    ]

    h = rng.randint(max(16, max(heights) + 8), 27)
    w = rng.randint(16, 29)
    grid = np.full((h, w), _BG, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    items = []
    for path in paths:
        items.append(("path", len(path["local_coords"]), path))
    for bar in bars:
        items.append(("bar", len(bar["local_coords"]), bar))
    items.sort(key=lambda item: item[1], reverse=True)

    for kind, _size, item in items:
        placed = _place_coords(forbidden, item["local_coords"], border=1, rng=rng)
        if placed is None:
            return None
        item["placed_coords"] = placed
        if kind == "path":
            for r, c in placed:
                grid[r, c] = item["body_color"]
            seed_r, seed_c = placed[0]
            grid[seed_r, seed_c] = item["seed_color"]
        else:
            for r, c in placed:
                grid[r, c] = item["color"]

    return grid


def _validate_case(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        out = _expected_output(grid, scene)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(out, actual):
        return False
    if np.array_equal(grid, out):
        return False

    beat_counts = []
    for path in scene["paths"]:
        beat = scene["bars"][path["seed_color"]]["height"]
        if len(path["ordered_cells"]) <= beat:
            return False
        beats = 1 + (len(path["ordered_cells"]) - 1) // beat
        beat_counts.append(beats)
        if beats < 2:
            return False
        if _turn_count(path["ordered_cells"]) < 1:
            return False

    if max(beat_counts) < 3 or min(beat_counts) > 2:
        return False

    path_order = [
        path["seed_color"]
        for path in sorted(scene["paths"], key=lambda path: (path["seed"][1], path["seed"][0]))
    ]
    bar_order = [
        color
        for color, _bar in sorted(scene["bars"].items(), key=lambda item: (item[1]["col"], item[1]["cells"][0][0]))
    ]
    if path_order == bar_order:
        return False

    changed = np.argwhere(grid != out)
    if changed.size == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _make_case(rng)
        if grid is not None and _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid 条高定节拍 input")
