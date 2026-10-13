import itertools
import random
from collections import Counter, defaultdict, deque

import numpy as np


LINE_COLORS = (1, 2, 3, 4)
LINE_KINDS = ("H", "V", "D", "U")
GEN_ATTEMPTS = 800


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _line_cells(shape: tuple[int, int], kind: str, param: int) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    if kind == "H":
        for c in range(w):
            cells.add((param, c))
    elif kind == "V":
        for r in range(h):
            cells.add((r, param))
    elif kind == "D":
        for r in range(h):
            c = r - param
            if 0 <= c < w:
                cells.add((r, c))
    elif kind == "U":
        for r in range(h):
            c = param - r
            if 0 <= c < w:
                cells.add((r, c))
    else:
        raise ValueError(f"unknown line kind: {kind}")
    return cells


def _infer_line(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == color)
    if len(coords) < 2:
        raise AssertionError(f"color {color} must have at least two visible pixels")

    rs = coords[:, 0]
    cs = coords[:, 1]
    candidates: list[tuple[str, int]] = []
    if np.all(rs == rs[0]):
        candidates.append(("H", int(rs[0])))
    if np.all(cs == cs[0]):
        candidates.append(("V", int(cs[0])))
    diffs = rs - cs
    if np.all(diffs == diffs[0]):
        candidates.append(("D", int(diffs[0])))
    sums = rs + cs
    if np.all(sums == sums[0]):
        candidates.append(("U", int(sums[0])))

    if len(candidates) != 1:
        raise AssertionError(f"color {color} is not uniquely recoverable as one straight line")

    kind, param = candidates[0]
    return {
        "color": int(color),
        "kind": kind,
        "param": int(param),
        "cells": _line_cells(grid.shape, kind, int(param)),
    }


def _recover_lines(grid: np.ndarray) -> dict[int, dict]:
    bg = _background_color(grid)
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    if set(colors) != set(LINE_COLORS):
        raise AssertionError("expected exactly the four line colors 1,2,3,4")

    lines = {color: _infer_line(grid, color) for color in colors}

    covered = set()
    for spec in lines.values():
        covered |= spec["cells"]

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg and (r, c) in covered:
                raise AssertionError("every recovered line pixel must be visibly occupied by some line")
            if grid[r, c] != bg and (r, c) not in lines[int(grid[r, c])]["cells"]:
                raise AssertionError("visible pixel must belong to its recovered line")
    return lines


def _cell_to_lines(lines: dict[int, dict]) -> dict[tuple[int, int], set[int]]:
    out: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, spec in lines.items():
        for cell in spec["cells"]:
            out[cell].add(int(color))
    return out


def _valid_orders(grid: np.ndarray, lines: dict[int, dict]) -> list[tuple[int, ...]]:
    cell_lines = _cell_to_lines(lines)
    colors = sorted(lines)
    below_edges: set[tuple[int, int]] = set()
    intersects: dict[int, set[int]] = {color: set() for color in colors}

    for cell, covering in cell_lines.items():
        if len(covering) < 2:
            continue
        visible = int(grid[cell])
        if visible not in covering:
            raise AssertionError("visible crossing color must be one of the covering lines")
        ordered_covering = sorted(covering)
        for a, b in itertools.combinations(ordered_covering, 2):
            intersects[a].add(b)
            intersects[b].add(a)
        for color in covering:
            if color != visible:
                below_edges.add((color, visible))

    valid = []
    for order in itertools.permutations(colors):
        pos = {color: idx for idx, color in enumerate(order)}
        if any(pos[below] >= pos[above] for below, above in below_edges):
            continue
        if any(order[i + 1] not in intersects[order[i]] for i in range(len(order) - 1)):
            continue
        valid.append(tuple(int(v) for v in order))
    return valid


def _regions(shape: tuple[int, int], lines: dict[int, dict]) -> list[dict]:
    h, w = shape
    cell_lines = _cell_to_lines(lines)
    barrier = set(cell_lines)
    seen: set[tuple[int, int]] = set()
    regions = []

    for r in range(h):
        for c in range(w):
            if (r, c) in barrier or (r, c) in seen:
                continue

            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []
            touches_edge = False
            adjacent_colors: set[int] = set()
            adjacent_counter: Counter[int] = Counter()

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in barrier:
                        for color in cell_lines[(nr, nc)]:
                            adjacent_colors.add(int(color))
                            adjacent_counter[int(color)] += 1
                    elif (nr, nc) not in seen:
                        seen.add((nr, nc))
                        queue.append((nr, nc))

            regions.append(
                {
                    "cells": cells,
                    "touches_edge": touches_edge,
                    "adjacent_colors": adjacent_colors,
                    "adjacent_counter": adjacent_counter,
                }
            )
    return regions


def _fill_regions(grid: np.ndarray, lines: dict[int, dict], order: tuple[int, ...]) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    rank = {color: idx for idx, color in enumerate(order)}
    regions = _regions(grid.shape, lines)

    for region in regions:
        if region["touches_edge"]:
            region["fill_color"] = 0
            continue
        if not region["adjacent_colors"]:
            raise AssertionError("interior region must touch at least one line")
        fill_color = min(region["adjacent_colors"], key=lambda color: rank[color])
        region["fill_color"] = int(fill_color)
        for r, c in region["cells"]:
            out[r, c] = int(fill_color)
    return out, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background region with the color of its lowest-layer boundary line.

    Input structure:
    - The background is black, and the only foreground colors are blue, red, green, and yellow.
    - Each foreground color is the visible part of one complete one-pixel straight line: horizontal,
      vertical, or a 45-degree diagonal.
    - Lines may cross and occlude each other, so some line pixels are hidden at crossings.

    Rule:
    - Recover every full line from its visible collinear pixels, including cells hidden by crossings.
    - Use crossing visibility to infer the unique bottom-to-top layer order: at each crossing,
      the visible color is above every other line through that cell.
    - Treat the recovered full lines as barriers that split the black background into 4-connected regions.
    - Recolor each region that does not touch the grid edge with the color of the lowest-layer line
      among all boundary lines adjacent to that region; keep edge-touching background black and keep
      all visible line pixels unchanged.

    Key insight:
    The fill color is not chosen from local boundary appearance alone: it comes from the global
    occlusion order inferred from line crossings, then applied to enclosed regions found by flood fill.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")

    lines = _recover_lines(g)
    orders = _valid_orders(g, lines)
    if len(orders) != 1:
        raise AssertionError("line layering must have a unique bottom-to-top order")
    out, _ = _fill_regions(g, lines, orders[0])
    return out


def _single_intersections(lines_by_kind: dict[str, set[tuple[int, int]]]) -> dict[tuple[str, str], tuple[int, int]] | None:
    intersections = {}
    points = []
    for a, b in itertools.combinations(LINE_KINDS, 2):
        inter = lines_by_kind[a] & lines_by_kind[b]
        if len(inter) != 1:
            return None
        point = next(iter(inter))
        intersections[(a, b)] = point
        points.append(point)
    if len(set(points)) != len(points):
        return None
    return intersections


def _render_input(shape: tuple[int, int], lines: dict[int, dict], order: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in order:
        for r, c in lines[int(color)]["cells"]:
            grid[r, c] = int(color)
    return grid


def _line_specs_from_kind_map(shape: tuple[int, int], kind_to_color: dict[str, int], params: dict[str, int]) -> dict[int, dict]:
    lines = {}
    for kind in LINE_KINDS:
        color = int(kind_to_color[kind])
        param = int(params[kind])
        lines[color] = {
            "color": color,
            "kind": kind,
            "param": param,
            "cells": _line_cells(shape, kind, param),
        }
    return lines


def _pick_sample(rng: random.Random) -> tuple[np.ndarray, tuple[int, ...]]:
    n = rng.randint(13, 17)
    params = {
        "H": rng.randint(2, n - 3),
        "V": rng.randint(2, n - 3),
        "D": rng.randint(-(n - 4), n - 4),
        "U": rng.randint(3, 2 * n - 4),
    }
    lines_by_kind = {kind: _line_cells((n, n), kind, params[kind]) for kind in LINE_KINDS}
    if _single_intersections(lines_by_kind) is None:
        raise AssertionError("pairwise intersections must be unique")

    colors = list(LINE_COLORS)
    rng.shuffle(colors)
    kind_to_color = {kind: colors[idx] for idx, kind in enumerate(LINE_KINDS)}
    lines = _line_specs_from_kind_map((n, n), kind_to_color, params)

    order_list = list(LINE_COLORS)
    rng.shuffle(order_list)
    order = tuple(order_list)

    grid = _render_input((n, n), lines, order)
    return grid, order


def _validate_generated(grid: np.ndarray, intended_order: tuple[int, ...]) -> bool:
    try:
        lines = _recover_lines(grid)
        orders = _valid_orders(grid, lines)
    except AssertionError:
        return False
    if orders != [tuple(int(v) for v in intended_order)]:
        return False

    expected, regions = _fill_regions(grid, lines, intended_order)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    internal = [region for region in regions if not region["touches_edge"]]
    if len(internal) < 2:
        return False
    if not any(len(region["adjacent_colors"]) >= 3 for region in internal):
        return False

    fills = [int(region["fill_color"]) for region in internal]
    if len(set(fills)) < 2:
        return False
    if np.array_equal(actual, grid):
        return False

    if not any(fill != min(region["adjacent_colors"]) for fill, region in zip(fills, internal)):
        return False
    if not any(fill != max(region["adjacent_colors"]) for fill, region in zip(fills, internal)):
        return False

    def mode_color(region: dict) -> int:
        return max(region["adjacent_counter"], key=lambda color: (region["adjacent_counter"][color], -color))

    if not any(fill != mode_color(region) for fill, region in zip(fills, internal)):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, order = _pick_sample(rng)
        except AssertionError:
            continue
        try:
            valid = _validate_generated(grid, order)
        except AssertionError:
            valid = False
        if valid:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
