import itertools
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

MIN_SIZE = 10
MAX_SIZE = 16
MIN_PATHS = 2
MAX_PATHS = 4
GEN_RETRIES = 500
PATH_RETRIES = 120

H_SIDES = ("top", "bottom")
V_SIDES = ("left", "right")
CROSSING_STATES = ("high", "low", "bg")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _on_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _line_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line_cells requires axis-aligned endpoints")


def _path_cells_from_bend(ep_h: Coord, ep_v: Coord, bend: Coord) -> set[Coord]:
    return set(_line_cells(ep_h, bend)) | set(_line_cells(bend, ep_v))


def _corner_distance(p: Coord, h: int, w: int) -> int:
    r, c = p
    return min(r + c, r + (w - 1 - c), (h - 1 - r) + c, (h - 1 - r) + (w - 1 - c))


def _positions_with_edge_distance(length: int, dist: int) -> list[int]:
    if dist <= 0:
        return []
    hi = length - 1 - dist
    if hi < dist:
        return []
    positions = {dist, hi}
    return sorted(p for p in positions if 0 < p < length - 1)


def _spec_endpoints(spec: dict, h: int, w: int) -> tuple[Coord, Coord]:
    row = int(spec["row"])
    col = int(spec["col"])
    ep_h = (0, col) if spec["h_side"] == "top" else (h - 1, col)
    ep_v = (row, 0) if spec["v_side"] == "left" else (row, w - 1)
    return ep_h, ep_v


def _spec_bends(spec: dict, h: int, w: int) -> tuple[Coord, Coord]:
    ep_h, ep_v = _spec_endpoints(spec, h, w)
    return (ep_v[0], ep_h[1]), (ep_h[0], ep_v[1])


def _spec_cells(spec: dict, h: int, w: int) -> set[Coord]:
    ep_h, ep_v = _spec_endpoints(spec, h, w)
    bend, _corner = _spec_bends(spec, h, w)
    return _path_cells_from_bend(ep_h, ep_v, bend)


def _orientation_at(spec: dict, cell: Coord, h: int, w: int) -> str | None:
    ep_h, ep_v = _spec_endpoints(spec, h, w)
    bend, _corner = _spec_bends(spec, h, w)
    if cell == bend or cell == ep_h or cell == ep_v:
        return None

    row = int(spec["row"])
    col = int(spec["col"])
    if cell[1] == col:
        lo = min(ep_h[0], row)
        hi = max(ep_h[0], row)
        if lo < cell[0] < hi:
            return "V"
    if cell[0] == row:
        lo = min(ep_v[1], col)
        hi = max(ep_v[1], col)
        if lo < cell[1] < hi:
            return "H"
    return None


def _render_output_from_specs(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    cover: dict[Coord, list[dict]] = {}
    for spec in specs:
        for cell in _spec_cells(spec, h, w):
            cover.setdefault(cell, []).append(spec)

    for cell, owners in cover.items():
        top = min(owners, key=lambda sp: tuple(sp["rank_key"]))
        out[cell] = int(top["color"])
    return out


def _infer_paths(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected at least one path color"

    paths: list[dict] = []
    for color in colors:
        coords = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        border = [p for p in coords if _on_border(p[0], p[1], h, w)]
        assert len(border) == 2, "each path color must have exactly two border endpoints"

        h_endpoints = [p for p in border if p[0] in (0, h - 1)]
        v_endpoints = [p for p in border if p[1] in (0, w - 1)]
        assert len(h_endpoints) == 1 and len(v_endpoints) == 1, (
            "each path must connect one horizontal edge to one vertical edge"
        )

        ep_h = h_endpoints[0]
        ep_v = v_endpoints[0]
        bend_inside = (ep_v[0], ep_h[1])
        bend_corner = (ep_h[0], ep_v[1])
        candidates = [
            _path_cells_from_bend(ep_h, ep_v, bend_inside),
            _path_cells_from_bend(ep_h, ep_v, bend_corner),
        ]

        valid: list[tuple[tuple[int, int, int], set[Coord], Coord]] = []
        for cells, bend in zip(candidates, (bend_inside, bend_corner)):
            if not coords.issubset(cells):
                continue
            occupied = sum(1 for r, c in cells if g[r, c] != bg)
            own = sum(1 for r, c in cells if g[r, c] == color)
            empties = sum(1 for r, c in cells if g[r, c] == bg)
            valid.append(((occupied, own, -empties), cells, bend))

        assert valid, "no candidate path matches visible pixels"
        valid.sort(key=lambda item: item[0], reverse=True)
        best_score, best_cells, best_bend = valid[0]
        if len(valid) > 1:
            assert best_score != valid[1][0], "path reconstruction is ambiguous"

        d0 = _corner_distance(ep_h, h, w)
        d1 = _corner_distance(ep_v, h, w)
        paths.append(
            {
                "color": color,
                "cells": best_cells,
                "bend": best_bend,
                "ep_h": ep_h,
                "ep_v": ep_v,
                "rank_key": (min(d0, d1), max(d0, d1)),
            }
        )

    ranks = [tuple(path["rank_key"]) for path in paths]
    assert len(ranks) == len(set(ranks)), "path layer order must be unique"
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fix intersection colors in a network of colored edge-to-edge orthogonal paths.

    Each path color appears on exactly two edge endpoints: one on a horizontal edge and one on a vertical edge.
    The intended path is the interior L-shape between those endpoints; the border-corner L-shape is a decoy.
    At cells where reconstructed paths cross, the path whose endpoints are closer to a corner has higher priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    inferred = _infer_paths(g)

    specs: list[dict] = []
    for path in inferred:
        specs.append(
            {
                "color": int(path["color"]),
                "rank_key": tuple(path["rank_key"]),
                "cells": set(path["cells"]),
            }
        )

    h, w = g.shape
    out = np.full_like(g, bg)
    cover: dict[Coord, list[dict]] = {}
    for spec in specs:
        for cell in spec["cells"]:
            cover.setdefault(cell, []).append(spec)

    for cell, owners in cover.items():
        top = min(owners, key=lambda sp: tuple(sp["rank_key"]))
        out[cell] = int(top["color"])
    return out


def _sample_path_spec(
    h: int,
    w: int,
    depth: int,
    color: int,
    used_rows: set[int],
    used_cols: set[int],
    rng: random.Random,
) -> dict | None:
    row_max = (h - 1) // 2
    col_max = (w - 1) // 2
    if depth > row_max or depth > col_max:
        return None

    for _ in range(PATH_RETRIES):
        h_side = rng.choice(H_SIDES)
        v_side = rng.choice(V_SIDES)
        mode = rng.choice(("row", "col", "both"))

        if mode == "row":
            row_dist = depth
            col_dist = rng.randint(depth + 1, col_max) if depth < col_max else depth
        elif mode == "col":
            col_dist = depth
            row_dist = rng.randint(depth + 1, row_max) if depth < row_max else depth
        else:
            row_dist = depth
            col_dist = depth

        row_choices = [r for r in _positions_with_edge_distance(h, row_dist) if r not in used_rows]
        col_choices = [c for c in _positions_with_edge_distance(w, col_dist) if c not in used_cols]
        if not row_choices or not col_choices:
            continue

        row = rng.choice(row_choices)
        col = rng.choice(col_choices)
        spec = {
            "color": color,
            "h_side": h_side,
            "v_side": v_side,
            "row": row,
            "col": col,
        }

        ep_h, ep_v = _spec_endpoints(spec, h, w)
        d0 = _corner_distance(ep_h, h, w)
        d1 = _corner_distance(ep_v, h, w)
        if min(d0, d1) != depth:
            continue
        if len(_spec_cells(spec, h, w)) < 5:
            continue
        return spec
    return None


def _crossings_for_specs(specs: list[dict], h: int, w: int) -> dict[Coord, tuple[int, int]]:
    crossings: dict[Coord, tuple[int, int]] = {}
    cover: dict[Coord, list[int]] = {}

    for idx, spec in enumerate(specs):
        for cell in _spec_cells(spec, h, w):
            cover.setdefault(cell, []).append(idx)

    if any(len(owners) > 2 for owners in cover.values()):
        return {}

    for i, j in itertools.combinations(range(len(specs)), 2):
        a = specs[i]
        b = specs[j]
        inter = _spec_cells(a, h, w) & _spec_cells(b, h, w)
        if len(inter) > 1:
            return {}
        if not inter:
            continue

        cell = next(iter(inter))
        if _orientation_at(a, cell, h, w) == "V" and _orientation_at(b, cell, h, w) == "H":
            crossings[cell] = (i, j)
            continue
        if _orientation_at(a, cell, h, w) == "H" and _orientation_at(b, cell, h, w) == "V":
            crossings[cell] = (i, j)
            continue
        return {}

    return crossings


def _render_input(specs: list[dict], shape: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    h, w = shape
    crossings = _crossings_for_specs(specs, h, w)
    if not crossings:
        return None

    by_cell: dict[Coord, list[dict]] = {}
    for spec in specs:
        for cell in _spec_cells(spec, h, w):
            by_cell.setdefault(cell, []).append(spec)

    states = {}
    cells = list(crossings.keys())
    for cell in cells:
        states[cell] = rng.choice(CROSSING_STATES)
    if all(states[cell] == "high" for cell in cells):
        flip = rng.choice(cells)
        states[flip] = rng.choice(("low", "bg"))

    grid = np.full((h, w), BG, dtype=int)
    for cell, owners in by_cell.items():
        if len(owners) == 1:
            grid[cell] = int(owners[0]["color"])
            continue

        top = min(owners, key=lambda sp: tuple(sp["rank_key"]))
        low = max(owners, key=lambda sp: tuple(sp["rank_key"]))
        state = states[cell]
        if state == "high":
            grid[cell] = int(top["color"])
        elif state == "low":
            grid[cell] = int(low["color"])
        else:
            grid[cell] = BG

    return grid


def _visible_path_pixels(grid: np.ndarray, color: int) -> set[Coord]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid == color)}


def _valid_rendered_input(grid: np.ndarray, specs: list[dict]) -> bool:
    h, w = grid.shape
    for spec in specs:
        color = int(spec["color"])
        visible = _visible_path_pixels(grid, color)
        border = {spec["ep_h"], spec["ep_v"]}
        if len(visible - border) < 2:
            return False

        ep_h = spec["ep_h"]
        ep_v = spec["ep_v"]
        inside_bend = (ep_v[0], ep_h[1])
        corner_bend = (ep_h[0], ep_v[1])
        inside = _path_cells_from_bend(ep_h, ep_v, inside_bend)
        corner = _path_cells_from_bend(ep_h, ep_v, corner_bend)
        if not visible.issubset(inside):
            return False
        if visible.issubset(corner):
            return False

    solved = main(grid)
    target = _render_output_from_specs(specs, grid.shape, BG)
    return np.array_equal(solved, target)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    max_depth = min((h - 1) // 2, (w - 1) // 2)
    if max_depth < MIN_PATHS:
        return None

    num_paths = rng.randint(MIN_PATHS, min(MAX_PATHS, max_depth))
    depths = rng.sample(list(range(1, max_depth + 1)), num_paths)
    colors = rng.sample(PALETTE, num_paths)

    specs: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for depth, color in zip(depths, colors):
        spec = _sample_path_spec(h, w, depth, color, used_rows, used_cols, rng)
        if spec is None:
            return None
        ep_h, ep_v = _spec_endpoints(spec, h, w)
        d0 = _corner_distance(ep_h, h, w)
        d1 = _corner_distance(ep_v, h, w)
        spec["ep_h"] = ep_h
        spec["ep_v"] = ep_v
        spec["rank_key"] = (min(d0, d1), max(d0, d1))
        specs.append(spec)
        used_rows.add(int(spec["row"]))
        used_cols.add(int(spec["col"]))

    if len({tuple(spec["rank_key"]) for spec in specs}) != len(specs):
        return None

    crossings = _crossings_for_specs(specs, h, w)
    if not crossings:
        return None
    if len(crossings) > max(4, num_paths + 1):
        return None

    grid = _render_input(specs, (h, w), rng)
    if grid is None:
        return None
    if np.array_equal(grid, _render_output_from_specs(specs, (h, w), BG)):
        return None
    if not _valid_rendered_input(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
