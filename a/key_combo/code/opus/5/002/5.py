from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SEED_COLORS = (RED, GREEN)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return out


def _find_axes(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    axes: list[dict] = []
    for r in range(h):
        if np.all(grid[r, :] == GREY):
            axes.append({"orient": "horizontal", "index": r})
    for c in range(w):
        if np.all(grid[:, c] == GREY):
            axes.append({"orient": "vertical", "index": c})
    return axes


def _component_side(component: dict, orient: str, axis: int) -> int | None:
    r0, c0, r1, c1 = component["bbox"]
    if orient == "horizontal":
        if r1 < axis:
            return -1
        if r0 > axis:
            return 1
        return None
    if c1 < axis:
        return -1
    if c0 > axis:
        return 1
    return None


def _touches_axis(component: dict, orient: str, axis: int) -> bool:
    if orient == "horizontal":
        return any(abs(r - axis) == 1 for r, _ in component["coords"])
    return any(abs(c - axis) == 1 for _, c in component["coords"])


def _mirror_coords(coords: list[tuple[int, int]], orient: str, axis: int) -> list[tuple[int, int]]:
    if orient == "horizontal":
        return [(2 * axis - r, c) for r, c in coords]
    return [(r, 2 * axis - c) for r, c in coords]


def _axis_distance(orient: str, axis: int, cell: tuple[int, int]) -> int:
    r, c = cell
    return abs(r - axis) if orient == "horizontal" else abs(c - axis)


def _source_for_axis(grid: np.ndarray, axis: dict, blue_components: list[dict]) -> dict:
    orient = axis["orient"]
    index = int(axis["index"])
    candidates = [
        comp
        for comp in blue_components
        if _component_side(comp, orient, index) is not None and _touches_axis(comp, orient, index)
    ]
    assert len(candidates) == 1, f"expected exactly one blue source for {orient} axis {index}"
    return candidates[0]


def _choose_fill_color(
    mirror_coords: list[tuple[int, int]],
    seed_components: list[dict],
    orient: str,
    axis: int,
) -> tuple[int, dict[int, int], dict[int, int]]:
    mirror_set = set(mirror_coords)
    counts = {RED: 0, GREEN: 0}
    dists = {RED: 10**9, GREEN: 10**9}

    for comp in seed_components:
        pairs = 0
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in mirror_set:
                    pairs += 1
        if pairs == 0:
            continue
        color = int(comp["color"])
        counts[color] += pairs
        comp_dist = min(_axis_distance(orient, axis, cell) for cell in comp["coords"])
        dists[color] = min(dists[color], comp_dist)

    best = sorted(
        SEED_COLORS,
        key=lambda color: (-counts[color], dists[color], color),
    )[0]
    assert counts[best] > 0, "mirror completion must touch at least one seed region"
    return best, counts, dists


def _solve(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    axes = _find_axes(g)
    assert len([a for a in axes if a["orient"] == "horizontal"]) == 1, "expected one full horizontal guide line"
    assert len([a for a in axes if a["orient"] == "vertical"]) == 1, "expected one full vertical guide line"

    blue_components = _components_of_color(g, BLUE)
    seed_components = _components_of_color(g, RED) + _components_of_color(g, GREEN)
    out = g.copy()
    filled_sets: list[set[tuple[int, int]]] = []
    trace: list[dict] = []

    for axis in axes:
        source = _source_for_axis(g, axis, blue_components)
        mirror_coords = _mirror_coords(source["coords"], axis["orient"], int(axis["index"]))
        assert all(_in_bounds(g.shape, r, c) for r, c in mirror_coords), "mirror left grid bounds"
        mirror_set = set(mirror_coords)
        for other in filled_sets:
            assert not (mirror_set & other), "mirror completion regions must be disjoint"
        fill_color, counts, dists = _choose_fill_color(
            mirror_coords,
            seed_components,
            axis["orient"],
            int(axis["index"]),
        )
        for r, c in mirror_coords:
            out[r, c] = fill_color
        filled_sets.append(mirror_set)
        trace.append(
            {
                "axis": axis,
                "source": source,
                "mirror_coords": mirror_coords,
                "fill_color": fill_color,
                "counts": counts,
                "dists": dists,
            }
        )

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each blue shape across its gray guide line, then color only the completed half from neighboring seed regions.

    Input structure:
    - A black background contains exactly one full gray horizontal line and one full gray vertical line.
    - Each guide line has one irregular 4-connected blue shape entirely on one side, with at least one blue cell touching the line.
    - On the opposite side of each guide line are small red and/or green seed regions near the would-be mirrored shape.

    Rule:
    - Detect every full gray row or column as a mirror axis.
    - For each axis, find the unique blue component that lies wholly on one side of that axis and touches it.
    - Reflect that blue component across the axis to obtain the completion region.
    - Count all 4-neighbor adjacency pairs between the completion region and red/green seed pixels.
    - Fill the completion region with the color having the larger total adjacency count; if tied, use the color whose adjacent seed region is closer to the guide line along the axis-normal direction.
    - Keep the original blue shapes, gray guide lines, and seed regions unchanged.

    Key insight:
    - The gray line determines where to mirror, but only the neighboring seed regions determine what color the mirrored completion receives.
    """
    return _solve(grid)


GRID_MIN = 18
GRID_MAX = 24
SHAPE_ATTEMPTS = 80
GRID_ATTEMPTS = 300
TEMPLATES = [
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]
CASE_MODES = ("red_wins", "green_wins", "tie_red_close", "tie_green_close")


def _grow_shape(depth: int, width: int, area: int) -> list[tuple[int, int]] | None:
    start = (0, random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < depth and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    if max(r for r, _ in cells) < 1:
        return None
    r0, c0, r1, c1 = _bbox(list(cells))
    if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
        return None
    return list(cells)


def _map_source_cells(
    orient: str,
    axis: int,
    side: int,
    tangential_origin: int,
    local_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    coords = []
    for d0, t0 in local_cells:
        dist = d0 + 1
        if orient == "horizontal":
            r = axis + side * dist
            c = tangential_origin + t0
        else:
            r = tangential_origin + t0
            c = axis + side * dist
        coords.append((r, c))
    return coords


def _component_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _candidate_regions(
    shape: tuple[int, int],
    orient: str,
    axis: int,
    mirror_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    rs = [r for r, _ in mirror_set]
    cs = [c for _, c in mirror_set]
    r0 = max(0, min(rs) - 2)
    r1 = min(h - 1, max(rs) + 2)
    c0 = max(0, min(cs) - 2)
    c1 = min(w - 1, max(cs) + 2)
    out: list[dict] = []
    seen = set()

    for template in TEMPLATES:
        tr_max = max(r for r, _ in template)
        tc_max = max(c for _, c in template)
        for ar in range(r0, r1 - tr_max + 1):
            for ac in range(c0, c1 - tc_max + 1):
                cells = {(ar + dr, ac + dc) for dr, dc in template}
                if cells & blocked:
                    continue
                if any(cell in mirror_set for cell in cells):
                    continue
                if orient == "horizontal":
                    dists = [abs(r - axis) for r, _ in cells]
                else:
                    dists = [abs(c - axis) for _, c in cells]
                if min(dists) < 1 or min(dists) > 3:
                    continue
                pairs = 0
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in mirror_set:
                            pairs += 1
                if pairs == 0:
                    continue
                key = tuple(sorted(cells))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "cells": cells,
                        "pairs": pairs,
                        "dist": min(dists),
                    }
                )
    random.shuffle(out)
    return out


def _pick_seed_pair(candidates: list[dict], mode: str) -> tuple[dict, dict] | None:
    valid: list[tuple[dict, dict]] = []
    for a in candidates:
        for b in candidates:
            if a is b:
                continue
            if a["cells"] & b["cells"]:
                continue
            if _component_adjacent(a["cells"], b["cells"]):
                continue
            if mode == "red_wins" and a["pairs"] > b["pairs"]:
                valid.append((a, b))
            elif mode == "green_wins" and b["pairs"] > a["pairs"]:
                valid.append((a, b))
            elif mode == "tie_red_close" and a["pairs"] == b["pairs"] and a["dist"] < b["dist"]:
                valid.append((a, b))
            elif mode == "tie_green_close" and a["pairs"] == b["pairs"] and b["dist"] < a["dist"]:
                valid.append((a, b))
    if not valid:
        return None
    return random.choice(valid)


def _place_case(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    *,
    orient: str,
    axis: int,
    tangential_span: tuple[int, int],
    side: int,
    mode: str,
) -> dict | None:
    h, w = grid.shape
    band_len = tangential_span[1] - tangential_span[0] + 1
    for _ in range(SHAPE_ATTEMPTS):
        depth = random.randint(2, 4)
        width = random.randint(3, min(6, band_len))
        area = random.randint(5, min(10, depth * width - 1))
        local = _grow_shape(depth, width, area)
        if local is None:
            continue
        tangential_origin = random.randint(tangential_span[0], tangential_span[1] - width + 1)
        source = _map_source_cells(orient, axis, side, tangential_origin, local)
        mirror = _mirror_coords(source, orient, axis)
        source_set = set(source)
        mirror_set = set(mirror)
        if len(mirror_set) != len(source_set):
            continue
        if any(not _in_bounds((h, w), r, c) for r, c in source + mirror):
            continue
        if source_set & occupied:
            continue
        if mirror_set & occupied:
            continue
        if source_set & mirror_set:
            continue
        blocked = occupied | source_set | mirror_set
        candidates = _candidate_regions((h, w), orient, axis, mirror_set, blocked)
        picked = _pick_seed_pair(candidates, mode)
        if picked is None:
            continue
        red_region, green_region = picked
        for r, c in source:
            grid[r, c] = BLUE
        for r, c in red_region["cells"]:
            grid[r, c] = RED
        for r, c in green_region["cells"]:
            grid[r, c] = GREEN
        occupied |= source_set | red_region["cells"] | green_region["cells"]
        return {
            "orient": orient,
            "axis": axis,
            "mode": mode,
            "source": source_set,
            "mirror": mirror_set,
            "seeds": {RED: red_region, GREEN: green_region},
        }
    return None


def _semantic_checks(grid: np.ndarray, meta: list[dict]) -> bool:
    try:
        out, trace = _solve(grid, return_trace=True)
    except AssertionError:
        return False

    if len(trace) != 2 or len(meta) != 2:
        return False
    if np.array_equal(out, grid):
        return False

    axes = {(item["axis"]["orient"], item["axis"]["index"]): item for item in trace}
    meta_axes = {(item["orient"], item["axis"]): item for item in meta}
    if set(axes) != set(meta_axes):
        return False

    all_mirror = set()
    seen_fill_colors = set()
    tie_seen = False
    win_seen = False

    for key, info in axes.items():
        expected = meta_axes[key]
        mirror_set = set(info["mirror_coords"])
        if mirror_set != expected["mirror"]:
            return False
        if mirror_set & all_mirror:
            return False
        all_mirror |= mirror_set
        if any(int(grid[r, c]) != BG for r, c in mirror_set):
            return False
        counts = info["counts"]
        dists = info["dists"]
        fill = int(info["fill_color"])
        seen_fill_colors.add(fill)
        if counts[RED] == counts[GREEN]:
            tie_seen = True
            if dists[RED] == dists[GREEN]:
                return False
        else:
            win_seen = True
        if fill not in SEED_COLORS:
            return False
        for color in SEED_COLORS:
            region = expected["seeds"][color]
            actual_pairs = 0
            for r, c in region["cells"]:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in mirror_set:
                        actual_pairs += 1
            if actual_pairs != region["pairs"]:
                return False
            if region["dist"] != min(_axis_distance(key[0], key[1], cell) for cell in region["cells"]):
                return False

    if not tie_seen and not win_seen:
        return False
    if len(seen_fill_colors) < 1:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    h_axis = random.randint(4, h - 5)
    v_axis = random.randint(4, w - 5)
    grid[h_axis, :] = GREY
    grid[:, v_axis] = GREY
    occupied = {(h_axis, c) for c in range(w)} | {(r, v_axis) for r in range(h)}

    left_cols = (1, v_axis - 4)
    right_cols = (v_axis + 4, w - 2)
    top_rows = (1, h_axis - 4)
    bottom_rows = (h_axis + 4, h - 2)
    horiz_spans = [span for span in (left_cols, right_cols) if span[1] - span[0] + 1 >= 3]
    vert_spans = [span for span in (top_rows, bottom_rows) if span[1] - span[0] + 1 >= 3]
    if not horiz_spans or not vert_spans:
        return None

    h_span = random.choice(horiz_spans)
    v_span = random.choice(vert_spans)
    h_side = random.choice([-1, 1])
    v_side = random.choice([-1, 1])

    modes = random.sample(CASE_MODES, 2)
    meta: list[dict] = []

    first = _place_case(
        grid,
        occupied,
        orient="horizontal",
        axis=h_axis,
        tangential_span=h_span,
        side=h_side,
        mode=modes[0],
    )
    if first is None:
        return None
    meta.append(first)

    second = _place_case(
        grid,
        occupied,
        orient="vertical",
        axis=v_axis,
        tangential_span=v_span,
        side=v_side,
        mode=modes[1],
    )
    if second is None:
        return None
    meta.append(second)

    if not _semantic_checks(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-complete seed-color input")
