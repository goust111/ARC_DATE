import random
from collections import Counter

import numpy as np


BG = 0
BREAK = 9
PATH_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES = ("h", "v", "L", "Zh", "Zv")
GEN_ATTEMPTS = 500
PATH_ATTEMPTS = 120


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _build_path(vertices: list[tuple[int, int]]) -> dict | None:
    points: list[tuple[int, int]] = []
    seg_cells: list[list[tuple[int, int]]] = []
    for i in range(len(vertices) - 1):
        seg = _segment_points(vertices[i], vertices[i + 1])
        seg_cells.append(seg)
        if i:
            seg = seg[1:]
        points.extend(seg)
    if len(set(points)) != len(points):
        return None
    index = {pt: i for i, pt in enumerate(points)}
    return {
        "vertices": tuple(vertices),
        "points": tuple(points),
        "cells": frozenset(points),
        "segments": tuple(tuple(seg) for seg in seg_cells),
        "turns": frozenset(vertices[1:-1]),
        "endpoints": frozenset((vertices[0], vertices[-1])),
        "index": index,
    }


def _point_orientation(path: dict, point: tuple[int, int]) -> str | None:
    idx = path["index"].get(point)
    if idx is None or idx == 0 or idx == len(path["points"]) - 1:
        return None
    prev_pt = path["points"][idx - 1]
    next_pt = path["points"][idx + 1]
    if prev_pt[0] == point[0] == next_pt[0]:
        return "h"
    if prev_pt[1] == point[1] == next_pt[1]:
        return "v"
    return None


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _legend_order(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    cols = np.flatnonzero(row != bg)
    assert cols.size >= 3
    start = int(cols[0])
    end = int(cols[-1])
    assert cols.size == end - start + 1
    colors = [int(v) for v in row[start : end + 1].tolist()]
    assert len(colors) == len(set(colors))
    return colors


def _candidate_vertices(r0: int, r1: int, c0: int, c1: int) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    if r0 == r1:
        out.append([(r0, c0), (r0, c1)])
        return out
    if c0 == c1:
        out.append([(r0, c0), (r1, c0)])
        return out

    out.extend(
        [
            [(r0, c1), (r0, c0), (r1, c0)],
            [(r0, c0), (r0, c1), (r1, c1)],
            [(r0, c0), (r1, c0), (r1, c1)],
            [(r0, c1), (r1, c1), (r1, c0)],
        ]
    )
    for cm in range(c0 + 1, c1):
        out.append([(r0, c0), (r0, cm), (r1, cm), (r1, c1)])
        out.append([(r0, c1), (r0, cm), (r1, cm), (r1, c0)])
    for rm in range(r0 + 1, r1):
        out.append([(r0, c0), (rm, c0), (rm, c1), (r1, c1)])
        out.append([(r0, c1), (rm, c1), (rm, c0), (r1, c0)])
    return out


def _candidate_valid(grid_main: np.ndarray, color: int, visible: set[tuple[int, int]], path: dict, bg: int) -> bool:
    if not visible.issubset(path["cells"]):
        return False
    if not path["endpoints"].issubset(visible):
        return False
    if not path["turns"].issubset(visible):
        return False

    for seg in path["segments"]:
        if not any(pt in visible for pt in seg):
            return False

    h, w = grid_main.shape
    hidden = path["cells"] - visible
    for r, c in hidden:
        if int(grid_main[r, c]) == bg:
            return False
        if (r, c) in path["turns"] or (r, c) in path["endpoints"]:
            return False
        orient = _point_orientation(path, (r, c))
        if orient is None:
            return False
        other = int(grid_main[r, c])
        if orient == "h":
            if r - 1 < 0 or r + 1 >= h:
                return False
            if int(grid_main[r - 1, c]) != other or int(grid_main[r + 1, c]) != other:
                return False
        else:
            if c - 1 < 0 or c + 1 >= w:
                return False
            if int(grid_main[r, c - 1]) != other or int(grid_main[r, c + 1]) != other:
                return False

    path["hidden"] = frozenset(hidden)
    path["score"] = (len(hidden), len(path["turns"]), len(path["cells"]))
    return True


def _enumerate_candidates(grid_main: np.ndarray, color: int, bg: int) -> list[dict]:
    coords = {(int(r), int(c)) for r, c in np.argwhere(grid_main == color)}
    assert coords
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    candidates: dict[frozenset[tuple[int, int]], dict] = {}
    for vertices in _candidate_vertices(r0, r1, c0, c1):
        path = _build_path(vertices)
        if path is None:
            continue
        key = path["cells"]
        if key in candidates:
            continue
        if _candidate_valid(grid_main, color, coords, path, bg):
            candidates[key] = path
    return sorted(candidates.values(), key=lambda cand: cand["score"])


def _touches_without_cross(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]], shared: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in b and (nr, nc) != (r, c):
                if (r, c) not in shared and (nr, nc) not in shared:
                    return True
    return False


def _partial_ok(color: int, cand: dict, chosen: dict[int, dict], grid_main: np.ndarray) -> bool:
    for other_color, other in chosen.items():
        shared = set(cand["cells"] & other["cells"])
        if len(shared) > 1:
            return False
        if len(shared) == 1:
            point = next(iter(shared))
            seen = int(grid_main[point[0], point[1]])
            if seen not in {color, other_color}:
                return False
            orient_a = _point_orientation(cand, point)
            orient_b = _point_orientation(other, point)
            if orient_a is None or orient_b is None or orient_a == orient_b:
                return False
        if _touches_without_cross(cand["cells"], other["cells"], shared):
            return False
        for point in cand["hidden"]:
            if int(grid_main[point[0], point[1]]) == other_color and point not in other["cells"]:
                return False
        for point in other["hidden"]:
            if int(grid_main[point[0], point[1]]) == color and point not in cand["cells"]:
                return False
    return True


def _final_ok(chosen: dict[int, dict], grid_main: np.ndarray) -> bool:
    owners: Counter[tuple[int, int]] = Counter()
    for cand in chosen.values():
        for point in cand["cells"]:
            owners[point] += 1
    if any(count > 2 for count in owners.values()):
        return False

    for color, cand in chosen.items():
        for point in cand["hidden"]:
            seen = int(grid_main[point[0], point[1]])
            if seen not in chosen or point not in chosen[seen]["cells"]:
                return False
        cross_indices: list[int] = []
        for point in cand["cells"]:
            if owners[point] != 2:
                continue
            idx = cand["index"][point]
            if idx == 0 or idx == len(cand["points"]) - 1:
                return False
            prev_pt = cand["points"][idx - 1]
            next_pt = cand["points"][idx + 1]
            if prev_pt in cand["turns"] or next_pt in cand["turns"]:
                return False
            if prev_pt in cand["endpoints"] or next_pt in cand["endpoints"]:
                return False
            if _point_orientation(cand, point) is None:
                return False
            cross_indices.append(idx)
        cross_indices.sort()
        for a, b in zip(cross_indices, cross_indices[1:]):
            if b - a <= 2:
                return False
    return True


def _select_paths(grid_main: np.ndarray, colors: list[int], bg: int) -> dict[int, dict]:
    candidates = {color: _enumerate_candidates(grid_main, color, bg) for color in colors}
    for color in colors:
        assert candidates[color], f"no candidate path for color {color}"

    order = sorted(colors, key=lambda color: (len(candidates[color]), len(candidates[color][0]["score"]), color))
    chosen: dict[int, dict] = {}

    def backtrack(i: int) -> bool:
        if i == len(order):
            return _final_ok(chosen, grid_main)
        color = order[i]
        for cand in candidates[color]:
            if not _partial_ok(color, cand, chosen, grid_main):
                continue
            chosen[color] = cand
            if backtrack(i + 1):
                return True
            del chosen[color]
        return False

    assert backtrack(0), "failed to reconstruct all paths"
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend = _legend_order(g, bg)
    grid_main = g[:-1, :]

    paths = _select_paths(grid_main, legend, bg)
    rank = {color: i for i, color in enumerate(legend)}

    out_main = np.full_like(grid_main, bg)
    for color in legend:
        for r, c in paths[color]["points"]:
            out_main[r, c] = color

    owners: Counter[tuple[int, int]] = Counter()
    for cand in paths.values():
        for point in cand["cells"]:
            owners[point] += 1

    for point, count in owners.items():
        if count != 2:
            continue
        colors_here = [color for color in legend if point in paths[color]["cells"]]
        assert len(colors_here) == 2
        top = max(colors_here, key=lambda color: rank[color])
        low = min(colors_here, key=lambda color: rank[color])
        out_main[point[0], point[1]] = top
        low_path = paths[low]
        idx = low_path["index"][point]
        for mark in (low_path["points"][idx - 1], low_path["points"][idx + 1]):
            out_main[mark[0], mark[1]] = BREAK

    out = np.full_like(g, bg)
    out[:-1, :] = out_main
    return out


def _sample_sorted(pool: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(pool) < count:
        return None
    for _ in range(60):
        vals = sorted(random.sample(pool, count))
        if all(b - a >= min_gap for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _random_path(row_pool: list[int], col_pool: list[int]) -> dict | None:
    shape = random.choice(SHAPES)
    vertices: list[tuple[int, int]] | None = None

    if shape == "h":
        cols = _sample_sorted(col_pool, 2, 4)
        if cols is None:
            return None
        vertices = [(random.choice(row_pool), cols[0]), (random.choice(row_pool), cols[1])]
        if vertices[0][0] != vertices[1][0]:
            vertices[1] = (vertices[0][0], vertices[1][1])
    elif shape == "v":
        rows = _sample_sorted(row_pool, 2, 4)
        if rows is None:
            return None
        vertices = [(rows[0], random.choice(col_pool)), (rows[1], random.choice(col_pool))]
        if vertices[0][1] != vertices[1][1]:
            vertices[1] = (vertices[1][0], vertices[0][1])
    elif shape == "L":
        rows = _sample_sorted(row_pool, 2, 4)
        cols = _sample_sorted(col_pool, 2, 4)
        if rows is None or cols is None:
            return None
        r0, r1 = rows
        c0, c1 = cols
        elbow = random.choice(("TL", "TR", "BL", "BR"))
        if elbow == "TL":
            vertices = [(r0, c1), (r0, c0), (r1, c0)]
        elif elbow == "TR":
            vertices = [(r0, c0), (r0, c1), (r1, c1)]
        elif elbow == "BL":
            vertices = [(r0, c0), (r1, c0), (r1, c1)]
        else:
            vertices = [(r0, c1), (r1, c1), (r1, c0)]
    elif shape == "Zh":
        rows = _sample_sorted(row_pool, 2, 4)
        cols = _sample_sorted(col_pool, 3, 2)
        if rows is None or cols is None or cols[-1] - cols[0] < 6:
            return None
        r0, r1 = rows
        c0, cm, c1 = cols
        if random.random() < 0.5:
            vertices = [(r0, c0), (r0, cm), (r1, cm), (r1, c1)]
        else:
            vertices = [(r0, c1), (r0, cm), (r1, cm), (r1, c0)]
    else:
        rows = _sample_sorted(row_pool, 3, 2)
        cols = _sample_sorted(col_pool, 2, 4)
        if rows is None or cols is None or rows[-1] - rows[0] < 6:
            return None
        r0, rm, r1 = rows
        c0, c1 = cols
        if random.random() < 0.5:
            vertices = [(r0, c0), (rm, c0), (rm, c1), (r1, c1)]
        else:
            vertices = [(r0, c1), (rm, c1), (rm, c0), (r1, c0)]

    assert vertices is not None
    if random.random() < 0.5:
        vertices = list(reversed(vertices))
    path = _build_path(vertices)
    if path is None or len(path["points"]) < 7:
        return None
    return path


def _pair_cross(a: dict, b: dict) -> tuple[bool, tuple[int, int] | None]:
    shared = a["cells"] & b["cells"]
    if len(shared) > 1:
        return False, None
    if len(shared) == 1:
        point = next(iter(shared))
        orient_a = _point_orientation(a, point)
        orient_b = _point_orientation(b, point)
        if orient_a is None or orient_b is None or orient_a == orient_b:
            return False, None
        return True, point
    if _touches_without_cross(a["cells"], b["cells"], set()):
        return False, None
    return True, None


def _layout_ok(paths: list[dict], min_crossings: int = 1) -> tuple[bool, list[tuple[int, int, int]]]:
    crossings: list[tuple[int, int, int]] = []
    cross_points: set[tuple[int, int]] = set()
    per_path_indices: list[list[int]] = [[] for _ in paths]

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            ok, point = _pair_cross(paths[i], paths[j])
            if not ok:
                return False, []
            if point is None:
                continue
            if point in cross_points:
                return False, []
            cross_points.add(point)
            crossings.append((i, j, paths[i]["index"][point]))
            per_path_indices[i].append(paths[i]["index"][point])
            per_path_indices[j].append(paths[j]["index"][point])

    if not (min_crossings <= len(crossings) <= 4):
        return False, []

    for path, indices in zip(paths, per_path_indices, strict=False):
        indices.sort()
        for idx in indices:
            prev_pt = path["points"][idx - 1]
            next_pt = path["points"][idx + 1]
            if prev_pt in path["turns"] or next_pt in path["turns"]:
                return False, []
            if prev_pt in path["endpoints"] or next_pt in path["endpoints"]:
                return False, []
        for a, b in zip(indices, indices[1:]):
            if b - a <= 2:
                return False, []
    return True, crossings


def _render_case(paths: list[dict], colors: list[int], legend: list[int], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for color, path in zip(colors, paths, strict=False):
        for r, c in path["points"]:
            inp[r, c] = color

    path_by_color = {color: path for color, path in zip(colors, paths, strict=False)}
    rank = {color: i for i, color in enumerate(legend)}

    owners: Counter[tuple[int, int]] = Counter()
    for path in paths:
        for point in path["cells"]:
            owners[point] += 1

    for point, count in owners.items():
        if count != 2:
            continue
        colors_here = [color for color in colors if point in path_by_color[color]["cells"]]
        visible = random.choice(colors_here)
        inp[point[0], point[1]] = visible

    for color in legend:
        for r, c in path_by_color[color]["points"]:
            out[r, c] = color

    for point, count in owners.items():
        if count != 2:
            continue
        colors_here = [color for color in colors if point in path_by_color[color]["cells"]]
        low = min(colors_here, key=lambda color: rank[color])
        low_path = path_by_color[low]
        idx = low_path["index"][point]
        for mark in (low_path["points"][idx - 1], low_path["points"][idx + 1]):
            out[mark[0], mark[1]] = BREAK

    start = random.randint(0, w - len(legend))
    inp[-1, start : start + len(legend)] = np.array(legend, dtype=int)
    out[-1, :] = BG
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    row_pool = list(range(1, h - 2, 2))
    col_pool = list(range(1, w - 1, 2))
    if len(row_pool) < 3 or len(col_pool) < 3:
        return None

    n_paths = random.randint(3, 5)
    colors = random.sample(PATH_COLORS, n_paths)
    legend = random.sample(colors, len(colors))

    paths: list[dict] = []
    current_crossings = 0
    for _ in range(n_paths):
        need_cross = current_crossings == 0 or (current_crossings < 3 and random.random() < 0.65)
        placed = None
        for _ in range(PATH_ATTEMPTS):
            cand = _random_path(row_pool, col_pool)
            if cand is None:
                continue
            trial = paths + [cand]
            ok, trial_crossings = _layout_ok(trial, min_crossings=0)
            if not ok:
                continue
            new_crosses = len(trial_crossings) - current_crossings
            if need_cross and paths and new_crosses == 0:
                continue
            placed = cand
            current_crossings = len(trial_crossings)
            break
        if placed is None:
            return None
        paths.append(placed)

    ok, _ = _layout_ok(paths, min_crossings=1)
    if not ok:
        return None

    inp, expected = _render_case(paths, colors, legend, h, w)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
