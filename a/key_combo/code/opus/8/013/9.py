from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def _sign(x: int) -> int:
    return 0 if int(x) == 0 else (1 if int(x) > 0 else -1)


def _repair_and_split_markers(g: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    repaired = np.array(g, dtype=int, copy=True)
    embedded: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    for r0, c0 in np.argwhere(g == ORANGE):
        r, c = int(r0), int(c0)
        blue_neighbors = sum(
            1
            for rr, cc in _neighbors4(r, c)
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == BLUE
        )
        if blue_neighbors >= 3:
            embedded.append((r, c))
            repaired[r, c] = BLUE
        else:
            isolated.append((r, c))
    return repaired, embedded, isolated


def _outline_mask(g: np.ndarray) -> np.ndarray:
    mask = g == BLUE
    out = np.zeros_like(mask, dtype=bool)
    for r0, c0 in np.argwhere(mask):
        r, c = int(r0), int(c0)
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(g.shape, rr, cc) or not bool(mask[rr, cc]):
                out[r, c] = True
                break
    return out


def _first_visible_boundary(
    g: np.ndarray, boundary: np.ndarray, marker: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = marker
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        dist = 1
        while _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == BG:
            rr += dr
            cc += dc
            dist += 1
        if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == BLUE and bool(boundary[rr, cc]):
            candidates.append((dist, (dr, dc), (int(rr), int(cc))))
    assert candidates, "isolated marker must see the repaired outline"
    candidates.sort(key=lambda item: item[0])
    best_dist = candidates[0][0]
    best = [item for item in candidates if item[0] == best_dist]
    assert len(best) == 1, "visible outline hit must be unique"
    _dist, direction, hit = best[0]
    return direction, hit


def _straight_segment(
    boundary: np.ndarray, hit: tuple[int, int], ray_dir: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = hit
    if ray_dir[0] == 0:
        r0 = r
        while r0 - 1 >= 0 and bool(boundary[r0 - 1, c]):
            r0 -= 1
        r1 = r
        while r1 + 1 < boundary.shape[0] and bool(boundary[r1 + 1, c]):
            r1 += 1
        return (int(r0), int(c)), (int(r1), int(c))

    c0 = c
    while c0 - 1 >= 0 and bool(boundary[r, c0 - 1]):
        c0 -= 1
    c1 = c
    while c1 + 1 < boundary.shape[1] and bool(boundary[r, c1 + 1]):
        c1 += 1
    return (int(r), int(c0)), (int(r), int(c1))


def _infer_triangle(
    g: np.ndarray, boundary: np.ndarray, marker: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    ray_dir, hit = _first_visible_boundary(g, boundary, marker)
    end_a, end_b = _straight_segment(boundary, hit, ray_dir)
    if ray_dir[0] == 0:
        shared = [pt for pt in (end_a, end_b) if pt[0] == marker[0]]
    else:
        shared = [pt for pt in (end_a, end_b) if pt[1] == marker[1]]
    assert len(shared) == 1, "marker must align with exactly one segment endpoint"
    near = shared[0]
    far = end_b if near == end_a else end_a
    seg_len = max(abs(end_a[0] - end_b[0]), abs(end_a[1] - end_b[1])) + 1
    assert seg_len >= 3, "outline side must be at least length 3"
    assert near[0] == marker[0] or near[1] == marker[1], "yellow edge must be axis-aligned"
    assert abs(far[0] - marker[0]) == abs(far[1] - marker[1]), "red edge must be a 45-degree diagonal"
    return near, far


def _draw_axis(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    dr = _sign(end[0] - start[0])
    dc = _sign(end[1] - start[1])
    assert (dr == 0) ^ (dc == 0), "axis edge must be horizontal or vertical"
    steps = abs(end[0] - start[0]) + abs(end[1] - start[1])
    for k in range(steps):
        rr = int(start[0] + dr * k)
        cc = int(start[1] + dc * k)
        out[rr, cc] = int(color)


def _draw_diag(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    dr = _sign(end[0] - start[0])
    dc = _sign(end[1] - start[1])
    steps = abs(end[0] - start[0])
    assert steps == abs(end[1] - start[1]) and steps > 0, "diagonal edge must be 45 degrees"
    for k in range(steps):
        rr = int(start[0] + dr * k)
        cc = int(start[1] + dc * k)
        out[rr, cc] = int(color)


def _solve(grid: np.ndarray, *, repair: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    work = g.copy()
    if repair:
        work, _embedded, isolated = _repair_and_split_markers(g)
    else:
        isolated = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    boundary = _outline_mask(work)
    out = work.copy()
    for marker in sorted(isolated, key=lambda rc: (rc[0], rc[1]), reverse=True):
        near, far = _infer_triangle(work, boundary, marker)
        _draw_diag(out, marker, far, RED)
        _draw_axis(out, marker, near, YELLOW)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded orange defects, then use isolated orange markers to complete triangles against the repaired blue outline.

    Input structure:
    - A black background contains one large 4-connected blue object and one full grey edge line on the outer border.
    - Orange pixels appear in two roles: some are embedded inside the blue object with at least three blue 4-neighbors, and some are isolated in the background.
    - Each isolated orange marker has a clear horizontal or vertical sightline to a straight repaired outline side of the blue object.

    Rule:
    - Recolor every embedded orange marker to blue, repairing the main object before any geometry is read.
    - Extract the repaired blue outline, then for each isolated orange marker find its nearest visible outline hit and the full straight outline side through that hit.
    - Treat the marker as the opposite triangle vertex: draw a red diagonal from the marker toward the far endpoint of that side, and draw a yellow axis-aligned edge from the marker toward the near endpoint, leaving the blue outline side itself in place.
    - When several isolated markers are present, process larger rows first and larger columns first; later triangles overwrite earlier ones where they overlap.

    Key insight:
    The two orange marker roles look identical locally, but only the embedded ones must be repaired first; that repair changes which straight blue outline side the isolated markers attach their triangles to.
    """
    return _solve(grid, repair=True)


SIDE_DEFS = {
    "left": {
        "length": lambda h, w: h,
        "endpoints": lambda r0, c0, r1, c1: {"top": (r0, c0), "bottom": (r1, c0)},
        "marker": lambda pt, d: (pt[0], pt[1] - d),
    },
    "right": {
        "length": lambda h, w: h,
        "endpoints": lambda r0, c0, r1, c1: {"top": (r0, c1), "bottom": (r1, c1)},
        "marker": lambda pt, d: (pt[0], pt[1] + d),
    },
    "top": {
        "length": lambda h, w: w,
        "endpoints": lambda r0, c0, r1, c1: {"left": (r0, c0), "right": (r0, c1)},
        "marker": lambda pt, d: (pt[0] - d, pt[1]),
    },
    "bottom": {
        "length": lambda h, w: w,
        "endpoints": lambda r0, c0, r1, c1: {"left": (r1, c0), "right": (r1, c1)},
        "marker": lambda pt, d: (pt[0] + d, pt[1]),
    },
}


def _valid_dims() -> list[tuple[int, int]]:
    dims = []
    for h in range(4, 7):
        for w in range(4, 7):
            if h * w <= 26:
                dims.append((h, w))
    return dims


def _sample_marker_specs() -> list[dict]:
    isolated_count = random.choice([1, 2])
    if isolated_count == 2 and random.random() < 0.35:
        side = random.choice(["left", "right", "top", "bottom"])
        if side in {"left", "right"}:
            return [
                {"side": side, "near": "top"},
                {"side": side, "near": "bottom"},
            ]
        return [
            {"side": side, "near": "left"},
            {"side": side, "near": "right"},
        ]

    if isolated_count == 1:
        side = random.choice(["left", "right", "top", "bottom"])
        near = random.choice(["top", "bottom"]) if side in {"left", "right"} else random.choice(["left", "right"])
        return [{"side": side, "near": near}]

    side_a, side_b = random.sample(["left", "right", "top", "bottom"], 2)
    near_a = random.choice(["top", "bottom"]) if side_a in {"left", "right"} else random.choice(["left", "right"])
    near_b = random.choice(["top", "bottom"]) if side_b in {"left", "right"} else random.choice(["left", "right"])
    return [{"side": side_a, "near": near_a}, {"side": side_b, "near": near_b}]


def _sample_dims(marker_specs: list[dict]) -> tuple[int, int]:
    side_counts: dict[str, int] = {}
    for spec in marker_specs:
        side_counts[spec["side"]] = side_counts.get(spec["side"], 0) + 1

    dims = _valid_dims()
    if any(count == 2 for count in side_counts.values()):
        repeated = next(side for side, count in side_counts.items() if count == 2)
        if repeated in {"left", "right"}:
            dims = [(h, w) for h, w in dims if h == 5]
        else:
            dims = [(h, w) for h, w in dims if w == 5]
    return random.choice(dims)


def _placement_ranges(H: int, W: int, h: int, w: int, marker_specs: list[dict]) -> tuple[range, range] | None:
    low_r = 2
    high_r = H - h - 2
    low_c = 2
    high_c = W - w - 2

    used_sides = {spec["side"] for spec in marker_specs}
    if "top" in used_sides:
        low_r = max(low_r, w)
    if "bottom" in used_sides:
        high_r = min(high_r, H - h - w)
    if "left" in used_sides:
        low_c = max(low_c, h)
    if "right" in used_sides:
        high_c = min(high_c, W - w - h)

    if low_r > high_r or low_c > high_c:
        return None
    return range(low_r, high_r + 1), range(low_c, high_c + 1)


def _corridor_cells(marker: tuple[int, int], near: tuple[int, int]) -> set[tuple[int, int]]:
    dr = _sign(near[0] - marker[0])
    dc = _sign(near[1] - marker[1])
    steps = abs(near[0] - marker[0]) + abs(near[1] - marker[1])
    return {(int(marker[0] + dr * k), int(marker[1] + dc * k)) for k in range(steps + 1)}


def _add_irregular_bump(
    object_cells: set[tuple[int, int]],
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    used_sides: set[str],
    forbidden: set[tuple[int, int]],
) -> bool:
    free_sides = [side for side in ["left", "right", "top", "bottom"] if side not in used_sides]
    random.shuffle(free_sides)
    for side in free_sides:
        if side in {"left", "right"}:
            rows = list(range(r0 + 1, r1))
            random.shuffle(rows)
            dc = -1 if side == "left" else 1
            col = c0 if side == "left" else c1
            for row in rows:
                first = (row, col + dc)
                second = (row, col + 2 * dc)
                if not _in_bounds((H, W), first[0], first[1]) or first in forbidden or first in object_cells:
                    continue
                object_cells.add(first)
                if random.random() < 0.6 and _in_bounds((H, W), second[0], second[1]) and second not in forbidden and second not in object_cells:
                    object_cells.add(second)
                return True
        else:
            cols = list(range(c0 + 1, c1))
            random.shuffle(cols)
            dr = -1 if side == "top" else 1
            row = r0 if side == "top" else r1
            for col in cols:
                first = (row + dr, col)
                second = (row + 2 * dr, col)
                if not _in_bounds((H, W), first[0], first[1]) or first in forbidden or first in object_cells:
                    continue
                object_cells.add(first)
                if random.random() < 0.6 and _in_bounds((H, W), second[0], second[1]) and second not in forbidden and second not in object_cells:
                    object_cells.add(second)
                return True
    return False


def _embedded_candidates(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r, c in sorted(cells):
        neighbors = sum((rr, cc) in cells for rr, cc in _neighbors4(r, c))
        if neighbors >= 3:
            out.append((r, c))
    return out


def _build_scene() -> tuple[np.ndarray, dict]:
    H = random.randint(12, 18)
    W = random.randint(12, 18)
    marker_specs = _sample_marker_specs()
    h, w = _sample_dims(marker_specs)
    ranges = _placement_ranges(H, W, h, w, marker_specs)
    if ranges is None:
        raise RuntimeError("no valid placement range")
    r0 = random.choice(list(ranges[0]))
    c0 = random.choice(list(ranges[1]))
    r1, c1 = r0 + h - 1, c0 + w - 1

    object_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    used_sides = {spec["side"] for spec in marker_specs}

    side_notches: dict[str, tuple[int, int]] = {}
    marker_meta: list[dict] = []
    forbidden = set()
    for spec in marker_specs:
        side = spec["side"]
        side_len = int(SIDE_DEFS[side]["length"](h, w))
        near = SIDE_DEFS[side]["endpoints"](r0, c0, r1, c1)[spec["near"]]
        far_choices = dict(SIDE_DEFS[side]["endpoints"](r0, c0, r1, c1))
        far_choices.pop(spec["near"])
        far = next(iter(far_choices.values()))
        marker = SIDE_DEFS[side]["marker"](near, side_len - 1)
        assert _in_bounds((H, W), marker[0], marker[1])
        marker_meta.append({"marker": marker, "near": near, "far": far, "side": side})
        forbidden |= _corridor_cells(marker, near)

    if not _add_irregular_bump(object_cells, H, W, r0, c0, r1, c1, used_sides, forbidden):
        raise RuntimeError("failed to add irregular bump")

    for side in sorted(used_sides):
        if side in {"left", "right"}:
            candidates = [(r, c0 if side == "left" else c1) for r in range(r0 + 1, r1)]
        else:
            candidates = [(r0 if side == "top" else r1, c) for c in range(c0 + 1, c1)]
        blocked = {meta["near"] for meta in marker_meta if meta["side"] == side}
        candidates = [cell for cell in candidates if cell not in blocked]
        candidates = [cell for cell in candidates if sum((rr, cc) in object_cells for rr, cc in _neighbors4(*cell)) >= 3]
        random.shuffle(candidates)
        chosen = None
        for cell in candidates:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in side_notches.values()):
                chosen = cell
                break
        if chosen is None:
            raise RuntimeError("failed to place a required embedded notch")
        side_notches[side] = chosen

    embedded = list(side_notches.values())
    if len(used_sides) == 1 and random.random() < 0.5:
        extra_candidates = [
            cell
            for cell in _embedded_candidates(object_cells)
            if cell not in embedded and all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in embedded)
        ]
        if extra_candidates:
            embedded.append(random.choice(extra_candidates))

    isolated = [meta["marker"] for meta in marker_meta]
    grid = np.full((H, W), BG, dtype=int)
    for r, c in object_cells:
        grid[r, c] = BLUE
    for r, c in embedded:
        grid[r, c] = ORANGE
    for r, c in isolated:
        grid[r, c] = ORANGE

    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        grid[0, :] = GREY
    elif edge == "bottom":
        grid[-1, :] = GREY
    elif edge == "left":
        grid[:, 0] = GREY
    else:
        grid[:, -1] = GREY

    meta = {
        "object_cells": set(object_cells),
        "embedded": list(embedded),
        "markers": list(marker_meta),
        "edge": edge,
    }
    return grid, meta


def _render_expected(meta: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in meta["object_cells"]:
        out[r, c] = BLUE
    edge = meta["edge"]
    if edge == "top":
        out[0, :] = GREY
    elif edge == "bottom":
        out[-1, :] = GREY
    elif edge == "left":
        out[:, 0] = GREY
    else:
        out[:, -1] = GREY

    for item in sorted(meta["markers"], key=lambda m: (m["marker"][0], m["marker"][1]), reverse=True):
        _draw_diag(out, item["marker"], item["far"], RED)
        _draw_axis(out, item["marker"], item["near"], YELLOW)
    return out


def _validate_scene(grid: np.ndarray, meta: dict) -> bool:
    repaired, embedded, isolated = _repair_and_split_markers(grid)
    if sorted(embedded) != sorted(meta["embedded"]):
        return False
    if sorted(isolated) != sorted(item["marker"] for item in meta["markers"]):
        return False
    if not (1 <= len(embedded) <= 2 and 1 <= len(isolated) <= 2):
        return False

    boundary = _outline_mask(repaired)
    for item in meta["markers"]:
        near, far = _infer_triangle(repaired, boundary, item["marker"])
        if near != item["near"] or far != item["far"]:
            return False
        if not all(int(grid[r, c]) == BG for r, c in _corridor_cells(item["marker"], item["near"]) - {item["marker"], item["near"]}):
            return False

    if not _add_irregular_signature_check(meta):
        return False

    expected = _render_expected(meta, grid.shape)
    if not np.array_equal(main(grid), expected):
        return False
    try:
        shortcut = _solve(grid, repair=False)
    except AssertionError:
        shortcut = None
    if shortcut is not None and np.array_equal(shortcut, expected):
        return False
    return True


def _add_irregular_signature_check(meta: dict) -> bool:
    rows = {}
    cols = {}
    for r, c in meta["object_cells"]:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)
    full_rect_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return len(meta["object_cells"]) < full_rect_area


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, meta = _build_scene()
        except RuntimeError:
            continue
        if _validate_scene(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-9 input")
