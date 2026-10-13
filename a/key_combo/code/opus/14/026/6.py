from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

HOLE_TO_COLOR = {
    0: BLUE,
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
}

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bg[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bg[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _extract_markers(grid: np.ndarray) -> list[dict]:
    markers: list[dict] = []
    for idx, coords in enumerate(_components_of_color(grid, GREY)):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
        markers.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "holes": _hole_count(local),
            }
        )
    return markers


def _marker_at(markers: list[dict], r: int, c: int) -> dict | None:
    for marker in markers:
        r0, c0, r1, c1 = marker["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return marker
    return None


def _ray_hit(
    shape: tuple[int, int],
    markers: list[dict],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[dict | None, tuple[int, int] | None, list[tuple[int, int]]]:
    dr, dc = direction
    rr, cc = start
    painted: list[tuple[int, int]] = []
    while True:
        nr, nc = rr + dr, cc + dc
        if not _in_bounds(shape, nr, nc):
            return None, None, painted
        marker = _marker_at(markers, nr, nc)
        if marker is not None:
            return marker, (rr, cc), painted
        painted.append((nr, nc))
        rr, cc = nr, nc


def _solve(grid: np.ndarray, *, want_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    starts = np.argwhere(g == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    start_r, start_c = map(int, starts[0])
    assert start_c == 0, "start pixel must be on the left edge"

    markers = _extract_markers(g)
    assert 3 <= len(markers) <= 4, "expected 3-4 grey marker objects"
    hole_counts = [int(marker["holes"]) for marker in markers]
    assert len(set(hole_counts)) == len(hole_counts), "marker hole counts must be distinct"
    assert set(hole_counts) <= {0, 1, 2, 3}

    out = g.copy()
    opened: set[int] = set()
    seen_segments: set[tuple[int, int, str, int]] = set()
    pending: list[tuple[int, int, str, int]] = [(start_r, start_c, "h", RED)]

    trace = {
        "opened_order": [],
        "hit_counts": {int(marker["id"]): 0 for marker in markers},
        "hits": [],
    }

    while pending:
        r, c, axis, color = pending.pop()
        seg_key = (r, c, axis, color)
        if seg_key in seen_segments:
            continue
        seen_segments.add(seg_key)

        if int(out[r, c]) == BLACK:
            out[r, c] = color

        if axis == "h":
            dirs = [(0, 1), (0, -1)]
            next_axis = "v"
        else:
            dirs = [(1, 0), (-1, 0)]
            next_axis = "h"

        children: list[tuple[int, int, str, int]] = []
        for dr, dc in dirs:
            marker, stop, painted = _ray_hit(g.shape, markers, (r, c), (dr, dc))
            for rr, cc in painted:
                if int(out[rr, cc]) == BLACK:
                    out[rr, cc] = color
            if marker is None:
                continue
            trace["hit_counts"][int(marker["id"])] += 1
            trace["hits"].append(
                {
                    "from": (r, c, axis, color),
                    "dir": (dr, dc),
                    "marker_id": int(marker["id"]),
                    "stop": stop,
                }
            )
            if int(marker["id"]) in opened:
                continue
            opened.add(int(marker["id"]))
            trace["opened_order"].append(int(marker["id"]))
            stop_r, stop_c = stop
            children.append((stop_r, stop_c, next_axis, HOLE_TO_COLOR[int(marker["holes"])]))

        for child in reversed(children):
            pending.append(child)

    if want_trace:
        trace["opened_set"] = set(opened)
        trace["markers"] = markers
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend a recursive ray network whose turn colors are encoded by each grey marker's hole count.

    Input structure:
    - A black background contains exactly one red start pixel on the left edge and 3-4 separated grey marker objects.
    - Each marker is one 4-connected grey rectangular blob whose bounding box is 3x3 to 5x5 and whose interior contains 0-3 enclosed black holes.
    - Marker objects stay fixed and act as blockers when a ray reaches their bounding-box boundary.

    Rule:
    - Fire a red horizontal ray rightward from the start pixel, painting background cells until the ray would enter a marker bounding box; the last painted cell just before that box is the stop cell.
    - When a marker is first reached, count its enclosed holes and map that count to the next ray color: 0->blue, 1->green, 2->yellow, 3->orange.
    - From the stop cell, emit the two perpendicular rays in depth-first order, using right/down before left/up; later hits on an already triggered marker only stop the incoming ray and do not emit more rays.
    - Rays stop at the grid edge, previously painted ray cells are not blockers, and all original grey marker pixels remain unchanged.

    Key insight:
    - The final pattern depends on both the recursive ray topology and the per-marker hole counts: topology decides which marker is reached first, while the hole count decides the color of the next emitted segment.
    """
    return _solve(grid, want_trace=False)


def _make_patch_with_holes(hole_count: int, h: int, w: int, rng: random.Random) -> np.ndarray | None:
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return None
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if hole_count == 0:
        patch = np.full((h, w), GREY, dtype=int)
        return patch

    candidates = interior[:]
    rng.shuffle(candidates)
    for _ in range(120):
        rng.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= 2 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == hole_count:
                    break
        if len(chosen) != hole_count:
            continue
        patch = np.full((h, w), GREY, dtype=int)
        for rr, cc in chosen:
            patch[rr, cc] = BLACK
        if _hole_count(patch == GREY) == hole_count:
            return patch
    return None


def _random_patch(hole_count: int, h: int | None, w: int | None, rng: random.Random) -> np.ndarray | None:
    dims: list[tuple[int, int]] = []
    if h is not None and w is not None:
        dims = [(h, w)]
    else:
        for hh in range(3, 6):
            for ww in range(3, 6):
                dims.append((hh, ww))
        rng.shuffle(dims)
    for hh, ww in dims:
        patch = _make_patch_with_holes(hole_count, hh, ww, rng)
        if patch is not None:
            return patch
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _choose_col_covering(anchor_c: int, width: int, lo: int, hi: int, rng: random.Random) -> int | None:
    starts = [c0 for c0 in range(lo, hi - width + 2) if c0 <= anchor_c <= c0 + width - 1]
    if not starts:
        return None
    return rng.choice(starts)


def _count_nonzero_colors(grid: np.ndarray, colors: set[int]) -> dict[int, int]:
    return {color: int(np.count_nonzero(grid == color)) for color in colors}


def _build_four_marker_case(rng: random.Random) -> np.ndarray | None:
    counts = [0, 1, 2, 3]
    rng.shuffle(counts)
    count_a, count_b, count_c, count_d = counts

    w_a = rng.randint(3, 4)
    patch_a = _random_patch(count_a, 3, w_a, rng)
    if patch_a is None:
        return None

    h_b = rng.randint(3, 5)
    w_b = rng.randint(3, 5)
    patch_b = _random_patch(count_b, h_b, w_b, rng)
    if patch_b is None:
        return None

    h_c = rng.randint(3, 5)
    w_c = rng.randint(3, 5)
    patch_c = _random_patch(count_c, h_c, w_c, rng)
    if patch_c is None:
        return None

    w_d = rng.randint(3, 5)
    patch_d = _random_patch(count_d, 5, w_d, rng)
    if patch_d is None:
        return None

    h = rng.randint(max(17, h_c + h_b + 7), 24)
    s_min = h_c + 3
    s_max = h - h_b - 4
    if s_min > s_max:
        return None
    start_r = rng.randint(s_min, s_max)

    y_c = start_r - 2
    y_b = start_r + 2
    r_c0 = start_r - h_c - 2
    r_b0 = start_r + 3
    r_d0 = y_c
    r_a0 = start_r - 1

    x_a = rng.randint(4, 7)
    c_a0 = x_a + 1
    c_b0 = _choose_col_covering(x_a, w_b, 1, x_a + 2, rng)
    c_c0 = _choose_col_covering(x_a, w_c, 1, x_a + 2, rng)
    if c_b0 is None or c_c0 is None:
        return None

    gap_to_d = rng.randint(patch_a.shape[1] + 2, patch_a.shape[1] + 5)
    c_d0 = x_a + gap_to_d
    w = max(c_d0 + w_d + 2, c_a0 + patch_a.shape[1] + 2, 16)
    w = rng.randint(w, min(26, w + 4))
    if c_d0 + w_d > w - 2:
        return None

    placements = [
        (r_a0, c_a0, patch_a),
        (r_b0, c_b0, patch_b),
        (r_c0, c_c0, patch_c),
        (r_d0, c_d0, patch_d),
    ]
    for r0, c0, patch in placements:
        ph, pw = patch.shape
        if r0 < 1 or c0 < 1 or r0 + ph > h - 1 or c0 + pw > w - 1:
            return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[start_r, 0] = RED
    for r0, c0, patch in placements:
        _paste(grid, patch, r0, c0)

    out, trace = _solve(grid, want_trace=True)
    markers = trace["markers"]
    by_bbox = {tuple(marker["bbox"]): marker for marker in markers}
    d_marker = by_bbox.get((r_d0, c_d0, r_d0 + 4, c_d0 + w_d - 1))
    if d_marker is None:
        return None

    d_id = int(d_marker["id"])
    if trace["hit_counts"][d_id] < 2:
        return None
    d_open_events = [hit for hit in trace["hits"] if int(hit["marker_id"]) == d_id]
    if not d_open_events:
        return None
    if tuple(d_open_events[0]["stop"]) != (y_b, c_d0 - 1):
        return None
    if trace["opened_set"] != {int(marker["id"]) for marker in markers}:
        return None

    mapped_colors = {HOLE_TO_COLOR[count_a], HOLE_TO_COLOR[count_b], HOLE_TO_COLOR[count_c], HOLE_TO_COLOR[count_d], RED}
    color_counts = _count_nonzero_colors(out, mapped_colors)
    if len([color for color, cnt in color_counts.items() if cnt > 0]) != 5:
        return None
    if color_counts[HOLE_TO_COLOR[count_d]] < 3:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def _build_three_marker_case(rng: random.Random) -> np.ndarray | None:
    counts = rng.sample([0, 1, 2, 3], 3)
    rng.shuffle(counts)
    count_a, count_b, count_c = counts

    patch_a = _random_patch(count_a, 3, rng.randint(3, 4), rng)
    patch_b = _random_patch(count_b, rng.randint(3, 5), rng.randint(3, 5), rng)
    patch_c = _random_patch(count_c, rng.randint(3, 5), rng.randint(3, 5), rng)
    if patch_a is None or patch_b is None or patch_c is None:
        return None

    h = rng.randint(max(15, patch_b.shape[0] + patch_c.shape[0] + 7), 22)
    s_min = patch_c.shape[0] + 3
    s_max = h - patch_b.shape[0] - 4
    if s_min > s_max:
        return None
    start_r = rng.randint(s_min, s_max)

    x_a = rng.randint(4, 8)
    c_a0 = x_a + 1
    c_b0 = _choose_col_covering(x_a, patch_b.shape[1], 1, x_a + 2, rng)
    c_c0 = _choose_col_covering(x_a, patch_c.shape[1], 1, x_a + 2, rng)
    if c_b0 is None or c_c0 is None:
        return None

    placements = [
        (start_r - 1, c_a0, patch_a),
        (start_r + 3, c_b0, patch_b),
        (start_r - patch_c.shape[0] - 2, c_c0, patch_c),
    ]

    right_extent = max(c0 + patch.shape[1] for _r0, c0, patch in placements)
    w = rng.randint(max(15, right_extent + 2), min(24, right_extent + 7))
    grid = np.full((h, w), BLACK, dtype=int)
    grid[start_r, 0] = RED

    for r0, c0, patch in placements:
        ph, pw = patch.shape
        if r0 < 1 or c0 < 1 or r0 + ph > h - 1 or c0 + pw > w - 1:
            return None
        _paste(grid, patch, r0, c0)

    out, trace = _solve(grid, want_trace=True)
    markers = trace["markers"]
    if trace["opened_set"] != {int(marker["id"]) for marker in markers}:
        return None

    hits_from_start = [
        hit
        for hit in trace["hits"]
        if tuple(hit["from"][:3]) == (start_r, 0, "h")
    ]
    if len(hits_from_start) != 1:
        return None

    colors = {RED, HOLE_TO_COLOR[count_a], HOLE_TO_COLOR[count_b], HOLE_TO_COLOR[count_c]}
    if len([c for c, n in _count_nonzero_colors(out, colors).items() if n > 0]) != 4:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        if rng.random() < 0.7:
            grid = _build_four_marker_case(rng)
        else:
            grid = _build_three_marker_case(rng)
        if grid is None:
            continue
        out = main(grid)
        if out.shape != grid.shape or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid recursive ray-hole puzzle input")
