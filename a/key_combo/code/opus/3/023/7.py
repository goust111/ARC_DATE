from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, in_bounds


GEN_HEIGHT_RANGE = (12, 18)
GEN_WIDTH_RANGE = (12, 18)
GEN_SEGMENT_LEN_RANGE = (7, 11)
GEN_BLOCKER_COUNTS = (2, 3)
BLOCKER_COLORS = (Color.RED, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_RETRY_LIMIT = 800


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _diagonal_segment(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    assert dr != 0 and abs(dr) == abs(dc)
    sr, sc = _sign(dr), _sign(dc)
    return [(ar + i * sr, ac + i * sc) for i in range(abs(dr) + 1)], (sr, sc)


def _full_diagonal(shape: tuple[int, int], anchor: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = anchor
    while in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc

    coords = []
    while in_bounds(shape, r, c):
        coords.append((int(r), int(c)))
        r += dr
        c += dc
    return coords


def _bounce_ray(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    coords = [(int(r), int(c))]

    while True:
        nr = r + dr
        if nr < 0 or nr >= h:
            break
        nc = c + dc
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
        coords.append((int(r), int(c)))
    return coords


def _blocker_trace(shape: tuple[int, int], blocker: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    up = _bounce_ray(shape, blocker, (-1, step[1]))
    down = _bounce_ray(shape, blocker, (1, -step[1]))
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for coord in itertools.chain(up, down[1:]):
        if coord not in seen:
            seen.add(coord)
            out.append(coord)
    return out


def _trace_sets(shape: tuple[int, int], blockers: list[dict], step: tuple[int, int]) -> list[set[tuple[int, int]]]:
    return [set(_blocker_trace(shape, blocker["coord"], step)) for blocker in blockers]


def _crossings_from_trace_sets(trace_sets: list[set[tuple[int, int]]]) -> list[tuple[int, int]]:
    hits: dict[tuple[int, int], int] = {}
    for coords in trace_sets:
        for coord in coords:
            hits[coord] = hits.get(coord, 0) + 1
    return sorted(coord for coord, count in hits.items() if count >= 2)


def _candidate_blue_segments(g: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int], list[dict]]]:
    blue = [(int(r), int(c)) for r, c in np.argwhere(g == int(Color.BLUE))]
    candidates = []
    for a, b in itertools.combinations(blue, 2):
        top, bottom = (a, b) if a[0] < b[0] else (b, a)
        if top[0] == bottom[0] or abs(bottom[0] - top[0]) != abs(bottom[1] - top[1]):
            continue
        segment, step = _diagonal_segment(top, bottom)
        blockers = []
        for idx, coord in enumerate(segment[1:-1], start=1):
            color = int(g[coord])
            if color not in (int(Color.BLACK), int(Color.BLUE)):
                blockers.append({"index": idx, "coord": coord, "color": color})
        if blockers:
            candidates.append((segment, step, blockers))
    return candidates


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _candidate_blue_segments(g)
    assert candidates, "expected a diagonal blue segment candidate"
    scored = sorted(candidates, key=lambda item: (len(item[2]), len(item[0])), reverse=True)
    best = scored[0]
    tied = [item for item in scored if (len(item[2]), len(item[0])) == (len(best[2]), len(best[0]))]
    assert len(tied) == 1, "expected a unique blue segment candidate"

    segment, step, blockers = best
    assert len(blockers) in GEN_BLOCKER_COUNTS
    blocker_colors = [int(blocker["color"]) for blocker in blockers]
    assert len(set(blocker_colors)) == len(blocker_colors)
    trace_sets = _trace_sets(g.shape, blockers, step)
    crossings = _crossings_from_trace_sets(trace_sets)

    return {
        "grid": g,
        "segment": segment,
        "segment_set": set(segment),
        "step": step,
        "blockers": blockers,
        "trace_sets": trace_sets,
        "crossings": crossings,
    }


def _render_base(g: np.ndarray, info: dict) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for coord in info["segment"]:
        if int(out[coord]) in (int(Color.BLACK), int(Color.BLUE)):
            out[coord] = int(Color.BLUE)

    for blocker in info["blockers"]:
        color = int(blocker["color"])
        for coord in _blocker_trace(out.shape, blocker["coord"], info["step"]):
            if int(out[coord]) == int(Color.BLACK):
                out[coord] = color
    return out


def _add_rings(base: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, copy=True)
    h, w = out.shape
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(Color.YELLOW)
    return out


def _render_straight_variant(g: np.ndarray, info: dict) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for coord in info["segment"]:
        if int(out[coord]) in (int(Color.BLACK), int(Color.BLUE)):
            out[coord] = int(Color.BLUE)

    perpendicular = (info["step"][0], -info["step"][1])
    trace_sets = []
    for blocker in info["blockers"]:
        coords = _full_diagonal(out.shape, blocker["coord"], perpendicular)
        trace_sets.append(set(coords))
        for coord in coords:
            if int(out[coord]) == int(Color.BLACK):
                out[coord] = int(blocker["color"])
    crossings = _crossings_from_trace_sets(trace_sets)
    return _add_rings(out, crossings)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the blue diagonal segment, let its blockers emit bouncing perpendicular diagonals, and ring every crossing.

    Input structure:
    - The grid is mostly black.
    - Two blue pixels are the endpoints of one diagonal segment.
    - Between them on that same segment sit 2 or 3 differently colored blocker pixels.

    Rule:
    - Repaint the whole endpoint-to-endpoint diagonal segment blue, but keep each blocker's original color.
    - From every blocker, trace the perpendicular diagonal family through the whole grid:
      one ray goes upward and one downward, and each ray reflects off the left/right walls while keeping its vertical direction.
    - Paint those traced cells in the blocker color without overwriting existing non-background cells.
    - Whenever two different blocker traces visit the same cell, treat that cell as a crossing and paint a yellow 3x3 ring around it, preserving the center.

    Key insight:
    - The blocker positions do not directly reveal the ring centers; each blocker first expands into a reflected diagonal trace, and only the trace intersections determine where the yellow rings appear.
    """
    info = _analyze_input(grid)
    base = _render_base(info["grid"], info)
    return _add_rings(base, info["crossings"])


def _sample_segment() -> tuple[int, int, list[tuple[int, int]]] | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    max_len = min(GEN_SEGMENT_LEN_RANGE[1], h - 2, w - 2)
    if max_len < GEN_SEGMENT_LEN_RANGE[0]:
        return None
    seg_len = random.randint(GEN_SEGMENT_LEN_RANGE[0], max_len)
    dc = random.choice((-1, 1))
    r0 = random.randint(1, h - seg_len - 1)
    if dc == 1:
        c0 = random.randint(1, w - seg_len - 1)
    else:
        c0 = random.randint(seg_len - 1, w - 2)
    segment = [(r0 + i, c0 + i * dc) for i in range(seg_len)]
    return h, w, segment


def _sample_blocker_indices(seg_len: int, count: int) -> list[int] | None:
    interior = list(range(1, seg_len - 1))
    for _ in range(40):
        idxs = sorted(random.sample(interior, count))
        if idxs[-1] - idxs[0] < count + 1:
            continue
        if any(b - a < 2 for a, b in zip(idxs, idxs[1:])):
            continue
        return idxs
    return None


def _ring_mask(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True
    return mask


def _crossings_are_well_spaced(crossings: list[tuple[int, int]]) -> bool:
    for a, b in itertools.combinations(crossings, 2):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 3:
            return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return False

    blockers = info["blockers"]
    crossings = info["crossings"]
    segment = info["segment"]
    if np.count_nonzero(grid == int(Color.BLUE)) != 2:
        return False
    if np.count_nonzero(grid != int(Color.BLACK)) != 2 + len(blockers):
        return False
    if len(crossings) == 0 or len(crossings) > 3:
        return False
    if any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in crossings):
        return False
    if any(coord in {blocker["coord"] for blocker in blockers} for coord in crossings):
        return False
    if not _crossings_are_well_spaced(crossings):
        return False
    if len(segment) - len(blockers) < 3:
        return False

    straight_trace_sets = []
    perpendicular = (info["step"][0], -info["step"][1])
    for blocker, bounced in zip(blockers, info["trace_sets"]):
        straight = set(_full_diagonal(grid.shape, blocker["coord"], perpendicular))
        straight_trace_sets.append(straight)
        if not (bounced - straight):
            return False
    if _crossings_from_trace_sets(straight_trace_sets):
        return False

    base = _render_base(grid, info)
    out = _add_rings(base, crossings)
    if np.array_equal(base, out):
        return False
    if np.array_equal(_render_straight_variant(grid, info), out):
        return False

    for blocker in blockers:
        if int(base[blocker["coord"]]) != int(blocker["color"]):
            return False
    for coord in segment:
        if coord not in {blocker["coord"] for blocker in blockers} and int(base[coord]) != int(Color.BLUE):
            return False

    yellow_mask = _ring_mask(grid.shape, crossings)
    if not np.array_equal(out == int(Color.YELLOW), yellow_mask):
        return False
    for r, c in crossings:
        if int(out[r, c]) == int(Color.YELLOW):
            return False
        if int(out[r, c]) != int(base[r, c]):
            return False

    return np.array_equal(main(grid), out)


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_segment()
    if sampled is None:
        return None
    h, w, segment = sampled
    blocker_count = 3 if random.random() < 0.9 else 2
    blocker_indices = _sample_blocker_indices(len(segment), blocker_count)
    if blocker_indices is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[segment[0]] = int(Color.BLUE)
    grid[segment[-1]] = int(Color.BLUE)
    colors = random.sample(list(BLOCKER_COLORS), blocker_count)
    for idx, color in zip(blocker_indices, colors):
        grid[segment[idx]] = int(color)

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-diagonal crossing input")
