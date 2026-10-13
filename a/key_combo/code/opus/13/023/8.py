from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
BLUE = 1
RED = 2
WALL = 5

EDGE_CHOICES = ("top", "bottom", "left", "right")
WALL_COUNT_WEIGHTS = ((2, 2), (3, 4), (4, 3))
FIRST_WALL_DIST_RANGE = (2, 4)
WALL_SPACING_RANGE = (2, 4)
FAR_MARGIN_RANGE = (2, 4)
SIDE_MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _edge_direction(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    directions: list[tuple[int, int]] = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))
    assert len(directions) == 1, "marker must lie on exactly one edge"
    return directions[0]


def _ray_color(depth: int) -> int:
    return BLUE if depth % 2 == 0 else RED


def _wall_map(g: np.ndarray, dr: int, dc: int) -> dict[int, tuple[int, int, tuple[int, ...]]]:
    walls: dict[int, tuple[int, int, tuple[int, ...]]] = {}
    if dr != 0:
        for r in np.flatnonzero(np.any(g == WALL, axis=1)):
            cols = np.flatnonzero(g[int(r)] == WALL)
            lo = int(cols[0])
            hi = int(cols[-1])
            gaps = tuple(int(c) for c in range(lo, hi + 1) if g[int(r), int(c)] != WALL)
            assert gaps, "each wall line must contain at least one gap"
            walls[int(r)] = (lo, hi, gaps)
    else:
        for c in np.flatnonzero(np.any(g == WALL, axis=0)):
            rows = np.flatnonzero(g[:, int(c)] == WALL)
            lo = int(rows[0])
            hi = int(rows[-1])
            gaps = tuple(int(r) for r in range(lo, hi + 1) if g[int(r), int(c)] != WALL)
            assert gaps, "each wall line must contain at least one gap"
            walls[int(c)] = (lo, hi, gaps)
    return walls


def main(grid: np.ndarray) -> np.ndarray:
    """Trace an inward branching ray tree through broken walls, alternating blue/red by wall depth.

    Input structure:
    - A black background contains exactly one blue marker pixel on one grid edge.
    - Inside the grid are 2-4 grey wall lines, all perpendicular to the marker's inward direction.
    - Each wall line occupies one row or column, spans only part of the grid, and contains one or more background gaps within that span.

    Rule:
    - Launch a ray from the blue edge marker straight inward; depth-0 segments are blue.
    - Paint open cells along the travel direction until the next cell is a grey wall.
    - When a wall is hit, look at that entire broken wall line, paint same-depth connector segments on the near side toward every gap, paint the gap cells, then continue from the far side of each gap.
    - After crossing one wall, the branch depth increases by 1, so the continuation color alternates blue, red, blue, red, ...
    - Stop each branch at the grid boundary; preserve every grey wall cell unchanged.

    Key insight:
    - The geometry comes from fanning through the gaps of the specific wall line that was hit, while the color of each segment reveals how many walls that branch has already crossed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_cells = np.argwhere(g == BLUE)
    assert len(marker_cells) == 1, "expected exactly one blue edge marker"
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)
    walls = _wall_map(g, dr, dc)

    out = g.copy()
    stack: list[tuple[int, int, int]] = [(start[0], start[1], 0)]
    seen: set[tuple[int, int, int]] = set()

    while stack:
        r, c, depth = stack.pop()
        if (r, c, depth) in seen:
            continue
        seen.add((r, c, depth))
        color = _ray_color(depth)

        if g[r, c] != WALL:
            out[r, c] = color

        cr, cc = r, c
        while True:
            nr = cr + dr
            nc = cc + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            if g[nr, nc] != WALL:
                out[nr, nc] = color
                cr, cc = nr, nc
                continue

            if dr != 0:
                wall_r = nr
                _lo, _hi, gaps = walls[wall_r]
                for gap_c in gaps:
                    c0 = min(cc, gap_c)
                    c1 = max(cc, gap_c)
                    for x in range(c0, c1 + 1):
                        if g[cr, x] != WALL:
                            out[cr, x] = color
                    out[wall_r, gap_c] = color
                    far_r = wall_r + dr
                    if _in_bounds(g.shape, far_r, gap_c) and g[far_r, gap_c] != WALL:
                        stack.append((far_r, gap_c, depth + 1))
            else:
                wall_c = nc
                _lo, _hi, gaps = walls[wall_c]
                for gap_r in gaps:
                    r0 = min(cr, gap_r)
                    r1 = max(cr, gap_r)
                    for y in range(r0, r1 + 1):
                        if g[y, cc] != WALL:
                            out[y, cc] = color
                    out[gap_r, wall_c] = color
                    far_c = wall_c + dc
                    if _in_bounds(g.shape, gap_r, far_c) and g[gap_r, far_c] != WALL:
                        stack.append((gap_r, far_c, depth + 1))
            break

    return out


@lru_cache(maxsize=None)
def _segment_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 0:
        return ((),) if total == 0 else ()

    out: list[tuple[int, ...]] = []
    min_rest = 3 * (parts - 1)
    max_rest = 10 * (parts - 1)
    for first in range(3, 11):
        remaining = total - first
        if remaining < min_rest or remaining > max_rest:
            continue
        for rest in _segment_compositions(remaining, parts - 1):
            out.append((first,) + rest)
    return tuple(out)


@lru_cache(maxsize=None)
def _gap_patterns(width: int, gap_count: int) -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = []
    wall_total = width - gap_count
    if wall_total < 3 * (gap_count + 1):
        return ()
    for segments in _segment_compositions(wall_total, gap_count + 1):
        cursor = segments[0]
        gaps: list[int] = []
        for seg_len in segments[1:]:
            gaps.append(cursor)
            cursor += 1 + seg_len
        patterns.append(tuple(gaps))
    return tuple(sorted(set(patterns)))


def _choose_wall_count() -> int:
    values = [value for value, _weight in WALL_COUNT_WEIGHTS]
    weights = [weight for _value, weight in WALL_COUNT_WEIGHTS]
    return random.choices(values, weights=weights, k=1)[0]


def _choose_gap_counts(wall_count: int) -> tuple[int, ...]:
    first = random.choices([1, 2, 3], weights=[1, 4, 3], k=1)[0]
    counts = [first]
    for _ in range(wall_count - 1):
        counts.append(random.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0])
    if max(counts) == 1:
        counts[random.randrange(wall_count)] = random.choice([2, 3])
    return tuple(counts)


def _min_width_for_gap_count(gap_count: int) -> int:
    return 4 * gap_count + 3


def _choose_width(gap_counts: tuple[int, ...]) -> int:
    lower = max(_min_width_for_gap_count(g) for g in gap_counts)
    upper = min(17, lower + 3)
    return random.randint(lower, upper)


def _choose_wall_distances(wall_count: int) -> tuple[int, ...]:
    distances = [random.randint(*FIRST_WALL_DIST_RANGE)]
    for _ in range(wall_count - 1):
        distances.append(distances[-1] + random.randint(*WALL_SPACING_RANGE))
    return tuple(distances)


def _marker_local_options(width: int, gaps: tuple[int, ...]) -> list[int]:
    gap_set = set(gaps)
    candidates = [x for x in range(width) if x not in gap_set]
    center = (width - 1) / 2.0
    in_band = [x for x in candidates if abs(x - center) <= width * 0.3]
    return in_band or candidates


def _choose_gap_pattern(width: int, gap_count: int, forbidden: set[int]) -> tuple[int, ...] | None:
    options = [pattern for pattern in _gap_patterns(width, gap_count) if not forbidden.intersection(pattern)]
    if not options:
        return None
    return random.choice(options)


def _build_grid(
    edge: str,
    axis_len: int,
    lateral_len: int,
    corridor_start: int,
    corridor_width: int,
    wall_distances: tuple[int, ...],
    gap_patterns: tuple[tuple[int, ...], ...],
    marker_local: int,
) -> np.ndarray:
    if edge in ("top", "bottom"):
        h, w = axis_len, lateral_len
    else:
        h, w = lateral_len, axis_len

    g = np.full((h, w), BG, dtype=int)
    marker_axis = 0 if edge in ("top", "left") else axis_len - 1
    marker_lateral = corridor_start + marker_local

    if edge == "top":
        g[marker_axis, marker_lateral] = BLUE
        wall_axes = wall_distances
    elif edge == "bottom":
        g[h - 1, marker_lateral] = BLUE
        wall_axes = tuple(h - 1 - d for d in wall_distances)
    elif edge == "left":
        g[marker_lateral, marker_axis] = BLUE
        wall_axes = wall_distances
    else:
        g[marker_lateral, w - 1] = BLUE
        wall_axes = tuple(w - 1 - d for d in wall_distances)

    for axis_pos, gaps in zip(wall_axes, gap_patterns):
        gap_set = set(gaps)
        for local in range(corridor_width):
            if local in gap_set:
                continue
            lateral = corridor_start + local
            if edge in ("top", "bottom"):
                g[int(axis_pos), lateral] = WALL
            else:
                g[lateral, int(axis_pos)] = WALL

    return g


def _trace_signature(g: np.ndarray) -> tuple[int, int, int]:
    marker = tuple(map(int, np.argwhere(g == BLUE)[0]))
    dr, dc = _edge_direction(marker, g.shape)
    walls = _wall_map(g, dr, dc)

    max_depth = 0
    max_branching = 1
    hit_count = 0
    stack = [(marker[0], marker[1], 0)]
    seen: set[tuple[int, int, int]] = set()

    while stack:
        r, c, depth = stack.pop()
        if (r, c, depth) in seen:
            continue
        seen.add((r, c, depth))
        cr, cc = r, c
        while True:
            nr = cr + dr
            nc = cc + dc
            if not _in_bounds(g.shape, nr, nc):
                max_depth = max(max_depth, depth)
                break
            if g[nr, nc] != WALL:
                cr, cc = nr, nc
                continue

            hit_count += 1
            if dr != 0:
                gaps = walls[nr][2]
                next_states = [(nr + dr, gap_c, depth + 1) for gap_c in gaps if _in_bounds(g.shape, nr + dr, gap_c)]
            else:
                gaps = walls[nc][2]
                next_states = [(gap_r, nc + dc, depth + 1) for gap_r in gaps if _in_bounds(g.shape, gap_r, nc + dc)]
            max_branching = max(max_branching, len(next_states))
            max_depth = max(max_depth, depth + 1)
            stack.extend(next_states)
            break

    return hit_count, max_depth, max_branching


def _attempt_generate() -> np.ndarray | None:
    edge = random.choice(EDGE_CHOICES)
    wall_count = _choose_wall_count()
    gap_counts = _choose_gap_counts(wall_count)
    corridor_width = _choose_width(gap_counts)
    side_margin_a = random.randint(*SIDE_MARGIN_RANGE)
    side_margin_b = random.randint(*SIDE_MARGIN_RANGE)
    lateral_len = corridor_width + side_margin_a + side_margin_b

    wall_distances = _choose_wall_distances(wall_count)
    far_margin = random.randint(*FAR_MARGIN_RANGE)
    axis_len = wall_distances[-1] + far_margin + 1
    if axis_len > 24 or lateral_len > 24:
        return None

    gap_patterns: list[tuple[int, ...]] = []
    first_pattern = _choose_gap_pattern(corridor_width, gap_counts[0], set())
    if first_pattern is None:
        return None
    gap_patterns.append(first_pattern)

    marker_options = _marker_local_options(corridor_width, first_pattern)
    marker_local = random.choice(marker_options)

    active = set(first_pattern)
    for gap_count in gap_counts[1:]:
        pattern = _choose_gap_pattern(corridor_width, gap_count, active)
        if pattern is None:
            return None
        gap_patterns.append(pattern)
        active = set(pattern)

    g = _build_grid(
        edge=edge,
        axis_len=axis_len,
        lateral_len=lateral_len,
        corridor_start=side_margin_a,
        corridor_width=corridor_width,
        wall_distances=wall_distances,
        gap_patterns=tuple(gap_patterns),
        marker_local=marker_local,
    )

    try:
        out = main(g)
    except Exception:
        return None

    if out.shape != g.shape or not np.any(out != g):
        return None
    if np.count_nonzero(out == RED) == 0:
        return None
    hit_count, max_depth, max_branching = _trace_signature(g)
    if hit_count < wall_count or max_depth < 2 or max_branching < 2:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 8")
