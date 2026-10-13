from __future__ import annotations

import numpy as np


BG_COLOR = 0
DIVIDER_COLOR = 5
POINT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400


def _find_divider(grid: np.ndarray) -> tuple[str, int] | None:
    h, w = grid.shape
    row_hits = [r for r in range(h) if np.all(grid[r, :] == DIVIDER_COLOR)]
    col_hits = [c for c in range(w) if np.all(grid[:, c] == DIVIDER_COLOR)]
    if len(row_hits) == 1 and not col_hits:
        return "h", int(row_hits[0])
    if len(col_hits) == 1 and not row_hits:
        return "v", int(col_hits[0])
    return None


def _point_coord(
    orientation: str,
    sep: int,
    track: int,
    side: int,
    distance: int,
) -> tuple[int, int]:
    if orientation == "v":
        return int(track), int(sep + side * distance)
    return int(sep + side * distance), int(track)


def _landing_positions(
    shape: tuple[int, int],
    orientation: str,
    sep: int,
    track: int,
    side: int,
    distance: int,
) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    k = 1
    while True:
        axis = int(sep - side * distance * k)
        if orientation == "v":
            r, c = int(track), axis
        else:
            r, c = axis, int(track)
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        k += 1
    return cells


def _render_input(
    shape: tuple[int, int],
    orientation: str,
    sep: int,
    points: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    if orientation == "v":
        grid[:, sep] = DIVIDER_COLOR
    else:
        grid[sep, :] = DIVIDER_COLOR
    for track, side, distance, color, _pid in points:
        r, c = _point_coord(orientation, sep, track, side, distance)
        grid[r, c] = int(color)
    return grid


def _paint_periodic_projections(
    base: np.ndarray,
    orientation: str,
    sep: int,
    points: list[tuple[int, int, int, int, int]],
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    out = np.array(base, copy=True)
    traces: dict[int, dict[str, object]] = {}
    ordered = sorted(points, key=lambda p: (p[2], p[0], p[1], p[3], p[4]))
    for track, side, distance, color, pid in ordered:
        painted: list[tuple[int, int]] = []
        blocked = 0
        for r, c in _landing_positions(out.shape, orientation, sep, track, side, distance):
            if int(out[r, c]) == BG_COLOR:
                out[r, c] = int(color)
                painted.append((int(r), int(c)))
            else:
                blocked += 1
        traces[int(pid)] = {
            "painted": painted,
            "blocked": int(blocked),
            "distance": int(distance),
            "track": int(track),
            "side": int(side),
            "color": int(color),
        }
    return out, traces


def _extract_points(
    grid: np.ndarray,
    orientation: str,
    sep: int,
) -> list[tuple[int, int, int, int, int]]:
    points: list[tuple[int, int, int, int, int]] = []
    for pid, (r, c) in enumerate(np.argwhere((grid != BG_COLOR) & (grid != DIVIDER_COLOR))):
        rr, cc = int(r), int(c)
        color = int(grid[rr, cc])
        if orientation == "v":
            side = -1 if cc < sep else 1
            distance = abs(cc - sep)
            track = rr
        else:
            side = -1 if rr < sep else 1
            distance = abs(rr - sep)
            track = cc
        points.append((int(track), int(side), int(distance), color, int(pid)))
    return points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each isolated colored pixel through the grey divider and repeat it periodically on the opposite side.

    Input structure:
    - A black background contains exactly one full grey divider line spanning an entire row or an entire column.
    - The other non-black cells are isolated colored singleton pixels placed on either side of that divider.
    - Each singleton lies some perpendicular distance away from the divider and keeps its own row or column track.

    Rule:
    - For each colored pixel, let d be its perpendicular distance to the grey divider.
    - Cast through the divider along the same row or column and color the cells on the opposite side at distances d, 2d, 3d, ... from the divider until the grid ends.
    - Process pixels from nearest to farthest from the divider; later projections never overwrite already colored cells. Keep the original pixels and the grey divider unchanged.

    Key insight:
    - The divider is a shared measuring line: the first hit is only the mirror image, while the true rule is a periodic extension whose step size equals the pixel's own distance from that divider.
    """
    g = np.asarray(grid, dtype=int)
    divider = _find_divider(g)
    if divider is None:
        return g.copy()
    orientation, sep = divider
    points = _extract_points(g, orientation, sep)
    out, _ = _paint_periodic_projections(g, orientation, sep, points)
    return out


def _choose_spaced_tracks(rng: np.random.Generator, length: int, count: int) -> list[int]:
    if count == 0:
        return []
    base = np.sort(rng.choice(length - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _build_points(
    rng: np.random.Generator,
    point_count: int,
    max_distance: int,
    track_count: int,
    with_overlap: bool,
) -> list[tuple[int, int, int]]:
    tracks = _choose_spaced_tracks(rng, track_count, point_count - 1 if with_overlap else point_count)
    if with_overlap:
        pair_options = [pair for pair in [(2, 4), (2, 6), (1, 3)] if pair[1] <= max_distance]
        near_d, far_d = pair_options[int(rng.integers(len(pair_options)))]
        used = {near_d, far_d}
        extra_needed = point_count - 2
        pool = [d for d in range(1, max_distance + 1) if d not in used]
        extra_distances = list(map(int, rng.choice(pool, size=extra_needed, replace=False))) if extra_needed else []

        pair_track = int(tracks[int(rng.integers(len(tracks)))])
        other_tracks = [int(t) for t in tracks if int(t) != pair_track]
        pair_side = -1 if int(rng.integers(2)) == 0 else 1
        other_sides = list(map(int, rng.choice(np.array([-1, 1], dtype=int), size=extra_needed, replace=True)))
        if extra_needed and all(side == pair_side for side in other_sides):
            other_sides[0] = -pair_side

        specs = [(pair_track, pair_side, near_d), (pair_track, pair_side, far_d)]
        for track, side, distance in zip(other_tracks, other_sides, extra_distances):
            specs.append((int(track), int(side), int(distance)))
        rng.shuffle(specs)
        return specs

    anchor_choices = [d for d in (2, 1) if d <= max_distance]
    anchor_d = int(anchor_choices[int(rng.integers(len(anchor_choices)))])
    used = {anchor_d}
    pool = [d for d in range(1, max_distance + 1) if d not in used]
    extra_distances = list(map(int, rng.choice(pool, size=point_count - 1, replace=False))) if point_count > 1 else []
    distances = [anchor_d] + extra_distances
    rng.shuffle(distances)

    sides = list(map(int, rng.choice(np.array([-1, 1], dtype=int), size=point_count, replace=True)))
    if all(side == sides[0] for side in sides):
        sides[0] = -sides[0]

    specs = [(int(track), int(side), int(distance)) for track, side, distance in zip(tracks, sides, distances)]
    rng.shuffle(specs)
    return specs


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    orientation: str,
    sep: int,
    points: list[tuple[int, int, int, int, int]],
    traces: dict[int, dict[str, object]],
    with_overlap: bool,
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    if orientation == "v":
        if sum(np.all(inp[:, c] == DIVIDER_COLOR) for c in range(inp.shape[1])) != 1:
            return False
        if np.any(np.all(inp == DIVIDER_COLOR, axis=1)):
            return False
    else:
        if sum(np.all(inp[r, :] == DIVIDER_COLOR) for r in range(inp.shape[0])) != 1:
            return False
        if np.any(np.all(inp == DIVIDER_COLOR, axis=0)):
            return False

    distances = [point[2] for point in points]
    if len(distances) != len(set(distances)):
        return False
    if {point[1] for point in points} != {-1, 1}:
        return False

    new_cells = int(np.count_nonzero(expected != inp))
    if new_cells < 4:
        return False

    painted_counts = [len(traces[point[4]]["painted"]) for point in points]
    blocked_counts = [int(traces[point[4]]["blocked"]) for point in points]
    if max(painted_counts, default=0) < 3:
        return False
    if with_overlap:
        if max(blocked_counts, default=0) == 0:
            return False
    else:
        if any(blocked_counts):
            return False

    for track, side, distance, color, _pid in points:
        r, c = _point_coord(orientation, sep, track, side, distance)
        if int(expected[r, c]) != int(color):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    orientation = "v" if int(rng.integers(2)) == 0 else "h"
    with_overlap = bool(rng.random() < 0.75)

    max_distance = int(rng.integers(4, 8))
    point_count = int(rng.integers(3, min(6, max_distance) + 1))
    left_len = max_distance + int(rng.integers(1, 5))
    right_len = max_distance + int(rng.integers(1, 5))
    cross_dim = left_len + 1 + right_len

    unique_tracks = point_count - 1 if with_overlap else point_count
    min_track_dim = max(8, 2 * unique_tracks + 1)
    max_track_dim = 18
    if min_track_dim > max_track_dim:
        return None
    track_dim = int(rng.integers(min_track_dim, max_track_dim + 1))

    specs = _build_points(rng, point_count, max_distance, track_dim, with_overlap)
    colors = list(map(int, rng.choice(POINT_COLORS, size=point_count, replace=False)))
    rng.shuffle(colors)
    points = [
        (int(track), int(side), int(distance), int(color), int(pid))
        for pid, ((track, side, distance), color) in enumerate(zip(specs, colors))
    ]

    sep = int(left_len)
    shape = (track_dim, cross_dim) if orientation == "v" else (cross_dim, track_dim)
    inp = _render_input(shape, orientation, sep, points)
    expected, traces = _paint_periodic_projections(inp, orientation, sep, points)

    if not _validate_case(inp, expected, orientation, sep, points, traces, with_overlap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
