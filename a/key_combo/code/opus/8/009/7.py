from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _ray_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    height, _ = grid.shape
    for r, c in np.argwhere(grid == GREY):
        mask[int(r) : height, int(c)] = True
    return mask


def _pair_records(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    colors = sorted(int(x) for x in np.unique(grid) if int(x) not in {BG, GREY})
    records: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        first = tuple(int(x) for x in coords[0])
        second = tuple(int(x) for x in coords[1])
        a, b = sorted((first, second), key=lambda rc: (rc[0], rc[1]))
        assert a[0] != b[0] and a[1] != b[1], f"pair {color} must not share row or column"
        records.append((a[0], a[1], b[0], b[1], color))
    records.sort(key=lambda item: (item[0], item[1]))
    return records


def _first_blocking_column(ray_mask: np.ndarray, row: int, start_col: int, end_col: int) -> int | None:
    step = 1 if end_col > start_col else -1
    col = start_col
    while col != end_col:
        next_col = col + step
        if ray_mask[row, next_col]:
            return next_col
        col = next_col
    return None


def _draw_horizontal(out: np.ndarray, ray_mask: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    for c in _inclusive_range(c0, c1):
        if not ray_mask[row, c]:
            out[row, c] = color


def _draw_vertical(out: np.ndarray, ray_mask: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    for r in _inclusive_range(r0, r1):
        if not ray_mask[r, col]:
            out[r, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend grey rays downward, then connect same-color markers with L paths that bend into Z-like detours when blocked.

    Input structure:
    - A black background contains 2-3 same-color marker pairs, with each non-grey
      color appearing exactly twice as single pixels.
    - One or two grey single pixels sit in the upper half and act as downward ray emitters.
    - Within each color pair, the two markers are in different rows and different columns.

    Rule:
    - First extend every grey pixel straight downward to the bottom edge, keeping the full
      grey ray in the output.
    - For each color pair, take the upper marker A and lower marker B and draw a horizontal-first path.
    - If the next horizontal step from A would enter a grey ray before reaching B's column,
      stop one column before the ray, draw vertically down to B's row, then draw horizontally
      toward B; otherwise draw the standard L path by reaching B's column first and then going down.
    - Path pixels never overwrite grey ray pixels, later paths overwrite earlier ones, and the
      original non-grey markers are restored at the end.

    Key insight:
    - The grey structures are obstacles rather than targets: they must be expanded first so each
      colored pair can decide whether its horizontal leg stays an L or is forced into a bent detour.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    rays = _ray_mask(g)
    out = g.copy()
    out[rays] = GREY

    for ra, ca, rb, cb, color in _pair_records(g):
        block_col = _first_blocking_column(rays, ra, ca, cb)
        if block_col is None:
            _draw_horizontal(out, rays, ra, ca, cb, color)
            _draw_vertical(out, rays, cb, ra, rb, color)
        else:
            step = 1 if cb > ca else -1
            stop_col = block_col - step
            _draw_horizontal(out, rays, ra, ca, stop_col, color)
            _draw_vertical(out, rays, stop_col, ra, rb, color)
            _draw_horizontal(out, rays, rb, stop_col, cb, color)

    marker_mask = (g != BG) & (g != GREY)
    out[marker_mask] = g[marker_mask]
    return out


def _sample_ray_columns(width: int, n_rays: int) -> list[int]:
    if n_rays == 1:
        return [random.randint(3, width - 4)]

    candidates = []
    for c1 in range(3, width - 6):
        for c2 in range(c1 + 3, width - 3):
            candidates.append((c1, c2))
    if not candidates:
        raise ValueError("no feasible two-ray layout")
    return list(random.choice(candidates))


def _zones_from_rays(width: int, ray_cols: list[int]) -> list[list[int]]:
    zones: list[list[int]] = []
    start = 1
    for col in ray_cols:
        zones.append(list(range(start, col)))
        start = col + 1
    zones.append(list(range(start, width - 1)))
    return zones


def _lane_rows(height: int, n_pairs: int) -> list[tuple[int, int]]:
    base_height = 5
    gap = 1
    lane_heights = [base_height] * n_pairs
    used = sum(lane_heights) + gap * (n_pairs - 1)
    extra = height - used - 2
    assert extra >= 0
    for _ in range(extra):
        lane_heights[random.randrange(n_pairs)] += 1

    start = 1
    lanes: list[tuple[int, int]] = []
    for idx, lane_height in enumerate(lane_heights):
        end = start + lane_height - 1
        lanes.append((start, end))
        start = end + 1 + (gap if idx < n_pairs - 1 else 0)
    return lanes


def _sample_pair_rows(lane: tuple[int, int]) -> tuple[int, int]:
    start, end = lane
    top = random.randint(start, min(start + 1, end - 3))
    bottom = random.randint(top + 3, end)
    return top, bottom


def _actual_blocked(ray_rows: dict[int, int], a_row: int, a_col: int, b_col: int) -> bool:
    step = 1 if b_col > a_col else -1
    for col in range(a_col + step, b_col, step):
        ray_row = ray_rows.get(col)
        if ray_row is not None and ray_row <= a_row:
            return True
    return False


def _build_candidate() -> np.ndarray | None:
    n_pairs = random.choice((2, 3))
    n_rays = random.choice((1, 2))

    min_height = 5 * n_pairs + (n_pairs - 1) + 2
    max_height = min(24, min_height + 4)
    min_width = 14 if n_rays == 1 else 16
    max_width = 22
    height = random.randint(min_height, max_height)
    width = random.randint(min_width, max_width)

    ray_cols = _sample_ray_columns(width, n_rays)
    zones = _zones_from_rays(width, ray_cols)
    if any(len(zone) == 0 for zone in zones):
        return None

    same_zone_options = [idx for idx, zone in enumerate(zones) if len(zone) >= 2]
    blocked_options = [idx for idx in range(n_rays) if zones[idx] and zones[idx + 1]]
    if not same_zone_options or not blocked_options:
        return None

    colors = random.sample(PAIR_COLORS, n_pairs)
    lanes = _lane_rows(height, n_pairs)

    pair_modes: list[tuple[str, int]] = [
        ("blocked", random.choice(blocked_options)),
        ("unblocked", random.choice(same_zone_options)),
    ]
    while len(pair_modes) < n_pairs:
        if random.random() < 0.5:
            pair_modes.append(("blocked", random.choice(blocked_options)))
        else:
            pair_modes.append(("unblocked", random.choice(same_zone_options)))
    random.shuffle(pair_modes)

    markers: list[tuple[int, int, int, int, int]] = []
    blocked_rows: dict[int, list[int]] = {col: [] for col in ray_cols}
    for color, lane, mode in zip(colors, lanes, pair_modes):
        a_row, b_row = _sample_pair_rows(lane)
        kind, idx = mode
        if kind == "blocked":
            left_col = random.choice(zones[idx])
            right_col = random.choice(zones[idx + 1])
            if random.random() < 0.5:
                a_col, b_col = left_col, right_col
            else:
                a_col, b_col = right_col, left_col
            blocked_rows[ray_cols[idx]].append(a_row)
        else:
            a_col, b_col = random.sample(zones[idx], 2)
        markers.append((color, a_row, a_col, b_row, b_col))

    upper_limit = height // 2 - 1
    if upper_limit < 0:
        return None

    ray_rows: dict[int, int] = {}
    for col in ray_cols:
        if blocked_rows[col]:
            limit = min(upper_limit, min(blocked_rows[col]))
            if limit < 0:
                return None
            ray_rows[col] = random.randint(0, limit)
        else:
            ray_rows[col] = random.randint(0, upper_limit)

    grid = np.zeros((height, width), dtype=int)
    for color, a_row, a_col, b_row, b_col in markers:
        grid[a_row, a_col] = color
        grid[b_row, b_col] = color
    for col, row in ray_rows.items():
        grid[row, col] = GREY

    statuses = [_actual_blocked(ray_rows, a_row, a_col, b_col) for _, a_row, a_col, _, b_col in markers]
    if not any(statuses) or all(statuses):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
