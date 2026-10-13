import math
import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3


SPRITE_SHAPES = [
    [(-1, 0), (0, 0), (1, 0)],
    [(0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, 0), (1, 0), (0, 1)],
    [(-1, 0), (0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
]


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sprite_center(coords):
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    assert center in set(coords)
    return center


def _color_components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _scan_hit(grid, track_row, col, step):
    r = track_row + step
    while 0 <= r < grid.shape[0]:
        if grid[r, col] == BLUE:
            return abs(r - track_row), r
        r += step
    return math.inf, None


def _infer_track_cols(visible_cols, covered_col):
    base = sorted(set(int(c) for c in visible_cols) | {int(covered_col)})
    assert len(base) >= 6
    diffs = [b - a for a, b in zip(base, base[1:])]
    spacing = 0
    for diff in diffs:
        spacing = diff if spacing == 0 else math.gcd(spacing, diff)
    assert spacing >= 2
    start = min(base)
    end = max(base)
    cols = list(range(start, end + 1, spacing))
    assert covered_col in cols
    assert len(cols) >= 6
    for c in visible_cols:
        assert c in cols
    return cols, spacing


def _extract_scene(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 3 <= len(red_coords) <= 5
    sprite_center_r, sprite_center_c = _sprite_center(red_coords)

    green_counts = [int(np.count_nonzero(g[r] == GREEN)) for r in range(g.shape[0])]
    track_row = int(np.argmax(green_counts))
    assert green_counts[track_row] >= 5
    assert sprite_center_r == track_row
    visible_track_cols = sorted(int(c) for c in np.where(g[track_row] == GREEN)[0])
    track_cols, spacing = _infer_track_cols(visible_track_cols, sprite_center_c)

    up = []
    down = []
    for col in track_cols:
        up.append(_scan_hit(g, track_row, col, -1))
        down.append(_scan_hit(g, track_row, col, 1))

    balance = [
        idx
        for idx, ((d_up, _), (d_down, _)) in enumerate(zip(up, down))
        if math.isfinite(d_up) and d_up == d_down
    ]
    assert len(balance) == 1

    return {
        "grid": g,
        "track_row": track_row,
        "track_cols": track_cols,
        "spacing": spacing,
        "sprite_coords": red_coords,
        "sprite_center": (sprite_center_r, sprite_center_c),
        "up": up,
        "down": down,
        "balance_idx": balance[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the red sprite to the unique track slot whose upward and downward blue-ray hits are equally far, then stamp both hits.

    Input structure:
    - Black background surrounds one horizontal row of evenly spaced green track dots.
    - A single small 4-connected red sprite overlaps exactly one interior track slot.
    - Blue foreground appears only as horizontal segments on rows above and below the track row, never on the track itself.

    Rule:
    - For each track slot column, scan upward and downward to the nearest blue cell in that column.
    - Find the unique slot whose finite upward and downward distances are equal.
    - Translate the whole red sprite horizontally so its center aligns with that balance slot, restoring the vacated covered slot to green.
    - At the upper and lower hit cells of the balance column, stamp a red 3x3 marker and then restore each marker center to blue.

    Key insight:
    - The target slot is not the next slot or a geometric midpoint; it is the only track position where the column-wise nearest blue hits above and below balance exactly.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    out = g.copy()

    for r, c in scene["sprite_coords"]:
        out[r, c] = BG
    for c in scene["track_cols"]:
        out[scene["track_row"], c] = GREEN

    balance_col = scene["track_cols"][scene["balance_idx"]]
    dc = balance_col - scene["sprite_center"][1]
    for r, c in scene["sprite_coords"]:
        out[r, c + dc] = RED

    for dist, hit_row in [scene["up"][scene["balance_idx"]], scene["down"][scene["balance_idx"]]]:
        assert math.isfinite(dist) and hit_row is not None
        for rr in range(hit_row - 1, hit_row + 2):
            for cc in range(balance_col - 1, balance_col + 2):
                if _in_bounds(out.shape, rr, cc):
                    out[rr, cc] = RED
        out[hit_row, balance_col] = BLUE

    return out


def _paint_segment(grid, row, c0, c1):
    assert 0 <= row < grid.shape[0]
    assert 0 <= c0 <= c1 < grid.shape[1]
    assert c1 - c0 + 1 >= 3
    grid[row, c0 : c1 + 1] = BLUE


def _build_segments(grid, track_cols, track_row, distances, sign):
    runs_by_row = defaultdict(list)
    for idx, dist in enumerate(distances):
        if dist is None:
            continue
        row = track_row + sign * dist
        runs_by_row[row].append(idx)

    for row, indices in runs_by_row.items():
        indices.sort()
        start = prev = indices[0]
        for idx in indices[1:] + [None]:
            if idx is not None and idx == prev + 1:
                prev = idx
                continue
            c0 = track_cols[start] - 1
            c1 = track_cols[prev] + 1
            _paint_segment(grid, row, c0, c1)
            if idx is None:
                break
            start = prev = idx


def _place_sprite(grid, center_r, center_c, shape):
    coords = []
    for dr, dc in shape:
        rr, cc = center_r + dr, center_c + dc
        assert _in_bounds(grid.shape, rr, cc)
        assert grid[rr, cc] in (BG, GREEN)
        coords.append((rr, cc))
    for rr, cc in coords:
        grid[rr, cc] = RED


def _blue_segment_count(grid, track_row, upper):
    blue = (grid == BLUE)
    if upper:
        blue[track_row:] = False
    else:
        blue[: track_row + 1] = False
    return len(_color_components(blue.astype(int), 1))


def _validate_input_semantics(grid):
    scene = _extract_scene(grid)
    g = np.asarray(grid, dtype=int)
    track_row = scene["track_row"]
    track_cols = scene["track_cols"]
    balance_idx = scene["balance_idx"]

    assert len(track_cols) >= 6
    assert 0 < balance_idx < len(track_cols) - 1
    assert scene["sprite_center"][1] != track_cols[balance_idx]

    finite_both = [
        idx
        for idx, ((du, _), (dd, _)) in enumerate(zip(scene["up"], scene["down"]))
        if math.isfinite(du) and math.isfinite(dd)
    ]
    assert len(finite_both) >= 3
    up_values = {int(d) for d, _ in scene["up"] if math.isfinite(d)}
    down_values = {int(d) for d, _ in scene["down"] if math.isfinite(d)}
    assert len(up_values) >= 2
    assert len(down_values) >= 2
    assert _blue_segment_count(g, track_row, upper=True) >= 2
    assert _blue_segment_count(g, track_row, upper=False) >= 2

    for comp in _color_components(g, BLUE):
        rows = {r for r, _ in comp}
        cols = sorted(c for _, c in comp)
        assert len(rows) == 1
        assert len(cols) >= 3
        assert cols[-1] - cols[0] + 1 == len(cols)
        assert len(cols) < g.shape[1]
        assert next(iter(rows)) != track_row

    midpoint_like = (len(track_cols) - 1) / 2.0
    if balance_idx == int(midpoint_like) and midpoint_like.is_integer():
        diff_profile = [abs(du - dd) for (du, _), (dd, _) in zip(scene["up"], scene["down"]) if math.isfinite(du) and math.isfinite(dd)]
        assert diff_profile.count(0) == 1

    out = main(g)
    assert not np.array_equal(out, g)
    assert np.count_nonzero(out == RED) >= np.count_nonzero(g == RED) + 10


def _candidate_distances(rng, n_slots, balance_idx, max_dist):
    balance_dist = rng.randint(2, max_dist)
    upper = [rng.randint(2, max_dist) for _ in range(n_slots)]
    lower = [rng.randint(2, max_dist) for _ in range(n_slots)]
    upper[balance_idx] = balance_dist
    lower[balance_idx] = balance_dist

    if n_slots >= 7 and rng.random() < 0.5:
        edge = 0 if rng.random() < 0.5 else n_slots - 1
        upper[edge] = None
    if n_slots >= 7 and rng.random() < 0.5:
        edge = 0 if rng.random() < 0.5 else n_slots - 1
        lower[edge] = None

    for idx in range(n_slots):
        if idx == balance_idx:
            continue
        if upper[idx] is None and lower[idx] is None:
            if rng.random() < 0.5:
                upper[idx] = rng.randint(2, max_dist)
            else:
                lower[idx] = rng.randint(2, max_dist)
        if upper[idx] is not None and lower[idx] is not None and upper[idx] == lower[idx]:
            choices = [d for d in range(2, max_dist + 1) if d != upper[idx]]
            lower[idx] = rng.choice(choices)

    return upper, lower


def _gen_attempt(rng):
    feasible_pairs = []
    for n_slots, spacing in [(6, 3), (6, 4), (7, 3), (7, 4), (8, 3)]:
        min_w = max(18, (n_slots - 1) * spacing + 7)
        if min_w <= 30:
            feasible_pairs.append((n_slots, spacing, min_w))
    n_slots, spacing, min_w = rng.choice(feasible_pairs)
    width = rng.randint(min_w, min(30, min_w + 4))
    height = rng.randint(15, 22)

    track_row_lo = 6
    track_row_hi = height - 7
    if track_row_lo > track_row_hi:
        return None
    track_row = rng.randint(track_row_lo, track_row_hi)

    balance_candidates = list(range(1, n_slots - 1))
    if rng.random() < 0.7:
        non_mid = [i for i in balance_candidates if i != n_slots // 2]
        if non_mid:
            balance_candidates = non_mid
    balance_idx = rng.choice(balance_candidates)

    start_candidates = [i for i in range(1, n_slots - 1) if i != balance_idx]
    start_idx = rng.choice(start_candidates)

    first_col = rng.randint(2, width - 3 - (n_slots - 1) * spacing)
    track_cols = [first_col + i * spacing for i in range(n_slots)]

    max_dist = min(track_row - 2, height - track_row - 3, 6)
    if max_dist < 3:
        return None

    for _ in range(80):
        upper, lower = _candidate_distances(rng, n_slots, balance_idx, max_dist)
        if len({d for d in upper if d is not None}) < 2 or len({d for d in lower if d is not None}) < 2:
            continue
        finite_both = sum(u is not None and d is not None for u, d in zip(upper, lower))
        if finite_both < 3:
            continue

        grid = np.zeros((height, width), dtype=int)
        for col in track_cols:
            grid[track_row, col] = GREEN
        _build_segments(grid, track_cols, track_row, upper, -1)
        _build_segments(grid, track_cols, track_row, lower, 1)

        shape = rng.choice(SPRITE_SHAPES)
        _place_sprite(grid, track_row, track_cols[start_idx], shape)

        try:
            _validate_input_semantics(grid)
        except AssertionError:
            continue

        scene = _extract_scene(grid)
        if scene["balance_idx"] != balance_idx:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid balance-track input")
