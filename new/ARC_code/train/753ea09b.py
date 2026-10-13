from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg, label_mask, neighbors4, paste
import itertools
import random


def _unique_largest_color(g: np.ndarray, colors: list[int]) -> int:
    counts = Counter(int(v) for v in g.ravel())
    ranked = sorted(colors, key=lambda color: counts[int(color)], reverse=True)
    assert len(ranked) >= 2, "expected at least two candidate colors"
    assert counts[ranked[0]] > counts[ranked[1]], "foreground majority must be unique"
    return int(ranked[0])


def _long_component_ids(lab: np.ndarray, n: int) -> set[int]:
    h = int(lab.shape[0])
    min_row_span = max(1, h // 3)
    ids: set[int] = set()
    for comp_id in range(1, n + 1):
        coords = np.argwhere(lab == comp_id)
        assert coords.size > 0, "component labels must be nonempty"
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        if r1 - r0 + 1 >= min_row_span:
            ids.add(int(comp_id))
    assert len(ids) >= 2, "expected at least two long contour components"
    return ids


def _two_exterior_bg_ids(bg_lab: np.ndarray, n_bg: int) -> set[int]:
    assert n_bg >= 3, "expected interior and exterior background regions"
    sizes = {comp_id: int((bg_lab == comp_id).sum()) for comp_id in range(1, n_bg + 1)}
    size_values = sorted(sizes.values(), reverse=True)
    assert size_values[1] > size_values[2], "two exterior regions must be separated by size"
    cutoff = size_values[1]
    exterior = {comp_id for comp_id, size in sizes.items() if size >= cutoff}
    assert len(exterior) == 2, "expected exactly two exterior background regions"
    return exterior


def _touches_long_contour(
    coords: np.ndarray,
    fill_lab: np.ndarray,
    long_ids: set[int],
) -> bool:
    for r, c in coords:
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < fill_lab.shape[0] and 0 <= cc < fill_lab.shape[1]:
                if int(fill_lab[rr, cc]) in long_ids:
                    return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the small background compartments bounded by the long contour color.

    Input structure:
    - The grid has one dominant background color and two foreground colors.
    - The more frequent foreground color forms long 8-connected contour fragments.
    - The other foreground color is a fixed distractor/obstacle color that partitions
      the background but is never recolored.

    Rule:
    - Split the background into 4-connected regions using all foreground pixels as
      barriers.
    - Keep the two largest background regions as the outside of the contour drawing.
    - Recolor every other background region that touches a long contour component
      to the contour color; leave all foreground pixels unchanged.

    Key insight:
    The apparent area fill is exactly a component-level fill of bounded background
    compartments, not a row-wise interpolation between individual contour pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 2, "expected exactly two foreground colors"

    fill_color = _unique_largest_color(g, foreground)
    fill_lab, _n_fill = label_mask(g == fill_color, connectivity=8)
    long_ids = _long_component_ids(fill_lab, _n_fill)

    bg_lab, n_bg = label_mask(g == bg, connectivity=4)
    exterior_ids = _two_exterior_bg_ids(bg_lab, n_bg)

    out = g.copy()
    for comp_id in range(1, n_bg + 1):
        if comp_id in exterior_ids:
            continue
        coords = np.argwhere(bg_lab == comp_id)
        if _touches_long_contour(coords, fill_lab, long_ids):
            out[bg_lab == comp_id] = fill_color

    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
H, W = 30, 30
GRID_SHAPE = (H, W)
ARC_COLOR_COUNT = 10

# Long contour path layout.
PATH_COUNT_OPTIONS = (2, 3)
PATH_STEP_OPTIONS = (-1, 0, 1)
PATH_GAP_MIN, PATH_GAP_MAX = 2, 5
PATH_LEFT_MIN_COL, PATH_RIGHT_MARGIN = 2, 3
PATH_START_MIN_COL = 3
PATH_START_RIGHT_MARGIN = 4
PATH_FAR_END_MIN_CHOICES, PATH_FAR_END_PROB, PATH_FAR_END_DELTA = 8, 0.7, 5
PATH_SCORING_CANDIDATES = 5
PATH_MIN_MOVES = 12

# Contour distractors and extra contour pixels.
OUTSIDE_MARGIN = 2
OUTSIDE_SIDE_OPTIONS = (-1, 1)
SHORT_FRAGMENT_COUNT_MIN, SHORT_FRAGMENT_COUNT_MAX = 3, 5
SHORT_FRAGMENT_SIZE_MIN, SHORT_FRAGMENT_SIZE_MAX = 1, 4
SHORT_FRAGMENT_ATTEMPTS = 200
SHORT_FRAGMENT_PATH_PADDING = 3
CONTOUR_PIXELS_MIN, CONTOUR_PIXELS_MAX = 81, 108

# Obstacle bars and scattered clusters.
BAR_SEGMENT_MAX = 11
SINGLE_CHANNEL_BARS_MIN, SINGLE_CHANNEL_BARS_MAX = 5, 11
MULTI_CHANNEL_TOTAL_BARS_MIN, MULTI_CHANNEL_TOTAL_BARS_MAX = 4, 10
MULTI_CHANNEL_INITIAL_BARS = 2
MULTI_CHANNEL_MAX_BARS = 5
BAR_WIDTH_MIN, BAR_WIDTH_MAX = 1, 5
OBSTACLE_PIXELS_MIN, OBSTACLE_PIXELS_MAX = 36, 63
OBSTACLE_EXTRA_PIXELS_MIN = 8
OBSTACLE_OUTSIDE_QUOTA_MIN, OBSTACLE_OUTSIDE_QUOTA_MAX = 8, 16
OBSTACLE_CLUSTER_SIZE_MIN, OBSTACLE_CLUSTER_SIZE_MAX = 1, 5
OBSTACLE_ATTEMPTS = 300
OBSTACLE_ZONE_OPTIONS = ("outside", "inside", "any")

# Component validation.
BG_REGIONS_MIN, BG_REGIONS_MAX = 8, 14
EXTERIOR_REGION_COUNT = 2
INTERIOR_REGIONS_MIN, INTERIOR_REGIONS_MAX = 6, 12
INTERIOR_REGION_AREA_MAX = 57
LONG_COMPONENT_MIN_SPAN = H // 3
LONG_COMPONENT_COUNT_OPTIONS = (2, 3)

# Retry limits and local neighborhoods.
GENERATION_ATTEMPTS = 300
NEIGHBOR_DELTAS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBOR_DELTAS_8 = NEIGHBOR_DELTAS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _color_choices() -> tuple[int, int, int]:
    colors = list(range(ARC_COLOR_COUNT))
    random.shuffle(colors)
    return int(colors[0]), int(colors[1]), int(colors[2])


def _path_columns(path_count: int) -> list[list[int]] | None:
    gaps = [random.randint(PATH_GAP_MIN, PATH_GAP_MAX) for _ in range(path_count - 1)]
    total_width = sum(gap + 1 for gap in gaps)
    start_hi = W - PATH_START_RIGHT_MARGIN - total_width
    if start_hi < PATH_START_MIN_COL:
        return None

    start = random.randint(PATH_START_MIN_COL, start_hi)
    end_choices = list(range(PATH_START_MIN_COL, start_hi + 1))
    if len(end_choices) > PATH_FAR_END_MIN_CHOICES and random.random() < PATH_FAR_END_PROB:
        far = [c for c in end_choices if abs(c - start) >= PATH_FAR_END_DELTA]
        if far:
            end_choices = far
    end = random.choice(end_choices)

    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)
    targets = [c + (end - start) for c in cols]

    paths = [[int(c)] for c in cols]
    for r in range(1, H):
        valid: list[list[int]] = []
        for steps in itertools.product(PATH_STEP_OPTIONS, repeat=path_count):
            nxt = [cols[i] + steps[i] for i in range(path_count)]
            if nxt[0] < PATH_LEFT_MIN_COL or nxt[-1] > W - PATH_RIGHT_MARGIN:
                continue
            if all(PATH_GAP_MIN <= nxt[i + 1] - nxt[i] - 1 <= PATH_GAP_MAX for i in range(path_count - 1)):
                valid.append(nxt)
        if not valid:
            return None
        ideal = [paths[i][0] + (targets[i] - paths[i][0]) * r / (H - 1) for i in range(path_count)]
        scored = sorted(
            valid,
            key=lambda nxt: sum(abs(nxt[i] - ideal[i]) for i in range(path_count)) + random.random(),
        )
        cols = random.choice(scored[: max(1, min(PATH_SCORING_CANDIDATES, len(scored)))])
        for i, c in enumerate(cols):
            paths[i].append(int(c))

    moves = sum(paths[i][r] != paths[i][r - 1] for i in range(path_count) for r in range(1, H))
    if moves < PATH_MIN_MOVES:
        return None
    return paths


def _outside_enclosure(paths: list[list[int]], r: int, c: int, margin: int = OUTSIDE_MARGIN) -> bool:
    return c <= paths[0][r] - margin or c >= paths[-1][r] + margin


def _inside_channel(paths: list[list[int]], r: int, c: int) -> bool:
    return any(paths[i][r] < c < paths[i + 1][r] for i in range(len(paths) - 1))


def _touches_mask(mask: np.ndarray, coords: list[tuple[int, int]], connectivity: int) -> bool:
    seen = set(coords)
    deltas = NEIGHBOR_DELTAS_8 if connectivity == 8 else NEIGHBOR_DELTAS_4
    for r, c in coords:
        for dr, dc in deltas:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in seen and mask[rr, cc]:
                return True
    return False


def _place_short_fragments(
    grid: np.ndarray, contour_mask: np.ndarray, paths: list[list[int]], contour: int, count: int
) -> bool:
    for _ in range(count):
        size = random.randint(SHORT_FRAGMENT_SIZE_MIN, SHORT_FRAGMENT_SIZE_MAX)
        placed = False
        for _attempt in range(SHORT_FRAGMENT_ATTEMPTS):
            r = random.randrange(H)
            side = random.choice(OUTSIDE_SIDE_OPTIONS)
            if side < 0:
                hi = paths[0][r] - SHORT_FRAGMENT_PATH_PADDING - size + 1
                if hi < 0:
                    continue
                c0 = random.randint(0, hi)
            else:
                lo = paths[-1][r] + SHORT_FRAGMENT_PATH_PADDING
                if lo + size > W:
                    continue
                c0 = random.randint(lo, W - size)
            coords = [(r, c0 + i) for i in range(size)]
            if any(grid[rr, cc] != grid[0, 0] for rr, cc in coords):
                continue
            if _touches_mask(contour_mask, coords, 8):
                continue
            patch = np.full((1, size), contour, dtype=int)
            paste(grid, patch, r, c0)
            for rr, cc in coords:
                contour_mask[rr, cc] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _add_contour_extras(grid: np.ndarray, long_mask: np.ndarray, paths: list[list[int]], contour: int, needed: int) -> bool:
    if needed < 0:
        return False
    candidates: list[tuple[int, int]] = []
    for side, path_i in [(-1, 0), (1, len(paths) - 1)]:
        for r, c in enumerate(paths[path_i]):
            cc = c + side
            if 0 <= cc < W and grid[r, cc] == grid[0, 0]:
                candidates.append((r, cc))
    if needed > len(candidates):
        return False
    random.shuffle(candidates)
    for r, c in candidates[:needed]:
        grid[r, c] = contour
        long_mask[r, c] = True
    return True


def _segments_for_bars(n_bars: int, max_segment: int = BAR_SEGMENT_MAX) -> list[int] | None:
    segments = [1] * (n_bars + 1)
    remaining = H - n_bars - len(segments)
    if remaining < 0:
        return None
    while remaining:
        choices = [i for i, value in enumerate(segments) if value < max_segment]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(max_segment - segments[i], remaining))
        segments[i] += add
        remaining -= add
    random.shuffle(segments)
    return segments


def _bar_rows(n_bars: int) -> list[int] | None:
    segments = _segments_for_bars(n_bars)
    if segments is None:
        return None
    rows: list[int] = []
    r = segments[0]
    for i in range(n_bars):
        rows.append(r)
        r += 1 + segments[i + 1]
    return rows


def _add_obstacle_bars(grid: np.ndarray, paths: list[list[int]], obstacle: int) -> bool:
    channel_count = len(paths) - 1
    if channel_count == 1:
        bars_per_channel = [random.randint(SINGLE_CHANNEL_BARS_MIN, SINGLE_CHANNEL_BARS_MAX)]
    else:
        total = random.randint(MULTI_CHANNEL_TOTAL_BARS_MIN, MULTI_CHANNEL_TOTAL_BARS_MAX)
        bars_per_channel = [MULTI_CHANNEL_INITIAL_BARS, MULTI_CHANNEL_INITIAL_BARS]
        for _ in range(total - MULTI_CHANNEL_TOTAL_BARS_MIN):
            choices = [i for i, value in enumerate(bars_per_channel) if value < MULTI_CHANNEL_MAX_BARS]
            if not choices:
                break
            bars_per_channel[random.choice(choices)] += 1

    for channel_i, n_bars in enumerate(bars_per_channel):
        rows = _bar_rows(n_bars)
        if rows is None:
            return False
        for r in rows:
            c0 = paths[channel_i][r] + 1
            c1 = paths[channel_i + 1][r]
            if c1 - c0 < BAR_WIDTH_MIN or c1 - c0 > BAR_WIDTH_MAX:
                return False
            if np.any(grid[r, c0:c1] != grid[0, 0]):
                return False
            grid[r, c0:c1] = obstacle
    return True


def _random_cluster(seed: tuple[int, int], size: int) -> list[tuple[int, int]]:
    coords = [seed]
    while len(coords) < size:
        r, c = random.choice(coords)
        dr, dc = random.choice(NEIGHBOR_DELTAS_4)
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in coords:
            coords.append((rr, cc))
    return coords


def _can_place_obstacle(
    grid: np.ndarray, obstacle_mask: np.ndarray, contour_mask: np.ndarray, coords: list[tuple[int, int]],
    bg: int, paths: list[list[int]], zone: str
) -> bool:
    if zone == "outside" and not all(_outside_enclosure(paths, r, c, OUTSIDE_MARGIN) for r, c in coords):
        return False
    if zone == "inside" and not all(_inside_channel(paths, r, c) for r, c in coords):
        return False
    if any(grid[r, c] != bg for r, c in coords):
        return False
    if _touches_mask(obstacle_mask, coords, 4):
        return False
    if _touches_mask(contour_mask, coords, 4):
        return False
    return True


def _scatter_obstacles(
    grid: np.ndarray, obstacle_mask: np.ndarray, contour_mask: np.ndarray, paths: list[list[int]], bg: int, obstacle: int
) -> bool:
    bar_pixels = int(obstacle_mask.sum())
    low = max(OBSTACLE_PIXELS_MIN, bar_pixels + OBSTACLE_EXTRA_PIXELS_MIN)
    if low > OBSTACLE_PIXELS_MAX:
        return False
    target = random.randint(low, OBSTACLE_PIXELS_MAX)
    outside_quota = min(target - bar_pixels, random.randint(OBSTACLE_OUTSIDE_QUOTA_MIN, OBSTACLE_OUTSIDE_QUOTA_MAX))

    while int(obstacle_mask.sum()) < target:
        remaining = target - int(obstacle_mask.sum())
        zone = "outside" if outside_quota > 0 else random.choice(OBSTACLE_ZONE_OPTIONS)
        quota_limit = outside_quota if outside_quota > 0 else OBSTACLE_CLUSTER_SIZE_MAX
        size = random.randint(OBSTACLE_CLUSTER_SIZE_MIN, min(OBSTACLE_CLUSTER_SIZE_MAX, remaining, quota_limit))
        placed = False
        for _attempt in range(OBSTACLE_ATTEMPTS):
            seed = (random.randrange(H), random.randrange(W))
            coords = _random_cluster(seed, size)
            if _can_place_obstacle(grid, obstacle_mask, contour_mask, coords, bg, paths, zone):
                for r, c in coords:
                    grid[r, c] = obstacle
                    obstacle_mask[r, c] = True
                if zone == "outside":
                    outside_quota -= size
                placed = True
                break
        if not placed:
            return False
    return True


def _gen_long_component_ids(fill_lab: np.ndarray, n: int) -> set[int]:
    ids: set[int] = set()
    for comp_id in range(1, n + 1):
        coords = np.argwhere(fill_lab == comp_id)
        if coords.size == 0:
            continue
        if int(coords[:, 0].max()) - int(coords[:, 0].min()) + 1 >= LONG_COMPONENT_MIN_SPAN:
            ids.add(int(comp_id))
    return ids


def _fillable_mask(grid: np.ndarray, bg: int, contour: int) -> tuple[np.ndarray, int, set[int], int] | None:
    fill_lab, n_fill = label_mask(grid == contour, connectivity=8)
    long_ids = _gen_long_component_ids(fill_lab, n_fill)
    if len(long_ids) not in LONG_COMPONENT_COUNT_OPTIONS:
        return None

    bg_lab, n_bg = label_mask(grid == bg, connectivity=4)
    if n_bg < BG_REGIONS_MIN or n_bg > BG_REGIONS_MAX:
        return None
    sizes = {comp_id: int((bg_lab == comp_id).sum()) for comp_id in range(1, n_bg + 1)}
    ranked = sorted(sizes, key=sizes.get, reverse=True)
    if len(ranked) < 3 or sizes[ranked[1]] <= sizes[ranked[2]]:
        return None
    exterior = {int(ranked[0]), int(ranked[1])}
    interior_count = n_bg - EXTERIOR_REGION_COUNT
    if not INTERIOR_REGIONS_MIN <= interior_count <= INTERIOR_REGIONS_MAX:
        return None
    if any(sizes[comp_id] > INTERIOR_REGION_AREA_MAX for comp_id in range(1, n_bg + 1) if comp_id not in exterior):
        return None

    fillable = np.zeros_like(grid, dtype=bool)
    for comp_id in range(1, n_bg + 1):
        if comp_id in exterior:
            continue
        coords = np.argwhere(bg_lab == comp_id)
        touches = False
        for r, c in coords:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(fill_lab[rr, cc]) in long_ids:
                    touches = True
                    break
            if touches:
                break
        if not touches:
            return None
        fillable[bg_lab == comp_id] = True
    return fillable, n_bg, exterior, len(long_ids)


def _validate(inp: np.ndarray, out: np.ndarray, paths: list[list[int]], bg: int, contour: int, obstacle: int, n_bg: int, long_count: int) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    values = set(int(v) for v in np.unique(inp))
    if values != {bg, contour, obstacle}:
        return False
    contour_count = int((inp == contour).sum())
    obstacle_count = int((inp == obstacle).sum())
    if not (CONTOUR_PIXELS_MIN <= contour_count <= CONTOUR_PIXELS_MAX and OBSTACLE_PIXELS_MIN <= obstacle_count <= OBSTACLE_PIXELS_MAX):
        return False
    if contour_count <= obstacle_count:
        return False
    interior_count = n_bg - EXTERIOR_REGION_COUNT
    if not (BG_REGIONS_MIN <= n_bg <= BG_REGIONS_MAX and INTERIOR_REGIONS_MIN <= interior_count <= INTERIOR_REGIONS_MAX and long_count in LONG_COMPONENT_COUNT_OPTIONS):
        return False

    contour_lab, n_contour = label_mask(inp == contour, connectivity=8)
    short = 0
    for comp_id in range(1, n_contour + 1):
        coords = np.argwhere(contour_lab == comp_id)
        span = int(coords[:, 0].max()) - int(coords[:, 0].min()) + 1
        area = int(coords.shape[0])
        if span >= LONG_COMPONENT_MIN_SPAN:
            continue
        short += 1
        if area < SHORT_FRAGMENT_SIZE_MIN or area > SHORT_FRAGMENT_SIZE_MAX:
            return False
    if not SHORT_FRAGMENT_COUNT_MIN <= short <= SHORT_FRAGMENT_COUNT_MAX:
        return False

    obs_lab, n_obs = label_mask(inp == obstacle, connectivity=4)
    if n_obs == 0:
        return False
    if any(int((obs_lab == comp_id).sum()) > OBSTACLE_CLUSTER_SIZE_MAX for comp_id in range(1, n_obs + 1)):
        return False
    inside_obs = any(inp[r, c] == obstacle and _inside_channel(paths, r, c) for r in range(H) for c in range(W))
    outside_obs = any(
        inp[r, c] == obstacle and _outside_enclosure(paths, r, c, OUTSIDE_MARGIN) for r in range(H) for c in range(W)
    )
    return bool(inside_obs and outside_obs)


def _draw_long_contours(bg: int, contour: int, paths: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Render the primary path columns and return the grid plus long-contour mask."""
    grid = np.full(GRID_SHAPE, bg, dtype=int)
    long_mask = np.zeros(GRID_SHAPE, dtype=bool)
    for path in paths:
        for r, c in enumerate(path):
            grid[r, c] = contour
            long_mask[r, c] = True
    return grid, long_mask


def _finish_contours(grid: np.ndarray, long_mask: np.ndarray, paths: list[list[int]], contour: int) -> bool:
    """Add short distractors, then pad long contours to the sampled pixel target."""
    contour_mask = grid == contour
    frag_count = random.randint(SHORT_FRAGMENT_COUNT_MIN, SHORT_FRAGMENT_COUNT_MAX)
    if not _place_short_fragments(grid, contour_mask, paths, contour, frag_count):
        return False

    current_contour = int((grid == contour).sum())
    if current_contour > CONTOUR_PIXELS_MAX:
        return False
    target_contour = random.randint(max(CONTOUR_PIXELS_MIN, current_contour), CONTOUR_PIXELS_MAX)
    return _add_contour_extras(grid, long_mask, paths, contour, target_contour - current_contour)


def _place_obstacles(
    grid: np.ndarray, contour_mask: np.ndarray, paths: list[list[int]], bg: int, obstacle: int
) -> bool:
    """Place channel bars first, then scatter isolated obstacle clusters."""
    obstacle_mask = np.zeros(GRID_SHAPE, dtype=bool)
    if not _add_obstacle_bars(grid, paths, obstacle):
        return False
    obstacle_mask = grid == obstacle
    return _scatter_obstacles(grid, obstacle_mask, contour_mask, paths, bg, obstacle)


def _make_input_output(grid: np.ndarray, bg: int, contour: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Build the output-first pair and carry validation metadata."""
    fill_info = _fillable_mask(grid, bg, contour)
    if fill_info is None:
        return None
    fillable, n_bg, _exterior, long_count = fill_info
    out = grid.copy()
    out[fillable] = contour
    inp = out.copy()
    inp[fillable] = bg
    return inp, out, n_bg, long_count


def _gen_attempt() -> np.ndarray | None:
    bg, contour, obstacle = _color_choices()
    path_count = random.choice(PATH_COUNT_OPTIONS)
    paths = _path_columns(path_count)
    if paths is None:
        return None

    grid, long_mask = _draw_long_contours(bg, contour, paths)
    if not _finish_contours(grid, long_mask, paths, contour):
        return None

    contour_mask = grid == contour
    if not _place_obstacles(grid, contour_mask, paths, bg, obstacle):
        return None

    pair = _make_input_output(grid, bg, contour)
    if pair is None:
        return None
    inp, out, n_bg, long_count = pair

    if not _validate(inp, out, paths, bg, contour, obstacle, n_bg, long_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 753ea09b input")
