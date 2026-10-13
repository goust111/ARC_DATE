import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, find_components, get_bg
import random


def _unique_largest_count_color(g: np.ndarray, colors: list[int]) -> int:
    counts = [(int(np.sum(g == color)), int(color)) for color in colors]
    max_count = max(count for count, _color in counts)
    winners = [color for count, color in counts if count == max_count]
    assert len(winners) == 1, "expected one most numerous non-background obstacle color"
    return winners[0]


def _marker_bbox(g: np.ndarray, marker_colors: list[int]) -> tuple[int, int, int, int]:
    mask = np.isin(g, marker_colors)
    candidates: list[tuple[int, int, int, int]] = []
    for coords in components_mask(mask, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        if r0 == 0:
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected one top-edge marker component"
    return candidates[0]


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _can_slide_right(base: np.ndarray, obstacle: int, bg: int, cols: list[int], marker_bottom_r: int, height: int) -> bool:
    H, W = base.shape
    assert cols
    if cols[-1] + 1 >= W:
        return False

    for floor_r in range(marker_bottom_r + 1, H):
        target_rows = np.arange(floor_r - height + 1, floor_r + 1)
        if int(target_rows[0]) < 0:
            continue

        shifted_cols = [c + 1 for c in cols]
        target_is_clear = all(np.all(base[target_rows, c] == bg) for c in shifted_cols)
        if not target_is_clear:
            continue

        for c in cols:
            if int(base[floor_r, c]) != obstacle:
                continue
            has_horizontal_neighbor = (
                (c > 0 and int(base[floor_r, c - 1]) == obstacle)
                or (c + 1 < W and int(base[floor_r, c + 1]) == obstacle)
            )
            if has_horizontal_neighbor:
                return True

    return False


def _clear_bottom_edge_gap(base: np.ndarray, out: np.ndarray, obstacle: int, bg: int, c: int, width: int) -> None:
    H, W = base.shape
    if int(base[H - 1, c]) != obstacle:
        return

    c1 = c
    while c1 < W and int(base[H - 1, c1]) == obstacle:
        c1 += 1

    if c1 - c >= width:
        out[H - 1, c + 1 : c + width] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the columns of the top marker downward, stopping blue-ended columns on grey obstacles and letting maroon-ended columns pass to the bottom.

    Input structure
    - The background is the dominant color.
    - The most numerous non-background color forms stationary grey obstacle pixels.
    - The other two colors form a single 3x3 top-edge marker whose top two rows
      are maroon and whose bottom row contains maroon or blue control pixels.

    Rule
    - Erase the top marker.
    - Treat each marker column as a vertical drop retaining its three colors.
    - A column whose bottom control pixel is blue stops immediately above the
      first grey pixel below that column, or bottom-aligns if none exists.
    - Contiguous maroon-ended columns pass through obstacles and bottom-align;
      if such a run has a clear down-right slide from a grey ledge, shift the
      run one column right before bottom-aligning.
    - When a maroon-ended drop lands through a grey bar on the bottom edge, clear
      the rest of the marker-width gap in that bottom bar.

    Key insight
    - The bottom row of the top marker is a control strip: blue cells mean
      "stick to the first obstacle," while maroon cells mean "fall through."
    """
    g = np.asarray(grid, dtype=int)
    H = g.shape[0]
    bg = int(get_bg(g))

    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 3, "expected obstacle color plus two marker colors"
    obstacle = _unique_largest_count_color(g, non_bg)
    marker_colors = [color for color in non_bg if color != obstacle]
    assert len(marker_colors) == 2, "expected exactly two marker colors"

    r0, c0, r1, c1 = _marker_bbox(g, marker_colors)
    marker = g[r0 : r1 + 1, c0 : c1 + 1]
    marker_h, marker_w = marker.shape
    assert (marker_h, marker_w) == (3, 3), "expected a 3x3 marker"
    assert np.all(np.isin(marker, marker_colors)), "marker bbox must contain only marker colors"

    pass_color = int(marker[0, 0])
    assert np.all(marker[:2, :] == pass_color), "top two marker rows must be the pass-through color"
    stop_colors = [color for color in marker_colors if color != pass_color]
    assert len(stop_colors) == 1, "expected one stop color"
    stop_color = stop_colors[0]

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    base = out.copy()

    pass_columns: list[int] = []
    for j in range(marker_w):
        c = c0 + j
        stack = marker[:, j]
        bottom_control = int(stack[-1])

        if bottom_control == stop_color:
            obstacles_below = np.flatnonzero(base[r1 + 1 :, c] == obstacle)
            if len(obstacles_below):
                floor_r = int(r1 + 1 + obstacles_below[0])
                top_r = floor_r - marker_h
            else:
                top_r = H - marker_h
            assert top_r >= 0, "stopped column must fit above its obstacle"
            out[top_r : top_r + marker_h, c] = stack
        else:
            assert bottom_control == pass_color, "bottom control must be one of the marker colors"
            pass_columns.append(j)

    for start_j, end_j in _runs(pass_columns):
        source_cols = list(range(c0 + start_j, c0 + end_j + 1))
        shift = 1 if _can_slide_right(base, obstacle, bg, source_cols, r1, marker_h) else 0
        top_r = H - marker_h

        for j in range(start_j, end_j + 1):
            c = c0 + j + shift
            _clear_bottom_edge_gap(base, out, obstacle, bg, c, marker_w)
            out[top_r : top_r + marker_h, c] = marker[:, j]

    return out

# --------------- generator (att1) ---------------

GEN_BG, GEN_OBSTACLE, GEN_PASS, GEN_STOP = 7, 5, 9, 1
GRID_SIZE_RANGE = (14, 16)
MARKER_SIZE = 3
STOP_COUNT_RANGE = (1, 3)
OBSTACLE_COMPONENT_RANGE = (7, 9)
OBSTACLE_PIXEL_RANGE = (20, 31)
COMPONENT_SIZE_RANGE = (1, 11)
MIN_STOP_OBSTACLE_ROW = 6
DROP_ZONE_HEIGHT = 3
COMPONENT_PLACEMENT_ATTEMPTS, GENERATION_ATTEMPTS = 100, 300


def _init_scene(H: int, W: int) -> tuple[np.ndarray, int, list[int], set[tuple[int, int]]]:
    grid = np.full((H, W), GEN_BG, dtype=int)
    c0 = random.randint(0, W - MARKER_SIZE)
    num_stops = random.randint(*STOP_COUNT_RANGE)
    positions = list(range(MARKER_SIZE))
    random.shuffle(positions)
    stop_positions = sorted(positions[:num_stops])
    marker = np.full((MARKER_SIZE, MARKER_SIZE), GEN_PASS, dtype=int)
    for j in stop_positions:
        marker[MARKER_SIZE - 1, j] = GEN_STOP
    grid[0:MARKER_SIZE, c0:c0 + MARKER_SIZE] = marker
    obstacle_pixels: set[tuple[int, int]] = set()
    for j in stop_positions:
        col = c0 + j
        first_obs_row = random.randint(MIN_STOP_OBSTACLE_ROW, H - 1)
        grid[first_obs_row, col] = GEN_OBSTACLE
        obstacle_pixels.add((first_obs_row, col))
    return grid, c0, stop_positions, obstacle_pixels


def _forbidden_cells(H: int, c0: int, stop_positions: list[int], obstacle_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    forbidden = {(r, c) for r in range(MARKER_SIZE) for c in range(c0, c0 + MARKER_SIZE)}
    for j in stop_positions:
        col = c0 + j
        for r in range(MARKER_SIZE, H):
            if (r, col) not in obstacle_pixels:
                continue
            for dr in range(1, DROP_ZONE_HEIGHT + 1):
                if r - dr >= 0:
                    forbidden.add((r - dr, col))
            break
    return forbidden


def _sample_component_sizes(total_pixels: int, component_count: int) -> list[int] | None:
    component_sizes = []
    left = total_pixels
    min_size, max_size = COMPONENT_SIZE_RANGE
    for i in range(component_count):
        remaining = component_count - i - 1
        if remaining == 0:
            size = left
        else:
            lo = max(min_size, left - max_size * remaining)
            hi = min(max_size, left - min_size * remaining)
            if lo > hi:
                return None
            size = random.randint(lo, hi)
        left -= size
        if size < min_size or size > max_size:
            return None
        component_sizes.append(size)
    random.shuffle(component_sizes)
    return component_sizes


def _grow_component(grid: np.ndarray, comp_size: int, forbidden: set[tuple[int, int]],
                    obstacle_pixels: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
    if (sr, sc) in forbidden or (sr, sc) in obstacle_pixels or grid[sr, sc] != GEN_BG:
        return None

    comp = {(sr, sc)}
    frontier = [(sr, sc)]
    while len(comp) < comp_size and frontier:
        random.shuffle(frontier)
        expanded = False
        for fr, fc in frontier:
            nbrs = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
            random.shuffle(nbrs)
            for nr, nc in nbrs:
                if (0 <= nr < H and 0 <= nc < W
                        and (nr, nc) not in forbidden
                        and (nr, nc) not in obstacle_pixels
                        and (nr, nc) not in comp
                        and grid[nr, nc] == GEN_BG):
                    comp.add((nr, nc))
                    frontier.append((nr, nc))
                    expanded = True
                    break
            if expanded:
                break
        if not expanded:
            break
    return comp if len(comp) == comp_size else None


def _place_obstacle_components(grid: np.ndarray, component_sizes: list[int], forbidden: set[tuple[int, int]],
                               obstacle_pixels: set[tuple[int, int]]) -> bool:
    for comp_size in component_sizes:
        placed = False
        for _ in range(COMPONENT_PLACEMENT_ATTEMPTS):
            comp = _grow_component(grid, comp_size, forbidden, obstacle_pixels)
            if comp is None:
                continue
            if any((pr + dr, pc + dc) in obstacle_pixels
                   for pr, pc in comp for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                continue
            for pr, pc in comp:
                grid[pr, pc] = GEN_OBSTACLE
                obstacle_pixels.add((pr, pc))
            placed = True
            break
        if not placed:
            return False
    return True


def _scatter_obstacles(grid: np.ndarray, c0: int, stop_positions: list[int],
                       obstacle_pixels: set[tuple[int, int]]) -> bool:
    H, _W = grid.shape
    num_target_components = random.randint(*OBSTACLE_COMPONENT_RANGE)
    total_target_pixels = random.randint(*OBSTACLE_PIXEL_RANGE)
    remaining_pixels = max(0, total_target_pixels - len(obstacle_pixels))
    remaining_components = max(1, num_target_components - len(stop_positions))
    if remaining_pixels <= 0 or remaining_components <= 0:
        return True

    component_sizes = _sample_component_sizes(remaining_pixels, remaining_components)
    if component_sizes is None:
        return False
    forbidden = _forbidden_cells(H, c0, stop_positions, obstacle_pixels)
    return _place_obstacle_components(grid, component_sizes, forbidden, obstacle_pixels)


def _has_clear_stop_drop_zones(grid: np.ndarray, c0: int, stop_positions: list[int]) -> bool:
    H, _W = grid.shape
    for j in stop_positions:
        col = c0 + j
        first_obs = next((r for r in range(MARKER_SIZE, H) if grid[r, col] == GEN_OBSTACLE), None)
        if first_obs is None:
            return False
        for dr in range(1, DROP_ZONE_HEIGHT + 1):
            if first_obs - dr < 0 or grid[first_obs - dr, col] != GEN_BG:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    grid, c0, stop_positions, obstacle_pixels = _init_scene(H, W)

    if not _scatter_obstacles(grid, c0, stop_positions, obstacle_pixels):
        return None
    if not _has_clear_stop_drop_zones(grid, c0, stop_positions):
        return None
    obs_comps = [c for c in find_components(grid, bg=GEN_BG, connectivity=4, monochromatic=True)
                 if c['color'] == GEN_OBSTACLE]
    if not (OBSTACLE_COMPONENT_RANGE[0] <= len(obs_comps) <= OBSTACLE_COMPONENT_RANGE[1]):
        return None

    # Run solver to verify no assertion failures
    try:
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
