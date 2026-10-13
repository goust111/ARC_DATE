import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8

NORTH, EAST, SOUTH, WEST = range(4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OBSTACLE_COLORS = [1, 2, 3, 4, 6, 7, 9]

MAX_PRIMARY_TURNS = 12
GEN_RETRIES = 800


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_from_mask(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    queue = deque([start])
    seen[start] = True
    cells: list[tuple[int, int]] = []
    while queue:
        r, c = queue.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return cells


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_source(grid: np.ndarray) -> tuple[int, int]:
    teal = grid == TEAL
    seen = np.zeros_like(teal, dtype=bool)
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not teal[r, c] or seen[r, c]:
                continue
            cells = _component_from_mask(teal, (r, c), seen)
            if len(cells) != 4:
                continue
            r0, c0, r1, c1 = _bbox(cells)
            if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL):
                candidates.append((r0, c0))
    assert len(candidates) == 1, "expected exactly one solid 2x2 teal source"
    return candidates[0]


def _find_obstacles(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    fg = (grid != BLACK) & (grid != TEAL)
    seen = np.zeros_like(fg, dtype=bool)
    h, w = grid.shape
    obstacles: list[dict] = []
    cell_to_obstacle: dict[tuple[int, int], dict] = {}
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            cells = _component_from_mask(fg & (grid == color), (r, c), seen)
            r0, c0, r1, c1 = _bbox(cells)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            assert hh != ww, "obstacles must be non-square"
            assert len(cells) == hh * ww, "obstacles must be solid rectangles"
            obs = {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": hh,
                "width": ww,
            }
            obstacles.append(obs)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    cell_to_obstacle[(rr, cc)] = obs
    return obstacles, cell_to_obstacle


def _ray_starts(src_r0: int, src_c0: int) -> list[tuple[int, int, int]]:
    # Discrete 2x2 source convention: use the outward-adjacent cells aligned to
    # the source's top-left, top-right, bottom-right, and bottom-left boundary cells.
    return [
        (src_r0 - 1, src_c0, NORTH),
        (src_r0, src_c0 + 2, EAST),
        (src_r0 + 2, src_c0 + 1, SOUTH),
        (src_r0 + 1, src_c0 - 1, WEST),
    ]


def _turn_direction(direction: int, obstacle: dict) -> int:
    if obstacle["width"] > obstacle["height"]:
        return (direction + 1) % 4
    return (direction - 1) % 4


def _turn_launch(direction: int, obstacle: dict) -> tuple[int, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = obstacle["bbox"]
    turned = _turn_direction(direction, obstacle)
    if direction == NORTH:
        face_r = r1
        face_c = c0 if turned == WEST else c1
    elif direction == SOUTH:
        face_r = r0
        face_c = c1 if turned == EAST else c0
    elif direction == EAST:
        face_c = c0
        face_r = r0 if turned == NORTH else r1
    else:
        face_c = c1
        face_r = r1 if turned == SOUTH else r0
    dr, dc = DIRS[turned]
    gap = (face_r + dr, face_c + dc)
    launch = gap
    return turned, launch, gap


def _trace_primary_rays(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    src_r0, src_c0 = _find_source(out)
    _obstacles, obstacle_map = _find_obstacles(out)

    turn_events: list[dict] = []
    ray_details: list[dict] = []
    for start_r, start_c, direction in _ray_starts(src_r0, src_c0):
        path: list[tuple[int, int]] = []
        states: set[tuple[int, int, int]] = set()
        r, c, d = start_r, start_c, direction
        turns = 0
        while _in_bounds(out.shape, r, c):
            state = (r, c, d)
            if state in states:
                break
            states.add(state)

            if out[r, c] == BLACK:
                out[r, c] = TEAL
                path.append((r, c))

            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                break
            if out[nr, nc] == TEAL:
                break

            obstacle = obstacle_map.get((nr, nc))
            if obstacle is not None:
                d, (r, c), gap = _turn_launch(d, obstacle)
                turn_events.append({"gap": gap, "direction": d})
                turns += 1
                if turns > MAX_PRIMARY_TURNS:
                    break
                if not _in_bounds(out.shape, r, c) or out[r, c] != BLACK:
                    break
                continue

            r, c = nr, nc

        ray_details.append({"path": path, "turns": turns})

    return out, ray_details, turn_events


def _enclosed_black_regions(board: np.ndarray) -> list[list[tuple[int, int]]]:
    black = board == BLACK
    seen = np.zeros_like(black, dtype=bool)
    h, w = board.shape
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            cells = _component_from_mask(black, (r, c), seen)
            touches_edge = any(rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1 for rr, cc in cells)
            if not touches_edge:
                regions.append(cells)
    return regions


def _region_touching_rays(region: list[tuple[int, int]], ray_details: list[dict]) -> int:
    region_set = set(region)
    touch_count = 0
    for ray in ray_details:
        path = ray["path"]
        if not path:
            continue
        path_set = set(path)
        touched = False
        for r, c in region_set:
            if any((nr, nc) in path_set for nr, nc in _neighbors8(r, c)):
                touched = True
                break
        if touched:
            touch_count += 1
    return touch_count


def _candidate_gap_events(
    primary: np.ndarray,
    region: list[tuple[int, int]],
    turn_events: list[dict],
) -> list[dict]:
    region_set = set(region)
    candidates: list[dict] = []
    seen_keys: set[tuple[int, int, int]] = set()
    for event in turn_events:
        gr, gc = event["gap"]
        if not _in_bounds(primary.shape, gr, gc):
            continue
        if (gr, gc) not in region_set and not any((nr, nc) in region_set for nr, nc in _neighbors8(gr, gc)):
            continue
        dr, dc = DIRS[event["direction"]]
        while _in_bounds(primary.shape, gr, gc) and (gr, gc) in region_set:
            gr += dr
            gc += dc
        while _in_bounds(primary.shape, gr, gc) and primary[gr, gc] != BLACK:
            gr += dr
            gc += dc
        if not _in_bounds(primary.shape, gr, gc):
            continue
        key = (gr, gc, event["direction"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append({"gap": (gr, gc), "direction": event["direction"]})
    return candidates


def _emit_secondary_ray(board: np.ndarray, start: tuple[int, int], direction: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    painted: list[tuple[int, int]] = []
    if _in_bounds(board.shape, r, c) and board[r, c] == BLACK:
        board[r, c] = GREY
        painted.append((r, c))
        nr, nc = r + dr, c + dc
        r, c = nr, nc
    else:
        r, c = r + dr, c + dc
    while _in_bounds(board.shape, r, c) and board[r, c] == BLACK:
        board[r, c] = GREY
        painted.append((r, c))
        nr, nc = r + dr, c + dc
        if not _in_bounds(board.shape, nr, nc) or board[nr, nc] != BLACK:
            break
        r, c = nr, nc
    return painted


def _transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    primary, ray_details, turn_events = _trace_primary_rays(grid)
    out = primary.copy()

    regions = _enclosed_black_regions(primary)
    for region in regions:
        for r, c in region:
            out[r, c] = GREY

    secondary_segments: list[list[tuple[int, int]]] = []
    gap_counts: list[int] = []
    for region in regions:
        candidates = _candidate_gap_events(primary, region, turn_events)
        gap_counts.append(len(candidates))
        if len(candidates) != 1:
            continue
        segment = _emit_secondary_ray(out, candidates[0]["gap"], candidates[0]["direction"])
        if segment:
            secondary_segments.append(segment)

    info = {
        "input": np.array(grid, dtype=int, copy=True),
        "primary": primary,
        "output": out,
        "regions": regions,
        "ray_details": ray_details,
        "turn_events": turn_events,
        "gap_counts": gap_counts,
        "secondary_segments": secondary_segments,
        "fill_area": sum(len(region) for region in regions),
        "secondary_added": int(np.count_nonzero(out == GREY) - sum(len(region) for region in regions)),
        "region_ray_touch_counts": [_region_touching_rays(region, ray_details) for region in regions],
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace teal rays around rectangular obstacles, fill the enclosed black pocket, and fire a grey follow-up ray from its single turn gap.

    Input structure:
    - The grid has a black background, exactly one solid 2x2 teal source, and 2 to 4 solid non-square rectangular obstacles of distinct colors.
    - Obstacles are monochromatic blocks that stay away from the grid edge; their aspect ratio is the only turn code.
    - The source is interpreted in the standard `(row, col)` grid frame, and every emitted ray has width 1.

    Rule:
    - Emit four teal rays from the 2x2 source. Because the source has even size, use the four outward-adjacent cells aligned with the source's top-left, top-right, bottom-right, and bottom-left boundary cells.
    - A ray paints black cells teal until it would leave the grid, hit an existing teal cell, or hit an obstacle. On obstacle contact, a wide rectangle turns the ray clockwise, and a tall rectangle turns it counterclockwise.
    - After a turn, resume from the cell adjacent to the relevant end of the contacted obstacle face, so the obstacle itself becomes part of the enclosing boundary; that turn pivot also defines a potential one-pixel gap location.
    - Once all primary rays are drawn, find every 4-connected black region that does not touch the grid edge and fill it grey.
    - For each filled region, inspect the turn-generated gap candidates. If exactly one turn pivot belongs to or borders that region, follow that turn's outward direction to the first black gap cell and emit a grey secondary ray from there until the next non-black cell or the grid edge.

    Key insight:
    - The obstacle shape, not the obstacle color, encodes the turn, and the turn leaves a geometric gap candidate that only matters if the traced rays really create a closed black pocket.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _info = _transform(g)
    return out


def _place_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> bool:
    gh, gw = grid.shape
    if r0 < 1 or c0 < 1 or r0 + h >= gh or c0 + w >= gw:
        return False
    if np.any(grid[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] != BLACK):
        return False
    grid[r0 : r0 + h, c0 : c0 + w] = color
    return True


def _random_obstacle_shape(force_orientation: int | None = None) -> tuple[int, int]:
    if force_orientation is None:
        force_orientation = random.randint(0, 1)
    while True:
        if force_orientation == 0:
            shape = (random.randint(3, 6), random.randint(2, 3))
        else:
            shape = (random.randint(2, 3), random.randint(3, 6))
        if shape[0] != shape[1]:
            return shape


def _make_random_case() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)

    src_r0 = random.randint(5, h - 7)
    src_c0 = random.randint(5, w - 7)
    grid[src_r0 : src_r0 + 2, src_c0 : src_c0 + 2] = TEAL

    n_obstacles = random.randint(2, 4)
    orientations = [0, 1]
    while len(orientations) < n_obstacles:
        orientations.append(random.randint(0, 1))
    random.shuffle(orientations)
    colors = random.sample(OBSTACLE_COLORS, n_obstacles)

    for color, orientation in zip(colors, orientations, strict=True):
        placed = False
        for _ in range(40):
            oh, ow = _random_obstacle_shape(orientation)
            r0 = random.randint(1, h - oh - 2)
            c0 = random.randint(1, w - ow - 2)
            if _place_rect(grid, r0, c0, oh, ow, color):
                placed = True
                break
        if not placed:
            return None

    return grid


def _case_is_valid(grid: np.ndarray, target_gap: bool) -> bool:
    try:
        out, info = _transform(grid)
    except AssertionError:
        return False

    if np.array_equal(grid, out):
        return False
    if len(info["regions"]) != 1:
        return False
    if info["fill_area"] < 6 or info["fill_area"] > 90:
        return False
    if info["region_ray_touch_counts"][0] < 2:
        return False

    nonempty_rays = sum(1 for ray in info["ray_details"] if ray["path"])
    turning_rays = sum(1 for ray in info["ray_details"] if ray["turns"] > 0)
    if nonempty_rays < 2 or turning_rays < 1:
        return False
    if any(ray["turns"] > 2 for ray in info["ray_details"] if ray["path"]):
        return False

    gap_regions = sum(count == 1 for count in info["gap_counts"])
    secondary_len = info["secondary_added"]
    if target_gap:
        if gap_regions != 1 or secondary_len < 1:
            return False
    else:
        if gap_regions != 0 or secondary_len != 0:
            return False

    return True


def generate_input() -> np.ndarray:
    target_gap = random.random() < 0.72
    for _ in range(GEN_RETRIES):
        grid = _make_random_case()
        if grid is None:
            continue
        if _case_is_valid(grid, target_gap=target_gap):
            return grid
    for fallback_gap in (True, False):
        for _ in range(GEN_RETRIES):
            grid = _make_random_case()
            if grid is None:
                continue
            if _case_is_valid(grid, target_gap=fallback_gap):
                return grid
    raise RuntimeError("failed to generate a valid ray-turn-fill ARC input")
