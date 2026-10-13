import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR_BY_MARKER = {
    RED: (1, 0),
    GREEN: (-1, 0),
    YELLOW: (0, -1),
    ORANGE: (0, 1),
}

MARKER_COLORS = tuple(DIR_BY_MARKER)
FREE_COLORS = (BLACK, BLUE)

GRID_MIN = 18
GRID_MAX = 24
MIN_REGION_SPAN = 5
GEN_RETRY_LIMIT = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = np.zeros((h, w), dtype=bool)
            queue = deque([(r, c)])
            seen[r, c] = True
            comp[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(component_masks(mask)) == 1


def free_region_masks(grid: np.ndarray) -> list[np.ndarray]:
    return component_masks(np.isin(grid, FREE_COLORS))


def region_marker(grid: np.ndarray, region_mask: np.ndarray) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    markers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(region_mask):
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in DIR_BY_MARKER:
                markers[(int(nr), int(nc))] = int(grid[nr, nc])
    assert len(markers) == 1, f"expected one marker, got {markers}"
    return next(iter(markers.items()))


def settle_single(region_mask: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = region_mask.shape
    dr, dc = direction
    r, c = start
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w and region_mask[nr, nc]):
            return int(r), int(c)
        r, c = nr, nc


def settle_region(
    region_mask: np.ndarray,
    waters: list[tuple[int, int]],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 1:
        ordered = sorted(waters, key=lambda rc: rc[0], reverse=True)
    elif dr == -1:
        ordered = sorted(waters, key=lambda rc: rc[0])
    elif dc == 1:
        ordered = sorted(waters, key=lambda rc: rc[1], reverse=True)
    else:
        ordered = sorted(waters, key=lambda rc: rc[1])

    h, w = region_mask.shape
    occupied: set[tuple[int, int]] = set()
    settled = []
    for r, c in ordered:
        rr, cc = int(r), int(c)
        while True:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w and region_mask[nr, nc]):
                break
            if (nr, nc) in occupied:
                break
            rr, cc = nr, nc
        occupied.add((rr, cc))
        settled.append((rr, cc))
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each region's blue water under the gravity direction encoded by its boundary marker.

    Input structure:
    - Grey walls partition the canvas into 3-4 enclosed regions whose interiors use black background and blue water.
    - Each region has exactly one non-grey marker on its boundary wall: red, green, yellow, or orange.
    - Marker pixels stay embedded in the walls, so regions are the 4-connected free-space components made of black and blue cells.

    Rule:
    - Identify each enclosed region independently.
    - Read the unique adjacent boundary marker for that region: red means down, green means up, yellow means left, orange means right.
    - Within that region, slide every blue water cell straight along that direction through black cells until the next step would hit a wall or already-settled water, processing the water closest to the target wall first.
    - Restore vacated cells to black and keep all walls and markers unchanged.

    Key insight:
    - The marker color does not act globally; it defines a separate gravity field for the specific enclosed region touching that marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for region_mask in free_region_masks(g):
        (_, _), marker_color = region_marker(g, region_mask)
        direction = DIR_BY_MARKER[marker_color]
        water_coords = [tuple(map(int, rc)) for rc in np.argwhere(region_mask & (g == BLUE))]
        if not water_coords:
            continue
        out[region_mask & (g == BLUE)] = BLACK
        for r, c in settle_region(region_mask, water_coords, direction):
            out[r, c] = BLUE
    return out


def random_transform(grid: np.ndarray) -> np.ndarray:
    choice = random.randrange(8)
    if choice == 0:
        return grid.copy()
    if choice == 1:
        return np.rot90(grid, 1).copy()
    if choice == 2:
        return np.rot90(grid, 2).copy()
    if choice == 3:
        return np.rot90(grid, 3).copy()
    if choice == 4:
        return np.flipud(grid).copy()
    if choice == 5:
        return np.fliplr(grid).copy()
    if choice == 6:
        return np.flipud(np.rot90(grid, 1)).copy()
    return np.fliplr(np.rot90(grid, 1)).copy()


def build_base_layout() -> tuple[np.ndarray, int]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    n_regions = random.choice((3, 4))
    c_split = random.randint(1 + MIN_REGION_SPAN, w - MIN_REGION_SPAN - 2)
    grid[1:-1, c_split] = GREY

    if n_regions == 3:
        split_left = random.choice((True, False))
        h_split = random.randint(1 + MIN_REGION_SPAN, h - MIN_REGION_SPAN - 2)
        if split_left:
            grid[h_split, 1:c_split] = GREY
        else:
            grid[h_split, c_split + 1 : w - 1] = GREY
    else:
        h_left = random.randint(1 + MIN_REGION_SPAN, h - MIN_REGION_SPAN - 2)
        h_right = random.randint(1 + MIN_REGION_SPAN, h - MIN_REGION_SPAN - 2)
        grid[h_left, 1:c_split] = GREY
        grid[h_right, c_split + 1 : w - 1] = GREY

    return grid, n_regions


def dent_rectangles(grid: np.ndarray, target_regions: int) -> np.ndarray | None:
    base_regions = free_region_masks(grid)
    random.shuffle(base_regions)

    dentable = []
    for region in base_regions:
        coords = np.argwhere(region)
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1) >= 6 or (c1 - c0 + 1) >= 6:
            dentable.append((region, (r0, c0, r1, c1)))
    if not dentable:
        return None

    out = grid.copy()
    wanted = min(len(dentable), 1 + int(random.random() < 0.7))
    added = 0
    for _region, rect in dentable:
        trial = out.copy()
        if not add_dent(trial, rect):
            continue
        if len(free_region_masks(trial)) != target_regions:
            continue
        out = trial
        added += 1
        if added >= wanted:
            break

    return out if added >= 1 else None


def add_dent(grid: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options = []

    if height >= 6 and width >= 4:
        for side in ("top", "bottom"):
            max_len = height - 2
            for col in range(c0 + 1, c1):
                options.append(("vertical", side, col, max_len))
    if width >= 6 and height >= 4:
        for side in ("left", "right"):
            max_len = width - 2
            for row in range(r0 + 1, r1):
                options.append(("horizontal", side, row, max_len))
    if not options:
        return False

    orient, side, anchor, max_len = random.choice(options)
    length = random.randint(2, max_len - 1)
    if orient == "vertical":
        col = anchor
        if side == "top":
            grid[r0 : r0 + length, col] = GREY
        else:
            grid[r1 - length + 1 : r1 + 1, col] = GREY
    else:
        row = anchor
        if side == "left":
            grid[row, c0 : c0 + length] = GREY
        else:
            grid[row, c1 - length + 1 : c1 + 1] = GREY
    return True


def outer_marker_candidates(grid: np.ndarray, region_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = set()
    for r, c in np.argwhere(region_mask):
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) != GREY:
                continue
            if nr not in (0, h - 1) and nc not in (0, w - 1):
                continue
            if (nr in (0, h - 1)) and (nc in (0, w - 1)):
                continue
            candidates.add((int(nr), int(nc)))
    return sorted(candidates)


def line_move_candidates(
    region_mask: np.ndarray,
    direction: tuple[int, int],
) -> dict[int, list[tuple[int, int]]]:
    by_line: dict[int, list[tuple[int, int]]] = defaultdict(list)
    vertical = direction[0] != 0
    for r, c in np.argwhere(region_mask):
        start = (int(r), int(c))
        end = settle_single(region_mask, start, direction)
        if end == start:
            continue
        line_key = int(c) if vertical else int(r)
        by_line[line_key].append(start)
    return by_line


def waters_are_spaced(waters: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(waters):
        for r1, c1 in waters[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) <= 1:
                return False
    return True


def choose_waters(
    region_mask: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    by_line = line_move_candidates(region_mask, direction)
    valid_lines = [line for line, coords in by_line.items() if coords]
    if len(valid_lines) < 3:
        return None

    max_count = min(6, len(valid_lines))
    count_choices = list(range(3, max_count + 1))
    random.shuffle(count_choices)
    for target_count in count_choices:
        for _ in range(200):
            chosen_lines = random.sample(valid_lines, target_count)
            waters = [random.choice(by_line[line]) for line in chosen_lines]
            if not waters_are_spaced(waters):
                continue
            black_mask = region_mask.copy()
            for r, c in waters:
                black_mask[r, c] = False
            if not is_connected(black_mask):
                continue
            settled = settle_region(region_mask, waters, direction)
            if len(set(settled)) != len(waters):
                continue
            if all(start == end for start, end in zip(waters, settled)):
                continue
            return waters
    return None


def annotate_regions(grid: np.ndarray) -> np.ndarray | None:
    regions = free_region_masks(grid)
    if len(regions) not in (3, 4):
        return None

    marker_colors = random.sample(MARKER_COLORS, len(regions))
    out = grid.copy()
    for region_mask, marker_color in zip(regions, marker_colors):
        candidates = outer_marker_candidates(out, region_mask)
        if not candidates:
            return None
        mr, mc = random.choice(candidates)
        out[mr, mc] = marker_color

        waters = choose_waters(region_mask, DIR_BY_MARKER[marker_color])
        if waters is None:
            return None
        for r, c in waters:
            out[r, c] = BLUE
    return out


def analyze_regions(grid: np.ndarray) -> list[dict]:
    regions = free_region_masks(grid)
    info = []
    for region in regions:
        coords = np.argwhere(region)
        waters = [tuple(map(int, rc)) for rc in np.argwhere(region & (grid == BLUE))]
        (_, _), marker_color = region_marker(grid, region)
        black_mask = region & (grid == BLACK)
        info.append(
            {
                "mask": region,
                "bbox": bbox_coords(coords),
                "area": int(region.sum()),
                "waters": waters,
                "marker": marker_color,
                "direction": DIR_BY_MARKER[marker_color],
                "black_connected": is_connected(black_mask),
            }
        )
    return info


def validate_instance(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        infos = analyze_regions(grid)
    except AssertionError:
        return False

    if len(infos) not in (3, 4):
        return False
    if np.array_equal(grid, out):
        return False

    markers = [info["marker"] for info in infos]
    if len(set(markers)) != len(markers):
        return False

    moved_regions = 0
    irregular = False
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        irregular |= info["area"] < bbox_area
        if not info["black_connected"]:
            return False
        if not (3 <= len(info["waters"]) <= 6):
            return False
        vertical = info["direction"][0] != 0
        keys = [c if vertical else r for r, c in info["waters"]]
        if len(set(keys)) != len(keys):
            return False
        settled = set(settle_region(info["mask"], info["waters"], info["direction"]))
        if settled != set(info["waters"]):
            moved_regions += 1

    if not irregular:
        return False
    if moved_regions != len(infos):
        return False
    return True


def generate_attempt() -> np.ndarray | None:
    grid, target_regions = build_base_layout()
    grid = dent_rectangles(grid, target_regions)
    if grid is None:
        return None
    grid = annotate_regions(grid)
    if grid is None:
        return None
    grid = random_transform(grid)
    if not validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
