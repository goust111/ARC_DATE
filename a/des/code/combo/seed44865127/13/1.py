import random
from collections import deque

import numpy as np


MARKER_COLOR = 5
GEN_RETRIES = 400
GRID_SIZE_RANGE = (20, 28)
RECT_SIZE_RANGE = (6, 8)
SAFE_MARGIN = 6
EDGE_BUFFER = 2
PLACEMENT_ATTEMPTS = 240
DIRECTIONS = ("up", "right", "down", "left")


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _direction_from_marker(box: np.ndarray) -> str | None:
    marker_pts = np.argwhere(box == MARKER_COLOR)
    if marker_pts.shape[0] != 1:
        return None
    mr, mc = map(int, marker_pts[0])
    dists = {
        "up": mr,
        "right": box.shape[1] - 1 - mc,
        "down": box.shape[0] - 1 - mr,
        "left": mc,
    }
    best = min(dists.values())
    matches = [name for name, dist in dists.items() if dist == best]
    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each rectangle's interior motif onto the nearest grid edge.

    The rectangle's grey marker decides the direction:
    top/right/bottom/left. Vertical projections collapse each original column
    to an edge-aligned bar, while horizontal projections do the same per row.
    The rectangle fill and the marker are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)
    out = np.full((h, w), bg, dtype=int)

    regions = []
    for coords in _components(g != bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(box == bg):
            continue
        direction = _direction_from_marker(box)
        if direction is None:
            continue

        non_marker = box[box != MARKER_COLOR]
        if non_marker.size == 0:
            continue
        values, counts = np.unique(non_marker, return_counts=True)
        base = int(values[np.argmax(counts)])
        pattern_mask = (box != base) & (box != MARKER_COLOR)
        if not pattern_mask.any():
            continue

        regions.append((r0, c0, box.copy(), base, direction))

    regions.sort(key=lambda item: (item[0], item[1]))

    for r0, c0, box, base, direction in regions:
        rh, rw = box.shape
        if direction in ("up", "down"):
            for dc in range(rw):
                colors = [
                    int(box[rr, dc])
                    for rr in range(rh)
                    if int(box[rr, dc]) not in (base, MARKER_COLOR)
                ]
                if not colors:
                    continue
                if direction == "up":
                    start_r = 0
                else:
                    start_r = h - len(colors)
                for offset, color in enumerate(colors):
                    out[start_r + offset, c0 + dc] = color
        else:
            for dr in range(rh):
                colors = [
                    int(box[dr, cc])
                    for cc in range(rw)
                    if int(box[dr, cc]) not in (base, MARKER_COLOR)
                ]
                if not colors:
                    continue
                if direction == "left":
                    start_c = 0
                else:
                    start_c = w - len(colors)
                for offset, color in enumerate(colors):
                    out[r0 + dr, start_c + offset] = color

    return out


def _choose_directions(rng: random.Random) -> list[str]:
    count = rng.randint(2, 4)
    if count == 2:
        dirs = [rng.choice(("up", "down")), rng.choice(("left", "right"))]
    elif count == 3:
        omitted = rng.choice(DIRECTIONS)
        dirs = [direction for direction in DIRECTIONS if direction != omitted]
    else:
        dirs = list(DIRECTIONS)
    rng.shuffle(dirs)
    return dirs


def _grow_pattern(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    area = height * width
    if area < 4:
        return None

    max_cells = min(area - 1, max(3, (2 * area) // 3))
    if max_cells < 3:
        return None
    target = rng.randint(3, max_cells)

    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target:
        return None

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_region_box(
    height: int,
    width: int,
    base_color: int,
    pattern_color: int,
    direction: str,
    rng: random.Random,
) -> np.ndarray | None:
    box = np.full((height, width), base_color, dtype=int)
    core = box[2 : height - 2, 2 : width - 2]
    pattern = _grow_pattern(core.shape[0], core.shape[1], rng)
    if pattern is None:
        return None
    core[pattern] = pattern_color

    if direction == "up":
        mr = 1
        mc = rng.randint(2, width - 3)
    elif direction == "down":
        mr = height - 2
        mc = rng.randint(2, width - 3)
    elif direction == "left":
        mr = rng.randint(2, height - 3)
        mc = 1
    else:
        mr = rng.randint(2, height - 3)
        mc = width - 2
    box[mr, mc] = MARKER_COLOR
    return box


def _rect_conflicts(
    bbox: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return True
    return False


def _sample_position(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
    direction: str,
    placed: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    r_min = EDGE_BUFFER
    r_max = height - EDGE_BUFFER - rect_h
    c_min = EDGE_BUFFER
    c_max = width - EDGE_BUFFER - rect_w

    if direction in ("left", "right"):
        r_min = max(r_min, SAFE_MARGIN)
        r_max = min(r_max, height - SAFE_MARGIN - rect_h)
    if direction in ("up", "down"):
        c_min = max(c_min, SAFE_MARGIN)
        c_max = min(c_max, width - SAFE_MARGIN - rect_w)

    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if not _rect_conflicts(bbox, placed):
            return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)
    directions = _choose_directions(rng)

    bg = rng.choice([color for color in range(10) if color != MARKER_COLOR])
    palette = [color for color in range(10) if color not in {bg, MARKER_COLOR}]
    rng.shuffle(palette)
    base_colors = palette[: len(directions)]
    pattern_colors = palette[len(directions) : 2 * len(directions)]

    grid = np.full((height, width), bg, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    pattern_pixels = 0

    for direction, base_color, pattern_color in zip(directions, base_colors, pattern_colors):
        placed_one = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rect_h = rng.randint(*RECT_SIZE_RANGE)
            rect_w = rng.randint(*RECT_SIZE_RANGE)
            box = _make_region_box(rect_h, rect_w, base_color, pattern_color, direction, rng)
            if box is None:
                continue
            pos = _sample_position(height, width, rect_h, rect_w, direction, placed, rng)
            if pos is None:
                continue
            r0, c0 = pos
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = box
            placed.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
            pattern_pixels += int(np.count_nonzero(box == pattern_color))
            placed_one = True
            break
        if not placed_one:
            return None

    if _bg_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if int(np.count_nonzero(out != bg)) != pattern_pixels:
        return None
    if MARKER_COLOR in set(map(int, out.ravel())):
        return None
    if any(color in set(map(int, out.ravel())) for color in base_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
