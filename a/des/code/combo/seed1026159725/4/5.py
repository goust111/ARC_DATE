import random
from collections import Counter, deque

import numpy as np


LAYER_COLORS = (4, 7, 2, 9, 6)
GEN_RETRIES = 300
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
UNIQUE_DEPTH_OPTIONS = (3, 4, 5)
EXTRA_OBJECT_OPTIONS = (0, 1, 2)
BACKGROUND_CHOICES = (0, 1, 3, 8)
INPUT_OBJECT_COLOR = 5
SIDE_NAMES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _bbox_depth(bbox: tuple[int, int, int, int], height: int, width: int) -> int:
    r0, c0, r1, c1 = bbox
    return min(r0, c0, height - 1 - r1, width - 1 - c1)


def _layer_palette(bg: int, input_color: int) -> tuple[int, ...]:
    preferred = list(LAYER_COLORS)
    fallback = [c for c in range(10) if c not in preferred]
    palette = [c for c in preferred + fallback if c not in {bg, input_color}]
    return tuple(palette)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome object by its bbox distance to the nearest border.

    The shallowest layer becomes the lightest color; deeper layers use later
    colors in a fixed visible sequence. Objects with the same depth share a
    color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _background_color(g)

    fg_colors = {int(v) for v in np.unique(g) if int(v) != bg}
    assert len(fg_colors) == 1, "expected one foreground color for all objects"

    comps = _find_components(g, bg)
    assert len(comps) >= 3, "expected at least three foreground objects"

    depths = sorted({_bbox_depth(comp["bbox"], h, w) for comp in comps})
    palette = _layer_palette(bg, next(iter(fg_colors)))
    assert len(depths) <= len(palette), "not enough output colors for distinct depths"
    color_for_depth = {depth: palette[i] for i, depth in enumerate(depths)}

    out = g.copy()
    for comp in comps:
        depth = _bbox_depth(comp["bbox"], h, w)
        new_color = color_for_depth[depth]
        for r, c in comp["coords"]:
            out[r, c] = new_color
    return out


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    horizontal_first = bool(random.getrandbits(1))
    while (r, c) != (tr, tc):
        move_h = c != tc and (horizontal_first or r == tr)
        if move_h:
            c += 1 if tc > c else -1
        else:
            r += 1 if tr > r else -1
        path.append((r, c))
    return path


def _random_shape(height: int, width: int) -> set[tuple[int, int]]:
    assert height >= 2 and width >= 2
    hub = (random.randrange(height), random.randrange(width))
    anchors = {
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    }

    cells: set[tuple[int, int]] = set()
    for anchor in anchors | {hub}:
        cells.update(_path_between(hub, anchor))

    max_area = height * width
    min_target = max(len(cells), min(max_area, 5))
    density_cap = max(min_target, int(round(max_area * random.uniform(0.45, 0.9))))
    target = random.randint(min_target, min(max_area, density_cap))

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _candidate_bbox(side: str, depth: int, height: int, width: int) -> tuple[int, int, int, int] | None:
    max_h = height - 2 * depth
    max_w = width - 2 * depth
    if max_h < 2 or max_w < 2:
        return None

    box_h = random.randint(2, min(7, max_h))
    box_w = random.randint(2, min(7, max_w))

    if side == "top":
        r0 = depth
        c0 = random.randint(depth, width - depth - box_w)
    elif side == "bottom":
        r0 = height - depth - box_h
        c0 = random.randint(depth, width - depth - box_w)
    elif side == "left":
        c0 = depth
        r0 = random.randint(depth, height - depth - box_h)
    else:
        c0 = width - depth - box_w
        r0 = random.randint(depth, height - depth - box_h)
    return r0, c0, box_h, box_w


def _is_clear(occupied: np.ndarray, coords: set[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if occupied[r, c]:
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    depth: int,
    color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(80):
        side = random.choice(SIDE_NAMES)
        bbox = _candidate_bbox(side, depth, h, w)
        if bbox is None:
            continue
        r0, c0, box_h, box_w = bbox
        local_shape = _random_shape(box_h, box_w)
        global_coords = {(r0 + r, c0 + c) for r, c in local_shape}
        if not _is_clear(occupied, global_coords):
            continue

        rows = [r for r, _ in global_coords]
        cols = [c for _, c in global_coords]
        actual_bbox = (min(rows), min(cols), max(rows), max(cols))
        if _bbox_depth(actual_bbox, h, w) != depth:
            continue

        for r, c in global_coords:
            grid[r, c] = color
            occupied[r, c] = True
        return True
    return False


def _sample_depths(height: int, width: int) -> list[int]:
    max_depth = min((height - 2) // 2, (width - 2) // 2)
    depth_count = min(random.choice(UNIQUE_DEPTH_OPTIONS), max_depth + 1)
    depths = sorted(random.sample(range(max_depth + 1), depth_count))
    extra = random.choice(EXTRA_OBJECT_OPTIONS)
    duplicated = [random.choice(depths) for _ in range(extra)]
    all_depths = sorted(depths + duplicated, reverse=True)
    return all_depths


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.choice(BACKGROUND_CHOICES)
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    depths = _sample_depths(height, width)
    if len(set(depths)) < 3:
        return None

    for depth in depths:
        if not _place_object(grid, occupied, depth, INPUT_OBJECT_COLOR):
            return None

    comps = _find_components(grid, bg)
    if len(comps) != len(depths):
        return None
    if any(int(comp["color"]) != INPUT_OBJECT_COLOR for comp in comps):
        return None

    actual_depths = sorted(_bbox_depth(comp["bbox"], height, width) for comp in comps)
    if actual_depths != sorted(depths):
        return None

    out = main(grid)
    used_output_colors = {int(v) for v in np.unique(out) if int(v) != bg}
    if len(used_output_colors) != len(set(depths)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
