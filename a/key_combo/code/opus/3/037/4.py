from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


Coord = tuple[int, int]


OBJECT_COLORS = [Color.BLUE, Color.RED, Color.PINK, Color.ORANGE, Color.MAROON]
WALL_COLOR = Color.TEAL
PATH_COLOR = Color.GREEN
MOVED_COLOR = Color.YELLOW
SEPARATOR_COLOR = Color.GREY
BACKGROUND = Color.BLACK

PANEL_HEIGHT_RANGE = (6, 11)
GRID_WIDTH_RANGE = (18, 29)
OBJECT_COUNT_RANGE = (3, 5)
CONFLICT_OBJECTS_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400
OBJECT_PLACEMENT_ATTEMPTS = 120

SEARCH_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
WALL_LENGTHS = [3, 4, 5]
WALL_ORIENTATIONS = ["h", "v"]

BASE_SHAPES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | None = None,
) -> list[dict]:
    h, w = grid.shape
    allowed = None if colors is None else set(map(int, colors))
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue
            if allowed is not None and color not in allowed:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in SEARCH_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )

    return comps


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    return coords - coords.min(axis=0, keepdims=True)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    coords = _normalize_coords(coords)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=int)
    mask[coords[:, 0], coords[:, 1]] = 1
    return mask


def _random_shape() -> np.ndarray:
    mask = _coords_to_mask(np.array(random.choice(BASE_SHAPES), dtype=int))
    mask = np.rot90(mask, k=random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return np.argwhere(mask > 0).astype(int)


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR_COLOR)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    sep = rows[0]
    assert grid.shape[0] == 2 * sep + 1, "separator must split equal-height panels"
    return int(sep)


def _top_objects(grid: np.ndarray, sep: int, bg: int) -> list[dict]:
    upper = grid[:sep]
    colors = [c for c in np.unique(upper) if int(c) not in (bg, SEPARATOR_COLOR)]
    comps = find_components(upper, bg=bg, colors=list(map(int, colors)))
    comps.sort(key=lambda comp: tuple(map(int, comp["coords"].min(axis=0))))
    return comps


def _mirror_coords(coords: np.ndarray, sep: int) -> np.ndarray:
    mirrored = np.asarray(coords, dtype=int).copy()
    mirrored[:, 0] = 2 * int(sep) - mirrored[:, 0]
    return mirrored


def _rounded_centroid(coords: np.ndarray) -> Coord:
    rr = float(coords[:, 0].mean())
    cc = float(coords[:, 1].mean())
    return int(np.floor(rr + 0.5)), int(np.floor(cc + 0.5))


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _placement_valid(coords: np.ndarray, canvas: np.ndarray, sep: int, bg: int) -> bool:
    h, w = canvas.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if rr <= int(sep):
            return False
        if int(canvas[rr, cc]) != int(bg):
            return False
    return True


def _relocate_coords(coords: np.ndarray, canvas: np.ndarray, sep: int, bg: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    min_dr = int(sep + 1 - coords[:, 0].min())
    max_dr = int(canvas.shape[0] - 1 - coords[:, 0].max())
    min_dc = int(-coords[:, 1].min())
    max_dc = int(canvas.shape[1] - 1 - coords[:, 1].max())

    queue = deque([(0, 0)])
    visited = {(0, 0)}
    while queue:
        dr, dc = queue.popleft()
        if (dr, dc) != (0, 0):
            shifted = coords + np.array([dr, dc], dtype=int)
            if _placement_valid(shifted, canvas, sep, bg):
                return shifted, (int(dr), int(dc))
        for step_r, step_c in SEARCH_DIRS:
            nxt = (dr + step_r, dc + step_c)
            if nxt in visited:
                continue
            if not (min_dr <= nxt[0] <= max_dr and min_dc <= nxt[1] <= max_dc):
                continue
            visited.add(nxt)
            queue.append(nxt)
    return None


def _draw_displacement_path(canvas: np.ndarray, start: Coord, end: Coord, bg: int) -> None:
    sr, sc = start
    er, ec = end

    step_c = 1 if ec >= sc else -1
    for c in range(sc, ec + step_c, step_c):
        if int(canvas[sr, c]) == int(bg):
            canvas[sr, c] = PATH_COLOR

    if er != sr:
        step_r = 1 if er >= sr else -1
        for r in range(sr + step_r, er + step_r, step_r):
            if int(canvas[r, ec]) == int(bg):
                canvas[r, ec] = PATH_COLOR


def _solve_case(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = get_bg(g)
    assert bg == BACKGROUND, "expected black background"
    sep = _separator_row(g)

    objects = _top_objects(g, sep, bg)
    assert objects, "expected at least one upper-panel object"

    out = g.copy()
    out[:sep, :] = bg

    placements = []
    for obj in objects:
        mirror = _mirror_coords(obj["coords"], sep)
        wall_hit = any(int(g[r, c]) == WALL_COLOR for r, c in mirror)
        target_center = _rounded_centroid(mirror)

        if _placement_valid(mirror, out, sep, bg):
            _paint_coords(out, mirror, obj["color"])
            placements.append(
                {
                    "original_color": int(obj["color"]),
                    "moved": False,
                    "distance": 0,
                    "offset": (0, 0),
                    "mirror_hits_wall": bool(wall_hit),
                    "mirror_coords": mirror,
                    "placed_coords": mirror.copy(),
                    "target_center": target_center,
                    "placed_center": target_center,
                }
            )
            continue

        relocation = _relocate_coords(mirror, out, sep, bg)
        assert relocation is not None, "expected a reachable free placement for every conflicted object"
        placed, offset = relocation
        placed_center = (target_center[0] + offset[0], target_center[1] + offset[1])
        _draw_displacement_path(out, target_center, placed_center, bg)
        _paint_coords(out, placed, MOVED_COLOR)
        placements.append(
            {
                "original_color": int(obj["color"]),
                "moved": True,
                "distance": int(abs(offset[0]) + abs(offset[1])),
                "offset": offset,
                "mirror_hits_wall": bool(wall_hit),
                "mirror_coords": mirror,
                "placed_coords": placed.copy(),
                "target_center": target_center,
                "placed_center": placed_center,
            }
        )

    if return_meta:
        return out, {"separator": sep, "objects": objects, "placements": placements}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each upper-panel object into the lower panel, rerouting blocked copies and marking moved ones yellow.

    Input structure:
    - The background is black.
    - One full-width grey row splits the grid into equal-height upper and lower panels.
    - The upper panel contains several small 4-connected single-color objects, while the lower panel contains teal wall segments.

    Rule:
    - Clear the upper-panel objects from the output but keep the separator row and all teal walls.
    - Reflect every upper-panel object across the grey row by mapping each pixel row `r` to `2*separator-r`, preserving columns.
    - Process reflected objects in row-major order: place an unblocked reflection in its original color; otherwise BFS over whole-object translations in up-right-down-left priority, draw a horizontal-then-vertical green path from target centroid to placed centroid through background cells only, and place the moved copy in yellow.

    Key insight:
    - The lower panel is not a simple mirror: each mirrored object first targets its exact reflected location, then obstacles and earlier placements force a nearest valid translated copy whose displacement is explicitly annotated by both a green route and a yellow recolor.
    """
    return _solve_case(grid)


def _cells_have_gap(cells: np.ndarray, occupied: set[Coord], h: int, w: int) -> bool:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return False
    return True


def _place_upper_objects(panel_h: int, width: int, count: int) -> list[dict] | None:
    occupied: set[Coord] = set()
    colors = random.sample(OBJECT_COLORS, count)
    objects: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            shape = _random_shape()
            shape = _normalize_coords(shape)
            h = int(shape[:, 0].max()) + 1
            w = int(shape[:, 1].max()) + 1
            if h > panel_h or w > width:
                continue
            r0 = random.randint(0, panel_h - h)
            c0 = random.randint(0, width - w)
            cells = shape + np.array([r0, c0], dtype=int)
            if not _cells_have_gap(cells, occupied, panel_h, width):
                continue
            objects.append({"color": int(color), "coords": cells})
            occupied.update((int(r), int(c)) for r, c in cells)
            placed = True
            break
        if not placed:
            return None

    objects.sort(key=lambda obj: tuple(map(int, obj["coords"].min(axis=0))))
    return objects


def _touches_existing_wall(segment: list[Coord], wall_cells: set[Coord]) -> bool:
    for r, c in segment:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in wall_cells:
                    return True
    return False


def _segment_candidates(hit: Coord, height: int, width: int, sep: int) -> list[list[Coord]]:
    hr, hc = hit
    candidates: list[list[Coord]] = []
    lengths = WALL_LENGTHS[:]
    random.shuffle(lengths)
    orientations = WALL_ORIENTATIONS[:]
    random.shuffle(orientations)

    for orientation in orientations:
        for length in lengths:
            indices = list(range(length))
            random.shuffle(indices)
            for idx in indices:
                if orientation == "h":
                    c0 = hc - idx
                    seg = [(hr, c0 + k) for k in range(length)]
                else:
                    r0 = hr - idx
                    seg = [(r0 + k, hc) for k in range(length)]
                if all(sep < r < height and 0 <= c < width for r, c in seg):
                    candidates.append(seg)
    random.shuffle(candidates)
    return candidates


def _choose_wall_for_object(
    mirror_cells: set[Coord],
    other_targets: set[Coord],
    wall_cells: set[Coord],
    height: int,
    width: int,
    sep: int,
) -> list[Coord] | None:
    hits = list(mirror_cells)
    random.shuffle(hits)
    for hit in hits:
        for seg in _segment_candidates(hit, height, width, sep):
            seg_set = set(seg)
            if not (seg_set & mirror_cells):
                continue
            if seg_set & other_targets:
                continue
            if _touches_existing_wall(seg, wall_cells):
                continue
            return seg
    return None


def _build_candidate() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    height = 2 * panel_h + 1
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)

    objects = _place_upper_objects(panel_h, width, count)
    if objects is None:
        return None

    sep = panel_h
    mirror_sets = [set(map(tuple, _mirror_coords(obj["coords"], sep))) for obj in objects]
    conflict_max = min(CONFLICT_OBJECTS_RANGE[1], count - 1)
    if conflict_max < CONFLICT_OBJECTS_RANGE[0]:
        return None
    n_conflicts = random.randint(CONFLICT_OBJECTS_RANGE[0], conflict_max)
    conflict_indices = set(random.sample(range(count), n_conflicts))

    wall_cells: set[Coord] = set()
    wall_segments: list[list[Coord]] = []
    for idx in sorted(conflict_indices):
        other_targets = set().union(*(mirror_sets[j] for j in range(count) if j != idx))
        segment = _choose_wall_for_object(mirror_sets[idx], other_targets, wall_cells, height, width, sep)
        if segment is None:
            return None
        wall_segments.append(segment)
        wall_cells.update(segment)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[sep, :] = SEPARATOR_COLOR
    for obj in objects:
        _paint_coords(grid, obj["coords"], obj["color"])
    for segment in wall_segments:
        for r, c in segment:
            grid[r, c] = WALL_COLOR
    return grid


def _straight_wall(component: dict) -> bool:
    coords = component["coords"]
    rows = {int(r) for r in coords[:, 0]}
    cols = {int(c) for c in coords[:, 1]}
    if len(rows) == 1:
        vals = sorted(int(c) for c in coords[:, 1])
        return vals == list(range(vals[0], vals[0] + len(vals)))
    if len(cols) == 1:
        vals = sorted(int(r) for r in coords[:, 0])
        return vals == list(range(vals[0], vals[0] + len(vals)))
    return False


def _valid_generated_case(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve_case(grid, return_meta=True)
    except AssertionError:
        return False

    if out.shape != grid.shape or out.ndim != 2 or max(out.shape) > 30:
        return False

    sep = meta["separator"]
    objects = meta["objects"]
    placements = meta["placements"]
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False

    lower = grid[sep + 1 :]
    walls = find_components(lower, bg=BACKGROUND, colors=[WALL_COLOR])
    if not (2 <= len(walls) <= 4):
        return False
    if any(not (3 <= wall["area"] <= 5) or not _straight_wall(wall) for wall in walls):
        return False

    mirror_wall_hits = sum(item["mirror_hits_wall"] for item in placements)
    moved = [item for item in placements if item["moved"]]
    direct = [item for item in placements if not item["moved"]]

    if mirror_wall_hits < 2:
        return False
    if len(moved) < 2 or len(direct) < 1:
        return False
    if any(item["distance"] > 5 for item in moved):
        return False
    if not any((not item["moved"]) and (not item["mirror_hits_wall"]) for item in placements):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _valid_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-reroute ARC input")
