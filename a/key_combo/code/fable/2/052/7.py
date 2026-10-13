from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, paste, random_free_location


SPIRAL_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
GEN_COLOR_POOL = [c for c in Color.NOT_BLACK if c != Color.GREY]

GEN_HEIGHT_RANGE = (14, 24)
GEN_WIDTH_RANGE = (14, 24)
GEN_OBJECT_COUNT_RANGE = (4, 6)
GEN_EXTRA_AREA_RANGE = (2, 6)
GEN_MAX_SPAN = 4
GEN_PADDING = 1
GEN_ATTEMPTS = 600


@lru_cache(maxsize=None)
def _spiral_positions(radius: int) -> tuple[tuple[int, int], ...]:
    radius = int(radius)
    assert radius >= 0
    limit = (2 * radius + 1) ** 2
    positions = [(0, 0)]
    r = c = 0
    step_len = 1
    direction_idx = 0

    while len(positions) < limit:
        for _ in range(2):
            dr, dc = SPIRAL_DIRS[direction_idx % 4]
            for _ in range(step_len):
                r += dr
                c += dc
                positions.append((r, c))
                if len(positions) >= limit:
                    break
            direction_idx += 1
            if len(positions) >= limit:
                break
        step_len += 1
    return tuple(positions)


@lru_cache(maxsize=None)
def _spiral_index_map(radius: int) -> dict[tuple[int, int], int]:
    return {pos: idx for idx, pos in enumerate(_spiral_positions(radius))}


def _origin_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == Color.GREY)
    assert coords.shape == (1, 2), "expected exactly one gray origin marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    assert bg == Color.BLACK, "expected black background"

    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True)
    objects: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        if color == Color.GREY:
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        objects.append(
            {
                "color": color,
                "coords": coords,
                "area": int(coords.shape[0]),
            }
        )

    assert len(objects) >= 2, "expected at least two colored objects"
    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors)), "expected all objects to have distinct colors"
    return objects


def _owner_grid(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for obj_id, obj in enumerate(objects):
        coords = obj["coords"]
        owner[coords[:, 0], coords[:, 1]] = obj_id
    return owner


def _max_object_radius(origin: tuple[int, int], objects: list[dict]) -> int:
    r0, c0 = origin
    return max(
        max(abs(int(r) - r0), abs(int(c) - c0))
        for obj in objects
        for r, c in obj["coords"]
    )


def _spiral_first_touch_order(
    shape: tuple[int, int],
    origin: tuple[int, int],
    owner: np.ndarray,
    n_objects: int,
) -> tuple[list[int], int]:
    r0, c0 = origin
    if n_objects:
        coords = np.argwhere(owner >= 0)
        radius = max(max(abs(int(r) - r0), abs(int(c) - c0)) for r, c in coords)
    else:
        radius = 0
    order: list[int] = []
    seen: set[int] = set()
    skipped = 0
    for dr, dc in _spiral_positions(radius):
        rr = r0 + dr
        cc = c0 + dc
        if not in_bounds(shape, rr, cc):
            skipped += 1
            continue
        obj_id = int(owner[rr, cc])
        if obj_id >= 0 and obj_id not in seen:
            seen.add(obj_id)
            order.append(obj_id)
            if len(order) == n_objects:
                break
    assert len(order) == n_objects, "spiral traversal failed to reach every object"
    return order, skipped


def _oracle_expected(grid: np.ndarray, origin: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, list[int]]:
    r0, c0 = origin
    radius = _max_object_radius(origin, objects)
    index_map = _spiral_index_map(radius)
    scored = []
    for obj_id, obj in enumerate(objects):
        coords = obj["coords"]
        rel = [(int(r) - r0, int(c) - c0) for r, c in coords]
        hit_index = min(index_map[cell] for cell in rel)
        scored.append((hit_index, obj_id))
    order = [obj_id for _hit, obj_id in sorted(scored)]

    out = np.array(grid, copy=True)
    colors = [objects[obj_id]["color"] for obj_id in order]
    for idx, obj_id in enumerate(order):
        coords = objects[obj_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = colors[idx - 1]
    out[origin] = Color.GREY
    return out, order


def _rowmajor_key(coords: np.ndarray, width: int) -> int:
    return min(int(r) * width + int(c) for r, c in coords)


def _distance_key(origin: tuple[int, int], coords: np.ndarray, width: int) -> tuple[int, int]:
    r0, c0 = origin
    ring = min(max(abs(int(r) - r0), abs(int(c) - c0)) for r, c in coords)
    return ring, _rowmajor_key(coords, width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle the object colors by the order in which a clipped square spiral from the gray marker first touches them.

    Input structure:
    - A black background contains several disconnected monochrome objects with pairwise distinct colors.
    - Exactly one gray single pixel marks the spiral origin and is not part of any object.
    - Objects may lie on different spiral rings, and edge clipping can remove parts of the infinite spiral path.

    Rule:
    - Start at the gray marker and traverse an outward square spiral cell by cell with the fixed order up, right, down, left and step lengths 1, 1, 2, 2, 3, 3, ...
    - Ignore spiral positions that fall outside the grid, but keep the same global traversal order.
    - Record the first time each object is touched by the spiral, then recolor every object with the previous object's original color in that order, wrapping the first object to the last object's color.

    Key insight:
    - The recoloring order is neither reading order nor pure distance from the marker; it is the first-hit order induced by the clipped square spiral.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    origin = _origin_coord(g)
    objects = _extract_objects(g)
    owner = _owner_grid(g.shape, objects)
    order, _skipped = _spiral_first_touch_order(g.shape, origin, owner, len(objects))

    out = g.copy()
    original_colors = [objects[obj_id]["color"] for obj_id in order]
    for idx, obj_id in enumerate(order):
        coords = objects[obj_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = original_colors[idx - 1]
    out[origin] = Color.GREY
    return out


def _sample_origin(height: int, width: int) -> tuple[int, int]:
    side = random.choice(("top", "bottom", "left", "right", "tl", "tr", "bl", "br"))
    band_r = list(range(min(3, height)))
    band_c = list(range(min(3, width)))
    inner_rows = list(range(2, height - 2))
    inner_cols = list(range(2, width - 2))

    if side == "top":
        return random.choice(band_r), random.choice(inner_cols)
    if side == "bottom":
        return height - 1 - random.choice(band_r), random.choice(inner_cols)
    if side == "left":
        return random.choice(inner_rows), random.choice(band_c)
    if side == "right":
        return random.choice(inner_rows), width - 1 - random.choice(band_c)
    if side == "tl":
        return random.choice(band_r), random.choice(band_c)
    if side == "tr":
        return random.choice(band_r), width - 1 - random.choice(band_c)
    if side == "bl":
        return height - 1 - random.choice(band_r), random.choice(band_c)
    return height - 1 - random.choice(band_r), width - 1 - random.choice(band_c)


def _random_polyomino(area: int, max_span: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in SPIRAL_DIRS:
                cand = (r + dr, c + dc)
                if cand in cells:
                    continue
                test = cells | {cand}
                rows = [rr for rr, _cc in test]
                cols = [cc for _rr, cc in test]
                if max(rows) - min(rows) + 1 <= max_span and max(cols) - min(cols) + 1 <= max_span:
                    frontier.add(cand)
        assert frontier, "polyomino growth frontier must be non-empty"
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    height = max(r for r, _c in norm) + 1
    width = max(c for _r, c in norm) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _choose_anchor_pair(
    height: int,
    width: int,
    origin: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0 = origin
    max_radius = max(r0, height - 1 - r0, c0, width - 1 - c0)
    index_map = _spiral_index_map(max_radius)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for ring in range(2, max_radius + 1):
        ring_cells = []
        for r in range(height):
            for c in range(width):
                if max(abs(r - r0), abs(c - c0)) == ring:
                    ring_cells.append((r, c))
        if len(ring_cells) < 2:
            continue
        random.shuffle(ring_cells)
        for i, a in enumerate(ring_cells):
            for b in ring_cells[i + 1 :]:
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                    continue
                sdiff = index_map[(a[0] - r0, a[1] - c0)] - index_map[(b[0] - r0, b[1] - c0)]
                rdiff = (a[0] * width + a[1]) - (b[0] * width + b[1])
                if sdiff * rdiff < 0:
                    candidates.append((a, b))
        if candidates:
            break

    return None if not candidates else random.choice(candidates)


def _objects_from_grid(grid: np.ndarray) -> list[dict]:
    origin = _origin_coord(grid)
    objects = _extract_objects(grid)
    owner = _owner_grid(grid.shape, objects)
    order, skipped = _spiral_first_touch_order(grid.shape, origin, owner, len(objects))
    for expected, obj_id in enumerate(order):
        objects[obj_id]["spiral_rank"] = expected
    objects.sort(key=lambda obj: obj["spiral_rank"])
    for obj in objects:
        coords = obj["coords"]
        width = grid.shape[1]
        obj["rowmajor_key"] = _rowmajor_key(coords, width)
        obj["distance_key"] = _distance_key(origin, coords, width)
    return objects


def _shape_sensitive(origin: tuple[int, int], objects: list[dict], width: int) -> bool:
    radius = _max_object_radius(origin, objects)
    index_map = _spiral_index_map(radius)
    r0, c0 = origin
    for obj in objects:
        if obj["area"] < 3:
            continue
        coords = obj["coords"]
        earliest = min(coords.tolist(), key=lambda rc: index_map[(int(rc[0]) - r0, int(rc[1]) - c0)])
        rowmajor = min(coords.tolist(), key=lambda rc: int(rc[0]) * width + int(rc[1]))
        if tuple(map(int, earliest)) != tuple(map(int, rowmajor)):
            return True
    return False


def _validate_case(grid: np.ndarray, expected: np.ndarray, anchor_cells: set[tuple[int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.count_nonzero(grid == Color.GREY)) != 1:
        return False
    if int(get_bg(grid)) != Color.BLACK:
        return False
    if np.array_equal(grid, expected):
        return False

    origin = _origin_coord(grid)
    objects = _objects_from_grid(grid)
    if len(objects) < 4:
        return False
    owner = _owner_grid(grid.shape, objects)
    _order, skipped = _spiral_first_touch_order(grid.shape, origin, owner, len(objects))
    if skipped <= 0:
        return False

    width = grid.shape[1]
    raster_order = sorted(objects, key=lambda obj: obj["rowmajor_key"])
    distance_order = sorted(objects, key=lambda obj: obj["distance_key"])
    spiral_colors = [obj["color"] for obj in objects]
    raster_colors = [obj["color"] for obj in raster_order]
    distance_colors = [obj["color"] for obj in distance_order]
    if spiral_colors == raster_colors:
        return False
    if spiral_colors == distance_colors:
        return False

    anchor_hit_colors = []
    for obj in objects:
        coords_set = {tuple(map(int, rc)) for rc in obj["coords"]}
        if coords_set & anchor_cells:
            anchor_hit_colors.append(obj["color"])
    if len(anchor_hit_colors) != 2:
        return False

    radius = _max_object_radius(origin, objects)
    index_map = _spiral_index_map(radius)
    anchor_list = list(anchor_cells)
    a, b = anchor_list[0], anchor_list[1]
    sdiff = index_map[(a[0] - origin[0], a[1] - origin[1])] - index_map[(b[0] - origin[0], b[1] - origin[1])]
    rdiff = (a[0] * width + a[1]) - (b[0] * width + b[1])
    if sdiff * rdiff >= 0:
        return False

    if not _shape_sensitive(origin, objects, width):
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    out_objects = _extract_objects(out)
    if len(out_objects) != len(objects):
        return False
    if int(np.count_nonzero(out == Color.GREY)) != 1 or int(out[origin]) != Color.GREY:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    origin = _sample_origin(height, width)
    anchor_pair = _choose_anchor_pair(height, width, origin)
    if anchor_pair is None:
        return None

    n_objects = random.randint(*GEN_OBJECT_COUNT_RANGE)
    colors = random.sample(GEN_COLOR_POOL, n_objects)
    grid = np.zeros((height, width), dtype=int)
    occupancy = np.zeros((height, width), dtype=int)

    grid[origin] = Color.GREY
    occupancy[origin] = 1

    objects: list[dict] = []
    anchor_cells = set(anchor_pair)
    for color, cell in zip(colors[:2], anchor_pair):
        r, c = cell
        if occupancy[r, c] != 0:
            return None
        grid[r, c] = int(color)
        occupancy[r, c] = 1
        objects.append(
            {
                "color": int(color),
                "coords": np.array([[r, c]], dtype=int),
                "area": 1,
            }
        )

    for idx, color in enumerate(colors[2:], start=2):
        min_area = 3 if idx == 2 else GEN_EXTRA_AREA_RANGE[0]
        area = random.randint(min_area, GEN_EXTRA_AREA_RANGE[1])
        mask = _random_polyomino(area=area, max_span=GEN_MAX_SPAN)
        patch = _patch_from_mask(mask, color)
        occ_patch = mask.astype(int)
        try:
            r0, c0 = random_free_location(
                occupancy,
                occ_patch,
                transparent=0,
                border=0,
                padding=GEN_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(grid, patch, r0, c0, transparent=0)
        paste(occupancy, occ_patch, r0, c0, transparent=0)
        coords = np.argwhere(mask) + np.array([[r0, c0]])
        objects.append(
            {
                "color": int(color),
                "coords": coords.astype(int),
                "area": int(coords.shape[0]),
            }
        )

    expected, _order = _oracle_expected(grid, origin, objects)
    if not _validate_case(grid, expected, anchor_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-cycle ARC input")
