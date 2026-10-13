from __future__ import annotations

import random
from collections import deque

import numpy as np


PAIR_COLOR = {
    8: 2,  # teal -> red
    3: 1,  # green -> blue
    5: 4,  # grey -> yellow
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INPUT_COLORS = tuple(PAIR_COLOR.keys())

GRID_SIZE_RANGE = (12, 18)
OBJECT_COUNT_CHOICES = (4, 5)
LARGEST_AREA_RANGE = (8, 12)
SMALL_AREAS = (3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coord_array = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coord_array)
            objects.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "area": int(coord_array.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return objects


def _has_hole(mask: np.ndarray) -> bool:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    h, w = padded.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not padded[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~padded) & (~seen)))


def _random_connected_shape(
    area: int,
    *,
    box_h_range: tuple[int, int],
    box_w_range: tuple[int, int],
    require_bbox_slack: bool,
    require_hole_free: bool,
    min_height: int = 2,
    min_width: int = 2,
) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(*box_h_range)
        box_w = random.randint(*box_w_range)
        if box_h * box_w < area:
            continue

        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = mask.T

        if mask.shape[0] < min_height or mask.shape[1] < min_width:
            continue
        if require_bbox_slack and mask.size == area:
            continue
        if require_hole_free and _has_hole(mask):
            continue
        return mask.astype(int)

    raise RuntimeError(f"failed to build a connected shape of area {area}")


def _chebyshev_gap(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    best = None
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            dist = max(abs(int(ra) - int(rb)), abs(int(ca) - int(cb)))
            if best is None or dist < best:
                best = dist
    return int(best if best is not None else 0)


def _place_patches(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    *,
    min_gap: int = 3,
    border_margin: int = 1,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    placed_coords: list[np.ndarray] = []

    order = sorted(range(len(patches)), key=lambda idx: -int(np.sum(patches[idx])))
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        row_min = border_margin
        row_max = h - ph - border_margin
        col_min = border_margin
        col_max = w - pw - border_margin
        if row_max < row_min or col_max < col_min:
            return None

        candidates = [(r, c) for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)]
        random.shuffle(candidates)
        local_coords = np.argwhere(patch != 0)

        chosen = None
        for r0, c0 in candidates:
            abs_coords = local_coords + np.array([[r0, c0]], dtype=int)
            if all(_chebyshev_gap(abs_coords, prev_coords) >= min_gap for prev_coords in placed_coords):
                chosen = (r0, c0)
                positions[idx] = chosen
                placed_coords.append(abs_coords)
                break
        if chosen is None:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _render_grid(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    colors: list[int],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for patch, color, (r0, c0) in zip(patches, colors, positions):
        coords = np.argwhere(patch != 0)
        for r, c in coords:
            grid[r0 + int(r), c0 + int(c)] = int(color)
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _find_objects(g)
    assert objects
    assert len({obj["area"] for obj in objects}) == len(objects)
    objects.sort(key=lambda obj: obj["area"], reverse=True)

    out = g.copy()
    largest = objects[0]
    for r, c in largest["coords"]:
        out[int(r), int(c)] = 0

    contour: set[tuple[int, int]] = set()
    for r, c in largest["coords"]:
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == 0:
                contour.add((nr, nc))
    for r, c in contour:
        out[r, c] = int(largest["color"])

    for obj in objects[1:]:
        pair = PAIR_COLOR[int(obj["color"])]
        r0, c0, r1, c1 = obj["bbox"]
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        transformed = crop.copy()
        transformed[crop == 0] = pair
        for r, c in obj["coords"]:
            transformed[int(r) - r0, int(c) - c0] = 0
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the largest object by its outside contour and invert every other object inside its own bounding box.

    Input structure:
    - The grid has black background and 4-5 monochromatic 4-connected foreground objects.
    - Foreground colors come from teal, green, and grey; object areas are all distinct.
    - The largest object is a hole-free blob, and the other objects are separated so their bounding boxes do not overlap.

    Rule:
    - Detect all foreground objects and rank them by area, largest first.
    - Erase the largest object, then paint every original black cell that was 4-neighbor-adjacent to it using that object's original color.
    - For every non-largest object, work only inside its own minimal bounding box: turn the object's pixels black and turn the original black pixels into the paired color (teal->red, green->blue, grey->yellow).

    Key insight:
    - Area ranking is the switch that chooses between two different transformations: outer contour tracing for rank 1, and bounding-box color inversion for every lower rank.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(np.isin(grid, [0, 3, 5, 8])):
        return False

    objects = _find_objects(grid)
    if len(objects) not in OBJECT_COUNT_CHOICES:
        return False

    areas = [obj["area"] for obj in objects]
    if len(set(areas)) != len(areas):
        return False

    objects.sort(key=lambda obj: obj["area"], reverse=True)
    largest = objects[0]
    if not (LARGEST_AREA_RANGE[0] <= largest["area"] <= LARGEST_AREA_RANGE[1]):
        return False

    largest_mask = np.zeros(
        (largest["bbox"][2] - largest["bbox"][0] + 1, largest["bbox"][3] - largest["bbox"][1] + 1),
        dtype=bool,
    )
    for r, c in largest["coords"]:
        largest_mask[int(r) - largest["bbox"][0], int(c) - largest["bbox"][1]] = True
    if _has_hole(largest_mask):
        return False

    nonmax_colors = {int(obj["color"]) for obj in objects[1:]}
    if nonmax_colors != set(INPUT_COLORS):
        return False

    for obj in objects[1:]:
        if obj["area"] not in SMALL_AREAS:
            return False
        r0, c0, r1, c1 = obj["bbox"]
        if (r1 - r0 + 1) * (c1 - c0 + 1) <= obj["area"]:
            return False

    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if _chebyshev_gap(obj_a["coords"], obj_b["coords"]) < 3:
                return False

    for i, obj_a in enumerate(objects[1:]):
        for obj_b in objects[i + 2 :]:
            if _bbox_overlap(obj_a["bbox"], obj_b["bbox"]):
                return False

    try:
        out = main(grid)
    except Exception:
        return False
    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape != grid.shape:
        return False
    return np.array_equal(out, _expected_output(grid))


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    object_count = random.choice(OBJECT_COUNT_CHOICES)

    largest_patch = _random_connected_shape(
        random.randint(*LARGEST_AREA_RANGE),
        box_h_range=(3, 5),
        box_w_range=(3, 5),
        require_bbox_slack=False,
        require_hole_free=True,
        min_height=3,
        min_width=3,
    )

    small_areas = random.sample(SMALL_AREAS, object_count - 1)
    small_patches = [
        _random_connected_shape(
            area,
            box_h_range=(2, 4),
            box_w_range=(2, 4),
            require_bbox_slack=True,
            require_hole_free=True,
            min_height=2,
            min_width=2,
        )
        for area in small_areas
    ]

    patches = [largest_patch] + small_patches
    largest_color = random.choice(INPUT_COLORS)
    extra_small_color = random.choice(INPUT_COLORS) if object_count == 5 else None
    small_colors = list(INPUT_COLORS)
    if extra_small_color is not None:
        small_colors.append(extra_small_color)
    random.shuffle(small_colors)
    colors = [largest_color] + small_colors

    positions = _place_patches((height, width), patches, min_gap=3, border_margin=1)
    if positions is None:
        return None

    grid = _render_grid((height, width), patches, colors, positions)
    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
