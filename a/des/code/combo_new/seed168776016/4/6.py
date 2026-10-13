from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG_COLOR = 0
OBJECT_COLOR = 5
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
EXTRA_OBJECTS_RANGE = (0, 2)
GRID_W_RANGE = (18, 30)
GRID_H_RANGE = (18, 30)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 150
HOLE_GEN_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    )


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(
    coords: list[tuple[int, int]],
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    if bbox is None:
        bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size != 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _hole_masks(fg_mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, fg_mask.shape)
    regions: list[np.ndarray] = []
    for coords in _components(~fg_mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        region = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            region[r, c] = True
        regions.append(_crop_true(region))
    return regions


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _scene_components(grid: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for coords in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        bbox = _bbox(coords)
        mask = _mask_from_coords(coords, bbox)
        holes = _hole_masks(mask)
        if len(holes) != 1:
            continue
        hole = holes[0]
        infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "color": next(iter(colors)),
                "mask": mask,
                "hole": hole,
                "hole_key": _shape_key(hole),
            }
        )
    return infos


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    infos = _scene_components(g, bg)
    assert infos, "expected monochrome components with one enclosed hole"

    legend_top = min(int(info["bbox"][0]) for info in infos)
    legends = [info for info in infos if int(info["bbox"][0]) == legend_top]
    objects = [info for info in infos if int(info["bbox"][0]) > legend_top]

    assert len(legends) >= 2, "expected at least two legend templates on the top row"
    assert objects, "expected at least one lower object"

    legend_map: dict[tuple[tuple[int, int], bytes], int] = {}
    for legend in legends:
        key = legend["hole_key"]
        assert key not in legend_map, "legend hole shapes must be unique"
        legend_map[key] = int(legend["color"])

    for obj in objects:
        assert obj["hole_key"] in legend_map, "every lower object hole must match a legend template"

    return bg, legends, objects, legend_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower object by matching its internal hole shape to a top-row legend template.

    Input structure:
    - The background is the dominant color.
    - Along the top edge sits a legend row of small monochrome square-like components, each with one enclosed background hole pattern and a distinct visible color.
    - Below the legend row are larger monochrome objects; each object is connected, contains exactly one enclosed background hole, and uses the same outer color before recoloring.

    Rule:
    - Extract the enclosed hole shape from every top-row legend component and map that shape to the legend component's color.
    - For each lower object, extract its unique enclosed hole shape and find the top-row legend entry with the exact same shape, without rotating or flipping it.
    - Repaint every non-background pixel of the lower object with the matched legend color.
    - Leave the legend row unchanged and preserve all hole cells as background.

    Key insight:
    - The internal hole shape is the hidden signature that identifies which legend color belongs to each larger object.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _legends, objects, legend_map = _analyze_scene(g)

    out = g.copy()
    for obj in objects:
        new_color = int(legend_map[obj["hole_key"]])
        for r, c in obj["coords"]:
            out[r, c] = new_color
    return out


def _random_connected_hole() -> np.ndarray:
    for _ in range(HOLE_GEN_RETRIES):
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        area = random.randint(1, h * w)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_true(mask)
        if len(_components(mask)) != 1:
            continue
        if _hole_masks(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a valid hole mask")


def _make_legend_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    hh, ww = map(int, hole_mask.shape)
    side = max(hh, ww) + random.randint(2, 3)
    patch = np.full((side, side), int(color), dtype=int)
    hole_r0 = random.randint(1, side - hh - 1)
    hole_c0 = random.randint(1, side - ww - 1)
    window = patch[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + ww]
    window[hole_mask] = BG_COLOR
    return patch


def _make_object_patch(hole_mask: np.ndarray) -> np.ndarray:
    hh, ww = map(int, hole_mask.shape)
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)

    base_h = hh + top + bottom
    base_w = ww + left + right
    canvas = np.zeros((base_h + 6, base_w + 6), dtype=bool)
    base_r0 = 3
    base_c0 = 3
    canvas[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    hole_r0 = base_r0 + top
    hole_c0 = base_c0 + left
    window = canvas[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + ww]
    window[hole_mask] = False

    sides = random.sample(("top", "bottom", "left", "right"), random.randint(1, 4))
    if random.random() < 0.5:
        sides.append(random.choice(("top", "bottom", "left", "right")))
    for side in sides:
        if side in ("top", "bottom"):
            arm_h = random.randint(1, 2)
            span = random.randint(1, base_w)
            offset = random.randint(0, base_w - span)
            c0 = base_c0 + offset
            c1 = c0 + span
            if side == "top":
                canvas[base_r0 - arm_h : base_r0, c0:c1] = True
            else:
                canvas[base_r0 + base_h : base_r0 + base_h + arm_h, c0:c1] = True
        else:
            arm_w = random.randint(1, 2)
            span = random.randint(1, base_h)
            offset = random.randint(0, base_h - span)
            r0 = base_r0 + offset
            r1 = r0 + span
            if side == "left":
                canvas[r0:r1, base_c0 - arm_w : base_c0] = True
            else:
                canvas[r0:r1, base_c0 + base_w : base_c0 + base_w + arm_w] = True

    shape = _crop_true(canvas)
    patch = np.full(shape.shape, BG_COLOR, dtype=int)
    patch[shape] = OBJECT_COLOR
    return patch


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    mask = patch != BG_COLOR
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    legend_colors = random.sample(LEGEND_COLOR_POOL, legend_count)

    hole_masks: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    while len(hole_masks) < legend_count:
        hole = _random_connected_hole()
        key = _shape_key(hole)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        hole_masks.append(hole)

    legends = [
        {
            "hole": hole,
            "color": color,
            "patch": _make_legend_patch(hole, color),
        }
        for hole, color in zip(hole_masks, legend_colors)
    ]
    random.shuffle(legends)

    extra_objects = random.randint(*EXTRA_OBJECTS_RANGE)
    assignments = list(range(legend_count)) + [random.randrange(legend_count) for _ in range(extra_objects)]
    random.shuffle(assignments)
    objects = []
    for idx in assignments:
        hole = legends[idx]["hole"]
        objects.append(
            {
                "legend_index": idx,
                "color": legends[idx]["color"],
                "patch": _make_object_patch(hole),
            }
        )

    legend_gap_values = [random.randint(1, 2) for _ in range(max(0, legend_count - 1))]
    legend_total_w = sum(int(item["patch"].shape[1]) for item in legends) + sum(legend_gap_values)
    legend_max_h = max(int(item["patch"].shape[0]) for item in legends)
    max_obj_h = max(int(item["patch"].shape[0]) for item in objects)
    max_obj_w = max(int(item["patch"].shape[1]) for item in objects)

    min_w = max(GRID_W_RANGE[0], legend_total_w, max_obj_w + 2)
    min_h = max(GRID_H_RANGE[0], legend_max_h + max_obj_h + 5, 12 + 2 * len(objects))
    if min_w > GRID_W_RANGE[1] or min_h > GRID_H_RANGE[1]:
        return None

    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    left_margin = random.randint(0, grid_w - legend_total_w)
    c = left_margin
    for i, legend in enumerate(legends):
        patch = legend["patch"]
        if not _reserve(occupied, 0, c, patch.shape[0], patch.shape[1], pad=1):
            return None
        legend["pos"] = (0, c)
        _paint_patch(grid, patch, 0, c)
        c += patch.shape[1]
        if i < len(legend_gap_values):
            c += legend_gap_values[i]

    object_zone_top = legend_max_h + random.randint(2, 4)
    if object_zone_top >= grid_h:
        return None

    for obj in objects:
        patch = obj["patch"]
        h, w = map(int, patch.shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(object_zone_top, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            snapshot = occupied.copy()
            if not _reserve(occupied, r0, c0, h, w, pad=1):
                occupied = snapshot
                continue
            obj["pos"] = (r0, c0)
            _paint_patch(grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    if _dominant_color(grid) != BG_COLOR:
        return None

    expected = grid.copy()
    for obj in objects:
        r0, c0 = obj["pos"]
        patch = obj["patch"]
        mask = patch != BG_COLOR
        expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = int(obj["color"])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    bg, legends_found, objects_found, legend_map = _analyze_scene(grid)
    if bg != BG_COLOR:
        return None
    if len(legends_found) != legend_count or len(objects_found) != len(objects):
        return None
    if len(legend_map) != legend_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-legend recoloring sample")
