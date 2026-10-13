from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 5
FG_COLORS = [1, 2, 3, 4]
GRID_SIDE_RANGE = (11, 15)
OBJECT_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 400
PLACEMENT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _mirror_mosaic(tile: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    h, w = tile.shape
    out = np.empty((3 * h, 3 * w), dtype=int)
    for sr in range(3):
        for sc in range(3):
            patch = tile
            if sr != 1:
                patch = np.flipud(patch)
            if sc != 1:
                patch = np.fliplr(patch)
            out[sr * h : (sr + 1) * h, sc * w : (sc + 1) * w] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the uniquely largest foreground component to its local bbox and mirror-tile that crop into a 3x3 mosaic.

    Input structure:
    - The grid has a dominant grey background and 3-4 monochromatic 4-connected foreground objects.
    - Foreground objects use distinct non-background colors, and exactly one object has strictly largest area.
    - The target object's own bounding box contains both its color and some background cells.

    Rule:
    - Find all non-background 4-connected components and select the one with strictly maximum pixel count.
    - Crop the input to that component's bounding box, keeping both the object's pixels and any background holes inside the box.
    - Use this crop as a tile T and build a 3x3 output mosaic: center = T, vertical neighbors = upside-down T, horizontal neighbors = left-right flipped T, corners = flipped both ways.

    Key insight:
    - The source tile is determined by hidden structure (largest component area), not by color, absolute position, or bbox size; all other objects are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected foreground objects"

    max_area = max(comp["area"] for comp in components)
    largest = [comp for comp in components if comp["area"] == max_area]
    assert len(largest) == 1, "expected a unique largest component"

    tile = _crop_bbox(g, largest[0]["bbox"])
    return _mirror_mosaic(tile)


def _connected_spanning_mask(
    h: int,
    w: int,
    target_area: int,
    *,
    require_hole: bool,
) -> np.ndarray | None:
    if target_area < h + w - 1:
        return None
    if require_hole and target_area >= h * w:
        return None

    pivot_r = random.randrange(h)
    pivot_c = random.randrange(w)
    mask = np.zeros((h, w), dtype=bool)
    mask[pivot_r, :] = True
    mask[:, pivot_c] = True

    while int(mask.sum()) < target_area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    rows, cols = np.where(mask)
    if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
        return None
    if require_hole and bool(mask.all()):
        return None
    return mask


def _sample_component(
    h: int,
    w: int,
    *,
    min_area: int | None = None,
    max_area: int | None = None,
    require_hole: bool = False,
    exact_area: int | None = None,
) -> tuple[np.ndarray, int] | None:
    low = h + w - 1 if min_area is None else int(min_area)
    high = h * w - (1 if require_hole else 0) if max_area is None else int(max_area)
    if exact_area is not None:
        low = high = int(exact_area)
    if low > high:
        return None

    for _ in range(80):
        area = random.randint(low, high)
        mask = _connected_spanning_mask(h, w, area, require_hole=require_hole)
        if mask is not None:
            return mask, area
    return None


def _boxes_separated(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    padding: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _place_objects(
    boxes: list[tuple[int, int]],
    largest_index: int,
    mode: str,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(boxes)))
    random.shuffle(order)

    for idx in order:
        bh, bw = boxes[idx]
        candidates = [(r, c) for r in range(grid_h - bh + 1) for c in range(grid_w - bw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if all(_boxes_separated(box, other, PLACEMENT_PADDING) for other in placed_boxes):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        positions.append(chosen)
        placed_boxes.append((chosen[0], chosen[1], chosen[0] + bh - 1, chosen[1] + bw - 1))

    ordered_positions = [None] * len(boxes)
    for idx, pos in zip(order, positions):
        ordered_positions[idx] = pos

    if mode == "position":
        largest_r, largest_c = ordered_positions[largest_index]
        others = [ordered_positions[i] for i in range(len(boxes)) if i != largest_index]
        if not any(r < largest_r for r, _ in others):
            return None
        if not any(c < largest_c for _, c in others):
            return None
    return ordered_positions


def _render_input(
    specs: list[dict],
    positions: list[tuple[int, int]],
    colors: list[int],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (r0, c0), color in zip(specs, positions, colors):
        mask = spec["mask"]
        h, w = mask.shape
        patch = np.where(mask, color, BG)
        view = grid[r0 : r0 + h, c0 : c0 + w]
        assert not np.any((view != BG) & mask)
        view[mask] = patch[mask]
    return grid


def _expected_from_largest(spec: dict, color: int) -> np.ndarray:
    tile = np.where(spec["mask"], color, BG)
    return _mirror_mosaic(tile)


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    largest_bbox_shape: tuple[int, int],
    largest_area: int,
    mode: str,
) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False

    comps = _find_components(grid, BG)
    if len(comps) not in OBJECT_COUNT_CHOICES:
        return False
    if len({comp["color"] for comp in comps}) != len(comps):
        return False

    areas = [comp["area"] for comp in comps]
    if areas.count(max(areas)) != 1:
        return False
    chosen = max(comps, key=lambda comp: comp["area"])
    if chosen["area"] != largest_area:
        return False

    tile = _crop_bbox(grid, chosen["bbox"])
    if tile.shape != largest_bbox_shape:
        return False
    if tile.shape[0] not in (3, 4) or tile.shape[1] not in (3, 4):
        return False
    if not np.any(tile == BG):
        return False
    if expected.shape != (tile.shape[0] * 3, tile.shape[1] * 3):
        return False
    if not np.array_equal(_mirror_mosaic(tile), expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    if mode == "position":
        r0, c0, _, _ = chosen["bbox"]
        others = [comp for comp in comps if comp is not chosen]
        if not any(comp["bbox"][0] < r0 for comp in others):
            return False
        if not any(comp["bbox"][1] < c0 for comp in others):
            return False
    elif mode == "bbox_decoy":
        largest_bbox_area = largest_bbox_shape[0] * largest_bbox_shape[1]
        if not any(
            comp["area"] < largest_area
            and (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1) >= largest_bbox_area
            for comp in comps
            if comp is not chosen
        ):
            return False
    elif mode == "near_area":
        if not any(comp["area"] == largest_area - 1 for comp in comps if comp is not chosen):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    largest_h = random.choice((3, 4))
    largest_w = random.choice((3, 4))
    largest_sample = _sample_component(
        largest_h,
        largest_w,
        min_area=largest_h + largest_w - 1,
        max_area=largest_h * largest_w - 1,
        require_hole=True,
    )
    if largest_sample is None:
        return None
    largest_mask, largest_area = largest_sample

    mode = random.choice(("position", "bbox_decoy", "near_area"))

    specs = [{"mask": largest_mask, "area": largest_area, "shape": (largest_h, largest_w)}]
    if mode == "bbox_decoy":
        decoy_dims = [(h, w) for h in (3, 4) for w in (3, 4) if h * w >= largest_h * largest_w]
        random.shuffle(decoy_dims)
        placed = False
        for dh, dw in decoy_dims:
            sample = _sample_component(
                dh,
                dw,
                min_area=dh + dw - 1,
                max_area=min(dh * dw, largest_area - 1),
                require_hole=random.choice((True, False)),
            )
            if sample is not None:
                mask, area = sample
                specs.append({"mask": mask, "area": area, "shape": (dh, dw)})
                placed = True
                break
        if not placed:
            return None

    while len(specs) < n_objects:
        dh = random.choice((2, 3, 4))
        dw = random.choice((2, 3, 4, 4))
        max_area = min(dh * dw, largest_area - 1)
        if max_area < dh + dw - 1:
            continue
        if mode == "near_area" and len(specs) == 1:
            target_area = largest_area - 1
            if target_area > max_area or target_area < dh + dw - 1:
                continue
            sample = _sample_component(dh, dw, exact_area=target_area, require_hole=False)
        else:
            sample = _sample_component(
                dh,
                dw,
                min_area=dh + dw - 1,
                max_area=max_area,
                require_hole=random.choice((False, True)) and max_area < dh * dw,
            )
        if sample is None:
            continue
        mask, area = sample
        specs.append({"mask": mask, "area": area, "shape": (dh, dw)})

    if specs[0]["area"] <= max(spec["area"] for spec in specs[1:]):
        return None

    colors = random.sample(FG_COLORS, n_objects)
    largest_color_index = random.randrange(n_objects)
    colors[0], colors[largest_color_index] = colors[largest_color_index], colors[0]

    boxes = [spec["shape"] for spec in specs]
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    positions = _place_objects(boxes, 0, mode, grid_h, grid_w)
    if positions is None:
        return None

    grid = _render_input(specs, positions, colors, grid_h, grid_w)
    expected = _expected_from_largest(specs[0], colors[0])
    if not _semantic_checks(grid, expected, (largest_h, largest_w), largest_area, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
