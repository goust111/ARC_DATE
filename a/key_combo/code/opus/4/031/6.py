from __future__ import annotations

import random
from collections import deque

import numpy as np


COLOR_POOL = (1, 2, 3, 4)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def bboxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def component_patch(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = int(component["color"])
    return patch


def _depths(components: list[dict]) -> list[int]:
    boxes = [comp["bbox"] for comp in components]
    return [sum(strict_contains(other, box) for other in boxes if other != box) for box in boxes]


def _transform_with_scales(grid: np.ndarray, scales: list[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = 0
    out = np.full_like(g, bg)
    components = find_components(g, bg=bg)
    depths = _depths(components)
    order = sorted(range(len(components)), key=lambda i: (depths[i], components[i]["bbox"], components[i]["color"]))

    for idx in order:
        comp = components[idx]
        r0, c0, _r1, _c1 = comp["bbox"]
        patch = component_patch(comp, bg)
        scaled = scale_sprite(crop_bbox(patch, (0, 0, patch.shape[0] - 1, patch.shape[1] - 1)), int(scales[idx]))
        rr1 = min(out.shape[0], r0 + scaled.shape[0])
        cc1 = min(out.shape[1], c0 + scaled.shape[1])
        if rr1 <= r0 or cc1 <= c0:
            continue
        view = scaled[: rr1 - r0, : cc1 - c0]
        mask = view != bg
        out[r0:rr1, c0:cc1][mask] = view[mask]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Scale each object by a factor encoded by how deeply its bounding box is nested inside other bounding boxes.

    Input structure:
    - The grid has black background and 3-4 small monochrome 4-connected objects in distinct non-black colors.
    - Some object bounding boxes strictly contain other object bounding boxes; every pair of boxes is either nested or disjoint.
    - At least one object is top-level, at least one is nested once, and at least one is nested twice.

    Rule:
    - Find every 4-connected component and compute its minimal bounding box.
    - The nesting depth of a component is the number of other component bounding boxes that strictly contain its box.
    - Crop the component to its own bounding box, keeping only that component on black background, and scale that patch by depth+1.
    - Paste each scaled patch back at its original bounding-box top-left corner, clipping at grid edges, with deeper objects drawn after shallower ones.

    Key insight:
    - The boxes themselves encode the scale factors: deeper spatial containment means larger pixel replication.
    """
    g = np.asarray(input_grid, dtype=int)
    components = find_components(g, bg=0)
    depths = _depths(components)
    scales = [depth + 1 for depth in depths]
    return _transform_with_scales(g, scales)


def _shape_from_bbox(height: int, width: int, rng: random.Random, corner: str | None = None) -> set[tuple[int, int]]:
    if height == 1:
        return {(0, c) for c in range(width)}
    if width == 1:
        return {(r, 0) for r in range(height)}

    corner = rng.choice(("tl", "tr", "bl", "br")) if corner is None else corner
    cells: set[tuple[int, int]] = set()
    if corner == "tl":
        cells.update((0, c) for c in range(width))
        cells.update((r, 0) for r in range(height))
    elif corner == "tr":
        cells.update((0, c) for c in range(width))
        cells.update((r, width - 1) for r in range(height))
    elif corner == "bl":
        cells.update((height - 1, c) for c in range(width))
        cells.update((r, 0) for r in range(height))
    else:
        cells.update((height - 1, c) for c in range(width))
        cells.update((r, width - 1) for r in range(height))
    return cells


def _paint_shape(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, cells: set[tuple[int, int]]) -> None:
    r0, c0, _r1, _c1 = bbox
    for rr, cc in cells:
        grid[r0 + rr, c0 + cc] = color


def _sample_chain_bboxes(rng: random.Random) -> list[tuple[int, int]]:
    inner_h, inner_w = rng.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
    middle_h = inner_h + 4
    middle_w = inner_w + 4
    outer_h = middle_h + 4
    outer_w = middle_w + 4
    return [(outer_h, outer_w), (middle_h, middle_w), (inner_h, inner_w)]


def _place_nested_bbox(
    outer: tuple[int, int, int, int],
    inner_size: tuple[int, int],
    occupied_corner: str,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inner_h, inner_w = inner_size
    top_margin = h - inner_h - 1 if occupied_corner[0] == "t" else 1
    left_margin = w - inner_w - 1 if occupied_corner[1] == "l" else 1
    inner_r0 = r0 + top_margin
    inner_c0 = c0 + left_margin
    return (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)


def _search_disjoint_bbox(
    canvas_shape: tuple[int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    sizes: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = canvas_shape
    rng.shuffle(sizes)
    for box_h, box_w in sizes:
        rows = list(range(0, h - box_h + 1))
        cols = list(range(0, w - box_w + 1))
        rng.shuffle(rows)
        rng.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                cand = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if all(bboxes_disjoint(cand, other) for other in existing_boxes):
                    return cand
    return None


def _boxes_only_nested_or_disjoint(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if strict_contains(a, b) or strict_contains(b, a) or bboxes_disjoint(a, b):
                continue
            return False
    return True


def _touching_components(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0 and g[nr, nc] != g[r, c]:
                    return True
    return False


def _rank_scales(values: list[int]) -> list[int]:
    uniq = sorted(set(values))
    mapping = {value: idx + 1 for idx, value in enumerate(uniq)}
    return [mapping[v] for v in values]


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    comps = find_components(grid, bg=0)
    if len(comps) not in (3, 4):
        return False
    colors = [comp["color"] for comp in comps]
    if len(set(colors)) != len(colors) or any(color not in COLOR_POOL for color in colors):
        return False
    boxes = [comp["bbox"] for comp in comps]
    if not _boxes_only_nested_or_disjoint(boxes):
        return False
    if _touching_components(grid):
        return False

    depths = _depths(comps)
    if not {0, 1, 2}.issubset(set(depths)):
        return False
    if sum(depth == 2 for depth in depths) != 1:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    unscaled = _transform_with_scales(grid, [1] * len(comps))
    if np.array_equal(out, unscaled):
        return False

    bbox_area_scales = _rank_scales([(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in boxes])
    if np.array_equal(out, _transform_with_scales(grid, bbox_area_scales)):
        return False

    pixel_area_scales = _rank_scales([comp["area"] for comp in comps])
    if np.array_equal(out, _transform_with_scales(grid, pixel_area_scales)):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random(random.getrandbits(64))

    for _ in range(500):
        chain_sizes = _sample_chain_bboxes(rng)
        outer_h, outer_w = chain_sizes[0]
        grid_h = rng.randint(max(outer_h + 4, 12), min(outer_h + 10, 18))
        grid_w = rng.randint(max(outer_w + 4, 12), min(outer_w + 10, 18))
        grid = np.zeros((grid_h, grid_w), dtype=int)

        outer_r0 = rng.randint(1, grid_h - outer_h - 1)
        outer_c0 = rng.randint(1, grid_w - outer_w - 1)
        outer_box = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
        outer_corner = rng.choice(("tl", "tr", "bl", "br"))
        middle_corner = rng.choice(("tl", "tr", "bl", "br"))
        middle_box = _place_nested_bbox(outer_box, chain_sizes[1], outer_corner)
        inner_box = _place_nested_bbox(middle_box, chain_sizes[2], middle_corner)

        boxes = [outer_box, middle_box, inner_box]

        if rng.random() < 0.65:
            extra_sizes = [(4, 4), (3, 4), (4, 3), (1, 4), (4, 1), (2, 3), (3, 2), (2, 2)]
            extra_box = _search_disjoint_bbox(grid.shape, [outer_box], extra_sizes, rng)
            if extra_box is not None:
                boxes.append(extra_box)

        rng.shuffle(boxes)
        colors = list(COLOR_POOL)
        rng.shuffle(colors)

        corners: dict[tuple[int, int, int, int], str | None] = {
            outer_box: outer_corner,
            middle_box: middle_corner,
            inner_box: None,
        }
        if len(boxes) == 4:
            for box in boxes:
                corners.setdefault(box, None)

        for box, color in zip(boxes, colors):
            h = box[2] - box[0] + 1
            w = box[3] - box[1] + 1
            _paint_shape(grid, box, color, _shape_from_bbox(h, w, rng, corners.get(box)))

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid nested-depth scaling puzzle")
