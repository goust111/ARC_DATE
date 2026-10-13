from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
BOX_PADDING = 1
GRID_MAX_SIDE = 30
GRID_MIN_BY_COUNT = {2: 15, 3: 20, 4: 24}

HOLE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _parse_object(grid: np.ndarray, color: int, coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = map(int, patch.shape)
    if height < 3 or width < 3:
        return None
    if not np.all((patch == int(color)) | (patch == BLACK)):
        return None

    hole_mask = patch == BLACK
    hole_area = int(hole_mask.sum())
    if hole_area < 1 or hole_area > 3:
        return None
    if bool(hole_mask[0].any()) or bool(hole_mask[-1].any()) or bool(hole_mask[:, 0].any()) or bool(hole_mask[:, -1].any()):
        return None

    hole_comps = _components(hole_mask)
    if len(hole_comps) != 1:
        return None
    hole_coords = hole_comps[0]
    hr0, hc0, hr1, hc1 = _bbox(hole_coords)

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "height": int(height),
        "width": int(width),
        "hole_key": _shape_key(hole_coords),
        "hole_offset": (int(hr0), int(hc0)),
        "hole_shape_hw": (int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)),
    }


def _grey_clusters(grid: np.ndarray) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for coords in _components(grid == GREY):
        area = int(coords.shape[0])
        if area < 1 or area > 3:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        clusters.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": _shape_key(coords),
            }
        )
    return clusters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each holed rectangle to the grey cluster that matches its internal hole shape, then fill the hole.

    Input structure:
    - The background is black.
    - The grid contains several monochrome rectangular objects, each with one small black connected hole strictly inside the rectangle.
    - Every rectangle uses a distinct non-grey foreground color.
    - The background also contains one grey connected component per object, and each grey component exactly matches one object's hole shape without rotation or reflection.

    Rule:
    - For each colored rectangle, extract the hole shape normalized by its own hole bbox and record the hole bbox offset inside the rectangle.
    - Find the unique grey component with the same normalized shape.
    - Translate the whole rectangle so that the grey component occupies the same relative position that the hole occupied in the source rectangle.
    - In the output, paint the translated rectangle as a solid block of its color, covering the grey component, and turn the source location black.

    Key insight:
    - The black hole is not damage but a positional key: its exact mini-shape chooses the destination anchor, and the hole's offset inside the rectangle fixes the translation.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    objects: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, GREY}):
        for coords in _components(g == int(color)):
            obj = _parse_object(g, int(color), coords)
            if obj is not None:
                objects.append(obj)

    assert objects, "expected at least one colored rectangle with an internal hole"

    clusters = _grey_clusters(g)
    by_shape: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = {}
    for cluster in clusters:
        by_shape.setdefault(cluster["shape"], []).append(cluster)

    assert len(clusters) == len(objects), "each object should have exactly one grey partner"

    out = np.full((height, width), BLACK, dtype=int)
    for obj in sorted(objects, key=lambda item: tuple(item["bbox"])):
        matches = by_shape.get(obj["hole_key"], [])
        assert len(matches) == 1, "hole shapes must map to unique grey clusters"
        cluster = matches[0]
        tr0 = int(cluster["bbox"][0]) - int(obj["hole_offset"][0])
        tc0 = int(cluster["bbox"][1]) - int(obj["hole_offset"][1])
        h = int(obj["height"])
        w = int(obj["width"])
        assert 0 <= tr0 and 0 <= tc0 and tr0 + h <= height and tc0 + w <= width, "translated rectangle leaves the grid"
        region = out[tr0 : tr0 + h, tc0 : tc0 + w]
        assert not np.any(region != BLACK), "translated rectangles should not overlap"
        region[:, :] = int(obj["color"])

    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = BOX_PADDING) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _find_box_slot(
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    box_h: int,
    box_w: int,
) -> tuple[int, int] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, height - box_h)
        c0 = random.randint(0, width - box_w)
        candidate = (int(r0), int(c0), int(r0 + box_h - 1), int(c0 + box_w - 1))
        if all(_boxes_separated(candidate, other) for other in boxes):
            return int(r0), int(c0)
    return None


def _paint_hole(grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = int(color)


def _rect_dims_for_shape(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    coords = np.array(shape, dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _sample_specs() -> list[dict[str, object]]:
    obj_count = random.randint(2, 4)
    colors = random.sample(OBJECT_COLORS, obj_count)
    shapes = random.sample(list(HOLE_SHAPES), obj_count)

    specs: list[dict[str, object]] = []
    for color, shape in zip(colors, shapes):
        hole_h, hole_w = _rect_dims_for_shape(shape)
        rect_h = random.randint(max(3, hole_h + 2), 8)
        rect_w = random.randint(max(3, hole_w + 2), 8)
        hole_r = random.randint(1, rect_h - hole_h - 1)
        hole_c = random.randint(1, rect_w - hole_w - 1)
        specs.append(
            {
                "color": int(color),
                "shape": shape,
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
                "hole_h": int(hole_h),
                "hole_w": int(hole_w),
                "hole_offset": (int(hole_r), int(hole_c)),
            }
        )
    return specs


def _build_expected(height: int, width: int, specs: list[dict[str, object]]) -> np.ndarray:
    expected = np.full((height, width), BLACK, dtype=int)
    for spec in specs:
        tr0, tc0 = spec["target_top_left"]
        h = int(spec["rect_h"])
        w = int(spec["rect_w"])
        expected[tr0 : tr0 + h, tc0 : tc0 + w] = int(spec["color"])
    return expected


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    obj_count = len(specs)
    max_rect_h = max(int(spec["rect_h"]) for spec in specs)
    max_rect_w = max(int(spec["rect_w"]) for spec in specs)
    min_side = max(GRID_MIN_BY_COUNT[obj_count], max_rect_h + 2, max_rect_w + 2)
    if min_side > GRID_MAX_SIDE:
        return None

    height = random.randint(min_side, GRID_MAX_SIDE)
    width = random.randint(min_side, GRID_MAX_SIDE)
    grid = np.full((height, width), BLACK, dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(obj_count), key=lambda idx: int(specs[idx]["rect_h"]) * int(specs[idx]["rect_w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        slot = _find_box_slot(boxes, height, width, int(spec["rect_h"]), int(spec["rect_w"]))
        if slot is None:
            return None
        spec["target_top_left"] = slot
        tr0, tc0 = slot
        boxes.append((int(tr0), int(tc0), int(tr0 + spec["rect_h"] - 1), int(tc0 + spec["rect_w"] - 1)))

    for idx in order:
        spec = specs[idx]
        slot = _find_box_slot(boxes, height, width, int(spec["rect_h"]), int(spec["rect_w"]))
        if slot is None:
            return None
        spec["source_top_left"] = slot
        sr0, sc0 = slot
        boxes.append((int(sr0), int(sc0), int(sr0 + spec["rect_h"] - 1), int(sc0 + spec["rect_w"] - 1)))

    expected = _build_expected(height, width, specs)
    for spec in specs:
        sr0, sc0 = spec["source_top_left"]
        tr0, tc0 = spec["target_top_left"]
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        color = int(spec["color"])
        hole_r, hole_c = spec["hole_offset"]
        shape = spec["shape"]

        grid[sr0 : sr0 + rect_h, sc0 : sc0 + rect_w] = color
        for dr, dc in shape:
            grid[sr0 + hole_r + dr, sc0 + hole_c + dc] = BLACK
        _paint_hole(grid, int(tr0 + hole_r), int(tc0 + hole_c), shape, GREY)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-key rectangle puzzle")
