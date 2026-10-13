import random
from collections import Counter

import numpy as np


ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BIG_BBOX_CHOICES = ((5, 5), (5, 7), (7, 5), (7, 7))
BIG_KIND_CHOICES = ("full", "cut_corners", "center_hole", "inner_corners")
GRID_HEIGHT_RANGE = (22, 31)
GRID_WIDTH_RANGE = (22, 31)
MAX_GENERATION_ATTEMPTS = 400
MIN_MOVE_DISTANCE = 4
SEPARATION = 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return objects


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _split_objects_by_area(objects: list[dict]) -> tuple[list[dict], list[dict]]:
    ranked = sorted(objects, key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1], obj["color"]))
    assert len(ranked) >= 2, "expected at least two objects"

    areas = [obj["area"] for obj in ranked]
    gaps = [areas[i + 1] - areas[i] for i in range(len(areas) - 1)]
    split_idx = int(np.argmax(gaps))
    assert gaps[split_idx] > 0, "expected a non-zero area gap"

    small = ranked[: split_idx + 1]
    big = ranked[split_idx + 1 :]
    assert small and big, "expected both small and big objects"
    return small, big


def _nearest_center(center: tuple[int, int], centers: list[tuple[int, int]]) -> int:
    dists = [(center[0] - rr) ** 2 + (center[1] - cc) ** 2 for rr, cc in centers]
    best = min(dists)
    assert dists.count(best) == 1, "nearest big center must be unique"
    return int(dists.index(best))


def main(grid: np.ndarray) -> np.ndarray:
    """Replace large objects with bbox crosses and move small objects onto the nearest big center."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    objects = _find_objects(g, bg)
    small_objects, big_objects = _split_objects_by_area(objects)

    out = np.full(g.shape, bg, dtype=int)
    big_centers = [_bbox_center(obj["bbox"]) for obj in big_objects]

    for obj in sorted(big_objects, key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = obj["bbox"]
        cr, cc = _bbox_center(obj["bbox"])
        out[cr, c0 : c1 + 1] = obj["color"]
        out[r0 : r1 + 1, cc] = obj["color"]

    for obj in sorted(small_objects, key=lambda item: item["bbox"]):
        src_center = _bbox_center(obj["bbox"])
        target_center = big_centers[_nearest_center(src_center, big_centers)]
        dr = target_center[0] - src_center[0]
        dc = target_center[1] - src_center[1]
        for r, c in obj["coords"]:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "translated small object left the grid"
            out[rr, cc] = obj["color"]

    return out


def _cells_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _big_shape(height: int, width: int, kind: str) -> set[tuple[int, int]]:
    mask = np.ones((height, width), dtype=bool)
    if kind == "cut_corners":
        mask[0, 0] = False
        mask[0, -1] = False
        mask[-1, 0] = False
        mask[-1, -1] = False
    elif kind == "center_hole":
        mask[height // 2, width // 2] = False
    elif kind == "inner_corners":
        mask[1, 1] = False
        mask[1, -2] = False
        mask[-2, 1] = False
        mask[-2, -2] = False
    elif kind != "full":
        raise ValueError(f"unknown big shape kind: {kind}")
    return _cells_from_mask(mask)


SMALL_SHAPES = (
    _cells_from_mask(np.array([[1]], dtype=bool)),
    _cells_from_mask(np.array([[1, 1, 1]], dtype=bool)),
    _cells_from_mask(np.array([[1], [1], [1]], dtype=bool)),
    _cells_from_mask(np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)),
    _cells_from_mask(np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool)),
    _cells_from_mask(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)),
)


def _shape_bbox(shape: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _offset_shape(shape: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r + r0, c + c0) for r, c in shape}


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bboxes_separated(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], gap: int = SEPARATION
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 < br0 - gap or br1 < ar0 - gap or ac1 < bc0 - gap or bc1 < ac0 - gap


def _placement_clear(
    r0: int, c0: int, shape_h: int, shape_w: int, placed_bboxes: list[tuple[int, int, int, int]]
) -> bool:
    bbox = (r0, c0, r0 + shape_h - 1, c0 + shape_w - 1)
    return all(_bboxes_separated(bbox, other) for other in placed_bboxes)


def _sample_big_objects(rng: random.Random, count: int) -> list[dict]:
    height, width = rng.choice(BIG_BBOX_CHOICES)
    big_objects = []
    for _ in range(count):
        kind = rng.choice(BIG_KIND_CHOICES)
        shape = _big_shape(height, width, kind)
        big_objects.append({"shape": shape, "bbox_hw": (height, width), "area": len(shape), "kind": kind})
    return big_objects


def _sample_small_objects(rng: random.Random, count: int) -> list[dict]:
    objects = []
    for _ in range(count):
        shape = rng.choice(SMALL_SHAPES)
        objects.append({"shape": shape, "bbox_hw": _shape_bbox(shape), "area": len(shape)})
    return objects


def _place_big_objects(
    rng: random.Random, height: int, width: int, big_objects: list[dict]
) -> list[dict] | None:
    placed_bboxes: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for obj in big_objects:
        box_h, box_w = obj["bbox_hw"]
        candidates = [(r, c) for r in range(height - box_h + 1) for c in range(width - box_w + 1)]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _placement_clear(r0, c0, box_h, box_w, placed_bboxes):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None

        cells = _offset_shape(obj["shape"], chosen[0], chosen[1])
        bbox = _bbox_from_cells(cells)
        placed_bboxes.append(bbox)
        placed.append({**obj, "cells": cells, "bbox": bbox, "center": _bbox_center(bbox)})

    return placed


def _candidate_small_positions(
    height: int,
    width: int,
    shape_h: int,
    shape_w: int,
    placed_bboxes: list[tuple[int, int, int, int]],
    target_idx: int,
    big_centers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r0 in range(height - shape_h + 1):
        for c0 in range(width - shape_w + 1):
            if not _placement_clear(r0, c0, shape_h, shape_w, placed_bboxes):
                continue

            bbox = (r0, c0, r0 + shape_h - 1, c0 + shape_w - 1)
            center = _bbox_center(bbox)
            dists = [(center[0] - rr) ** 2 + (center[1] - cc) ** 2 for rr, cc in big_centers]
            best = min(dists)
            if dists.count(best) != 1 or int(dists.index(best)) != target_idx:
                continue
            if abs(center[0] - big_centers[target_idx][0]) + abs(center[1] - big_centers[target_idx][1]) < MIN_MOVE_DISTANCE:
                continue
            candidates.append((r0, c0))
    return candidates


def _place_small_objects(
    rng: random.Random,
    height: int,
    width: int,
    small_objects: list[dict],
    big_centers: list[tuple[int, int]],
    placed_bboxes: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    target_order = rng.sample(range(len(big_centers)), k=len(small_objects))
    rng.shuffle(target_order)

    placed = []
    current_bboxes = list(placed_bboxes)
    for obj, target_idx in zip(small_objects, target_order):
        box_h, box_w = obj["bbox_hw"]
        candidates = _candidate_small_positions(height, width, box_h, box_w, current_bboxes, target_idx, big_centers)
        if not candidates:
            return None

        r0, c0 = rng.choice(candidates)
        cells = _offset_shape(obj["shape"], r0, c0)
        bbox = _bbox_from_cells(cells)
        current_bboxes.append(bbox)
        placed.append({**obj, "cells": cells, "bbox": bbox, "center": _bbox_center(bbox), "target_idx": target_idx})

    return placed


def _render_input(
    height: int, width: int, bg: int, big_objects: list[dict], small_objects: list[dict], colors: list[int]
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, obj in zip(colors, big_objects + small_objects):
        for r, c in obj["cells"]:
            grid[r, c] = color
    return grid


def _expected_output(
    input_grid: np.ndarray, bg: int, big_objects: list[dict], small_objects: list[dict], colors: list[int]
) -> np.ndarray:
    out = np.full(input_grid.shape, bg, dtype=int)
    big_colors = colors[: len(big_objects)]
    small_colors = colors[len(big_objects) :]

    for color, obj in zip(big_colors, big_objects):
        r0, c0, r1, c1 = obj["bbox"]
        cr, cc = obj["center"]
        out[cr, c0 : c1 + 1] = color
        out[r0 : r1 + 1, cc] = color

    big_centers = [obj["center"] for obj in big_objects]
    for color, obj in zip(small_colors, small_objects):
        src_r, src_c = obj["center"]
        dst_r, dst_c = big_centers[obj["target_idx"]]
        dr = dst_r - src_r
        dc = dst_c - src_c
        for r, c in obj["cells"]:
            out[r + dr, c + dc] = color

    return out


def _class_split_matches_design(grid: np.ndarray, big_count: int, small_count: int) -> bool:
    bg = _background_color(grid)
    objects = _find_objects(grid, bg)
    if len(objects) != big_count + small_count:
        return False
    small, big = _split_objects_by_area(objects)
    return len(big) == big_count and len(small) == small_count


def _generation_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    big_count = rng.choice((2, 3))
    small_count = 2 if big_count == 2 else rng.choice((2, 3))

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    colors = palette[1 : 1 + big_count + small_count]

    big_objects = _sample_big_objects(rng, big_count)
    small_objects = _sample_small_objects(rng, small_count)

    max_small_area = max(obj["area"] for obj in small_objects)
    min_big_area = min(obj["area"] for obj in big_objects)
    if min_big_area < 4 * max_small_area:
        return None

    placed_big = _place_big_objects(rng, height, width, big_objects)
    if placed_big is None:
        return None

    placed_small = _place_small_objects(
        rng,
        height,
        width,
        small_objects,
        [obj["center"] for obj in placed_big],
        [obj["bbox"] for obj in placed_big],
    )
    if placed_small is None:
        return None

    grid = _render_input(height, width, bg, placed_big, placed_small, colors)
    if Counter(grid.ravel()).most_common(1)[0][0] != bg:
        return None
    if not _class_split_matches_design(grid, big_count, small_count):
        return None

    expected = _expected_output(grid, bg, placed_big, placed_small, colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
