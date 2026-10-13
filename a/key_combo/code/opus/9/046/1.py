from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
RING_COLORS = (1, 3, 5, 8)

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _bbox_of_color(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        raise ValueError(f"color {color} not present")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _point_on_border(point: Coord, box: BBox) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1 and (r in (r0, r1) or c in (c0, c1))


def _infer_rings(grid: np.ndarray) -> list[dict[str, object]]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, RED)]
    rings = [{"color": color, "bbox": _bbox_of_color(grid, color)} for color in colors]
    rings.sort(key=lambda item: (-_bbox_area(item["bbox"]), item["bbox"]))
    return rings


def _marker_depth(marker: Coord, rings: list[dict[str, object]]) -> int:
    matches = [depth for depth, ring in enumerate(rings, start=1) if _point_on_border(marker, ring["bbox"])]
    if len(matches) != 1:
        raise ValueError(f"marker {marker} must lie on exactly one ring border, got {matches}")
    return matches[0]


def _ray_dirs_for_depth(depth: int) -> tuple[Coord, Coord]:
    return ((0, -1), (0, 1)) if depth % 2 == 1 else ((-1, 0), (1, 0))


def _paint_rays(grid: np.ndarray, out: np.ndarray, marker: Coord, depth: int) -> set[Coord]:
    h, w = grid.shape
    painted = {marker}
    out[marker] = YELLOW
    for dr, dc in _ray_dirs_for_depth(depth):
        r, c = marker[0] + dr, marker[1] + dc
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BLACK:
            out[r, c] = YELLOW
            painted.add((r, c))
            r += dr
            c += dc
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each red marker on a concentric rectangular ring into a yellow ray source whose axis is set by ring-depth parity.

    Input structure:
    - Black background with 3-4 one-cell-thick concentric rectangular rings in distinct non-red colors.
    - Red marker pixels replace some ring pixels, with at most one marker on any ring.
    - The rings are ordered from the outermost rectangle to the innermost rectangle by nesting depth.

    Rule:
    - Identify every concentric ring and assign depth 1 to the outermost, depth 2 to the next, and so on.
    - For each red marker, find the unique ring border that contains it and read that ring's depth.
    - If the depth is odd, recolor the marker yellow and cast yellow rays left and right; if the depth is even, cast yellow rays up and down.
    - Each ray advances only through black cells and stops just before the first non-black cell or the grid boundary.

    Key insight:
    - The marker's local position on the ring does not determine the ray direction; only the enclosing ring's depth parity does, so the nested-ring structure must be recovered first.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    out = g.copy()
    rings = _infer_rings(g)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]

    for marker in markers:
        depth = _marker_depth(marker, rings)
        _paint_rays(g, out, marker, depth)
    return out


def _draw_ring(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _ring_boxes(top: int, left: int, outer_h: int, outer_w: int, layers: int) -> list[BBox]:
    boxes = []
    for depth in range(layers):
        inset = 2 * depth
        boxes.append((top + inset, left + inset, top + outer_h - 1 - inset, left + outer_w - 1 - inset))
    return boxes


def _corner_coord(box: BBox, name: str) -> Coord:
    r0, c0, r1, c1 = box
    lookup = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    return lookup[name]


def _pick_from_line(values: list[int], mode: str) -> int:
    if mode == "near_start":
        return values[0]
    if mode == "near_end":
        return values[-1]
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return values[mid - 1 if random.random() < 0.5 else mid]


def _odd_side_coord(box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    rows = list(range(r0 + 1, r1))
    side = random.choice(["left", "right"])
    mode = random.choice(["near_start", "middle", "near_end"])
    r = _pick_from_line(rows, mode)
    c = c0 if side == "left" else c1
    return r, c


def _even_side_coord(box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    cols = list(range(c0 + 1, c1))
    side = random.choice(["top", "bottom"])
    mode = random.choice(["near_start", "middle", "near_end"])
    r = r0 if side == "top" else r1
    c = _pick_from_line(cols, mode)
    return r, c


def _expected_from_truth(grid: np.ndarray, marker_depths: dict[Coord, int]) -> tuple[np.ndarray, dict[Coord, set[Coord]]]:
    out = grid.copy()
    contributions: dict[Coord, set[Coord]] = {}
    for marker, depth in marker_depths.items():
        contributions[marker] = _paint_rays(grid, out, marker, depth)
    return out, contributions


def _sample_marker_depths(layers: int) -> list[int]:
    options = [depth for depth in range(1, layers + 1)]
    if layers == 3:
        count = random.choices([2, 3], weights=[1, 4])[0]
    else:
        count = random.choices([2, 3, 4], weights=[1, 3, 2])[0]
    for _ in range(50):
        chosen = sorted(random.sample(options, count))
        if any(depth % 2 == 1 for depth in chosen) and any(depth % 2 == 0 for depth in chosen):
            return chosen
    raise RuntimeError("failed to sample mixed-parity marker depths")


def _assign_markers(boxes: list[BBox], marker_depths: list[int]) -> tuple[dict[Coord, int], set[tuple[int, str]]]:
    assignments: dict[Coord, int] = {}
    tags: set[tuple[int, str]] = set()

    odd_depth = random.choice([depth for depth in marker_depths if depth % 2 == 1])
    even_depth = random.choice([depth for depth in marker_depths if depth % 2 == 0])
    shared_corner = random.choice(["tl", "tr", "bl", "br"])

    odd_corner = _corner_coord(boxes[odd_depth - 1], shared_corner)
    even_corner = _corner_coord(boxes[even_depth - 1], shared_corner)
    assignments[odd_corner] = odd_depth
    assignments[even_corner] = even_depth
    tags.add((odd_depth, f"corner:{shared_corner}"))
    tags.add((even_depth, f"corner:{shared_corner}"))

    remaining = [depth for depth in marker_depths if depth not in (odd_depth, even_depth)]
    random.shuffle(remaining)
    for depth in remaining:
        if depth % 2 == 1:
            coord = _odd_side_coord(boxes[depth - 1])
            tag = "odd_side"
        else:
            coord = _even_side_coord(boxes[depth - 1])
            tag = "even_side"
        assignments[coord] = depth
        tags.add((depth, tag))

    return assignments, tags


def _validate_input(
    grid: np.ndarray,
    boxes: list[BBox],
    marker_depths: dict[Coord, int],
    tags: set[tuple[int, str]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    inferred = _infer_rings(grid)
    inferred_boxes = [ring["bbox"] for ring in inferred]
    if inferred_boxes != boxes:
        return False

    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    if len(markers) != len(marker_depths):
        return False

    inferred_depths = {marker: _marker_depth(marker, inferred) for marker in markers}
    if inferred_depths != marker_depths:
        return False

    if len({depth for depth in marker_depths.values()}) != len(marker_depths):
        return False
    if not any(depth % 2 == 1 for depth in marker_depths.values()):
        return False
    if not any(depth % 2 == 0 for depth in marker_depths.values()):
        return False

    corner_tags = [tag for tag in tags if tag[1].startswith("corner:")]
    if len(corner_tags) < 2:
        return False
    same_corner = {}
    for depth, label in corner_tags:
        same_corner.setdefault(label, []).append(depth)
    if not any(len(depths) >= 2 and {depth % 2 for depth in depths} == {0, 1} for depths in same_corner.values()):
        return False

    expected, contributions = _expected_from_truth(grid, marker_depths)
    if not np.array_equal(main(grid), expected):
        return False

    total_extra = 0
    for marker, depth in marker_depths.items():
        extra = len(contributions[marker]) - 1
        if extra < 1:
            return False
        total_extra += extra
        rows = {r for r, _ in contributions[marker]}
        cols = {c for _, c in contributions[marker]}
        if depth % 2 == 1 and len(rows) != 1:
            return False
        if depth % 2 == 0 and len(cols) != 1:
            return False
    if total_extra < len(marker_depths) + 1:
        return False

    if len(marker_depths) >= 3:
        has_odd_side = any(label == "odd_side" for _depth, label in tags)
        has_even_side = any(label == "even_side" for _depth, label in tags)
        if not (has_odd_side or has_even_side):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    layers = random.choice([3, 4])
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 7)
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    height = top_margin + outer_h + bottom_margin
    width = left_margin + outer_w + right_margin
    if max(height, width) > 30:
        return None

    boxes = _ring_boxes(top_margin, left_margin, outer_h, outer_w, layers)
    palette = random.sample(list(RING_COLORS), layers)

    grid = np.zeros((height, width), dtype=int)
    for box, color in zip(boxes, palette):
        _draw_ring(grid, box, color)

    marker_layers = _sample_marker_depths(layers)
    marker_depths, tags = _assign_markers(boxes, marker_layers)
    for marker in marker_depths:
        grid[marker] = RED

    if not _validate_input(grid, boxes, marker_depths, tags):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring ray puzzle")
