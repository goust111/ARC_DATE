from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Box = tuple[int, int, int, int]

DIR_UP = "up"
DIR_RIGHT = "right"
DIR_DOWN = "down"
DIR_LEFT = "left"
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator bounds.
GRID_HEIGHT_RANGE = (11, 18)
GRID_WIDTH_RANGE = (11, 18)
REGION_COUNT_RANGE = (3, 6)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (2, 5)
MIN_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: list[Coord]) -> Box:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            parts.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return parts


def _touches_grid_corner(box: Box, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = map(int, shape)
    return (r0, c0, r1, c1) in {
        (0, 0, 1, 1),
        (0, w - 2, 1, w - 1),
        (h - 2, 0, h - 1, 1),
        (h - 2, w - 2, h - 1, w - 1),
    }


def _marker_direction(component: dict, shape: tuple[int, int]) -> str:
    assert component["area"] == 3
    r0, c0, r1, c1 = component["bbox"]
    assert r1 - r0 == 1 and c1 - c0 == 1
    assert _touches_grid_corner(component["bbox"], shape)

    local = {(r - r0, c - c0) for r, c in component["coords"]}
    elbow = None
    for cell in local:
        neighbors = sum((nr, nc) in local for nr, nc in _neighbors4(*cell))
        if neighbors == 2:
            elbow = cell
            break
    assert elbow is not None, "marker must be a 3-cell L"

    mapping = {
        (0, 0): DIR_UP,
        (0, 1): DIR_RIGHT,
        (1, 1): DIR_DOWN,
        (1, 0): DIR_LEFT,
    }
    assert elbow in mapping
    return mapping[elbow]


def _is_filled_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return component["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _region_order_key(direction: str, component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    if direction == DIR_UP:
        return r0, c0
    if direction == DIR_RIGHT:
        return c0, r0
    if direction == DIR_DOWN:
        return -r1, c0
    if direction == DIR_LEFT:
        return -c1, r0
    raise AssertionError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate rectangle colors according to the scan order selected by a corner L-marker.

    Input structure:
    - The dominant color is the background.
    - One corner contains a 3-cell L-shaped indicator.
    - Every other non-background object is one solid monochrome rectangle, and each such
      color appears on exactly one rectangle.

    Rule:
    - Read the indicator direction from the L elbow: top-left -> up, top-right -> right,
      bottom-right -> down, bottom-left -> left.
    - Order the rectangles by the corresponding scan order.
    - Keep every rectangle in place, but recolor rectangle k with rectangle k+1's color
      cyclically.
    - Leave the indicator unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    components = _component_records(g, bg)
    assert len(components) >= 4, "expected one marker and multiple rectangles"

    marker_candidates = [
        component
        for component in components
        if component["area"] == 3
        and component["bbox"][2] - component["bbox"][0] == 1
        and component["bbox"][3] - component["bbox"][1] == 1
        and _touches_grid_corner(component["bbox"], g.shape)
    ]
    assert len(marker_candidates) == 1, "expected exactly one corner L marker"
    marker = marker_candidates[0]
    direction = _marker_direction(marker, g.shape)

    regions = [component for component in components if component is not marker]
    assert len(regions) >= 2, "expected at least two rectangles to rotate"
    assert len({component["color"] for component in regions}) == len(regions), "rectangle colors must be unique"
    for component in regions:
        assert _is_filled_rectangle(component), "every non-marker object must be a solid rectangle"

    ordered = sorted(regions, key=lambda component: _region_order_key(direction, component))

    out = g.copy()
    for i, component in enumerate(ordered):
        new_color = int(ordered[(i + 1) % len(ordered)]["color"])
        rows, cols = zip(*component["coords"])
        out[list(rows), list(cols)] = new_color
    return out


def _boxes_overlap(a: Box, b: Box) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_box(box: Box, gap: int) -> Box:
    r0, c0, r1, c1 = box
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _marker_cells(direction: str, corner: str, shape: tuple[int, int]) -> list[Coord]:
    local_by_direction = {
        DIR_UP: [(0, 0), (0, 1), (1, 0)],
        DIR_RIGHT: [(0, 0), (0, 1), (1, 1)],
        DIR_DOWN: [(0, 1), (1, 0), (1, 1)],
        DIR_LEFT: [(0, 0), (1, 0), (1, 1)],
    }
    h, w = map(int, shape)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - 2, 0
    elif corner == "br":
        r0, c0 = h - 2, w - 2
    else:
        raise AssertionError(corner)
    return [(r0 + dr, c0 + dc) for dr, dc in local_by_direction[direction]]


def _marker_box(corner: str, shape: tuple[int, int]) -> Box:
    h, w = map(int, shape)
    if corner == "tl":
        return 0, 0, 1, 1
    if corner == "tr":
        return 0, w - 2, 1, w - 1
    if corner == "bl":
        return h - 2, 0, h - 1, 1
    if corner == "br":
        return h - 2, w - 2, h - 1, w - 1
    raise AssertionError(corner)


def _paint_regions(
    shape: tuple[int, int],
    bg: int,
    marker_color: int,
    marker_coords: list[Coord],
    regions: list[dict],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for r, c in marker_coords:
        grid[r, c] = marker_color
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = region["color"]
    return grid


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    marker_color: int,
    marker_coords: list[Coord],
    direction: str,
    regions: list[dict],
) -> np.ndarray:
    out = _paint_regions(shape, bg, marker_color, marker_coords, regions)
    ordered = sorted(regions, key=lambda region: _region_order_key(direction, region))
    for i, region in enumerate(ordered):
        color = int(ordered[(i + 1) % len(ordered)]["color"])
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _sample_rectangle(shape: tuple[int, int], existing: list[Box], forbidden: list[Box]) -> Box | None:
    h, w = map(int, shape)
    rect_h = random.randint(RECT_HEIGHT_RANGE[0], min(RECT_HEIGHT_RANGE[1], h - 3))
    rect_w = random.randint(RECT_WIDTH_RANGE[0], min(RECT_WIDTH_RANGE[1], w - 3))
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    padded = _expand_box(box, MIN_GAP)
    for other in existing + forbidden:
        if _boxes_overlap(padded, other):
            return None
    return box


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    shape = (height, width)

    direction = random.choice([DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT])
    corner = random.choice(["tl", "tr", "bl", "br"])
    region_count = random.randint(*REGION_COUNT_RANGE)

    if region_count + 2 > 10:
        return None
    palette = random.sample(range(10), region_count + 2)
    bg = palette[0]
    marker_color = palette[1]
    region_colors = palette[2:]

    marker_coords = _marker_cells(direction, corner, shape)
    marker_box = _marker_box(corner, shape)
    forbidden = [_expand_box(marker_box, MIN_GAP)]

    regions: list[dict] = []
    boxes: list[Box] = []
    total_area = 0
    for color in region_colors:
        box = None
        for _ in range(80):
            box = _sample_rectangle(shape, boxes, forbidden)
            if box is None:
                continue
            area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            if total_area + area >= (height * width) // 2:
                box = None
                continue
            break
        if box is None:
            return None
        boxes.append(box)
        total_area += (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        regions.append({"color": color, "bbox": box})

    grid = _paint_regions(shape, bg, marker_color, marker_coords, regions)
    color_counts = Counter(map(int, grid.reshape(-1)))
    if color_counts[bg] <= max(count for color, count in color_counts.items() if color != bg):
        return None

    expected = _expected_output(shape, bg, marker_color, marker_coords, direction, regions)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
