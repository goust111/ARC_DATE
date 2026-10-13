import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 30
CONTAINER_COUNT_RANGE = (2, 4)
OUTER_HEIGHT_RANGE = (4, 8)
OUTER_WIDTH_RANGE = (4, 8)
PLACEMENT_MARGIN = 2
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]], bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    frame_mask = np.zeros_like(patch, dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True
    if not np.all(patch[frame_mask] == color):
        return False
    if not np.all(patch[1:-1, 1:-1] == bg):
        return False
    return True


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components_mask(grid == color):
            if not _is_rect_frame(grid, color, comp, bg):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            containers.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "interior": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
                }
            )
    assert containers, "expected at least one hollow rectangular container"
    return containers


def _target_container_index(
    containers: list[dict[str, object]], p0: tuple[int, int], p1: tuple[int, int]
) -> int:
    if p0[0] == p1[0]:
        row = p0[0]
        c_lo, c_hi = sorted((p0[1], p1[1]))
        hits = []
        for idx, container in enumerate(containers):
            r0, c0, r1, c1 = container["bbox"]
            if r0 < row < r1 and max(c_lo + 1, c0 + 1) <= min(c_hi - 1, c1 - 1):
                hits.append(idx)
    else:
        col = p0[1]
        r_lo, r_hi = sorted((p0[0], p1[0]))
        hits = []
        for idx, container in enumerate(containers):
            r0, c0, r1, c1 = container["bbox"]
            if c0 < col < c1 and max(r_lo + 1, r0 + 1) <= min(r_hi - 1, r1 - 1):
                hits.append(idx)
    assert len(hits) == 1, "each color pair must cross exactly one container interior"
    return int(hits[0])


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    containers = _find_containers(g, bg)
    frame_colors = {int(container["color"]) for container in containers}
    assert len(frame_colors) == 1, "all containers must share one frame color"
    frame_color = next(iter(frame_colors))

    out = g.copy()
    pair_targets: dict[int, int] = {}
    for color in [int(v) for v in np.unique(g) if int(v) not in {bg, frame_color}]:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each non-frame color must appear exactly twice"
        p0 = tuple(int(x) for x in coords[0])
        p1 = tuple(int(x) for x in coords[1])
        assert p0[0] == p1[0] or p0[1] == p1[1], "same-color markers must share a row or column"
        idx = _target_container_index(containers, p0, p1)
        assert idx not in pair_targets.values(), "each container must be targeted by at most one color pair"
        pair_targets[color] = idx
        interior_rows, interior_cols = containers[idx]["interior"]
        out[interior_rows, interior_cols] = color

    return out, containers, pair_targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each same-color pixel pair as a row or column pointer to fill the unique hollow container that its connecting line crosses.

    Input structure:
    - The background is black.
    - One or more grey one-pixel-thick rectangular frames form hollow containers whose interiors start as background.
    - Several other colors appear as isolated marker pairs, with exactly two pixels per color.
    - Each same-color pair lies on a common row or column, and the segment between them crosses exactly one container interior.

    Rule:
    - Detect every hollow rectangular grey container.
    - For each non-grey color, find its two marker pixels and determine whether they define a horizontal or vertical pointer line.
    - Identify the unique container interior intersected by that line segment between the markers.
    - Fill the entire interior of that container with the marker color, while leaving the marker pixels and grey borders unchanged.

    Key insight:
    - The colored pixels must be grouped by color first, then interpreted geometrically: the pair defines a hidden pointer, and the container it pierces determines where that color should be applied.
    """
    out, _containers, _pair_targets = _solve(grid)
    return out


def _inflate(bbox: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_orientations(count: int) -> list[str]:
    orientations = [random.choice(("h", "v")) for _ in range(count)]
    if count >= 2 and len(set(orientations)) == 1 and random.random() < 0.75:
        orientations[0] = "h"
        orientations[1] = "v"
        random.shuffle(orientations)
    return orientations


def _place_containers(
    height: int, width: int, outer_shapes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    inflated: list[tuple[int, int, int, int]] = []
    for box_h, box_w in outer_shapes:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - box_h - 1)
            c0 = random.randint(1, width - box_w - 1)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_boxes_overlap(_inflate(bbox, PLACEMENT_MARGIN), prev) for prev in inflated):
                continue
            boxes.append(bbox)
            inflated.append(_inflate(bbox, PLACEMENT_MARGIN))
            placed = True
            break
        if not placed:
            return None
    return boxes


def _build_input_from_plan(
    height: int,
    width: int,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    orientations: list[str],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], np.ndarray]:
    grid = np.full((height, width), BLACK, dtype=int)
    expected = grid.copy()
    color_to_bbox: dict[int, tuple[int, int, int, int]] = {}

    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        expected[r0, c0 : c1 + 1] = GREY
        expected[r1, c0 : c1 + 1] = GREY
        expected[r0 : r1 + 1, c0] = GREY
        expected[r0 : r1 + 1, c1] = GREY

    used_markers: set[tuple[int, int]] = set()
    for bbox, color, orientation in zip(boxes, colors, orientations):
        r0, c0, r1, c1 = bbox
        if orientation == "h":
            row = random.randint(r0 + 1, r1 - 1)
            left = (row, c0 - 1)
            right = (row, c1 + 1)
            markers = (left, right)
        else:
            col = random.randint(c0 + 1, c1 - 1)
            top = (r0 - 1, col)
            bottom = (r1 + 1, col)
            markers = (top, bottom)

        assert all(grid[r, c] == BLACK for r, c in markers), "marker cells must stay on background"
        assert not any(marker in used_markers for marker in markers), "marker cells must be unique"
        used_markers.update(markers)

        for r, c in markers:
            grid[r, c] = color
            expected[r, c] = color
        expected[r0 + 1 : r1, c0 + 1 : c1] = color
        color_to_bbox[color] = bbox

    return grid, color_to_bbox, expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    count = random.randint(*CONTAINER_COUNT_RANGE)
    outer_shapes = [
        (
            random.randint(*OUTER_HEIGHT_RANGE),
            random.randint(*OUTER_WIDTH_RANGE),
        )
        for _ in range(count)
    ]

    total_padded_area = sum((h + 2 * PLACEMENT_MARGIN) * (w + 2 * PLACEMENT_MARGIN) for h, w in outer_shapes)
    if total_padded_area > height * width:
        return None

    boxes = _place_containers(height, width, outer_shapes)
    if boxes is None:
        return None

    colors = random.sample(COLOR_POOL, count)
    orientations = _sample_orientations(count)
    grid, color_to_bbox, expected = _build_input_from_plan(height, width, boxes, colors, orientations)

    try:
        solved, containers, pair_targets = _solve(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if len(containers) != count or len(pair_targets) != count:
        return None

    recovered = {
        color: tuple(int(x) for x in containers[idx]["bbox"])
        for color, idx in pair_targets.items()
    }
    if recovered != color_to_bbox:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
