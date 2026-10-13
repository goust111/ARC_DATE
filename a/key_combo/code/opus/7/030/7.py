from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NONZERO_COLORS = list(range(1, 10))
INNER_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
]

GRID_MIN = 16
GRID_MAX = 24
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
OBJECT_COUNTS = [2, 2, 3]
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _ring_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_rectangular_ring(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return np.array_equal(mask, _ring_mask(mask.shape, (r0, c0, r1, c1)))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _detect_structures(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground color"

    frame_boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        mask = g == color
        if _is_rectangular_ring(mask):
            frame_boxes[color] = _bbox(mask)

    assert frame_boxes, "expected at least one rectangular frame color"

    inner_to_frames: dict[int, list[int]] = {}
    for color in colors:
        if color in frame_boxes:
            continue
        inner_box = _bbox(g == color)
        owners = [frame_color for frame_color, frame_box in frame_boxes.items() if _strictly_contains(frame_box, inner_box)]
        assert len(owners) == 1, f"inner color {color} must belong to exactly one frame"
        inner_to_frames[color] = owners

    frame_to_inners: dict[int, list[int]] = {frame_color: [] for frame_color in frame_boxes}
    for inner_color, owners in inner_to_frames.items():
        frame_to_inners[owners[0]].append(inner_color)

    structures = []
    for frame_color, inner_colors in frame_to_inners.items():
        assert len(inner_colors) == 1, f"frame color {frame_color} must contain exactly one inner color"
        inner_color = inner_colors[0]
        inner_mask = g == inner_color
        inner_box = _bbox(inner_mask)
        r0, c0, r1, c1 = frame_boxes[frame_color]
        assert 0 < r0 and 0 < c0 and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1], "frame needs output margin"
        assert 1 <= int(inner_mask.sum()) <= 3, "inner foreground must contain 1..3 pixels"
        structures.append(
            {
                "frame_color": int(frame_color),
                "inner_color": int(inner_color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": inner_box,
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    assert structures, "expected at least one frame/inner structure"
    return bg, structures


def _expand_frames(base: np.ndarray, structures: list[dict[str, int | tuple[int, int, int, int]]], bg: int) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    for structure in structures:
        frame_color = int(structure["frame_color"])
        r0, c0, r1, c1 = structure["bbox"]
        out[r0 - 1, c0 : c1 + 1] = frame_color
        out[r1 + 1, c0 : c1 + 1] = frame_color
        out[r0 : r1 + 1, c0 - 1] = frame_color
        out[r0 : r1 + 1, c1 + 1] = frame_color
        out[r0 - 1, c0 - 1] = bg
        out[r0 - 1, c1 + 1] = bg
        out[r1 + 1, c0 - 1] = bg
        out[r1 + 1, c1 + 1] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each framed object by one cell and paint its diagonal corner gaps with the inner color.

    Input structure:
    - The grid has black background and 2-3 separated two-color objects.
    - In each object, one color forms a one-pixel rectangular ring.
    - A different inner color appears as a 1-3 pixel cluster strictly inside that ring.

    Rule:
    - For every rectangular ring, add a one-cell outer extrusion on the four sides while leaving the four new outer corners black.
    - Keep the original ring and all existing inner-color pixels unchanged.
    - After extrusion, recolor each remaining background pixel that is diagonally adjacent to that object's original ring color but not orthogonally adjacent to it; this paints the four new corner gaps with the inner color.

    Key insight: The rectangular ring identifies both the extrusion boundary and the diagonal-only corner targets, while the enclosed color supplies the paint used for those new corner cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, structures = _detect_structures(g)

    out = _expand_frames(g, structures, bg)
    ordered = sorted(structures, key=lambda item: int(item["area"]))
    for structure in ordered:
        frame_color = int(structure["frame_color"])
        inner_color = int(structure["inner_color"])
        frame_mask = g == frame_color

        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if int(out[r, c]) != bg:
                    continue
                has_diagonal = any(
                    _in_bounds(g.shape, r + dr, c + dc) and bool(frame_mask[r + dr, c + dc])
                    for dr, dc in DIAGONALS
                )
                if not has_diagonal:
                    continue
                has_orthogonal = any(
                    _in_bounds(g.shape, r + dr, c + dc) and bool(frame_mask[r + dr, c + dc])
                    for dr, dc in ORTHOGONALS
                )
                if not has_orthogonal:
                    out[r, c] = inner_color

    return out


def _shape_size(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _choose_inner_cells(frame_h: int, frame_w: int) -> list[tuple[int, int]] | None:
    feasible = []
    for shape in INNER_SHAPES:
        shape_h, shape_w = _shape_size(shape)
        if shape_h <= frame_h - 2 and shape_w <= frame_w - 2:
            feasible.append(shape)
    if not feasible:
        return None

    shape = random.choice(feasible)
    shape_h, shape_w = _shape_size(shape)
    r_off = random.randint(1, frame_h - shape_h - 1)
    c_off = random.randint(1, frame_w - shape_w - 1)
    return [(r_off + r, c_off + c) for r, c in shape]


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _paint_object(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    inner_color: int,
    inner_cells: list[tuple[int, int]],
) -> None:
    ring = _ring_mask(grid.shape, bbox)
    grid[ring] = frame_color
    r0, c0, _r1, _c1 = bbox
    for dr, dc in inner_cells:
        grid[r0 + dr, c0 + dc] = inner_color


def _build_candidate_grid() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    n_objects = random.choice(OBJECT_COUNTS)
    if 2 * n_objects > len(NONZERO_COLORS):
        return None

    color_pool = random.sample(NONZERO_COLORS, 2 * n_objects)
    random.shuffle(color_pool)
    pairs = [(color_pool[2 * i], color_pool[2 * i + 1]) for i in range(n_objects)]

    grid = np.zeros((height, width), dtype=int)
    occupied_output_boxes: list[tuple[int, int, int, int]] = []

    for frame_color, inner_color in pairs:
        placed = False
        for _ in range(120):
            frame_h = random.randint(*FRAME_H_RANGE)
            frame_w = random.randint(*FRAME_W_RANGE)
            if frame_h + 2 > height or frame_w + 2 > width:
                continue

            r0 = random.randint(1, height - frame_h - 1)
            c0 = random.randint(1, width - frame_w - 1)
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            output_box = (r0 - 1, c0 - 1, r0 + frame_h, c0 + frame_w)
            if any(not _separated(output_box, other) for other in occupied_output_boxes):
                continue

            inner_cells = _choose_inner_cells(frame_h, frame_w)
            if inner_cells is None:
                continue

            _paint_object(grid, bbox, frame_color, inner_color, inner_cells)
            occupied_output_boxes.append(output_box)
            placed = True
            break

        if not placed:
            return None

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    bg, structures = _detect_structures(grid)
    if bg != 0 or not (2 <= len(structures) <= 3):
        return False

    out = main(grid)
    for structure in structures:
        inner_color = int(structure["inner_color"])
        r0, c0, r1, c1 = structure["bbox"]
        corners = [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]
        if any(int(out[r, c]) != inner_color for r, c in corners):
            return False

    delta = int(np.count_nonzero(out != grid))
    expected = sum(2 * ((s["bbox"][2] - s["bbox"][0] + 1) + (s["bbox"][3] - s["bbox"][1] + 1)) + 4 for s in structures)
    return delta == expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
