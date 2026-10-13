from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MARKER_COLORS = (BLUE, RED, GREEN)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _nearest_marker_color(coords: np.ndarray, markers: list[tuple[int, int, int]]) -> int | None:
    best_distance: int | None = None
    best_colors: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for mr, mc, color in markers:
            distance = (rr - int(mr)) ** 2 + (cc - int(mc)) ** 2
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_colors = {int(color)}
            elif distance == best_distance:
                best_colors.add(int(color))
    if len(best_colors) != 1:
        return None
    return next(iter(best_colors))


def _expanded_perimeter(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    cells: list[tuple[int, int]] = []
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                cells.append((r, c))
    return cells


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object by its nearest marker, erase the markers, then frame the recolored components by merged area tier.

    Input structure:
    - The grid is a black background with several small gray 4-connected objects.
    - A few isolated single-pixel markers use blue, red, and/or green.
    - Gray objects are separate in the input, while markers do not belong to those objects.

    Rule:
    - For each gray 4-connected object, find the minimum squared Euclidean distance from any object pixel to any marker pixel; recolor the whole object to that marker color.
    - Erase all marker pixels to black.
    - Recompute monochromatic 4-connected components on the recolored grid.
    - Leave area-1 components unframed; frame area-2 or area-3 components with a green perimeter of the bbox expanded by one cell, and frame area-4-or-larger components the same way in yellow.
    - When frame perimeters overlap, larger-area components override smaller ones; equal areas override in increasing bbox top-left order.

    Key insight:
    - The outline color depends on the post-recolor component area, so marker proximity indirectly controls both recoloring and which frame tier appears.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    markers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(np.isin(grid, MARKER_COLORS))
    ]
    gray_objects = _components_of_color(grid, GRAY)
    assert markers, "expected at least one marker"
    assert gray_objects, "expected at least one gray object"

    recolored = grid.copy()
    for coords in gray_objects:
        color = _nearest_marker_color(coords, markers)
        assert color is not None, "nearest marker color must be unambiguous"
        recolored[coords[:, 0], coords[:, 1]] = int(color)

    for r, c, _color in markers:
        recolored[r, c] = BLACK

    base = recolored.copy()
    frame_jobs: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(base) if int(v) != BLACK):
        for coords in _components_of_color(base, color):
            area = int(coords.shape[0])
            if area <= 1:
                continue
            border_color = GREEN if area <= 3 else YELLOW
            box = _bbox(coords)
            paintable = [
                (r, c)
                for r, c in _expanded_perimeter(box, base.shape)
                if int(base[r, c]) == BLACK
            ]
            if paintable:
                frame_jobs.append((area, box[0], box[1], border_color, paintable))

    out = base.copy()
    frame_jobs.sort(key=lambda item: (item[0], item[1], item[2]))
    for _area, _r0, _c0, border_color, paintable in frame_jobs:
        for r, c in paintable:
            out[r, c] = int(border_color)
    return out


SHAPES_BY_AREA: dict[int, list[tuple[tuple[int, int], ...]]] = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
    ],
}

ZONE_KEYS = ("tl", "tr", "bl", "br", "tc", "bc")
GEN_MAX_ATTEMPTS = 500


def _zone_layouts(h: int, w: int) -> dict[str, dict[str, tuple[int, int, int, int] | list[tuple[int, int]]]]:
    top_band = max(2, h // 2 - 1)
    left_band = max(2, w // 2 - 1)
    bottom_band = min(h - 3, h // 2)
    right_band = min(w - 3, w // 2)
    center_left = max(2, w // 3)
    center_right = min(w - 3, (2 * w) // 3)
    center_mid = w // 2
    return {
        "tl": {
            "marker_cells": [(0, 0), (0, 1), (1, 0), (1, 1)],
            "main": (2, 2, top_band, left_band),
            "tight": (2, 2, min(top_band, 6), min(left_band, 6)),
        },
        "tr": {
            "marker_cells": [(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)],
            "main": (2, right_band, top_band, w - 3),
            "tight": (2, max(right_band, w - 7), min(top_band, 6), w - 3),
        },
        "bl": {
            "marker_cells": [(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)],
            "main": (bottom_band, 2, h - 3, left_band),
            "tight": (max(bottom_band, h - 7), 2, h - 3, min(left_band, 6)),
        },
        "br": {
            "marker_cells": [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)],
            "main": (bottom_band, right_band, h - 3, w - 3),
            "tight": (max(bottom_band, h - 7), max(right_band, w - 7), h - 3, w - 3),
        },
        "tc": {
            "marker_cells": [(0, center_mid - 1), (0, center_mid), (1, center_mid - 1), (1, center_mid)],
            "main": (2, center_left, top_band, center_right),
            "tight": (2, max(2, center_mid - 2), min(top_band, 6), min(w - 3, center_mid + 2)),
        },
        "bc": {
            "marker_cells": [(h - 2, center_mid - 1), (h - 2, center_mid), (h - 1, center_mid - 1), (h - 1, center_mid)],
            "main": (bottom_band, center_left, h - 3, center_right),
            "tight": (max(bottom_band, h - 7), max(2, center_mid - 2), h - 3, min(w - 3, center_mid + 2)),
        },
    }


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> np.ndarray:
    return np.array([(r0 + r, c0 + c) for r, c in shape], dtype=int)


def _bbox_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    row_gap = max(box_a[0] - box_b[2] - 1, box_b[0] - box_a[2] - 1, 0)
    col_gap = max(box_a[1] - box_b[3] - 1, box_b[1] - box_a[3] - 1, 0)
    return int(row_gap + col_gap)


def _bbox_within(box: tuple[int, int, int, int], region: tuple[int, int, int, int]) -> bool:
    return box[0] >= region[0] and box[1] >= region[1] and box[2] <= region[2] and box[3] <= region[3]


def _can_place_object(grid: np.ndarray, coords: np.ndarray) -> bool:
    h, w = grid.shape
    shape_cells = {tuple(map(int, coord)) for coord in coords}
    for r, c in shape_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape_cells and int(grid[nr, nc]) == GRAY:
                return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in MARKER_COLORS:
                    return False
    return True


def _placement_candidates(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    target_color: int,
    markers: list[tuple[int, int, int]],
    region: tuple[int, int, int, int] | None,
) -> list[np.ndarray]:
    h, w = grid.shape
    _r0, _c0, r1, c1 = _shape_bbox(shape)
    height = r1 + 1
    width = c1 + 1
    candidates: list[np.ndarray] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            coords = _translate_shape(shape, top, left)
            box = _bbox(coords)
            if region is not None and not _bbox_within(box, region):
                continue
            if not _can_place_object(grid, coords):
                continue
            if _nearest_marker_color(coords, markers) != int(target_color):
                continue
            candidates.append(coords)
    return candidates


def _same_color_close_pairs(grid: np.ndarray) -> int:
    markers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(np.isin(grid, MARKER_COLORS))
    ]
    gray_objects = _components_of_color(grid, GRAY)
    pairs = 0
    labeled = [(_nearest_marker_color(coords, markers), _bbox(coords)) for coords in gray_objects]
    for i in range(len(labeled)):
        color_i, box_i = labeled[i]
        for j in range(i + 1, len(labeled)):
            color_j, box_j = labeled[j]
            if color_i is not None and color_i == color_j and _bbox_gap(box_i, box_j) <= 1:
                pairs += 1
    return pairs


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    grid = np.zeros((h, w), dtype=int)

    layouts = _zone_layouts(h, w)
    active_colors = random.sample(MARKER_COLORS, k=random.choice((2, 3)))
    zone_keys = random.sample(ZONE_KEYS, k=len(active_colors))
    color_regions: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    markers: list[tuple[int, int, int]] = []
    for color, zone_key in zip(active_colors, zone_keys):
        zone = layouts[zone_key]
        mr, mc = random.choice(zone["marker_cells"])  # type: ignore[index]
        grid[mr, mc] = int(color)
        markers.append((int(mr), int(mc), int(color)))
        color_regions[int(color)] = {
            "main": zone["main"],  # type: ignore[assignment]
            "tight": zone["tight"],  # type: ignore[assignment]
        }

    n_objects = random.randint(5, 8)
    cluster_color = random.choice(active_colors)
    other_colors = [color for color in active_colors if color != cluster_color]
    cluster_count = 3 if n_objects >= 6 else 2

    specs: list[tuple[int, int, str]] = [(cluster_color, 4, "tight")]
    specs.append((cluster_color, random.choice((2, 3)), "tight"))
    if cluster_count == 3:
        specs.append((cluster_color, random.choice((2, 3)), "tight"))
    if other_colors:
        specs.append((random.choice(other_colors), random.choice((2, 3, 4)), "main"))
    while len(specs) < n_objects:
        specs.append((random.choice(active_colors), random.choice((2, 3, 4)), "main"))
    random.shuffle(specs[cluster_count + 1 :])
    specs.sort(key=lambda item: (item[2] != "tight", -item[1], random.random()))

    placed_boxes: list[tuple[int, int, int, int]] = []
    cluster_boxes: list[tuple[int, int, int, int]] = []
    for target_color, area, region_kind in specs:
        shape = random.choice(SHAPES_BY_AREA[int(area)])
        regions = [color_regions[int(target_color)][region_kind]]
        if region_kind == "tight":
            regions.append(color_regions[int(target_color)]["main"])
        regions.append(None)

        candidates: list[np.ndarray] = []
        for region in regions:
            candidates = _placement_candidates(grid, shape, target_color, markers, region)
            if candidates:
                break
        if not candidates:
            return None

        random.shuffle(candidates)
        if region_kind == "tight" and cluster_boxes:
            candidates.sort(key=lambda coords: min(_bbox_gap(_bbox(coords), box) for box in cluster_boxes))
            chosen = candidates[0]
        else:
            chosen = candidates[0]

        grid[chosen[:, 0], chosen[:, 1]] = GRAY
        box = _bbox(chosen)
        placed_boxes.append(box)
        if int(target_color) == int(cluster_color):
            cluster_boxes.append(box)

    out = main(grid)
    if GREEN not in out or YELLOW not in out:
        return None
    if len(set(int(v) for v in np.unique(grid) if int(v) in MARKER_COLORS)) < 2:
        return None
    if _same_color_close_pairs(grid) < 2:
        return None
    if len(_components_of_color(grid, GRAY)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
