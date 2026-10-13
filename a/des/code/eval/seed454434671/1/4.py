from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# The missing corner of the 2x2 L marks the direction.
DIR_TO_MISSING = {
    "up": (0, 0),
    "right": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
}

PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
WALL_COLOR = 5
BG_COLOR = 0
MAX_GEN_ATTEMPTS = 200

BLOB_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _decode_pointer(pointer_coords: np.ndarray) -> str:
    assert len(pointer_coords) == 3, "pointer must contain exactly 3 cells"
    r0, c0, r1, c1 = _bbox(pointer_coords)
    assert r1 - r0 == 1 and c1 - c0 == 1, "pointer must be a 2x2 L"
    cells = {(int(r), int(c)) for r, c in pointer_coords}
    full = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(full - cells)
    assert len(missing) == 1, "pointer must have one missing corner"
    rel = (missing[0][0] - r0, missing[0][1] - c0)
    for direction, miss in DIR_TO_MISSING.items():
        if rel == miss:
            return direction
    raise AssertionError("invalid pointer orientation")


def _find_neighbor_index(
    bbox: tuple[int, int, int, int],
    direction: str,
    bboxes: list[tuple[int, int, int, int]],
) -> int:
    r0, c0, r1, c1 = bbox
    best_idx = -1
    best_overlap = 0
    ambiguous = False

    for idx, other in enumerate(bboxes):
        or0, oc0, or1, oc1 = other
        overlap = 0
        if direction == "up" and or1 + 2 == r0:
            overlap = _interval_overlap(c0, c1, oc0, oc1)
        elif direction == "down" and r1 + 2 == or0:
            overlap = _interval_overlap(c0, c1, oc0, oc1)
        elif direction == "left" and oc1 + 2 == c0:
            overlap = _interval_overlap(r0, r1, or0, or1)
        elif direction == "right" and c1 + 2 == oc0:
            overlap = _interval_overlap(r0, r1, or0, or1)

        if overlap > best_overlap:
            best_idx = idx
            best_overlap = overlap
            ambiguous = False
        elif overlap > 0 and overlap == best_overlap:
            ambiguous = True

    assert best_idx >= 0 and not ambiguous, "direction must select a unique adjacent region"
    return best_idx


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    counts = Counter(int(v) for v in g.ravel())
    bg = int(counts.most_common(1)[0][0])
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected walls and regions"
    wall = max(non_bg, key=lambda color: counts[color])

    regions = [
        comp
        for comp in _components(g != wall)
        if not _touches_boundary(comp, h, w)
    ]
    regions.sort(key=lambda comp: (_bbox(comp)[0], _bbox(comp)[1]))
    assert len(regions) >= 2, "expected multiple enclosed regions"

    appearances = Counter()
    region_unique_colors: list[list[int]] = []
    for coords in regions:
        values = [int(g[r, c]) for r, c in coords]
        colors = sorted({v for v in values if v not in (bg, wall)})
        assert len(colors) >= 2, "each region needs a pointer and a fill block"
        region_unique_colors.append(colors)
        for color in colors:
            appearances[color] += 1

    full_coverage = [color for color, seen in appearances.items() if seen == len(regions)]
    candidates = full_coverage or [color for color, seen in appearances.items() if seen == max(appearances.values())]
    pointer_color = min(candidates, key=lambda color: abs(counts[color] - 3 * len(regions)))

    parsed_regions = []
    for coords, colors in zip(regions, region_unique_colors):
        values = np.array([int(g[r, c]) for r, c in coords], dtype=int)
        assert pointer_color in colors, "pointer color must appear in every region"
        fill_candidates = [color for color in colors if color != pointer_color]
        assert fill_candidates, "region is missing its fill color"
        fill_color = max(fill_candidates, key=lambda color: int(np.sum(values == color)))
        pointer_coords = coords[values == pointer_color]
        fill_coords = coords[values == fill_color]
        parsed_regions.append(
            {
                "bbox": _bbox(coords),
                "direction": _decode_pointer(pointer_coords),
                "fill_color": int(fill_color),
                "fill_coords": fill_coords,
            }
        )

    bboxes = [region["bbox"] for region in parsed_regions]
    out = g.copy()
    for region in parsed_regions:
        neighbor_idx = _find_neighbor_index(region["bbox"], region["direction"], bboxes)
        new_color = parsed_regions[neighbor_idx]["fill_color"]
        fill_coords = region["fill_coords"]
        out[fill_coords[:, 0], fill_coords[:, 1]] = new_color
    return out


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        if not any(rotated.shape == other.shape and np.array_equal(rotated, other) for other in variants):
            variants.append(rotated.copy())
    return variants


def _all_placements(region_h: int, region_w: int, mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mh, mw = mask.shape
    placements: list[set[tuple[int, int]]] = []
    for top in range(1, region_h - mh):
        for left in range(1, region_w - mw):
            cells = {
                (top + dr, left + dc)
                for dr in range(mh)
                for dc in range(mw)
                if mask[dr, dc]
            }
            placements.append(cells)
    return placements


def _buffered(cells: set[tuple[int, int]], region_h: int, region_w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < region_h and 0 <= nc < region_w:
                    out.add((nr, nc))
    return out


def _place_region_objects(region_h: int, region_w: int, direction: str, rng: random.Random):
    pointer_mask = np.ones((2, 2), dtype=bool)
    pointer_mask[DIR_TO_MISSING[direction]] = False
    pointer_options = _all_placements(region_h, region_w, pointer_mask)
    rng.shuffle(pointer_options)

    shape_variants: list[np.ndarray] = []
    for shape in BLOB_SHAPES:
        shape_variants.extend(_unique_rotations(shape))
    rng.shuffle(shape_variants)

    for pointer_cells in pointer_options:
        blocked = _buffered(pointer_cells, region_h, region_w)
        blob_options: list[set[tuple[int, int]]] = []
        for shape in shape_variants:
            for blob_cells in _all_placements(region_h, region_w, shape):
                if blob_cells.isdisjoint(blocked):
                    blob_options.append(blob_cells)
        if blob_options:
            return pointer_cells, rng.choice(blob_options)
    raise ValueError("failed to place pointer and blob")


def _perimeter_cycle(rows: int, cols: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for c in range(cols):
        order.append((0, c))
    for r in range(1, rows):
        order.append((r, cols - 1))
    for c in range(cols - 2, -1, -1):
        order.append((rows - 1, c))
    for r in range(rows - 2, 0, -1):
        order.append((r, 0))
    return order


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for direction, (rr, cc) in DIRS.items():
        if (dr, dc) == (rr, cc):
            return direction
    raise AssertionError("cycle must move through side-adjacent cells")


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    by_cell = {spec["cell"]: spec for spec in specs}
    for spec in specs:
        successor = by_cell[spec["next_cell"]]
        for r, c in spec["fill_cells"]:
            out[r, c] = successor["fill_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice([(2, 2), (2, 3), (3, 2)])
    heights = [rng.randint(6, 8) for _ in range(rows)]
    widths = [rng.randint(6, 8) for _ in range(cols)]
    margin_top = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    margin_right = rng.randint(1, 3)

    arrangement_h = sum(heights) + rows + 1
    arrangement_w = sum(widths) + cols + 1
    h = margin_top + arrangement_h + margin_bottom
    w = margin_left + arrangement_w + margin_right
    grid = np.full((h, w), BG_COLOR, dtype=int)

    pointer_color = rng.choice(PALETTE)
    fill_colors = rng.sample([color for color in PALETTE if color != pointer_color], rows * cols)

    row_starts = []
    cursor = margin_top + 1
    for height in heights:
        row_starts.append(cursor)
        cursor += height + 1

    col_starts = []
    cursor = margin_left + 1
    for width in widths:
        col_starts.append(cursor)
        cursor += width + 1

    top = margin_top
    left = margin_left
    grid[top : top + arrangement_h, left] = WALL_COLOR
    grid[top : top + arrangement_h, left + arrangement_w - 1] = WALL_COLOR
    grid[top, left : left + arrangement_w] = WALL_COLOR
    grid[top + arrangement_h - 1, left : left + arrangement_w] = WALL_COLOR

    for boundary in range(rows - 1):
        wall_row = row_starts[boundary] + heights[boundary]
        grid[wall_row, left : left + arrangement_w] = WALL_COLOR
    for boundary in range(cols - 1):
        wall_col = col_starts[boundary] + widths[boundary]
        grid[top : top + arrangement_h, wall_col] = WALL_COLOR

    order = _perimeter_cycle(rows, cols)
    if rng.random() < 0.5:
        order = list(reversed(order))
    successor = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}

    specs: list[dict] = []
    color_iter = iter(fill_colors)
    for r in range(rows):
        for c in range(cols):
            region_h = heights[r]
            region_w = widths[c]
            next_cell = successor[(r, c)]
            direction = _direction_between((r, c), next_cell)
            pointer_cells_local, blob_cells_local = _place_region_objects(region_h, region_w, direction, rng)
            region_r0 = row_starts[r]
            region_c0 = col_starts[c]

            for rr, cc in pointer_cells_local:
                grid[region_r0 + rr, region_c0 + cc] = pointer_color

            fill_color = next(color_iter)
            fill_cells_global = []
            for rr, cc in blob_cells_local:
                gr = region_r0 + rr
                gc = region_c0 + cc
                grid[gr, gc] = fill_color
                fill_cells_global.append((gr, gc))

            specs.append(
                {
                    "cell": (r, c),
                    "next_cell": next_cell,
                    "fill_color": fill_color,
                    "fill_cells": fill_cells_global,
                }
            )

    expected = _build_expected(grid, specs)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
