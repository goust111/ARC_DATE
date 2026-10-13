from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DEFAULT_BORDER = 1
LEGEND_COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _side_cells(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    return cells


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    legend = [int(v) for v in grid[:, -1].tolist() if int(v) != bg]
    return {idx + 1: color for idx, color in enumerate(legend)}


def _detect_border_color(grid: np.ndarray, bg: int) -> int:
    body = grid[:, :-1]
    vals, cnts = np.unique(body[body != bg], return_counts=True)
    if vals.size == 0:
        return DEFAULT_BORDER
    return int(vals[np.argmax(cnts)])


def _count_gaps(grid: np.ndarray, border: int, bg: int, bbox: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != border for r, c in corners):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != bg):
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != bg) & (sub != border)):
        return None

    sides = _side_cells(r0, c0, r1, c1)
    gap_count = 0
    for cells in sides.values():
        miss = [idx for idx, (r, c) in enumerate(cells) if int(grid[r, c]) == bg]
        if any(b - a == 1 for a, b in zip(miss, miss[1:], strict=False)):
            return None
        gap_count += len(miss)

    for r, c in _perimeter_cells(r0, c0, r1, c1):
        if int(grid[r, c]) not in (bg, border):
            return None

    if np.count_nonzero(sub == border) != len(_perimeter_cells(r0, c0, r1, c1)) - gap_count:
        return None

    return int(gap_count)


def _find_rectangles(grid: np.ndarray, border: int, bg: int) -> list[tuple[tuple[int, int, int, int], int]]:
    mask = grid[:, :-1] == border
    inflated = _dilate4(mask)
    rectangles: list[tuple[tuple[int, int, int, int], int]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()

    for comp in _components(inflated):
        pts = [(r, c) for r, c in comp if mask[r, c]]
        if not pts:
            continue
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        if bbox in seen_bboxes:
            continue
        gap_count = _count_gaps(grid, border, bg, bbox)
        if gap_count is None:
            continue
        seen_bboxes.add(bbox)
        rectangles.append((bbox, gap_count))

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue rectangular enclosure according to how many single-pixel gates its border contains.

    Input structure:
    - The background is black.
    - Several single-pixel-thick blue rectangular outlines appear in the grid; each outline may have zero or more one-cell gaps on its border.
    - The rightmost column is a legend made of isolated colored pixels, ordered from top to bottom with one background cell between neighboring legend pixels.

    Rule:
    - Count the number of border gaps on each blue rectangle.
    - Read the legend positionally: the first legend color means "1 gate", the second means "2 gates", and so on.
    - Fill the strict interior of each rectangle with the color indexed by its gate count, while leaving the blue border unchanged.
    - Rectangles with zero gates are not filled.
    - Erase the entire legend column to background in the output.

    Key insight:
    The legend does not map from border color to fill color; instead, its vertical order encodes a lookup table from gate count to fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a legend column"

    bg = BACKGROUND
    border = _detect_border_color(g, bg)
    legend = _legend_mapping(g, bg)

    out = g.copy()
    out[:, -1] = bg

    for (r0, c0, r1, c1), gap_count in _find_rectangles(g, border, bg):
        if gap_count <= 0 or gap_count not in legend:
            continue
        fill = legend[gap_count]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = int(fill)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 30)
GRID_WIDTH_RANGE = (18, 30)
LEGEND_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (5, 8)
RECT_WIDTH_RANGE = (5, 8)
RECT_MARGIN = 1
RECT_CLEARANCE = 3
EXTRA_RECTANGLES_RANGE = (0, 2)
PLACEMENT_TRIES = 200
GENERATION_TRIES = 400


def _rectangles_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], clearance: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + clearance < br0 or br1 + clearance < ar0 or ac1 + clearance < bc0 or bc1 + clearance < ac0


def _sample_gates(height: int, width: int, gate_count: int) -> set[tuple[int, int]]:
    assert 0 <= gate_count <= 4
    if gate_count == 0:
        return set()

    side_offsets = {
        "top": [(0, c) for c in range(1, width - 1)],
        "bottom": [(height - 1, c) for c in range(1, width - 1)],
        "left": [(r, 0) for r in range(1, height - 1)],
        "right": [(r, width - 1) for r in range(1, height - 1)],
    }
    chosen_sides = random.sample(list(side_offsets), gate_count)
    gates = {random.choice(side_offsets[side]) for side in chosen_sides}
    return {(int(r), int(c)) for r, c in gates}


def _draw_rectangle(grid: np.ndarray, top: int, left: int, height: int, width: int, gate_count: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    gates = _sample_gates(height, width, gate_count)

    for r, c in _perimeter_cells(top, left, bottom, right):
        local = (r - top, c - left)
        if local in gates:
            continue
        grid[r, c] = DEFAULT_BORDER


def _expected_from_layout(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    out[:, -1] = BACKGROUND
    for top, left, height, width, gate_count in rectangles:
        if gate_count == 0:
            continue
        color = int(legend_colors[gate_count - 1])
        out[top + 1 : top + height - 1, left + 1 : left + width - 1] = color
    return out


def _place_rectangles(
    height: int,
    width: int,
    gate_counts: list[int],
) -> list[tuple[int, int, int, int, int]] | None:
    specs = [
        (
            random.randint(*RECT_HEIGHT_RANGE),
            random.randint(*RECT_WIDTH_RANGE),
            int(gate_count),
        )
        for gate_count in gate_counts
    ]
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    rectangles: list[tuple[int, int, int, int, int]] = []
    max_top = height - RECT_MARGIN - RECT_HEIGHT_RANGE[0]
    max_left = width - 2 - RECT_MARGIN - RECT_WIDTH_RANGE[0]
    if max_top < RECT_MARGIN or max_left < RECT_MARGIN:
        return None

    for rect_h, rect_w, gate_count in specs:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(RECT_MARGIN, height - rect_h - RECT_MARGIN)
            left = random.randint(RECT_MARGIN, width - 2 - rect_w - RECT_MARGIN)
            bbox = (top, left, top + rect_h - 1, left + rect_w - 1)
            if any(not _rectangles_far_apart(bbox, (r0, c0, r0 + rh - 1, c0 + rw - 1), RECT_CLEARANCE) for r0, c0, rh, rw, _ in rectangles):
                continue
            rectangles.append((int(top), int(left), int(rect_h), int(rect_w), int(gate_count)))
            placed = True
            break
        if not placed:
            return None

    random.shuffle(rectangles)
    return rectangles


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        legend_count = random.randint(*LEGEND_COUNT_RANGE)
        legend_colors = random.sample(LEGEND_COLOR_POOL, legend_count)

        gate_counts = list(range(legend_count + 1))
        extra_count = random.randint(*EXTRA_RECTANGLES_RANGE)
        gate_counts.extend(random.randint(0, legend_count) for _ in range(extra_count))
        random.shuffle(gate_counts)

        grid_h = random.randint(max(GRID_HEIGHT_RANGE[0], 2 * legend_count), GRID_HEIGHT_RANGE[1])
        grid_w = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
        rectangles = _place_rectangles(grid_h, grid_w, gate_counts)
        if rectangles is None:
            continue

        grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

        legend_start = random.randint(0, grid_h - (2 * legend_count - 1))
        for idx, color in enumerate(legend_colors):
            grid[legend_start + 2 * idx, grid_w - 1] = int(color)

        for top, left, rect_h, rect_w, gate_count in rectangles:
            _draw_rectangle(grid, top, left, rect_h, rect_w, gate_count)

        expected = _expected_from_layout(grid, rectangles, legend_colors)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
