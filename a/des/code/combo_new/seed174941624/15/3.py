from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_TRIES = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_info(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    grey_mask = grid == GREY
    assert grey_mask.any(), "legend border must exist"
    r0, c0, r1, c1 = _bbox_from_mask(grey_mask)

    border_mask = np.zeros_like(grey_mask, dtype=bool)
    border_mask[r0, c0 : c1 + 1] = True
    border_mask[r1, c0 : c1 + 1] = True
    border_mask[r0 : r1 + 1, c0] = True
    border_mask[r0 : r1 + 1, c1] = True
    assert np.array_equal(grey_mask, border_mask), "grey pixels must form one rectangular border"

    legend_mask = np.zeros_like(grey_mask, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping: dict[int, int] = {}
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    for row in interior:
        maroon_count = int(np.count_nonzero(row == MAROON))
        if maroon_count == 0:
            continue
        fill_colors = [int(v) for v in row.tolist() if int(v) not in (bg, MAROON)]
        assert fill_colors, "legend row with maroon count must specify a fill color"
        assert len(set(fill_colors)) == 1, "each legend row must encode exactly one fill color"
        mapping[maroon_count] = fill_colors[0]

    assert mapping, "legend mapping must be non-empty"
    return mapping, legend_mask


def _background_components(grid: np.ndarray, legend_mask: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or legend_mask[r, c] or int(grid[r, c]) != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or legend_mask[nr, nc] or int(grid[nr, nc]) != bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(comp)
    return components


def _touches_edge(component: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in component)


def _boundary_color_count(
    grid: np.ndarray, component: list[tuple[int, int]], legend_mask: np.ndarray, bg: int
) -> int:
    h, w = grid.shape
    colors: set[int] = set()
    for r, c in component:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if legend_mask[nr, nc]:
                continue
            val = int(grid[nr, nc])
            if val != bg:
                colors.add(val)
    assert colors, "closed region must have a colored boundary"
    return len(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed region using the legend entry keyed by its boundary color-count.

    Input structure:
    - The background is black and most of the grid is open background.
    - One-pixel-wide colored line segments connect to form one or more closed outlines whose interiors remain background.
    - A small grey rectangular legend sits in the lower-right area; inside each legend row, a run of 1-4 maroon cells encodes a count and a single colored cell encodes the fill color for that count.

    Rule:
    - Locate the grey legend frame and read every interior row as count -> fill-color, where the count is the number of maroon cells in that row.
    - Find every background connected component that is not part of the legend and does not touch the grid boundary; these are the enclosed regions.
    - For each enclosed region, collect the distinct colors on its 4-neighbor boundary, count how many different colors appear there, look up that count in the legend, and fill the whole region with the mapped color.
    - Keep the colored line segments, the legend, and every background region that touches the grid edge unchanged.

    Key insight:
    - The maroon blocks in the legend do not label line colors directly; they encode the number of different boundary colors surrounding a closed region, and that count determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())
    mapping, legend_mask = _legend_info(g, bg)

    out = g.copy()
    for component in _background_components(g, legend_mask, bg):
        if _touches_edge(component, g.shape):
            continue
        color_count = _boundary_color_count(g, component, legend_mask, bg)
        assert color_count in mapping, "legend must cover every enclosed region count"
        fill = int(mapping[color_count])
        for r, c in component:
            out[r, c] = fill
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _draw_legend(grid: np.ndarray, mapping: dict[int, int]) -> tuple[int, int, int, int]:
    h, w = grid.shape
    inner_h = len(mapping)
    inner_w = 6
    legend_h = inner_h + 2
    legend_w = inner_w + 2
    bottom_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    r0 = h - legend_h - bottom_margin
    c0 = w - legend_w - right_margin
    r1 = r0 + legend_h - 1
    c1 = c0 + legend_w - 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    row_counts = list(mapping)
    random.shuffle(row_counts)
    for i, count in enumerate(row_counts):
        rr = r0 + 1 + i
        grid[rr, c0 + 1 : c0 + 1 + count] = MAROON
        grid[rr, c1 - 1] = int(mapping[count])
    return r0, c0, r1, c1


def _side_assignment(count: int) -> dict[str, int]:
    unique_colors = random.sample(PALETTE, count)
    sides = ["top", "right", "bottom", "left"]
    random.shuffle(sides)
    assignment: dict[str, int] = {}
    for idx, side in enumerate(sides):
        if idx < count:
            assignment[side] = unique_colors[idx]
        else:
            assignment[side] = random.choice(unique_colors)
    return assignment


def _draw_region(
    grid: np.ndarray, r0: int, c0: int, height: int, width: int, count: int
) -> tuple[tuple[int, int, int, int], int]:
    colors = _side_assignment(count)
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = colors["top"]
    grid[r1, c0 : c1 + 1] = colors["bottom"]
    grid[r0 : r1 + 1, c0] = colors["left"]
    grid[r0 : r1 + 1, c1] = colors["right"]
    return (r0, c0, r1, c1), count


def _expected_output(inp: np.ndarray, mapping: dict[int, int], regions: list[tuple[tuple[int, int, int, int], int]]) -> np.ndarray:
    out = inp.copy()
    for (r0, c0, r1, c1), count in regions:
        out[r0 + 1 : r1, c0 + 1 : c1] = int(mapping[count])
    return out


def _region_box_options(
    grid_shape: tuple[int, int], legend_box: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    h, w = grid_shape
    options: list[tuple[int, int, int, int]] = []
    for height in range(4, 8):
        for width in range(4, 8):
            for r0 in range(1, h - height):
                for c0 in range(1, w - width):
                    box = (r0, c0, r0 + height - 1, c0 + width - 1)
                    padded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
                    if _boxes_overlap(padded, legend_box):
                        continue
                    if any(_boxes_overlap(padded, other) for other in occupied):
                        continue
                    options.append(box)
    return options


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)

    fill_colors = random.sample(PALETTE, 4)
    mapping = {idx + 1: fill_colors[idx] for idx in range(4)}
    legend_box = _draw_legend(grid, mapping)
    legend_padded = (legend_box[0] - 1, legend_box[1] - 1, legend_box[2] + 1, legend_box[3] + 1)

    region_count = random.randint(2, 4)
    counts = random.sample([1, 2, 3, 4], region_count)
    random.shuffle(counts)

    occupied = [legend_padded]
    regions: list[tuple[tuple[int, int, int, int], int]] = []

    for count in counts:
        options = _region_box_options(grid.shape, legend_padded, occupied)
        if not options:
            return None
        box = random.choice(options)
        r0, c0, r1, c1 = box
        regions.append(_draw_region(grid, r0, c0, r1 - r0 + 1, c1 - c0 + 1, count))
        occupied.append((r0 - 1, c0 - 1, r1 + 1, c1 + 1))

    out = _expected_output(grid, mapping, regions)
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, expected = attempt
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
