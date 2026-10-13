import random

import numpy as np


AZURE = 8
RED = 2
GREY = 5
CONTENT_COLORS = (1, 3, 4, 6, 7, 9)

GEN_RETRIES = 300
REGIONS_PER_AXIS = (1, 3)
INTERIOR_SIZE_RANGE = (4, 8)
PATCH_SPAN_MAX = 4
PATCH_CELL_MAX = 6
REGION_RETRIES = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _region_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components(grid != GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(region != GREY), "each enclosed region must be a solid rectangle"
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    assert boxes, "expected at least one enclosed region"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each region's colored content to the wall indicated by its red two-pixel compass.

    Input structure:
    - The background is azure.
    - Grey one-cell walls partition the grid into enclosed rectangular regions.
    - Each region contains exactly one red two-pixel compass marker and some non-azure content pixels in other colors.

    Rule:
    - Process each enclosed region independently.
    - Read the compass from its two adjacent red cells: a horizontal pair points right, and a vertical pair points down.
    - Remove the compass, then translate every other non-azure pixel in that region together along that direction until the translated set first touches the grey wall.
    - Keep the grey walls and all untouched azure cells unchanged.

    Key insight:
    - The red domino does not mark a destination cell; its axis gives the motion axis, and the visible scan order of the pair fixes the positive direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _region_boxes(g):
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        red_coords = np.argwhere(region == RED)
        assert red_coords.shape == (2, 2), "each region must contain one two-cell red compass"

        (ar, ac), (br, bc) = [tuple(map(int, pos)) for pos in red_coords]
        assert abs(ar - br) + abs(ac - bc) == 1, "compass cells must be orthogonally adjacent"
        dr, dc = ((0, 1) if ar == br else (1, 0))

        region_out = np.full(region.shape, AZURE, dtype=int)
        content_mask = (region != AZURE) & (region != RED)
        content = np.argwhere(content_mask)
        if content.size:
            if dc == 1:
                shift = int(region.shape[1] - 1 - content[:, 1].max())
            else:
                shift = int(region.shape[0] - 1 - content[:, 0].max())
            for rr, cc in content:
                rr = int(rr)
                cc = int(cc)
                region_out[rr + dr * shift, cc + dc * shift] = int(region[rr, cc])

        out[r0 : r1 + 1, c0 : c1 + 1] = region_out

    return out


def _balanced_directions(count: int) -> list[str]:
    directions = ["right"] * (count // 2) + ["down"] * (count // 2)
    if count % 2:
        directions.append(random.choice(["right", "down"]))
    random.shuffle(directions)
    return directions


def _build_partition(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    total_h = sum(row_heights) + len(row_heights) + 1
    total_w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((total_h, total_w), AZURE, dtype=int)

    divider_rows = [0]
    pos = 0
    for height in row_heights:
        pos += height + 1
        divider_rows.append(pos)
    divider_cols = [0]
    pos = 0
    for width in col_widths:
        pos += width + 1
        divider_cols.append(pos)

    grid[divider_rows, :] = GREY
    grid[:, divider_cols] = GREY

    boxes: list[tuple[int, int, int, int]] = []
    top = 1
    for height in row_heights:
        left = 1
        for width in col_widths:
            boxes.append((top, left, height, width))
            left += width + 1
        top += height + 1

    return grid, boxes


def _sample_shape_cells(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    boundary = [cell for cell in all_cells if (cell[1] == width - 1 if direction == "right" else cell[0] == height - 1)]
    max_count = min(PATCH_CELL_MAX, len(all_cells))
    min_count = 2 if max_count >= 2 else 1
    count = random.randint(min_count, max_count)

    chosen = [random.choice(boundary)]
    remaining = [cell for cell in all_cells if cell != chosen[0]]
    if count > 1:
        chosen.extend(random.sample(remaining, count - 1))
    return chosen


def _assign_colors(cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    n_colors = random.randint(1, min(3, len(cells), len(CONTENT_COLORS)))
    palette = random.sample(CONTENT_COLORS, n_colors)
    order = cells[:]
    random.shuffle(order)

    colors = {cell: palette[i] for i, cell in enumerate(order[:n_colors])}
    for cell in order[n_colors:]:
        colors[cell] = random.choice(palette)
    return colors


def _compass_options(
    height: int,
    width: int,
    direction: str,
    input_cells: dict[tuple[int, int], int],
    output_cells: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    blocked = set(input_cells) | set(output_cells)
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if direction == "right":
        for r in range(height):
            for c in range(width - 1):
                pair = ((r, c), (r, c + 1))
                if pair[0] not in blocked and pair[1] not in blocked:
                    options.append(pair)
    else:
        for r in range(height - 1):
            for c in range(width):
                pair = ((r, c), (r + 1, c))
                if pair[0] not in blocked and pair[1] not in blocked:
                    options.append(pair)
    return options


def _decorate_region(
    inp: np.ndarray,
    out: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    direction: str,
) -> bool:
    for _ in range(REGION_RETRIES):
        if direction == "right":
            patch_h = random.randint(1, min(PATCH_SPAN_MAX, height))
            patch_w = random.randint(1, min(PATCH_SPAN_MAX, width - 1))
            out_top = random.randint(0, height - patch_h)
            out_left = width - patch_w
            max_gap = out_left
            if max_gap < 1:
                continue
            gap = random.randint(1, max_gap)
            in_top = out_top
            in_left = out_left - gap
        else:
            patch_h = random.randint(1, min(PATCH_SPAN_MAX, height - 1))
            patch_w = random.randint(1, min(PATCH_SPAN_MAX, width))
            out_top = height - patch_h
            out_left = random.randint(0, width - patch_w)
            max_gap = out_top
            if max_gap < 1:
                continue
            gap = random.randint(1, max_gap)
            in_top = out_top - gap
            in_left = out_left

        cells = _sample_shape_cells(patch_h, patch_w, direction)
        colors = _assign_colors(cells)
        input_cells = {(in_top + r, in_left + c): color for (r, c), color in colors.items()}
        output_cells = {(out_top + r, out_left + c): color for (r, c), color in colors.items()}

        options = _compass_options(height, width, direction, input_cells, output_cells)
        if not options:
            continue

        marker = random.choice(options)
        for (rr, cc), color in input_cells.items():
            inp[top + rr, left + cc] = color
        for rr, cc in marker:
            inp[top + rr, left + cc] = RED
        for (rr, cc), color in output_cells.items():
            out[top + rr, left + cc] = color
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*REGIONS_PER_AXIS)
    n_cols = random.randint(*REGIONS_PER_AXIS)
    if n_rows * n_cols < 2:
        return None

    row_heights = [random.randint(*INTERIOR_SIZE_RANGE) for _ in range(n_rows)]
    col_widths = [random.randint(*INTERIOR_SIZE_RANGE) for _ in range(n_cols)]
    base, boxes = _build_partition(row_heights, col_widths)
    inp = base.copy()
    out = base.copy()

    directions = _balanced_directions(len(boxes))
    for (top, left, height, width), direction in zip(boxes, directions):
        if not _decorate_region(inp, out, top, left, height, width, direction):
            return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
