from __future__ import annotations

import random

import numpy as np


DIGIT_PATTERNS: dict[int, tuple[str, ...]] = {
    0: ("111", "101", "101", "101", "111"),
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
}

BG_COLOR = 0
SWATCH_SIZE = 3
DIGIT_HEIGHT = len(DIGIT_PATTERNS[0])
DIGIT_WIDTH = len(DIGIT_PATTERNS[0][0])
LEGEND_V_GAP = 1
LEGEND_ENTRY_GAP_RANGE = (1, 2)
ROW_GAP_RANGE = (2, 4)
SHAPE_LEGEND_GAP_MIN = 1
GEN_RETRIES = 400
SHAPE_RETRIES = 200
ARC_COLORS = tuple(range(10))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_edge = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_edge:
                holes += 1
    return holes


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _component_hole_count(coords: np.ndarray) -> int:
    return _hole_count(_component_mask(coords))


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _dominant_foreground_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected non-background content"
    return max(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _legend_mapping(grid: np.ndarray, bg: int, shape_color: int) -> tuple[dict[int, int], int]:
    other_colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, shape_color)]
    assert len(other_colors) == 4, "expected exactly four legend colors"

    entries: list[tuple[int, int, int]] = []
    for color in other_colors:
        comps = _components(grid == color)
        assert len(comps) == 1, "each legend color must form one swatch"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "legend swatch must be solid"
        assert r1 - r0 + 1 >= 2 and c1 - c0 + 1 >= 2, "legend swatch must be at least 2x2"
        entries.append((c0, r0, color))

    entries.sort()
    mapping = {index: color for index, (_c0, _r0, color) in enumerate(entries)}
    legend_top = min(r0 for _c0, r0, _color in entries)
    return mapping, int(legend_top)


def _shape_components(grid: np.ndarray, shape_color: int, legend_top: int) -> list[np.ndarray]:
    comps: list[np.ndarray] = []
    for coords in _components(grid == shape_color):
        _r0, _c0, r1, _c1 = _bbox(coords)
        if r1 < legend_top:
            comps.append(coords)
    assert comps, "expected at least one non-legend shape"
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed foreground shape by matching its number of background holes to the bottom legend.

    Input structure:
    - The background is the most common color.
    - Several disconnected shapes are drawn in the same foreground color; they may have 0, 1, 2, or 3 enclosed background holes.
    - Along the bottom sits a horizontal legend of four colored swatches ordered left-to-right for counts 0, 1, 2, and 3, with same-color digits beneath them.

    Rule:
    - Read the four bottom swatches from left to right as the color mapping for hole counts 0 through 3.
    - Ignore the bottom legend itself and examine every upper foreground component.
    - Count how many enclosed background regions each shape contains, then recolor that whole shape to the legend color for the same count.
    - Leave the legend and all background holes unchanged.

    Key insight:
    - The important quantity is the topology of each connected shape: distinct enclosed background regions, not the outer silhouette, determine which legend color to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    shape_color = _dominant_foreground_color(g, bg)
    mapping, legend_top = _legend_mapping(g, bg, shape_color)
    shape_components = _shape_components(g, shape_color, legend_top)

    out = g.copy()
    for coords in shape_components:
        hole_count = _component_hole_count(coords)
        assert hole_count in mapping, "shape hole count must be in 0..3"
        out[coords[:, 0], coords[:, 1]] = int(mapping[hole_count])
    return out


def _draw_digit(grid: np.ndarray, top: int, left: int, color: int, digit: int) -> None:
    for r, row in enumerate(DIGIT_PATTERNS[digit]):
        for c, cell in enumerate(row):
            if cell == "1":
                grid[top + r, left + c] = int(color)


def _carve_holes(mask: np.ndarray, hole_count: int) -> bool:
    if hole_count == 0:
        return True

    h, w = mask.shape
    guard = np.zeros((h, w), dtype=bool)
    for _ in range(hole_count):
        placed = False
        for _ in range(80):
            hole_h = random.choice((1, 1, 2))
            hole_w = random.choice((1, 1, 2))
            if h - hole_h - 1 <= 1 or w - hole_w - 1 <= 1:
                return False
            r0 = random.randint(1, h - hole_h - 1)
            c0 = random.randint(1, w - hole_w - 1)
            rr = slice(max(1, r0 - 1), min(h - 1, r0 + hole_h + 1))
            cc = slice(max(1, c0 - 1), min(w - 1, c0 + hole_w + 1))
            if np.any(guard[rr, cc]):
                continue
            mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False
            guard[rr, cc] = True
            placed = True
            break
        if not placed:
            return False
    return _is_connected(mask) and _hole_count(mask) == hole_count


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w) or not mask[rr, cc]:
                    cells.append((r, c))
                    break
    return cells


def _build_shape(hole_count: int) -> np.ndarray:
    min_side = {0: 5, 1: 6, 2: 7, 3: 8}[hole_count]
    max_side = {0: 8, 1: 9, 2: 10, 3: 11}[hole_count]
    min_area = 10 + 6 * hole_count

    for _ in range(SHAPE_RETRIES):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        mask = np.ones((h, w), dtype=bool)
        if not _carve_holes(mask, hole_count):
            continue

        removals_left = random.randint(1, 3 + hole_count)
        for _ in range(150):
            if removals_left <= 0:
                break
            boundary = _boundary_cells(mask)
            if not boundary:
                break
            r, c = random.choice(boundary)
            mask[r, c] = False
            if (
                np.count_nonzero(mask) >= min_area
                and _is_connected(mask)
                and _hole_count(mask) == hole_count
            ):
                removals_left -= 1
            else:
                mask[r, c] = True

        if not _is_connected(mask) or _hole_count(mask) != hole_count:
            continue
        coords = np.argwhere(mask)
        if int(coords.shape[0]) < min_area:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(cropped.shape) < 3:
            continue
        return cropped

    raise RuntimeError(f"failed to build shape with {hole_count} holes")


def _pack_rows(shapes: list[np.ndarray], legend_width: int, legend_height: int) -> tuple[int, int, list[tuple[int, int, np.ndarray]]] | None:
    count = len(shapes)
    possible_row_counts = [2] if count <= 5 else [2, 3]
    random.shuffle(possible_row_counts)

    for row_count in possible_row_counts:
        rows: list[list[np.ndarray]] = [[] for _ in range(row_count)]
        widths = [0] * row_count
        shuffled = shapes[:]
        random.shuffle(shuffled)
        shuffled.sort(key=lambda mask: int(mask.shape[1]), reverse=True)

        for mask in shuffled:
            idx = min(range(row_count), key=lambda i: widths[i])
            rows[idx].append(mask)
            widths[idx] += int(mask.shape[1]) + 3

        row_specs: list[tuple[list[np.ndarray], list[int], int, int]] = []
        max_row_width = 0
        total_row_height = 0
        for row in rows:
            if not row:
                continue
            gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(len(row) - 1)]
            row_height = max(int(mask.shape[0]) for mask in row)
            row_width = sum(int(mask.shape[1]) for mask in row) + sum(gaps)
            row_specs.append((row, gaps, row_height, row_width))
            max_row_width = max(max_row_width, row_width)
            total_row_height += row_height

        row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(max(0, len(row_specs) - 1))]
        shape_block_height = total_row_height + sum(row_gaps)
        min_width = max(legend_width, max_row_width) + 2
        min_height = shape_block_height + legend_height + 2
        if min_width > 30 or min_height > 30:
            continue

        grid_width = random.randint(min_width, min(30, min_width + 4))
        grid_height = random.randint(min_height, min(30, min_height + 4))
        legend_top = grid_height - legend_height
        max_shape_top = legend_top - shape_block_height - SHAPE_LEGEND_GAP_MIN
        if max_shape_top < 1:
            continue
        top = random.randint(1, max_shape_top)

        placements: list[tuple[int, int, np.ndarray]] = []
        cursor_y = top
        for row_index, (row, gaps, row_height, row_width) in enumerate(row_specs):
            max_left = grid_width - row_width - 1
            if max_left < 1:
                break
            left = random.randint(1, max_left)
            cursor_x = left
            for idx, mask in enumerate(row):
                y_jitter = random.randint(0, row_height - int(mask.shape[0]))
                placements.append((cursor_y + y_jitter, cursor_x, mask))
                cursor_x += int(mask.shape[1])
                if idx < len(gaps):
                    cursor_x += gaps[idx]
            cursor_y += row_height
            if row_index < len(row_gaps):
                cursor_y += row_gaps[row_index]
        else:
            return grid_height, grid_width, placements

    return None


def _legend_dimensions(entry_gap: int) -> tuple[int, int]:
    width = 4 * DIGIT_WIDTH + 3 * entry_gap
    height = SWATCH_SIZE + LEGEND_V_GAP + DIGIT_HEIGHT
    return width, height


def _draw_legend(grid: np.ndarray, top: int, left: int, shape_color: int, palette: list[int], entry_gap: int) -> None:
    for count, swatch_color in enumerate(palette):
        col = left + count * (DIGIT_WIDTH + entry_gap)
        grid[top : top + SWATCH_SIZE, col : col + SWATCH_SIZE] = int(swatch_color)
        _draw_digit(grid, top + SWATCH_SIZE + LEGEND_V_GAP, col, shape_color, count)


def _generate_attempt() -> np.ndarray | None:
    shape_color = random.choice([c for c in ARC_COLORS if c != BG_COLOR])
    palette = random.sample([c for c in ARC_COLORS if c not in (BG_COLOR, shape_color)], 4)
    entry_gap = random.randint(*LEGEND_ENTRY_GAP_RANGE)
    legend_width, legend_height = _legend_dimensions(entry_gap)

    total_shapes = random.randint(4, 6)
    hole_counts = [0, 1, 2, 3] + [random.randint(0, 3) for _ in range(total_shapes - 4)]
    shapes = [_build_shape(hole_count) for hole_count in hole_counts]
    packed = _pack_rows(shapes, legend_width, legend_height)
    if packed is None:
        return None

    grid_height, grid_width, placements = packed
    grid = np.full((grid_height, grid_width), BG_COLOR, dtype=int)

    for top, left, mask in placements:
        h, w = mask.shape
        block = grid[top : top + h, left : left + w]
        if np.any(block[mask] != BG_COLOR):
            return None
        block[mask] = shape_color

    legend_left = random.randint(1, grid_width - legend_width - 1)
    legend_top = grid_height - legend_height
    if np.any(grid[legend_top:, legend_left : legend_left + legend_width] != BG_COLOR):
        return None
    _draw_legend(grid, legend_top, legend_left, shape_color, palette, entry_gap)

    mapping, parsed_legend_top = _legend_mapping(grid, BG_COLOR, shape_color)
    if parsed_legend_top != legend_top:
        return None
    seen_counts = {_component_hole_count(coords) for coords in _shape_components(grid, shape_color, legend_top)}
    if seen_counts != {0, 1, 2, 3}:
        return None
    if [mapping[i] for i in range(4)] != palette:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
