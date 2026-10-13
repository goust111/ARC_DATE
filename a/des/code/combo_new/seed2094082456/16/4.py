from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 8
ARROW = 7
REVEAL = 5
RECT_COLORS = [1, 2, 3, 4, 6, 9]
DIRECTIONS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    values = [int(v) for v in grid.flat]
    return Counter(values).most_common(1)[0][0]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    assert connectivity == 8
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _find_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_cells_for_bbox(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    if direction == "right":
        mid = (r0 + r1) // 2
        arrow = [(mid, c1 + 1), (mid - 1, c1 + 2), (mid + 1, c1 + 2)]
        moved = [(r, c1 + 1) for r in range(r0, r1 + 1)]
    elif direction == "left":
        mid = (r0 + r1) // 2
        arrow = [(mid, c0 - 1), (mid - 1, c0 - 2), (mid + 1, c0 - 2)]
        moved = [(r, c0 - 1) for r in range(r0, r1 + 1)]
    elif direction == "down":
        mid = (c0 + c1) // 2
        arrow = [(r1 + 1, mid), (r1 + 2, mid - 1), (r1 + 2, mid + 1)]
        moved = [(r1 + 1, c) for c in range(c0, c1 + 1)]
    else:
        assert direction == "up"
        mid = (c0 + c1) // 2
        arrow = [(r0 - 1, mid), (r0 - 2, mid - 1), (r0 - 2, mid + 1)]
        moved = [(r0 - 1, c) for c in range(c0, c1 + 1)]
    return arrow, moved


def _match_arrow(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    arrow_index: dict[tuple[int, int], int],
    arrow_components: list[list[tuple[int, int]]],
) -> tuple[str, int]:
    h, w = grid.shape
    matches: list[tuple[str, int]] = []
    for direction in DIRECTIONS:
        arrow_cells, _ = _arrow_cells_for_bbox(bbox, direction)
        if not all(0 <= r < h and 0 <= c < w for r, c in arrow_cells):
            continue
        ids = {arrow_index.get((r, c), -1) for r, c in arrow_cells}
        if len(ids) != 1:
            continue
        arrow_id = ids.pop()
        if arrow_id < 0:
            continue
        comp_cells = set(arrow_components[arrow_id])
        if comp_cells != set(arrow_cells):
            continue
        matches.append((direction, arrow_id))

    assert len(matches) == 1, "each solid rectangle must have exactly one matching arrow"
    return matches[0]


def _analyze_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int, int, str]], set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    arrow_components = _find_components(g == ARROW, connectivity=8)
    assert arrow_components, "expected at least one orange arrow"
    assert all(len(comp) == 3 for comp in arrow_components), "each arrow must contain exactly three pixels"
    arrow_index = {
        (int(r), int(c)): idx
        for idx, comp in enumerate(arrow_components)
        for r, c in comp
    }

    rect_mask = (g != bg) & (g != ARROW)
    rect_components = _find_components(rect_mask, connectivity=4)
    assert rect_components, "expected at least one colored rectangle"

    rectangles: list[tuple[int, int, int, int, int, str]] = []
    used_arrows: set[int] = set()
    seen_colors: set[int] = set()
    for comp in rect_components:
        r0, c0, r1, c1 = _bbox(comp)
        color = int(g[comp[0]])
        assert color not in {bg, ARROW}, "rectangle colors must differ from background and arrow color"
        assert all(int(g[r, c]) == color for r, c in comp), "rectangle must be monochrome"
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "rectangle must be solid"
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "rectangle must be at least 3x3"
        assert color not in seen_colors, "rectangles must use distinct colors"
        seen_colors.add(color)

        direction, arrow_id = _match_arrow(g, (r0, c0, r1, c1), arrow_index, arrow_components)
        assert arrow_id not in used_arrows, "each arrow must belong to exactly one rectangle"
        used_arrows.add(arrow_id)
        rectangles.append((r0, c0, r1, c1, color, direction))

    assert used_arrows == set(range(len(arrow_components))), "every arrow must be assigned to a rectangle"
    return bg, rectangles, used_arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel one outer layer from each colored rectangle in the direction marked by its adjacent arrow.

    Input structure:
    - The background is a dominant color, and each foreground object is a solid monochrome rectangle of size at least 3x3.
    - Every rectangle has exactly one adjacent orange 3-pixel V-shaped arrow outside one of its four sides.
    - The arrow points away from the rectangle and identifies the side whose outer layer will be peeled.

    Rule:
    - Detect each solid rectangle and match it to its unique adjacent arrow.
    - Remove the arrow from the output, then shift the rectangle's facing outer row or column one cell outward in the arrow direction.
    - Recolor the exposed row or column inside the original rectangle to grey, while leaving the moved strip in the rectangle's original color.

    Key insight:
    - The arrow does not move the whole rectangle; it marks only the single boundary layer that detaches, so the visible change is a peel plus a grey reveal.
    """
    g = np.asarray(grid, dtype=int)
    bg, rectangles, _ = _analyze_rectangles(g)

    out = g.copy()
    out[g == ARROW] = bg

    for r0, c0, r1, c1, color, direction in rectangles:
        if direction == "right":
            out[r0 : r1 + 1, c1] = REVEAL
            out[r0 : r1 + 1, c1 + 1] = color
        elif direction == "left":
            out[r0 : r1 + 1, c0] = REVEAL
            out[r0 : r1 + 1, c0 - 1] = color
        elif direction == "down":
            out[r1, c0 : c1 + 1] = REVEAL
            out[r1 + 1, c0 : c1 + 1] = color
        else:
            assert direction == "up"
            out[r0, c0 : c1 + 1] = REVEAL
            out[r0 - 1, c0 : c1 + 1] = color

    return out


def _direction_size_options(direction: str, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    options: list[tuple[int, int]] = []
    odd_lengths = [3, 5, 7, 9]
    if direction in {"left", "right"}:
        for rect_h in odd_lengths:
            for rect_w in range(3, 9):
                if rect_h + 2 <= h and rect_w + 4 <= w:
                    options.append((rect_h, rect_w))
    else:
        for rect_h in range(3, 9):
            for rect_w in odd_lengths:
                if rect_h + 4 <= h and rect_w + 2 <= w:
                    options.append((rect_h, rect_w))
    return options


def _placement_ranges(
    direction: str, rect_h: int, rect_w: int, grid_shape: tuple[int, int]
) -> tuple[range, range] | None:
    h, w = grid_shape
    if direction == "right":
        r_range = range(1, h - rect_h)
        c_range = range(1, w - rect_w - 2)
    elif direction == "left":
        r_range = range(1, h - rect_h)
        c_range = range(2, w - rect_w - 1)
    elif direction == "down":
        r_range = range(1, h - rect_h - 2)
        c_range = range(1, w - rect_w)
    else:
        assert direction == "up"
        r_range = range(2, h - rect_h - 1)
        c_range = range(1, w - rect_w)
    if not r_range or not c_range:
        return None
    return r_range, c_range


def _expand_with_halo(
    cells: set[tuple[int, int]], grid_shape: tuple[int, int], radius: int = 1
) -> set[tuple[int, int]]:
    h, w = grid_shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _footprint(
    bbox: tuple[int, int, int, int], direction: str, grid_shape: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    arrow_cells, moved_cells = _arrow_cells_for_bbox(bbox, direction)
    occupied = rect_cells | set(arrow_cells) | set(moved_cells)
    return occupied, _expand_with_halo(occupied, grid_shape, radius=1)


def _render_input(specs: list[tuple[int, int, int, int, int, str]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r0, c0, r1, c1, color, direction in specs:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        arrow_cells, _ = _arrow_cells_for_bbox((r0, c0, r1, c1), direction)
        for r, c in arrow_cells:
            grid[r, c] = ARROW
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(2, 4)
    h = random.randint(max(14, 4 * n_rects + 8), 26)
    w = random.randint(max(14, 4 * n_rects + 8), 26)
    colors = random.sample(RECT_COLORS, n_rects)
    directions = random.sample(list(DIRECTIONS), n_rects)

    reserved: set[tuple[int, int]] = set()
    specs: list[tuple[int, int, int, int, int, str]] = []

    for color, direction in zip(colors, directions):
        size_options = _direction_size_options(direction, (h, w))
        random.shuffle(size_options)
        placed = False
        for rect_h, rect_w in size_options[:]:
            ranges = _placement_ranges(direction, rect_h, rect_w, (h, w))
            if ranges is None:
                continue
            r_range, c_range = ranges
            positions = [(r0, c0) for r0 in r_range for c0 in c_range]
            random.shuffle(positions)
            for r0, c0 in positions[:PLACEMENT_ATTEMPTS]:
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                occupied, halo = _footprint(bbox, direction, (h, w))
                if occupied & reserved or halo & reserved:
                    continue
                specs.append((bbox[0], bbox[1], bbox[2], bbox[3], color, direction))
                reserved |= halo
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    grid = _render_input(specs, (h, w))
    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == REVEAL):
        return None
    if np.any(out == ARROW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid peeling-rectangle input")
