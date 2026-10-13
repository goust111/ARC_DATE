from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
FRAME_COLORS = tuple(color for color in range(1, 10) if color != MAROON)
DIRECTIONS = ("up", "right", "down", "left")
MISSING_TO_DIRECTION = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_thin_rect_frame(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return False

    expected_area = 2 * (height + width) - 4
    if len(cells) != expected_area:
        return False

    cell_set = set(cells)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False
    return True


def _read_frame_side_colors(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, int] | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    top_vals = np.unique(grid[r0, c0 : c1 + 1])
    bottom_vals = np.unique(grid[r1, c0 : c1 + 1])
    left_vals = np.unique(grid[r0 + 1 : r1, c0])
    right_vals = np.unique(grid[r0 + 1 : r1, c1])
    if not (top_vals.size == bottom_vals.size == left_vals.size == right_vals.size == 1):
        return None

    colors = {
        "up": int(top_vals[0]),
        "down": int(bottom_vals[0]),
        "left": int(left_vals[0]),
        "right": int(right_vals[0]),
    }
    if BLACK in colors.values():
        return None
    if len(set(colors.values())) != 4:
        return None
    return colors


def _opening_direction(interior: np.ndarray) -> str | None:
    maroon_components = _components(interior == MAROON)
    if len(maroon_components) != 1:
        return None
    comp = maroon_components[0]
    if len(comp) != 3:
        return None

    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None

    local_cells = {(r - r0, c - c0) for r, c in comp}
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - local_cells
    if len(missing) != 1:
        return None
    return MISSING_TO_DIRECTION[missing.pop()]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored rectangular fence interior by reading the opening direction of its maroon L marker.

    Input structure:
    - The background is black.
    - One or more hollow rectangular fences appear as one-pixel-thick multicolor rings.
      Each fence has four distinct side colors, with the corner pixels belonging to the
      top or bottom side colors.
    - Inside each fence there is exactly one maroon 3-pixel L-shape, while the rest of
      the fence interior is black.

    Rule:
    - Detect every closed rectangular fence and the maroon L-shape inside it.
    - Infer the L opening direction from its missing corner using the cardinal mapping
      top-left -> up, top-right -> right, bottom-right -> down, bottom-left -> left.
    - Use the color of the fence side that matches that direction to fill every black
      pixel inside the fence, leaving the fence and the maroon L unchanged.

    Key insight:
    - The interior L is a directional cue: its orientation chooses one of the four side
      colors, and only then can the enclosed black region be filled correctly.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g != BLACK):
        if not _is_thin_rect_frame(comp):
            continue

        bbox = _bbox(comp)
        side_colors = _read_frame_side_colors(g, bbox)
        if side_colors is None:
            continue

        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        direction = _opening_direction(interior)
        if direction is None:
            continue

        fill_color = side_colors[direction]
        out_region = out[r0 + 1 : r1, c0 + 1 : c1]
        out_region[out_region == BLACK] = fill_color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, colors: dict[str, int]) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = colors["up"]
    grid[r1, c0 : c1 + 1] = colors["down"]
    grid[r0 + 1 : r1, c0] = colors["left"]
    grid[r0 + 1 : r1, c1] = colors["right"]


def _l_patch(direction: str) -> np.ndarray:
    missing = next(key for key, value in MISSING_TO_DIRECTION.items() if value == direction)
    patch = np.full((2, 2), MAROON, dtype=int)
    patch[missing] = BLACK
    return patch


def _choose_canvas_shape() -> tuple[int, int]:
    rows = random.randint(18, 30)
    cols = random.randint(18, 30)
    if random.random() < 0.55:
        cols = max(18, min(30, rows + random.randint(-3, 3)))
    return rows, cols


def _choose_fence_count(rows: int, cols: int) -> int:
    max_count = 1
    if min(rows, cols) >= 20:
        max_count = 2
    if min(rows, cols) >= 23:
        max_count = 3
    if min(rows, cols) >= 27:
        max_count = 4
    return random.randint(1, max_count)


def _sample_sizes(rows: int, cols: int, count: int) -> list[tuple[int, int]]:
    size_cap = {1: 13, 2: 11, 3: 9, 4: 8}[count]
    sizes: list[tuple[int, int]] = []
    for _ in range(count):
        h = random.randint(6, min(size_cap, rows - 2))
        w = random.randint(6, min(size_cap, cols - 2))
        sizes.append((h, w))
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    return sizes


def _place_rectangles(
    rows: int, cols: int, sizes: list[tuple[int, int]], pad: int = 1
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((rows, cols), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []

    for height, width in sizes:
        positions: list[tuple[int, int]] = []
        for r0 in range(1, rows - height):
            for c0 in range(1, cols - width):
                if bool(blocked[r0 : r0 + height, c0 : c0 + width].any()):
                    continue
                positions.append((r0, c0))
        if not positions:
            return None

        r0, c0 = random.choice(positions)
        placed.append((r0, c0, height, width))
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(rows, r0 + height + pad)
        cc1 = min(cols, c0 + width + pad)
        blocked[rr0:rr1, cc0:cc1] = True

    return placed


def _sample_l_origin(r0: int, c0: int, height: int, width: int) -> tuple[int, int]:
    interior_h = height - 2
    interior_w = width - 2
    local_r = random.randint(1, interior_h - 3)
    local_c = random.randint(1, interior_w - 3)
    return r0 + 1 + local_r, c0 + 1 + local_c


def _build_case(rows: int, cols: int, placements: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BLACK, dtype=int)
    out = np.full((rows, cols), BLACK, dtype=int)

    for r0, c0, height, width in placements:
        top, bottom, left, right = random.sample(FRAME_COLORS, 4)
        colors = {"up": top, "down": bottom, "left": left, "right": right}
        direction = random.choice(DIRECTIONS)
        lr, lc = _sample_l_origin(r0, c0, height, width)
        patch = _l_patch(direction)

        _draw_frame(inp, r0, c0, height, width, colors)
        _draw_frame(out, r0, c0, height, width, colors)

        inp[lr : lr + 2, lc : lc + 2] = patch

        out[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1] = colors[direction]
        out[lr : lr + 2, lc : lc + 2][patch == MAROON] = MAROON

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rows, cols = _choose_canvas_shape()
        count = _choose_fence_count(rows, cols)
        sizes = _sample_sizes(rows, cols, count)
        placements = _place_rectangles(rows, cols, sizes)
        if placements is None:
            continue

        inp, expected = _build_case(rows, cols, placements)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid fence-pointer sample")
