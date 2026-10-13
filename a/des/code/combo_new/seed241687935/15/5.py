import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


BG = 8
NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_RETRIES = 400
PALETTE_RESERVED_COLS = 5
PALETTE_FILL_WIDTH = 1


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(np.asarray(grid, dtype=int).ravel()).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_palette(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    mapping: dict[tuple[int, int], int] = {}
    mask = np.zeros((h, w), dtype=bool)
    r = 0

    while r + 1 < h and w >= 3:
        a = int(grid[r, 0])
        b = int(grid[r, 1])
        if a == bg or b == bg or a == b:
            break
        if not (np.all(grid[r : r + 2, 0] == a) and np.all(grid[r : r + 2, 1] == b)):
            break

        fill_color = None
        fill_width = 0
        for c in range(2, min(w, 4)):
            col = grid[r : r + 2, c]
            if not np.all(col == col[0]) or int(col[0]) == bg:
                break
            if fill_color is None:
                fill_color = int(col[0])
                fill_width = 1
            elif int(col[0]) == fill_color:
                fill_width += 1
            else:
                break
        if fill_color is None:
            break

        key = tuple(sorted((a, b)))
        if key in mapping:
            assert mapping[key] == fill_color, "duplicate palette entry must map to the same fill color"
        mapping[key] = fill_color
        mask[r : r + 2, : 2 + fill_width] = True

        if r + 2 >= h:
            r += 2
            break
        if np.all(grid[r + 2, : 2 + fill_width] == bg):
            r += 3
            continue
        r += 2
        break

    assert mapping, "expected a non-empty top-left palette"
    return mapping, mask


def _rectangle_colors(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    if set(coords) != expected:
        return None

    top_color = int(grid[r0, c0])
    if top_color == bg:
        return None
    if not (np.all(grid[r0, c0 : c1 + 1] == top_color) and np.all(grid[r1, c0 : c1 + 1] == top_color)):
        return None

    left_side = grid[r0 + 1 : r1, c0]
    right_side = grid[r0 + 1 : r1, c1]
    if left_side.size == 0:
        return None
    side_color = int(left_side[0])
    if side_color == bg or side_color == top_color:
        return None
    if not (np.all(left_side == side_color) and np.all(right_side == side_color)):
        return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None

    return tuple(sorted((top_color, side_color))), (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each two-color rectangular enclosure with the indicator color looked up from the corner palette.

    Input structure:
    - Azure is the dominant background color.
    - The top-left corner holds a vertical palette of 2x2 color-pair blocks; each block is followed by a solid vertical indicator strip, and palette entries are separated by one Azure row.
    - Elsewhere the grid contains one-pixel-thick rectangular outlines whose top and bottom edges share one color while the left and right edges share another; corners use the horizontal-edge color.
    - Every rectangle interior is entirely Azure.

    Rule:
    - Read each palette entry as an unordered pair of boundary colors mapped to the solid indicator color on its right.
    - For every closed rectangular outline, identify its horizontal-edge color and vertical-edge color and form the corresponding unordered pair.
    - Look up that pair in the palette and replace all Azure cells inside the rectangle with the mapped fill color.
    - Leave the palette and all outline pixels unchanged.

    Key insight:
    - The border colors do not directly specify the fill; they only act as an order-independent key into a separate palette embedded in the grid corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, palette_mask = _parse_palette(g, bg)

    out = g.copy()
    shape_mask = (g != bg) & ~palette_mask
    for coords in _components(shape_mask):
        result = _rectangle_colors(g, coords, bg)
        if result is None:
            continue
        key, (r0, c0, r1, c1) = result
        assert key in mapping, "every rectangle color pair must appear in the palette"
        out[r0 + 1 : r1, c0 + 1 : c1] = int(mapping[key])
    return out


def _paint_rectangle(
    grid: np.ndarray, r0: int, c0: int, height: int, width: int, horiz: int, vert: int
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = horiz
    grid[r1, c0 : c1 + 1] = horiz
    grid[r0 + 1 : r1, c0] = vert
    grid[r0 + 1 : r1, c1] = vert


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_entries = rng.randint(2, 4)
    palette_height = 3 * n_entries - 1

    max_edge_colors = min(6, len(NON_BG_COLORS) - 2)
    edge_color_count = rng.randint(4, max_edge_colors)
    edge_colors = rng.sample(NON_BG_COLORS, edge_color_count)
    pair_choices = list(combinations(edge_colors, 2))
    if len(pair_choices) < n_entries:
        return None
    rng.shuffle(pair_choices)
    chosen_pairs = pair_choices[:n_entries]

    fills: dict[tuple[int, int], int] = {}
    palette_entries: list[tuple[int, int, int]] = []
    for pair in chosen_pairs:
        fill_options = [c for c in NON_BG_COLORS if c not in pair]
        if not fill_options:
            return None
        fill = rng.choice(fill_options)
        fills[tuple(sorted(pair))] = fill
        left, right = pair if rng.random() < 0.5 else (pair[1], pair[0])
        palette_entries.append((left, right, fill))

    n_rectangles = rng.randint(n_entries, min(n_entries + 2, 6))
    assignments = list(chosen_pairs)
    while len(assignments) < n_rectangles:
        assignments.append(rng.choice(chosen_pairs))
    rng.shuffle(assignments)

    min_height = max(palette_height + 1, 12)
    grid_h = rng.randint(min_height, 30)
    grid_w = rng.randint(16, 30)
    if grid_w - PALETTE_RESERVED_COLS < 5:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for idx, (left, right, fill) in enumerate(palette_entries):
        r0 = 3 * idx
        grid[r0 : r0 + 2, 0] = left
        grid[r0 : r0 + 2, 1] = right
        grid[r0 : r0 + 2, 2 : 2 + PALETTE_FILL_WIDTH] = fill

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[:, :PALETTE_RESERVED_COLS] = True

    if n_rectangles >= 5:
        h_max, w_max = 6, 7
    elif n_rectangles == 4:
        h_max, w_max = 7, 8
    else:
        h_max, w_max = 8, 10
    h_max = min(h_max, grid_h)
    w_max = min(w_max, grid_w - PALETTE_RESERVED_COLS)
    if h_max < 4 or w_max < 4:
        return None

    for pair in assignments:
        placed = False
        for _ in range(300):
            height = rng.randint(4, h_max)
            width = rng.randint(4, w_max)
            r0 = rng.randint(0, grid_h - height)
            c0 = rng.randint(PALETTE_RESERVED_COLS, grid_w - width)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + height + 1)
            cc1 = min(grid_w, c0 + width + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            colors = list(pair)
            if rng.random() < 0.5:
                horiz, vert = colors[0], colors[1]
            else:
                horiz, vert = colors[1], colors[0]
            _paint_rectangle(grid, r0, c0, height, width, horiz, vert)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_RETRIES} attempts")
