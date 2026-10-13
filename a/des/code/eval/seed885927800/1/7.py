import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
TOP_COLORS = (RED, BLUE, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mono_components(grid: np.ndarray, ignored: set[int]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(g[r, c])
            if color in ignored:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": np.array(coords, dtype=int)})

    return comps


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    bg = _most_common_color(grid)
    pairs = []
    for comp in _mono_components(grid, ignored={bg}):
        coords = comp["coords"]
        assert isinstance(coords, np.ndarray)
        if len(coords) != 2:
            continue
        rows = {int(v) for v in coords[:, 0]}
        cols = sorted(int(v) for v in coords[:, 1])
        if len(rows) == 1 and cols[1] == cols[0] + 1:
            pairs.append(comp)

    assert len(pairs) == 4
    col_counts = Counter(int(c) for comp in pairs for c in comp["coords"][:, 1])
    anchor_cols = [col for col, count in col_counts.items() if count == 4]
    assert len(anchor_cols) == 1
    anchor = int(anchor_cols[0])

    direction_by_color: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    legend_rows = set()

    for comp in pairs:
        color = int(comp["color"])
        coords = comp["coords"]
        cols = sorted(int(v) for v in coords[:, 1])
        row = int(coords[0, 0])
        assert anchor in cols
        other = cols[0] if cols[1] == anchor else cols[1]
        direction_by_color[color] = -1 if other < anchor else 1
        legend_rows.add(row)
        for rr, cc in coords:
            legend_cells.add((int(rr), int(cc)))

    assert len(direction_by_color) == 4
    assert len(legend_rows) == 4
    return direction_by_color, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a bottom-right color legend, then use each tower's height and top color
    to light part of the adjacent gray strip in the mapped direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = _most_common_color(g)
    direction_by_color, legend_cells = _find_legend(g)

    counts = Counter(map(int, g.ravel()))
    gray = max((color for color in counts if color != bg), key=lambda color: counts[color])

    comps = _mono_components(g, ignored={bg, gray})
    tower_tops: list[tuple[int, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        assert isinstance(coords, np.ndarray)
        if len(coords) != 1:
            continue
        r, c = map(int, coords[0])
        if (r, c) in legend_cells:
            continue
        color = int(comp["color"])
        if color in direction_by_color:
            tower_tops.append((r, c, color))

    assert tower_tops

    for r, c, color in tower_tops:
        rr = r + 1
        while rr < h and int(g[rr, c]) == gray:
            rr += 1
        body_len = rr - r - 1
        assert body_len >= 1
        base_r = rr - 1
        height = body_len + 1

        left_is_strip = c - 1 >= 0 and int(g[base_r, c - 1]) == gray
        right_is_strip = c + 1 < w and int(g[base_r, c + 1]) == gray
        assert left_is_strip ^ right_is_strip
        strip_dir = -1 if left_is_strip else 1
        assert strip_dir == direction_by_color[color]

        cc = c + strip_dir
        lit = 0
        while 0 <= cc < w and int(g[base_r, cc]) == gray and lit < height:
            out[base_r, cc] = color
            cc += strip_dir
            lit += 1

    return out


def _tower_coords(base_r: int, base_c: int, height: int, length: int, direction: int) -> list[tuple[int, int]]:
    top_r = base_r - height + 1
    coords = [(top_r, base_c)]
    coords.extend((rr, base_c) for rr in range(top_r + 1, base_r + 1))
    coords.extend((base_r, base_c + direction * step) for step in range(1, length + 1))
    return coords


def _can_place(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _paint_legend(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    direction_by_color: dict[int, int],
    top: int,
    left: int,
) -> None:
    mid = left + 1
    for row_offset, color in enumerate(TOP_COLORS):
        r = top + row_offset
        edge = left if direction_by_color[color] == -1 else left + 2
        inp[r, mid] = color
        inp[r, edge] = color
        out[r, mid] = color
        out[r, edge] = color
        occupied[r, mid] = True
        occupied[r, edge] = True


def _paint_tower(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    base_r: int,
    base_c: int,
    height: int,
    length: int,
    direction: int,
    color: int,
) -> None:
    top_r = base_r - height + 1
    inp[top_r, base_c] = color
    out[top_r, base_c] = color
    occupied[top_r, base_c] = True

    for rr in range(top_r + 1, base_r + 1):
        inp[rr, base_c] = GRAY
        out[rr, base_c] = GRAY
        occupied[rr, base_c] = True

    for step in range(1, length + 1):
        cc = base_c + direction * step
        inp[base_r, cc] = GRAY
        out[base_r, cc] = color if step <= height else GRAY
        occupied[base_r, cc] = True


def _gen_attempt() -> np.ndarray | None:
    n_towers = random.randint(4, 6)
    h = random.randint(20 + n_towers, 28 + n_towers)
    w = random.randint(28 + 2 * (n_towers - 4), 40 + 2 * (n_towers - 4))

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    left_colors = set(random.sample(TOP_COLORS, 2))
    direction_by_color = {color: (-1 if color in left_colors else 1) for color in TOP_COLORS}

    legend_bottom_margin = random.randint(0, 1)
    legend_right_margin = random.randint(0, 2)
    legend_top = h - 4 - legend_bottom_margin
    legend_left = w - 3 - legend_right_margin
    _paint_legend(inp, out, occupied, direction_by_color, legend_top, legend_left)

    tower_colors = list(TOP_COLORS) + random.choices(TOP_COLORS, k=n_towers - len(TOP_COLORS))
    random.shuffle(tower_colors)

    for color in tower_colors:
        direction = direction_by_color[color]
        placed = False
        for _ in range(300):
            height = random.randint(2, 7)
            max_length = min(16, w - 4)
            if max_length < 8:
                return None
            length = random.randint(8, max_length)
            base_r = random.randint(height, h - 2)

            if direction == 1:
                c_min = 1
                c_max = w - length - 2
            else:
                c_min = length + 1
                c_max = w - 2
            if c_min > c_max:
                continue
            base_c = random.randint(c_min, c_max)

            coords = _tower_coords(base_r, base_c, height, length, direction)
            if not _can_place(occupied, coords):
                continue

            _paint_tower(inp, out, occupied, base_r, base_c, height, length, direction, color)
            placed = True
            break

        if not placed:
            return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
