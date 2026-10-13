import random
from collections import Counter

import numpy as np


BG = 0
OUT_BLUE = 1
BARRIER = 2
OUT_GREEN = 3
OUT_YELLOW = 4
GLYPH = 5
OUT_PINK = 6
OUT_ORANGE = 7

LENGTH_TO_COLOR = {
    1: OUT_BLUE,
    2: OUT_GREEN,
    3: OUT_YELLOW,
    4: OUT_PINK,
    5: OUT_ORANGE,
}

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

PLUS_MASK = frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)})
VERTICAL_MASK = frozenset({(-1, 0), (0, 0), (1, 0)})
HORIZONTAL_MASK = frozenset({(0, -1), (0, 0), (0, 1)})
L_MASK = frozenset({(-1, 0), (0, -1), (0, 0)})

SHAPE_TO_DIRECTION = {
    PLUS_MASK: UP,
    VERTICAL_MASK: DOWN,
    HORIZONTAL_MASK: RIGHT,
    L_MASK: LEFT,
}

GEN_SHAPES = [
    ("plus", PLUS_MASK, UP),
    ("vertical", VERTICAL_MASK, DOWN),
    ("horizontal", HORIZONTAL_MASK, RIGHT),
    ("lshape", L_MASK, LEFT),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _component_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    pts = set(coords)
    best = None
    best_deg = -1
    for r, c in coords:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        key = (deg, -r, -c)
        if best is None or key > best_deg_key:
            best = (r, c)
            best_deg = deg
            best_deg_key = key
    assert best is not None
    return best


def _classify_glyph(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    center = _component_center(coords)
    offsets = frozenset((r - center[0], c - center[1]) for r, c in coords)
    assert offsets in SHAPE_TO_DIRECTION, f"unknown glyph shape: {sorted(offsets)}"
    return center, SHAPE_TO_DIRECTION[offsets], set(coords)


def _ray_length(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], own_cells: set[tuple[int, int]]) -> int:
    h, w = grid.shape
    dr, dc = direction
    r, c = center

    while 0 <= r < h and 0 <= c < w and (r, c) in own_cells:
        r += dr
        c += dc

    length = 0
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != BARRIER:
        length += 1
        r += dr
        c += dc
    return length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each gray glyph into a ray direction, measure how far that ray runs before a red blocker, and compress the lengths into a one-row color code.

    Input structure:
    - The grid is mostly black background with several separate gray glyphs and red barrier pixels or short red line segments.
    - Each glyph is a small 4-connected shape built from gray cells: plus, vertical bar, horizontal bar, or a 3-pixel L.
    - Glyphs do not overlap each other, and the relevant barrier for each glyph lies somewhere along its decoded direction.

    Rule:
    - Identify every gray glyph, locate its center pixel, and decode its shape as a direction: plus -> up, vertical bar -> down, horizontal bar -> right, L -> left.
    - Starting from the glyph center and moving outward past the glyph itself, count the cells along that direction until the next step would hit a red barrier or leave the grid; that count is the ray length.
    - Sort glyphs by center position from top to bottom, then left to right, and output a 1xN row whose colors encode the corresponding lengths: 1 -> blue, 2 -> green, 3 -> yellow, 4 -> pink, 5 -> orange.

    Key insight:
    - The gray shape does not name an output color directly; it first encodes a direction, the barrier fixes a distance along that direction, and only that measured distance determines the final output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_color_components(g, GLYPH)
    assert 3 <= len(components) <= 5, "expected 3-5 gray glyphs"

    decoded: list[tuple[tuple[int, int], int]] = []
    for coords in components:
        center, direction, own_cells = _classify_glyph(coords)
        length = _ray_length(g, center, direction, own_cells)
        assert length in LENGTH_TO_COLOR, f"unsupported ray length: {length}"
        decoded.append((center, LENGTH_TO_COLOR[length]))

    decoded.sort(key=lambda item: item[0])
    return np.array([[color for _center, color in decoded]], dtype=int)


def _glyph_pixels(center: tuple[int, int], offsets: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in offsets}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _valid_center_bounds(h: int, w: int, direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            stop_r = r + dr * (length + 2)
            stop_c = c + dc * (length + 2)
            if 0 <= stop_r < h and 0 <= stop_c < w:
                centers.append((r, c))
    random.shuffle(centers)
    return centers


def _barrier_segments(
    h: int,
    w: int,
    stop: tuple[int, int],
    direction: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    sr, sc = stop
    segments: list[set[tuple[int, int]]] = []
    lengths = [1, 2, 3]
    random.shuffle(lengths)
    if direction in (UP, DOWN):
        for seg_len in lengths:
            starts = list(range(sc - seg_len + 1, sc + 1))
            random.shuffle(starts)
            for c0 in starts:
                if c0 < 0 or c0 + seg_len > w:
                    continue
                seg = {(sr, c0 + i) for i in range(seg_len)}
                if stop in seg:
                    segments.append(seg)
    else:
        for seg_len in lengths:
            starts = list(range(sr - seg_len + 1, sr + 1))
            random.shuffle(starts)
            for r0 in starts:
                if r0 < 0 or r0 + seg_len > h:
                    continue
                seg = {(r0 + i, sc) for i in range(seg_len)}
                if stop in seg:
                    segments.append(seg)
    return segments


def _empty_corridor(center: tuple[int, int], direction: tuple[int, int], length: int, own_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = center
    while (r, c) in own_cells:
        r += dr
        c += dc
    cells = []
    for _ in range(length):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _place_glyphs(
    h: int,
    w: int,
    specs: list[tuple[str, frozenset[tuple[int, int]], tuple[int, int], int]],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved_corridors: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    placed: list[dict] = []

    ordered_specs = sorted(specs, key=lambda item: item[3], reverse=True)
    for shape_name, offsets, direction, length in ordered_specs:
        placed_this = False
        for center in _valid_center_bounds(h, w, direction, length):
            if any(_chebyshev(center, other) < 4 for other in centers):
                continue

            glyph = _glyph_pixels(center, offsets)
            if glyph & occupied:
                continue

            corridor = _empty_corridor(center, direction, length, glyph)
            if any(cell in occupied for cell in corridor):
                continue

            stop = (corridor[-1][0] + direction[0], corridor[-1][1] + direction[1])
            for segment in _barrier_segments(h, w, stop, direction):
                if segment & occupied:
                    continue
                if segment & reserved_corridors:
                    continue

                for r, c in glyph:
                    grid[r, c] = GLYPH
                for r, c in segment:
                    grid[r, c] = BARRIER

                occupied.update(glyph)
                occupied.update(segment)
                reserved_corridors.update(corridor)
                centers.append(center)
                placed.append(
                    {
                        "shape": shape_name,
                        "center": center,
                        "direction": direction,
                        "length": length,
                        "glyph": glyph,
                        "corridor": corridor,
                        "segment": segment,
                    }
                )
                placed_this = True
                break
            if placed_this:
                break
        if not placed_this:
            return None
    return grid, placed


def _add_distractors(grid: np.ndarray, placed: list[dict]) -> None:
    h, w = grid.shape
    blocked = set()
    for item in placed:
        blocked.update(item["glyph"])
        blocked.update(item["corridor"])
        blocked.update(item["segment"])

    n_extra = random.randint(0, 2)
    for _ in range(n_extra):
        vertical = bool(random.getrandbits(1))
        seg_len = random.randint(1, 3)
        for _attempt in range(50):
            if vertical:
                r0 = random.randint(0, h - seg_len)
                c0 = random.randint(0, w - 1)
                seg = {(r0 + i, c0) for i in range(seg_len)}
            else:
                r0 = random.randint(0, h - 1)
                c0 = random.randint(0, w - seg_len)
                seg = {(r0, c0 + i) for i in range(seg_len)}
            if any(int(grid[r, c]) != BG for r, c in seg):
                continue
            if seg & blocked:
                continue
            for r, c in seg:
                grid[r, c] = BARRIER
            blocked.update(seg)
            break


def _specs_for_example(h: int, w: int) -> list[tuple[str, frozenset[tuple[int, int]], tuple[int, int], int]]:
    min_dim = min(h, w)
    if min_dim <= 12:
        n_glyphs = random.randint(3, 4)
    else:
        n_glyphs = random.randint(4, 5)

    base_shapes = random.sample(GEN_SHAPES, min(4, n_glyphs))
    if n_glyphs > 4:
        base_shapes.append(random.choice(GEN_SHAPES))
    random.shuffle(base_shapes)

    lengths = random.sample([1, 2, 3, 4, 5], n_glyphs)
    random.shuffle(lengths)
    return [
        (shape_name, offsets, direction, length)
        for (shape_name, offsets, direction), length in zip(base_shapes, lengths)
    ]


def _oracle_output(placed: list[dict]) -> np.ndarray:
    ordered = sorted(placed, key=lambda item: item["center"])
    return np.array([[LENGTH_TO_COLOR[item["length"]] for item in ordered]], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 14)
    w = random.randint(11, 14)
    specs = _specs_for_example(h, w)

    placed_result = _place_glyphs(h, w, specs)
    if placed_result is None:
        return None
    grid, placed = placed_result
    _add_distractors(grid, placed)

    if len(_find_color_components(grid, GLYPH)) != len(placed):
        return None

    expected = _oracle_output(placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
