import random
from collections import defaultdict

import numpy as np


BG = 0
ORANGE = 7
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

PAIR_COUNT_RANGE = (3, 5)
HEIGHT_MAX = 28
WIDTH_MAX = 28
ROW_GAP = 1
MIN_INTERSECTIONS = 2
MAX_INTERSECTIONS = 7
MAX_GENERATION_ATTEMPTS = 500


def _sample_spaced_positions(n: int, lo: int, hi: int, gap: int) -> list[int] | None:
    """Sample sorted positions in [lo, hi] with at least `gap` empty cells between neighbors."""
    span = hi - lo + 1
    needed = n + (n - 1) * gap
    if span < needed:
        return None
    compressed_span = span - (n - 1) * gap
    picks = sorted(random.sample(range(compressed_span), n))
    return [lo + picks[i] + i * gap for i in range(n)]


def _phase_marker(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marked = [(r, c) for r, c in corners if int(grid[r, c]) != BG]
    assert len(marked) == 1, "expected exactly one non-background corner marker"
    r, c = marked[0]
    assert int(grid[r, c]) == ORANGE, "corner marker must be orange"
    return r, c


def _path_specs(grid: np.ndarray) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, ORANGE)]
    assert colors, "expected at least one color pair"
    for color in colors:
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = sorted((tuple(map(int, rc)) for rc in coords), key=lambda rc: (rc[1], rc[0]))
        assert r0 != r1 and c0 != c1, f"color {color} endpoints must differ in both row and column"
        specs.append(
            {
                "color": color,
                "left_r": r0,
                "left_c": c0,
                "right_r": r1,
                "right_c": c1,
                "h_row": r0,
                "h_c0": c0,
                "h_c1": c1,
                "v_col": c1,
                "v_r0": min(r0, r1),
                "v_r1": max(r0, r1),
            }
        )
    return specs


def _collect_occupancy(paths: list[dict[str, int]]) -> dict[tuple[int, int], dict[str, set[int]]]:
    occ: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(lambda: {"h": set(), "v": set()})
    for path in paths:
        color = int(path["color"])
        for c in range(path["h_c0"], path["h_c1"] + 1):
            occ[(path["h_row"], c)]["h"].add(color)
        for r in range(path["v_r0"], path["v_r1"] + 1):
            occ[(r, path["v_col"])]["v"].add(color)
    return occ


def _intersections(occ: dict[tuple[int, int], dict[str, set[int]]]) -> list[tuple[int, int, int, int]]:
    crossings: list[tuple[int, int, int, int]] = []
    for (r, c), segs in occ.items():
        if len(segs["h"]) > 1 or len(segs["v"]) > 1:
            raise AssertionError("paths may only meet as one horizontal segment crossing one vertical segment")
        if not segs["h"] or not segs["v"]:
            continue
        h_color = next(iter(segs["h"]))
        v_color = next(iter(segs["v"]))
        if h_color == v_color:
            continue
        crossings.append((r, c, h_color, v_color))
    crossings.sort(key=lambda item: (item[1], item[0]))
    return crossings


def _endpoints(paths: list[dict[str, int]]) -> dict[tuple[int, int], int]:
    pts: dict[tuple[int, int], int] = {}
    for path in paths:
        left = (path["left_r"], path["left_c"])
        right = (path["right_r"], path["right_c"])
        assert left not in pts and right not in pts, "endpoints must not overlap"
        pts[left] = int(path["color"])
        pts[right] = int(path["color"])
    return pts


def _validate_path_layout(paths: list[dict[str, int]]) -> list[tuple[int, int, int, int]] | None:
    occ = _collect_occupancy(paths)
    endpoints = _endpoints(paths)

    for coord, color in endpoints.items():
        segs = occ.get(coord)
        if segs is None:
            return None
        if segs["h"] - {color} or segs["v"] - {color}:
            return None

    crossings = _intersections(occ)
    if not (MIN_INTERSECTIONS <= len(crossings) <= MAX_INTERSECTIONS):
        return None
    return crossings


def _render_input(shape: tuple[int, int], paths: list[dict[str, int]], phase: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[phase] = ORANGE
    for path in paths:
        grid[path["left_r"], path["left_c"]] = path["color"]
        grid[path["right_r"], path["right_c"]] = path["color"]
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each color pair with an L-shaped wire and alternate the visible color at crossings.

    Input structure:
    - The grid has a black background, several non-orange colors that each occur exactly twice, and one orange corner pixel.
    - Each same-color pair marks two endpoints on different rows and different columns.
    - The orange marker is in one of the four corners and acts only as a phase cue.

    Rule:
    - Keep the orange corner marker in place.
    - For each same-color pair, treat the leftmost endpoint as the start, draw horizontally to the other endpoint's column, then draw vertically in that column to the other endpoint.
    - Whenever a horizontal segment of one color and a vertical segment of another color share a cell, sort those crossings by increasing column and then increasing row.
    - If the orange marker is on the left side, the first crossing is numbered 1; if it is on the right side, the first crossing is numbered 2. Odd crossings show the horizontal color, and even crossings show the vertical color.

    Key insight:
    - Every pair uses the same left-to-right L routing, while the orange corner flips the global over-under weaving parity across the ordered crossing list.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert int(np.min(g)) >= 0, "grid colors must be non-negative"
    assert int(np.count_nonzero(g == ORANGE)) == 1, "expected exactly one orange phase marker"

    phase_r, phase_c = _phase_marker(g)
    paths = _path_specs(g)
    occ = _collect_occupancy(paths)
    assert (phase_r, phase_c) not in occ, "phase marker must not lie on a path"

    out = np.zeros_like(g)
    out[phase_r, phase_c] = ORANGE

    for path in paths:
        color = int(path["color"])
        out[path["h_row"], path["h_c0"] : path["h_c1"] + 1] = color
        out[path["v_r0"] : path["v_r1"] + 1, path["v_col"]] = color

    first_number = 1 if phase_c < g.shape[1] / 2 else 2
    for offset, (r, c, h_color, v_color) in enumerate(_intersections(occ)):
        out[r, c] = h_color if (first_number + offset) % 2 == 1 else v_color

    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    min_height = max(4 * pair_count + 1, 15)
    height = random.randint(min_height, HEIGHT_MAX)
    width = random.randint(15, WIDTH_MAX)

    row_slots = _sample_spaced_positions(2 * pair_count, 1, height - 2, ROW_GAP)
    if row_slots is None:
        return None

    left_band = list(range(1, width // 3 + 1))
    right_lo = max(width // 2, width // 3 + 2)
    right_band = list(range(right_lo, width - 1))
    if len(left_band) < pair_count or len(right_band) < pair_count:
        return None

    left_even_slots = random.choice((False, True))
    left_rows = row_slots[::2] if left_even_slots else row_slots[1::2]
    right_rows_pool = row_slots[1::2] if left_even_slots else row_slots[::2]
    right_rows = right_rows_pool[:]
    random.shuffle(right_rows)

    left_cols = random.sample(left_band, pair_count)
    right_cols = random.sample(right_band, pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)

    paths = []
    for i in range(pair_count):
        paths.append(
            {
                "color": int(colors[i]),
                "left_r": int(left_rows[i]),
                "left_c": int(left_cols[i]),
                "right_r": int(right_rows[i]),
                "right_c": int(right_cols[i]),
                "h_row": int(left_rows[i]),
                "h_c0": int(left_cols[i]),
                "h_c1": int(right_cols[i]),
                "v_col": int(right_cols[i]),
                "v_r0": int(min(left_rows[i], right_rows[i])),
                "v_r1": int(max(left_rows[i], right_rows[i])),
            }
        )

    crossings = _validate_path_layout(paths)
    if crossings is None:
        return None

    phase = random.choice(((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    inp = _render_input((height, width), paths, phase)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape or np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-path ARC input")
