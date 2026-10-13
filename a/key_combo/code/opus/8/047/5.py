from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:  # pragma: no cover - fallback keeps standalone imports working
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(cell) for cell in row) for row in np.asarray(mask, dtype=int))


GLYPH_SPECS = (
    (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int), Color.BLUE),
    (np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int), Color.RED),
    (np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int), Color.GREEN),
    (np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int), Color.YELLOW),
)
GLYPH_TO_COLOR = {_mask_key(mask): int(color) for mask, color in GLYPH_SPECS}
KEY_TO_MASK = {key: np.array(key, dtype=bool) for key in GLYPH_TO_COLOR}
COLOR_TO_CORNERS = {
    int(Color.BLUE): ((-1, -1), (2, 2)),
    int(Color.RED): ((-1, 2), (2, -1)),
    int(Color.GREEN): ((-1, -1), (-1, 2), (2, -1), (2, 2)),
    int(Color.YELLOW): ((-1, -1), (-1, 2), (2, -1), (2, 2)),
}

INPUT_COLORS = {int(Color.BLACK), int(Color.GREY)}
PAIR_COUNT_OPTIONS = (3, 4)
GEN_HEIGHT_RANGE = (15, 20)
GEN_WIDTH_RANGE = (15, 20)
GEN_ATTEMPTS = 400
ROW_STEP = 7
COL_STEP = 4
_GEN_CALL_COUNT = 0


def _grey_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != Color.GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == Color.GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _extract_pairs(grid: np.ndarray) -> list[tuple[int, int, tuple[tuple[int, ...], ...]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "expected only black and grey input colors"

    squares: list[tuple[int, int]] = []
    glyph_like_components = 0
    for cells in _grey_components(g):
        rows = cells[:, 0]
        cols = cells[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        area = int(cells.shape[0])
        if area == 4 and r1 - r0 == 1 and c1 - c0 == 1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY):
            squares.append((r0, c0))
        else:
            glyph_like_components += 1

    assert len(squares) in PAIR_COUNT_OPTIONS, "expected 3 or 4 square/glyph pairs"
    assert glyph_like_components >= len(squares), "expected at least one glyph component per square"

    pairs: list[tuple[int, int, tuple[tuple[int, ...], ...]]] = []
    seen_keys: set[tuple[tuple[int, ...], ...]] = set()
    h, w = g.shape
    for r0, c0 in sorted(squares):
        assert 4 <= r0 <= h - 3, "square is too close to the top or bottom edge"
        assert 1 <= c0 <= w - 3, "square is too close to the left or right edge"
        glyph = g[r0 - 4 : r0 - 1, c0 : c0 + 3]
        assert glyph.shape == (3, 3), "expected a 3x3 glyph directly above each square"
        assert np.count_nonzero(glyph == Color.GREY) == 5, "expected a five-pixel glyph"
        assert np.all(g[r0 - 1, c0 : c0 + 3] == Color.BLACK), "expected a one-row black gap below each glyph"
        key = _mask_key(glyph == Color.GREY)
        assert key in GLYPH_TO_COLOR, "unrecognized glyph pattern"
        assert key not in seen_keys, "expected pairwise distinct glyph patterns"
        seen_keys.add(key)
        pairs.append((r0, c0, key))

    assert len(seen_keys) >= 3, "expected at least three distinct glyphs"
    return pairs


def _paint_decorations(grid: np.ndarray, pairs: list[tuple[int, int, tuple[tuple[int, ...], ...]]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for r0, c0, key in pairs:
        color = GLYPH_TO_COLOR[key]
        for dr, dc in COLOR_TO_CORNERS[color]:
            rr, cc = r0 + dr, c0 + dc
            assert 0 <= rr < h and 0 <= cc < w, "corner decoration must stay in bounds"
            assert out[rr, cc] == Color.BLACK, "corner decoration must land on black background"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each 3x3 grey glyph above a grey 2x2 square, then decorate only the color-selected outside corners.

    Input structure:
    - The grid is a black background containing 3 or 4 grey 2x2 solid squares.
    - Each square has a unique 3x3 grey five-pixel glyph directly above it, separated by one black row.
    - The glyph starts in the same column as the square and is decoded with a fixed codebook:
      cross -> blue, bottom-T -> red, left-L -> green, right-L -> yellow.

    Rule:
    - Preserve every glyph and every grey square.
    - Decode the glyph above each square to a color.
    - Blue paints only the northwest and southeast outside corners of its paired square.
    - Red paints only the northeast and southwest outside corners of its paired square.
    - Green and yellow paint all four outside diagonal corners of their paired square.

    Key insight:
    - The glyph does not move or recolor the square itself; it first selects a color, and that color then selects which corner markers are emitted.
    """
    pairs = _extract_pairs(np.asarray(grid, dtype=int))
    return _paint_decorations(np.asarray(grid, dtype=int), pairs)


def _candidate_square_positions(height: int, width: int) -> list[tuple[int, int]]:
    rows = [r for r in range(4, height - 1, ROW_STEP) if r + 2 < height]
    cols = [c for c in range(1, width - 1, COL_STEP) if c + 2 < width]
    return [(r, c) for r in rows for c in cols]


def _build_input(
    height: int,
    width: int,
    placements: list[tuple[int, int]],
    glyph_keys: list[tuple[tuple[int, ...], ...]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r0, c0), key in zip(placements, glyph_keys):
        glyph_mask = KEY_TO_MASK[key]
        glyph = grid[r0 - 4 : r0 - 1, c0 : c0 + 3]
        glyph[glyph_mask] = Color.GREY
        grid[r0 : r0 + 2, c0 : c0 + 2] = Color.GREY
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return _paint_decorations(grid, _extract_pairs(grid))


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, count: int) -> bool:
    try:
        pairs = _extract_pairs(grid)
    except AssertionError:
        return False
    if len(pairs) != count:
        return False
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False
    if np.array_equal(grid, expected):
        return False
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    for r0, c0, key in pairs:
        color = GLYPH_TO_COLOR[key]
        corners = {(r0 + dr, c0 + dc) for dr, dc in COLOR_TO_CORNERS[color]}
        for rr, cc in corners:
            if actual[rr, cc] != color:
                return False
        if color == Color.BLUE and len(corners) != 2:
            return False
        if color == Color.RED and len(corners) != 2:
            return False
        if color in (Color.GREEN, Color.YELLOW) and len(corners) != 4:
            return False
        if not np.array_equal(actual[r0 : r0 + 2, c0 : c0 + 2], grid[r0 : r0 + 2, c0 : c0 + 2]):
            return False
        if not np.array_equal(actual[r0 - 4 : r0 - 1, c0 : c0 + 3], grid[r0 - 4 : r0 - 1, c0 : c0 + 3]):
            return False
    return True


def _gen_attempt(counter: int) -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    candidates = _candidate_square_positions(height, width)
    count_options = [count for count in PAIR_COUNT_OPTIONS if count <= len(candidates)]
    if not count_options:
        return None

    count = count_options[counter % len(count_options)]
    placements = random.sample(candidates, count)
    glyph_items = random.sample(list(GLYPH_TO_COLOR.items()), count)
    random.shuffle(glyph_items)
    glyph_keys = [key for key, _ in glyph_items]

    grid = _build_input(height, width, placements, glyph_keys)
    expected = _expected_output(grid)
    if not _semantic_checks(grid, expected, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    base_counter = _GEN_CALL_COUNT
    for extra in range(GEN_ATTEMPTS):
        grid = _gen_attempt(base_counter + extra)
        if grid is not None:
            _GEN_CALL_COUNT = base_counter + extra + 1
            return grid
    raise RuntimeError("failed to generate a valid glyph-corner-decoration input")
