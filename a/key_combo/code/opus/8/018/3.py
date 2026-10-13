from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


TOP_COLOR = Color.RED
BOTTOM_COLOR = Color.BLUE
BLOCKER_COLOR = Color.GREY
BG_COLOR = Color.BLACK

HEIGHT_CHOICES = (13, 15, 17)
WIDTH_CHOICES = (13, 15, 17)
BLOCKER_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 400
MIDDLE_BLOCKER_PROB = 0.4


def _is_gridline_pixel(r: int, c: int) -> bool:
    return (r % 2 == 0) or (c % 2 == 0)


def _nearest_edge_color(r: int, h: int) -> int:
    d_top = r
    d_bottom = (h - 1) - r
    return TOP_COLOR if d_top <= d_bottom else BOTTOM_COLOR


def _blockers(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == BLOCKER_COLOR)
    return [(int(r), int(c)) for r, c in coords]


def _has_visible_truncation(r: int, c: int, h: int) -> bool:
    mid = (h - 1) // 2
    if r <= mid:
        rows = range(1, r)
    else:
        rows = range(r + 1, h - 1)
    return any(_is_gridline_pixel(rr, c) for rr in rows)


def _validate_input(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    h, w = g.shape
    assert h in HEIGHT_CHOICES and w in WIDTH_CHOICES, "unexpected grid size"
    assert h % 2 == 1 and w % 2 == 1, "grid dimensions must be odd"
    assert np.all(g[0] == TOP_COLOR), "top edge bar must be solid red"
    assert np.all(g[h - 1] == BOTTOM_COLOR), "bottom edge bar must be solid blue"
    assert np.all((g[1 : h - 1] == BG_COLOR) | (g[1 : h - 1] == BLOCKER_COLOR)), "interior must contain only background or blockers"

    blockers = _blockers(g)
    assert BLOCKER_COUNT_RANGE[0] <= len(blockers) <= BLOCKER_COUNT_RANGE[1], "unexpected blocker count"
    cols = [c for _r, c in blockers]
    assert len(set(cols)) == len(cols), "each column may contain at most one blocker"
    for r, c in blockers:
        assert 1 <= r <= h - 2, "blocker must stay off the edge bars"
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) != (r, c):
                assert g[nr, nc] != BLOCKER_COLOR, "blockers must be 4-disconnected"
    return blockers


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        color = _nearest_edge_color(r, h)
        for c in range(w):
            if _is_gridline_pixel(r, c):
                out[r, c] = color
            else:
                out[r, c] = BG_COLOR

    mid = (h - 1) // 2
    for rb, cb in _blockers(g):
        if rb <= mid:
            rr = range(1, rb)
        else:
            rr = range(rb + 1, h - 1)
        for r in rr:
            if _is_gridline_pixel(r, cb):
                out[r, cb] = BG_COLOR
        out[rb, cb] = BLOCKER_COLOR

    out[0, :] = TOP_COLOR
    out[h - 1, :] = BOTTOM_COLOR
    return out


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    h, w = inp.shape
    blockers = _validate_input(inp)
    expected = _apply_rule(inp)
    assert np.array_equal(out, expected), "output must match the deterministic rule"

    mid = (h - 1) // 2
    top_branch = any(r <= mid for r, _c in blockers)
    bottom_branch = any(r > mid for r, _c in blockers)
    assert top_branch and bottom_branch, "generator must expose both blocker directions"

    has_gridline_blocker = any(_is_gridline_pixel(r, c) for r, c in blockers)
    has_non_gridline_blocker = any(not _is_gridline_pixel(r, c) for r, c in blockers)
    assert has_gridline_blocker and has_non_gridline_blocker, "generator should mix blocker parities"

    erased_top = 0
    erased_bottom = 0
    for rb, cb in blockers:
        if rb <= mid:
            for r in range(1, rb):
                if _is_gridline_pixel(r, cb):
                    assert out[r, cb] == BG_COLOR, "upper blocker must erase toward the top edge"
                    erased_top += 1
            for r in range(rb + 1, h - 1):
                if _is_gridline_pixel(r, cb):
                    assert out[r, cb] == _nearest_edge_color(r, h), "upper blocker must not erase the far side"
        else:
            for r in range(rb + 1, h - 1):
                if _is_gridline_pixel(r, cb):
                    assert out[r, cb] == BG_COLOR, "lower blocker must erase toward the bottom edge"
                    erased_bottom += 1
            for r in range(1, rb):
                if _is_gridline_pixel(r, cb):
                    assert out[r, cb] == _nearest_edge_color(r, h), "lower blocker must not erase the far side"
        assert out[rb, cb] == BLOCKER_COLOR, "blocker must remain visible"

    assert erased_top > 0 and erased_bottom > 0, "both blocker directions must visibly affect the output"

    red_interior = int(np.count_nonzero(out[1 : mid + 1] == TOP_COLOR))
    blue_interior = int(np.count_nonzero(out[mid + 1 : h - 1] == BOTTOM_COLOR))
    assert red_interior > 0 and blue_interior > 0, "distance-based coloring must leave visible red and blue interiors"

    untouched_oddodd = 0
    for r in range(1, h - 1):
        for c in range(w):
            if not _is_gridline_pixel(r, c) and inp[r, c] == BG_COLOR:
                assert out[r, c] == BG_COLOR, "odd-odd non-grid cells must stay black unless occupied by a blocker"
                untouched_oddodd += 1
    assert untouched_oddodd > 0, "output should visibly preserve non-grid cells"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a parity-defined grid between two edge bars, color each grid pixel by the nearer edge, and erase the near-side segment in blocker columns.

    Input structure:
    - The canvas is an odd-by-odd black grid with a solid red top row and a solid blue bottom row.
    - The interior contains only black background plus several isolated gray single-pixel blockers.
    - Each blocker lies off the edge bars, blockers are 4-disconnected, and no two blockers share a column.

    Rule:
    - Consider only interior rows. A cell is a grid-line pixel iff its row index is even or its column index is even, using 0-based coordinates.
    - Color each interior grid-line pixel red when it is at least as close to the top bar as to the bottom bar; otherwise color it blue. Interior odd-odd cells stay black.
    - For each gray blocker, erase grid-line pixels only in its own column and only on the side of the nearer edge bar: rows above it when the blocker is in the upper half (including the middle row), or rows below it when the blocker is in the lower half.
    - Keep the blocker pixels and both edge bars unchanged.

    Key insight:
    - The pattern is the conjunction of three independent-looking but necessary cues: parity chooses where lines exist, vertical distance chooses each line color, and blocker height chooses which side of its column gets truncated.
    """
    _validate_input(grid)
    return _apply_rule(grid)


def _sample_nonadjacent_position(
    rng: random.Random,
    h: int,
    w: int,
    used: set[tuple[int, int]],
    columns: set[int],
    row_candidates: list[int],
    parity: str,
    require_effect: bool = False,
) -> tuple[int, int] | None:
    cells = [(r, c) for r in row_candidates for c in range(w) if c not in columns]
    rng.shuffle(cells)
    for r, c in cells:
        if parity == "grid" and not _is_gridline_pixel(r, c):
            continue
        if parity == "nongrid" and _is_gridline_pixel(r, c):
            continue
        if require_effect and not _has_visible_truncation(r, c, h):
            continue
        if all(abs(r - ur) + abs(c - uc) != 1 for ur, uc in used):
            return r, c
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(HEIGHT_CHOICES)
    w = rng.choice(WIDTH_CHOICES)
    blocker_count = rng.randint(*BLOCKER_COUNT_RANGE)
    mid = (h - 1) // 2

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[0, :] = TOP_COLOR
    grid[h - 1, :] = BOTTOM_COLOR

    used: set[tuple[int, int]] = set()
    columns: set[int] = set()
    blockers: list[tuple[int, int]] = []

    required_specs: list[tuple[list[int], str]] = [
        (list(range(2, mid + 1)), "grid"),
        (list(range(mid + 1, h - 2)), "nongrid"),
    ]
    if rng.random() < MIDDLE_BLOCKER_PROB and blocker_count >= 4:
        required_specs.append(([mid], "grid"))

    for rows, parity in required_specs:
        pos = _sample_nonadjacent_position(rng, h, w, used, columns, rows, parity, require_effect=True)
        if pos is None:
            return None
        blockers.append(pos)
        used.add(pos)
        columns.add(pos[1])

    remaining = blocker_count - len(blockers)
    parity_options = ("grid", "nongrid")
    half_options = ("top", "bottom")
    for _ in range(remaining):
        specs = [(half, parity) for half in half_options for parity in parity_options]
        rng.shuffle(specs)
        pos = None
        for half, parity in specs:
            rows = list(range(1, mid + 1)) if half == "top" else list(range(mid + 1, h - 1))
            pos = _sample_nonadjacent_position(rng, h, w, used, columns, rows, parity)
            if pos is not None:
                break
        if pos is None:
            return None
        blockers.append(pos)
        used.add(pos)
        columns.add(pos[1])

    rng.shuffle(blockers)
    for r, c in blockers:
        grid[r, c] = BLOCKER_COLOR

    out = _apply_rule(grid)
    _validate_output_semantics(grid, out)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 3")
