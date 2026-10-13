from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


def _load_color_namespace():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if arc_utils_path.is_file():
        try:
            mod = SourceFileLoader("_arc_utils_027_6", str(arc_utils_path)).load_module()
            return mod.Color
        except Exception:
            pass

    class _Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        PINK = 6

    return _Color


Color = _load_color_namespace()


def _paint_half_border(grid: np.ndarray) -> None:
    h, w = grid.shape
    mid = h // 2
    grid[0, :] = Color.RED
    grid[h - 1, :] = Color.BLUE
    for r in range(h):
        border_color = Color.RED if r < mid else Color.BLUE
        grid[r, 0] = border_color
        grid[r, w - 1] = border_color


def _extract_roles(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    red = np.argwhere(g == Color.RED)
    blue = np.argwhere(g == Color.BLUE)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREEN)]

    assert g.ndim == 2
    assert g.shape[0] % 2 == 0
    assert red.shape[0] == 1
    assert blue.shape[0] == 1
    assert len(markers) >= 2

    red_rc = tuple(map(int, red[0]))
    blue_rc = tuple(map(int, blue[0]))
    assert red_rc[0] < blue_rc[0]
    return red_rc, blue_rc, markers


def _ray_cover_masks(base: np.ndarray, markers: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = base.shape
    mid = h // 2
    red_cover = np.zeros((h, w), dtype=bool)
    blue_cover = np.zeros((h, w), dtype=bool)

    for r, c in markers:
        cover = red_cover if r < mid else blue_cover

        for rr in range(r - 1, 0, -1):
            if int(base[rr, c]) == Color.BLACK:
                cover[rr, c] = True
        for rr in range(r + 1, h - 1):
            if int(base[rr, c]) == Color.BLACK:
                cover[rr, c] = True
        for cc in range(c - 1, 0, -1):
            if int(base[r, cc]) == Color.BLACK:
                cover[r, cc] = True
        for cc in range(c + 1, w - 1):
            if int(base[r, cc]) == Color.BLACK:
                cover[r, cc] = True

    return red_cover, blue_cover


def _base_canvas(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    (red_r, _red_c), (blue_r, _blue_c), markers = _extract_roles(g)
    out = g.copy()
    out[red_r, :] = Color.RED
    out[blue_r, :] = Color.BLUE
    _paint_half_border(out)
    return out, markers, g.shape[0] // 2


def _expected_from_semantics(grid: np.ndarray) -> np.ndarray:
    base, markers, mid = _base_canvas(grid)
    expected = base.copy()

    upper_rows = {r for r, _ in markers if r < mid}
    upper_cols = {c for r, c in markers if r < mid}
    lower_rows = {r for r, _ in markers if r >= mid}
    lower_cols = {c for r, c in markers if r >= mid}

    for r in range(1, base.shape[0] - 1):
        for c in range(1, base.shape[1] - 1):
            if int(base[r, c]) != Color.BLACK:
                continue
            upper_hit = r in upper_rows or c in upper_cols
            lower_hit = r in lower_rows or c in lower_cols
            if upper_hit and lower_hit:
                expected[r, c] = Color.PINK
            elif upper_hit:
                expected[r, c] = Color.RED
            elif lower_hit:
                expected[r, c] = Color.BLUE

    return expected


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30 or g.shape[0] % 2 != 0:
            return False

        colors = set(map(int, np.unique(g)))
        if not colors.issubset({Color.BLACK, Color.BLUE, Color.RED, Color.GREEN}):
            return False

        red_rc, blue_rc, markers = _extract_roles(g)
        if len(markers) < 2 or len(markers) > 4:
            return False

        h, w = g.shape
        mid = h // 2
        upper_markers = [(r, c) for r, c in markers if r < mid]
        lower_markers = [(r, c) for r, c in markers if r >= mid]
        if not upper_markers or not lower_markers:
            return False

        seen_rows = set()
        seen_cols = set()
        for r, c in markers:
            if r in (0, h - 1) or c in (0, w - 1):
                return False
            if r in {red_rc[0], blue_rc[0]} or c in {red_rc[1], blue_rc[1]}:
                return False
            if r in seen_rows or c in seen_cols:
                return False
            seen_rows.add(r)
            seen_cols.add(c)

        base, _, _ = _base_canvas(g)
        crossable = 0
        for ur, uc in upper_markers:
            for lr, lc in lower_markers:
                if int(base[ur, lc]) == Color.BLACK:
                    crossable += 1
                if int(base[lr, uc]) == Color.BLACK:
                    crossable += 1
        if crossable == 0:
            return False

        expected = _expected_from_semantics(g)
        actual = main(g)
        if not np.array_equal(actual, expected):
            return False
        if np.count_nonzero(actual == Color.PINK) == 0:
            return False
        if np.count_nonzero(actual != g) < h + w:
            return False
        return True
    except Exception:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """Draw half-colored frame lines and half-dependent rays from green markers, turning red/blue overlaps magenta.

    Input structure:
    - Black is the background on an even-height grid.
    - Exactly one red pixel lies in the upper half and exactly one blue pixel lies in the lower half.
    - Two to four green marker pixels sit away from the border, with at least one in each half.

    Rule:
    - Draw a full-width red horizontal line through the red pixel row and a full-width blue horizontal line through the blue pixel row.
    - Draw a one-cell rectangular border around the whole grid, coloring border rows 0..H/2-1 red and rows H/2..H-1 blue.
    - Each green marker emits four orthogonal rays whose color is determined by its half: upper-half markers emit red and lower-half markers emit blue.
    - Rays extend until the border, never repaint the border or any non-black cell, and any black cell covered by both a red ray and a blue ray becomes magenta while markers stay green.

    Key insight:
    - The green markers do not carry their own output color; their row half decides the ray color, and the half-colored border only limits reach while cross-half row/column intersections reveal themselves as magenta.
    """
    base, markers, _mid = _base_canvas(grid)
    out = base.copy()
    red_cover, blue_cover = _ray_cover_masks(base, markers)

    both = red_cover & blue_cover
    red_only = red_cover & ~blue_cover
    blue_only = blue_cover & ~red_cover

    out[red_only] = Color.RED
    out[blue_only] = Color.BLUE
    out[both] = Color.PINK
    return out


GRID_HEIGHTS = (12, 14, 16, 18)
GRID_WIDTH_RANGE = (12, 18)
MARKER_COUNTS = (2, 3, 4)
GEN_ATTEMPTS = 400


def _sample_rows(pool: list[int], count: int) -> list[int]:
    picks = random.sample(pool, count)
    random.shuffle(picks)
    return picks


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_HEIGHTS)
    w = random.randint(*GRID_WIDTH_RANGE)
    mid = h // 2

    if w < 8:
        return None

    red_row = random.randint(1, mid - 2)
    blue_row = random.randint(mid + 1, h - 2)
    red_col, blue_col = random.sample(range(1, w - 1), 2)

    marker_count = random.choice(MARKER_COUNTS)
    upper_count = random.randint(1, marker_count - 1)
    lower_count = marker_count - upper_count

    upper_row_pool = [r for r in range(1, mid) if r != red_row]
    lower_row_pool = [r for r in range(mid, h - 1) if r != blue_row]
    col_pool = [c for c in range(1, w - 1) if c not in {red_col, blue_col}]
    if len(upper_row_pool) < upper_count or len(lower_row_pool) < lower_count or len(col_pool) < marker_count:
        return None

    upper_rows = _sample_rows(upper_row_pool, upper_count)
    lower_rows = _sample_rows(lower_row_pool, lower_count)
    marker_cols = _sample_rows(col_pool, marker_count)

    markers: list[tuple[int, int]] = []
    for idx, r in enumerate(upper_rows):
        markers.append((r, marker_cols[idx]))
    for idx, r in enumerate(lower_rows, start=upper_count):
        markers.append((r, marker_cols[idx]))
    random.shuffle(markers)

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[red_row, red_col] = Color.RED
    grid[blue_row, blue_col] = Color.BLUE
    for r, c in markers:
        grid[r, c] = Color.GREEN

    if _validate_generated_input(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
