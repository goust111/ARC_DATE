from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ROW_COLOR_POOL = (2, 3, 4)
COL_COLOR_POOL = (1, 6, 7)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _extract_bars(grid: np.ndarray) -> tuple[dict[int, int], dict[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    for comp in _find_components(g, bg):
        assert comp["area"] == 2, "each foreground object must be a two-pixel domino"
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == r1 and c1 == c0 + 1:
            assert r0 not in row_colors, "duplicate horizontal marker row"
            row_colors[r0] = color
        elif c0 == c1 and r1 == r0 + 1:
            assert c0 not in col_colors, "duplicate vertical marker column"
            col_colors[c0] = color
        else:
            raise AssertionError("foreground domino must be horizontal or vertical")

    assert 2 <= len(row_colors) <= 3, "expected 2-3 horizontal dominoes"
    assert 2 <= len(col_colors) <= 3, "expected 2-3 vertical dominoes"
    return row_colors, col_colors, bg


def _paint_bars(
    shape: tuple[int, int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
    bg: int,
    *,
    rows_over_cols: bool,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    if rows_over_cols:
        for c, color in sorted(col_colors.items()):
            out[:, c] = int(color)
        for r, color in sorted(row_colors.items()):
            out[r, :] = int(color)
    else:
        for r, color in sorted(row_colors.items()):
            out[r, :] = int(color)
        for c, color in sorted(col_colors.items()):
            out[:, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend domino markers into full row/column bars, then crop the union of all bar intersections.

    Input structure:
    - The grid is mostly black background with several isolated same-color two-pixel dominoes.
    - Horizontal dominoes mark rows and vertical dominoes mark columns.
    - Every foreground component is exactly one horizontal or vertical domino.

    Rule:
    - Extend each vertical domino into a full-height column bar of its color.
    - Extend each horizontal domino into a full-width row bar of its color after the columns, so rows overwrite columns at crossings.
    - Take the smallest bounding box containing every row/column intersection and return that crop.

    Key insight:
    - The output is not the whole painted grid: only the rectangle spanned by the row-bar rows and column-bar columns survives, and the crossing colors reveal the row-over-column priority.
    """
    g = np.asarray(grid, dtype=int)
    row_colors, col_colors, bg = _extract_bars(g)
    rows = sorted(row_colors)
    cols = sorted(col_colors)
    painted = _paint_bars(g.shape, row_colors, col_colors, bg, rows_over_cols=True)
    return _crop_bbox(painted, (rows[0], cols[0], rows[-1], cols[-1]))


def _sample_axis_positions(length: int, count: int, rng: random.Random) -> list[int]:
    assert length >= 8
    choices = list(range(1, length - 1))
    for _ in range(200):
        picked = sorted(rng.sample(choices, count))
        if picked[-1] - picked[0] + 1 > count:
            return picked
    raise RuntimeError("failed to sample non-degenerate axis positions")


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG and (nr, nc) not in cells:
                return False
    return True


def _place_horizontal_markers(
    grid: np.ndarray,
    rows: list[int],
    colors: list[int],
    blocked_cols: set[int],
    rng: random.Random,
) -> None:
    width = grid.shape[1]
    row_order = rows[:]
    rng.shuffle(row_order)
    for row in row_order:
        color = colors[rows.index(row)]
        starts = []
        for c0 in range(width - 1):
            cells = [(row, c0), (row, c0 + 1)]
            if c0 in blocked_cols or c0 + 1 in blocked_cols:
                continue
            if _can_place(grid, cells):
                starts.append(c0)
        assert starts, "no valid placement for horizontal marker"
        c0 = rng.choice(starts)
        grid[row, c0 : c0 + 2] = color


def _place_vertical_markers(
    grid: np.ndarray,
    cols: list[int],
    colors: list[int],
    blocked_rows: set[int],
    rng: random.Random,
) -> None:
    height = grid.shape[0]
    col_order = cols[:]
    rng.shuffle(col_order)
    for col in col_order:
        color = colors[cols.index(col)]
        starts = []
        for r0 in range(height - 1):
            cells = [(r0, col), (r0 + 1, col)]
            if r0 in blocked_rows or r0 + 1 in blocked_rows:
                continue
            if _can_place(grid, cells):
                starts.append(r0)
        assert starts, "no valid placement for vertical marker"
        r0 = rng.choice(starts)
        grid[r0 : r0 + 2, col] = color


def _validate_generated_case(grid: np.ndarray) -> None:
    row_colors, col_colors, bg = _extract_bars(grid)
    h, w = grid.shape
    rows = sorted(row_colors)
    cols = sorted(col_colors)

    assert bg == BG
    assert rows[0] > 0 and rows[-1] < h - 1, "crop must be smaller than input vertically"
    assert cols[0] > 0 and cols[-1] < w - 1, "crop must be smaller than input horizontally"
    assert rows[-1] - rows[0] + 1 > len(rows), "need at least one non-row gap inside crop"
    assert cols[-1] - cols[0] + 1 > len(cols), "need at least one non-column gap inside crop"

    bbox = (rows[0], cols[0], rows[-1], cols[-1])
    out = main(grid)
    full = _paint_bars(grid.shape, row_colors, col_colors, bg, rows_over_cols=True)
    alt = _paint_bars(grid.shape, row_colors, col_colors, bg, rows_over_cols=False)
    row_only = _crop_bbox(_paint_bars(grid.shape, row_colors, {}, bg, rows_over_cols=True), bbox)
    col_only = _crop_bbox(_paint_bars(grid.shape, {}, col_colors, bg, rows_over_cols=True), bbox)

    assert np.array_equal(out, _crop_bbox(full, bbox))
    assert not np.array_equal(out, _crop_bbox(alt, bbox)), "row-over-column priority must matter"
    assert not np.array_equal(out, row_only), "column bars must affect the output"
    assert not np.array_equal(out, col_only), "row bars must affect the output"
    assert bg in set(map(int, out.flatten())), "output should retain visible background cells"
    for color in row_colors.values():
        assert int(color) in out
    for color in col_colors.values():
        assert int(color) in out

    local_rows = {r - rows[0]: color for r, color in row_colors.items()}
    local_cols = {c - cols[0]: color for c, color in col_colors.items()}
    for rr, color in local_rows.items():
        for cc in local_cols:
            assert int(out[rr, cc]) == int(color), "intersections must show row colors"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 17)
    width = rng.randint(12, 17)
    h_count = rng.randint(2, 3)
    v_count = rng.randint(2, 3)

    rows = _sample_axis_positions(height, h_count, rng)
    cols = _sample_axis_positions(width, v_count, rng)
    row_colors = rng.sample(list(ROW_COLOR_POOL), h_count)
    col_colors = rng.sample(list(COL_COLOR_POOL), v_count)

    grid = np.full((height, width), BG, dtype=int)
    try:
        _place_vertical_markers(grid, cols, col_colors, set(rows), rng)
        _place_horizontal_markers(grid, rows, row_colors, set(cols), rng)
        _validate_generated_case(grid)
    except AssertionError:
        return None
    except RuntimeError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
