from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ORANGE = 7
PALETTE = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return comp["area"] == area and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict], int, int]:
    bg = _get_bg(grid)
    rects = [comp for comp in _find_components(grid, bg) if _is_solid_rectangle(grid, comp)]
    assert len(rects) >= 6, "expected a grid of repeated solid rectangles"

    shapes = {(comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1) for comp in rects}
    assert len(shapes) == 1, "all rectangles must share one size"
    rect_h, rect_w = next(iter(shapes))
    assert rect_h == 2 and rect_w in {2, 3}, "expected 2x2 or 2x3 rectangles"

    rects.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return bg, rects, rect_h, rect_w


def _constant_step(values: list[int]) -> int:
    diffs = [b - a for a, b in zip(values, values[1:])]
    assert diffs and min(diffs) > 0, "need repeated positions to infer a period"
    step = min(diffs)
    assert all(diff == step for diff in diffs), "rectangle starts must form a regular lattice"
    return int(step)


def _minimal_row_period(matrix: np.ndarray) -> int:
    rows = matrix.shape[0]
    for period in range(1, rows + 1):
        if np.array_equal(matrix, matrix[np.arange(rows) % period, :]):
            return period
    raise AssertionError("row period not found")


def _minimal_col_period(matrix: np.ndarray) -> int:
    cols = matrix.shape[1]
    for period in range(1, cols + 1):
        if np.array_equal(matrix, matrix[:, np.arange(cols) % period]):
            return period
    raise AssertionError("column period not found")


def _first_positions(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color != bg and color not in out:
                out[color] = (r, c)
    return out


def _rank_colors(grid: np.ndarray, input_grid: np.ndarray, bg: int) -> list[int]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != bg)
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    first = _first_positions(input_grid, bg)
    assert all(color in first for color in colors), "every completed color must already be visible in the input"
    return sorted(colors, key=lambda color: (-counts[color], first[color][0], first[color][1]))


def _recolor_minorities(grid: np.ndarray, keep: set[int], bg: int) -> np.ndarray:
    out = grid.copy()
    mask = (out != bg) & (~np.isin(out, sorted(keep)))
    out[mask] = ORANGE
    return out


def _completed_pattern(grid: np.ndarray) -> tuple[np.ndarray, int]:
    bg, rects, rect_h, rect_w = _extract_rectangles(grid)
    row_starts = sorted({comp["bbox"][0] for comp in rects})
    col_starts = sorted({comp["bbox"][1] for comp in rects})
    assert len(row_starts) >= 2 and len(col_starts) >= 3, "need multiple rows and visible columns"

    step_r = _constant_step(row_starts)
    step_c = _constant_step(col_starts)

    row_index = {value: idx for idx, value in enumerate(row_starts)}
    col_index = {value: idx for idx, value in enumerate(col_starts)}
    visible = np.full((len(row_starts), len(col_starts)), -1, dtype=int)
    for comp in rects:
        r0, c0, _r1, _c1 = comp["bbox"]
        ri = row_index[r0]
        ci = col_index[c0]
        assert visible[ri, ci] == -1, "each lattice slot should contain exactly one rectangle"
        visible[ri, ci] = int(comp["color"])
    assert np.all(visible != -1), "visible lattice must be a full rectangular prefix"

    total_cols = 1 + (grid.shape[1] - rect_w - col_starts[0]) // step_c
    assert total_cols > len(col_starts), "input must be truncated on the right"

    row_period = _minimal_row_period(visible)
    col_period = _minimal_col_period(visible)
    tile = visible[:row_period, :col_period]
    complete_colors = tile[np.arange(len(row_starts)) % row_period][:, np.arange(total_cols) % col_period]
    assert np.array_equal(complete_colors[:, : visible.shape[1]], visible), "visible rectangles must be a prefix of the periodic matrix"

    out = np.asarray(grid, dtype=int).copy()
    first_missing_col = col_starts[0] + len(col_starts) * step_c
    assert np.all(out[:, first_missing_col:] == bg), "the truncated right region should be pure background"
    for ri, r0 in enumerate(row_starts):
        for ci in range(total_cols):
            c0 = col_starts[0] + ci * step_c
            color = int(complete_colors[ri, ci])
            patch = out[r0 : r0 + rect_h, c0 : c0 + rect_w]
            assert patch.shape == (rect_h, rect_w), "completed rectangle would leave the grid"
            assert np.all((patch == bg) | (patch == color)), "completed rectangles must not collide with conflicting pixels"
            out[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
    return out, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the truncated periodic rectangle matrix and recolor all but the two most frequent completed colors to orange.

    Input structure:
    - The background is black, and every foreground object is a same-sized solid 2x2 or 2x3 rectangle.
    - Rectangle top-left corners form a regular row/column lattice, and the visible rectangles occupy a complete rectangular prefix of that lattice.
    - The right side of the grid is truncated to pure background, but the rectangle colors follow a repeating row/column pattern across the full lattice.

    Rule:
    - Infer the rectangle spacing and the smallest repeating color matrix that explains the visible lattice.
    - Extend that periodic matrix to every rectangle position that still fits in the blank right-side truncated region.
    - On the completed grid, count foreground pixels by color, keep the two highest-ranked colors, and recolor every other rectangle pixel to orange.
    - Break any frequency ties by the earliest row-major appearance of that color in the original input grid.

    Key insight:
    - The blank right region is not empty by rule; it is a clipped suffix of a periodic rectangle matrix, and filling it can change which colors are the two true majorities.
    """
    completed, bg = _completed_pattern(np.asarray(grid, dtype=int))
    ranked = _rank_colors(completed, np.asarray(grid, dtype=int), bg)
    return _recolor_minorities(completed, set(ranked[:2]), bg)


def _rank_from_counts(counts: Counter, first: dict[int, tuple[int, int]]) -> list[int]:
    return sorted(counts, key=lambda color: (-int(counts[color]), first[color][0], first[color][1]))


def _lattice_first_positions(matrix: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            color = int(matrix[r, c])
            if color not in out:
                out[color] = (r, c)
    return out


def _render_lattice(
    rows: int,
    cols: int,
    rect_h: int,
    rect_w: int,
    step_r: int,
    step_c: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    lattice: np.ndarray,
) -> np.ndarray:
    height = top + (rows - 1) * step_r + rect_h + bottom
    width = left + (cols - 1) * step_c + rect_w + right
    grid = np.zeros((height, width), dtype=int)
    for r in range(rows):
        rr = top + r * step_r
        for c in range(cols):
            cc = left + c * step_c
            grid[rr : rr + rect_h, cc : cc + rect_w] = int(lattice[r, c])
    return grid


def _tile_matrix(tile: np.ndarray, rows: int, cols: int) -> np.ndarray:
    pr, pc = tile.shape
    return tile[np.arange(rows) % pr][:, np.arange(cols) % pc]


def _sample_geometry() -> tuple[int, int, int, int, int, int, int, int, int]:
    rect_h = 2
    rect_w = random.choice((2, 3))
    step_r = random.choice((3, 4))
    step_c = random.choice((4, 5))
    rows = random.randint(3, 5)
    total_cols = 6 if step_c == 4 and random.random() < 0.6 else 5
    visible_cols = total_cols - 2

    top = random.randint(0, step_r - 1)
    left = random.randint(0, step_c - 1)
    bottom = random.randint(0, 2)

    right_candidates = []
    for right in range(step_c):
        width = left + (total_cols - 1) * step_c + rect_w + right
        if width > 30:
            continue
        blank_start = left + visible_cols * step_c
        blank_ratio = (width - blank_start) / width
        if 0.28 <= blank_ratio <= 0.45:
            right_candidates.append(right)
    if not right_candidates:
        raise RuntimeError("no valid right margin")
    right = random.choice(right_candidates)

    height = top + (rows - 1) * step_r + rect_h + bottom
    assert height <= 30
    return rows, total_cols, visible_cols, rect_h, rect_w, step_r, step_c, top, left, bottom, right


def _sample_tile(
    rows: int,
    total_cols: int,
    visible_cols: int,
    want_tie: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]] | None:
    shape_choices = []
    if rows >= 2:
        shape_choices.extend([(2, 2), (2, 3)])
    shape_choices.append((1, 3))
    random.shuffle(shape_choices)

    for pr, pc in shape_choices:
        if pc > visible_cols:
            continue
        palette_size = 4 if pr * pc >= 4 and random.random() < 0.55 else 3
        for _ in range(160):
            colors = random.sample(PALETTE, palette_size)
            tile = np.array(random.choices(colors, k=pr * pc), dtype=int).reshape(pr, pc)
            if len(set(map(int, tile.flatten()))) != palette_size:
                continue
            if _minimal_row_period(tile) != pr or _minimal_col_period(tile) != pc:
                continue

            visible = _tile_matrix(tile, rows, visible_cols)
            full = _tile_matrix(tile, rows, total_cols)
            if len(set(map(int, np.unique(visible)))) < 3:
                continue
            if set(map(int, np.unique(full))) != set(map(int, np.unique(visible))):
                continue
            if _minimal_row_period(visible) != pr or _minimal_col_period(visible) != pc:
                continue

            first = _lattice_first_positions(visible)
            visible_rank = _rank_from_counts(Counter(map(int, visible.flatten())), first)
            full_counts = Counter(map(int, full.flatten()))
            full_rank = _rank_from_counts(full_counts, first)
            if len(visible_rank) < 3 or len(full_rank) < 3:
                continue

            promoted = [color for color in full_rank[:2] if color not in visible_rank[:2]]
            demoted = [color for color in visible_rank[:2] if color not in full_rank[:2]]
            stable = [color for color in full_rank[:2] if color in visible_rank[:2]]
            if len(promoted) != 1 or len(demoted) != 1 or len(stable) != 1:
                continue
            if visible_rank.index(promoted[0]) < 2:
                continue

            second_tied = full_counts[full_rank[1]] == full_counts[full_rank[2]]
            if second_tied != want_tie:
                continue
            if not second_tied and full_counts[full_rank[1]] <= full_counts[full_rank[2]]:
                continue
            return tile, visible, full, visible_rank, full_rank
    return None


def _expected_output(full_grid: np.ndarray, input_grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(input_grid)
    ranked = _rank_colors(full_grid, input_grid, bg)
    return _recolor_minorities(full_grid, set(ranked[:2]), bg)


def _wrong_shortcut_output(full_grid: np.ndarray, input_grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(input_grid)
    ranked = _rank_colors(input_grid, input_grid, bg)
    return _recolor_minorities(full_grid, set(ranked[:2]), bg)


def _attempt_generate() -> np.ndarray | None:
    rows, total_cols, visible_cols, rect_h, rect_w, step_r, step_c, top, left, bottom, right = _sample_geometry()
    want_tie = random.random() < 0.45
    tile_info = _sample_tile(rows, total_cols, visible_cols, want_tie)
    if tile_info is None:
        return None

    _tile, visible_lattice, full_lattice, visible_rank, full_rank = tile_info
    full_grid = _render_lattice(
        rows,
        total_cols,
        rect_h,
        rect_w,
        step_r,
        step_c,
        top,
        left,
        bottom,
        right,
        full_lattice,
    )
    input_grid = _render_lattice(
        rows,
        visible_cols,
        rect_h,
        rect_w,
        step_r,
        step_c,
        top,
        left,
        bottom,
        right + (total_cols - visible_cols) * step_c,
        visible_lattice,
    )

    blank_start = left + visible_cols * step_c
    if not np.all(input_grid[:, blank_start:] == 0):
        return None
    if full_grid.shape != input_grid.shape or max(full_grid.shape) > 30:
        return None

    expected = _expected_output(full_grid, input_grid)
    shortcut = _wrong_shortcut_output(full_grid, input_grid)
    if np.array_equal(expected, shortcut):
        return None
    if np.array_equal(visible_rank[:2], full_rank[:2]):
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    promoted = [color for color in full_rank[:2] if color not in visible_rank[:2]]
    if len(promoted) != 1:
        return None
    full_counts = Counter(map(int, full_lattice.flatten()))
    visible_counts = Counter(map(int, visible_lattice.flatten()))
    if full_counts[promoted[0]] <= visible_counts[promoted[0]]:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-completion input")
