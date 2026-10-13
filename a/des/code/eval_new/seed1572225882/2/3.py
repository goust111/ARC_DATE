import numpy as np
import random


def _find_separator(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    # Background is expected to be the dominant color, black in the generator.
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])

    row_candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            row_candidates.append((r, color))

    col_candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_candidates.append((c, color))

    crosses = [
        (r, c, color)
        for r, color in row_candidates
        for c, c_color in col_candidates
        if c_color == color
    ]
    assert len(crosses) == 1, "expected exactly one grey separator cross"

    cross_row, cross_col, sep = crosses[0]
    assert bg != sep, "separator color must differ from background"
    assert 0 < cross_row < h - 1 and 0 < cross_col < w - 1
    return cross_row, cross_col, bg, sep


def _line_orientations(tl: np.ndarray, bg: int, sep: int) -> tuple[list[int], list[int]]:
    colors = [int(v) for v in np.unique(tl) if int(v) not in (bg, sep)]
    horizontal: list[int] = []
    vertical: list[int] = []

    for color in colors:
        mask = tl == color
        assert mask.any(), "each candidate line color must exist"
        rows, cols = np.where(mask)
        row_count = len(np.unique(rows))
        col_count = len(np.unique(cols))

        if row_count == 1 and col_count > 1:
            horizontal.append(color)
            continue
        if col_count == 1 and row_count > 1:
            vertical.append(color)
            continue

        h_links = int(np.count_nonzero(mask[:, 1:] & mask[:, :-1]))
        v_links = int(np.count_nonzero(mask[1:, :] & mask[:-1, :]))
        if h_links >= v_links:
            horizontal.append(color)
        else:
            vertical.append(color)

    return horizontal, vertical


def _reconstruct_horizontal_lines(tl: np.ndarray, color: int, bg: int, sep: int) -> np.ndarray:
    h, w = map(int, tl.shape)
    out = np.full((h, w), bg, dtype=int)
    positions = np.argwhere(tl == color)
    for row in sorted(set(int(r) for r, _c in positions)):
        cols = sorted(set(int(c) for r, c in positions if int(r) == row))
        if not cols:
            continue
        if len(cols) == 1:
            out[row, cols[0]] = color
            continue

        active = set(cols)
        for a, b in zip(cols, cols[1:]):
            if b - a == 2:
                mid = a + 1
                if (
                    0 < row < h - 1
                    and 0 < mid < w - 1
                    and int(tl[row, mid]) not in (bg, sep)
                    and int(tl[row - 1, mid]) not in (bg, sep)
                    and int(tl[row + 1, mid]) not in (bg, sep)
                ):
                    active.add(mid)

        cols = sorted(active)
        start = cols[0]
        prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
                continue
            out[row, start : prev + 1] = color
            start = prev = c
        out[row, start : prev + 1] = color

    return out


def _reconstruct_vertical_lines(tl: np.ndarray, color: int, bg: int, sep: int) -> np.ndarray:
    h, w = map(int, tl.shape)
    out = np.full((h, w), bg, dtype=int)
    positions = np.argwhere(tl == color)
    for col in sorted(set(int(c) for _r, c in positions)):
        rows = sorted(set(int(r) for r, c in positions if int(c) == col))
        if not rows:
            continue
        if len(rows) == 1:
            out[rows[0], col] = color
            continue

        active = set(rows)
        for a, b in zip(rows, rows[1:]):
            if b - a == 2:
                mid = a + 1
                if (
                    0 < mid < h - 1
                    and 0 < col < w - 1
                    and int(tl[mid, col]) not in (bg, sep)
                    and int(tl[mid, col - 1]) not in (bg, sep)
                    and int(tl[mid, col + 1]) not in (bg, sep)
                ):
                    active.add(mid)

        rows = sorted(active)
        start = rows[0]
        prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
                continue
            out[start : prev + 1, col] = color
            start = prev = r
        out[start : prev + 1, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover horizontal and vertical source line layers from the visible overlay and route them into separate output quadrants.

    Input structure:
    - A black background with one grey full row and one grey full column forming a 2x2 cross.
    - The left-top quadrant contains axis-aligned one-pixel lines of non-grey colors, with some pixels covered by crossings.
    - At a crossing only the top layer remains visible, so the hidden layer color is missing at those single pixels.

    Rule:
    - Detect the separator cross and isolate the left-top quadrant.
    - Infer each line color's direction (horizontal or vertical) from its structural evidence.
    - Reconstruct interrupted line segments by continuity, allowing one-cell gaps where an opposite-oriented line passes through.
    - In the right-top quadrant draw only horizontal reconstructed lines; in the left-bottom only vertical lines.
    - In the right-bottom quadrant draw each intersection pixel with the color of the hidden layer at that position.
    - Keep grey separator lines unchanged, everything else black.

    Key insight:
    - The visible left-top pixel is a layer composition, but crossing lines are still recoverable by axis continuity and one-pixel gap inference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cross_row, cross_col, bg, sep = _find_separator(g)
    assert 0 < cross_row < g.shape[0] - 1
    assert 0 < cross_col < g.shape[1] - 1

    tl = g[:cross_row, :cross_col]
    horizontal_colors, vertical_colors = _line_orientations(tl, bg, sep)
    assert horizontal_colors or vertical_colors

    h_lines = np.full_like(tl, bg, dtype=int)
    for color in horizontal_colors:
        h_lines |= _reconstruct_horizontal_lines(tl, color, bg, sep)

    v_lines = np.full_like(tl, bg, dtype=int)
    for color in vertical_colors:
        v_lines |= _reconstruct_vertical_lines(tl, color, bg, sep)

    out = np.full_like(g, bg, dtype=int)
    out[cross_row, :] = sep
    out[:, cross_col] = sep

    for r in range(cross_row):
        for c in range(cross_col):
            hc = int(h_lines[r, c])
            vc = int(v_lines[r, c])
            if hc != bg:
                out[r, cross_col + 1 + c] = hc
            if vc != bg:
                out[cross_row + 1 + r, c] = vc

            if hc != bg and vc != bg:
                visible = int(tl[r, c])
                if visible == hc and visible != vc:
                    out[cross_row + 1 + r, cross_col + 1 + c] = vc
                elif visible == vc and visible != hc:
                    out[cross_row + 1 + r, cross_col + 1 + c] = hc

    return out


# -------------------------- generator --------------------------

GRID_HALF_RANGE = (5, 11)  # q gives total dimension = 2*q + 1
HALF_LINE_MIN = 2
LINES_PER_AXIS = (1, 4)
MAX_GEN_ATTEMPTS = 300
BACKGROUND = 0
GREY = 1
PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]


def _pick_positions(space: int, count: int, min_gap: int) -> list[int] | None:
    if space < 3:
        return None
    candidates = list(range(1, space - 1))
    if len(candidates) < count:
        return None

    for _ in range(120):
        rows = sorted(random.sample(candidates, count))
        good = True
        for a, b in zip(rows, rows[1:]):
            if b - a < min_gap:
                good = False
                break
        if good:
            return rows
    return None


def _pick_span(axis: int, min_len: int) -> tuple[int, int] | None:
    max_len = axis - 2
    if max_len < min_len:
        return None
    length = random.randint(min_len, max_len)
    length = max(min_len, length)
    start = random.randint(1, axis - length - 1)
    return start, start + length - 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        qh = random.randint(*GRID_HALF_RANGE)
        qw = random.randint(*GRID_HALF_RANGE)
        if 2 * qh + 1 > 30 or 2 * qw + 1 > 30:
            continue

        h_axes = qh - 1
        v_axes = qw - 1
        h_count = random.randint(LINES_PER_AXIS[0], min(LINES_PER_AXIS[1], max(1, h_axes // 2)))
        v_count = random.randint(LINES_PER_AXIS[0], min(LINES_PER_AXIS[1], max(1, v_axes // 2)))
        if h_count <= 0 or v_count <= 0:
            continue

        h_rows = _pick_positions(qh, h_count, min_gap=2)
        v_cols = _pick_positions(qw, v_count, min_gap=2)
        if h_rows is None or v_cols is None:
            continue

        palette = PALETTE[:]
        random.shuffle(palette)
        if h_count + v_count > len(palette):
            continue

        horizontal = []
        vertical = []
        for i, row in enumerate(h_rows):
            span = _pick_span(qw, HALF_LINE_MIN)
            if span is None:
                break
            start, end = span
            horizontal.append({"row": row, "start": start, "end": end, "color": palette[i]})
        if len(horizontal) != h_count:
            continue

        for i, col in enumerate(v_cols):
            span = _pick_span(qh, HALF_LINE_MIN)
            if span is None:
                break
            start, end = span
            vertical.append(
                {
                    "col": col,
                    "start": start,
                    "end": end,
                    "color": palette[h_count + i],
                }
            )
        if len(vertical) != v_count:
            continue

        tl = np.full((qh, qw), BACKGROUND, dtype=int)
        for line in horizontal:
            tl[line["row"], line["start"] : line["end"] + 1] = line["color"]
        for line in vertical:
            tl[line["start"] : line["end"] + 1, line["col"]] = line["color"]

        crossings = []
        for hi, h_line in enumerate(horizontal):
            for vi, v_line in enumerate(vertical):
                if (
                    h_line["start"] <= v_line["col"] <= h_line["end"]
                    and v_line["start"] <= h_line["row"] <= v_line["end"]
                ):
                    crossings.append((h_line["row"], v_line["col"], hi, vi))

        if len(crossings) < 2:
            continue

        top_vertical = [random.choice((True, False)) for _ in crossings]
        if all(top_vertical) or not any(top_vertical):
            idx = random.randrange(len(top_vertical))
            top_vertical[idx] = not top_vertical[idx]

        for (r, c, hi, vi), show_vertical in zip(crossings, top_vertical):
            if not show_vertical:
                tl[r, c] = horizontal[hi]["color"]

        grid = np.full((2 * qh + 1, 2 * qw + 1), BACKGROUND, dtype=int)
        grid[qh, :] = GREY
        grid[:, qw] = GREY
        grid[:qh, :qw] = tl

        try:
            transformed = main(grid)
            assert transformed.shape == grid.shape
            assert np.all(transformed[qh, :] == GREY)
            assert np.all(transformed[:, qw] == GREY)
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate valid input")
