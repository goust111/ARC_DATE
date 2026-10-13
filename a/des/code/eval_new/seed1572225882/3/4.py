import random

import numpy as np


BG = 0
MARK = 9
COLORS = (1, 2, 3, 4, 5, 6, 7, 8)


def _axis_match_score(base: np.ndarray, orient: str, a2: int) -> int:
    rows, cols = base.shape
    coords = np.argwhere(base != BG)
    matches = 0
    conflicts = 0
    offgrid = 0

    for r, c in coords:
        if orient == "h":
            rr, cc = a2 - r, c
            in_bounds = 0 <= rr < rows
        else:
            rr, cc = r, a2 - c
            in_bounds = 0 <= cc < cols

        if not in_bounds:
            offgrid += 1
            continue

        other = base[rr, cc]
        if other == base[r, c]:
            matches += 1
        elif other != BG:
            conflicts += 1

    return matches * 5 - conflicts * 40 - offgrid * 8


def _line_count(base: np.ndarray, orient: str, line: int) -> int:
    if orient == "h":
        return int(np.count_nonzero(base[line, :]))
    return int(np.count_nonzero(base[:, line]))


def _candidate_score(base: np.ndarray, markers: np.ndarray, orient: str, a2: int) -> int:
    rows, cols = base.shape
    limit = rows if orient == "h" else cols
    marked_lines = np.where(np.any(markers, axis=1 if orient == "h" else 0))[0]
    score = _axis_match_score(base, orient, a2)

    for line in marked_lines:
        mirror = a2 - int(line)
        if not 0 <= mirror < limit:
            score -= 300
            continue

        dst_count = _line_count(base, orient, int(line))
        src_count = _line_count(base, orient, mirror)
        marker_count = int(np.count_nonzero(markers[line, :])) if orient == "h" else int(np.count_nonzero(markers[:, line]))

        if dst_count == 0 and src_count > 0:
            score += 220 + min(src_count, 18) * 8 + marker_count * 4
        elif src_count > 0:
            score -= 90
        else:
            score -= 220

    return score


def _best_axis(base: np.ndarray, markers: np.ndarray, orient: str) -> tuple[int, int]:
    limit = base.shape[0] if orient == "h" else base.shape[1]
    best_a2 = 0
    best_score = None
    for a2 in range(2 * limit - 1):
        score = _candidate_score(base, markers, orient, a2)
        if best_score is None or score > best_score:
            best_score = score
            best_a2 = a2
    return best_a2, int(best_score if best_score is not None else 0)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-fill only the marked erased rows or columns across an inferred hidden symmetry axis.

    Input structure:
    The background is black, with one large colored pattern that is mostly symmetric across an
    unseen horizontal or vertical axis. Some complete rows or columns on one side of the pattern
    have been erased to background. Maroon pixels appear only on selected erased rows or columns
    as activation markers, while other erased lines have no marker.

    Rule:
    Remove all maroon markers, infer whether the remaining colored pattern is horizontally or
    vertically symmetric and where that axis lies, then inspect the marked erased lines. Each
    marked erased row or column is replaced by the mirror row or column from the opposite side of
    the inferred axis. Erased rows or columns without maroon markers stay black.

    Key insight:
    The hidden symmetry gives the missing source content, but maroon pixels are selective switches:
    symmetry alone is not enough, because only activated erased lines are restored.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    markers = out == MARK
    out[markers] = BG

    if not np.any(markers):
        return out

    h_axis, h_score = _best_axis(out, markers, "h")
    v_axis, v_score = _best_axis(out, markers, "v")

    if h_score >= v_score:
        rows = out.shape[0]
        marked_rows = np.where(np.any(markers, axis=1))[0]
        for r in marked_rows:
            src = h_axis - int(r)
            if 0 <= src < rows and np.count_nonzero(out[src, :]) > 0:
                out[r, :] = out[src, :]
    else:
        cols = out.shape[1]
        marked_cols = np.where(np.any(markers, axis=0))[0]
        for c in marked_cols:
            src = v_axis - int(c)
            if 0 <= src < cols and np.count_nonzero(out[:, src]) > 0:
                out[:, c] = out[:, src]

    out[out == MARK] = BG
    return out


def _colored_segments(length: int, colors: tuple[int, ...], rng: random.Random) -> list[tuple[int, int, int]]:
    first_len = rng.randint(3, max(3, min(7, length - 3)))
    start1 = rng.randint(0, length - first_len)
    segments = [(start1, start1 + first_len, rng.choice(colors))]

    if length >= 9 and rng.random() < 0.65:
        for _ in range(12):
            second_len = rng.randint(2, max(2, min(5, length - 2)))
            start2 = rng.randint(0, length - second_len)
            end2 = start2 + second_len
            if end2 < start1 - 1 or start2 > start1 + first_len + 1:
                segments.append((start2, end2, rng.choice(colors)))
                break
    return segments


def _paint_horizontal(base: np.ndarray, top: int, bottom: int, left: int, width: int, colors: tuple[int, ...], rng: random.Random) -> None:
    a2 = top + bottom
    for r in range(top, bottom + 1):
        mirror = a2 - r
        if r > mirror:
            continue
        segments = _colored_segments(width, colors, rng)
        for start, end, color in segments:
            base[r, left + start:left + end] = color
            base[mirror, left + start:left + end] = color

    for _ in range(rng.randint(3, 7)):
        c = left + rng.randrange(width)
        span_top = rng.randint(top, bottom)
        span_bottom = a2 - span_top
        lo, hi = sorted((span_top, span_bottom))
        color = rng.choice(colors)
        base[lo:hi + 1, c] = color


def _paint_vertical(base: np.ndarray, left: int, right: int, top: int, height: int, colors: tuple[int, ...], rng: random.Random) -> None:
    a2 = left + right
    for c in range(left, right + 1):
        mirror = a2 - c
        if c > mirror:
            continue
        segments = _colored_segments(height, colors, rng)
        for start, end, color in segments:
            base[top + start:top + end, c] = color
            base[top + start:top + end, mirror] = color

    for _ in range(rng.randint(3, 7)):
        r = top + rng.randrange(height)
        span_left = rng.randint(left, right)
        span_right = a2 - span_left
        lo, hi = sorted((span_left, span_right))
        color = rng.choice(colors)
        base[r, lo:hi + 1] = color


def _choose_missing_lines(lines: list[int], rng: random.Random) -> tuple[list[int], set[int]]:
    count = rng.randint(2, min(5, len(lines)))
    missing = rng.sample(lines, count)
    marked_count = rng.randint(1, count - 1)
    marked = set(rng.sample(missing, marked_count))
    return missing, marked


def _mark_row(inp: np.ndarray, source: np.ndarray, row: int, src: int, rng: random.Random) -> None:
    cols = np.where(source[src, :] != BG)[0].tolist()
    if not cols:
        return
    endpoints = sorted({cols[0], cols[-1]})
    inner = [c for c in cols if c not in endpoints]
    rng.shuffle(inner)
    picks = endpoints + inner[:rng.randint(0, min(2, len(inner)))]
    for c in picks:
        inp[row, c] = MARK


def _mark_col(inp: np.ndarray, source: np.ndarray, col: int, src: int, rng: random.Random) -> None:
    rows = np.where(source[:, src] != BG)[0].tolist()
    if not rows:
        return
    endpoints = sorted({rows[0], rows[-1]})
    inner = [r for r in rows if r not in endpoints]
    rng.shuffle(inner)
    picks = endpoints + inner[:rng.randint(0, min(2, len(inner)))]
    for r in picks:
        inp[r, col] = MARK


def generate_input() -> np.ndarray:
    rng = random.Random()
    orient = rng.choice(("h", "v"))
    rows = rng.randint(13, 30)
    cols = rng.randint(13, 30)
    grid = np.zeros((rows, cols), dtype=int)
    colors = tuple(rng.sample(COLORS, rng.randint(2, 4)))

    if orient == "h":
        height = rng.randint(9, min(19, rows - 2))
        width = rng.randint(8, min(22, cols - 2))
        top = rng.randint(1, rows - height - 1)
        bottom = top + height - 1
        left = rng.randint(1, cols - width - 1)
        axis2 = top + bottom
        _paint_horizontal(grid, top, bottom, left, width, colors, rng)

        above = [r for r in range(top + 1, (axis2 + 1) // 2) if axis2 - r != r]
        below = [axis2 - r for r in above]
        side_lines = rng.choice((above, below))
        if len(side_lines) < 2:
            side_lines = [r for r in range(top + 1, bottom) if axis2 - r != r]
        missing, marked = _choose_missing_lines(side_lines, rng)

        inp = grid.copy()
        for r in missing:
            inp[r, :] = BG
        for r in marked:
            _mark_row(inp, grid, r, axis2 - r, rng)
        return inp

    height = rng.randint(8, min(22, rows - 2))
    width = rng.randint(9, min(19, cols - 2))
    top = rng.randint(1, rows - height - 1)
    left = rng.randint(1, cols - width - 1)
    right = left + width - 1
    axis2 = left + right
    _paint_vertical(grid, left, right, top, height, colors, rng)

    left_side = [c for c in range(left + 1, (axis2 + 1) // 2) if axis2 - c != c]
    right_side = [axis2 - c for c in left_side]
    side_lines = rng.choice((left_side, right_side))
    if len(side_lines) < 2:
        side_lines = [c for c in range(left + 1, right) if axis2 - c != c]
    missing, marked = _choose_missing_lines(side_lines, rng)

    inp = grid.copy()
    for c in missing:
        inp[:, c] = BG
    for c in marked:
        _mark_col(inp, grid, c, axis2 - c, rng)
    return inp
