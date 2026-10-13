import itertools
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
EVIDENCE_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_evidence(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    left_hits: dict[int, int] = {}
    right_hits: dict[int, int] = {}
    top_hits: dict[int, int] = {}
    bottom_hits: dict[int, int] = {}
    full_rows: set[int] = set()
    full_cols: set[int] = set()

    for r in range(h):
        if g[r, 0] != bg:
            c = 0
            while c < w and g[r, c] != bg:
                c += 1
            if c == w:
                full_rows.add(r)
            else:
                left_hits[r] = c
        if g[r, w - 1] != bg:
            c = w - 1
            while c >= 0 and g[r, c] != bg:
                c -= 1
            if c < 0:
                full_rows.add(r)
            else:
                right_hits[r] = c

    for c in range(w):
        if g[0, c] != bg:
            r = 0
            while r < h and g[r, c] != bg:
                r += 1
            if r == h:
                full_cols.add(c)
            else:
                top_hits[c] = r
        if g[h - 1, c] != bg:
            r = h - 1
            while r >= 0 and g[r, c] != bg:
                r -= 1
            if r < 0:
                full_cols.add(c)
            else:
                bottom_hits[c] = r

    return {
        "h": h,
        "w": w,
        "bg": bg,
        "left_hits": left_hits,
        "right_hits": right_hits,
        "top_hits": top_hits,
        "bottom_hits": bottom_hits,
        "full_rows": full_rows,
        "full_cols": full_cols,
    }


def _covers_row(rect: tuple[int, int, int, int], row: int) -> bool:
    return rect[0] <= row <= rect[2]


def _covers_col(rect: tuple[int, int, int, int], col: int) -> bool:
    return rect[1] <= col <= rect[3]


def _expected_left(rects: tuple[tuple[int, int, int, int], ...], row: int) -> int | None:
    edges = [rect[1] for rect in rects if _covers_row(rect, row)]
    return None if not edges else min(edges)


def _expected_right(rects: tuple[tuple[int, int, int, int], ...], row: int) -> int | None:
    edges = [rect[3] for rect in rects if _covers_row(rect, row)]
    return None if not edges else max(edges)


def _expected_top(rects: tuple[tuple[int, int, int, int], ...], col: int) -> int | None:
    edges = [rect[0] for rect in rects if _covers_col(rect, col)]
    return None if not edges else min(edges)


def _expected_bottom(rects: tuple[tuple[int, int, int, int], ...], col: int) -> int | None:
    edges = [rect[2] for rect in rects if _covers_col(rect, col)]
    return None if not edges else max(edges)


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _candidate_rectangles(evidence: dict) -> list[tuple[int, int, int, int]]:
    lefts = sorted(set(evidence["left_hits"].values()))
    rights = sorted(set(evidence["right_hits"].values()))
    tops = sorted(set(evidence["top_hits"].values()))
    bottoms = sorted(set(evidence["bottom_hits"].values()))

    out: set[tuple[int, int, int, int]] = set()
    for r0 in tops:
        for r1 in bottoms:
            if r1 < r0:
                continue
            if any(row in evidence["full_rows"] for row in range(r0, r1 + 1)):
                continue
            for c0 in lefts:
                for c1 in rights:
                    if c1 < c0:
                        continue
                    if any(col in evidence["full_cols"] for col in range(c0, c1 + 1)):
                        continue
                    rows = range(r0, r1 + 1)
                    cols = range(c0, c1 + 1)
                    if not any(evidence["left_hits"].get(row) == c0 for row in rows):
                        continue
                    if not any(evidence["right_hits"].get(row) == c1 for row in rows):
                        continue
                    if not any(evidence["top_hits"].get(col) == r0 for col in cols):
                        continue
                    if not any(evidence["bottom_hits"].get(col) == r1 for col in cols):
                        continue
                    out.add((r0, c0, r1, c1))
    return sorted(out)


def _pair_is_consistent(
    rects: tuple[tuple[int, int, int, int], ...],
    evidence: dict,
) -> bool:
    for row, col in evidence["left_hits"].items():
        if _expected_left(rects, row) != col:
            return False
    for row, col in evidence["right_hits"].items():
        if _expected_right(rects, row) != col:
            return False
    for col, row in evidence["top_hits"].items():
        if _expected_top(rects, col) != row:
            return False
    for col, row in evidence["bottom_hits"].items():
        if _expected_bottom(rects, col) != row:
            return False
    for row in evidence["full_rows"]:
        if _expected_left(rects, row) is not None or _expected_right(rects, row) is not None:
            return False
    for col in evidence["full_cols"]:
        if _expected_top(rects, col) is not None or _expected_bottom(rects, col) is not None:
            return False
    return True


def _solve_rectangles(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    evidence = _extract_evidence(grid)
    candidates = _candidate_rectangles(evidence)
    solutions: set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = set()

    for a, b in itertools.combinations(candidates, 2):
        if _rectangles_overlap(a, b):
            continue
        pair = tuple(sorted((a, b)))
        if _pair_is_consistent(pair, evidence):
            solutions.add(pair)

    if len(solutions) != 1:
        raise ValueError(f"expected exactly one rectangle pair, found {len(solutions)}")
    return next(iter(solutions))


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the two invisible rectangles from where boundary-entering segments get truncated.

    Input structure:
    - The grid has a uniform background and one-cell-thick straight evidence segments entering from the four borders.
    - Every segment is horizontal or vertical; some traverse the whole grid while others stop one cell before an unseen solid rectangle.
    - Exactly two non-overlapping rectangles are absent from the input pixels; blue is reserved for the recovered output rectangles.

    Rule:
    - Interpret each truncated left/right segment endpoint as revealing a rectangle's left/right edge on that row, and each truncated top/bottom endpoint as revealing a top/bottom edge on that column.
    - Use full-through rows and columns to rule out rectangle coverage there, then search for the unique pair of rectangles whose row and column spans explain every hit and every pass-through.
    - Keep all evidence segments unchanged and fill both recovered rectangles in blue.

    Key insight:
    - The rectangles are never drawn directly; they are uniquely determined by the causal pattern of which boundary rays stop, which pass through, and which row or column ranges can consistently belong to the same hidden box.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    rects = _solve_rectangles(g)
    out = g.copy()
    for r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return out


def _draw_left(grid: np.ndarray, row: int, stop_col: int, color: int) -> None:
    grid[row, :stop_col] = color


def _draw_right(grid: np.ndarray, row: int, stop_col: int, color: int) -> None:
    grid[row, stop_col + 1 :] = color


def _draw_top(grid: np.ndarray, col: int, stop_row: int, color: int) -> None:
    grid[:stop_row, col] = color


def _draw_bottom(grid: np.ndarray, col: int, stop_row: int, color: int) -> None:
    grid[stop_row + 1 :, col] = color


def _rows(rect: tuple[int, int, int, int]) -> set[int]:
    return set(range(rect[0], rect[2] + 1))


def _cols(rect: tuple[int, int, int, int]) -> set[int]:
    return set(range(rect[1], rect[3] + 1))


def _pick_some(values: set[int], min_count: int, max_count: int) -> set[int]:
    ordered = sorted(values)
    count = random.randint(min_count, min(max_count, len(ordered)))
    return set(random.sample(ordered, count))


def _sample_side_by_side(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    shared = random.randint(2, 4)
    top_only = random.randint(1, 3)
    bottom_only = random.randint(1, 3)
    height1 = shared + top_only
    height2 = shared + bottom_only
    width1 = random.randint(3, 6)
    width2 = random.randint(3, 6)
    gap = random.randint(2, 5)

    total_h = top_only + shared + bottom_only
    total_w = width1 + gap + width2
    if total_h > h - 2 or total_w > w - 2:
        return None

    r0 = random.randint(1, h - total_h - 1)
    c0 = random.randint(1, w - total_w - 1)

    rect1 = (r0, c0, r0 + height1 - 1, c0 + width1 - 1)
    rect2 = (
        r0 + top_only,
        c0 + width1 + gap,
        r0 + top_only + height2 - 1,
        c0 + width1 + gap + width2 - 1,
    )
    return rect1, rect2


def _sample_stacked(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    shared = random.randint(2, 4)
    left_only = random.randint(1, 3)
    right_only = random.randint(1, 3)
    width1 = shared + left_only
    width2 = shared + right_only
    height1 = random.randint(3, 6)
    height2 = random.randint(3, 6)
    gap = random.randint(2, 5)

    total_h = height1 + gap + height2
    total_w = left_only + shared + right_only
    if total_h > h - 2 or total_w > w - 2:
        return None

    r0 = random.randint(1, h - total_h - 1)
    c0 = random.randint(1, w - total_w - 1)

    rect1 = (r0, c0, r0 + height1 - 1, c0 + width1 - 1)
    rect2 = (
        r0 + height1 + gap,
        c0 + left_only,
        r0 + height1 + gap + height2 - 1,
        c0 + left_only + width2 - 1,
    )
    return rect1, rect2


def _build_side_by_side_case(
    h: int,
    w: int,
    color: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] | None:
    sampled = _sample_side_by_side(h, w)
    if sampled is None:
        return None
    rect1, rect2 = sampled

    rows1 = _rows(rect1)
    rows2 = _rows(rect2)
    cols1 = _cols(rect1)
    cols2 = _cols(rect2)
    shared_rows = rows1 & rows2
    only1 = rows1 - rows2
    only2 = rows2 - rows1
    outside_rows = set(range(1, h - 1)) - (rows1 | rows2)
    outside_cols = set(range(1, w - 1)) - (cols1 | cols2)

    left_rows = {random.choice(sorted(rows1)), random.choice(sorted(only2))}
    right_rows = {random.choice(sorted(only1)), random.choice(sorted(rows2))}
    left_rows |= _pick_some(rows1 | rows2, 0, 3)
    right_rows |= _pick_some(rows1 | rows2, 0, 3)

    top_cols = {random.choice(sorted(cols1)), random.choice(sorted(cols2))}
    bottom_cols = {random.choice(sorted(cols1)), random.choice(sorted(cols2))}
    top_cols |= _pick_some(cols1 | cols2, 0, 2)
    bottom_cols |= _pick_some(cols1 | cols2, 0, 2)

    if outside_rows:
        left_rows |= _pick_some(outside_rows, 1, min(3, len(outside_rows)))
    if outside_cols:
        top_cols |= _pick_some(outside_cols, 1, min(3, len(outside_cols)))

    grid = np.full((h, w), BACKGROUND, dtype=int)

    for row in left_rows:
        if row in rows1 or row in rows2:
            boundary = rect1[1] if row in rows1 else rect2[1]
            _draw_left(grid, row, boundary, color)
        else:
            _draw_left(grid, row, w, color)
    for row in right_rows:
        if row in rows1 or row in rows2:
            boundary = rect2[3] if row in rows2 else rect1[3]
            _draw_right(grid, row, boundary, color)
        else:
            _draw_right(grid, row, -1, color)
    for col in top_cols:
        if col in cols1 or col in cols2:
            boundary = rect1[0] if col in cols1 else rect2[0]
            _draw_top(grid, col, boundary, color)
        else:
            _draw_top(grid, col, h, color)
    for col in bottom_cols:
        if col in cols1 or col in cols2:
            boundary = rect2[2] if col in cols2 else rect1[2]
            _draw_bottom(grid, col, boundary, color)
        else:
            _draw_bottom(grid, col, -1, color)

    if not shared_rows or not only1 or not only2:
        return None
    return grid, tuple(sorted((rect1, rect2)))


def _build_stacked_case(
    h: int,
    w: int,
    color: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] | None:
    sampled = _sample_stacked(h, w)
    if sampled is None:
        return None
    rect1, rect2 = sampled

    rows1 = _rows(rect1)
    rows2 = _rows(rect2)
    cols1 = _cols(rect1)
    cols2 = _cols(rect2)
    shared_cols = cols1 & cols2
    only1 = cols1 - cols2
    only2 = cols2 - cols1
    outside_rows = set(range(1, h - 1)) - (rows1 | rows2)
    outside_cols = set(range(1, w - 1)) - (cols1 | cols2)

    left_rows = {random.choice(sorted(rows1)), random.choice(sorted(rows2))}
    right_rows = {random.choice(sorted(rows1)), random.choice(sorted(rows2))}
    left_rows |= _pick_some(rows1 | rows2, 0, 3)
    right_rows |= _pick_some(rows1 | rows2, 0, 3)

    top_cols = {random.choice(sorted(cols1)), random.choice(sorted(only2))}
    bottom_cols = {random.choice(sorted(only1)), random.choice(sorted(cols2))}
    top_cols |= _pick_some(cols1 | cols2, 0, 3)
    bottom_cols |= _pick_some(cols1 | cols2, 0, 3)

    if outside_rows:
        left_rows |= _pick_some(outside_rows, 1, min(3, len(outside_rows)))
    if outside_cols:
        top_cols |= _pick_some(outside_cols, 1, min(3, len(outside_cols)))

    grid = np.full((h, w), BACKGROUND, dtype=int)

    for row in left_rows:
        if row in rows1:
            _draw_left(grid, row, rect1[1], color)
        elif row in rows2:
            _draw_left(grid, row, rect2[1], color)
        else:
            _draw_left(grid, row, w, color)
    for row in right_rows:
        if row in rows1:
            _draw_right(grid, row, rect1[3], color)
        elif row in rows2:
            _draw_right(grid, row, rect2[3], color)
        else:
            _draw_right(grid, row, -1, color)
    for col in top_cols:
        if col in cols1 or col in cols2:
            boundary = rect1[0] if col in cols1 else rect2[0]
            _draw_top(grid, col, boundary, color)
        else:
            _draw_top(grid, col, h, color)
    for col in bottom_cols:
        if col in cols1 or col in cols2:
            boundary = rect2[2] if col in cols2 else rect1[2]
            _draw_bottom(grid, col, boundary, color)
        else:
            _draw_bottom(grid, col, -1, color)

    if not shared_cols or not only1 or not only2:
        return None
    return grid, tuple(sorted((rect1, rect2)))


def _oracle_output(
    grid: np.ndarray,
    rects: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    color = random.choice(EVIDENCE_COLORS)

    builder = _build_side_by_side_case if random.random() < 0.5 else _build_stacked_case
    built = builder(h, w, color)
    if built is None:
        return None

    inp, rects = built
    out = _oracle_output(inp, rects)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid invisible-rectangle input")
