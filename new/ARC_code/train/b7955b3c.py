from __future__ import annotations

import numpy as np
from arc_utils import paste


UNKNOWN = 8
PLUS_SHAPE = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def _assert_plus_masks(g: np.ndarray) -> None:
    mask = g == UNKNOWN
    coords = np.argwhere(mask)
    assert coords.size, "expected at least one color-8 mask"

    seen = np.zeros(g.shape, dtype=bool)
    h, w = g.shape
    for r0, c0 in coords:
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        top, left = min(rs), min(cs)
        rel = {(r - top, c - left) for r, c in comp}
        assert rel == PLUS_SHAPE, "each color-8 component must be a 3x3 plus mask"


def _component_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _same_color_components(a: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue

            color = int(a[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(a[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append((color, comp))

    return components


def _background_color(a: np.ndarray) -> int:
    colors, counts = np.unique(a, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _plus_bars(g: np.ndarray) -> list[dict[str, set[tuple[int, int]]]]:
    mask = g == UNKNOWN
    seen = np.zeros(g.shape, dtype=bool)
    h, w = g.shape
    bars: list[dict[str, set[tuple[int, int]]]] = []

    for r0, c0 in map(tuple, np.argwhere(mask)):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        cr = (min(rs) + max(rs)) // 2
        cc = (min(cs) + max(cs)) // 2
        bars.append(
            {
                "hbar": {(cr, cc - 1), (cr, cc), (cr, cc + 1)},
                "vbar": {(cr - 1, cc), (cr, cc), (cr + 1, cc)},
            }
        )

    return bars


def _assert_mask_preserves_non_bg_extents(g: np.ndarray, out: np.ndarray) -> None:
    mask = g == UNKNOWN
    bg = _background_color(out)

    for color, comp in _same_color_components(out):
        if color == bg:
            continue

        hidden = [(r, c) for r, c in comp if mask[r, c]]
        if not hidden:
            continue

        visible = [(r, c) for r, c in comp if not mask[r, c]]
        assert visible, "mask cannot fully cover a non-background component"
        assert _component_bbox(visible) == _component_bbox(comp), (
            "mask cannot hide the visible height or width of a non-background component"
        )


def _assert_non_bg_support_is_irregular(out: np.ndarray) -> None:
    bg = _background_color(out)
    support = [(int(r), int(c)) for r, c in np.argwhere(out != bg)]
    assert support, "expected non-background mosaic cells"
    r0, r1, c0, c1 = _component_bbox(support)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert len(support) < bbox_area, "non-background cells must form an irregular support, not a filled rectangle"


def _assert_no_visible_short_side_against_plus(g: np.ndarray, out: np.ndarray) -> None:
    mask = g == UNKNOWN
    bars = _plus_bars(g)
    bg = _background_color(out)
    h, w = out.shape

    for color, comp_list in _same_color_components(out):
        if color == bg:
            continue

        comp = set(comp_list)
        r0, r1, c0, c1 = _component_bbox(comp_list)
        comp_h = r1 - r0 + 1
        comp_w = c1 - c0 + 1
        sides: list[tuple[list[tuple[int, int]], tuple[int, int], str]] = []
        if comp_w < 3:
            sides.append(([(r0, c) for c in range(c0, c1 + 1)], (-1, 0), "hbar"))
            sides.append(([(r1, c) for c in range(c0, c1 + 1)], (1, 0), "hbar"))
        if comp_h < 3:
            sides.append(([(r, c0) for r in range(r0, r1 + 1)], (0, -1), "vbar"))
            sides.append(([(r, c1) for r in range(r0, r1 + 1)], (0, 1), "vbar"))

        for side_cells, (dr, dc), bar_name in sides:
            if not all(cell in comp for cell in side_cells):
                continue
            if any(mask[r, c] for r, c in side_cells):
                continue

            adjacent = [(r + dr, c + dc) for r, c in side_cells]
            if not all(0 <= r < h and 0 <= c < w for r, c in adjacent):
                continue
            assert not any(
                all(cell in bar[bar_name] for cell in adjacent)
                for bar in bars
            ), "visible side shorter than 3 cannot be fully adjacent to a plus bar"


def _exact_parallel_line_fill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    h, w = out.shape

    changed = True
    while changed:
        changed = False
        for axis in (0, 1):
            n_lines = h if axis == 0 else w
            lines = [
                out[i, :].copy() if axis == 0 else out[:, i].copy()
                for i in range(n_lines)
            ]

            for i, line in enumerate(lines):
                for pos in np.flatnonzero(line == UNKNOWN):
                    best_support = -1
                    best_distance = n_lines + 1
                    best_values: set[int] = set()

                    for j, other in enumerate(lines):
                        if i == j or other[pos] == UNKNOWN:
                            continue

                        known = (line != UNKNOWN) & (other != UNKNOWN)
                        if not np.any(known) or np.any(line[known] != other[known]):
                            continue

                        support = int(np.sum(known))
                        distance = abs(i - j)
                        value = int(other[pos])
                        if support > best_support or (
                            support == best_support and distance < best_distance
                        ):
                            best_support = support
                            best_distance = distance
                            best_values = {value}
                        elif support == best_support and distance == best_distance:
                            best_values.add(value)

                    if len(best_values) == 1:
                        value = next(iter(best_values))
                        if axis == 0:
                            out[i, pos] = value
                        else:
                            out[pos, i] = value
                        changed = True

    return out


def _opposite_pair_fill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    h, w = out.shape

    changed = True
    while changed:
        changed = False
        for r, c in map(tuple, np.argwhere(out == UNKNOWN)):
            r = int(r)
            c = int(c)
            candidates: list[int] = []
            if r > 0 and r + 1 < h and out[r - 1, c] == out[r + 1, c] != UNKNOWN:
                candidates.append(int(out[r - 1, c]))
            if c > 0 and c + 1 < w and out[r, c - 1] == out[r, c + 1] != UNKNOWN:
                candidates.append(int(out[r, c - 1]))
            if candidates and len(set(candidates)) == 1:
                out[r, c] = candidates[0]
                changed = True

    return out


def _run_length(a: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    h, w = a.shape
    length = 0
    while 0 <= r < h and 0 <= c < w and int(a[r, c]) == color:
        length += 1
        r += dr
        c += dc
    return length


def _fill_remaining_runs(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    h, w = out.shape

    changed = True
    while changed and np.any(out == UNKNOWN):
        changed = False

        for r in range(h):
            c = 0
            while c < w:
                if out[r, c] != UNKNOWN:
                    c += 1
                    continue
                start = c
                while c < w and out[r, c] == UNKNOWN:
                    c += 1
                end = c - 1

                if start == 0 or end + 1 == w:
                    continue
                left = int(out[r, start - 1])
                right = int(out[r, end + 1])
                if left == UNKNOWN or right == UNKNOWN:
                    continue

                fill = _supported_horizontal_color(out, r, start, end, left, right)
                if fill is not None:
                    out[r, start : end + 1] = fill
                    changed = True

        for c in range(w):
            r = 0
            while r < h:
                if out[r, c] != UNKNOWN:
                    r += 1
                    continue
                start = r
                while r < h and out[r, c] == UNKNOWN:
                    r += 1
                end = r - 1

                if start == 0 or end + 1 == h:
                    continue
                up = int(out[start - 1, c])
                down = int(out[end + 1, c])
                if up == UNKNOWN or down == UNKNOWN:
                    continue

                fill = _supported_vertical_color(out, start, end, c, up, down)
                if fill is not None:
                    out[start : end + 1, c] = fill
                    changed = True

    return out


def _supported_horizontal_color(
    a: np.ndarray, r: int, start: int, end: int, left: int, right: int
) -> int | None:
    if left == right:
        return left

    h, _ = a.shape
    left_vertical = sum(
        1
        for rr in range(max(0, r - 1), min(h, r + 2))
        if int(a[rr, start - 1]) == left
    )
    right_vertical = sum(
        1
        for rr in range(max(0, r - 1), min(h, r + 2))
        if int(a[rr, end + 1]) == right
    )
    left_horizontal = _run_length(a, r, start - 1, 0, -1, left)
    right_horizontal = _run_length(a, r, end + 1, 0, 1, right)

    left_score = 2 * left_vertical + left_horizontal
    right_score = 2 * right_vertical + right_horizontal
    if left_score > right_score:
        return left
    if right_score > left_score:
        return right
    return None


def _supported_vertical_color(
    a: np.ndarray, start: int, end: int, c: int, up: int, down: int
) -> int | None:
    if up == down:
        return up

    _, w = a.shape
    up_horizontal = sum(
        1
        for cc in range(max(0, c - 1), min(w, c + 2))
        if int(a[start - 1, cc]) == up
    )
    down_horizontal = sum(
        1
        for cc in range(max(0, c - 1), min(w, c + 2))
        if int(a[end + 1, cc]) == down
    )
    up_vertical = _run_length(a, start - 1, c, -1, 0, up)
    down_vertical = _run_length(a, end + 1, c, 1, 0, down)

    up_score = 2 * up_horizontal + up_vertical
    down_score = 2 * down_horizontal + down_vertical
    if up_score > down_score:
        return up
    if down_score > up_score:
        return down
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace gray plus-shaped damage masks with the hidden colors of the rectilinear mosaic.

    Input structure:
    - Color 8 appears only as isolated 3x3 plus masks.
    - The non-mask pixels are visible parts of a clean grid made from horizontal
      and vertical color regions.

    Rule:
    - Treat every color-8 pixel as unknown damage and leave all other pixels fixed.
    - First copy unknown pixels from exact matching parallel rows or columns.
    - Then fill any remaining mask cells from opposite-side continuations and the
      better-supported local straight edge.

    Key insight:
    - The gray pluses are occluders, not objects; the underlying colors are
      determined by repeated line templates and nearby axis-aligned boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    _assert_plus_masks(g)

    out = _exact_parallel_line_fill(g)
    out = _opposite_pair_fill(out)
    out = _fill_remaining_runs(out)
    assert not np.any(out == UNKNOWN), "all plus-mask pixels must be repaired"
    _assert_mask_preserves_non_bg_extents(g, out)
    _assert_non_bg_support_is_irregular(out)
    _assert_no_visible_short_side_against_plus(g, out)
    return out

# --------------- generator (att2) ---------------

_GEN_UNKNOWN = 8
_GEN_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GEN_PLUS_MASK = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_GEN_PLUS_PATCH = np.full((3, 3), _GEN_UNKNOWN, dtype=int)

# Canvas dimensions are sampled by partitioning into small row/column blocks.
_GEN_ROW_TOTAL_RANGE = (8, 26)
_GEN_COL_TOTAL_RANGE = (8, 22)
_GEN_ROW_BLOCK_COUNT_RANGE = (5, 11)
_GEN_COL_BLOCK_COUNT_RANGE = (5, 12)
_GEN_ROW_BLOCK_SIZE_RANGE = (1, 5)
_GEN_COL_BLOCK_SIZE_RANGE = (1, 4)

# Palette and local rectangle painting controls.
_GEN_COLOR_COUNT_RANGE = (3, 7)
_GEN_PAINT_MODE_COUNT = 4
_GEN_KEEP_BASE_MODE_PROB = 0.65
_GEN_EXTRA_RECT_MIN, _GEN_EXTRA_RECT_SLACK = 0, 4

# Plus-mask placement controls.
_GEN_MASK_COUNT_RANGE = (1, 6)
_GEN_EDGE_MASK_PROB = 0.12
_GEN_CENTER_MARGIN, _GEN_CENTER_MIN_MANHATTAN = 1, 6

# Generation retry limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_partition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    values: list[int] = []
    remaining = int(total)
    for i in range(int(parts)):
        slots = int(parts) - i - 1
        min_v = max(int(lo), remaining - int(hi) * slots)
        max_v = min(int(hi), remaining - int(lo) * slots)
        if min_v > max_v:
            return None
        v = int(np.random.randint(min_v, max_v + 1))
        values.append(v)
        remaining -= v
    np.random.shuffle(values)
    return values


def _gen_choose_partition(
    total_lo: int, total_hi: int, n_lo: int, n_hi: int, part_lo: int, part_hi: int
) -> list[int] | None:
    total = int(np.random.randint(total_lo, total_hi + 1))
    valid = [n for n in range(n_lo, n_hi + 1) if part_lo * n <= total <= part_hi * n]
    if not valid:
        return None
    n = int(np.random.choice(valid))
    return _gen_partition(total, n, part_lo, part_hi)


def _gen_paint_rect(cells: np.ndarray, color: int, mode: int) -> None:
    rows, cols = cells.shape
    inner_h = rows - 2
    inner_w = cols - 2
    if inner_h <= 0 or inner_w <= 0:
        return

    if mode == 0:
        h = int(np.random.randint(1, inner_h + 1))
        w = int(np.random.randint(1, inner_w + 1))
    elif mode == 1:
        h = int(np.random.randint(1, inner_h + 1))
        w = int(np.random.randint(max(1, inner_w // 2), inner_w + 1))
    elif mode == 2:
        h = int(np.random.randint(max(1, inner_h // 2), inner_h + 1))
        w = int(np.random.randint(1, inner_w + 1))
    else:
        h = int(np.random.randint(1, max(2, inner_h // 2 + 1)))
        w = int(np.random.randint(1, max(2, inner_w // 2 + 1)))

    r0 = int(np.random.randint(1, rows - h))
    c0 = int(np.random.randint(1, cols - w))
    cells[r0 : r0 + h, c0 : c0 + w] = int(color)


def _gen_random_paint_mode() -> int:
    return int(np.random.randint(0, _GEN_PAINT_MODE_COUNT))


def _gen_weight_counts(
    cells: np.ndarray, row_heights: list[int], col_widths: list[int]
) -> dict[int, int]:
    weights = np.asarray(row_heights, dtype=int)[:, None] * np.asarray(col_widths, dtype=int)[None, :]
    return {
        int(color): int(weights[cells == int(color)].sum())
        for color in np.unique(cells)
    }


def _gen_rebalance_background(
    cells: np.ndarray, row_heights: list[int], col_widths: list[int], bg: int, fill_colors: list[int]
) -> bool:
    weights = np.asarray(row_heights, dtype=int)[:, None] * np.asarray(col_widths, dtype=int)[None, :]
    for _ in range(cells.size):
        counts = _gen_weight_counts(cells, row_heights, col_widths)
        bg_count = counts.get(int(bg), 0)
        heavy = max((c for c in counts if c != int(bg)), key=lambda c: counts[c], default=None)
        if heavy is None or bg_count > counts[heavy]:
            return all(np.any(cells == int(color)) for color in fill_colors)

        coords = np.argwhere(cells[1:-1, 1:-1] == int(heavy)) + 1
        if len(coords) <= 1:
            return False
        coord_weights = weights[coords[:, 0], coords[:, 1]]
        r, c = map(int, coords[int(np.argmax(coord_weights))])
        cells[r, c] = int(bg)

    return False


def _gen_expand_mosaic(
    cells: np.ndarray, row_heights: list[int], col_widths: list[int]
) -> np.ndarray:
    return np.repeat(np.repeat(cells, row_heights, axis=0), col_widths, axis=1)


def _gen_plus_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c - 1), (r, c), (r, c + 1), (r + 1, c)]


def _gen_boundary_marks(sizes: list[int], limit: int) -> set[int]:
    marks: set[int] = set()
    for cut in np.cumsum(sizes)[:-1]:
        for p in (int(cut) - 1, int(cut)):
            if _GEN_CENTER_MARGIN <= p <= int(limit) - _GEN_CENTER_MARGIN - 1:
                marks.add(p)
    return marks


def _gen_is_edge_touch(shape: tuple[int, int], center: tuple[int, int]) -> bool:
    h, w = shape
    r, c = center
    return (
        r == _GEN_CENTER_MARGIN
        or c == _GEN_CENTER_MARGIN
        or r == h - _GEN_CENTER_MARGIN - 1
        or c == w - _GEN_CENTER_MARGIN - 1
    )


def _gen_spaced(
    selected: list[tuple[int, int]], center: tuple[int, int], shape: tuple[int, int]
) -> bool:
    if sum(_gen_is_edge_touch(shape, p) for p in selected) and _gen_is_edge_touch(shape, center):
        return False
    r, c = center
    return all(abs(r - rr) + abs(c - cc) >= _GEN_CENTER_MIN_MANHATTAN for rr, cc in selected)


def _gen_candidate_centers(
    out: np.ndarray, row_heights: list[int], col_widths: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = out.shape
    row_marks = _gen_boundary_marks(row_heights, h)
    col_marks = _gen_boundary_marks(col_widths, w)
    boundary: list[tuple[int, int]] = []
    all_centers: list[tuple[int, int]] = []

    for r in range(_GEN_CENTER_MARGIN, h - _GEN_CENTER_MARGIN):
        for c in range(_GEN_CENTER_MARGIN, w - _GEN_CENTER_MARGIN):
            center = (r, c)
            values = {int(out[rr, cc]) for rr, cc in _gen_plus_cells(r, c)}
            all_centers.append(center)
            if len(values) > 1 and (r in row_marks or c in col_marks):
                boundary.append(center)
    return boundary, all_centers


def _gen_select_centers(
    out: np.ndarray, row_heights: list[int], col_widths: list[int], count: int
) -> list[tuple[int, int]] | None:
    boundary, all_centers = _gen_candidate_centers(out, row_heights, col_widths)
    allow_edge = bool(np.random.random() < _GEN_EDGE_MASK_PROB)
    if not allow_edge:
        boundary = [p for p in boundary if not _gen_is_edge_touch(out.shape, p)]
        all_centers = [p for p in all_centers if not _gen_is_edge_touch(out.shape, p)]

    np.random.shuffle(boundary)
    np.random.shuffle(all_centers)
    selected: list[tuple[int, int]] = []

    for pool in (boundary, all_centers):
        changed = True
        while changed and len(selected) < int(count):
            changed = False
            for center in pool:
                if center in selected or not _gen_spaced(selected, center, out.shape):
                    continue
                selected.append(center)
                changed = True
                break

    if len(selected) != int(count):
        return None
    return selected


def _gen_paint_mosaic_cells(
    row_heights: list[int], col_widths: list[int], bg: int, fill_colors: list[int]
) -> np.ndarray:
    cells = np.full((len(row_heights), len(col_widths)), bg, dtype=int)
    mode = _gen_random_paint_mode()
    extra_hi = max(2, len(row_heights) + len(col_widths) - _GEN_EXTRA_RECT_SLACK)
    extra = int(np.random.randint(_GEN_EXTRA_RECT_MIN, extra_hi))
    draw_colors = list(map(int, np.random.choice(fill_colors, extra, replace=True)))
    for color in draw_colors:
        paint_mode = mode if np.random.random() < _GEN_KEEP_BASE_MODE_PROB else _gen_random_paint_mode()
        _gen_paint_rect(cells, int(color), paint_mode)

    required_colors = list(fill_colors)
    np.random.shuffle(required_colors)
    for color in required_colors:
        _gen_paint_rect(cells, int(color), _gen_random_paint_mode())

    for color in fill_colors:
        if not np.any(cells[1:-1, 1:-1] == int(color)):
            _gen_paint_rect(cells, int(color), _GEN_PAINT_MODE_COUNT - 1)
        if not np.any(cells[1:-1, 1:-1] == int(color)):
            r = int(np.random.randint(1, cells.shape[0] - 1))
            c = int(np.random.randint(1, cells.shape[1] - 1))
            cells[r, c] = int(color)
    return cells


def _gen_build_output() -> tuple[np.ndarray, list[int], list[int]] | None:
    row_heights = _gen_choose_partition(
        *_GEN_ROW_TOTAL_RANGE, *_GEN_ROW_BLOCK_COUNT_RANGE, *_GEN_ROW_BLOCK_SIZE_RANGE
    )
    col_widths = _gen_choose_partition(
        *_GEN_COL_TOTAL_RANGE, *_GEN_COL_BLOCK_COUNT_RANGE, *_GEN_COL_BLOCK_SIZE_RANGE
    )
    if row_heights is None or col_widths is None:
        return None

    n_colors = int(np.random.randint(_GEN_COLOR_COUNT_RANGE[0], _GEN_COLOR_COUNT_RANGE[1] + 1))
    palette = list(map(int, np.random.choice(_GEN_COLORS, n_colors, replace=False)))
    bg = int(np.random.choice(palette))
    fill_colors = [color for color in palette if color != bg]
    cells = _gen_paint_mosaic_cells(row_heights, col_widths, bg, fill_colors)

    if not _gen_rebalance_background(cells, row_heights, col_widths, bg, fill_colors):
        return None
    return _gen_expand_mosaic(cells, row_heights, col_widths), row_heights, col_widths


def _gen_stamp_masks(out: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    inp = out.copy()
    for r, c in centers:
        paste(inp, _GEN_PLUS_PATCH, r - 1, c - 1, mask=_GEN_PLUS_MASK)
    return inp


def _gen_attempt() -> np.ndarray | None:
    built = _gen_build_output()
    if built is None:
        return None
    out, row_heights, col_widths = built

    count = int(np.random.randint(_GEN_MASK_COUNT_RANGE[0], _GEN_MASK_COUNT_RANGE[1] + 1))
    centers = _gen_select_centers(out, row_heights, col_widths, count)
    if centers is None:
        return None

    inp = _gen_stamp_masks(out, centers)
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid b7955b3c input")
