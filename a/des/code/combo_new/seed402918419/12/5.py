from __future__ import annotations

import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    cell_set = set(cells)
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(cell_set) != expected:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in cell_set):
                return False
    return True


def _find_target_box(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for cells in _components(grid == color):
            if not _is_frame_component(cells):
                continue
            r0, c0, r1, c1 = _bbox(cells)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not bool(np.all(inner == bg)):
                continue
            area = int(inner.shape[0] * inner.shape[1])
            candidates.append((area, color, (r0, c0, r1, c1)))
    assert candidates, "expected a framed target box"
    _area, color, bbox = max(candidates, key=lambda item: (item[0], item[2][0], item[2][1]))
    return int(color), bbox


def _legend_pattern(grid: np.ndarray, row: int, col: int, border: int, bg: int) -> bool:
    return (
        grid[row, col - 1] == border
        and grid[row, col + 1] == border
        and grid[row, col] != border
        and grid[row, col] != bg
    )


def _find_legend(
    grid: np.ndarray,
    bg: int,
    border: int,
    box_bbox: tuple[int, int, int, int],
) -> tuple[int, list[int], list[int]]:
    box_top = int(box_bbox[0])
    k = int(box_bbox[2] - box_bbox[0] - 1)
    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[int]]] = []
    for col in range(1, w - 1):
        row = 0
        while row < box_top:
            if not _legend_pattern(grid, row, col, border, bg):
                row += 1
                continue
            start = row
            colors: list[int] = []
            while row < box_top and _legend_pattern(grid, row, col, border, bg):
                colors.append(int(grid[row, col]))
                row += 1
            if len(colors) == k and len(set(colors)) == k:
                candidates.append((start, row - 1, col, colors))
    assert candidates, "expected a bordered legend column above the target box"
    start, end, col, colors = min(
        candidates,
        key=lambda item: (box_top - item[1], item[0], abs(item[2] - (box_bbox[1] + box_bbox[3]) // 2)),
    )
    rows = list(range(int(start), int(end) + 1))
    return int(col), rows, [int(color) for color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble colored horizontal fragments into stacked rows ordered by the right-side legend.

    Input structure:
    - The background is black.
    - The left part of the grid contains several one-pixel-high horizontal fragments in multiple colors.
    - The right side contains a vertical legend column: each row has one color pixel between two border-color vertical rails.
    - Below the legend is an empty rectangular target box drawn in the same border color.

    Rule:
    - Read the legend colors from top to bottom; this fixes the top-to-bottom order of rows inside the target box.
    - For each legend color, collect every left-side fragment pixel of that color and copy it into the matching target row at the same column offset.
    - If the same color contributes to the same target column multiple times, keep the pixel coming from the uppermost source row.
    - Erase the original left-side fragments to black while keeping the legend and border structure unchanged.

    Key insight:
    - The legend determines vertical ordering, but the fragment pixels keep their original horizontal coordinates, so each output row is formed by overlaying same-color fragments rather than concatenating them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0

    border, box = _find_target_box(g, bg)
    _legend_col, _legend_rows, order = _find_legend(g, bg, border, box)

    r0, c0, r1, c1 = box
    left_width = int(c1 - c0 - 1)
    assert left_width > 0
    assert len(order) == int(r1 - r0 - 1)

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = bg

    color_set = set(order)
    left = g[:, :left_width]
    erase_mask = np.isin(left, list(color_set))
    left_out = out[:, :left_width]
    left_out[erase_mask] = bg

    for idx, color in enumerate(order):
        target_row = int(r0 + 1 + idx)
        for src_row in range(g.shape[0]):
            row_mask = g[src_row, :left_width] == int(color)
            if not bool(row_mask.any()):
                continue
            target = out[target_row, c0 + 1 : c1]
            write_mask = row_mask & (target == bg)
            target[write_mask] = int(color)

    return out


# --------------- generator (GPT-5) ---------------

_RNG = np.random.default_rng()

_GEN_RETRIES = 400
_COLOR_COUNT_RANGE = (2, 5)
_LEFT_WIDTH_RANGE = (7, 11)
_GAP_X_RANGE = (2, 5)
_TOP_MARGIN_RANGE = (0, 3)
_LEGEND_BOX_GAP_RANGE = (1, 3)
_BOTTOM_MARGIN_RANGE = (0, 4)
_RIGHT_MARGIN_RANGE = (0, 1)


def _sample_parts(total: int, count: int, min_each: int = 1) -> list[int]:
    assert count >= 1
    assert total >= count * min_each
    base = np.full(count, min_each, dtype=int)
    remaining = int(total - count * min_each)
    if remaining:
        base += _RNG.multinomial(remaining, np.full(count, 1.0 / count))
    return [int(v) for v in base]


def _sample_final_intervals(width: int, require_long: bool) -> list[tuple[int, int]] | None:
    max_segments = min(3, width)
    for _ in range(200):
        n_segments = int(_RNG.integers(1, max_segments + 1))
        min_cover = max(n_segments, min(width - 1, max(3, width // 3)))
        max_cover = int(width - (n_segments - 1))
        if n_segments == 1:
            max_cover = min(max_cover, width - 1)
        if min_cover > max_cover:
            continue
        cover = int(_RNG.integers(min_cover, max_cover + 1))
        if cover + (n_segments - 1) > width:
            continue
        lengths = _sample_parts(cover, n_segments, min_each=1)
        if require_long and max(lengths) < 4:
            continue
        slack = int(width - cover - (n_segments - 1))
        extras = _RNG.multinomial(slack, np.full(n_segments + 1, 1.0 / (n_segments + 1)))
        cursor = int(extras[0])
        intervals: list[tuple[int, int]] = []
        for i, seg_len in enumerate(lengths):
            start = cursor
            end = start + seg_len - 1
            intervals.append((int(start), int(end)))
            cursor = end + 1
            if i < n_segments - 1:
                cursor += 1 + int(extras[i + 1])
        if cursor + int(extras[-1]) != width:
            continue
        if len(intervals) == 1 and intervals[0] == (0, width - 1):
            continue
        return intervals
    return None


def _fragment_one_interval(
    start: int,
    end: int,
    must_overlap: bool,
    may_overlap: bool,
) -> tuple[list[tuple[int, int]], bool] | None:
    length = int(end - start + 1)
    if length <= 0:
        return None
    if must_overlap and length < 4:
        return None

    if must_overlap:
        pieces = 2 if length < 6 else int(_RNG.integers(2, min(4, length // 2 + 1)))
        part_lengths = _sample_parts(length, pieces, min_each=2)
    else:
        pieces = int(_RNG.integers(1, min(3, length) + 1))
        min_each = 1
        if pieces == 1:
            part_lengths = [length]
        else:
            part_lengths = _sample_parts(length, pieces, min_each=min_each)

    starts: list[int] = []
    ends: list[int] = []
    cursor = start
    for part_len in part_lengths:
        starts.append(int(cursor))
        ends.append(int(cursor + part_len - 1))
        cursor += int(part_len)

    used_overlap = False
    if pieces > 1 and (must_overlap or may_overlap):
        forced_boundary = int(_RNG.integers(pieces - 1)) if must_overlap else -1
        for i in range(pieces - 1):
            left_len = int(part_lengths[i])
            right_len = int(part_lengths[i + 1])
            max_ov = min(2, left_len - 1, right_len - 1)
            if max_ov <= 0:
                continue
            if i == forced_boundary:
                ov = int(_RNG.integers(1, max_ov + 1))
            elif may_overlap and bool(_RNG.integers(2)):
                ov = int(_RNG.integers(1, max_ov + 1))
            else:
                ov = 0
            if ov:
                ends[i] += ov
                starts[i + 1] -= ov
                used_overlap = True
    if must_overlap and not used_overlap:
        return None
    return list(zip(starts, ends)), used_overlap


def _sample_color_spec(width: int, force_overlap: bool) -> dict[str, object] | None:
    final_intervals = _sample_final_intervals(width, require_long=force_overlap)
    if final_intervals is None:
        return None

    overlap_host = None
    if force_overlap:
        long_indices = [i for i, (s, e) in enumerate(final_intervals) if e - s + 1 >= 4]
        if not long_indices:
            return None
        overlap_host = int(_RNG.choice(np.array(long_indices, dtype=int)))

    fragments: list[tuple[int, int]] = []
    used_overlap = False
    for i, (start, end) in enumerate(final_intervals):
        piece = _fragment_one_interval(
            int(start),
            int(end),
            must_overlap=(overlap_host == i),
            may_overlap=(force_overlap or bool(_RNG.integers(2))),
        )
        if piece is None:
            return None
        piece_intervals, piece_overlap = piece
        used_overlap = used_overlap or piece_overlap
        fragments.extend(piece_intervals)

    if len(fragments) < 2:
        return None
    return {
        "final_intervals": [(int(s), int(e)) for s, e in final_intervals],
        "fragments": [(int(s), int(e)) for s, e in fragments],
        "used_overlap": bool(used_overlap),
    }


def _render_case(
    height: int,
    width: int,
    left_width: int,
    border: int,
    box_bbox: tuple[int, int, int, int],
    legend_col: int,
    legend_top: int,
    order: list[int],
    specs: list[dict[str, object]],
    fragment_rows: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)

    r0, c0, r1, c1 = box_bbox
    inp[r0, c0 : c1 + 1] = border
    inp[r1, c0 : c1 + 1] = border
    inp[r0 : r1 + 1, c0] = border
    inp[r0 : r1 + 1, c1] = border

    for i, color in enumerate(order):
        rr = int(legend_top + i)
        inp[rr, legend_col - 1] = border
        inp[rr, legend_col] = int(color)
        inp[rr, legend_col + 1] = border

    for color, spec, rows in zip(order, specs, fragment_rows):
        fragments = spec["fragments"]
        for row, (start, end) in zip(rows, fragments):
            inp[int(row), int(start) : int(end) + 1] = int(color)

    out = inp.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = 0
    left_out = out[:, :left_width]
    left_out[np.isin(left_out, np.array(order, dtype=int))] = 0

    for idx, (color, spec) in enumerate(zip(order, specs)):
        target_row = int(r0 + 1 + idx)
        for start, end in spec["final_intervals"]:
            out[target_row, c0 + 1 + int(start) : c0 + 1 + int(end) + 1] = int(color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    k = int(_RNG.integers(_COLOR_COUNT_RANGE[0], _COLOR_COUNT_RANGE[1] + 1))
    left_width = int(_RNG.integers(_LEFT_WIDTH_RANGE[0], _LEFT_WIDTH_RANGE[1] + 1))
    gap_x = int(_RNG.integers(_GAP_X_RANGE[0], _GAP_X_RANGE[1] + 1))
    top_margin = int(_RNG.integers(_TOP_MARGIN_RANGE[0], _TOP_MARGIN_RANGE[1] + 1))
    legend_gap = int(_RNG.integers(_LEGEND_BOX_GAP_RANGE[0], _LEGEND_BOX_GAP_RANGE[1] + 1))
    bottom_margin = int(_RNG.integers(_BOTTOM_MARGIN_RANGE[0], _BOTTOM_MARGIN_RANGE[1] + 1))
    right_margin = int(_RNG.integers(_RIGHT_MARGIN_RANGE[0], _RIGHT_MARGIN_RANGE[1] + 1))

    gapless_palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chosen = list(_RNG.choice(np.array(gapless_palette, dtype=int), size=k + 1, replace=False))
    border = int(chosen[0])
    colors = [int(v) for v in chosen[1:]]
    order = [int(v) for v in _RNG.permutation(np.array(colors, dtype=int))]

    need_overlap = bool(_RNG.integers(2))
    overlap_color = int(_RNG.integers(k)) if need_overlap else -1

    specs: list[dict[str, object]] = []
    for i in range(k):
        spec = _sample_color_spec(left_width, force_overlap=(i == overlap_color))
        if spec is None:
            return None
        specs.append(spec)

    if need_overlap and not any(bool(spec["used_overlap"]) for spec in specs):
        return None
    if not any(len(spec["final_intervals"]) > 1 for spec in specs):
        return None

    total_fragments = sum(len(spec["fragments"]) for spec in specs)

    legend_top = top_margin
    box_top = int(legend_top + k + legend_gap)
    box_left = int(left_width + gap_x)
    box_bbox = (box_top, box_left, box_top + k + 1, box_left + left_width + 1)

    min_height = max(int(box_bbox[2] + 1 + bottom_margin), total_fragments + 2)
    if min_height > 30:
        return None
    height = int(_RNG.integers(min_height, min(30, min_height + 6) + 1))
    width = int(box_bbox[3] + 1 + right_margin)
    if width > 30:
        return None

    row_pool = list(_RNG.permutation(np.arange(height, dtype=int)))
    if len(row_pool) < total_fragments:
        return None

    fragment_rows: list[list[int]] = []
    cursor = 0
    for spec in specs:
        count = len(spec["fragments"])
        rows = sorted(int(v) for v in row_pool[cursor : cursor + count])
        cursor += count
        fragment_rows.append(rows)

    legend_col = int(_RNG.integers(box_left + 1, box_left + left_width + 1))
    inp, expected = _render_case(
        height=height,
        width=width,
        left_width=left_width,
        border=border,
        box_bbox=box_bbox,
        legend_col=legend_col,
        legend_top=legend_top,
        order=order,
        specs=specs,
        fragment_rows=fragment_rows,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
