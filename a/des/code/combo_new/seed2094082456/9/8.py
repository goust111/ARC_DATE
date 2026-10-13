from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIGIT_PATTERNS = {
    2: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}

_LEGEND_SIZE = 7
_MARGIN_WITH_LEGEND = _LEGEND_SIZE + 2
_SMALL_COUNTS = {2: (2,), 3: (2, 3)}
_LARGE_COUNTS = {2: (3, 4, 5), 3: (4, 5)}
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _count_components(mask: np.ndarray) -> int:
    return len(_component_coords(mask))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h != w:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _component_coords(grid == color):
            if not _is_rectangular_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if not np.any((interior != bg) & (interior != color)):
                continue
            corner_dist = min(
                r0 + c0,
                r0 + (w - 1 - c1),
                (h - 1 - r1) + c0,
                (h - 1 - r1) + (w - 1 - c1),
            )
            candidates.append((corner_dist, (r1 - r0 + 1) * (c1 - c0 + 1), color, (r0, c0, r1, c1)))
    assert candidates, "legend frame not found"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _corner_dist, _area, grey, bbox = candidates[0]
    return int(grey), bbox


def _read_threshold(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int, grey: int) -> int:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    ink = (interior != bg) & (interior != grey)
    coords = np.argwhere(ink)
    assert coords.size > 0, "legend digit not found"
    rr0, cc0, rr1, cc1 = _bbox(coords)
    glyph = ink[rr0 : rr1 + 1, cc0 : cc1 + 1].astype(int)
    for threshold, pattern in _DIGIT_PATTERNS.items():
        if glyph.shape == pattern.shape and np.array_equal(glyph, pattern):
            return int(threshold)
    raise AssertionError("unrecognized legend digit")


def _largest_arithmetic_run(indices: list[int]) -> list[int]:
    if len(indices) <= 2:
        return indices[:]
    index_set = set(indices)
    best: list[int] = []
    for i, start in enumerate(indices):
        for next_index in indices[i + 1 :]:
            step = next_index - start
            run = [start]
            value = start + step
            while value in index_set:
                run.append(value)
                value += step
            if len(run) > len(best):
                best = run
    return best if best else indices[:]


def _detect_grid_lines(
    grid: np.ndarray,
    grey: int,
    legend_bbox: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    grey_mask = grid == grey
    r0, c0, r1, c1 = legend_bbox
    grey_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    row_counts = grey_mask.sum(axis=1)
    col_counts = grey_mask.sum(axis=0)
    row_threshold = max(2, int(row_counts.max()) // 2 + 1)
    col_threshold = max(2, int(col_counts.max()) // 2 + 1)

    row_candidates = [int(r) for r, count in enumerate(row_counts.tolist()) if count >= row_threshold]
    col_candidates = [int(c) for c, count in enumerate(col_counts.tolist()) if count >= col_threshold]
    row_lines = _largest_arithmetic_run(row_candidates)
    col_lines = _largest_arithmetic_run(col_candidates)

    assert len(row_lines) >= 3 and len(col_lines) >= 3, "grid lines not found"
    if len(row_lines) >= 2:
        row_step = row_lines[1] - row_lines[0]
        assert row_step > 1 and all(b - a == row_step for a, b in zip(row_lines, row_lines[1:]))
    if len(col_lines) >= 2:
        col_step = col_lines[1] - col_lines[0]
        assert col_step > 1 and all(b - a == col_step for a, b in zip(col_lines, col_lines[1:]))
    return row_lines, col_lines


def _cell_spans(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for left, right in zip(lines, lines[1:]):
        spans.append((int(left + 1), int(right - 1)))
    assert spans and all(a <= b for a, b in spans)
    return spans


def _scene_without_guides(
    grid: np.ndarray,
    bg: int,
    grey: int,
    legend_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    scene = np.array(grid, dtype=int)
    r0, c0, r1, c1 = legend_bbox
    scene[r0 : r1 + 1, c0 : c1 + 1] = bg
    scene[scene == grey] = bg
    return scene


def _occupied_cells_by_color(
    scene: np.ndarray,
    bg: int,
    row_lines: list[int],
    col_lines: list[int],
) -> dict[int, set[tuple[int, int]]]:
    row_spans = _cell_spans(row_lines)
    col_spans = _cell_spans(col_lines)
    occupied: dict[int, set[tuple[int, int]]] = {}
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            sub = scene[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(v) for v in np.unique(sub) if int(v) != bg]
            assert len(colors) <= 1, "cell interior contains multiple colors"
            if colors:
                occupied.setdefault(colors[0], set()).add((i, j))
    return occupied


def _render_object(
    out: np.ndarray,
    color: int,
    cells: set[tuple[int, int]],
    row_lines: list[int],
    col_lines: list[int],
) -> None:
    row_spans = _cell_spans(row_lines)
    col_spans = _cell_spans(col_lines)
    for i, j in cells:
        r0, r1 = row_spans[i]
        c0, c1 = col_spans[j]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        if (i, j + 1) in cells:
            bridge_col = col_lines[j + 1]
            out[r0 : r1 + 1, bridge_col] = color
        if (i + 1, j) in cells:
            bridge_row = row_lines[i + 1]
            out[bridge_row, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only objects whose gridline-cut fragment count exceeds the legend threshold.

    Input structure:
    - A background canvas contains one dashed grey rectangular grid, with horizontal and vertical dashed lines at uniform spacing.
    - Several different-colored 4-connected solid objects occupy multiple grid cells and are visibly split where the grey grid lines overwrite them.
    - In one corner, a separate grey square frame contains a small digit glyph that encodes the threshold value 2 or 3.

    Rule:
    - Find the grey legend frame and read its interior digit as the threshold T.
    - Ignore the legend and treat grey gridline pixels as erased cuts; for each object color, count the remaining 4-connected fragments.
    - If an object's fragment count is strictly greater than T, keep that object and restore its full cross-cell shape by filling back the erased gridline bridges between its occupied cells.
    - If an object's fragment count is less than or equal to T, delete it entirely; remove all grey grid lines and the legend from the output.

    Key insight:
    - The grey dashed grid is an active cutting mechanism rather than decoration, and the corner legend gives the comparison threshold that decides which cut-apart objects survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey, legend_bbox = _find_legend(g, bg)
    threshold = _read_threshold(g, legend_bbox, bg, grey)
    row_lines, col_lines = _detect_grid_lines(g, grey, legend_bbox)
    scene = _scene_without_guides(g, bg, grey, legend_bbox)
    occupied = _occupied_cells_by_color(scene, bg, row_lines, col_lines)

    out = np.full_like(g, bg)
    for color, cells in occupied.items():
        fragments = _count_components(scene == color)
        if fragments > threshold:
            _render_object(out, int(color), cells, row_lines, col_lines)
    return out


def _build_legend(bg: int, grey: int, digit_color: int, threshold: int) -> np.ndarray:
    legend = np.full((_LEGEND_SIZE, _LEGEND_SIZE), bg, dtype=int)
    legend[0, :] = grey
    legend[-1, :] = grey
    legend[:, 0] = grey
    legend[:, -1] = grey
    pattern = _DIGIT_PATTERNS[int(threshold)]
    legend[1:6, 2:5][pattern.astype(bool)] = digit_color
    return legend


def _choose_field_axis() -> tuple[int, int]:
    choices = []
    for cells in (4, 5):
        for cell_size in (3, 4):
            field_size = cells * cell_size + (cells + 1)
            if field_size + _MARGIN_WITH_LEGEND <= 30:
                choices.append((cells, cell_size))
    return random.choice(choices)


def _corner_layout(field_h: int, field_w: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "top_left":
        return _MARGIN_WITH_LEGEND, _MARGIN_WITH_LEGEND, field_h + _MARGIN_WITH_LEGEND, field_w + _MARGIN_WITH_LEGEND
    if corner == "top_right":
        return _MARGIN_WITH_LEGEND, 1, field_h + _MARGIN_WITH_LEGEND, field_w + _MARGIN_WITH_LEGEND
    if corner == "bottom_left":
        return 1, _MARGIN_WITH_LEGEND, field_h + _MARGIN_WITH_LEGEND, field_w + _MARGIN_WITH_LEGEND
    if corner == "bottom_right":
        return 1, 1, field_h + _MARGIN_WITH_LEGEND, field_w + _MARGIN_WITH_LEGEND
    raise AssertionError(f"unknown corner {corner}")


def _place_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    if corner == "top_left":
        r0, c0 = 0, 0
    elif corner == "top_right":
        r0, c0 = 0, w - _LEGEND_SIZE
    elif corner == "bottom_left":
        r0, c0 = h - _LEGEND_SIZE, 0
    else:
        r0, c0 = h - _LEGEND_SIZE, w - _LEGEND_SIZE
    grid[r0 : r0 + _LEGEND_SIZE, c0 : c0 + _LEGEND_SIZE] = legend


def _grid_lines(start: int, cell_size: int, cell_count: int) -> list[int]:
    return [int(start + i * (cell_size + 1)) for i in range(cell_count + 1)]


def _draw_dashed_grid(
    grid: np.ndarray,
    bg: int,
    grey: int,
    row_lines: list[int],
    col_lines: list[int],
    row_phase: int,
    col_phase: int,
) -> None:
    c0, c1 = col_lines[0], col_lines[-1]
    r0, r1 = row_lines[0], row_lines[-1]
    for r in row_lines:
        grid[r, c0 : c1 + 1] = bg
    for c in col_lines:
        grid[r0 : r1 + 1, c] = bg
    for r in row_lines:
        for c in range(c0, c1 + 1):
            if (c - c0 + row_phase) % 2 == 0:
                grid[r, c] = grey
    for c in col_lines:
        for r in range(r0, r1 + 1):
            if (r - r0 + col_phase) % 2 == 0:
                grid[r, c] = grey


def _balanced_branch_flags(object_count: int) -> list[bool]:
    flags = [False, True]
    while len(flags) < object_count:
        flags.append(len([v for v in flags if v]) < len(flags) / 2 or random.random() < 0.5)
    random.shuffle(flags)
    return flags


def _choose_fragment_counts(threshold: int, object_count: int) -> list[int]:
    flags = _balanced_branch_flags(object_count)
    counts = []
    for keep in flags:
        pool = _LARGE_COUNTS[threshold] if keep else _SMALL_COUNTS[threshold]
        counts.append(int(random.choice(pool)))
    return counts


def _free_cells(cell_rows: int, cell_cols: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(cell_rows)
        for c in range(cell_cols)
        if (r, c) not in blocked
    ]


def _grow_polyomino(
    cell_rows: int,
    cell_cols: int,
    size: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    starts = _free_cells(cell_rows, cell_cols, blocked)
    random.shuffle(starts)
    for start in starts:
        cells = {start}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, cell_rows, cell_cols):
                    if (nr, nc) not in cells and (nr, nc) not in blocked:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == size:
            return cells
    return None


def _expand_blocked(cells: set[tuple[int, int]], cell_rows: int, cell_cols: int) -> set[tuple[int, int]]:
    return set(cells)


def _place_objects(
    cell_rows: int,
    cell_cols: int,
    counts: list[int],
) -> list[set[tuple[int, int]]] | None:
    order = sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True)
    shapes: list[set[tuple[int, int]] | None] = [None] * len(counts)
    blocked: set[tuple[int, int]] = set()
    for idx in order:
        shape = _grow_polyomino(cell_rows, cell_cols, counts[idx], blocked)
        if shape is None:
            return None
        shapes[idx] = shape
        blocked |= _expand_blocked(shape, cell_rows, cell_cols)
    return [shape for shape in shapes if shape is not None]


def _render_expected(
    shape_specs: list[tuple[int, set[tuple[int, int]], bool]],
    bg: int,
    out_shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    out = np.full(out_shape, bg, dtype=int)
    for color, cells, keep in shape_specs:
        if keep:
            _render_object(out, color, cells, row_lines, col_lines)
    return out


def _render_input_objects(
    shape_specs: list[tuple[int, set[tuple[int, int]], bool]],
    bg: int,
    out_shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    inp = np.full(out_shape, bg, dtype=int)
    for color, cells, _keep in shape_specs:
        _render_object(inp, color, cells, row_lines, col_lines)
    return inp


def _gen_attempt() -> np.ndarray | None:
    threshold = random.choice((2, 3))
    cell_rows, cell_h = _choose_field_axis()
    cell_cols, cell_w = _choose_field_axis()
    field_h = cell_rows * cell_h + (cell_rows + 1)
    field_w = cell_cols * cell_w + (cell_cols + 1)
    corner = random.choice(_CORNERS)
    field_r0, field_c0, height, width = _corner_layout(field_h, field_w, corner)

    colors = list(range(10))
    random.shuffle(colors)
    bg = colors.pop()
    grey = colors.pop()
    digit_color = colors.pop()
    object_count = random.randint(2, min(4, len(colors)))
    counts = _choose_fragment_counts(threshold, object_count)
    shapes = _place_objects(cell_rows, cell_cols, counts)
    if shapes is None:
        return None

    keep_flags = [count > threshold for count in counts]
    object_colors = colors[:object_count]
    shape_specs = [
        (int(color), cells, bool(keep))
        for color, cells, keep in zip(object_colors, shapes, keep_flags, strict=False)
    ]

    row_lines = _grid_lines(field_r0, cell_h, cell_rows)
    col_lines = _grid_lines(field_c0, cell_w, cell_cols)
    out = _render_expected(shape_specs, bg, (height, width), row_lines, col_lines)

    inp = _render_input_objects(shape_specs, bg, (height, width), row_lines, col_lines)
    _draw_dashed_grid(
        inp,
        bg,
        grey,
        row_lines,
        col_lines,
        row_phase=random.randint(0, 1),
        col_phase=random.randint(0, 1),
    )
    legend = _build_legend(bg, grey, digit_color, threshold)
    _place_legend(inp, legend, corner)

    # Reject inputs that accidentally lose one of the two threshold branches.
    counts_present = Counter(count > threshold for count in counts)
    if counts_present[True] == 0 or counts_present[False] == 0:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
