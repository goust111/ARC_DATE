import numpy as np


BG = 0
ORANGE = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray, bg: int, orange: int) -> tuple[list[tuple[int, int]], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for col in (0, w - 1):
        rows: list[int] = []
        colors: list[int] = []
        ok = True
        for r in range(h):
            v = int(grid[r, col])
            if v in (bg, orange):
                continue
            if any(int(grid[nr, nc]) != bg for nr, nc in _neighbors4(r, col, h, w)):
                ok = False
                break
            rows.append(r)
            colors.append(v)
        if not ok or len(rows) < 2:
            continue
        if any(rows[i + 1] - rows[i] != 2 for i in range(len(rows) - 1)):
            continue
        candidates.append((col, rows, colors))
    assert candidates, "failed to detect edge legend"
    col, rows, colors = max(candidates, key=lambda x: (len(x[1]), -abs(x[0])))
    return [(r, col) for r in rows], colors


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    legend_cells, legend_colors = _find_legend(g, bg, ORANGE)

    mask = (g != bg) & (g != ORANGE)
    for r, c in legend_cells:
        mask[r, c] = False

    segments = _components(mask)
    assert segments, "expected at least one path segment"

    unique_lengths = sorted({len(seg) for seg in segments})
    assert len(unique_lengths) <= len(legend_colors), "legend too short for segment lengths"
    length_to_color = {length: int(legend_colors[i]) for i, length in enumerate(unique_lengths)}

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg
    for seg in segments:
        new_color = length_to_color[len(seg)]
        for r, c in seg:
            out[r, c] = new_color
    return out


def _assign_lengths(rng: np.random.Generator, unique_lengths: list[int], slot_count: int) -> list[int]:
    values = list(unique_lengths)
    while len(values) < slot_count:
        values.append(int(rng.choice(unique_lengths)))
    rng.shuffle(values)
    return values


def _build_horizontal_case(
    rng: np.random.Generator,
    legend_side: str,
    legend_colors: list[int],
    path_colors: list[int],
    assigned_lengths: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cross_count = len(path_colors) - 1
    spine_lengths = assigned_lengths[: cross_count + 1]
    up_lengths = assigned_lengths[cross_count + 1 :: 2]
    down_lengths = assigned_lengths[cross_count + 2 :: 2]

    path_w = sum(spine_lengths) + cross_count
    path_h = max(up_lengths) + 1 + max(down_lengths)
    legend_h = 2 * len(legend_colors) - 1

    h = max(path_h, legend_h) + 3 + int(rng.integers(0, 4))
    w = path_w + 4 + int(rng.integers(0, 5))
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    legend_col = 0 if legend_side == "left" else w - 1
    legend_start = int(rng.integers(0, h - legend_h + 1))
    legend_cells: list[tuple[int, int]] = []
    for i, color in enumerate(legend_colors):
        rr = legend_start + 2 * i
        grid[rr, legend_col] = int(color)
        legend_cells.append((rr, legend_col))

    min_left = 2 if legend_side == "left" else 1
    max_left = w - path_w - (2 if legend_side == "right" else 1)
    left = int(rng.integers(min_left, max_left + 1))
    top = int(rng.integers(1, h - path_h))
    spine_row = top + max(up_lengths)

    cross_cols: list[int] = []
    col = left + spine_lengths[0]
    cross_cols.append(col)
    for middle_len in spine_lengths[1:-1]:
        col += middle_len + 1
        cross_cols.append(col)

    right = cross_cols[-1] + spine_lengths[-1]
    for cc in range(left, right + 1):
        grid[spine_row, cc] = int(path_colors[0])

    segment_cells: list[list[tuple[int, int]]] = []
    segment_lengths: list[int] = []

    left_seg = [(spine_row, cc) for cc in range(left, cross_cols[0])]
    segment_cells.append(left_seg)
    segment_lengths.append(len(left_seg))

    for i in range(cross_count - 1):
        mid_seg = [(spine_row, cc) for cc in range(cross_cols[i] + 1, cross_cols[i + 1])]
        segment_cells.append(mid_seg)
        segment_lengths.append(len(mid_seg))

    right_seg = [(spine_row, cc) for cc in range(cross_cols[-1] + 1, right + 1)]
    segment_cells.append(right_seg)
    segment_lengths.append(len(right_seg))

    for i, cross_col in enumerate(cross_cols):
        top_r = spine_row - up_lengths[i]
        bot_r = spine_row + down_lengths[i]
        for rr in range(top_r, bot_r + 1):
            grid[rr, cross_col] = int(path_colors[i + 1])

        up_seg = [(rr, cross_col) for rr in range(top_r, spine_row)]
        down_seg = [(rr, cross_col) for rr in range(spine_row + 1, bot_r + 1)]
        segment_cells.append(up_seg)
        segment_lengths.append(len(up_seg))
        segment_cells.append(down_seg)
        segment_lengths.append(len(down_seg))

    for cross_col in cross_cols:
        grid[spine_row, cross_col] = ORANGE
        expected[spine_row, cross_col] = ORANGE

    length_to_color = {
        length: int(legend_colors[i])
        for i, length in enumerate(sorted(set(segment_lengths)))
    }
    assert len(length_to_color) == len(legend_colors)

    for cells, length in zip(segment_cells, segment_lengths):
        color = length_to_color[length]
        for rr, cc in cells:
            expected[rr, cc] = color

    assert all(grid[r, c] != BG for r, c in legend_cells)
    return grid, expected


def _build_vertical_case(
    rng: np.random.Generator,
    legend_side: str,
    legend_colors: list[int],
    path_colors: list[int],
    assigned_lengths: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cross_count = len(path_colors) - 1
    spine_lengths = assigned_lengths[: cross_count + 1]
    left_lengths = assigned_lengths[cross_count + 1 :: 2]
    right_lengths = assigned_lengths[cross_count + 2 :: 2]

    path_h = sum(spine_lengths) + cross_count
    path_w = max(left_lengths) + 1 + max(right_lengths)
    legend_h = 2 * len(legend_colors) - 1

    h = max(path_h, legend_h) + 3 + int(rng.integers(0, 4))
    w = path_w + 4 + int(rng.integers(0, 5))
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    legend_col = 0 if legend_side == "left" else w - 1
    legend_start = int(rng.integers(0, h - legend_h + 1))
    legend_cells: list[tuple[int, int]] = []
    for i, color in enumerate(legend_colors):
        rr = legend_start + 2 * i
        grid[rr, legend_col] = int(color)
        legend_cells.append((rr, legend_col))

    top = int(rng.integers(1, h - path_h))
    min_left = 2 if legend_side == "left" else 1
    max_left = w - path_w - (2 if legend_side == "right" else 1)
    left = int(rng.integers(min_left, max_left + 1))
    spine_col = left + max(left_lengths)

    cross_rows: list[int] = []
    row = top + spine_lengths[0]
    cross_rows.append(row)
    for middle_len in spine_lengths[1:-1]:
        row += middle_len + 1
        cross_rows.append(row)

    bottom = cross_rows[-1] + spine_lengths[-1]
    for rr in range(top, bottom + 1):
        grid[rr, spine_col] = int(path_colors[0])

    segment_cells: list[list[tuple[int, int]]] = []
    segment_lengths: list[int] = []

    top_seg = [(rr, spine_col) for rr in range(top, cross_rows[0])]
    segment_cells.append(top_seg)
    segment_lengths.append(len(top_seg))

    for i in range(cross_count - 1):
        mid_seg = [(rr, spine_col) for rr in range(cross_rows[i] + 1, cross_rows[i + 1])]
        segment_cells.append(mid_seg)
        segment_lengths.append(len(mid_seg))

    bottom_seg = [(rr, spine_col) for rr in range(cross_rows[-1] + 1, bottom + 1)]
    segment_cells.append(bottom_seg)
    segment_lengths.append(len(bottom_seg))

    for i, cross_row in enumerate(cross_rows):
        left_c = spine_col - left_lengths[i]
        right_c = spine_col + right_lengths[i]
        for cc in range(left_c, right_c + 1):
            grid[cross_row, cc] = int(path_colors[i + 1])

        left_seg = [(cross_row, cc) for cc in range(left_c, spine_col)]
        right_seg = [(cross_row, cc) for cc in range(spine_col + 1, right_c + 1)]
        segment_cells.append(left_seg)
        segment_lengths.append(len(left_seg))
        segment_cells.append(right_seg)
        segment_lengths.append(len(right_seg))

    for cross_row in cross_rows:
        grid[cross_row, spine_col] = ORANGE
        expected[cross_row, spine_col] = ORANGE

    length_to_color = {
        length: int(legend_colors[i])
        for i, length in enumerate(sorted(set(segment_lengths)))
    }
    assert len(length_to_color) == len(legend_colors)

    for cells, length in zip(segment_cells, segment_lengths):
        color = length_to_color[length]
        for rr, cc in cells:
            expected[rr, cc] = color

    assert all(grid[r, c] != BG for r, c in legend_cells)
    return grid, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cross_count = int(rng.integers(2, 4))
    legend_count = int(rng.integers(3, 5))
    if legend_count + cross_count + 1 > len(PALETTE):
        return None

    unique_lengths = sorted(int(v) for v in rng.choice(np.arange(1, 6), size=legend_count, replace=False))
    slot_count = 3 * cross_count + 1
    assigned_lengths = _assign_lengths(rng, unique_lengths, slot_count)

    colors = list(PALETTE)
    rng.shuffle(colors)
    legend_colors = colors[:legend_count]
    path_colors = colors[legend_count : legend_count + cross_count + 1]
    if len(set(path_colors)) != cross_count + 1:
        return None

    legend_side = "left" if rng.random() < 0.5 else "right"
    if rng.random() < 0.5:
        grid, expected = _build_horizontal_case(
            rng, legend_side, legend_colors, path_colors, assigned_lengths
        )
    else:
        grid, expected = _build_vertical_case(
            rng, legend_side, legend_colors, path_colors, assigned_lengths
        )

    legend_cells, detected_colors = _find_legend(grid, BG, ORANGE)
    if detected_colors != legend_colors:
        return None

    mask = (grid != BG) & (grid != ORANGE)
    for r, c in legend_cells:
        mask[r, c] = False
    seg_lengths = sorted(len(comp) for comp in _components(mask))
    if sorted(set(seg_lengths)) != unique_lengths:
        return None
    if len(seg_lengths) != slot_count:
        return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
