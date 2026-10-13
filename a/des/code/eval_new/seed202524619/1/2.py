import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_anchor_fragments(
    grid: np.ndarray, bg: int
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int, int, int]]]:
    anchors: dict[int, tuple[int, int]] = {}
    fragments: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        comps = _components(grid == color)
        if len(comps) != 2:
            continue

        singletons = [comp for comp in comps if len(comp) == 1]
        rectangles: list[tuple[int, int, int, int]] = []
        for comp in comps:
            if len(comp) == 1:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h >= 2 and w >= 2 and len(comp) == h * w:
                rectangles.append((r0, c0, r1, c1))
        if len(singletons) == 1 and len(rectangles) == 1:
            anchors[color] = tuple(map(int, singletons[0][0]))
            fragments[color] = rectangles[0]

    assert len(anchors) >= 2, "expected at least two anchor/fragment color pairs"
    return anchors, fragments


def _segment_color(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], bg: int
) -> int:
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        segment = grid[r0, lo + 1 : hi]
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        segment = grid[lo + 1 : hi, c0]
    else:
        return bg
    if segment.size == 0 or np.any(segment == bg):
        return bg
    colors = np.unique(segment)
    return int(colors[0]) if len(colors) == 1 else bg


def _offsets(sizes: list[int]) -> list[int]:
    out: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        out.append(cursor)
        cursor += int(size)
        if i + 1 < len(sizes):
            cursor += 1
    return out


def _assemble_output(
    grid: np.ndarray,
    bg: int,
    cells: dict[tuple[int, int], int],
    anchors: dict[int, tuple[int, int]],
    fragments: dict[int, tuple[int, int, int, int]],
    row_heights: list[int],
    col_widths: list[int],
) -> np.ndarray:
    n_rows = len(row_heights)
    n_cols = len(col_widths)
    row_starts = _offsets(row_heights)
    col_starts = _offsets(col_widths)
    out_h = row_starts[-1] + row_heights[-1]
    out_w = col_starts[-1] + col_widths[-1]
    out = np.full((out_h, out_w), bg, dtype=int)

    for (ri, cj), color in cells.items():
        fr0, fc0, fr1, fc1 = fragments[color]
        patch = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
        h, w = patch.shape
        r0 = row_starts[ri]
        c0 = col_starts[cj]
        out[r0 : r0 + h, c0 : c0 + w] = patch

    for ri in range(n_rows):
        for cj in range(n_cols - 1):
            left = cells.get((ri, cj))
            right = cells.get((ri, cj + 1))
            if left is None or right is None:
                continue
            seam_color = _segment_color(grid, anchors[left], anchors[right], bg)
            if seam_color == bg:
                continue
            seam_col = col_starts[cj] + col_widths[cj]
            r0 = row_starts[ri]
            out[r0 : r0 + row_heights[ri], seam_col] = seam_color

    for ri in range(n_rows - 1):
        for cj in range(n_cols):
            top = cells.get((ri, cj))
            bottom = cells.get((ri + 1, cj))
            if top is None or bottom is None:
                continue
            seam_color = _segment_color(grid, anchors[top], anchors[bottom], bg)
            if seam_color == bg:
                continue
            seam_row = row_starts[ri] + row_heights[ri]
            c0 = col_starts[cj]
            out[seam_row, c0 : c0 + col_widths[cj]] = seam_color

    for ri in range(n_rows - 1):
        seam_row = row_starts[ri] + row_heights[ri]
        for cj in range(n_cols - 1):
            seam_col = col_starts[cj] + col_widths[cj]
            out[seam_row, seam_col] = bg

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered same-color rectangles by reading anchor positions and seam colors from the small template.

    Input structure:
    - The background is black.
    - A small template region contains single-pixel anchor colors arranged on a grid, with optional horizontal or vertical colored line segments between adjacent anchors.
    - Elsewhere, each anchor color appears again as one scattered solid rectangular fragment, with one fragment per anchor color.

    Rule:
    - Detect the colors that appear exactly as one single-pixel anchor plus one solid rectangle fragment.
    - Use the anchor coordinates to recover the template rows and columns, then place the matching fragments into the same row/column arrangement with a one-pixel seam between neighboring cells.
    - For each pair of adjacent anchors, fill the corresponding seam with the template line color if a direct uniform line segment connects them; otherwise keep that seam as background.

    Key insight:
    The template encodes two layers at once: anchor positions define the fragment layout, and the colors of direct connecting segments define the seam fills between already-matched fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    anchors, fragments = _extract_anchor_fragments(g, bg)

    row_values = sorted({int(r) for r, _ in anchors.values()})
    col_values = sorted({int(c) for _, c in anchors.values()})
    row_index = {value: i for i, value in enumerate(row_values)}
    col_index = {value: i for i, value in enumerate(col_values)}

    cells: dict[tuple[int, int], int] = {}
    row_heights = [0] * len(row_values)
    col_widths = [0] * len(col_values)
    for color, (r, c) in anchors.items():
        ri = row_index[int(r)]
        cj = col_index[int(c)]
        cells[(ri, cj)] = int(color)
        fr0, fc0, fr1, fc1 = fragments[color]
        row_heights[ri] = max(row_heights[ri], fr1 - fr0 + 1)
        col_widths[cj] = max(col_widths[cj], fc1 - fc0 + 1)

    return _assemble_output(g, bg, cells, anchors, fragments, row_heights, col_widths)


_BG = 0
_GEN_RETRIES = 300
_PLACE_RETRIES = 300
_ROWCOL_GAPS = (2, 3)
_TEMPLATE_ORIENTATIONS = ("left", "top")
_LAYOUT_MODES = ("row", "col", "grid")
_GRID_CASES = ((2, 2), (2, 3), (3, 2))
_PIECE_SIZE_RANGE = (2, 5)
_COLOR_POOL = tuple(range(1, 10))


def _sample_layout_shape() -> tuple[int, int]:
    mode = random.choice(_LAYOUT_MODES)
    if mode == "row":
        return 1, random.randint(2, 4)
    if mode == "col":
        return random.randint(2, 4), 1
    return random.choice(_GRID_CASES)


def _row_col_positions(count: int) -> list[int]:
    positions = [0]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.choice(_ROWCOL_GAPS))
    return positions


def _sample_seams(
    n_rows: int, n_cols: int, line_palette: list[int]
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    edge_keys = [("h", r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    edge_keys += [("v", r, c) for r in range(n_rows - 1) for c in range(n_cols)]

    if len(edge_keys) == 1:
        connected = [random.choice([False, True])]
    else:
        connected = [random.choice([False, True]) for _ in edge_keys]
        if all(connected) or not any(connected):
            flip = random.randrange(len(edge_keys))
            connected[flip] = not connected[flip]

    h_seams: dict[tuple[int, int], int] = {}
    v_seams: dict[tuple[int, int], int] = {}
    for key, is_connected in zip(edge_keys, connected):
        if not is_connected:
            continue
        seam_color = int(random.choice(line_palette))
        kind, r, c = key
        if kind == "h":
            h_seams[(r, c)] = seam_color
        else:
            v_seams[(r, c)] = seam_color
    return h_seams, v_seams


def _build_template(
    row_positions: list[int],
    col_positions: list[int],
    cell_colors: dict[tuple[int, int], int],
    h_seams: dict[tuple[int, int], int],
    v_seams: dict[tuple[int, int], int],
) -> np.ndarray:
    h = row_positions[-1] + 1
    w = col_positions[-1] + 1
    template = np.full((h, w), _BG, dtype=int)

    for (ri, cj), color in cell_colors.items():
        template[row_positions[ri], col_positions[cj]] = int(color)

    for (ri, cj), color in h_seams.items():
        r = row_positions[ri]
        c0 = col_positions[cj]
        c1 = col_positions[cj + 1]
        template[r, c0 + 1 : c1] = int(color)

    for (ri, cj), color in v_seams.items():
        c = col_positions[cj]
        r0 = row_positions[ri]
        r1 = row_positions[ri + 1]
        template[r0 + 1 : r1, c] = int(color)

    return template


def _expected_output(
    row_heights: list[int],
    col_widths: list[int],
    cell_colors: dict[tuple[int, int], int],
    h_seams: dict[tuple[int, int], int],
    v_seams: dict[tuple[int, int], int],
) -> np.ndarray:
    row_starts = _offsets(row_heights)
    col_starts = _offsets(col_widths)
    out_h = row_starts[-1] + row_heights[-1]
    out_w = col_starts[-1] + col_widths[-1]
    out = np.full((out_h, out_w), _BG, dtype=int)

    for (ri, cj), color in cell_colors.items():
        r0 = row_starts[ri]
        c0 = col_starts[cj]
        out[r0 : r0 + row_heights[ri], c0 : c0 + col_widths[cj]] = int(color)

    for (ri, cj), color in h_seams.items():
        seam_col = col_starts[cj] + col_widths[cj]
        r0 = row_starts[ri]
        out[r0 : r0 + row_heights[ri], seam_col] = int(color)

    for (ri, cj), color in v_seams.items():
        seam_row = row_starts[ri] + row_heights[ri]
        c0 = col_starts[cj]
        out[seam_row, c0 : c0 + col_widths[cj]] = int(color)

    for ri in range(len(row_heights) - 1):
        seam_row = row_starts[ri] + row_heights[ri]
        for cj in range(len(col_widths) - 1):
            seam_col = col_starts[cj] + col_widths[cj]
            out[seam_row, seam_col] = _BG

    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_grid, r0 + h + 1)
    cc1 = min(w_grid, c0 + w + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_fragments(
    grid: np.ndarray,
    occupied: np.ndarray,
    fragment_sizes: dict[int, tuple[int, int]],
    orientation: str,
    template_bbox: tuple[int, int, int, int],
) -> bool:
    h_grid, w_grid = grid.shape
    tr0, tc0, tr1, tc1 = template_bbox
    if orientation == "left":
        row_min = 1
        col_min = tc1 + 3
    else:
        row_min = tr1 + 3
        col_min = 1

    items = list(fragment_sizes.items())
    random.shuffle(items)
    items.sort(key=lambda item: -(item[1][0] * item[1][1]))

    for color, (h, w) in items:
        row_max = h_grid - h - 1
        col_max = w_grid - w - 1
        if row_min > row_max or col_min > col_max:
            return False
        for _ in range(_PLACE_RETRIES):
            r0 = random.randint(row_min, row_max)
            c0 = random.randint(col_min, col_max)
            if not _can_place(occupied, r0, c0, h, w):
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = int(color)
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            break
        else:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = _sample_layout_shape()
    row_heights = [
        random.randint(_PIECE_SIZE_RANGE[0], _PIECE_SIZE_RANGE[1]) for _ in range(n_rows)
    ]
    col_widths = [
        random.randint(_PIECE_SIZE_RANGE[0], _PIECE_SIZE_RANGE[1]) for _ in range(n_cols)
    ]

    piece_count = n_rows * n_cols
    anchor_colors = random.sample(list(_COLOR_POOL), piece_count)
    remaining = [c for c in _COLOR_POOL if c not in anchor_colors]
    palette_size = random.randint(1, min(3, len(remaining)))
    line_palette = random.sample(remaining, palette_size)

    row_positions = _row_col_positions(n_rows)
    col_positions = _row_col_positions(n_cols)

    cell_colors: dict[tuple[int, int], int] = {}
    shuffled_colors = anchor_colors[:]
    random.shuffle(shuffled_colors)
    k = 0
    for ri in range(n_rows):
        for cj in range(n_cols):
            cell_colors[(ri, cj)] = int(shuffled_colors[k])
            k += 1

    h_seams, v_seams = _sample_seams(n_rows, n_cols, line_palette)
    template = _build_template(row_positions, col_positions, cell_colors, h_seams, v_seams)
    th, tw = template.shape

    orientation = random.choice(_TEMPLATE_ORIENTATIONS)
    if orientation == "left":
        min_h = max(16, th + 2)
        min_w = max(18, tw + 12)
    else:
        min_h = max(18, th + 12)
        min_w = max(16, tw + 2)

    if min_h > 30 or min_w > 30:
        return None
    h_grid = random.randint(min_h, 30)
    w_grid = random.randint(min_w, 30)
    grid = np.full((h_grid, w_grid), _BG, dtype=int)

    if orientation == "left":
        tr0 = random.randint(1, h_grid - th - 1)
        tc0 = random.randint(1, 2)
    else:
        tr0 = random.randint(1, 2)
        tc0 = random.randint(1, w_grid - tw - 1)

    grid[tr0 : tr0 + th, tc0 : tc0 + tw] = template
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[tr0 : tr0 + th, tc0 : tc0 + tw] = template != _BG

    fragment_sizes = {
        color: (row_heights[ri], col_widths[cj]) for (ri, cj), color in cell_colors.items()
    }
    if not _place_fragments(
        grid, occupied, fragment_sizes, orientation, (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)
    ):
        return None

    expected = _expected_output(row_heights, col_widths, cell_colors, h_seams, v_seams)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-fragment assembly input")
