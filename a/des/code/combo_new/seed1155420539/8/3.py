from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _scan_corner_legend(
    g: np.ndarray, row: int, start_col: int, step: int, bg: int, neutral: int
) -> tuple[list[Coord], list[int]] | None:
    h, w = g.shape
    seq: list[int] = []
    coords: list[Coord] = []
    c = start_col
    while 0 <= c < w:
        value = int(g[row, c])
        if value in (bg, neutral):
            break
        seq.append(value)
        coords.append((row, c))
        c += step

    if step < 0:
        seq.reverse()
        coords.reverse()

    if not (3 <= len(seq) <= 4):
        return None
    if len(set(seq)) != len(seq):
        return None

    adj_row = row + 1 if row == 0 else row - 1
    if 0 <= adj_row < h:
        if any(int(g[adj_row, col]) != bg for _, col in coords):
            return None
    return coords, seq


def _find_legend(g: np.ndarray, bg: int, neutral: int) -> tuple[np.ndarray, list[int]]:
    h, w = g.shape
    candidates: list[tuple[list[Coord], list[int]]] = []
    probes = [
        (0, 0, 1),
        (0, w - 1, -1),
        (h - 1, 0, 1),
        (h - 1, w - 1, -1),
    ]
    for row, start_col, step in probes:
        found = _scan_corner_legend(g, row, start_col, step, bg, neutral)
        if found is not None:
            candidates.append(found)

    assert len(candidates) == 1, "expected exactly one corner legend"
    coords, seq = candidates[0]
    mask = np.zeros(g.shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask, seq


def _network_neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    h, w = mask.shape
    r, c = cell
    out: list[Coord] = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _is_intersection(g: np.ndarray, cell: Coord, neutral: int) -> bool:
    r, c = cell
    if int(g[r, c]) != neutral:
        return False
    h, w = g.shape
    grey_neighbors = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == neutral:
            grey_neighbors += 1
    return grey_neighbors >= 3


def _choose_next(prev: Coord, cur: Coord, mask: np.ndarray) -> Coord | None:
    candidates = [cell for cell in _network_neighbors(mask, cur) if cell != prev]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    dr = cur[0] - prev[0]
    dc = cur[1] - prev[1]
    straight = (cur[0] + dr, cur[1] + dc)
    if straight in candidates:
        return straight
    raise AssertionError("ambiguous continuation at intersection")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each path's source color along the neutral path, rotating at crossings according to the corner legend.

    Input structure:
    - The background is the dominant color.
    - A one-row color legend of length 3 or 4 sits in exactly one corner on the outer border.
    - Outside the legend, one neutral color forms single-pixel-wide 4-connected paths.
    - Each path has one colored endpoint and one neutral endpoint, and different paths may cross at shared neutral cells.

    Rule:
    - Read the corner legend from left to right as a cyclic color order.
    - Treat every non-background, non-legend cell as part of the path network, and start from each colored degree-1 endpoint.
    - Walk along the unique path toward the opposite endpoint; normal cells inherit the current color.
    - Whenever the walk arrives at a neutral cell with at least three neutral neighbors, rotate to the next legend color, color that crossing cell with the rotated color, and continue.
    - Leave the legend unchanged.

    Key insight:
    - The legend is a visible cyclic instruction table, and the path color only changes when the walk reaches a crossing, not at a fixed distance or by any hidden color ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common(g.ravel())
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected foreground cells"
    neutral = _most_common(non_bg)

    legend_mask, legend_seq = _find_legend(g, bg, neutral)
    ring = {legend_seq[i]: legend_seq[(i + 1) % len(legend_seq)] for i in range(len(legend_seq))}

    network_mask = (g != bg) & (~legend_mask)
    network_coords = [tuple(map(int, rc)) for rc in np.argwhere(network_mask)]
    assert network_coords, "expected a path network"

    degrees = {cell: len(_network_neighbors(network_mask, cell)) for cell in network_coords}
    endpoints = [cell for cell in network_coords if degrees[cell] == 1]
    sources = sorted(cell for cell in endpoints if int(g[cell]) != neutral)
    sinks = [cell for cell in endpoints if int(g[cell]) == neutral]

    assert sources, "expected colored path endpoints"
    assert len(sources) == len(sinks), "each path must have one colored and one neutral endpoint"
    assert all(int(g[cell]) in ring for cell in sources), "source colors must come from the legend"

    out = g.copy()
    for src in sources:
        color = int(g[src])
        next_cells = _network_neighbors(network_mask, src)
        assert len(next_cells) == 1, "source endpoint must have degree 1"
        prev = src
        cur = next_cells[0]

        while True:
            if _is_intersection(g, cur, neutral):
                color = int(ring[color])

            existing = int(out[cur])
            assert existing in (neutral, color), "conflicting propagated colors"
            out[cur] = color

            nxt = _choose_next(prev, cur, network_mask)
            if nxt is None:
                break
            prev, cur = cur, nxt

    return out


def _line_positions(count: int) -> list[int]:
    pos = [0]
    for _ in range(1, count):
        pos.append(pos[-1] + random.randint(2, 4))
    return pos


def _encounter_index(order_len: int, index: int, from_low_side: bool) -> int:
    return index + 1 if from_low_side else order_len - index


def _paint_expected_path(
    out: np.ndarray,
    start: Coord,
    end: Coord,
    step: Coord,
    start_color: int,
    crossings: set[Coord],
    ring: dict[int, int],
    neutral: int,
) -> bool:
    color = int(start_color)
    r, c = start
    while (r, c) != end:
        r += step[0]
        c += step[1]
        if (r, c) in crossings:
            color = int(ring[color])
        existing = int(out[r, c])
        if existing not in (neutral, color):
            return False
        out[r, c] = color
    return True


def _legend_cells(shape: tuple[int, int], corner: str, legend_len: int) -> list[Coord]:
    h, w = shape
    if corner == "tl":
        return [(0, c) for c in range(legend_len)]
    if corner == "tr":
        return [(0, w - legend_len + c) for c in range(legend_len)]
    if corner == "bl":
        return [(h - 1, c) for c in range(legend_len)]
    if corner == "br":
        return [(h - 1, w - legend_len + c) for c in range(legend_len)]
    raise AssertionError("unknown corner")


def _repair_corner_conflict(
    h_src_extras: list[int],
    v_src_extras: list[int],
    h_from_left: bool,
    v_from_top: bool,
) -> None:
    hi = 0 if v_from_top else len(h_src_extras) - 1
    vj = 0 if h_from_left else len(v_src_extras) - 1
    if h_src_extras[hi] == 1 and v_src_extras[vj] == 1:
        if random.choice((False, True)):
            h_src_extras[hi] = random.randint(2, 4)
        else:
            v_src_extras[vj] = random.randint(2, 4)


def _ensure_mixed_case(
    h_src_extras: list[int],
    v_src_extras: list[int],
    h_from_left: bool,
    v_from_top: bool,
) -> None:
    _repair_corner_conflict(h_src_extras, v_src_extras, h_from_left, v_from_top)
    if any(extra == 1 for extra in h_src_extras + v_src_extras):
        return

    hi = 0 if v_from_top else len(h_src_extras) - 1
    vj = 0 if h_from_left else len(v_src_extras) - 1
    choices: list[tuple[str, int]] = []
    for i in range(len(h_src_extras)):
        if not (i == hi and v_src_extras[vj] == 1):
            choices.append(("h", i))
    for j in range(len(v_src_extras)):
        if not (j == vj and h_src_extras[hi] == 1):
            choices.append(("v", j))

    kind, index = random.choice(choices)
    if kind == "h":
        h_src_extras[index] = 1
    else:
        v_src_extras[index] = 1
    _repair_corner_conflict(h_src_extras, v_src_extras, h_from_left, v_from_top)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    legend_len = random.choice((3, 4))
    colors = random.sample(range(10), legend_len + 2)
    bg = colors[0]
    neutral = colors[1]
    legend_seq = colors[2:]
    ring = {legend_seq[i]: legend_seq[(i + 1) % legend_len] for i in range(legend_len)}

    h_count = random.choice((1, 2, 2, 3))
    v_count = random.choice((1, 2, 2, 3))
    rows_rel = _line_positions(h_count)
    cols_rel = _line_positions(v_count)

    h_from_left = random.choice((False, True))
    v_from_top = random.choice((False, True))
    mode = random.choice(("all4", "mixed"))

    h_far_extras = [random.randint(1, 4) for _ in range(h_count)]
    v_far_extras = [random.randint(1, 4) for _ in range(v_count)]

    if mode == "all4":
        h_src_extras = [random.randint(2, 4) for _ in range(h_count)]
        v_src_extras = [random.randint(2, 4) for _ in range(v_count)]
    else:
        h_src_extras = [random.randint(1, 4) for _ in range(h_count)]
        v_src_extras = [random.randint(1, 4) for _ in range(v_count)]
        _ensure_mixed_case(h_src_extras, v_src_extras, h_from_left, v_from_top)

    if h_from_left:
        left_extras, right_extras = h_src_extras, h_far_extras
    else:
        left_extras, right_extras = h_far_extras, h_src_extras
    if v_from_top:
        top_extras, bottom_extras = v_src_extras, v_far_extras
    else:
        top_extras, bottom_extras = v_far_extras, v_src_extras

    top_occ = rows_rel[0] - max(top_extras)
    bottom_occ = rows_rel[-1] + max(bottom_extras)
    left_occ = cols_rel[0] - max(left_extras)
    right_occ = cols_rel[-1] + max(right_extras)

    legend_top = random.choice((False, True))
    corner = random.choice(("tl", "tr")) if legend_top else random.choice(("bl", "br"))
    pad_top = random.randint(2, 4) if legend_top else random.randint(1, 3)
    pad_bottom = random.randint(2, 4) if not legend_top else random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 3)

    height = (bottom_occ - top_occ + 1) + pad_top + pad_bottom
    width = (right_occ - left_occ + 1) + pad_left + pad_right
    if max(height, width) > 30:
        return None

    row_shift = pad_top - top_occ
    col_shift = pad_left - left_occ
    rows = [row + row_shift for row in rows_rel]
    cols = [col + col_shift for col in cols_rel]

    grid = np.full((height, width), bg, dtype=int)
    legend_coords = _legend_cells(grid.shape, corner, legend_len)
    for (r, c), color in zip(legend_coords, legend_seq, strict=False):
        grid[r, c] = int(color)

    crossings = {(row, col) for row in rows for col in cols}

    for i, row in enumerate(rows):
        c0 = cols[0] - left_extras[i]
        c1 = cols[-1] + right_extras[i]
        grid[row, c0 : c1 + 1] = neutral

    base_shift = random.randrange(legend_len)
    h_start_colors = [
        int(legend_seq[(base_shift + _encounter_index(h_count, i, v_from_top)) % legend_len])
        for i in range(h_count)
    ]
    v_start_colors = [
        int(legend_seq[(base_shift + _encounter_index(v_count, j, h_from_left)) % legend_len])
        for j in range(v_count)
    ]

    for j, col in enumerate(cols):
        r0 = rows[0] - top_extras[j]
        r1 = rows[-1] + bottom_extras[j]
        grid[r0 : r1 + 1, col] = neutral

    h_paths: list[tuple[Coord, Coord, Coord, int]] = []
    for i, row in enumerate(rows):
        left_end = (row, cols[0] - left_extras[i])
        right_end = (row, cols[-1] + right_extras[i])
        if h_from_left:
            src, dst, step = left_end, right_end, (0, 1)
        else:
            src, dst, step = right_end, left_end, (0, -1)
        grid[src] = h_start_colors[i]
        h_paths.append((src, dst, step, h_start_colors[i]))

    v_paths: list[tuple[Coord, Coord, Coord, int]] = []
    for j, col in enumerate(cols):
        top_end = (rows[0] - top_extras[j], col)
        bottom_end = (rows[-1] + bottom_extras[j], col)
        if v_from_top:
            src, dst, step = top_end, bottom_end, (1, 0)
        else:
            src, dst, step = bottom_end, top_end, (-1, 0)
        grid[src] = v_start_colors[j]
        v_paths.append((src, dst, step, v_start_colors[j]))

    if int(_most_common(grid.ravel())) != bg:
        return None
    if int(_most_common(grid[grid != bg])) != neutral:
        return None

    expected = grid.copy()
    for src, dst, step, color in h_paths + v_paths:
        if not _paint_expected_path(expected, src, dst, step, color, crossings, ring, neutral):
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Ensure the generated grid really contains the intended intersection cases.
    intersection_types = {
        sum(int(grid[nr, nc]) == neutral for nr, nc in _neighbors4(r, c))
        for r, c in crossings
    }
    if not intersection_types <= {3, 4}:
        return None
    if mode == "all4" and intersection_types != {4}:
        return None
    if mode == "mixed" and 3 not in intersection_types:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
