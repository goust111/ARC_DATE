from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, draw_line, find_components, get_bg


def _line_cells(r: int, c: int, end_r: int, end_c: int) -> list[tuple[int, int]]:
    assert r == end_r or c == end_c, "marker path must be horizontal or vertical"
    if r == end_r:
        step = 1 if end_c >= c else -1
        return [(r, cc) for cc in range(c, end_c + step, step)]

    step = 1 if end_r >= r else -1
    return [(rr, c) for rr in range(r, end_r + step, step)]


def _anchor_adjacent_endpoint(
    r: int, c: int, anchor_bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = anchor_bbox
    if r0 <= r <= r1:
        if c < c0:
            return r, c0 - 1
        if c > c1:
            return r, c1 + 1
    if c0 <= c <= c1:
        if r < r0:
            return r0 - 1, c
        if r > r1:
            return r1 + 1, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every colored marker that is row- or column-aligned with the 2x2 anchor square until it touches that square.

    Input structure
    - The background contains one non-background 2x2 square anchor.
    - All other foreground objects are single colored marker pixels.
    - Some marker pixels share one row or one column with the anchor square.

    Rule
    - Keep the anchor and all original marker pixels unchanged.
    - For each marker on an anchor row, paint horizontally from the marker to the cell next to the anchor.
    - For each marker on an anchor column, paint vertically from the marker to the cell next to the anchor.
    - Ignore markers that are not aligned with any anchor row or column.

    Key insight
    - The 2x2 square supplies the only two active rows and two active columns; aligned singleton markers emit straight segments into the nearest side of the square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    anchors = [
        comp
        for comp in components
        if comp["area"] == 4 and bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2
    ]
    assert len(anchors) == 1, "expected exactly one 2x2 foreground anchor"

    anchor = anchors[0]
    out = g.copy()
    painted_any = False
    for comp in components:
        if comp is anchor:
            continue

        assert comp["area"] == 1, "expected every non-anchor object to be one marker pixel"
        r, c = map(int, comp["coords"][0])
        endpoint = _anchor_adjacent_endpoint(r, c, anchor["bbox"])
        if endpoint is None:
            continue

        for rr, cc in _line_cells(r, c, *endpoint)[1:]:
            assert int(g[rr, cc]) == bg, "expected a clear background path to the anchor"
        draw_line(out, r, c, end_r=endpoint[0], end_c=endpoint[1], color=int(comp["color"]))
        painted_any = True

    assert painted_any, "expected at least one marker aligned to the anchor"
    return out

# --------------- generator (o5x) ---------------

# Canvas and anchor geometry.
GRID_SIZE = 10
BG_COLOR = 0
ANCHOR_COLOR = 3
ANCHOR_SIZE = 2
ANCHOR_ROW_RANGE = (2, 7)
ANCHOR_COL_RANGE = (2, 6)

# Marker counts and palette choices.
TOTAL_MARKER_RANGE = (4, 11)
MAX_ALIGNED_MARKERS = 4
MAX_UNALIGNED_MARKERS = 6
MARKER_COLOR_COUNT_RANGE = (1, 3)
MARKER_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)

# Line layout and retry limits.
SIDE_NAMES = ["top", "bottom", "left", "right"]
LINE_LENGTH_RANGE = (2, 6)
GEN_RETRY_LIMIT = 300


def _gen_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and int(grid[nr, nc]) != BG_COLOR:
            return False
    return True


def _gen_place_unaligned(
    inp_seed: np.ndarray,
    out: np.ndarray,
    color: int,
    anchor_rows: set[int],
    anchor_cols: set[int],
) -> bool:
    """Place one unaligned singleton marker in a currently valid free cell."""
    choices: list[tuple[int, int]] = []
    for r in range(GRID_SIZE):
        if r in anchor_rows:
            continue
        for c in range(GRID_SIZE):
            if c in anchor_cols or int(out[r, c]) != BG_COLOR:
                continue
            if _gen_is_isolated(inp_seed, r, c) and _gen_is_isolated(out, r, c):
                choices.append((r, c))
    if not choices:
        return False
    r, c = choices[int(np.random.randint(len(choices)))]
    inp_seed[r, c] = color
    out[r, c] = color
    return True


def _gen_paint_aligned_lines(
    out: np.ndarray, inp_seed: np.ndarray, ar: int, ac: int, colors: list[int], aligned_count: int
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]] | None:
    """Sample aligned sides, paint output lines, and keep marker-to-line records."""
    side_defs = {
        "top": ar,
        "bottom": GRID_SIZE - ANCHOR_SIZE - ar,
        "left": ac,
        "right": GRID_SIZE - ANCHOR_SIZE - ac,
    }
    sides = list(np.random.permutation(SIDE_NAMES)[:aligned_count])
    line_records: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    for i, side in enumerate(sides):
        color = int(colors[i])
        max_len = min(LINE_LENGTH_RANGE[1], int(side_defs[side]))
        if max_len < LINE_LENGTH_RANGE[0]:
            return None
        length = int(np.random.randint(LINE_LENGTH_RANGE[0], max_len + 1))
        if side == "top":
            col = int(np.random.choice([ac, ac + 1]))
            marker = (ar - length, col)
            endpoint = (ar - 1, col)
        elif side == "bottom":
            col = int(np.random.choice([ac, ac + 1]))
            marker = (ar + 1 + length, col)
            endpoint = (ar + ANCHOR_SIZE, col)
        elif side == "left":
            row = int(np.random.choice([ar, ar + 1]))
            marker = (row, ac - length)
            endpoint = (row, ac - 1)
        else:
            row = int(np.random.choice([ar, ar + 1]))
            marker = (row, ac + 1 + length)
            endpoint = (row, ac + ANCHOR_SIZE)

        mr, mc = marker
        cells = _line_cells(mr, mc, endpoint[0], endpoint[1])
        if any(int(out[r, c]) != BG_COLOR for r, c in cells):
            return None
        if not _gen_is_isolated(inp_seed, mr, mc):
            return None

        for r, c in cells:
            out[r, c] = color
        inp_seed[mr, mc] = color
        line_records.append((marker, cells))

    return line_records


def _gen_attempt() -> np.ndarray | None:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    inp_seed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    ar = int(np.random.randint(*ANCHOR_ROW_RANGE))
    ac = int(np.random.randint(*ANCHOR_COL_RANGE))
    out[ar : ar + ANCHOR_SIZE, ac : ac + ANCHOR_SIZE] = ANCHOR_COLOR
    inp_seed[ar : ar + ANCHOR_SIZE, ac : ac + ANCHOR_SIZE] = ANCHOR_COLOR
    anchor_rows = {ar, ar + 1}
    anchor_cols = {ac, ac + 1}

    total_count = int(np.random.randint(*TOTAL_MARKER_RANGE))
    aligned_min = max(1, total_count - MAX_UNALIGNED_MARKERS)
    aligned_max = min(MAX_ALIGNED_MARKERS, total_count - 1)
    aligned_count = int(np.random.randint(aligned_min, aligned_max + 1))

    color_count = int(np.random.randint(*MARKER_COLOR_COUNT_RANGE))
    palette = np.random.choice(MARKER_COLORS, size=color_count, replace=False).astype(int).tolist()
    if color_count == 1:
        colors = [palette[0]] * total_count
    else:
        colors = [palette[0], palette[1]]
        colors.extend(int(np.random.choice(palette)) for _ in range(total_count - 2))
        np.random.shuffle(colors)

    line_records = _gen_paint_aligned_lines(out, inp_seed, ar, ac, colors, aligned_count)
    if line_records is None:
        return None

    for color in colors[aligned_count:]:
        if not _gen_place_unaligned(inp_seed, out, int(color), anchor_rows, anchor_cols):
            return None

    inp = out.copy()
    for marker, cells in line_records:
        for r, c in cells:
            if (r, c) != marker:
                inp[r, c] = BG_COLOR

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d43fd935 input")
