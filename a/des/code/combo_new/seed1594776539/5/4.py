import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PATTERN_COLORS = (1, 2, 3, 4, 6, 8, 9)
GRID_OPTIONS = ((2, 2), (2, 3), (3, 2))
MIN_PANEL_SIZE = 5
MAX_PANEL_SIZE = 9
MAX_MARKERS = 3
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _split_bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator rows/cols may not be adjacent"
        bands.append((start, sep))
        start = sep + 1
    assert start < length, "separator may not touch the final border"
    bands.append((start, length))
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (start, end) in enumerate(bands):
        if start <= pos < end:
            return idx
    raise AssertionError("position does not belong to any band")


def _offset_side(local_index: int, length: int) -> int:
    pos = local_index + 1
    before = pos
    after = length - pos
    assert before > 0 and after > 0, "marker must not sit at a segment endpoint"
    assert before != after, "marker may not sit at the exact segment midpoint"
    return -1 if before < after else 1


def _panel_pattern(panel: np.ndarray) -> tuple[int, np.ndarray]:
    coords = np.argwhere(panel != BLACK)
    assert len(coords) > 0, "every panel must contain a pattern"
    colors = {int(panel[int(r), int(c)]) for r, c in coords}
    assert len(colors) == 1, "each panel pattern must use exactly one color"
    color = next(iter(colors))
    assert color not in {GREY, ORANGE}, "panel colors must differ from separator colors"
    return color, coords


def _panel_slices(
    row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], idx: tuple[int, int]
) -> tuple[slice, slice]:
    ri, ci = idx
    r0, r1 = row_bands[ri]
    c0, c1 = col_bands[ci]
    return slice(r0, r1), slice(c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay each source panel's pattern into a neighboring target panel based on orange divider markers.

    Input structure:
    - Black background with a grey 1-pixel separator lattice that splits the grid
      into equally sized rectangular panels.
    - Each panel contains one small monochrome pattern in a color different from
      black, grey, and orange.
    - Some separator segments contain one orange pixel instead of grey.

    Rule:
    - Recover the panel bands from the full grey divider rows and columns.
    - For a marker on a vertical divider segment, an upper-half marker copies the
      left panel into the right panel, while a lower-half marker copies right to left.
      For a marker on a horizontal divider segment, a left-half marker copies the
      upper panel downward, while a right-half marker copies the lower panel upward.
    - Copy only the source pattern's occupied relative coordinates; paint them in
      the source color inside the target panel, overwriting any target pixels there.
    - Source panels stay unchanged, unrelated panels stay unchanged, and all orange
      divider pixels are restored to grey.

    Key insight:
    - The grey lattice identifies equal panel cells, and each orange marker encodes
      a directed neighbor-to-neighbor transfer whose payload is the source panel's
      relative shape rather than its absolute placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    sep_rows = [r for r in range(h) if bool(np.all(np.isin(g[r, :], (GREY, ORANGE))))]
    sep_cols = [c for c in range(w) if bool(np.all(np.isin(g[:, c], (GREY, ORANGE))))]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"

    row_bands = _split_bands(h, sep_rows)
    col_bands = _split_bands(w, sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-panel grid"

    panel_heights = {r1 - r0 for r0, r1 in row_bands}
    panel_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "all panels must be equally sized"
    panel_h = next(iter(panel_heights))
    panel_w = next(iter(panel_widths))

    sep_row_mask = np.zeros(h, dtype=bool)
    sep_col_mask = np.zeros(w, dtype=bool)
    sep_row_mask[sep_rows] = True
    sep_col_mask[sep_cols] = True

    for r, c in np.argwhere((g == GREY) | (g == ORANGE)):
        rr = int(r)
        cc = int(c)
        assert bool(sep_row_mask[rr] or sep_col_mask[cc]), "separator colors may only appear on the lattice"
        if sep_row_mask[rr] and sep_col_mask[cc]:
            assert int(g[rr, cc]) == GREY, "orange markers may not occupy lattice intersections"

    patterns: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for ri in range(len(row_bands)):
        for ci in range(len(col_bands)):
            rs, cs = _panel_slices(row_bands, col_bands, (ri, ci))
            patterns[(ri, ci)] = _panel_pattern(g[rs, cs])

    out = g.copy()
    out[g == ORANGE] = GREY

    sep_row_to_idx = {sep: idx for idx, sep in enumerate(sep_rows)}
    sep_col_to_idx = {sep: idx for idx, sep in enumerate(sep_cols)}
    used_segments: set[tuple[str, int, int]] = set()

    orange_positions = sorted((int(r), int(c)) for r, c in np.argwhere(g == ORANGE))
    if not orange_positions:
        return out

    for r, c in orange_positions:
        on_sep_row = bool(sep_row_mask[r])
        on_sep_col = bool(sep_col_mask[c])
        assert on_sep_row ^ on_sep_col, "orange marker must lie on exactly one separator direction"

        if on_sep_col:
            row_idx = _band_index(r, row_bands)
            col_idx = sep_col_to_idx[c]
            side = _offset_side(r - row_bands[row_idx][0], panel_h)
            src = (row_idx, col_idx) if side < 0 else (row_idx, col_idx + 1)
            dst = (row_idx, col_idx + 1) if side < 0 else (row_idx, col_idx)
            seg_id = ("V", row_idx, col_idx)
        else:
            row_idx = sep_row_to_idx[r]
            col_idx = _band_index(c, col_bands)
            side = _offset_side(c - col_bands[col_idx][0], panel_w)
            src = (row_idx, col_idx) if side < 0 else (row_idx + 1, col_idx)
            dst = (row_idx + 1, col_idx) if side < 0 else (row_idx, col_idx)
            seg_id = ("H", row_idx, col_idx)

        assert seg_id not in used_segments, "each separator segment may carry at most one marker"
        used_segments.add(seg_id)

        color, coords = patterns[src]
        rs, cs = _panel_slices(row_bands, col_bands, dst)
        dst_panel = out[rs, cs]
        for rr, cc in coords:
            dst_panel[int(rr), int(cc)] = color

    return out


def _random_connected_shape(height: int, width: int, cells: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    shape = {start}
    frontier = {start}

    while len(shape) < cells:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in shape:
                    candidates.add((rr, cc))
        if not candidates:
            return None
        nxt = random.choice(tuple(candidates))
        shape.add(nxt)
        frontier.add(nxt)

    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    r0 = min(rows)
    c0 = min(cols)
    return sorted((r - r0, c - c0) for r, c in shape)


def _random_panel_pattern(panel_h: int, panel_w: int) -> list[tuple[int, int]] | None:
    bbox_h = random.randint(2, min(5, panel_h))
    bbox_w = random.randint(2, min(5, panel_w))
    max_cells = min(6, bbox_h * bbox_w - 1)
    cells = random.randint(2, max_cells)
    shape = _random_connected_shape(bbox_h, bbox_w, cells)
    if shape is None:
        return None
    top = random.randint(0, panel_h - bbox_h)
    left = random.randint(0, panel_w - bbox_w)
    return [(top + r, left + c) for r, c in shape]


def _pick_marker_local(length: int, side: int) -> int:
    choices = []
    for idx in range(1, length - 1):
        pos = idx + 1
        before = pos
        after = length - pos
        if before == after:
            continue
        if (before < after and side < 0) or (before > after and side > 0):
            choices.append(idx)
    assert choices, "panel side must admit both directional marker branches"
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = random.choice(GRID_OPTIONS)
    max_h = min(MAX_PANEL_SIZE, (30 - (panel_rows - 1)) // panel_rows)
    max_w = min(MAX_PANEL_SIZE, (30 - (panel_cols - 1)) // panel_cols)
    if max_h < MIN_PANEL_SIZE or max_w < MIN_PANEL_SIZE:
        return None

    panel_h = random.randint(MIN_PANEL_SIZE, max_h)
    panel_w = random.randint(MIN_PANEL_SIZE, max_w)
    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    sep_rows = [(idx + 1) * panel_h + idx for idx in range(panel_rows - 1)]
    sep_cols = [(idx + 1) * panel_w + idx for idx in range(panel_cols - 1)]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    row_bands = _split_bands(grid_h, sep_rows)
    col_bands = _split_bands(grid_w, sep_cols)
    colors = random.sample(PATTERN_COLORS, panel_rows * panel_cols)

    color_idx = 0
    for ri in range(panel_rows):
        for ci in range(panel_cols):
            pattern = _random_panel_pattern(panel_h, panel_w)
            if pattern is None:
                return None
            color = colors[color_idx]
            color_idx += 1
            r0, r1 = row_bands[ri]
            c0, c1 = col_bands[ci]
            panel = grid[r0:r1, c0:c1]
            for rr, cc in pattern:
                panel[rr, cc] = color

    candidates: list[tuple[str, int, int, int, tuple[int, int], tuple[int, int]]] = []
    for ri in range(panel_rows):
        for ci in range(panel_cols - 1):
            for side in (-1, 1):
                src = (ri, ci) if side < 0 else (ri, ci + 1)
                dst = (ri, ci + 1) if side < 0 else (ri, ci)
                candidates.append(("V", ri, ci, side, src, dst))
    for ri in range(panel_rows - 1):
        for ci in range(panel_cols):
            for side in (-1, 1):
                src = (ri, ci) if side < 0 else (ri + 1, ci)
                dst = (ri + 1, ci) if side < 0 else (ri, ci)
                candidates.append(("H", ri, ci, side, src, dst))

    random.shuffle(candidates)
    marker_count = random.randint(1, min(MAX_MARKERS, len(candidates)))
    chosen: list[tuple[str, int, int, int, tuple[int, int], tuple[int, int]]] = []
    used_segments: set[tuple[str, int, int]] = set()
    used_targets: set[tuple[int, int]] = set()

    for cand in candidates:
        orient, ri, ci, side, _src, dst = cand
        seg_id = (orient, ri, ci)
        if seg_id in used_segments or dst in used_targets:
            continue
        chosen.append(cand)
        used_segments.add(seg_id)
        used_targets.add(dst)
        if len(chosen) == marker_count:
            break

    if not chosen:
        return None

    for orient, ri, ci, side, _src, _dst in chosen:
        if orient == "V":
            local = _pick_marker_local(panel_h, side)
            r = row_bands[ri][0] + local
            c = sep_cols[ci]
        else:
            local = _pick_marker_local(panel_w, side)
            r = sep_rows[ri]
            c = col_bands[ci][0] + local
        grid[r, c] = ORANGE

    base = grid.copy()
    base[base == ORANGE] = GREY
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != base), "at least one target panel must change"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid panel-overlay input")
