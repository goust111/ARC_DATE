import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _separator_columns(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None
    for color in sorted(set(int(x) for x in np.unique(g)) - {BG}):
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not cols:
            continue
        bands = _bands(w, cols)
        if len(bands) not in (3, 4):
            continue
        widths = [c1 - c0 + 1 for c0, c1 in bands]
        if len(set(widths)) != 1:
            continue
        score = (len(cols), len(bands))
        if best is None or score > best[0]:
            best = (score, color, cols)
    assert best is not None, "failed to detect panel separators"
    return best[1], best[2]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    seps = sorted(set(int(x) for x in separators))
    bands = []
    start = 0
    for c in seps:
        if start <= c - 1:
            bands.append((start, c - 1))
        start = c + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _extract_sprite(panel: np.ndarray, sep: int) -> tuple[tuple[int, int], np.ndarray] | None:
    mask = (panel != BG) & (panel != sep)
    if not np.any(mask):
        return None
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    sub = panel[r0 : r1 + 1, c0 : c1 + 1]
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    patch[submask] = sub[submask]
    return (r0, c0), patch


def _paste_clipped(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = dst.shape
    ph, pw = patch.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top) + ph)
    c1 = min(w, int(left) + pw)
    if r0 >= r1 or c0 >= c1:
        return
    src_r0 = r0 - int(top)
    src_c0 = c0 - int(left)
    src = patch[src_r0 : src_r0 + (r1 - r0), src_c0 : src_c0 + (c1 - c0)]
    mask = src != BG
    dst_view = dst[r0:r1, c0:c1]
    dst_view[mask] = src[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate a moving sprite across equal-width panels and draw the next frame.

    The last panel is blank. The sprite moves by a constant row/column offset from
    panel to panel. When the motion points left, mirror horizontally; when it
    points up, mirror vertically. Paint the predicted next frame into the last
    panel, clipping any out-of-bounds cells.
    """
    g = np.asarray(grid, dtype=int)
    sep, sep_cols = _separator_columns(g)
    panel_cols = _bands(g.shape[1], sep_cols)
    panels = [g[:, c0 : c1 + 1] for c0, c1 in panel_cols]

    observed = []
    for idx, panel in enumerate(panels):
        found = _extract_sprite(panel, sep)
        if found is not None:
            pos, patch = found
            observed.append((idx, pos, patch))

    assert len(observed) >= 2, "need at least two observed frames"
    assert observed[-1][0] == len(panels) - 2, "blank frame must be the last panel"

    positions = [pos for _idx, pos, _patch in observed]
    deltas = [
        (positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        for i in range(len(positions) - 1)
    ]
    dr, dc = deltas[0]
    assert all(delta == (dr, dc) for delta in deltas), "motion must be linear"

    base_patch = max(observed, key=lambda item: int(np.count_nonzero(item[2])))[2]
    next_patch = base_patch.copy()
    if dc < 0:
        next_patch = np.fliplr(next_patch)
    if dr < 0:
        next_patch = np.flipud(next_patch)

    next_r = positions[-1][0] + dr
    next_c = positions[-1][1] + dc

    out = g.copy()
    blank_c0, blank_c1 = panel_cols[-1]
    out[:, blank_c0 : blank_c1 + 1] = BG
    _paste_clipped(out[:, blank_c0 : blank_c1 + 1], next_patch, next_r, next_c)
    for c in sep_cols:
        out[:, c] = sep
    return out


def _random_connected_cells(h: int, w: int, area: int) -> set[tuple[int, int]]:
    start = (int(RNG.integers(h)), int(RNG.integers(w)))
    cells = {start}
    frontier = {start}
    while len(cells) < area:
        anchor = list(frontier)[int(RNG.integers(len(frontier)))]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = anchor[0] + dr, anchor[1] + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                neighbors.append((rr, cc))
        if neighbors:
            cell = neighbors[int(RNG.integers(len(neighbors)))]
            cells.add(cell)
            frontier.add(cell)
        else:
            frontier.remove(anchor)
            if not frontier:
                frontier = set(cells)
    return cells


def _make_sprite(require_hflip: bool, require_vflip: bool) -> np.ndarray | None:
    for _ in range(120):
        h = int(RNG.integers(2, 6))
        w = int(RNG.integers(2, 6))
        area = int(RNG.integers(max(3, (h * w) // 2), h * w + 1))
        cells = _random_connected_cells(h, w, area)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        patch = np.full((hh, ww), BG, dtype=int)
        color_count = 1 if RNG.random() < 0.45 else int(RNG.integers(2, min(4, len(PALETTE)) + 1))
        colors = RNG.choice(PALETTE, size=color_count, replace=False)
        for r, c in cells:
            patch[r - r0, c - c0] = int(colors[int(RNG.integers(color_count))])
        if require_hflip and np.array_equal(patch, np.fliplr(patch)):
            continue
        if require_vflip and np.array_equal(patch, np.flipud(patch)):
            continue
        return patch
    return None


def _visible_after_paste(panel_h: int, panel_w: int, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    return not (
        top >= panel_h
        or left >= panel_w
        or top + ph <= 0
        or left + pw <= 0
    )


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(RNG.choice([3, 4]))
    observed_count = panel_count - 1
    panel_h = int(RNG.integers(8, 15))
    panel_w = int(RNG.integers(8, 15))

    dr = int(RNG.choice([-2, -1, 1, 2]))
    dc = int(RNG.choice([-2, -1, 1, 2]))
    sprite = _make_sprite(require_hflip=dc < 0, require_vflip=dr < 0)
    if sprite is None:
        return None
    sh, sw = sprite.shape
    if sh >= panel_h or sw >= panel_w:
        return None

    valid = []
    for r0 in range(panel_h - sh + 1):
        for c0 in range(panel_w - sw + 1):
            ok = True
            for k in range(observed_count):
                rr = r0 + k * dr
                cc = c0 + k * dc
                if not (0 <= rr <= panel_h - sh and 0 <= cc <= panel_w - sw):
                    ok = False
                    break
            if not ok:
                continue
            next_r = r0 + observed_count * dr
            next_c = c0 + observed_count * dc
            if not _visible_after_paste(panel_h, panel_w, sprite, next_r, next_c):
                continue
            valid.append((r0, c0, next_r, next_c))
    if not valid:
        return None

    clipped_candidates = [
        item
        for item in valid
        if not (0 <= item[2] <= panel_h - sh and 0 <= item[3] <= panel_w - sw)
    ]
    if clipped_candidates and RNG.random() < 0.5:
        r0, c0, next_r, next_c = clipped_candidates[int(RNG.integers(len(clipped_candidates)))]
    else:
        r0, c0, next_r, next_c = valid[int(RNG.integers(len(valid)))]

    grid = np.full((panel_h, panel_count * panel_w + (panel_count - 1)), BG, dtype=int)
    for p in range(panel_count - 1):
        grid[:, (p + 1) * panel_w + p] = SEP

    for k in range(observed_count):
        rr = r0 + k * dr
        cc = c0 + k * dc
        panel_left = k * (panel_w + 1)
        _paste_clipped(grid[:, panel_left : panel_left + panel_w], sprite, rr, cc)

    out = main(grid)
    blank_left = (panel_count - 1) * (panel_w + 1)
    blank_panel = out[:, blank_left : blank_left + panel_w]
    if not np.any(blank_panel != BG):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
