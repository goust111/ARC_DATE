import numpy as np


BG_COLOR = 0
CROSS_COLOR = 5
STAMP_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANT_KEYS = ("tl", "tr", "bl", "br")


def _components(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _quadrant_bounds(h: int, w: int, cross_r: int, cross_c: int):
    return {
        "tl": (0, cross_r, 0, cross_c),
        "tr": (0, cross_r, cross_c + 1, w),
        "bl": (cross_r + 1, h, 0, cross_c),
        "br": (cross_r + 1, h, cross_c + 1, w),
    }


def _transform_mask(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return mask.copy()
    if quadrant == "tr":
        return np.fliplr(mask)
    if quadrant == "bl":
        return np.flipud(mask)
    if quadrant == "br":
        return np.flipud(np.fliplr(mask))
    raise ValueError(f"unknown quadrant {quadrant}")


def _bbox_mask(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in np.unique(g):
        if int(np.count_nonzero(g == color)) != h + w - 1:
            continue
        rows = [r for r in range(1, h - 1) if np.all(g[r, :] == color)]
        cols = [c for c in range(1, w - 1) if np.all(g[:, c] == color)]
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((rows[0], cols[0], int(color)))

    assert len(candidates) == 1
    cross_r, cross_c, cross_color = candidates[0]

    cross_mask = np.zeros_like(g, dtype=bool)
    cross_mask[cross_r, :] = True
    cross_mask[:, cross_c] = True

    payload = g[~cross_mask]
    vals, cnts = np.unique(payload, return_counts=True)
    bg = int(vals[np.argmax(cnts)])

    infos = []
    original_stamp_mask = np.zeros_like(g, dtype=bool)
    for key, (r0, r1, c0, c1) in _quadrant_bounds(h, w, cross_r, cross_c).items():
        sub = g[r0:r1, c0:c1]
        mask = sub != bg
        comps = _components(mask)
        stamp_comps = [comp for comp in comps if len(comp) > 1]
        marker_comps = [comp for comp in comps if len(comp) == 1]
        assert len(stamp_comps) == 1
        assert len(marker_comps) >= 1

        stamp_rel = stamp_comps[0]
        stamp_global = [(r0 + r, c0 + c) for r, c in stamp_rel]
        color = int(g[stamp_global[0]])
        assert all(int(g[r, c]) == color for r, c in stamp_global)
        assert all(int(g[r0 + r, c0 + c]) == color for comp in marker_comps for r, c in comp)

        stamp_box = _bbox_mask(stamp_rel)
        sr, sc = stamp_box.shape
        assert sr % 2 == 1 and sc % 2 == 1
        assert stamp_box[sr // 2, sc // 2]
        assert not np.array_equal(stamp_box, np.fliplr(stamp_box))
        assert not np.array_equal(stamp_box, np.flipud(stamp_box))

        for r, c in stamp_global:
            original_stamp_mask[r, c] = True

        marker_points = [(r0 + r, c0 + c) for comp in marker_comps for r, c in comp]
        infos.append(
            {
                "quadrant": key,
                "color": color,
                "stamp": stamp_box,
                "markers": marker_points,
            }
        )

    return cross_mask, original_stamp_mask, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the grey cross, split the grid into four quadrants, and treat each quadrant
    as one stamp component plus several singleton markers. Copy the quadrant's stamp
    onto every marker, using a quadrant-dependent reflection:
    TL none, TR horizontal, BL vertical, BR both.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cross_mask, original_stamp_mask, infos = _parse_scene(g)
    h, w = g.shape
    out = g.copy()

    for info in infos:
        stamp = _transform_mask(info["stamp"], info["quadrant"])
        center_r = stamp.shape[0] // 2
        center_c = stamp.shape[1] // 2
        cells = np.argwhere(stamp)

        for mr, mc in info["markers"]:
            for rr, cc in cells:
                tr = int(mr + rr - center_r)
                tc = int(mc + cc - center_c)
                if not (0 <= tr < h and 0 <= tc < w):
                    continue
                if cross_mask[tr, tc] or original_stamp_mask[tr, tc]:
                    continue
                out[tr, tc] = info["color"]

    return out


def _random_stamp(rng):
    for _ in range(200):
        h = int(rng.choice([3, 5]))
        w = int(rng.choice([3, 5]))
        min_area = max(5, (h * w) // 3)
        max_area = min(h * w - 1, (2 * h * w) // 3 + 1)
        target_area = int(rng.integers(min_area, max_area + 1))

        cells = {(h // 2, w // 2)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric stamp")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int):
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _footprint(mask: np.ndarray, center, shape):
    h, w = shape
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    cells = set()
    for rr, cc in np.argwhere(mask):
        tr = int(center[0] + rr - center_r)
        tc = int(center[1] + cc - center_c)
        if 0 <= tr < h and 0 <= tc < w:
            cells.add((tr, tc))
    return cells, int(mask.sum())


def _choose_markers(
    rng,
    grid: np.ndarray,
    quadrant: str,
    bounds,
    stamp_mask: np.ndarray,
    stamp_cells,
    color: int,
    original_stamp_mask: np.ndarray,
):
    r0, r1, c0, c1 = bounds
    count = int(rng.integers(1, 4))
    same_color_cells = set(stamp_cells)
    selected = []
    copy_cells = set()
    prefer_clipped = bool(rng.integers(2))

    for idx in range(count):
        candidates = []
        clipped_candidates = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                if grid[r, c] != BG_COLOR or (r, c) in same_color_cells:
                    continue
                if any((r + dr, c + dc) in same_color_cells for dr, dc in NEIGHBORS4):
                    continue

                footprint, full_area = _footprint(stamp_mask, (r, c), grid.shape)
                if not footprint:
                    continue
                if any(not (r0 <= rr < r1 and c0 <= cc < c1) for rr, cc in footprint):
                    continue
                if any(original_stamp_mask[rr, cc] for rr, cc in footprint):
                    continue
                if footprint & copy_cells:
                    continue

                clipped = len(footprint) < full_area
                entry = (r, c, footprint, clipped)
                candidates.append(entry)
                if clipped:
                    clipped_candidates.append(entry)

        if not candidates:
            return None

        pool = candidates
        if idx == 0 and prefer_clipped and clipped_candidates:
            pool = clipped_candidates
        choice = pool[int(rng.integers(len(pool)))]
        r, c, footprint, _ = choice
        selected.append((r, c))
        same_color_cells.add((r, c))
        copy_cells |= footprint
        grid[r, c] = color

    return selected


def _gen_attempt():
    rng = np.random.default_rng()
    base_stamp = _random_stamp(rng)
    stamp_h, stamp_w = base_stamp.shape

    top_h = int(rng.integers(stamp_h + 3, stamp_h + 9))
    bottom_h = int(rng.integers(stamp_h + 3, stamp_h + 9))
    left_w = int(rng.integers(stamp_w + 3, stamp_w + 9))
    right_w = int(rng.integers(stamp_w + 3, stamp_w + 9))

    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    cross_r = top_h
    cross_c = left_w

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[cross_r, :] = CROSS_COLOR
    grid[:, cross_c] = CROSS_COLOR

    colors = list(rng.choice(STAMP_COLOR_POOL, size=4, replace=False))
    bounds = _quadrant_bounds(h, w, cross_r, cross_c)
    placements = {}

    for key in QUADRANT_KEYS:
        r0, r1, c0, c1 = bounds[key]
        qh = r1 - r0
        qw = c1 - c0

        row_gap = int(rng.integers(1, min(4, qh - stamp_h + 1)))
        col_gap = int(rng.integers(1, min(4, qw - stamp_w + 1)))

        if key == "tl":
            top = r1 - stamp_h - row_gap
            left = c1 - stamp_w - col_gap
        elif key == "tr":
            top = r1 - stamp_h - row_gap
            left = c0 + col_gap
        elif key == "bl":
            top = r0 + row_gap
            left = c1 - stamp_w - col_gap
        else:
            top = r0 + row_gap
            left = c0 + col_gap

        placements[key] = (top, left)

    original_stamp_mask = np.zeros_like(grid, dtype=bool)
    stamp_cells_by_quadrant = {}
    for key, color in zip(QUADRANT_KEYS, colors):
        top, left = placements[key]
        _paint_mask(grid, base_stamp, top, left, color)
        cells = set()
        for rr, cc in np.argwhere(base_stamp):
            cells.add((top + int(rr), left + int(cc)))
            original_stamp_mask[top + int(rr), left + int(cc)] = True
        stamp_cells_by_quadrant[key] = cells

    for key, color in zip(QUADRANT_KEYS, colors):
        markers = _choose_markers(
            rng=rng,
            grid=grid,
            quadrant=key,
            bounds=bounds[key],
            stamp_mask=_transform_mask(base_stamp, key),
            stamp_cells=stamp_cells_by_quadrant[key],
            color=color,
            original_stamp_mask=original_stamp_mask,
        )
        if markers is None:
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
