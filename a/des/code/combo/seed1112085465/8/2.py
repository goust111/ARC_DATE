import numpy as np
from collections import Counter, deque


GEN_BG = 7
GEN_SEPARATOR = 5
GEN_FRAME = 9
GEN_TEMPLATE_COLORS = [1, 2, 3, 4, 6, 8]

GEN_ATTEMPTS = 400
PANEL_H_RANGE = (9, 16)
PANEL_W_RANGE = (9, 16)
TEMPLATE_MARGIN_RANGE = (1, 3)
FRAME_INSET_RANGE = (0, 2)
HOLE_ATTEMPTS = 80


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_size(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sr, sc = map(int, coords[0])
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    size = 0
    while q:
        r, c = q.popleft()
        size += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return size


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) == _component_size(mask)


def _tight_crop(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    assert coords.size > 0, "tight crop requires at least one non-background pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _find_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        if c != w - c - 1:
            continue
        if np.all(grid[:, c] == grid[0, c]):
            candidates.append(c)
    assert len(candidates) == 1, "expected one full-height separator column"
    return int(candidates[0])


def _infer_background(grid: np.ndarray, sep_c: int) -> int:
    h, w = grid.shape
    boundary = []
    for c in range(w):
        if c != sep_c:
            boundary.append(int(grid[0, c]))
            boundary.append(int(grid[h - 1, c]))
    for r in range(1, h - 1):
        boundary.append(int(grid[r, 0]))
        boundary.append(int(grid[r, w - 1]))
    return Counter(boundary).most_common(1)[0][0]


def _enclosed_background_mask(panel: np.ndarray, bg: int) -> np.ndarray:
    h, w = panel.shape
    bg_mask = panel == bg
    outside = np.zeros_like(bg_mask, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    hole = bg_mask & ~outside
    assert hole.any(), "expected one enclosed background hole"
    assert _is_connected(hole), "hole must be connected"
    return hole


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep left-panel template pixels only where the right-panel mask exposes a hole,
    then tight-crop the result and rotate it 90 degrees clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_c = _find_separator_col(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert left.shape == right.shape

    bg = _infer_background(g, sep_c)
    hole = _enclosed_background_mask(right, bg)

    clipped = np.full(left.shape, bg, dtype=int)
    clipped[hole] = left[hole]
    cropped = _tight_crop(clipped, bg)
    return np.rot90(cropped, -1)


def _template_pattern(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    color_count = int(rng.integers(3, min(len(GEN_TEMPLATE_COLORS), 5) + 1))
    colors = rng.choice(GEN_TEMPLATE_COLORS, size=color_count, replace=False).astype(int)
    rr, cc = np.indices((height, width))
    mode = int(rng.integers(0, 3))

    if mode == 0:
        motif_h = int(rng.integers(2, min(height, 4) + 1))
        motif_w = int(rng.integers(2, min(width, 4) + 1))
        motif = rng.choice(colors, size=(motif_h, motif_w), replace=True).astype(int)
        while len(np.unique(motif)) < min(3, len(colors)):
            motif = rng.choice(colors, size=(motif_h, motif_w), replace=True).astype(int)
        pattern = motif[rr % motif_h, cc % motif_w]
    elif mode == 1:
        band_h = int(rng.integers(1, 3))
        band_w = int(rng.integers(1, 3))
        pattern = colors[(rr // band_h + 2 * (cc // band_w)) % len(colors)]
    else:
        pattern = colors[(rr + 2 * cc + rr // 2) % len(colors)]
        accent = (rr % 3 == 1) & (cc % 4 == 0)
        pattern = pattern.copy()
        pattern[accent] = colors[-1]

    assert len(np.unique(pattern)) >= 3
    return pattern.astype(int)


def _hole_valid(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    if area < 4:
        return False
    if not _is_connected(mask):
        return False

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)
    bbox_area = bbox_h * bbox_w

    if bbox_h < 2 or bbox_w < 2:
        return False
    if area == bbox_area:
        return False
    if area <= max(bbox_h, bbox_w):
        return False
    if area >= int(mask.size * 0.82):
        return False
    return True


def _smooth_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if out[r, c]:
                continue
            degree = sum(int(out[nr, nc]) for nr, nc in _neighbors4(r, c, h, w))
            if degree >= 3 and rng.random() < 0.55:
                out[r, c] = True
    return out


def _hole_by_growth(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    target = int(rng.integers(max(4, (height * width) // 4), max(5, int(height * width * 0.62)) + 1))
    r = int(rng.integers(0, height))
    c = int(rng.integers(0, width))
    mask[r, c] = True

    while int(mask.sum()) < target:
        frontier = []
        weights = []
        for rr in range(height):
            for cc in range(width):
                if mask[rr, cc]:
                    continue
                degree = sum(int(mask[nr, nc]) for nr, nc in _neighbors4(rr, cc, height, width))
                if degree:
                    frontier.append((rr, cc))
                    weights.append(degree * degree)
        if not frontier:
            break
        idx = int(rng.choice(len(frontier), p=np.array(weights, dtype=float) / np.sum(weights)))
        rr, cc = frontier[idx]
        mask[rr, cc] = True

    return _smooth_mask(mask, rng)


def _hole_by_rectangles(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rects = int(rng.integers(2, 5))

    for i in range(rects):
        rh = int(rng.integers(1, max(2, min(height, max(3, height // 2 + 1))) + 1))
        rw = int(rng.integers(1, max(2, min(width, max(3, width // 2 + 1))) + 1))
        rh = min(rh, height)
        rw = min(rw, width)

        if i == 0 or not mask.any():
            r0 = int(rng.integers(0, height - rh + 1))
            c0 = int(rng.integers(0, width - rw + 1))
        else:
            coords = np.argwhere(mask)
            ar, ac = map(int, coords[int(rng.integers(0, len(coords)))])
            r_lo = max(0, ar - rh + 1)
            r_hi = min(ar, height - rh)
            c_lo = max(0, ac - rw + 1)
            c_hi = min(ac, width - rw)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))

        mask[r0 : r0 + rh, c0 : c0 + rw] = True

    coords = np.argwhere(mask)
    if coords.size:
        for _ in range(3):
            rr, cc = map(int, coords[int(rng.integers(0, len(coords)))])
            if sum(int(mask[nr, nc]) for nr, nc in _neighbors4(rr, cc, height, width)) >= 3:
                trial = mask.copy()
                trial[rr, cc] = False
                if trial.any() and _is_connected(trial):
                    mask = trial
                    break
    return mask


def _hole_by_walk(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    steps = int(rng.integers(max(5, (height * width) // 3), max(6, int(height * width * 0.8)) + 1))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    r = int(rng.integers(0, height))
    c = int(rng.integers(0, width))
    mask[r, c] = True

    for _ in range(steps):
        dr, dc = dirs[int(rng.integers(0, 4))]
        seg_len = int(rng.integers(1, 4))
        for _ in range(seg_len):
            preferred = (r + dr, c + dc)
            choices = [(nr, nc) for nr, nc in _neighbors4(r, c, height, width)]
            if not choices:
                break
            if 0 <= preferred[0] < height and 0 <= preferred[1] < width and rng.random() < 0.7:
                r, c = preferred
            else:
                r, c = choices[int(rng.integers(0, len(choices)))]
            mask[r, c] = True

            if rng.random() < 0.45:
                if dr == 0:
                    side = [(r - 1, c), (r + 1, c)]
                else:
                    side = [(r, c - 1), (r, c + 1)]
                side = [(nr, nc) for nr, nc in side if 0 <= nr < height and 0 <= nc < width]
                if side:
                    nr, nc = side[int(rng.integers(0, len(side)))]
                    mask[nr, nc] = True

    return _smooth_mask(mask, rng)


def _random_hole(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(HOLE_ATTEMPTS):
        mode = int(rng.integers(0, 3))
        if mode == 0:
            hole = _hole_by_growth(height, width, rng)
        elif mode == 1:
            hole = _hole_by_rectangles(height, width, rng)
        else:
            hole = _hole_by_walk(height, width, rng)
        if _hole_valid(hole):
            return hole
    return None


def _template_bbox(panel_h: int, panel_w: int, rng: np.random.Generator):
    top = int(rng.integers(*TEMPLATE_MARGIN_RANGE))
    left = int(rng.integers(*TEMPLATE_MARGIN_RANGE))
    bottom_margin = int(rng.integers(*TEMPLATE_MARGIN_RANGE))
    right_margin = int(rng.integers(*TEMPLATE_MARGIN_RANGE))
    bottom = panel_h - bottom_margin - 1
    right = panel_w - right_margin - 1
    if bottom - top + 1 < 6 or right - left + 1 < 6:
        return None
    return top, left, bottom, right


def _frame_bbox(tbox, rng: np.random.Generator):
    tr0, tc0, tr1, tc1 = tbox
    inset_top = int(rng.integers(*FRAME_INSET_RANGE))
    inset_left = int(rng.integers(*FRAME_INSET_RANGE))
    inset_bottom = int(rng.integers(*FRAME_INSET_RANGE))
    inset_right = int(rng.integers(*FRAME_INSET_RANGE))
    fr0 = tr0 + inset_top
    fc0 = tc0 + inset_left
    fr1 = tr1 - inset_bottom
    fc1 = tc1 - inset_right
    if fr1 - fr0 + 1 < 5 or fc1 - fc0 + 1 < 5:
        return None
    return fr0, fc0, fr1, fc1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_h = int(rng.integers(*PANEL_H_RANGE))
    panel_w = int(rng.integers(*PANEL_W_RANGE))

    tbox = _template_bbox(panel_h, panel_w, rng)
    if tbox is None:
        return None
    fbox = _frame_bbox(tbox, rng)
    if fbox is None:
        return None

    tr0, tc0, tr1, tc1 = tbox
    fr0, fc0, fr1, fc1 = fbox
    frame_h = fr1 - fr0 + 1
    frame_w = fc1 - fc0 + 1

    hole = _random_hole(frame_h - 2, frame_w - 2, rng)
    if hole is None:
        return None

    left = np.full((panel_h, panel_w), GEN_BG, dtype=int)
    template = _template_pattern(tr1 - tr0 + 1, tc1 - tc0 + 1, rng)
    left[tr0 : tr1 + 1, tc0 : tc1 + 1] = template

    right = np.full((panel_h, panel_w), GEN_BG, dtype=int)
    right[fr0 : fr1 + 1, fc0 : fc1 + 1] = GEN_FRAME
    right[fr0 + 1 : fr1, fc0 + 1 : fc1][hole] = GEN_BG

    panel_hole = np.zeros((panel_h, panel_w), dtype=bool)
    panel_hole[fr0 + 1 : fr1, fc0 + 1 : fc1] = hole
    visible = left[panel_hole]
    if len(set(map(int, visible))) < 2:
        return None

    grid = np.full((panel_h, panel_w * 2 + 1), GEN_BG, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w] = GEN_SEPARATOR
    grid[:, panel_w + 1 :] = right

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] < 2 or out.shape[1] < 2:
        return None
    if len(set(map(int, out[out != GEN_BG].ravel()))) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
