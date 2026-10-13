import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(pts)
    return out


def _bbox(points):
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_points(points):
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _mask_key(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return (0, 0, b"")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return (int(cropped.shape[0]), int(cropped.shape[1]), cropped.tobytes())


def _points_key(points):
    return _mask_key(_mask_from_points(points))


def _ring_mask(h, w, thickness):
    rr, cc = np.indices((h, w))
    return (rr < thickness) | (rr >= h - thickness) | (cc < thickness) | (cc >= w - thickness)


def _legend_and_frame(g):
    grey_comps = _components(g == GREY)
    assert grey_comps, "expected grey frame and grey legend templates"
    frame_idx = max(range(len(grey_comps)), key=lambda i: len(grey_comps[i]))
    frame = grey_comps[frame_idx]

    legend = {}
    legend_mask = np.zeros_like(g, dtype=bool)
    for i, comp in enumerate(grey_comps):
        if i == frame_idx:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if c1 + 1 >= g.shape[1]:
            continue
        color_cells = []
        for r in range(r0, r1 + 1):
            v = int(g[r, c1 + 1])
            if v not in (BG, GREY):
                color_cells.append((r, c1 + 1, v))
        if len(color_cells) != 1:
            continue
        rr, cc, color = color_cells[0]
        key = _points_key(comp)
        assert key not in legend, "legend templates must be unique"
        legend[key] = color
        for r, c in comp:
            legend_mask[r, c] = True
        legend_mask[rr, cc] = True

    assert legend, "expected at least one bottom legend entry"
    return frame, legend, legend_mask


def _infer_frame(frame_points):
    fr0, fc0, fr1, fc1 = _bbox(frame_points)
    fh, fw = fr1 - fr0 + 1, fc1 - fc0 + 1
    rel = [(r - fr0, c - fc0) for r, c in frame_points]
    grey_mask = np.zeros((fh, fw), dtype=bool)
    for r, c in rel:
        grey_mask[r, c] = True

    best_t = None
    for t in range(1, min(fh, fw) // 2 + 1):
        ring = _ring_mask(fh, fw, t)
        if np.all(ring[grey_mask]):
            best_t = t
            break
    assert best_t is not None, "could not infer rectangular frame thickness"
    return (fr0, fc0, fr1, fc1), best_t, grey_mask, _ring_mask(fh, fw, best_t)


def _local_shape_key(grey_mask, gap_points):
    local = set(gap_points)
    h, w = grey_mask.shape
    for r, c in gap_points:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grey_mask[nr, nc]:
                local.add((nr, nc))
    return _points_key(local)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit colored fragments into missing parts of a central grey frame and recolor them through a shape legend.

    Input structure:
    - The background is black, with one large grey rectangular frame near the center.
    - Several small grey cells are missing from the frame boundary, forming 4-connected irregular gaps.
    - Outside the frame are separate monochrome colored fragments; their colors are distractors.
    - Along the bottom edge, each legend entry is a small grey shape template with one colored square immediately to its right.

    Rule:
    - Reconstruct the complete rectangular grey frame and extract each missing boundary gap.
    - Match every outside fragment to the gap with the same normalized shape, using translation only.
    - For that gap, form the local completed boundary shape from the gap cells plus adjacent original grey frame cells.
    - Look up this local shape in the bottom legend and recolor only the inserted fragment cells with the legend color.
    - Erase the fragments' original positions and erase the legend marks; all original grey frame pixels stay grey.

    Key insight:
    - A fragment's input color is irrelevant: the output color is determined only after the fragment is fitted into a frame gap and the resulting local shape is matched to the legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_points, legend, legend_mask = _legend_and_frame(g)
    (fr0, fc0, fr1, fc1), _thickness, grey_mask, full_ring = _infer_frame(frame_points)
    fh, fw = grey_mask.shape

    missing = full_ring & ~grey_mask
    gaps = _components(missing)
    assert gaps, "expected at least one frame gap"

    piece_mask = (g != BG) & (g != GREY) & ~legend_mask
    pieces = []
    for comp in _components(piece_mask):
        if any(fr0 <= r <= fr1 and fc0 <= c <= fc1 for r, c in comp):
            continue
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "fragments must be monochrome"
        pieces.append({"points": comp, "key": _points_key(comp)})
    assert len(pieces) == len(gaps), "every gap must have one outside fragment"

    used = set()
    for gap in gaps:
        gap_key = _points_key(gap)
        matches = [i for i, piece in enumerate(pieces) if i not in used and piece["key"] == gap_key]
        assert len(matches) == 1, "gap-to-fragment shape matching must be unique"
        pi = matches[0]
        used.add(pi)

        template_key = _local_shape_key(grey_mask, gap)
        assert template_key in legend, "completed local frame shape must appear in the legend"
        color = int(legend[template_key])

        for r, c in pieces[pi]["points"]:
            out[r, c] = BG
        for rr, cc in gap:
            out[fr0 + rr, fc0 + cc] = color

    out[legend_mask] = BG
    return out


BASE_GAPS = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=bool),
]


def _oriented_gap(mask, side):
    if side == "top":
        return mask.copy()
    if side == "bottom":
        return np.flipud(mask).copy()
    if side == "left":
        return mask.T.copy()
    if side == "right":
        return np.fliplr(mask.T).copy()
    raise ValueError(side)


def _gap_positions(frame_shape, thickness, side, mask, rng, blocked):
    fh, fw = frame_shape
    gh, gw = mask.shape
    if side in ("top", "bottom"):
        lo, hi = thickness + 1, fw - thickness - gw - 1
        if hi < lo:
            return None
        c0 = rng.randint(lo, hi)
        r0 = 0 if side == "top" else fh - gh
    else:
        lo, hi = thickness + 1, fh - thickness - gh - 1
        if hi < lo:
            return None
        r0 = rng.randint(lo, hi)
        c0 = 0 if side == "left" else fw - gw

    pts = [(r0 + r, c0 + c) for r, c in zip(*np.nonzero(mask))]
    for r, c in pts:
        if blocked[r, c]:
            return None
        for nr in range(max(0, r - 1), min(fh, r + 2)):
            for nc in range(max(0, c - 1), min(fw, c + 2)):
                if blocked[nr, nc]:
                    return None
    return pts


def _mark_gap_blocked(blocked, pts):
    h, w = blocked.shape
    for r, c in pts:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _paste_mask(grid, mask, r0, c0, color):
    rr, cc = np.nonzero(mask)
    grid[r0 + rr, c0 + cc] = color


def _can_place(grid, occupied, mask, r0, c0, frame_bbox):
    h, w = grid.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    fr0, fc0, fr1, fc1 = frame_bbox
    if not (r0 + mh - 1 < fr0 or r0 > fr1 or c0 + mw - 1 < fc0 or c0 > fc1):
        return False
    rlo, rhi = max(0, r0 - 1), min(h, r0 + mh + 1)
    clo, chi = max(0, c0 - 1), min(w, c0 + mw + 1)
    patch = np.zeros((rhi - rlo, chi - clo), dtype=bool)
    patch[r0 - rlo : r0 - rlo + mh, c0 - clo : c0 - clo + mw] = mask
    return not np.any(occupied[rlo:rhi, clo:chi] & patch)


def _place_piece(grid, occupied, mask, color, frame_bbox, legend_top, rng):
    h, w = grid.shape
    mh, mw = mask.shape
    candidates = []
    for r0 in range(1, legend_top - mh):
        for c0 in range(1, w - mw):
            if _can_place(grid, occupied, mask, r0, c0, frame_bbox):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = rng.choice(candidates)
    _paste_mask(grid, mask, r0, c0, color)
    occupied[max(0, r0 - 1) : min(h, r0 + mh + 1), max(0, c0 - 1) : min(w, c0 + mw + 1)] = True
    return [(r0 + r, c0 + c) for r, c in zip(*np.nonzero(mask))]


def _connected(mask):
    return len(_components(mask)) == 1


def _build_attempt(rng):
    n = rng.randint(2, 4)
    thickness = rng.choice([2, 3])
    h = rng.randint(23, 30)
    w = rng.randint(23, 30)

    legend_band = 7
    legend_top = h - legend_band
    max_frame_h = legend_top - 5
    if max_frame_h < 10:
        return None
    fh = rng.randint(9, min(14, max_frame_h))
    fw = rng.randint(11, min(18, w - 6))
    fr0 = rng.randint(2, legend_top - fh - 2)
    fc0 = rng.randint(2, w - fw - 2)
    frame_bbox = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)

    full_ring = _ring_mask(fh, fw, thickness)
    missing = np.zeros((fh, fw), dtype=bool)
    blocked = np.zeros((fh, fw), dtype=bool)

    side_order = rng.sample(["top", "bottom", "left", "right"], 4)
    sides = [side_order[i % 4] for i in range(n)]
    rng.shuffle(sides)
    base_choices = [m for m in BASE_GAPS if m.shape[0] <= thickness and _connected(m)]
    rng.shuffle(base_choices)

    gaps = []
    gap_keys = set()
    local_keys = set()
    for side in sides:
        placed = None
        rng.shuffle(base_choices)
        for base in base_choices:
            oriented = _oriented_gap(base, side)
            pts = _gap_positions((fh, fw), thickness, side, oriented, rng, blocked)
            if pts is None:
                continue
            gkey = _points_key(pts)
            if gkey in gap_keys:
                continue
            grey_for_key = full_ring.copy()
            for r, c in pts:
                grey_for_key[r, c] = False
            lkey = _local_shape_key(grey_for_key, pts)
            if lkey in local_keys:
                continue
            placed = (oriented, pts, gkey, lkey)
            break
        if placed is None:
            return None
        oriented, pts, gkey, lkey = placed
        for r, c in pts:
            missing[r, c] = True
        _mark_gap_blocked(blocked, pts)
        gap_keys.add(gkey)
        local_keys.add(lkey)
        gaps.append({"mask": _mask_from_points(pts), "pts": pts, "local_key": lkey})

    frame = full_ring & ~missing
    if len(_components(frame)) != 1:
        return None

    colors = rng.sample(COLOR_POOL, 2 * n)
    piece_colors = colors[:n]
    output_colors = colors[n:]
    rng.shuffle(output_colors)

    grid = np.zeros((h, w), dtype=int)
    grid[fr0 : fr0 + fh, fc0 : fc0 + fw][frame] = GREY

    occupied = grid != BG
    occupied[legend_top:, :] = True

    piece_points = []
    order = list(range(n))
    rng.shuffle(order)
    for idx in order:
        pts = _place_piece(grid, occupied, gaps[idx]["mask"], piece_colors[idx], frame_bbox, legend_top, rng)
        if pts is None:
            return None
        piece_points.append(pts)

    entries = []
    for gap, color in zip(gaps, output_colors):
        grey_for_key = full_ring.copy()
        for r, c in gap["pts"]:
            grey_for_key[r, c] = False
        local = set(gap["pts"])
        for r, c in gap["pts"]:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < fh and 0 <= nc < fw and grey_for_key[nr, nc]:
                    local.add((nr, nc))
        tmpl = _mask_from_points(list(local))
        entries.append((tmpl, color, gap["pts"]))
    rng.shuffle(entries)

    total_w = sum(t.shape[1] + 2 for t, _c, _pts in entries) + len(entries) - 1
    if total_w > w - 2:
        return None
    c = rng.randint(1, w - total_w - 1)
    legend_mask = np.zeros_like(grid, dtype=bool)
    max_th = max(t.shape[0] for t, _c, _pts in entries)
    if max_th > legend_band:
        return None
    top = legend_top + rng.randint(0, legend_band - max_th)
    planned_colors = {}
    for tmpl, color, pts in entries:
        r0 = top + (max_th - tmpl.shape[0])
        _paste_mask(grid, tmpl, r0, c, GREY)
        tr, tc = np.nonzero(tmpl)
        legend_mask[r0 + tr, c + tc] = True
        right_rows = [int(r) for r in np.nonzero(tmpl[:, -1])[0]]
        sq_r = r0 + rng.choice(right_rows)
        sq_c = c + tmpl.shape[1]
        grid[sq_r, sq_c] = color
        legend_mask[sq_r, sq_c] = True
        planned_colors[tuple(sorted(pts))] = color
        c += tmpl.shape[1] + 3

    expected = grid.copy()
    expected[legend_mask] = BG
    for pts in piece_points:
        for r, c in pts:
            expected[r, c] = BG
    for gap, color in zip(gaps, output_colors):
        for rr, cc in gap["pts"]:
            expected[fr0 + rr, fc0 + cc] = color

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        try:
            grid = _build_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
