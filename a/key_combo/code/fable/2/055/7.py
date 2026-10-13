from __future__ import annotations

from collections import deque
import random

import numpy as np


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400
GRID_MIN = 14
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
FRAME_OUTER_H_RANGE = (4, 6)
FRAME_OUTER_W_RANGE = (4, 6)
FRAME_MARGIN = 1
FRAME_GAP = 1
PERIOD_H_RANGE = (2, 4)
PERIOD_W_RANGE = (2, 4)
TILE_COLOR_COUNT_RANGE = (3, 5)
TAMPER_DENSITY_RANGE = (0.20, 0.45)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _tile_fill(shape: tuple[int, int], tile: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    r_off, c_off = offset
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[(r - r_off) % th, (c - c_off) % tw])
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_hollow_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    fh = r1 - r0 + 1
    fw = c1 - c0 + 1
    if fh < 3 or fw < 3:
        return False
    border_area = 2 * fh + 2 * fw - 4
    if int(coords.shape[0]) != border_area:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.ones((fh, fw), dtype=bool)
    border[1:-1, 1:-1] = False
    if not np.all(patch[border] == int(color)):
        return False
    if np.any(patch[1:-1, 1:-1] == int(color)):
        return False
    return True


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for color in np.unique(g):
        for coords in _components(g == int(color)):
            if not _is_hollow_frame(g, int(color), coords):
                continue
            bbox = _bbox_coords(coords)
            fh = bbox[2] - bbox[0] + 1
            fw = bbox[3] - bbox[1] + 1
            groups.setdefault((int(color), fh, fw), []).append(bbox)

    candidates = [sorted(bboxes) for bboxes in groups.values() if len(bboxes) >= 2]
    assert len(candidates) == 1, "expected one frame group"
    return candidates[0]


def _period_candidates(
    grid: np.ndarray,
    observed_mask: np.ndarray,
    *,
    max_area: int | None = None,
) -> list[tuple[int, int, dict[tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    rs, cs = np.nonzero(observed_mask)
    vals = g[rs, cs].astype(int)
    h, w = g.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    rmods = {ph: rs % ph for ph in range(1, h + 1)}
    cmods = {pw: cs % pw for pw in range(1, w + 1)}
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if max_area is not None and ph * pw > max_area:
                continue
            residue = np.full((ph, pw), -1, dtype=int)
            rr = rmods[ph]
            cc = cmods[pw]
            ok = True
            for i, value in enumerate(vals):
                prev = int(residue[rr[i], cc[i]])
                if prev == -1:
                    residue[rr[i], cc[i]] = int(value)
                elif prev != int(value):
                    ok = False
                    break
            if ok:
                mapping = {
                    (r, c): int(residue[r, c])
                    for r in range(ph)
                    for c in range(pw)
                    if int(residue[r, c]) != -1
                }
                candidates.append((ph, pw, mapping))
    return candidates


def _infer_period_from_observed(grid: np.ndarray, observed_mask: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates = _period_candidates(grid, observed_mask)
    assert candidates, "no compatible period"
    min_area = min(ph * pw for ph, pw, _mapping in candidates)
    best = [(ph, pw, mapping) for ph, pw, mapping in candidates if ph * pw == min_area]
    assert len(best) == 1, "period must be uniquely determined"
    ph, pw, mapping = best[0]
    assert len(mapping) == ph * pw, "every residue must be visible outside the frames"
    tile = np.empty((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            tile[r, c] = int(mapping[(r, c)])
    return ph, pw, tile


def _interior_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _interior_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    return _crop_bbox(grid, _interior_bbox(bbox))


def _frame_order(bboxes: list[tuple[int, int, int, int]]) -> tuple[str, list[int]]:
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in bboxes]
    row_span = max(r for r, _c in centers) - min(r for r, _c in centers)
    col_span = max(c for _r, c in centers) - min(c for _r, c in centers)
    if col_span > row_span:
        order = sorted(range(len(bboxes)), key=lambda i: (centers[i][1], centers[i][0]))
        return "horizontal", order
    order = sorted(range(len(bboxes)), key=lambda i: (centers[i][0], centers[i][1]))
    return "vertical", order


def _pack_windows(
    windows: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    *,
    axis: str | None = None,
    order: list[int] | None = None,
) -> np.ndarray:
    assert windows and len(windows) == len(bboxes)
    shapes = {window.shape for window in windows}
    assert len(shapes) == 1
    wh, ww = next(iter(shapes))
    if axis is None or order is None:
        axis2, order2 = _frame_order(bboxes)
        axis = axis if axis is not None else axis2
        order = order if order is not None else order2
    if axis == "horizontal":
        out = np.empty((wh, ww * len(order)), dtype=int)
        for k, idx in enumerate(order):
            _paste(out, windows[idx], 0, k * ww)
        return out
    out = np.empty((wh * len(order), ww), dtype=int)
    for k, idx in enumerate(order):
        _paste(out, windows[idx], k * wh, 0)
    return out


def _repair_windows(
    bboxes: list[tuple[int, int, int, int]],
    ph: int,
    pw: int,
    tile: np.ndarray,
) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for bbox in bboxes:
        r0, c0, r1, c1 = _interior_bbox(bbox)
        patch = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                patch[rr - r0, cc - c0] = int(tile[rr % ph, cc % pw])
        windows.append(patch)
    return windows


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    mask = np.ones(tile.shape, dtype=bool)
    ph, pw, _ = _infer_period_from_observed(tile, mask)
    return ph, pw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each framed viewport from the global period, then bind the restored interiors into one strip.

    Input structure:
    - The whole grid is a rectangular tiling of one periodic color pattern.
    - Several non-overlapping same-color hollow rectangular frames of equal size sit on top of that tiling.
    - Cells inside each frame may be tampered, while the pattern outside every frame is intact.

    Rule:
    - Detect the frame rectangles and treat every full frame bbox as hidden from period inference.
    - Infer the unique minimal 2D period from the untampered cells outside those bboxes.
    - Restore each frame interior to the color predicted by that global period.
    - If frame centers spread more by column than row, sort left-to-right and concatenate interiors horizontally; otherwise sort top-to-bottom and concatenate vertically.

    Key insight:
    - The frames are only viewports: the visible content must be repaired from the exterior global tiling before any crop-and-pack step can be correct.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    bboxes = _find_frames(g)
    observed = np.ones(g.shape, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        observed[r0 : r1 + 1, c0 : c1 + 1] = False

    ph, pw, tile = _infer_period_from_observed(g, observed)
    windows = _repair_windows(bboxes, ph, pw, tile)
    return _pack_windows(windows, bboxes)


def _sample_minimal_tile(ph: int, pw: int, frame_color: int) -> np.ndarray | None:
    colors = random.sample([c for c in PALETTE if c != frame_color], random.randint(*TILE_COLOR_COUNT_RANGE))
    for _ in range(200):
        tile = np.array([[random.choice(colors) for _ in range(pw)] for _ in range(ph)], dtype=int)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _minimal_period(tile) != (ph, pw):
            continue
        return tile
    return None


def _gen_spaced_positions(size: int, count: int, obj_size: int) -> list[int] | None:
    lo = FRAME_MARGIN
    hi = size - obj_size - FRAME_MARGIN
    step = obj_size + FRAME_GAP
    slack = hi - lo - (count - 1) * step
    if slack < 0:
        return None
    offsets = sorted(random.choices(range(slack + 1), k=count))
    return [lo + i * step + offsets[i] for i in range(count)]


def _gen_limited_positions(size: int, count: int, obj_size: int, max_span: int) -> list[int] | None:
    lo = FRAME_MARGIN
    hi = size - obj_size - FRAME_MARGIN
    if hi < lo:
        return None
    span_limit = min(max_span, hi - lo)
    if span_limit < 0:
        return None
    span = random.randint(0, span_limit)
    base = random.randint(lo, hi - span)
    return [base + random.randint(0, span) for _ in range(count)]


def _sample_layout(height: int, width: int, fh: int, fw: int, n_frames: int, axis: str) -> list[tuple[int, int, int, int]] | None:
    if axis == "horizontal":
        cols = _gen_spaced_positions(width, n_frames, fw)
        if cols is None:
            return None
        col_span = max(cols) - min(cols)
        rows = _gen_limited_positions(height, n_frames, fh, col_span - 1)
        if rows is None:
            return None
        return [(r, c, r + fh - 1, c + fw - 1) for r, c in zip(rows, cols)]

    rows = _gen_spaced_positions(height, n_frames, fh)
    if rows is None:
        return None
    row_span = max(rows) - min(rows)
    cols = _gen_limited_positions(width, n_frames, fw, row_span - 1)
    if cols is None:
        return None
    return [(r, c, r + fh - 1, c + fw - 1) for r, c in zip(rows, cols)]


def _render_input(
    base: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    frame_color: int,
) -> np.ndarray | None:
    g = base.copy()
    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        g[r0, c0 : c1 + 1] = frame_color
        g[r1, c0 : c1 + 1] = frame_color
        g[r0 : r1 + 1, c0] = frame_color
        g[r0 : r1 + 1, c1] = frame_color

        inner = _interior_bbox(bbox)
        ir0, ic0, ir1, ic1 = inner
        coords = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
        random.shuffle(coords)
        area = len(coords)
        max_tampered = max(1, int(np.ceil(area * TAMPER_DENSITY_RANGE[1])))
        min_tampered = max(1, int(np.ceil(area * TAMPER_DENSITY_RANGE[0])))
        tamper_count = random.randint(min_tampered, max_tampered)
        for r, c in coords[:tamper_count]:
            expected = int(base[r, c])
            choices = [color for color in PALETTE if color not in (expected, frame_color)]
            g[r, c] = random.choice(choices)
    return g


def _validate_instance(
    inp: np.ndarray,
    base: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    true_period: tuple[int, int],
) -> bool:
    ph, pw = true_period
    observed = np.ones(inp.shape, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        observed[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates = _period_candidates(inp, observed, max_area=ph * pw)
    if not candidates:
        return False
    min_area = min(ph0 * pw0 for ph0, pw0, _mapping in candidates)
    best = [(ph0, pw0, mapping) for ph0, pw0, mapping in candidates if ph0 * pw0 == min_area]
    if len(best) != 1:
        return False
    if best[0][0:2] != (ph, pw):
        return False
    if len(best[0][2]) != ph * pw:
        return False

    true_windows = [_interior_patch(base, bbox) for bbox in bboxes]
    raw_windows = [_interior_patch(inp, bbox) for bbox in bboxes]

    if any(np.array_equal(raw, truth) for raw, truth in zip(raw_windows, true_windows)):
        return False
    if len({window.tobytes() for window in true_windows}) != len(true_windows):
        return False

    axis, order = _frame_order(bboxes)
    row_order = sorted(range(len(bboxes)), key=lambda i: ((bboxes[i][0] + bboxes[i][2]) / 2.0, (bboxes[i][1] + bboxes[i][3]) / 2.0))
    col_order = sorted(range(len(bboxes)), key=lambda i: ((bboxes[i][1] + bboxes[i][3]) / 2.0, (bboxes[i][0] + bboxes[i][2]) / 2.0))
    if axis == "horizontal":
        if order != col_order or order == row_order:
            return False
        wrong = _pack_windows(true_windows, bboxes, axis="vertical", order=row_order)
    else:
        if order != row_order or order == col_order:
            return False
        wrong = _pack_windows(true_windows, bboxes, axis="horizontal", order=col_order)

    expected = _pack_windows(true_windows, bboxes)
    raw = _pack_windows(raw_windows, bboxes)
    if np.array_equal(expected, raw):
        return False
    if expected.shape == wrong.shape and np.array_equal(expected, wrong):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    fh = random.randint(*FRAME_OUTER_H_RANGE)
    fw = random.randint(*FRAME_OUTER_W_RANGE)
    ih = fh - 2
    iw = fw - 2
    axis = random.choice(("horizontal", "vertical"))

    ph = random.randint(*PERIOD_H_RANGE)
    pw = random.randint(*PERIOD_W_RANGE)
    frame_color = random.choice(PALETTE)
    tile = _sample_minimal_tile(ph, pw, frame_color)
    if tile is None:
        return None

    min_h = max(GRID_MIN, fh + 2 * FRAME_MARGIN, ph * 3)
    min_w = max(GRID_MIN, fw + 2 * FRAME_MARGIN, pw * 3)
    if axis == "horizontal":
        min_w = max(min_w, 2 * FRAME_MARGIN + n_frames * fw + (n_frames - 1) * FRAME_GAP)
    else:
        min_h = max(min_h, 2 * FRAME_MARGIN + n_frames * fh + (n_frames - 1) * FRAME_GAP)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, min(GRID_MAX, min_h + 6))
    width = random.randint(min_w, min(GRID_MAX, min_w + 6))
    bboxes = _sample_layout(height, width, fh, fw, n_frames, axis)
    if bboxes is None:
        return None

    phase = (random.randrange(ph), random.randrange(pw))
    base = _tile_fill((height, width), tile, phase)
    inp = _render_input(base, bboxes, frame_color)
    if inp is None:
        return None

    if max(n_frames * iw if axis == "horizontal" else n_frames * ih, iw if axis == "vertical" else ih) > 30:
        return None

    if not _validate_instance(inp, base, bboxes, (ph, pw)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid viewport-repair instance")
