from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


def _occupied_rows(grid: np.ndarray) -> np.ndarray:
    rows = np.where(np.any(grid != BG, axis=1))[0]
    assert rows.size > 0, "expected one non-empty sheared band"
    assert int(rows[-1] - rows[0] + 1) == int(rows.size), "band rows must be contiguous"
    return rows


def _rectify_band(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    rows = _occupied_rows(g)
    top = int(rows[0])
    bottom = int(rows[-1])
    h = int(rows.size)
    assert h >= 2

    firsts: list[int] = []
    lasts: list[int] = []
    for r in rows:
        cols = np.where(g[int(r)] != BG)[0]
        assert cols.size > 0
        firsts.append(int(cols[0]))
        lasts.append(int(cols[-1]))

    diffs = [firsts[i + 1] - firsts[i] for i in range(h - 1)]
    assert diffs and len(set(diffs)) == 1, "left edge must be one diagonal"
    k = int(diffs[0])
    assert k > 0, "expected a positive row-shear step"

    left = min(firsts)
    right = max(lasts)
    band = g[top : bottom + 1, left : right + 1]
    band_w = int(band.shape[1])
    rect_w = band_w - k * (h - 1)
    assert rect_w > 0, "invalid band width for recovered shear"

    rectified = np.full((h, band_w), BG, dtype=int)
    for i in range(h):
        local_start = int(firsts[i] - left)
        local_end = int(lasts[i] - left)
        assert local_start == k * i, "left edge slope must match the shear step"
        assert local_end == local_start + rect_w - 1, "right edge must be consistent across rows"
        rectified[i, : band_w - local_start] = band[i, local_start:]

    assert np.all(rectified[:, rect_w:] == BG), "rectified canvas should only have trailing background"
    return rectified[:, :rect_w].copy(), k


def _period_consistent(rect: np.ndarray, q: int) -> bool:
    h, _w = rect.shape
    step = int(q)
    if not 1 <= step < h:
        return False
    for r in range(h - step):
        a = rect[r]
        b = rect[r + step]
        mask = (a != BG) & (b != BG)
        if np.any(a[mask] != b[mask]):
            return False
    return True


def _minimal_vertical_period(rect: np.ndarray) -> int:
    h, _w = rect.shape
    for q in range(1, h):
        if _period_consistent(rect, q):
            return q
    raise AssertionError("no valid vertical period found")


def _fill_by_period(rect: np.ndarray, q: int) -> np.ndarray:
    out = np.array(rect, copy=True)
    h, w = out.shape
    for c in range(w):
        for phase in range(q):
            values = [int(out[r, c]) for r in range(phase, h, q) if int(out[r, c]) != BG]
            assert values, "every periodic chain must keep at least one visible witness"
            color = values[0]
            assert all(v == color for v in values), "one periodic chain cannot carry multiple colors"
            for r in range(phase, h, q):
                if int(out[r, c]) == BG:
                    out[r, c] = color
    assert not np.any(out == BG), "all holes must be fillable from the recovered period"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo a row-sheared periodic band, then repair its black holes from the hidden vertical period.

    Input structure:
    - A black background contains one colored parallelogram band whose rows come from an HxW rectangle shifted right by a fixed step each row.
    - The band edges stay intact, while some interior cells are punched out to black.
    - In the upright rectangle, rows repeat with one minimal vertical period q < H.

    Rule:
    - Read the left-edge slope to recover the shear step k, then shift row i left by k*i to undo the shear.
    - Crop the rectified result back to the original HxW rectangle.
    - Find the smallest vertical period consistent with the visible cells, and fill every black hole from same-column cells spaced by that period.

    Key insight:
    - The repetition direction is diagonal in the input but vertical after rectification; only after undoing the shear does the minimal period become directly visible and usable for repair.
    """
    rect_with_holes, _k = _rectify_band(np.asarray(grid, dtype=int))
    q = _minimal_vertical_period(rect_with_holes)
    return _fill_by_period(rect_with_holes, q)


# --------------- generator ---------------

MIN_Q = 2
MAX_Q = 4
MIN_W = 5
MAX_W = 12
MAX_K = 3
MAX_EXTRA_MARGIN = 4
GEN_ATTEMPTS = 500
ROW_ATTEMPTS = 200


def _has_short_horizontal_period(row: np.ndarray) -> bool:
    w = int(row.size)
    for step in range(1, w):
        if np.array_equal(row[:-step], row[step:]):
            return True
    return False


def _sample_row(width: int, palette: list[int]) -> np.ndarray:
    row = np.empty(width, dtype=int)
    mode = random.choice(("runs", "runs", "noise"))
    if mode == "noise":
        row[:] = np.random.choice(palette, size=width)
    else:
        cut_count = random.randint(1, min(width - 1, 4))
        cuts = sorted(random.sample(range(1, width), cut_count))
        starts = [0] + cuts
        ends = cuts + [width]
        last = None
        for c0, c1 in zip(starts, ends):
            choices = [color for color in palette if color != last]
            color = random.choice(choices)
            row[c0:c1] = color
            last = color
        for _ in range(random.randint(1, 2)):
            idx = random.randrange(width)
            choices = [color for color in palette if color != int(row[idx])]
            row[idx] = random.choice(choices)
    return row


def _generate_prototype_rows(q: int, width: int, palette: list[int]) -> list[np.ndarray] | None:
    rows: list[np.ndarray] = []
    min_distance = max(2, width // 4)
    for _ in range(ROW_ATTEMPTS):
        if len(rows) == q:
            return rows
        row = _sample_row(width, palette)
        if len(set(map(int, row))) < 2:
            continue
        if _has_short_horizontal_period(row):
            continue
        if any(np.array_equal(row, prev) for prev in rows):
            continue
        if any(int(np.count_nonzero(row != prev)) < min_distance for prev in rows):
            continue
        rows.append(row)
    return None


def _repeat_rows(prototypes: list[np.ndarray], height: int) -> np.ndarray:
    q = len(prototypes)
    return np.stack([prototypes[r % q] for r in range(height)], axis=0).astype(int)


def _sample_holes(rect: np.ndarray, q: int) -> np.ndarray | None:
    h, w = rect.shape
    if h < 3 or w < 3:
        return None

    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    chains_left = {(phase, c): len(range(phase, h, q)) for phase in range(q) for c in range(w)}
    holes = np.zeros((h, w), dtype=bool)

    interior_area = (h - 2) * (w - 2)
    min_holes = max(3, interior_area // 10)
    max_holes = min(12, max(min_holes, interior_area // 4))
    target = random.randint(min_holes, max_holes)

    for r, c in candidates:
        if int(holes.sum()) >= target:
            break
        key = (r % q, c)
        if chains_left[key] <= 1:
            continue
        if random.random() < 0.55:
            holes[r, c] = True
            chains_left[key] -= 1
            if random.random() < 0.30:
                neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                random.shuffle(neighbors)
                for rr, cc in neighbors:
                    if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
                        continue
                    if holes[rr, cc]:
                        continue
                    key2 = (rr % q, cc)
                    if chains_left[key2] <= 1:
                        continue
                    holes[rr, cc] = True
                    chains_left[key2] -= 1
                    break

    if int(holes.sum()) < min_holes:
        return None
    return holes


def _apply_holes(rect: np.ndarray, holes: np.ndarray) -> np.ndarray:
    out = np.array(rect, copy=True)
    out[holes] = BG
    return out


def _shear_rect(rect: np.ndarray, k: int) -> np.ndarray:
    h, w = rect.shape
    out = np.zeros((h, w + k * (h - 1)), dtype=int)
    for r in range(h):
        out[r, k * r : k * r + w] = rect[r]
    return out


def _place_on_canvas(band: np.ndarray) -> np.ndarray:
    h, w = band.shape
    extra_h = random.randint(0, min(MAX_EXTRA_MARGIN, 30 - h))
    extra_w = random.randint(0, min(MAX_EXTRA_MARGIN, 30 - w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    canvas = np.zeros((h + extra_h, w + extra_w), dtype=int)
    canvas[top : top + h, left : left + w] = band
    return canvas


def _nontrivial_hole_exists(rect: np.ndarray, holes: np.ndarray) -> bool:
    h, w = rect.shape
    for r, c in np.argwhere(holes):
        rr = int(r)
        cc = int(c)
        if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
            continue
        color = int(rect[rr, cc])
        left = int(rect[rr, cc - 1])
        right = int(rect[rr, cc + 1])
        if left != color or right != color:
            return True
    return False


def _visible_uniquely_reveals_period(rect_with_holes: np.ndarray, q: int) -> bool:
    return all(not _period_consistent(rect_with_holes, d) for d in range(1, q))


def _input_bbox_has_no_vertical_period(inp: np.ndarray) -> bool:
    rows = _occupied_rows(inp)
    cols = np.where(np.any(inp != BG, axis=0))[0]
    bbox = inp[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]
    return all(not _period_consistent(bbox, d) for d in range(1, bbox.shape[0]))


def _sample_dimensions() -> tuple[int, int, int, int] | None:
    q = random.randint(MIN_Q, MAX_Q)
    height_lo = 2 * q
    height_hi = min(10, 2 * q + 3)
    if height_lo > height_hi:
        return None
    h = random.randint(height_lo, height_hi)
    k_options = [k for k in range(1, MAX_K + 1) if MIN_W + k * (h - 1) <= 30]
    if not k_options:
        return None
    k = random.choice(k_options)
    max_w = min(MAX_W, 30 - k * (h - 1))
    if max_w < MIN_W:
        return None
    w = random.randint(MIN_W, max_w)
    return h, w, q, k


def _gen_attempt() -> np.ndarray | None:
    dims = _sample_dimensions()
    if dims is None:
        return None
    h, w, q, k = dims

    palette_size = random.randint(3, min(6, len(PALETTE)))
    palette = random.sample(list(PALETTE), palette_size)
    prototypes = _generate_prototype_rows(q, w, palette)
    if prototypes is None:
        return None

    rect = _repeat_rows(prototypes, h)
    if len(set(map(int, rect.ravel()))) < 3:
        return None

    holes = _sample_holes(rect, q)
    if holes is None:
        return None
    if len(set(int(r) for r, _c in np.argwhere(holes))) < 2:
        return None
    if len(set(int(c) for _r, c in np.argwhere(holes))) < 2:
        return None
    if not _nontrivial_hole_exists(rect, holes):
        return None

    rect_with_holes = _apply_holes(rect, holes)
    if not _visible_uniquely_reveals_period(rect_with_holes, q):
        return None
    if not np.array_equal(_fill_by_period(rect_with_holes, q), rect):
        return None

    sheared = _shear_rect(rect_with_holes, k)
    inp = _place_on_canvas(sheared)

    if not _input_bbox_has_no_vertical_period(inp):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, rect):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sheared-period repair puzzle")
