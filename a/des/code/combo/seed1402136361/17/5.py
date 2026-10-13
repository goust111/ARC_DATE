from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))

# Generator tuning.
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
BORDER_MARGIN = 1
OBJECT_CLEARANCE = 2
MAX_FOREGROUND_RATIO = 0.38

ALL_MODES = ("rect", "notch", "L", "T", "plus", "union")
IRREGULAR_MODES = ("notch", "L", "T", "plus", "union")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _outside_background(bg_mask: np.ndarray) -> np.ndarray:
    bg = np.asarray(bg_mask, dtype=bool)
    h, w = bg.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((int(nr), int(nc)))

    return outside


def _outer_boundary_mask(mask: np.ndarray) -> np.ndarray:
    obj = np.asarray(mask, dtype=bool)
    h, w = obj.shape
    outside = _outside_background(~obj)
    boundary = np.zeros((h, w), dtype=bool)

    rows, cols = np.where(obj)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        on_boundary = r == 0 or c == 0 or r == h - 1 or c == w - 1
        if not on_boundary:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not obj[nr, nc] and outside[nr, nc]:
                    on_boundary = True
                    break
        if on_boundary:
            boundary[int(r), int(c)] = True

    return boundary


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("mask must contain at least one filled cell")
    return np.asarray(
        mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1],
        dtype=bool,
    )


def _is_connected(mask: np.ndarray) -> bool:
    obj = np.asarray(mask, dtype=bool)
    rows, cols = np.where(obj)
    if len(rows) == 0:
        return False

    start = (int(rows[0]), int(cols[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    h, w = obj.shape

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and obj[nr, nc] and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                queue.append((int(nr), int(nc)))

    return len(seen) == int(obj.sum())


def _has_hole(mask: np.ndarray) -> bool:
    obj = np.asarray(mask, dtype=bool)
    outside = _outside_background(~obj)
    return bool(np.any((~obj) & (~outside)))


def _random_symmetry(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(_crop_mask(out), dtype=bool)


def _rect_mask() -> np.ndarray:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    return np.ones((h, w), dtype=bool)


def _notched_rect_mask() -> np.ndarray:
    h = random.randint(6, 11)
    w = random.randint(6, 11)
    mask = np.ones((h, w), dtype=bool)
    side = random.choice(("top", "bottom", "left", "right"))

    if side in ("top", "bottom"):
        notch_h = random.randint(1, h - 3)
        notch_w = random.randint(1, w - 2)
        c0 = random.randint(1, w - notch_w - 1)
        if side == "top":
            mask[:notch_h, c0 : c0 + notch_w] = False
        else:
            mask[h - notch_h :, c0 : c0 + notch_w] = False
    else:
        notch_w = random.randint(1, w - 3)
        notch_h = random.randint(1, h - 2)
        r0 = random.randint(1, h - notch_h - 1)
        if side == "left":
            mask[r0 : r0 + notch_h, :notch_w] = False
        else:
            mask[r0 : r0 + notch_h, w - notch_w :] = False

    return _random_symmetry(mask)


def _l_mask() -> np.ndarray:
    vert_h = random.randint(6, 10)
    horiz_w = random.randint(6, 10)
    vert_w = random.randint(2, 4)
    horiz_h = random.randint(2, 4)

    mask = np.zeros((vert_h, horiz_w), dtype=bool)
    mask[:, :vert_w] = True
    mask[vert_h - horiz_h :, :] = True
    return _random_symmetry(mask)


def _t_mask() -> np.ndarray:
    cap_h = random.randint(2, 4)
    width = random.randint(7, 11)
    stem_w = random.randint(2, min(4, width - 2))
    stem_h = random.randint(5, 9)
    stem_c = random.randint(1, width - stem_w - 1)

    mask = np.zeros((cap_h + stem_h - 1, width), dtype=bool)
    mask[:cap_h, :] = True
    mask[cap_h - 1 :, stem_c : stem_c + stem_w] = True
    return _random_symmetry(mask)


def _plus_mask() -> np.ndarray:
    h = random.randint(7, 11)
    w = random.randint(7, 11)
    band_h = random.randint(2, 4)
    band_w = random.randint(2, 4)
    row0 = random.randint(1, h - band_h - 1)
    col0 = random.randint(1, w - band_w - 1)

    mask = np.zeros((h, w), dtype=bool)
    mask[row0 : row0 + band_h, :] = True
    mask[:, col0 : col0 + band_w] = True
    return _random_symmetry(mask)


def _rect_touches(mask: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> bool:
    h, w = mask.shape
    for r in range(r0, r0 + rh):
        for c in range(c0, c0 + rw):
            if mask[r, c]:
                return True
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    return True
    return False


def _union_mask() -> np.ndarray:
    h = random.randint(8, 12)
    w = random.randint(8, 12)
    mask = np.zeros((h, w), dtype=bool)

    first_h = random.randint(3, min(6, h))
    first_w = random.randint(3, min(6, w))
    first_r = random.randint(0, h - first_h)
    first_c = random.randint(0, w - first_w)
    mask[first_r : first_r + first_h, first_c : first_c + first_w] = True

    n_rects = random.randint(2, 5)
    for _ in range(n_rects - 1):
        for _attempt in range(80):
            rh = random.randint(2, min(6, h))
            rw = random.randint(2, min(6, w))
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            if not _rect_touches(mask, r0, c0, rh, rw):
                continue
            updated = mask.copy()
            updated[r0 : r0 + rh, c0 : c0 + rw] = True
            if np.array_equal(updated, mask):
                continue
            mask = updated
            break

    return _random_symmetry(mask)


def _make_shape(mode: str) -> np.ndarray:
    if mode == "rect":
        return _rect_mask()
    if mode == "notch":
        return _notched_rect_mask()
    if mode == "L":
        return _l_mask()
    if mode == "T":
        return _t_mask()
    if mode == "plus":
        return _plus_mask()
    if mode == "union":
        return _union_mask()
    raise ValueError(f"unknown mode: {mode}")


def _valid_object_mask(mask: np.ndarray) -> bool:
    obj = _crop_mask(np.asarray(mask, dtype=bool))
    h, w = obj.shape
    if h < 3 or w < 3:
        return False
    if int(obj.sum()) < 12:
        return False
    if not _is_connected(obj):
        return False
    if _has_hole(obj):
        return False

    boundary = _outer_boundary_mask(obj)
    interior = obj & (~boundary)
    if not bool(np.any(interior)):
        return False
    if int(boundary.sum()) >= int(obj.sum()):
        return False
    return True


def _sample_object_mask(modes: tuple[str, ...]) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        mode = random.choice(modes)
        mask = _make_shape(mode)
        if _valid_object_mask(mask):
            return mask
    return None


def _mark_forbidden(forbidden: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = forbidden.shape
    rows, cols = np.where(mask)
    for lr, lc in zip(rows.tolist(), cols.tolist(), strict=False):
        gr = r0 + int(lr)
        gc = c0 + int(lc)
        r1 = max(0, gr - OBJECT_CLEARANCE)
        r2 = min(h, gr + OBJECT_CLEARANCE + 1)
        c1 = max(0, gc - OBJECT_CLEARANCE)
        c2 = min(w, gc + OBJECT_CLEARANCE + 1)
        forbidden[r1:r2, c1:c2] = True


def _place_object(
    grid: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if h < mh + 2 * BORDER_MARGIN or w < mw + 2 * BORDER_MARGIN:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(BORDER_MARGIN, h - mh - BORDER_MARGIN)
        c0 = random.randint(BORDER_MARGIN, w - mw - BORDER_MARGIN)
        if bool(np.any(forbidden[r0 : r0 + mh, c0 : c0 + mw] & mask)):
            continue
        window = grid[r0 : r0 + mh, c0 : c0 + mw]
        window[mask] = int(color)
        _mark_forbidden(forbidden, mask, r0, c0)
        return True

    return False


def _component_count_by_color(grid: np.ndarray, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            count += 1
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))

    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only each object's corner-supported outline by redrawing its outer boundary.

    Input structure:
    - The background is dominant.
    - Foreground objects are separated monochrome 4-connected solid orthogonal polygons.

    Rule:
    - Treat each object's background-reachable exterior as the relevant boundary.
    - The described "keep corner pixels, erase the rest, then reconnect adjacent corners"
      reconstructs exactly this one-pixel outer shell.
    - Paint only those shell pixels in the object's color and leave everything else as bg.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    bg_mask = g == int(bg)
    outside = _outside_background(bg_mask)
    h, w = g.shape

    out = np.full_like(g, int(bg))
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            keep = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if not keep:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if int(g[nr, nc]) == int(bg) and outside[nr, nc]:
                        keep = True
                        break
            if keep:
                out[r, c] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)

    bg = random.choice(PALETTE)
    colors = [color for color in PALETTE if color != bg]
    random.shuffle(colors)
    obj_colors = colors[:n_objects]

    grid = np.full((h, w), int(bg), dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    object_modes = [random.choice(IRREGULAR_MODES)]
    while len(object_modes) < n_objects:
        object_modes.append(random.choice(ALL_MODES))
    random.shuffle(object_modes)

    total_fg = 0
    for color, mode in zip(obj_colors, object_modes, strict=False):
        modes = (mode,)
        mask = _sample_object_mask(modes)
        if mask is None:
            return None
        if not _place_object(grid, forbidden, mask, color):
            return None
        total_fg += int(mask.sum())

    if total_fg >= int(h * w * MAX_FOREGROUND_RATIO):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    if _component_count_by_color(grid, bg) != _component_count_by_color(out, bg):
        return None

    if not bool(np.any((grid != bg) & (out == bg))):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
