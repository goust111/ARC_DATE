from __future__ import annotations

from collections import deque

import numpy as np


DEPTH0_COLOR = 4  # yellow
DEPTH1_COLOR = 7  # orange
DEPTH2_COLOR = 2  # red
DEPTH3P_COLOR = 9  # maroon
OUTPUT_COLORS = (DEPTH0_COLOR, DEPTH1_COLOR, DEPTH2_COLOR, DEPTH3P_COLOR)

GEN_ATTEMPTS = 300
INPUT_COLORS = tuple(c for c in range(10) if c not in OUTPUT_COLORS)
SIDES = ("top", "bottom", "left", "right")
RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _local_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    return local, int(r0), int(c0)


def _depth_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    depths = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            is_boundary = False
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    is_boundary = True
                    break
            if is_boundary:
                depths[r, c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and depths[nr, nc] == -1:
                depths[nr, nc] = depths[r, c] + 1
                q.append((nr, nc))

    return depths


def _depth_to_color(depth: int) -> int:
    if depth <= 0:
        return DEPTH0_COLOR
    if depth == 1:
        return DEPTH1_COLOR
    if depth == 2:
        return DEPTH2_COLOR
    return DEPTH3P_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by 4-neighbor depth from its boundary.

    Boundary pixels have depth 0. Pixels one step inward have depth 1, etc.
    The recoloring is fixed:
    - depth 0 -> yellow (4)
    - depth 1 -> orange (7)
    - depth 2 -> red (2)
    - depth 3+ -> maroon (9)
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _background_color(g)
    out = g.copy()
    for comp in _components(g != bg):
        local, r0, c0 = _local_mask(comp)
        depths = _depth_map(local)
        rows, cols = np.nonzero(local)
        for lr, lc in zip(rows, cols):
            out[r0 + int(lr), c0 + int(lc)] = _depth_to_color(int(depths[int(lr), int(lc)]))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _trim(mask: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _attach_wing(mask: np.ndarray) -> None:
    r0, r1, c0, c1 = _bbox(mask)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    side = str(RNG.choice(SIDES))

    if side in ("top", "bottom"):
        ext = int(RNG.integers(1, 4))
        min_span = max(2, width // 2)
        max_span = min(mask.shape[1], width + 2)
        span = int(RNG.integers(min_span, max_span + 1))
        if span <= 0:
            return
        lo = max(0, c0 - 2)
        hi = min(mask.shape[1] - span, c1 - span + 1 + 2)
        if hi < lo:
            return
        left = int(RNG.integers(lo, hi + 1))
        if side == "top":
            top = r0 - ext
            if top < 0:
                return
        else:
            top = r1 + 1
            if top + ext > mask.shape[0]:
                return
        mask[top : top + ext, left : left + span] = True
        return

    ext = int(RNG.integers(1, 4))
    min_span = max(2, height // 2)
    max_span = min(mask.shape[0], height + 2)
    span = int(RNG.integers(min_span, max_span + 1))
    if span <= 0:
        return
    lo = max(0, r0 - 2)
    hi = min(mask.shape[0] - span, r1 - span + 1 + 2)
    if hi < lo:
        return
    top = int(RNG.integers(lo, hi + 1))
    if side == "left":
        left = c0 - ext
        if left < 0:
            return
    else:
        left = c1 + 1
        if left + ext > mask.shape[1]:
            return
    mask[top : top + span, left : left + ext] = True


def _make_object_mask(target_depth: int) -> np.ndarray:
    min_dim = 2 * target_depth + 1
    core_h = int(RNG.integers(min_dim, min_dim + 3))
    core_w = int(RNG.integers(min_dim, min_dim + 3))
    pad = 6
    canvas = np.zeros((core_h + 2 * pad + 4, core_w + 2 * pad + 4), dtype=bool)

    top = pad + int(RNG.integers(0, 3))
    left = pad + int(RNG.integers(0, 3))
    canvas[top : top + core_h, left : left + core_w] = True

    extra_wings = int(RNG.integers(1, 5))
    for _ in range(extra_wings):
        _attach_wing(canvas)

    obj = _trim(canvas)
    depths = _depth_map(obj)
    if int(depths[obj].max()) < target_depth:
        return _make_object_mask(target_depth)
    return obj


def _can_place(blocked: np.ndarray, obj: np.ndarray, top: int, left: int) -> bool:
    h, w = obj.shape
    return not np.any(blocked[top : top + h, left : left + w] & obj)


def _mark_blocked(blocked: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    h, w = obj.shape
    rows, cols = np.nonzero(obj)
    for r, c in zip(rows, cols):
        rr = top + int(r)
        cc = left + int(c)
        for nr in range(max(0, rr - 1), min(blocked.shape[0], rr + 2)):
            for nc in range(max(0, cc - 1), min(blocked.shape[1], cc + 2)):
                blocked[nr, nc] = True


def _place_objects(
    grid: np.ndarray, fg: int, objects: list[np.ndarray]
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    for obj in sorted(objects, key=lambda x: int(x.sum()), reverse=True):
        oh, ow = obj.shape
        candidates: list[tuple[int, int]] = []
        for top in range(1, h - oh):
            for left in range(1, w - ow):
                if _can_place(blocked, obj, top, left):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return False
        top, left = candidates[int(RNG.integers(0, len(candidates)))]
        grid[top : top + oh, left : left + ow][obj] = fg
        _mark_blocked(blocked, obj, top, left)
    return True


def _valid_grid(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    fg_colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    if len(fg_colors) != 1:
        return False
    if bg in OUTPUT_COLORS or fg_colors[0] in OUTPUT_COLORS:
        return False
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return False

    comps = _components(grid != bg)
    if not (2 <= len(comps) <= 4):
        return False

    has_deep_object = False
    for comp in comps:
        if len(comp) < 9:
            return False
        local, _, _ = _local_mask(comp)
        depths = _depth_map(local)
        if int(depths[local].max()) >= 3:
            has_deep_object = True
    return has_deep_object


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(22, 31))
    w = int(RNG.integers(22, 31))
    bg = int(RNG.choice(INPUT_COLORS))
    fg_choices = [c for c in INPUT_COLORS if c != bg]
    fg = int(RNG.choice(fg_choices))

    object_count = int(RNG.integers(2, 5))
    depths = [int(RNG.choice((3, 4)))]
    for _ in range(object_count - 1):
        depths.append(int(RNG.choice((1, 2, 2, 3, 4))))

    objects = [_make_object_mask(depth) for depth in depths]
    total_area = sum(int(obj.sum()) for obj in objects)
    if total_area >= int(0.42 * h * w):
        return None

    grid = np.full((h, w), bg, dtype=int)
    if not _place_objects(grid, fg, objects):
        return None
    if not _valid_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
