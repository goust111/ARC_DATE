from __future__ import annotations

import numpy as np

from arc_utils import bbox, paste
import random


_SOURCE_COLOR = 8
_BLOCK_COLOR = 1
_PAINT_COLOR = 4


def _solid_source_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g == _SOURCE_COLOR
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height > 1 and width > 1
    assert int(mask.sum()) == height * width
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return b


def _paint_ray(
    g: np.ndarray,
    out: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    fillable: set[int],
) -> None:
    H, W = map(int, g.shape)
    while 0 <= r < H and 0 <= c < W:
        value = int(g[r, c])
        if value == _BLOCK_COLOR:
            break
        assert value == _SOURCE_COLOR or value in fillable
        if value != _SOURCE_COLOR:
            out[r, c] = _PAINT_COLOR
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast yellow horizontal and vertical rays from the teal rectangle until blue blockers stop them.

    Input structure:
    - A black-background grid containing one solid teal rectangle.
    - Blue objects are blockers; other non-background objects may lie in the ray paths.
    - Yellow is absent from the input and is the output paint color.

    Rule:
    - From every row of the teal rectangle, cast rays left and right to the grid edge.
    - From every column of the teal rectangle, cast rays up and down to the grid edge.
    - A ray stops before a blue pixel; all crossed non-blue, non-teal pixels become yellow.

    Key insight:
    The teal rectangle defines a full cross-shaped projection, while blue objects are walls and red/background cells are simply paintable space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _PAINT_COLOR not in set(map(int, np.unique(g)))
    assert _BLOCK_COLOR in set(map(int, np.unique(g)))

    r0, c0, r1, c1 = _solid_source_bbox(g)
    fillable = set(map(int, np.unique(g))) - {_SOURCE_COLOR, _BLOCK_COLOR}

    out = g.copy()
    for r in range(r0, r1 + 1):
        _paint_ray(g, out, r, c0 - 1, 0, -1, fillable)
        _paint_ray(g, out, r, c1 + 1, 0, 1, fillable)

    for c in range(c0, c1 + 1):
        _paint_ray(g, out, r0 - 1, c, -1, 0, fillable)
        _paint_ray(g, out, r1 + 1, c, 1, 0, fillable)

    return out

# --------------- generator (att2) ---------------

# Input palette.
_BG, _BLUE, _RED, _YELLOW, _TEAL = 0, 1, 2, 4, 8
_REQUIRED_INPUT_COLORS = {_BG, _BLUE, _RED, _TEAL}

# Canvas and source rectangle bounds.
_GRID_HEIGHT = 19
_GRID_WIDTH_RANGE = (18, 19)
_SOURCE_SIZE_RANGE = (4, 7)
_SOURCE_EDGE_GAP, _SOURCE_MAX_TOP, _SOURCE_CENTER_JITTER = 4, 7, 2

# Object separation and blocker rectangle bounds.
_MIN_OBJECT_GAP, _RAY_OBJECT_GAP = 1, 1
_BLOCKER_COUNT_RANGE = (1, 4)
_BLOCKER_HEIGHTS = range(2, 9)
_BLOCKER_WIDTHS = range(2, 10)

# Passthrough red rectangle bounds.
_RED_COUNT_RANGE = (2, 6)
_RED_PATH_COUNT_MIN = 1
_RED_HEIGHTS = range(2, 7)
_RED_WIDTHS = range(2, 14)

# Retry budget for valid oracle-checked samples.
_MAX_GENERATION_ATTEMPTS = 300


def _fits(rect, rects, H, W):
    r0, c0, r1, c1 = rect
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    for or0, oc0, or1, oc1 in rects:
        if not (
            r1 < or0 - _MIN_OBJECT_GAP
            or r0 > or1 + _MIN_OBJECT_GAP
            or c1 < oc0 - _MIN_OBJECT_GAP
            or c0 > oc1 + _MIN_OBJECT_GAP
        ):
            return False
    return True


def _place(grid, rect, color):
    r0, c0, r1, c1 = rect
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), color, dtype=int)
    paste(grid, patch, r0, c0)


def _source_rect(H, W):
    h = random.randint(*_SOURCE_SIZE_RANGE)
    w = random.randint(*_SOURCE_SIZE_RANGE)
    r_hi = min(_SOURCE_MAX_TOP, H - h - _SOURCE_EDGE_GAP)
    r0 = random.randint(_SOURCE_EDGE_GAP, r_hi)

    c_lo, c_hi = _SOURCE_EDGE_GAP, W - w - _SOURCE_EDGE_GAP
    center = (W - w) // 2
    c_lo = max(c_lo, center - _SOURCE_CENTER_JITTER)
    c_hi = min(c_hi, center + _SOURCE_CENTER_JITTER)
    c0 = random.randint(c_lo, c_hi)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _choose_grouped(candidates):
    if not candidates:
        return None
    groups = {}
    for item in candidates:
        groups.setdefault(item[1], []).append(item[0])
    return random.choice(random.choice(list(groups.values())))


def _is_visible_blocker(grid, source, rect, axis, side):
    sr0, sc0, sr1, sc1 = source
    r0, c0, r1, c1 = rect
    if axis == "h":
        for r in range(max(sr0, r0), min(sr1, r1) + 1):
            if side == "left":
                if not np.any(grid[r, c1 + 1 : sc0] == _BLUE):
                    return True
            else:
                if not np.any(grid[r, sc1 + 1 : c0] == _BLUE):
                    return True
    else:
        for c in range(max(sc0, c0), min(sc1, c1) + 1):
            if side == "up":
                if not np.any(grid[r1 + 1 : sr0, c] == _BLUE):
                    return True
            else:
                if not np.any(grid[sr1 + 1 : r0, c] == _BLUE):
                    return True
    return False


def _blocker_candidates(grid, source, rects):
    H, W = grid.shape
    sr0, sc0, sr1, sc1 = source
    candidates = []

    for h in _BLOCKER_HEIGHTS:
        for w in _BLOCKER_WIDTHS:
            for r in range(max(0, sr0 - h + 1), min(sr1, H - h) + 1):
                for side, c_range in (
                    ("left", range(0, sc0 - w - _RAY_OBJECT_GAP + 1)),
                    ("right", range(sc1 + _RAY_OBJECT_GAP + 1, W - w + 1)),
                ):
                    for c in c_range:
                        rect = (r, c, r + h - 1, c + w - 1)
                        if _fits(rect, rects, H, W) and _is_visible_blocker(grid, source, rect, "h", side):
                            candidates.append((rect, ("h", side, h, w)))

            for c in range(max(0, sc0 - w + 1), min(sc1, W - w) + 1):
                for side, r_range in (
                    ("up", range(0, sr0 - h - _RAY_OBJECT_GAP + 1)),
                    ("down", range(sr1 + _RAY_OBJECT_GAP + 1, H - h + 1)),
                ):
                    for r in r_range:
                        rect = (r, c, r + h - 1, c + w - 1)
                        if _fits(rect, rects, H, W) and _is_visible_blocker(grid, source, rect, "v", side):
                            candidates.append((rect, ("v", side, h, w)))

    return candidates


def _place_blockers(grid, source, rects):
    """Place visible blue blockers; return False if any required choice fails."""
    for _ in range(random.randint(*_BLOCKER_COUNT_RANGE)):
        rect = _choose_grouped(_blocker_candidates(grid, source, rects))
        if rect is None:
            return False
        _place(grid, rect, _BLUE)
        rects.append(rect)
    return True


def _ray_mask(grid, source):
    H, W = grid.shape
    sr0, sc0, sr1, sc1 = source
    mask = np.zeros((H, W), dtype=bool)
    for r in range(sr0, sr1 + 1):
        for c in range(sc0 - 1, -1, -1):
            if grid[r, c] == _BLUE:
                break
            mask[r, c] = True
        for c in range(sc1 + 1, W):
            if grid[r, c] == _BLUE:
                break
            mask[r, c] = True
    for c in range(sc0, sc1 + 1):
        for r in range(sr0 - 1, -1, -1):
            if grid[r, c] == _BLUE:
                break
            mask[r, c] = True
        for r in range(sr1 + 1, H):
            if grid[r, c] == _BLUE:
                break
            mask[r, c] = True
    return mask


def _red_candidates(ray_mask, rects, want_path):
    H, W = ray_mask.shape
    candidates = []
    for h in _RED_HEIGHTS:
        for w in _RED_WIDTHS:
            for r in range(0, H - h + 1):
                for c in range(0, W - w + 1):
                    rect = (r, c, r + h - 1, c + w - 1)
                    if not _fits(rect, rects, H, W):
                        continue
                    in_path = bool(np.any(ray_mask[r : r + h, c : c + w]))
                    if in_path == want_path:
                        candidates.append((rect, (h, w)))
    return candidates


def _place_red_objects(grid, ray_mask, rects):
    """Place red rectangles with the preselected path/non-path role mix."""
    red_count = random.randint(*_RED_COUNT_RANGE)
    path_count = random.randint(_RED_PATH_COUNT_MIN, red_count - 1)
    red_roles = [True] * path_count + [False] * (red_count - path_count)
    random.shuffle(red_roles)
    for want_path in red_roles:
        rect = _choose_grouped(_red_candidates(ray_mask, rects, want_path))
        if rect is None:
            return False
        _place(grid, rect, _RED)
        rects.append(rect)
    return True


def _gen_attempt():
    W = random.randint(*_GRID_WIDTH_RANGE)
    grid = np.zeros((_GRID_HEIGHT, W), dtype=int)

    source = _source_rect(*grid.shape)
    rects = [source]
    _place(grid, source, _TEAL)
    if not _place_blockers(grid, source, rects):
        return None

    ray_mask = _ray_mask(grid, source)
    if not _place_red_objects(grid, ray_mask, rects):
        return None

    colors = set(map(int, np.unique(grid)))
    if colors != _REQUIRED_INPUT_COLORS or _YELLOW in colors:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        if out.shape == inp.shape and not np.array_equal(out, inp):
            return inp
    raise RuntimeError("failed to generate a valid 6ffe8f07 input")
