import numpy as np
from collections import deque


BLACK = 0
RED = 2
CURVE_COLORS = (1, 3, 4, 6, 7, 8, 9)
OPEN_SIDES = ("top", "right", "bottom", "left")
_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _neighbors4(shape, r, c):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_from_mask(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochromatic_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(grid.shape, rr, cc):
                    if grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _inside_mask(barrier):
    h, w = barrier.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(barrier.shape, r, c):
            if not barrier[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~outside) & (~barrier)


def _curve_info(component, shape):
    coords = component["coords"]
    color = int(component["color"])
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    endpoints = []
    for r, c in coords:
        degree = sum(1 for nr, nc in _neighbors4(shape, int(r), int(c)) if mask[nr, nc])
        if degree == 1:
            endpoints.append((int(r), int(c)))
        elif degree != 2:
            raise AssertionError("curve must be a simple open chain or simple loop")
    chord = np.zeros(shape, dtype=bool)
    is_open = len(endpoints) == 2
    if is_open:
        (r0, c0), (r1, c1) = endpoints
        if r0 == r1:
            for cc in range(min(c0, c1) + 1, max(c0, c1)):
                if mask[r0, cc]:
                    raise AssertionError("open-chain closing chord must pass through background")
                chord[r0, cc] = True
        elif c0 == c1:
            for rr in range(min(r0, r1) + 1, max(r0, r1)):
                if mask[rr, c0]:
                    raise AssertionError("open-chain closing chord must pass through background")
                chord[rr, c0] = True
        else:
            raise AssertionError("open-chain endpoints must share a row or a column")
    elif len(endpoints) != 0:
        raise AssertionError("curve must have zero or two endpoints")
    barrier = mask | chord
    inside = _inside_mask(barrier)
    if not inside.any():
        raise AssertionError("closed curve must enclose at least one background cell")
    return {
        "color": color,
        "mask": mask,
        "barrier": barrier,
        "chord": chord,
        "inside": inside,
        "inside_area": int(inside.sum()),
        "is_open": is_open,
    }


def _transform(grid, *, include_open=True, use_parity=True):
    g = np.asarray(grid, dtype=int)
    curves = []
    out = g.copy()
    for component in _monochromatic_components(g):
        info = _curve_info(component, g.shape)
        if info["is_open"] and not include_open:
            continue
        curves.append(info)
        out[info["chord"]] = RED
    for region in _components_from_mask(out == BLACK):
        sr, sc = map(int, region[0])
        containing = [curve for curve in curves if curve["inside"][sr, sc]]
        if not containing:
            continue
        if use_parity and len(containing) % 2 == 0:
            continue
        innermost = min(containing, key=lambda curve: curve["inside_area"])
        out[region[:, 0], region[:, 1]] = int(innermost["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Close every open curve with a red chord, then fill odd-depth enclosed regions by the innermost enclosing curve color.

    Input structure:
    - A black background contains several pairwise non-touching 1-cell-wide monochrome curves.
    - Each curve is either a simple closed loop or a simple open chain.
    - Every open chain has exactly two endpoints, and those endpoints share one row or one column.

    Rule:
    - For each open chain, draw a straight red chord along the shared row or column between its two endpoints; keep the original curve pixels unchanged.
    - Treat every original loop and every newly closed open chain as a closed boundary.
    - For each black 4-connected background region, count how many such boundaries enclose it: if the count is odd, fill the region with the color of the innermost enclosing curve; if the count is even, leave it black.

    Key insight:
    - The red chord changes which background regions are enclosed, while nested boundaries must be interpreted by odd-even depth rather than by filling every apparent interior.
    """
    return _transform(grid, include_open=True, use_parity=True)


def _draw_curve(grid, spec):
    r0, c0, r1, c1 = spec["bbox"]
    color = int(spec["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if spec["kind"] == "open":
        side = spec["side"]
        if side == "top":
            grid[r0, c0 : c1 + 1] = BLACK
        elif side == "bottom":
            grid[r1, c0 : c1 + 1] = BLACK
        elif side == "left":
            grid[r0 : r1 + 1, c0] = BLACK
        else:
            grid[r0 : r1 + 1, c1] = BLACK


def _interior_bbox(spec):
    r0, c0, r1, c1 = spec["bbox"]
    top, left, bottom, right = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    if spec["kind"] == "open":
        side = spec["side"]
        if side == "top":
            top += 1
        elif side == "bottom":
            bottom -= 1
        elif side == "left":
            left += 1
        else:
            right -= 1
    return top, left, bottom, right


def _inset_bbox(bbox, margin):
    r0, c0, r1, c1 = bbox
    return (r0 + margin, c0 + margin, r1 - margin, c1 - margin)


def _box_size(bbox):
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _make_spec(bbox, color, kind, side=None):
    spec = {"bbox": tuple(map(int, bbox)), "color": int(color), "kind": kind, "side": side}
    h, w = _box_size(spec["bbox"])
    assert h >= 5 and w >= 5
    top, left, bottom, right = _interior_bbox(spec)
    assert top <= bottom and left <= right
    return spec


def _sample_colors(count):
    return [int(c) for c in _RNG.choice(np.array(CURVE_COLORS), size=count, replace=False)]


def _chain3_layout(side_offset):
    outer_h = int(_RNG.integers(17, 20))
    outer_w = int(_RNG.integers(17, 20))
    h = outer_h + int(_RNG.integers(3, 7))
    w = outer_w + int(_RNG.integers(3, 7))
    r0 = int(_RNG.integers(1, h - outer_h))
    c0 = int(_RNG.integers(1, w - outer_w))
    outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    middle = _inset_bbox(outer, 3)
    inner = _inset_bbox(middle, 3)
    colors = _sample_colors(3)
    kinds = ("closed", "open", "closed") if side_offset % 2 == 0 else ("open", "closed", "open")
    specs = []
    for i, bbox in enumerate((outer, middle, inner)):
        side = OPEN_SIDES[(side_offset + i) % 4] if kinds[i] == "open" else None
        specs.append(_make_spec(bbox, colors[i], kinds[i], side))
    return (h, w), specs


def _sibling_layout(side_offset):
    left_h = int(_RNG.integers(13, 16))
    left_w = int(_RNG.integers(13, 16))
    right_h = int(_RNG.integers(7, 10))
    right_w = int(_RNG.integers(7, 10))
    gap = int(_RNG.integers(3, 5))
    top_margin = int(_RNG.integers(1, 4))
    bottom_margin = int(_RNG.integers(1, 4))
    left_margin = int(_RNG.integers(1, 3))
    right_margin = int(_RNG.integers(1, 4))
    h = max(left_h, right_h) + top_margin + bottom_margin
    w = left_w + gap + right_w + left_margin + right_margin
    left_box = (top_margin, left_margin, top_margin + left_h - 1, left_margin + left_w - 1)
    child_box = _inset_bbox(left_box, 3)
    right_top = int(_RNG.integers(top_margin, h - bottom_margin - right_h + 1))
    right_left = left_margin + left_w + gap
    right_box = (right_top, right_left, right_top + right_h - 1, right_left + right_w - 1)
    colors = _sample_colors(3)
    specs = [
        _make_spec(left_box, colors[0], "open", OPEN_SIDES[side_offset % 4]),
        _make_spec(child_box, colors[1], "closed"),
        _make_spec(right_box, colors[2], "open", OPEN_SIDES[(side_offset + 2) % 4]),
    ]
    return (h, w), specs


def _mixed_layout(side_offset):
    outer_h = int(_RNG.integers(14, 18))
    outer_w = int(_RNG.integers(14, 17))
    inner_margin = 3
    inner_h = outer_h - 2 * inner_margin
    inner_w = outer_w - 2 * inner_margin
    extra_h = int(_RNG.integers(8, 12))
    extra_w = int(_RNG.integers(7, 10))
    gap = int(_RNG.integers(3, 5))
    h = max(outer_h + 4, extra_h + 4)
    w = outer_w + extra_w + gap + int(_RNG.integers(3, 6))
    outer_top = int(_RNG.integers(1, h - outer_h))
    outer_left = int(_RNG.integers(1, max(2, w - (outer_w + extra_w + gap + 1))))
    outer = (outer_top, outer_left, outer_top + outer_h - 1, outer_left + outer_w - 1)
    inner = _inset_bbox(outer, inner_margin)
    extra_top = int(_RNG.integers(1, h - extra_h))
    extra_left = outer_left + outer_w + gap
    extra = (extra_top, extra_left, extra_top + extra_h - 1, extra_left + extra_w - 1)
    colors = _sample_colors(4)
    specs = [
        _make_spec(outer, colors[0], "open", OPEN_SIDES[(side_offset + 1) % 4]),
        _make_spec(inner, colors[1], "open", OPEN_SIDES[(side_offset + 3) % 4]),
        _make_spec(extra, colors[2], "closed"),
    ]
    if _RNG.random() < 0.5:
        small_h = int(_RNG.integers(5, 7))
        small_w = int(_RNG.integers(5, 8))
        small_top = h - small_h - 1
        small_left = int(_RNG.integers(1, max(2, outer_left - small_w - 1)))
        if small_left >= 1:
            small = (small_top, small_left, small_top + small_h - 1, small_left + small_w - 1)
            specs.append(_make_spec(small, colors[3], "open", OPEN_SIDES[side_offset % 4]))
    return (h, w), specs


def _build_pattern(pattern, side_offset):
    if pattern == 0:
        return _chain3_layout(side_offset)
    if pattern == 1:
        return _sibling_layout(side_offset)
    return _mixed_layout(side_offset)


def _validate_spacing(grid):
    fg = grid != BLACK
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not fg[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if fg[nr, nc] and grid[nr, nc] != grid[r, c]:
                        return False
    return True


def _generated_grid():
    global _GEN_COUNTER
    pattern = _GEN_COUNTER % 3
    side_offset = _GEN_COUNTER
    _GEN_COUNTER += 1
    shape, specs = _build_pattern(pattern, side_offset)
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        _draw_curve(grid, spec)
    return grid


def _validate_generated(grid):
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _validate_spacing(grid):
        return False
    try:
        components = [_curve_info(comp, grid.shape) for comp in _monochromatic_components(grid)]
    except AssertionError:
        return False
    if len(components) < 3:
        return False
    if not any(comp["is_open"] for comp in components):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    black_regions = _components_from_mask(out == BLACK)
    if not any(not np.any(region[:, 0] == 0) and not np.any(region[:, 0] == out.shape[0] - 1)
               and not np.any(region[:, 1] == 0) and not np.any(region[:, 1] == out.shape[1] - 1)
               for region in black_regions):
        return False
    filled_colors = set(map(int, np.unique(out))) - {BLACK, RED}
    if len(filled_colors) < 2:
        return False
    if np.array_equal(out, _transform(grid, include_open=False, use_parity=True)):
        return False
    if np.array_equal(out, _transform(grid, include_open=True, use_parity=False)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generated_grid()
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
