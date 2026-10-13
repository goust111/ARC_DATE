from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each small monochromatic foreground object green while leaving larger objects unchanged.

    Input structure
    - The dominant color is the background.
    - The foreground is split into monochromatic 4-connected objects of several colors.
    - Green is absent from the input and is the output marker color.

    Rule
    - Find every non-background same-color 4-connected object.
    - If an object's area is one or two pixels, recolor all of its pixels green.
    - Keep objects with area at least three unchanged.

    Key insight
    The selected pixels are the complete small color-components, independent of their color or position.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg != Color.GREEN, "background must not be the output marker color"
    assert Color.GREEN not in set(map(int, g.ravel())), "green must be absent from the input"

    out = g.copy()
    recolored_area = 0
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        area = int(comp["area"])
        assert area >= 1, "components must be nonempty"
        if area <= 2:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.GREEN
            recolored_area += area

    assert recolored_area > 0, "expected at least one small foreground object"
    return out

# --------------- generator (o5x) ---------------

# Palette excludes black background and green output marker.
_GEN_PALETTE = [Color.BLUE, Color.RED, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

# Canvas and palette bounds.
_GEN_GRID_ROWS, _GEN_GRID_COLS, _GEN_COLOR_COUNTS = (3, 9), (3, 8), [3, 4]

# Foreground density and background dominance.
_GEN_PERCENT, _GEN_MIN_FG_CELLS = 100, 5
_GEN_FG_DENSITY_PCT = (45, 67)
_GEN_MIN_BG_CELLS, _GEN_COLOR_BG_MARGIN = 4, 1

# Component count and area bounds.
_GEN_LARGE_COUNT_RANGE, _GEN_SMALL_COUNT_RANGE = (1, 7), (2, 15)
_GEN_MAX_COMPONENTS = 19
_GEN_LARGE_AREA_RANGE, _GEN_SMALL_AREA_RANGE = (3, 5), (1, 2)

# Shape geometry and retry limits.
_GEN_SHAPE_BOUNDS = (3, 4)
_GEN_CONNECTIVITY, _GEN_RETRY_LIMIT = 4, 300
_GEN_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _gen_shape_size(shape):
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _gen_normalize(coords):
    arr = np.array(list(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _gen_orientations(shape, max_h, max_w):
    base = np.array(shape, dtype=int)
    seen = set()
    out = []
    for swap in (False, True):
        arr = base[:, ::-1] if swap else base.copy()
        for flip_r in (False, True):
            for flip_c in (False, True):
                cur = arr.copy()
                if flip_r:
                    cur[:, 0] = int(cur[:, 0].max()) - cur[:, 0]
                if flip_c:
                    cur[:, 1] = int(cur[:, 1].max()) - cur[:, 1]
                cur = _gen_normalize(map(tuple, cur))
                h, w = _gen_shape_size(cur)
                key = tuple(map(tuple, cur.tolist()))
                if h <= max_h and w <= max_w and key not in seen:
                    seen.add(key)
                    out.append(cur)
    random.shuffle(out)
    return out


def _gen_polyomino(area, rows, cols):
    shape_rows, shape_cols = _GEN_SHAPE_BOUNDS
    max_h = min(shape_rows, rows)
    max_w = min(shape_cols, cols)
    cells = {(0, 0)}
    while len(cells) < area:
        candidates = []
        for r, c in cells:
            for dr, dc in _GEN_NEIGHBORS4:
                nb = (r + dr, c + dc)
                if nb in cells:
                    continue
                trial = _gen_normalize(cells | {nb})
                h, w = _gen_shape_size(trial)
                if h <= max_h and w <= max_w:
                    candidates.append(nb)
        if not candidates:
            return None
        cells.add(random.choice(candidates))
        cells = set(map(tuple, _gen_normalize(cells).tolist()))
    shape = _gen_normalize(cells)
    return random.choice(_gen_orientations(shape, max_h, max_w))


def _gen_distribute(total, lows, highs):
    values = list(lows)
    remaining = int(total) - sum(values)
    caps = [hi - lo for lo, hi in zip(lows, highs)]
    while remaining > 0:
        choices = [i for i, cap in enumerate(caps) if cap > 0]
        if not choices:
            return None
        i = random.choice(choices)
        values[i] += 1
        caps[i] -= 1
        remaining -= 1
    return values


def _gen_assign_colors(areas, colors, bg_count):
    """Assign all selected colors while keeping black strictly dominant."""
    cap = int(bg_count) - _GEN_COLOR_BG_MARGIN
    if cap < 1 or sum(areas) > cap * len(colors) or max(areas) > cap:
        return None

    order = sorted(range(len(areas)), key=lambda i: (-areas[i], random.random()))
    totals = {c: 0 for c in colors}
    used = set()
    assigned = {}

    def rec(pos):
        if pos == len(order):
            return len(used) == len(colors)
        remaining = len(order) - pos
        if remaining < len(colors) - len(used):
            return False
        if sum(areas[i] for i in order[pos:]) > sum(cap - totals[c] for c in colors):
            return False

        idx = order[pos]
        area = int(areas[idx])
        options = list(colors)
        random.shuffle(options)
        options.sort(key=lambda c: (totals[c], c in used))
        for color in options:
            if totals[color] + area > cap:
                continue
            totals[color] += area
            was_used = color in used
            used.add(color)
            assigned[idx] = color
            if rec(pos + 1):
                return True
            totals[color] -= area
            if not was_used:
                used.remove(color)
            assigned.pop(idx, None)
        return False

    if not rec(0):
        return None
    return [assigned[i] for i in range(len(areas))]


def _gen_valid_options(grid, shapes, color):
    """Return placements that avoid overlap and same-color 4-neighbor merges."""
    H, W = grid.shape
    options = []
    for shape in shapes:
        h, w = _gen_shape_size(shape)
        rel_cells = set(map(tuple, shape.tolist()))
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                abs_cells = {(r0 + int(r), c0 + int(c)) for r, c in rel_cells}
                ok = True
                for r, c in abs_cells:
                    if grid[r, c] != Color.BLACK:
                        ok = False
                        break
                    for dr, dc in _GEN_NEIGHBORS4:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in abs_cells:
                            continue
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == color:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    options.append((shape, r0, c0))
    return options


def _gen_paste_shape(grid, shape, r0, c0, color):
    h, w = _gen_shape_size(shape)
    patch = np.zeros((h, w), dtype=int)
    patch[shape[:, 0], shape[:, 1]] = int(color)
    paste(grid, patch, int(r0), int(c0), transparent=0)
    return np.array([[int(r0) + int(r), int(c0) + int(c)] for r, c in shape], dtype=int)


def _gen_expected_output(grid):
    out = grid.copy()
    for comp in find_components(grid, bg=Color.BLACK, connectivity=_GEN_CONNECTIVITY, monochromatic=True):
        if int(comp["area"]) <= _GEN_SMALL_AREA_RANGE[1]:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.GREEN
    return out


def _gen_plan_components(total, colors):
    """Choose target area/counts, then allocate component areas and colors."""
    color_count = len(colors)
    min_pct, max_pct = _GEN_FG_DENSITY_PCT
    min_fg = max(_GEN_MIN_FG_CELLS, (min_pct * total + _GEN_PERCENT - 1) // _GEN_PERCENT)
    max_fg = (max_pct * total) // _GEN_PERCENT
    target_options = [
        fg
        for fg in range(min_fg, max_fg + 1)
        if total - fg >= _GEN_MIN_BG_CELLS and fg <= color_count * (total - fg - _GEN_COLOR_BG_MARGIN)
    ]
    if not target_options:
        return None
    target_fg = random.choice(target_options)
    bg_count = total - target_fg

    pairs = []
    large_min, large_max = _GEN_LARGE_COUNT_RANGE
    small_min, small_max = _GEN_SMALL_COUNT_RANGE
    large_area_min, large_area_max = _GEN_LARGE_AREA_RANGE
    small_area_min, small_area_max = _GEN_SMALL_AREA_RANGE
    for large_count in range(large_min, large_max + 1):
        for small_count in range(small_min, small_max + 1):
            if large_count + small_count > _GEN_MAX_COMPONENTS or large_count + small_count < color_count:
                continue
            lo = large_area_min * large_count + small_area_min * small_count
            hi = large_area_max * large_count + small_area_max * small_count
            if lo <= target_fg <= hi:
                pairs.append((large_count, small_count))
    if not pairs:
        return None
    large_count, small_count = random.choice(pairs)

    kinds = ["large"] * large_count + ["small"] * small_count
    lows = [large_area_min if kind == "large" else small_area_min for kind in kinds]
    highs = [large_area_max if kind == "large" else small_area_max for kind in kinds]
    areas = _gen_distribute(target_fg, lows, highs)
    if areas is None:
        return None

    component_colors = _gen_assign_colors(areas, colors, bg_count)
    if component_colors is None:
        return None

    components = [
        {"kind": kind, "area": area, "color": color}
        for kind, area, color in zip(kinds, areas, component_colors)
    ]
    large = [c for c in components if c["kind"] == "large"]
    small = [c for c in components if c["kind"] == "small"]
    random.shuffle(large)
    random.shuffle(small)
    return target_fg, large_count, small_count, large + small


def _gen_render_components(rows, cols, components):
    """Place every planned component on an initially black grid."""
    grid = np.zeros((rows, cols), dtype=int)
    for comp in components:
        if comp["kind"] == "large":
            shape = _gen_polyomino(comp["area"], rows, cols)
            if shape is None:
                return None
            shape_rows, shape_cols = _GEN_SHAPE_BOUNDS
            shapes = _gen_orientations(shape, min(shape_rows, rows), min(shape_cols, cols))
        elif comp["area"] == _GEN_SMALL_AREA_RANGE[0]:
            shapes = [np.array([[0, 0]], dtype=int)]
        else:
            shapes = [
                np.array([[0, 0], [0, 1]], dtype=int),
                np.array([[0, 0], [1, 0]], dtype=int),
            ]
            random.shuffle(shapes)

        options = _gen_valid_options(grid, shapes, comp["color"])
        if not options:
            return None
        shape, r0, c0 = random.choice(options)
        _gen_paste_shape(grid, shape, r0, c0, comp["color"])
    return grid


def _gen_grid_is_valid(grid, target_fg, total, colors, small_count, large_count):
    """Run invariant and solver-oracle checks for a completed grid."""
    if get_bg(grid) != Color.BLACK or Color.GREEN in set(map(int, grid.ravel())):
        return False
    fg = int(np.count_nonzero(grid))
    min_pct, max_pct = _GEN_FG_DENSITY_PCT
    if fg != target_fg or not (min_pct / _GEN_PERCENT <= fg / total <= max_pct / _GEN_PERCENT):
        return False

    comps = find_components(grid, bg=Color.BLACK, connectivity=_GEN_CONNECTIVITY, monochromatic=True)
    large_area_min, large_area_max = _GEN_LARGE_AREA_RANGE
    small_area_min, small_area_max = _GEN_SMALL_AREA_RANGE
    small_areas = [int(c["area"]) for c in comps if int(c["area"]) <= small_area_max]
    large_areas = [int(c["area"]) for c in comps if int(c["area"]) >= large_area_min]
    if len(small_areas) != small_count or len(large_areas) != large_count:
        return False
    if any(a < small_area_min or a > small_area_max for a in small_areas):
        return False
    if any(a < large_area_min or a > large_area_max for a in large_areas):
        return False
    present = set(map(int, grid.ravel())) - {Color.BLACK}
    if present != set(colors):
        return False

    out = _gen_expected_output(grid)
    return np.array_equal(main(grid), out)


def _gen_attempt():
    rows = random.randint(*_GEN_GRID_ROWS)
    cols = random.randint(*_GEN_GRID_COLS)
    color_count = random.choice(_GEN_COLOR_COUNTS)
    colors = random.sample(_GEN_PALETTE, color_count)
    total = rows * cols

    plan = _gen_plan_components(total, colors)
    if plan is None:
        return None
    target_fg, large_count, small_count, components = plan

    grid = _gen_render_components(rows, cols, components)
    if grid is None:
        return None

    if not _gen_grid_is_valid(grid, target_fg, total, colors, small_count, large_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 12eac192 input")
