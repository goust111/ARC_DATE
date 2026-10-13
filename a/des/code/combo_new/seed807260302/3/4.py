import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _find_separator(g: np.ndarray) -> int:
    full_grey_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert full_grey_rows.size > 0, "expected a complete grey horizontal separator"
    for r in map(int, full_grey_rows):
        has_legend = bool(np.any((g[:r] != BLACK) & (g[:r] != GREY)))
        has_shape = bool(np.any(g[r + 1 :] == GREY))
        if has_legend and has_shape:
            return r
    return int(full_grey_rows[0])


def _legend_sequence(g: np.ndarray, sep_r: int) -> list[int]:
    top = g[:sep_r]
    mask = (top != BLACK) & (top != GREY)
    bars = []
    for pts in _components(mask):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        vals = {int(top[r, c]) for r, c in pts}
        assert len(vals) == 1, "each legend bar must have one color"
        assert c0 == c1, "legend bars must be one pixel wide"
        height = r1 - r0 + 1
        assert len(pts) == height, "legend bars must be solid vertical runs"
        bars.append((height, c0, next(iter(vals))))

    assert bars, "expected at least one colored legend bar"
    assert len({h for h, _, _ in bars}) == len(bars), "legend bar heights must be unique"
    bars.sort(key=lambda x: (x[0], x[1]))
    return [color for _height, _col, color in bars]


def _paint_concentric_layers(
    out: np.ndarray, pts: list[tuple[int, int]], colors: list[int]
) -> None:
    remaining = set((int(r), int(c)) for r, c in pts)
    layer = 0
    while remaining:
        boundary = []
        for r, c in remaining:
            for dr, dc in NEIGHBORS4:
                if (r + dr, c + dc) not in remaining:
                    boundary.append((r, c))
                    break
        assert boundary, "peeling must remove at least one cell per layer"
        color = int(colors[layer % len(colors)])
        for r, c in boundary:
            out[r, c] = color
        for p in boundary:
            remaining.remove(p)
        layer += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey solid shape by concentric layers using a height-sorted color legend.

    Input structure:
    - Black background with one complete one-pixel grey horizontal separator.
    - Above the separator, colored one-pixel-wide vertical legend bars have distinct heights.
    - Below the separator, separated grey 4-connected solid shapes sit on the black background.

    Rule:
    - Read the colored legend bars and sort their colors by bar height from shortest to tallest.
    - For each lower grey shape, repeatedly peel its current 4-connected boundary layer.
    - Paint the outer layer with the first sorted legend color, the next layer with the second color, cycling when the shape has more layers than colors.
    - Keep the separator and the entire upper legend area unchanged.

    Key insight:
    The legend's visible bar heights define the color order, and each grey object is then solved by iterative boundary peeling rather than by its raw size or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_r = _find_separator(g)
    colors = _legend_sequence(g, sep_r)

    out = g.copy()
    lower_mask = g == GREY
    lower_mask[: sep_r + 1, :] = False
    for pts in _components(lower_mask):
        _paint_concentric_layers(out, pts, colors)
    return out


GRID_H_RANGE = (15, 25)
GRID_W_RANGE = (15, 25)
TOP_MIN = 5
BOTTOM_MIN = 6
BAR_COUNT_RANGE = (3, 5)
BAR_HEIGHT_MAX = 5
SHAPE_COUNT_RANGE = (2, 5)
SMALL_RECT_MAX = 8
FORCE_CYCLE_PROB = 0.5
TOUCH_SEPARATOR_PROB = 0.28
GEN_RETRIES = 400


def _separated_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - old) >= 2 for old in chosen):
            chosen.append(c)
            if len(chosen) == count:
                chosen.sort()
                return chosen
    return None


def _rect_layer_count(h: int, w: int) -> int:
    return (min(h, w) + 1) // 2


def _free_rect(
    occupied: np.ndarray, sep_r: int, r0: int, c0: int, h: int, w: int
) -> bool:
    rows, cols = occupied.shape
    if r0 < sep_r + 1 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0, rr1 = max(sep_r + 1, r0 - 1), min(rows, r0 + h + 1)
    cc0, cc1 = max(0, c0 - 1), min(cols, c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_rect(
    grid: np.ndarray,
    occupied: np.ndarray,
    sep_r: int,
    h: int,
    w: int,
    prefer_touch_separator: bool,
) -> bool:
    rows, cols = grid.shape
    all_positions = []
    inner_positions = []
    touch_positions = []
    for r in range(sep_r + 1, rows - h + 1):
        for c in range(0, cols - w + 1):
            if not _free_rect(occupied, sep_r, r, c, h, w):
                continue
            all_positions.append((r, c))
            if r == sep_r + 1:
                touch_positions.append((r, c))
            if r > sep_r + 1 and c > 0 and r + h < rows and c + w < cols:
                inner_positions.append((r, c))
    if not all_positions:
        return False
    if prefer_touch_separator and touch_positions:
        r, c = random.choice(touch_positions)
    elif inner_positions:
        r, c = random.choice(inner_positions)
    else:
        r, c = random.choice(all_positions)
    grid[r : r + h, c : c + w] = GREY
    occupied[r : r + h, c : c + w] = True
    return True


def _sample_shape_dims(
    bottom_h: int, width: int, color_count: int, force_cycle: bool
) -> list[tuple[int, int]] | None:
    count = random.randint(*SHAPE_COUNT_RANGE)
    dims: list[tuple[int, int]] = []
    if force_cycle:
        need = 2 * color_count + 1
        if bottom_h < need or width < need:
            return None
        h = random.randint(need, min(bottom_h, need + 4))
        w = random.randint(need, min(width, need + 5))
        if random.random() < 0.5:
            h, w = w if w <= bottom_h else h, h if h <= width else w
        if h > bottom_h or w > width or _rect_layer_count(h, w) <= color_count:
            return None
        dims.append((h, w))

    max_h = min(SMALL_RECT_MAX, bottom_h)
    max_w = min(SMALL_RECT_MAX, width)
    if max_h < 2 or max_w < 2:
        return None
    while len(dims) < count:
        if random.random() < 0.35:
            side = random.randint(2, min(max_h, max_w))
            h, w = side, side
        else:
            h = random.randint(2, max_h)
            w = random.randint(2, max_w)
        dims.append((h, w))

    dims.sort(key=lambda x: x[0] * x[1], reverse=True)
    return dims


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_H_RANGE)
    cols = random.randint(*GRID_W_RANGE)
    if rows < TOP_MIN + 1 + BOTTOM_MIN:
        return None

    sep_r = random.randint(TOP_MIN, rows - BOTTOM_MIN - 1)
    bottom_h = rows - sep_r - 1
    force_cycle = random.random() < FORCE_CYCLE_PROB

    max_bars = min(BAR_COUNT_RANGE[1], BAR_HEIGHT_MAX, sep_r, (cols - 1) // 2)
    if force_cycle:
        max_bars = min(max_bars, 4)
    if max_bars < BAR_COUNT_RANGE[0]:
        return None
    bar_count = random.randint(BAR_COUNT_RANGE[0], max_bars)
    heights = random.sample(range(1, min(BAR_HEIGHT_MAX, sep_r) + 1), bar_count)
    colors = random.sample(COLOR_POOL, bar_count)
    bar_cols = _separated_columns(cols, bar_count)
    if bar_cols is None:
        return None

    grid = np.full((rows, cols), BLACK, dtype=int)
    grid[sep_r, :] = GREY
    for col, height, color in zip(bar_cols, heights, colors):
        grid[sep_r - height : sep_r, col] = int(color)

    dims = _sample_shape_dims(bottom_h, cols, bar_count, force_cycle)
    if dims is None:
        return None

    occupied = np.zeros((rows, cols), dtype=bool)
    has_cycle_shape = False
    for i, (h, w) in enumerate(dims):
        touch = i == 0 and random.random() < TOUCH_SEPARATOR_PROB
        if not _place_rect(grid, occupied, sep_r, h, w, touch):
            return None
        has_cycle_shape = has_cycle_shape or _rect_layer_count(h, w) > bar_count

    if force_cycle and not has_cycle_shape:
        return None
    if int(np.sum(grid == BLACK)) <= int(np.sum(grid != BLACK)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if not np.array_equal(out[: sep_r + 1], grid[: sep_r + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
