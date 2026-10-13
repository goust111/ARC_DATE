import random
from collections import Counter, deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _shape_array(shape_sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in shape_sig) + 1
    w = max(c for _, c in shape_sig) + 1
    arr = np.zeros((h, w), dtype=bool)
    for r, c in shape_sig:
        arr[r, c] = True
    return arr


def _band_rows(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, int]]:
    rows: list[int] = []
    colors: dict[int, int] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color != bg and np.all(row == color):
            rows.append(r)
            colors[r] = color
    return rows, colors


def _enclosed_holes(template: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(template)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in template:
        mask[r - r0, c - c0] = True

    holes: list[list[tuple[int, int]]] = []
    for pts in _components(~mask, connectivity=4):
        if any(
            rr == 0 or cc == 0 or rr == mask.shape[0] - 1 or cc == mask.shape[1] - 1
            for rr, cc in pts
        ):
            continue
        holes.append([(rr + r0, cc + c0) for rr, cc in pts])
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered shape-matched fragments into a holed template and recolor them by crossed channel bands.

    Input structure:
    - The background is black.
    - A large connected template object sits in the upper part of the grid; it contains several internal black holes of different shapes.
    - Several small colored fragments are scattered below the template, and each fragment has exactly the same shape as one template hole under translation only.
    - Between the template and the fragments are several full-width, single-row horizontal color bands.

    Rule:
    - Ignore fragment colors and identify the template as the largest non-band connected object.
    - Extract the enclosed black holes inside the template and match each lower fragment to the unique hole with the same normalized shape.
    - Move each fragment into its matched hole, erase the fragment's original position, and fill the hole with the color of the first horizontal band crossed while moving upward from the fragment to that hole.
    - Keep the template outline and all color bands unchanged.

    Key insight:
    - Shape determines where each fragment belongs, but the fill color is determined by the nearest crossed horizontal band above the fragment, not by the fragment's original color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    band_rows, band_colors = _band_rows(g, bg)
    assert band_rows, "expected at least one full-width color band"

    fg = g != bg
    for r in band_rows:
        fg[r, :] = False

    comps = _components(fg, connectivity=4)
    assert comps, "expected at least one non-band foreground component"
    template = max(comps, key=len)
    holes = _enclosed_holes(template)
    assert holes, "template must contain enclosed holes"

    remaining = fg.copy()
    for r, c in template:
        remaining[r, c] = False
    fragments = _components(remaining, connectivity=4)
    assert len(fragments) == len(holes), "fragment count must match hole count"

    hole_by_shape: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for hole in holes:
        sig = _normalize(hole)
        assert sig not in hole_by_shape, "hole shapes must be unique"
        hole_by_shape[sig] = hole

    out = g.copy()
    for frag in fragments:
        for r, c in frag:
            out[r, c] = bg

    for frag in fragments:
        sig = _normalize(frag)
        assert sig in hole_by_shape, "every fragment must match a hole shape"
        hole = hole_by_shape[sig]
        frag_top = min(r for r, _ in frag)
        hole_top = min(r for r, _ in hole)
        crossed = [r for r in band_rows if hole_top < r < frag_top]
        assert crossed, "fragment must cross at least one band"
        fill_color = band_colors[max(crossed)]
        for r, c in hole:
            out[r, c] = fill_color

    return out


def _grow_shape(rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    box_h = rng.randint(2, 3)
    box_w = rng.randint(2, 4)
    area = rng.randint(3, min(6, box_h * box_w))
    start = (rng.randrange(box_h), rng.randrange(box_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    sig = _normalize(list(cells))
    arr = _shape_array(sig)
    if arr.shape[0] == 1 or arr.shape[1] == 1:
        return None
    return sig


def _make_shapes(rng: random.Random, count: int) -> list[tuple[tuple[int, int], ...]]:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    attempts = 0
    while len(shapes) < count and attempts < 500:
        attempts += 1
        sig = _grow_shape(rng)
        if sig is None or sig in seen:
            continue
        seen.add(sig)
        shapes.append(sig)
    if len(shapes) != count:
        raise RuntimeError("failed to build enough unique shapes")
    return shapes


def _place_shape_with_gap(
    blocked: np.ndarray, shape: np.ndarray, top: int, left: int
) -> bool:
    h, w = blocked.shape
    sh, sw = shape.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    cells = np.argwhere(shape)
    for rr, cc in cells:
        if blocked[top + int(rr), left + int(cc)]:
            return False
    return True


def _stamp_blocked(blocked: np.ndarray, shape: np.ndarray, top: int, left: int) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(shape):
        r = top + int(rr)
        c = left + int(cc)
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_holes(
    rng: random.Random,
    shapes: list[tuple[tuple[int, int], ...]],
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: len(shapes[i]), reverse=True)
    for idx in order:
        shape = _shape_array(shapes[idx])
        sh, sw = shape.shape
        candidates = []
        for top in range(inner_h - sh + 1):
            for left in range(inner_w - sw + 1):
                if _place_shape_with_gap(blocked, shape, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placements[idx] = (top, left)
        _stamp_blocked(blocked, shape, top, left)
    return [p for p in placements if p is not None]


def _template_patch(
    shapes: list[tuple[tuple[int, int], ...]],
    placements: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    patch = np.full((inner_h + 2, inner_w + 2), TEMPLATE_COLOR, dtype=int)
    hole_points: list[list[tuple[int, int]]] = []
    for sig, (top, left) in zip(shapes, placements, strict=False):
        pts = []
        for rr, cc in sig:
            ar = top + rr + 1
            ac = left + cc + 1
            patch[ar, ac] = BG
            pts.append((ar, ac))
        hole_points.append(pts)
    return patch, hole_points


def _place_fragment_candidates(
    blocked: np.ndarray,
    shape: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_margin: int = 1,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    sh, sw = shape.shape
    if row_lo + sh - 1 > row_hi:
        return []
    candidates = []
    for top in range(row_lo, row_hi - sh + 2):
        for left in range(col_margin, w - sw - col_margin + 1):
            if _place_shape_with_gap(blocked, shape, top, left):
                candidates.append((top, left))
    return candidates


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    pts = []
    for rr, cc in np.argwhere(shape):
        r = top + int(rr)
        c = left + int(cc)
        grid[r, c] = color
        pts.append((r, c))
    return pts


def _expected_output(
    grid: np.ndarray,
    hole_abs: list[list[tuple[int, int]]],
    fragment_abs: list[list[tuple[int, int]]],
    levels: list[int],
    band_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for frag in fragment_abs:
        for r, c in frag:
            out[r, c] = BG
    for hole, level in zip(hole_abs, levels, strict=False):
        fill = band_colors[level]
        for r, c in hole:
            out[r, c] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    band_count = rng.choice([2, 3])
    shape_count = band_count + rng.choice([0, 1])
    shapes = _make_shapes(rng, shape_count)

    inner_h = rng.randint(5, 7)
    inner_w = rng.randint(10, 14)
    placements = _place_holes(rng, shapes, inner_h, inner_w)
    if placements is None:
        return None

    template_patch, hole_rel = _template_patch(shapes, placements, inner_h, inner_w)
    template_h, template_w = template_patch.shape

    grid_w_min = max(18, template_w + 4)
    grid_w_max = min(28, template_w + 10)
    if grid_w_min > grid_w_max:
        return None
    grid_w = rng.randint(grid_w_min, grid_w_max)

    template_top = rng.randint(1, 3)
    center_left = (grid_w - template_w) // 2
    template_left = max(1, min(grid_w - template_w - 1, center_left + rng.randint(-1, 1)))

    levels = list(range(band_count))
    while len(levels) < shape_count:
        levels.append(rng.randrange(band_count))
    rng.shuffle(levels)

    band_heights = {i: 0 for i in range(band_count)}
    for idx, sig in enumerate(shapes):
        band_heights[levels[idx]] = max(band_heights[levels[idx]], _shape_array(sig).shape[0])

    band_rows: list[int] = []
    layer_ranges: list[tuple[int, int]] = []
    current = template_top + template_h + 1
    for i in range(band_count):
        band_rows.append(current)
        row_lo = current + 2
        layer_height = band_heights[i] + rng.choice([0, 1])
        row_hi = row_lo + layer_height - 1
        layer_ranges.append((row_lo, row_hi))
        current = row_hi + 2
    grid_h = current + rng.randint(1, 2)
    if grid_h > 30:
        return None

    colors = rng.sample(COLOR_POOL, band_count + shape_count)
    band_colors = colors[:band_count]
    fragment_colors = colors[band_count:]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[template_top : template_top + template_h, template_left : template_left + template_w] = template_patch
    for row, color in zip(band_rows, band_colors, strict=False):
        grid[row, :] = color

    hole_abs = []
    for pts in hole_rel:
        hole_abs.append([(template_top + r, template_left + c) for r, c in pts])

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[grid != BG] = True
    fragment_abs: list[list[tuple[int, int]]] = []
    for idx, sig in enumerate(shapes):
        shape = _shape_array(sig)
        row_lo, row_hi = layer_ranges[levels[idx]]
        candidates = _place_fragment_candidates(blocked, shape, row_lo, row_hi, col_margin=1)
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        pts = _paint_shape(grid, shape, top, left, fragment_colors[idx])
        fragment_abs.append(pts)
        _stamp_blocked(blocked, shape, top, left)

    if np.count_nonzero(grid) >= grid.size // 2:
        return None

    expected = _expected_output(grid, hole_abs, fragment_abs, levels, band_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
