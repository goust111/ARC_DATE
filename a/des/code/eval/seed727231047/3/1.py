import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250

PATTERN_LIBRARY = {
    2: (
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 0]], dtype=int),
        np.array([[1, 0], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
    ),
    3: (
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    ),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(points, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _spans_between(lines, size: int):
    ordered = sorted(set(lines))
    assert ordered and ordered[0] == 0 and ordered[-1] == size - 1
    spans = []
    for left, right in zip(ordered, ordered[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _reconstruct_tile(cell: np.ndarray, color: int) -> np.ndarray:
    ch, cw = cell.shape
    assert ch == cw and ch in (3, 4)
    tile_size = ch - 1
    tile = np.zeros((tile_size, tile_size), dtype=int)
    for tr in range(tile_size):
        for tc in range(tile_size):
            vals = set()
            for r in range(ch):
                for c in range(cw):
                    if r % tile_size == tr and c % tile_size == tc and not (r == 0 and c == 0):
                        val = int(cell[r, c])
                        assert val in (BLACK, color)
                        vals.add(val)
            assert vals
            tile[tr, tc] = color if color in vals else BLACK
    assert np.any(tile == color) and np.any(tile == BLACK)
    return tile


def _extract_legend(grid: np.ndarray):
    h, w = grid.shape
    candidates = [
        comp for comp in _components(grid == GREY) if _touches_border(comp, h, w)
    ]
    assert candidates, "expected a unique edge-touching grey legend"
    legend = max(candidates, key=len)
    r0, c0, r1, c1 = _bbox(legend)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    full_rows = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == GREY)]
    full_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == GREY)]
    row_spans = _spans_between(full_rows, patch.shape[0])
    col_spans = _spans_between(full_cols, patch.shape[1])

    if len(row_spans) == 1 and len(col_spans) >= 2:
        cell_spans = [((row_spans[0][0], row_spans[0][1]), col_span) for col_span in col_spans]
    else:
        assert len(col_spans) == 1 and len(row_spans) >= 2
        cell_spans = [(row_span, (col_spans[0][0], col_spans[0][1])) for row_span in row_spans]

    mapping = {}
    for (rs, re), (cs, ce) in cell_spans:
        cell = patch[rs : re + 1, cs : ce + 1]
        color = int(cell[0, 0])
        assert color not in (BLACK, GREY)
        mapping[color] = _reconstruct_tile(cell, color)

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert mapping
    return legend_mask, mapping


def _target_regions(grid: np.ndarray, legend_mask: np.ndarray):
    h, w = grid.shape
    fillable = (grid != GREY) & (~legend_mask)
    regions = []
    for comp in _components(fillable):
        if _touches_border(comp, h, w):
            continue
        vals = [int(grid[r, c]) for r, c in comp]
        colors = {v for v in vals if v != BLACK}
        if len(colors) != 1:
            continue
        seed_color = next(iter(colors))
        assert vals.count(seed_color) == 1
        boundary = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and not legend_mask[nr, nc]:
                    boundary.add((nr, nc))
        assert boundary
        regions.append((comp, seed_color, _bbox(boundary)))
    assert regions
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region by looking up the seed color's periodic pattern
    in the edge legend, then tile that pattern with the boundary-box top-left as
    the phase origin.
    """
    g = np.asarray(grid, dtype=int)
    legend_mask, mapping = _extract_legend(g)
    out = g.copy()

    for cells, seed_color, (r0, c0, _r1, _c1) in _target_regions(g, legend_mask):
        assert seed_color in mapping
        tile = mapping[seed_color]
        th, tw = tile.shape
        for r, c in cells:
            out[r, c] = tile[(r - r0) % th, (c - c0) % tw]

    return out


def _make_legend_cell(pattern: np.ndarray, color: int) -> np.ndarray:
    tile_size = pattern.shape[0]
    cell_size = tile_size + 1
    cell = np.zeros((cell_size, cell_size), dtype=int)
    for r in range(cell_size):
        for c in range(cell_size):
            if pattern[r % tile_size, c % tile_size]:
                cell[r, c] = color
    cell[0, 0] = color
    return cell


def _build_legend(colors, patterns, side: str) -> np.ndarray:
    tile_size = patterns[0].shape[0]
    cell_size = tile_size + 1
    count = len(colors)
    if side in ("top", "bottom"):
        patch = np.full((cell_size + 2, count * (cell_size + 1) + 1), GREY, dtype=int)
        for idx, (color, pattern) in enumerate(zip(colors, patterns)):
            c0 = 1 + idx * (cell_size + 1)
            patch[1 : 1 + cell_size, c0 : c0 + cell_size] = _make_legend_cell(pattern, color)
    else:
        patch = np.full((count * (cell_size + 1) + 1, cell_size + 2), GREY, dtype=int)
        for idx, (color, pattern) in enumerate(zip(colors, patterns)):
            r0 = 1 + idx * (cell_size + 1)
            patch[r0 : r0 + cell_size, 1 : 1 + cell_size] = _make_legend_cell(pattern, color)
    return patch


def _normalize(points):
    r0, c0, r1, c1 = _bbox(points)
    shifted = {(r - r0 + 1, c - c0 + 1) for r, c in points}
    return shifted, (r1 - r0 + 3, c1 - c0 + 3)


def _make_region_shape(rng: random.Random):
    for _ in range(120):
        canvas_h = rng.randint(5, 7)
        canvas_w = rng.randint(5, 7)
        base_h = rng.randint(2, min(4, canvas_h - 2))
        base_w = rng.randint(2, min(4, canvas_w - 2))
        base_r = rng.randint(1, canvas_h - base_h - 1)
        base_c = rng.randint(1, canvas_w - base_w - 1)

        cells = {
            (r, c)
            for r in range(base_r, base_r + base_h)
            for c in range(base_c, base_c + base_w)
        }

        sides = list(SIDES)
        rng.shuffle(sides)
        added_arm = False
        for side in sides[: rng.randint(1, 2)]:
            if side == "top" and base_r > 0:
                arm_len = rng.randint(1, base_r)
                arm_w = rng.randint(1, max(1, base_w - 1))
                arm_c = rng.randint(base_c, base_c + base_w - arm_w)
                for r in range(base_r - arm_len, base_r):
                    for c in range(arm_c, arm_c + arm_w):
                        cells.add((r, c))
                added_arm = True
            elif side == "bottom" and base_r + base_h < canvas_h:
                arm_len = rng.randint(1, canvas_h - (base_r + base_h))
                arm_w = rng.randint(1, max(1, base_w - 1))
                arm_c = rng.randint(base_c, base_c + base_w - arm_w)
                for r in range(base_r + base_h, base_r + base_h + arm_len):
                    for c in range(arm_c, arm_c + arm_w):
                        cells.add((r, c))
                added_arm = True
            elif side == "left" and base_c > 0:
                arm_len = rng.randint(1, base_c)
                arm_h = rng.randint(1, max(1, base_h - 1))
                arm_r = rng.randint(base_r, base_r + base_h - arm_h)
                for r in range(arm_r, arm_r + arm_h):
                    for c in range(base_c - arm_len, base_c):
                        cells.add((r, c))
                added_arm = True
            elif side == "right" and base_c + base_w < canvas_w:
                arm_len = rng.randint(1, canvas_w - (base_c + base_w))
                arm_h = rng.randint(1, max(1, base_h - 1))
                arm_r = rng.randint(base_r, base_r + base_h - arm_h)
                for r in range(arm_r, arm_r + arm_h):
                    for c in range(base_c + base_w, base_c + base_w + arm_len):
                        cells.add((r, c))
                added_arm = True

        if not added_arm:
            continue

        shifted, (patch_h, patch_w) = _normalize(cells)
        ring = set()
        for r, c in shifted:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in shifted:
                    ring.add((nr, nc))

        if any(r < 0 or c < 0 or r >= patch_h or c >= patch_w for r, c in ring):
            continue
        if len(shifted) < 5:
            continue
        return shifted, ring, patch_h, patch_w

    raise RuntimeError("failed to build region shape")


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1):
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _validate_instance(grid: np.ndarray, expected_colors, expected_regions: int):
    legend_mask, mapping = _extract_legend(grid)
    assert set(mapping) == set(expected_colors)
    regions = _target_regions(grid, legend_mask)
    assert len(regions) == expected_regions
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)


def _gen_attempt(rng: random.Random):
    tile_size = rng.choice((2, 3))
    side = rng.choice(SIDES)
    color_count = rng.randint(2, 4)
    colors = rng.sample(SEED_COLORS, color_count)
    patterns = [pattern.copy() for pattern in rng.sample(PATTERN_LIBRARY[tile_size], color_count)]
    legend = _build_legend(colors, patterns, side)
    legend_h, legend_w = legend.shape

    if side in ("top", "bottom"):
        height_lo = max(16, legend_h + 8)
        width_lo = max(18, legend_w + 2)
    else:
        height_lo = max(18, legend_h + 2)
        width_lo = max(16, legend_w + 8)

    height = rng.randint(height_lo, height_lo + 8)
    width = rng.randint(width_lo, width_lo + 8)
    grid = np.zeros((height, width), dtype=int)

    if side == "top":
        lr0, lc0 = 0, rng.randint(0, width - legend_w)
    elif side == "bottom":
        lr0, lc0 = height - legend_h, rng.randint(0, width - legend_w)
    elif side == "left":
        lr0, lc0 = rng.randint(0, height - legend_h), 0
    else:
        lr0, lc0 = rng.randint(0, height - legend_h), width - legend_w
    grid[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w] = legend

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    _mark_blocked(blocked, lr0, lc0, lr0 + legend_h - 1, lc0 + legend_w - 1, pad=1)

    region_count = rng.randint(color_count, color_count + 2)
    region_colors = list(colors)
    while len(region_colors) < region_count:
        region_colors.append(rng.choice(colors))
    rng.shuffle(region_colors)

    for color in region_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            interior, ring, patch_h, patch_w = _make_region_shape(rng)
            if patch_h + 2 >= height or patch_w + 2 >= width:
                continue
            pr0 = rng.randint(1, height - patch_h - 1)
            pc0 = rng.randint(1, width - patch_w - 1)
            if np.any(blocked[pr0 : pr0 + patch_h, pc0 : pc0 + patch_w]):
                continue

            for r, c in ring:
                grid[pr0 + r, pc0 + c] = GREY
            seed_r, seed_c = rng.choice(tuple(interior))
            grid[pr0 + seed_r, pc0 + seed_c] = color
            _mark_blocked(blocked, pr0, pc0, pr0 + patch_h - 1, pc0 + patch_w - 1, pad=1)
            placed = True
            break
        if not placed:
            return None

    _validate_instance(grid, colors, region_count)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
