from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
TEXTURE_RETRIES = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _infer_texture_unit(block: np.ndarray) -> np.ndarray:
    bh, bw = block.shape
    for size in (2, 3):
        if bh % size != 0 or bw % size != 0:
            continue
        unit = np.zeros((size, size), dtype=int)
        ok = True
        for rr in range(size):
            for cc in range(size):
                vals = {
                    int(block[r, c])
                    for r in range(rr, bh, size)
                    for c in range(cc, bw, size)
                    if not (r == 0 and c == 0)
                }
                if len(vals) != 1:
                    ok = False
                    break
                unit[rr, cc] = next(iter(vals))
            if not ok:
                break
        if ok:
            return unit
    raise AssertionError("palette block must contain a repeated 2x2 or 3x3 texture")


def _find_palette_mapping(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != BLACK) & (g != GREY)
    palette_mask = np.zeros_like(mask, dtype=bool)
    mapping: dict[int, np.ndarray] = {}

    for pts in _components(mask):
        if not _touches_boundary(pts, h, w):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if len(pts) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        label = int(block[0, 0])
        if label in (BLACK, GREY):
            continue
        mapping[label] = _infer_texture_unit(block)
        palette_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert mapping, "expected at least one edge palette block"
    return mapping, palette_mask


def _find_regions(grid: np.ndarray, palette_mask: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = (g != GREY) & (~palette_mask)
    regions: list[dict[str, object]] = []

    for pts in _components(passable):
        if _touches_boundary(pts, h, w):
            continue
        markers = [(r, c, int(g[r, c])) for r, c in pts if int(g[r, c]) != BLACK]
        assert len(markers) == 1, "each closed region must contain exactly one colored marker"
        r0, c0, r1, c1 = _bbox(pts)
        mr, mc, color = markers[0]
        regions.append(
            {
                "points": pts,
                "bbox": (r0, c0, r1, c1),
                "marker_pos": (mr, mc),
                "marker_color": color,
            }
        )

    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, palette_mask = _find_palette_mapping(g)
    regions = _find_regions(g, palette_mask)

    out = g.copy()
    for region in regions:
        marker_color = int(region["marker_color"])
        assert marker_color in mapping, "every region marker must have a palette block"
        unit = mapping[marker_color]
        uh, uw = unit.shape
        r0, c0, _, _ = region["bbox"]
        for r, c in region["points"]:
            if out[r, c] == BLACK:
                out[r, c] = int(unit[(r - r0) % uh, (c - c0) % uw])
    return out


def _row_period(arr: np.ndarray) -> int:
    n = arr.shape[0]
    for p in range(1, n):
        if n % p != 0:
            continue
        if all(np.array_equal(arr[r], arr[r % p]) for r in range(n)):
            return p
    return n


def _col_period(arr: np.ndarray) -> int:
    n = arr.shape[1]
    for p in range(1, n):
        if n % p != 0:
            continue
        if all(np.array_equal(arr[:, c], arr[:, c % p]) for c in range(n)):
            return p
    return n


def _random_texture(size: int, forbidden: set[int], rng: random.Random) -> np.ndarray:
    palette = [color for color in COLORS if color not in forbidden]
    assert len(palette) >= 3
    for _ in range(TEXTURE_RETRIES):
        color_count = 2 if rng.random() < 0.65 else 3
        colors = rng.sample(palette, color_count)
        unit = np.array(
            [[rng.choice(colors) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        if len(set(map(int, unit.reshape(-1)))) < 2:
            continue
        if _row_period(unit) != size or _col_period(unit) != size:
            continue
        return unit
    raise RuntimeError("failed to build a non-degenerate texture unit")


def _random_connected_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(max(4, (h * w) // 3), max(4, h * w - 1))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while frontier and len(cells) < target:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], h, w):
            if nbr not in cells:
                frontier.add(nbr)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_region_patch(marker_color: int, region_count: int, rng: random.Random) -> np.ndarray:
    max_inner = 6 if region_count <= 2 else 5
    inner_h = rng.randint(3, max_inner)
    inner_w = rng.randint(3, max_inner)
    interior = _random_connected_mask(inner_h, inner_w, rng)

    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    inside = {(r + 1, c + 1) for r, c in zip(*np.where(interior))}
    boundary: set[tuple[int, int]] = set()
    for r, c in inside:
        for nr, nc in _neighbors4(r, c, patch.shape[0], patch.shape[1]):
            if (nr, nc) not in inside:
                boundary.add((nr, nc))

    for r, c in boundary:
        patch[r, c] = GREY

    marker_r, marker_c = rng.choice(tuple(inside))
    patch[marker_r, marker_c] = marker_color
    return patch


def _pack_positions(lengths: list[int], span: int, rng: random.Random) -> list[int] | None:
    if not lengths:
        return []
    base = sum(lengths) + (len(lengths) - 1)
    if base > span:
        return None
    extra = span - base
    gaps = [0] * (len(lengths) + 1)
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1
    pos = gaps[0]
    out: list[int] = []
    for idx, length in enumerate(lengths):
        out.append(pos)
        pos += length
        if idx + 1 < len(lengths):
            pos += 1 + gaps[idx + 1]
    return out


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    r0, r1 = row_range
    c0, c1 = col_range
    if ph > r1 - r0 + 1 or pw > c1 - c0 + 1:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(r0, r1 - ph + 1)
        left = rng.randint(c0, c1 - pw + 1)
        if blocked[top : top + ph, left : left + pw].any():
            continue
        mask = patch != BLACK
        if np.any(grid[top : top + ph, left : left + pw][mask] != BLACK):
            continue
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        blocked[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)] = True
        return True
    return False


def _layout_canvas(blocks: list[np.ndarray], region_count: int, side: str, rng: random.Random):
    long_need = 10 + 4 * (region_count - 1)
    if side in ("top", "bottom"):
        lengths = [block.shape[1] for block in blocks]
        strip = max(block.shape[0] for block in blocks) + 1
        min_w = max(18, long_need, sum(lengths) + len(lengths) - 1)
        min_h = max(16, strip + 10 + 3 * (region_count - 1))
        if min_w > 30 or min_h > 30:
            return None
        w = rng.randint(min_w, 30)
        h = rng.randint(min_h, 30)
        xs = _pack_positions(lengths, w, rng)
        if xs is None:
            return None
        grid = np.zeros((h, w), dtype=int)
        for block, x in zip(blocks, xs):
            y = 0 if side == "top" else h - block.shape[0]
            grid[y : y + block.shape[0], x : x + block.shape[1]] = block
        row_range = (strip, h - 2) if side == "top" else (1, h - strip - 1)
        col_range = (1, w - 2)
        return grid, row_range, col_range

    lengths = [block.shape[0] for block in blocks]
    strip = max(block.shape[1] for block in blocks) + 1
    min_h = max(18, long_need, sum(lengths) + len(lengths) - 1)
    min_w = max(16, strip + 10 + 3 * (region_count - 1))
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    ys = _pack_positions(lengths, h, rng)
    if ys is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    for block, y in zip(blocks, ys):
        x = 0 if side == "left" else w - block.shape[1]
        grid[y : y + block.shape[0], x : x + block.shape[1]] = block
    row_range = (1, h - 2)
    col_range = (strip, w - 2) if side == "left" else (1, w - strip - 1)
    return grid, row_range, col_range


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.choice((1, 2, 3))
    marker_colors = rng.sample(COLORS, region_count)
    side = rng.choice(SIDES)
    forbidden = {BLACK, GREY, *marker_colors}

    blocks: list[np.ndarray] = []
    for marker_color in marker_colors:
        size = rng.choice((2, 3))
        unit = _random_texture(size, forbidden, rng)
        tile_rows = rng.choice((2, 3))
        tile_cols = rng.choice((2, 3))
        block = np.tile(unit, (tile_rows, tile_cols))
        block[0, 0] = marker_color
        blocks.append(block)

    rng.shuffle(blocks)
    layout = _layout_canvas(blocks, region_count, side, rng)
    if layout is None:
        return None
    grid, row_range, col_range = layout
    blocked = np.zeros_like(grid, dtype=bool)

    for marker_color in marker_colors:
        patch = _make_region_patch(marker_color, region_count, rng)
        if not _place_region(grid, blocked, patch, row_range, col_range, rng):
            return None

    try:
        mapping, palette_mask = _find_palette_mapping(grid)
        regions = _find_regions(grid, palette_mask)
        if set(mapping.keys()) != set(marker_colors):
            return None
        if len(regions) != region_count:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[palette_mask], grid[palette_mask]):
        return None
    if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return None

    for region in regions:
        pts = region["points"]
        marker_pos = region["marker_pos"]
        if sum(1 for r, c in pts if grid[r, c] == BLACK) == 0:
            return None
        if any((r, c) != marker_pos and out[r, c] == BLACK for r, c in pts):
            return None
        if out[marker_pos] != grid[marker_pos]:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
