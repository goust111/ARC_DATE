from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

DIGIT_TEMPLATES = {
    1: np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    2: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    4: np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}

PALETTE_COLORS = (BLUE, GREEN, YELLOW, MAGENTA, ORANGE, BROWN)

GRID_H_RANGE = (30, 42)
GRID_W_RANGE = (30, 44)
PALETTE_GAP_OPTIONS = (1, 2)
PALETTE_BLOCK_OFFSETS = (0, 1, 2)
PALETTE_PAD = 2
SHAPE_GEN_RETRIES = 120
PLACE_RETRIES = 120
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_local_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _inside_regions(boundary_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = boundary_mask.shape
    regions: list[list[tuple[int, int]]] = []
    for pts in _components(~boundary_mask):
        if not _touches_local_edge(pts, h, w):
            regions.append(pts)
    return regions


def _outside_background_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _parse_palette(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    best: dict[int, tuple[int, int]] = {}
    for r in range(h - 4):
        for c in range(w - 5):
            digit_area = grid[r : r + 5, c : c + 3]
            if not np.all((digit_area == BG) | (digit_area == AZURE)):
                continue
            mask = digit_area == AZURE
            digit_value = None
            for value, pattern in DIGIT_TEMPLATES.items():
                if np.array_equal(mask, pattern):
                    digit_value = value
                    break
            if digit_value is None:
                continue
            for offset in PALETTE_BLOCK_OFFSETS:
                block = grid[r + offset : r + offset + 3, c + 3 : c + 6]
                if block.shape != (3, 3):
                    continue
                color = int(block[0, 0])
                if color in (BG, AZURE) or not np.all(block == color):
                    continue
                dist = min(
                    r + c,
                    r + (w - (c + 6)),
                    (h - (r + 5)) + c,
                    (h - (r + 5)) + (w - (c + 6)),
                )
                prev = best.get(digit_value)
                if prev is None or dist < prev[0]:
                    best[digit_value] = (dist, color)
                break
    mapping = {digit: color for digit, (_dist, color) in best.items()}
    assert set(mapping) == {1, 2, 3, 4}, "palette must provide digits 1..4"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red anchors inside each grey closed region, use the corner palette to map
    that count to a color, then grow that many 4-connected outer layers from the grey border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _parse_palette(g)
    outside_bg = _outside_background_mask(g)
    out = g.copy()
    h, w = g.shape

    shapes: list[tuple[int, int, list[tuple[int, int]], int, int]] = []
    for grey_pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(grey_pts)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in grey_pts:
            local[r - r0, c - c0] = True
        inside = _inside_regions(local)
        if not inside:
            continue
        anchor_count = 0
        for region in inside:
            for lr, lc in region:
                if int(g[r0 + lr, c0 + lc]) == RED:
                    anchor_count += 1
        if 1 <= anchor_count <= 4:
            shapes.append((r0, c0, grey_pts, anchor_count, int(palette[anchor_count])))

    shapes.sort(key=lambda item: (item[0], item[1]))

    for _r0, _c0, grey_pts, layers, color in shapes:
        frontier = set(grey_pts)
        for _ in range(layers):
            next_layer: set[tuple[int, int]] = set()
            for r, c in frontier:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and outside_bg[nr, nc] and out[nr, nc] == BG:
                        next_layer.add((int(nr), int(nc)))
            if not next_layer:
                break
            for r, c in next_layer:
                out[r, c] = color
            frontier = next_layer

    return out


def _is_simple_cycle(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if len(comps) != 1:
        return False
    for r, c in comps[0]:
        degree = 0
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                degree += 1
        if degree != 2:
            return False
    return True


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _add_bump(mask: np.ndarray, side: str, r0: int, c0: int, r1: int, c1: int, rng: random.Random) -> bool:
    if side == "top":
        if r0 < 1 or c1 - c0 < 4:
            return False
        a_lo, a_hi = c0 + 1, c1 - 3
        if a_lo > a_hi:
            return False
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(a + 2, c1 - 1)
        depth = rng.randint(1, min(2, r0))
        top = r0 - depth
        mask[r0, a + 1 : b] = False
        mask[top : r0 + 1, a] = True
        mask[top : r0 + 1, b] = True
        mask[top, a : b + 1] = True
        return True
    if side == "bottom":
        if mask.shape[0] - 1 - r1 < 1 or c1 - c0 < 4:
            return False
        a_lo, a_hi = c0 + 1, c1 - 3
        if a_lo > a_hi:
            return False
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(a + 2, c1 - 1)
        depth = rng.randint(1, min(2, mask.shape[0] - 1 - r1))
        bottom = r1 + depth
        mask[r1, a + 1 : b] = False
        mask[r1 : bottom + 1, a] = True
        mask[r1 : bottom + 1, b] = True
        mask[bottom, a : b + 1] = True
        return True
    if side == "left":
        if c0 < 1 or r1 - r0 < 4:
            return False
        a_lo, a_hi = r0 + 1, r1 - 3
        if a_lo > a_hi:
            return False
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(a + 2, r1 - 1)
        depth = rng.randint(1, min(2, c0))
        left = c0 - depth
        mask[a + 1 : b, c0] = False
        mask[a, left : c0 + 1] = True
        mask[b, left : c0 + 1] = True
        mask[a : b + 1, left] = True
        return True
    if side == "right":
        if mask.shape[1] - 1 - c1 < 1 or r1 - r0 < 4:
            return False
        a_lo, a_hi = r0 + 1, r1 - 3
        if a_lo > a_hi:
            return False
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(a + 2, r1 - 1)
        depth = rng.randint(1, min(2, mask.shape[1] - 1 - c1))
        right = c1 + depth
        mask[a + 1 : b, c1] = False
        mask[a, c1 : right + 1] = True
        mask[b, c1 : right + 1] = True
        mask[a : b + 1, right] = True
        return True
    return False


def _make_shape(rng: random.Random, min_inside: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(SHAPE_GEN_RETRIES):
        h = rng.randint(12, 16)
        w = rng.randint(12, 16)
        mask = np.zeros((h, w), dtype=bool)
        r0 = rng.randint(2, 4)
        c0 = rng.randint(2, 4)
        r1 = rng.randint(h - 6, h - 3)
        c1 = rng.randint(w - 6, w - 3)
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue

        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        applied = 0
        for side in sides[: rng.randint(1, 3)]:
            applied += int(_add_bump(mask, side, r0, c0, r1, c1, rng))
        if applied == 0:
            continue

        cropped = _crop_mask(mask)
        if not _is_simple_cycle(cropped):
            continue
        inside = _inside_regions(cropped)
        if len(inside) != 1 or len(inside[0]) < min_inside:
            continue
        return cropped, inside[0]
    return None


def _sample_anchor_cells(rng: random.Random, points: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    pool = points[:]
    rng.shuffle(pool)
    chosen = [pool.pop()]
    while len(chosen) < count:
        best = None
        best_score = -1
        for pt in pool:
            score = min(abs(pt[0] - rr) + abs(pt[1] - cc) for rr, cc in chosen)
            if score > best_score:
                best_score = score
                best = pt
        assert best is not None
        chosen.append(best)
        pool.remove(best)
    return chosen


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    gh, gw = reserved.shape
    if top < pad or left < pad or top + h + pad > gh or left + w + pad > gw:
        return False
    return not reserved[top - pad : top + h + pad, left - pad : left + w + pad].any()


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    reserved[top - pad : top + h + pad, left - pad : left + w + pad] = True


def _place_palette(grid: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    h, w = grid.shape
    colors = rng.sample(PALETTE_COLORS, 4)
    gap = rng.choice(PALETTE_GAP_OPTIONS)
    palette_h = 4 * 5 + 3 * gap
    palette_w = 6
    corner = rng.choice(("tl", "tr", "bl", "br"))
    top = 0 if "t" in corner else h - palette_h
    left = 0 if "l" in corner else w - palette_w

    for idx, digit in enumerate((1, 2, 3, 4)):
        r = top + idx * (5 + gap)
        pattern = DIGIT_TEMPLATES[digit]
        digit_area = grid[r : r + 5, left : left + 3]
        digit_area[pattern] = AZURE
        offset = rng.choice(PALETTE_BLOCK_OFFSETS)
        grid[r + offset : r + offset + 3, left + 3 : left + 6] = colors[idx]

    return top, left, palette_h, palette_w


def _draw_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    anchor_count: int,
) -> bool:
    made = _make_shape(rng, min_inside=max(6, anchor_count * 2))
    if made is None:
        return False
    boundary_mask, inside = made
    sh, sw = boundary_mask.shape
    pad = anchor_count + 2
    gh, gw = grid.shape
    if gh - sh - pad < pad or gw - sw - pad < pad:
        return False

    candidates: list[tuple[int, int]] = []
    for _ in range(PLACE_RETRIES):
        top = rng.randint(pad, gh - sh - pad)
        left = rng.randint(pad, gw - sw - pad)
        if _can_place(reserved, top, left, sh, sw, pad):
            candidates.append((top, left))
    if not candidates:
        return False

    top, left = rng.choice(candidates)
    coords = np.argwhere(boundary_mask)
    for lr, lc in coords:
        grid[top + int(lr), left + int(lc)] = GREY

    for lr, lc in _sample_anchor_cells(rng, inside, anchor_count):
        grid[top + lr, left + lc] = RED

    _reserve(reserved, top, left, sh, sw, pad)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    p_top, p_left, p_h, p_w = _place_palette(grid, rng)
    _reserve(reserved, p_top, p_left, p_h, p_w, PALETTE_PAD)

    shape_count = rng.randint(2, 4)
    anchor_counts = list(range(1, 5))
    rng.shuffle(anchor_counts)
    anchor_counts = anchor_counts[:shape_count]
    for anchor_count in sorted(anchor_counts, reverse=True):
        if not _draw_shape(grid, reserved, rng, anchor_count):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
