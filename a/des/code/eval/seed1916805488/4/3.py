import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_seed_mask(grid: np.ndarray, comp: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(comp)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True
    return mask


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size, "legend frame missing"
    comps = _components(grid == GREY)
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if border != set(comp):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        seed_mask = (inner != BG) & (inner != GREY)
        seed_comps = _components(seed_mask)
        if not seed_comps:
            continue
        seeds: dict[int, np.ndarray] = {}
        ok = True
        for seed in seed_comps:
            colors = {int(grid[r0 + 1 + r, c0 + 1 + c]) for r, c in seed}
            if len(colors) != 1:
                ok = False
                break
            color = next(iter(colors))
            if color in seeds:
                ok = False
                break
            seeds[color] = _extract_seed_mask(inner, seed)
        if ok:
            return (r0, c0, r1, c1), seeds
    raise AssertionError("legend not found")


def _enclosed_regions(grid: np.ndarray) -> list[tuple[int, set[tuple[int, int]], int]]:
    h, w = grid.shape
    bg_mask = grid == BG
    comps = _components(bg_mask)
    regions: list[tuple[int, set[tuple[int, int]], int]] = []
    for comp in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        border_colors: list[int] = []
        cell_set = set(comp)
        touches_grey = False
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set:
                    continue
                color = int(grid[nr, nc])
                if color == GREY:
                    touches_grey = True
                if color not in (BG, GREY):
                    border_colors.append(color)
        if touches_grey:
            continue
        if not border_colors:
            continue
        counts = Counter(border_colors)
        color, _ = counts.most_common(1)[0]
        regions.append((color, cell_set, len(comp)))
    return regions


def _tile_fill(out: np.ndarray, region: set[tuple[int, int]], color: int, seed: np.ndarray):
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    sh, sw = seed.shape
    for ar in range(r0, r1 + 1):
        for ac in range(c0, c1 + 1):
            for sr in range(sh):
                for sc in range(sw):
                    if not seed[sr, sc]:
                        continue
                    rr, cc = ar + sr, ac + sc
                    if (rr, cc) in region and out[rr, cc] == BG:
                        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox, seeds = _find_legend(g)
    out = g.copy()

    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    regions = _enclosed_regions(g)
    assert regions, "no enclosed regions found"
    for color, region, _size in regions:
        assert color in seeds, f"missing seed for color {color}"
        _tile_fill(out, region, color, seeds[color])
    return out


def _random_seed_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    target = rng.randint(3, min(h * w, max(4, h * w - 1)))
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        opts = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                opts.append((nr, nc))
        if opts:
            cells.add(rng.choice(opts))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    if mask.shape == (1, 1):
        mask = np.array([[1, 1], [1, 0]], dtype=bool)
    return mask


def _random_region_mask(rng: random.Random) -> np.ndarray:
    base_h = rng.randint(5, 8)
    base_w = rng.randint(5, 8)
    canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
    top = 3
    left = 3
    h = rng.randint(5, base_h)
    w = rng.randint(5, base_w)
    canvas[top : top + h, left : left + w] = True
    for _ in range(rng.randint(1, 3)):
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            arm_w = rng.randint(max(2, w // 3), max(2, w - 1))
            arm_h = rng.randint(1, 3)
            arm_c = left + rng.randint(0, w - arm_w)
            if side == "top":
                arm_r = top - arm_h
            else:
                arm_r = top + h
            canvas[arm_r : arm_r + arm_h, arm_c : arm_c + arm_w] = True
        else:
            arm_h = rng.randint(max(2, h // 3), max(2, h - 1))
            arm_w = rng.randint(1, 3)
            arm_r = top + rng.randint(0, h - arm_h)
            if side == "left":
                arm_c = left - arm_w
            else:
                arm_c = left + w
            canvas[arm_r : arm_r + arm_h, arm_c : arm_c + arm_w] = True
    coords = list(map(tuple, np.argwhere(canvas)))
    r0, c0, r1, c1 = _bbox(coords)
    mask = canvas[r0 : r1 + 1, c0 : c1 + 1]

    border = np.zeros_like(mask, dtype=bool)
    mh, mw = mask.shape
    for r in range(mh):
        for c in range(mw):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < mh and 0 <= nc < mw) or not mask[nr, nc]:
                    border[r, c] = True
                    break
    interior = mask & ~border
    assert interior.sum() >= 6
    return mask


def _can_place(occ: np.ndarray, top: int, left: int, shape: np.ndarray, gap: int = 1) -> bool:
    h, w = occ.shape
    sh, sw = shape.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + sh + gap)
    c1 = min(w, left + sw + gap)
    return not occ[r0:r1, c0:c1].any()


def _stamp_region(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int):
    mh, mw = mask.shape
    for r in range(mh):
        for c in range(mw):
            if not mask[r, c]:
                continue
            border = False
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < mh and 0 <= nc < mw) or not mask[nr, nc]:
                    border = True
                    break
            if border:
                grid[top + r, left + c] = color


def _place_legend(grid: np.ndarray, rng: random.Random, seeds: dict[int, np.ndarray], occ: np.ndarray):
    dims = [mask.shape for mask in seeds.values()]
    inner_h = max(h for h, _ in dims)
    inner_w = sum(w for _, w in dims) + (len(dims) - 1) * 1
    inner_h += 2
    inner_w += 2
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    corners = [
        (1, 1),
        (1, grid.shape[1] - outer_w - 1),
        (grid.shape[0] - outer_h - 1, 1),
        (grid.shape[0] - outer_h - 1, grid.shape[1] - outer_w - 1),
    ]
    rng.shuffle(corners)
    for top, left in corners:
        if top < 0 or left < 0:
            continue
        shape = np.ones((outer_h, outer_w), dtype=bool)
        if _can_place(occ, top, left, shape, gap=1):
            grid[top, left : left + outer_w] = GREY
            grid[top + outer_h - 1, left : left + outer_w] = GREY
            grid[top : top + outer_h, left] = GREY
            grid[top : top + outer_h, left + outer_w - 1] = GREY
            occ[top : top + outer_h, left : left + outer_w] = True
            cur_c = left + 2
            for color, mask in seeds.items():
                sh, sw = mask.shape
                sr = top + 2 + rng.randint(0, inner_h - 2 - sh)
                for r in range(sh):
                    for c in range(sw):
                        if mask[r, c]:
                            grid[sr + r, cur_c + c] = color
                cur_c += sw + 1
            return
    raise AssertionError("legend placement failed")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 29)
    w = rng.randint(20, 31)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    count = rng.randint(2, 3)
    colors = rng.sample(COLOR_POOL, count)
    seeds = {color: _random_seed_mask(rng) for color in colors}
    try:
        _place_legend(grid, rng, seeds, occ)
    except AssertionError:
        return None

    for color in colors:
        placed = False
        mask = _random_region_mask(rng)
        mh, mw = mask.shape
        for _ in range(80):
            top = rng.randint(1, h - mh - 2)
            left = rng.randint(1, w - mw - 2)
            if not _can_place(occ, top, left, mask, gap=2):
                continue
            _stamp_region(grid, mask, color, top, left)
            occ[top : top + mh, left : left + mw] |= mask
            placed = True
            break
        if not placed:
            return None

    regions = _enclosed_regions(grid)
    if len(regions) != count:
        return None
    found_colors = {color for color, _cells, _size in regions}
    if found_colors != set(colors):
        return None
    try:
        out = main(grid)
    except AssertionError:
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
    raise RuntimeError("failed to generate valid input")
