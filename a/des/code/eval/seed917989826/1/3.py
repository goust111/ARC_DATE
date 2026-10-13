import random
from collections import Counter, deque

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
ARRANGEMENTS = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 1), (3, 2))
INNER_SIZE_RANGE = (4, 6)
LOCK_GAP_RANGE = (2, 4)
CANVAS_EXTRA_RANGE = (8, 14)
LOCK_PAD = 1
COLOR_POOL = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.array(coords, dtype=int)
        r0 = int(arr[:, 0].min())
        c0 = int(arr[:, 1].min())
        r1 = int(arr[:, 0].max())
        c1 = int(arr[:, 1].max())
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sub[arr[:, 0] - r0, arr[:, 1] - c0] = True
        comps.append({"coords": arr, "bbox": (r0, c0, r1, c1), "mask": sub})
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _bbox_inside(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    ir0, ic0, ir1, ic1 = inner
    return r0 <= ir0 <= ir1 <= r1 and c0 <= ic0 <= ic1 <= c1


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _candidate_locks(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    found: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        matches: list[dict] = []
        for size in range(4, min(h, w) + 1):
            for r0 in range(h - size + 1):
                r1 = r0 + size - 1
                for c0 in range(w - size + 1):
                    c1 = c0 + size - 1
                    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if not (
                        np.all(sub[0, :] == color)
                        and np.all(sub[-1, :] == color)
                        and np.all(sub[:, 0] == color)
                        and np.all(sub[:, -1] == color)
                    ):
                        continue
                    inner = sub[1:-1, 1:-1]
                    if inner.size == 0 or not np.any(inner == color) or not np.any(inner != color):
                        continue
                    visible = inner != color
                    if not _touches_all_sides(visible):
                        continue

                    square_bbox = (r0, c0, r1, c1)
                    outside = []
                    valid = True
                    for comp in comps:
                        bbox = comp["bbox"]
                        if _bbox_inside(square_bbox, bbox):
                            continue
                        if _bbox_disjoint(square_bbox, bbox):
                            outside.append(comp)
                            continue
                        valid = False
                        break
                    if not valid or len(outside) != 1:
                        continue
                    key = outside[0]
                    if key["mask"].shape != visible.shape:
                        continue
                    if not np.array_equal(key["mask"], visible):
                        continue

                    cleaned = inner.copy()
                    cleaned[cleaned == color] = bg
                    matches.append(
                        {
                            "color": color,
                            "bbox": square_bbox,
                            "inner": cleaned,
                        }
                    )
        unique = {}
        for item in matches:
            unique[item["bbox"]] = item
        if len(unique) == 1:
            found.append(next(iter(unique.values())))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to its same-colored lock, remove that lock's same-colored
    occlusion pixels, and assemble the revealed lock interiors in spatial order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    locks = _candidate_locks(g, bg)
    assert locks, "expected at least one lock"

    inner_shapes = {tuple(item["inner"].shape) for item in locks}
    assert len(inner_shapes) == 1, "all locks must share one interior size"
    inner_h, inner_w = next(iter(inner_shapes))

    row_positions = sorted({item["bbox"][0] for item in locks})
    col_positions = sorted({item["bbox"][1] for item in locks})
    assert len(row_positions) * len(col_positions) == len(locks), "locks must form a full row/column layout"

    lookup = {(item["bbox"][0], item["bbox"][1]): item["inner"] for item in locks}
    out = np.full((len(row_positions) * inner_h, len(col_positions) * inner_w), bg, dtype=int)
    for rr, r0 in enumerate(row_positions):
        for cc, c0 in enumerate(col_positions):
            patch = lookup[(r0, c0)]
            out[rr * inner_h : (rr + 1) * inner_h, cc * inner_w : (cc + 1) * inner_w] = patch
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _grow_connected_mask(n: int, target: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    seed = (random.randrange(n), random.randrange(n))
    mask[seed] = True
    fringe = set(_frontier(mask))
    while int(mask.sum()) < target and fringe:
        cell = random.choice(tuple(fringe))
        fringe.remove(cell)
        mask[cell] = True
        for nxt in _frontier(mask):
            if not mask[nxt]:
                fringe.add(nxt)
    return mask


def _nearest_mask_cell(mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    coords = np.argwhere(mask)
    dists = np.abs(coords[:, 0] - target[0]) + np.abs(coords[:, 1] - target[1])
    idx = int(np.argmin(dists))
    return int(coords[idx, 0]), int(coords[idx, 1])


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    row_first = random.random() < 0.5
    if row_first:
        step = 1 if tr >= r else -1
        for rr in range(r, tr + step, step):
            mask[rr, c] = True
        step = 1 if tc >= c else -1
        for cc in range(c, tc + step, step):
            mask[tr, cc] = True
    else:
        step = 1 if tc >= c else -1
        for cc in range(c, tc + step, step):
            mask[r, cc] = True
        step = 1 if tr >= r else -1
        for rr in range(r, tr + step, step):
            mask[rr, tc] = True


def _make_visible_mask(n: int) -> np.ndarray | None:
    area = n * n
    min_visible = max(n + 2, int(np.ceil(area * 0.45)))
    max_visible = min(area - 3, int(np.floor(area * 0.75)))
    if min_visible > max_visible:
        return None
    for _ in range(80):
        target = random.randint(min_visible, max_visible)
        mask = _grow_connected_mask(n, target)
        goals: list[tuple[int, int]] = []
        if not mask[0, :].any():
            goals.append((0, random.randrange(n)))
        if not mask[-1, :].any():
            goals.append((n - 1, random.randrange(n)))
        if not mask[:, 0].any():
            goals.append((random.randrange(n), 0))
        if not mask[:, -1].any():
            goals.append((random.randrange(n), n - 1))
        for goal in goals:
            start = _nearest_mask_cell(mask, goal)
            _draw_path(mask, start, goal)
        for _ in range(random.randint(0, n)):
            fringe = _frontier(mask)
            if not fringe or int(mask.sum()) >= area - 2:
                break
            cell = random.choice(fringe)
            mask[cell] = True
        if not _touches_all_sides(mask):
            continue
        if int(mask.sum()) >= area - 1:
            continue
        if int((~mask).sum()) < 2:
            continue
        return mask
    return None


def _sample_pattern(visible: np.ndarray, palette: list[int]) -> np.ndarray | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(visible)]
    if len(coords) < 2:
        return None
    n_colors = random.randint(1, min(3, len(palette), len(coords)))
    colors = random.sample(palette, n_colors)
    centers = random.sample(coords, n_colors)
    fill_prob = random.uniform(0.45, 0.8)

    patch = np.zeros(visible.shape, dtype=int)
    for color, (r, c) in zip(colors, centers):
        patch[r, c] = color
    for r, c in coords:
        if patch[r, c] != BG:
            continue
        if random.random() > fill_prob:
            continue
        dists = [abs(r - cr) + abs(c - cc) for cr, cc in centers]
        best = min(dists)
        choices = [i for i, dist in enumerate(dists) if dist == best]
        patch[r, c] = colors[random.choice(choices)]

    visible_count = int(visible.sum())
    colored_count = int(np.count_nonzero(patch))
    if colored_count == 0 or colored_count >= visible_count:
        return None
    return patch


def _build_lock(lock_color: int, visible: np.ndarray, pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = visible.shape[0]
    inner = pattern.copy()
    inner[~visible] = lock_color

    lock = np.full((n + 2, n + 2), lock_color, dtype=int)
    lock[1:-1, 1:-1] = inner
    key = np.where(visible, lock_color, BG).astype(int)

    clean = inner.copy()
    clean[clean == lock_color] = BG
    return lock, key, clean


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = LOCK_PAD) -> bool:
    h0, w0 = occ.shape
    r1 = r0 + h
    c1 = c0 + w
    return not occ[max(0, r0 - pad) : min(h0, r1 + pad), max(0, c0 - pad) : min(w0, c1 + pad)].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = LOCK_PAD) -> None:
    h0, w0 = occ.shape
    r1 = r0 + h
    c1 = c0 + w
    occ[max(0, r0 - pad) : min(h0, r1 + pad), max(0, c0 - pad) : min(w0, c1 + pad)] = True


def _free_locations(occ: np.ndarray, h: int, w: int, pad: int = LOCK_PAD) -> list[tuple[int, int]]:
    h0, w0 = occ.shape
    spots = []
    for r0 in range(1, h0 - h):
        for c0 in range(1, w0 - w):
            if _can_place(occ, r0, c0, h, w, pad=pad):
                spots.append((r0, c0))
    return spots


def _bg_dominates(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.flat))
    return counts.get(BG, 0) > max(count for color, count in counts.items() if color != BG)


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(ARRANGEMENTS)
    n_locks = rows * cols
    inner = random.randint(*INNER_SIZE_RANGE)
    outer = inner + 2
    gap_r = random.randint(*LOCK_GAP_RANGE)
    gap_c = random.randint(*LOCK_GAP_RANGE)

    lock_colors = random.sample(list(COLOR_POOL), n_locks)
    pattern_pool = [color for color in COLOR_POOL if color not in lock_colors]
    palette_size = random.randint(1, min(3, len(pattern_pool)))
    pattern_palette = random.sample(pattern_pool, palette_size)

    lock_patches: list[np.ndarray] = []
    key_patches: list[tuple[int, np.ndarray]] = []
    clean_patches: list[np.ndarray] = []
    for color in lock_colors:
        visible = _make_visible_mask(inner)
        if visible is None:
            return None
        pattern = _sample_pattern(visible, pattern_palette)
        if pattern is None:
            return None
        lock_patch, key_patch, clean_patch = _build_lock(color, visible, pattern)
        lock_patches.append(lock_patch)
        key_patches.append((color, key_patch))
        clean_patches.append(clean_patch)

    lock_area_h = rows * outer + (rows - 1) * gap_r
    lock_area_w = cols * outer + (cols - 1) * gap_c
    extra_h = inner + random.randint(*CANVAS_EXTRA_RANGE)
    extra_w = inner + random.randint(*CANVAS_EXTRA_RANGE)
    if n_locks >= 5:
        extra_h += inner // 2
        extra_w += inner // 2
    h = lock_area_h + extra_h
    w = lock_area_w + extra_w

    if h <= lock_area_h + 1 or w <= lock_area_w + 1:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    base_r = random.randint(1, h - lock_area_h - 1)
    base_c = random.randint(1, w - lock_area_w - 1)

    output = np.full((rows * inner, cols * inner), BG, dtype=int)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = base_r + rr * (outer + gap_r)
            c0 = base_c + cc * (outer + gap_c)
            _paste(grid, lock_patches[idx], r0, c0)
            _mark_occ(occ, r0, c0, outer, outer)
            output[rr * inner : (rr + 1) * inner, cc * inner : (cc + 1) * inner] = clean_patches[idx]
            idx += 1

    random.shuffle(key_patches)
    for _color, key_patch in key_patches:
        locations = _free_locations(occ, inner, inner)
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        _paste(grid, key_patch, r0, c0)
        _mark_occ(occ, r0, c0, inner, inner)

    if not _bg_dominates(grid):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
