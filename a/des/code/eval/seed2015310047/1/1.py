import random
from collections import deque

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400

N_COLORS_CHOICES = (2, 3, 4, 5)
SEED_H_RANGE = (3, 7)
SEED_W_RANGE = (3, 7)
SEED_MARGIN_RANGE = (1, 3)
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
PLACEMENT_RETRIES = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BG)
    assert box is not None, "expected at least one non-background cell"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        border_count = 2 * (h + w) - 4
        if int(coords.shape[0]) != border_count:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == color):
            continue
        if np.any(patch[~border] == color):
            continue
        if not np.any(patch[1:-1, 1:-1] != BG):
            continue
        return int(r0), int(c0), int(r1), int(c1), int(color)
    raise AssertionError("failed to locate the unique rectangular frame")


def _find_palette(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame_box
    outside = np.zeros_like(grid, dtype=bool)
    outside[grid != BG] = True
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    box = _bbox(outside)
    assert box is not None, "expected a palette outside the frame"
    pr0, pc0, pr1, pc1 = box
    patch_mask = outside[pr0 : pr1 + 1, pc0 : pc1 + 1]
    patch = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
    assert patch_mask.all(), "palette must be a tight contiguous strip"

    if pr0 == pr1:
        colors = [int(v) for v in patch[0, :]]
    else:
        assert pc0 == pc1, "palette must be horizontal or vertical"
        colors = [int(v) for v in patch[:, 0]]

    assert len(colors) == len(set(colors)), "palette colors must be unique"
    return colors


def _expand_seed(seed: np.ndarray, palette: list[int]) -> np.ndarray:
    seed = _crop_nonzero(seed)
    pad = max(0, len(palette) - 1)
    work = np.pad(seed, pad_width=pad, mode="constant", constant_values=BG)
    for rank, color in enumerate(palette):
        radius = rank
        if radius == 0 or not np.any(work == color):
            continue
        region = work == color
        for _ in range(radius):
            frontier = _dilate4(region) & (work == BG)
            if not np.any(frontier):
                break
            work[frontier] = color
            region |= frontier
    return _crop_nonzero(work)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed color by the palette order, then crop away the carrier frame.
    """
    g = np.asarray(grid, dtype=int)
    fr0, fc0, fr1, fc1, _frame_color = _find_frame(g)
    palette = _find_palette(g, (fr0, fc0, fr1, fc1))
    seed = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    return _expand_seed(seed, palette)


def _has_hole(mask: np.ndarray) -> bool:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = padded.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if seen[rr, cc] or padded[rr, cc]:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))
    return bool(np.any((~padded) & (~seen)))


def _random_compact_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = []
        weights = []
        for r, c in np.argwhere(mask):
            for rr, cc in _neighbors4(int(r), int(c), h, w):
                if mask[rr, cc]:
                    continue
                if (rr, cc) in frontier:
                    continue
                frontier.append((rr, cc))
                adj = sum(mask[nr, nc] for nr, nc in _neighbors4(rr, cc, h, w))
                weights.append(int(adj))
        if not frontier:
            return None
        choice = random.choices(frontier, weights=weights, k=1)[0]
        mask[choice] = True
    if _has_hole(mask):
        return None
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _partition_mask(mask: np.ndarray, n_regions: int) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(cells) < 2 * n_regions:
        return None
    seeds = random.sample(cells, n_regions)
    owner = np.full(mask.shape, -1, dtype=int)
    frontiers = [deque() for _ in range(n_regions)]
    for idx, (r, c) in enumerate(seeds):
        owner[r, c] = idx
        frontiers[idx].append((r, c))

    while np.any((owner == -1) & mask):
        active = [i for i, q in enumerate(frontiers) if q]
        if not active:
            return None
        idx = random.choice(active)
        r, c = frontiers[idx].popleft()
        nbrs = [
            (rr, cc)
            for rr, cc in _neighbors4(r, c, *mask.shape)
            if mask[rr, cc] and owner[rr, cc] == -1
        ]
        random.shuffle(nbrs)
        for rr, cc in nbrs:
            owner[rr, cc] = idx
            frontiers[idx].append((rr, cc))

    counts = [(owner == idx).sum() for idx in range(n_regions)]
    if min(int(v) for v in counts) < 2:
        return None
    return owner


def _random_transform(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for _ in range(random.randrange(4)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _region_touches_bg(seed: np.ndarray, color: int) -> bool:
    mask = seed == color
    return bool(np.any(_dilate4(mask) & (seed == BG)))


def _build_seed(n_colors: int) -> tuple[np.ndarray, list[int]] | None:
    for _ in range(80):
        h = random.randint(*SEED_H_RANGE)
        w = random.randint(*SEED_W_RANGE)
        area_lo = max(2 * n_colors, h + w - 2)
        area_hi = min(h * w, 3 * n_colors + 8)
        if area_lo > area_hi:
            continue
        area = random.randint(area_lo, area_hi)
        mask = _random_compact_mask(h, w, area)
        if mask is None:
            continue
        owner = _partition_mask(mask, n_colors)
        if owner is None:
            continue
        owner = _random_transform(owner)
        used_colors = random.sample(ALL_COLORS, n_colors)
        region_to_color = {idx: used_colors[idx] for idx in range(n_colors)}
        seed = np.zeros(owner.shape, dtype=int)
        for idx, color in region_to_color.items():
            seed[owner == idx] = int(color)
        palette_region_order = random.sample(list(range(n_colors)), n_colors)
        palette = [region_to_color[idx] for idx in palette_region_order]
        if any(not _region_touches_bg(seed, color) for color in palette[1:]):
            continue
        out = _expand_seed(seed, palette)
        if out.shape == seed.shape:
            continue
        if int(np.count_nonzero(out)) <= int(np.count_nonzero(seed)):
            continue
        return seed, palette
    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_rectangles(
    grid_h: int, grid_w: int, frame_h: int, frame_w: int, pal_h: int, pal_w: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        fr = random.randint(0, grid_h - frame_h)
        fc = random.randint(0, grid_w - frame_w)
        pr = random.randint(0, grid_h - pal_h)
        pc = random.randint(0, grid_w - pal_w)
        frame_box = (fr, fc, fr + frame_h - 1, fc + frame_w - 1)
        pal_box = (pr, pc, pr + pal_h - 1, pc + pal_w - 1)
        if _separated(frame_box, pal_box, gap=1):
            return (fr, fc), (pr, pc)
    return None


def _render_input(seed: np.ndarray, palette: list[int], frame_color: int) -> np.ndarray | None:
    seed_h, seed_w = seed.shape
    pad_top = random.randint(*SEED_MARGIN_RANGE)
    pad_bottom = random.randint(*SEED_MARGIN_RANGE)
    pad_left = random.randint(*SEED_MARGIN_RANGE)
    pad_right = random.randint(*SEED_MARGIN_RANGE)
    inner_h = seed_h + pad_top + pad_bottom
    inner_w = seed_w + pad_left + pad_right
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if random.random() < 0.5:
        pal_h, pal_w = 1, len(palette)
    else:
        pal_h, pal_w = len(palette), 1

    min_h = max(frame_h, pal_h) + 3
    min_w = max(frame_w, pal_w) + 3
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
    placements = _place_rectangles(grid_h, grid_w, frame_h, frame_w, pal_h, pal_w)
    if placements is None:
        return None
    (fr, fc), (pr, pc) = placements

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[fr, fc : fc + frame_w] = frame_color
    grid[fr + frame_h - 1, fc : fc + frame_w] = frame_color
    grid[fr : fr + frame_h, fc] = frame_color
    grid[fr : fr + frame_h, fc + frame_w - 1] = frame_color

    seed_r = fr + 1 + pad_top
    seed_c = fc + 1 + pad_left
    grid[seed_r : seed_r + seed_h, seed_c : seed_c + seed_w] = seed

    if pal_h == 1:
        grid[pr, pc : pc + pal_w] = np.array(palette, dtype=int)
    else:
        grid[pr : pr + pal_h, pc] = np.array(palette, dtype=int)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice(N_COLORS_CHOICES)
    built = _build_seed(n_colors)
    if built is None:
        return None
    seed, palette = built

    used = set(map(int, np.unique(seed))) - {BG}
    frame_choices = [c for c in ALL_COLORS if c not in used]
    if not frame_choices:
        return None
    frame_color = random.choice(frame_choices)
    inp = _render_input(seed, palette, frame_color)
    if inp is None:
        return None
    expected = _expand_seed(seed, palette)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC example")
