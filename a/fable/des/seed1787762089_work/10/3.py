from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
TILE_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
TILE_COLOR_POOL = tuple(c for c in range(10) if c not in {BLACK, GREY})
NOISE_COLOR_POOL = tuple(c for c in range(1, 10))
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
SHAPE_ATTEMPTS = 100
TILE_ATTEMPTS = 100


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.ravel()))
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _infer_clean_tile(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    best_key = None
    best_tile = None
    for tile_h in _divisors(h):
        if h // tile_h < 2:
            continue
        for tile_w in _divisors(w):
            if w // tile_w < 2:
                continue
            tile = np.zeros((tile_h, tile_w), dtype=int)
            for r in range(tile_h):
                for c in range(tile_w):
                    tile[r, c] = _majority_color(region[r:h:tile_h, c:w:tile_w])
            rebuilt = np.tile(tile, (h // tile_h, w // tile_w))
            key = (int(np.count_nonzero(rebuilt != region)), tile_h * tile_w, tile_h, tile_w)
            if best_key is None or key < best_key:
                best_key = key
                best_tile = tile
    assert best_tile is not None
    return best_tile


def _split_puzzle(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = _components(grid != BLACK)
    assert len(parts) == 2

    pattern_region = None
    control_mask = None
    for coords in parts:
        colors = set(map(int, grid[tuple(coords.T)]))
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if colors.issubset({GREY}):
            assert control_mask is None
            control_mask = patch == GREY
        else:
            assert pattern_region is None
            pattern_region = patch

    assert pattern_region is not None and control_mask is not None
    return pattern_region, control_mask


def _stamp(tile: np.ndarray, control_mask: np.ndarray) -> np.ndarray:
    tile_h, tile_w = tile.shape
    mask_h, mask_w = control_mask.shape
    out = np.zeros((mask_h * tile_h, mask_w * tile_w), dtype=int)
    for r, c in zip(*np.where(control_mask)):
        out[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pattern_region, control_mask = _split_puzzle(g)
    tile = _infer_clean_tile(pattern_region)
    return _stamp(tile, control_mask)


def _has_exact_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for sub_h in _divisors(h):
        for sub_w in _divisors(w):
            if sub_h == h and sub_w == w:
                continue
            if np.array_equal(tile, np.tile(tile[:sub_h, :sub_w], (h // sub_h, w // sub_w))):
                return False
    return True


def _random_tile(rng: np.random.Generator) -> np.ndarray | None:
    tile_h, tile_w = TILE_SHAPES[int(rng.integers(len(TILE_SHAPES)))]
    color_count = int(rng.integers(2, 4))
    colors = np.array(rng.choice(TILE_COLOR_POOL, size=color_count, replace=False), dtype=int)
    for _ in range(TILE_ATTEMPTS):
        tile = np.array(rng.choice(colors, size=(tile_h, tile_w)), dtype=int)
        if len(set(map(int, tile.ravel()))) != color_count:
            continue
        if _has_exact_period(tile):
            return tile
    return None


def _random_control_mask(rng: np.random.Generator) -> np.ndarray | None:
    box_h = int(rng.integers(2, 5))
    box_w = int(rng.integers(2, 5))
    min_cells = max(2, (box_h * box_w) // 3)
    max_cells = box_h * box_w - 1
    if min_cells > max_cells:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        target = int(rng.integers(min_cells, max_cells + 1))
        mask = np.zeros((box_h, box_w), dtype=bool)
        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = []
            for r, c in zip(*np.where(mask)):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if mask.all():
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != box_h - 1 or cols.min() != 0 or cols.max() != box_w - 1:
            continue
        return mask
    return None


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + pad <= br or br + bh + pad <= ar or ac + aw + pad <= bc or bc + bw + pad <= ac


def _place_two_boxes(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    a_size: tuple[int, int],
    b_size: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ah, aw = a_size
    bh, bw = b_size
    for _ in range(PLACEMENT_ATTEMPTS):
        ar = int(rng.integers(0, grid_h - ah + 1))
        ac = int(rng.integers(0, grid_w - aw + 1))
        br = int(rng.integers(0, grid_h - bh + 1))
        bc = int(rng.integers(0, grid_w - bw + 1))
        if _boxes_separated((ar, ac, ah, aw), (br, bc, bh, bw)):
            return (ar, ac), (br, bc)
    return None


def _apply_noise(
    rng: np.random.Generator,
    clean_region: np.ndarray,
    tile: np.ndarray,
    rep_h: int,
    rep_w: int,
) -> np.ndarray | None:
    noisy = clean_region.copy()
    tile_h, tile_w = tile.shape
    copies_per_slot = rep_h * rep_w
    max_per_slot = (copies_per_slot - 1) // 2
    if max_per_slot <= 0:
        return None

    slot_counts: dict[tuple[int, int], int] = {(r, c): 0 for r in range(tile_h) for c in range(tile_w)}
    options = [(r, c) for r in range(clean_region.shape[0]) for c in range(clean_region.shape[1])]
    rng.shuffle(options)

    max_noise = min(4, sum(max_per_slot for _ in slot_counts))
    noise_count = int(rng.integers(2, max_noise + 1))
    placed = 0
    for r, c in options:
        key = (r % tile_h, c % tile_w)
        if slot_counts[key] >= max_per_slot:
            continue
        current = int(noisy[r, c])
        palette = [color for color in NOISE_COLOR_POOL if color != current and color not in set(map(int, tile.ravel()))]
        if not palette:
            continue
        noisy[r, c] = int(palette[int(rng.integers(len(palette)))])
        slot_counts[key] += 1
        placed += 1
        if placed == noise_count:
            return noisy
    return None


def _expected_output(tile: np.ndarray, control_mask: np.ndarray) -> np.ndarray:
    return _stamp(tile, control_mask)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile = _random_tile(rng)
    if tile is None:
        return None

    rep_h = int(rng.integers(2, 5))
    rep_w = int(rng.integers(2, 5))
    clean_region = np.tile(tile, (rep_h, rep_w))
    noisy_region = _apply_noise(rng, clean_region, tile, rep_h, rep_w)
    if noisy_region is None:
        return None

    control_mask = _random_control_mask(rng)
    if control_mask is None:
        return None

    region_h, region_w = noisy_region.shape
    control_h, control_w = control_mask.shape
    grid_h = region_h + control_h + int(rng.integers(5, 12))
    grid_w = region_w + control_w + int(rng.integers(5, 12))
    placement = _place_two_boxes(rng, grid_h, grid_w, (region_h, region_w), (control_h, control_w))
    if placement is None:
        return None

    (region_r, region_c), (control_r, control_c) = placement
    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[region_r : region_r + region_h, region_c : region_c + region_w] = noisy_region
    grid[control_r : control_r + control_h, control_c : control_c + control_w][control_mask] = GREY

    if not np.array_equal(_infer_clean_tile(noisy_region), tile):
        return None

    expected = _expected_output(tile, control_mask)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
