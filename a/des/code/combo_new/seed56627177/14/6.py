import random

import numpy as np


BACKGROUND = 0
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 300
REGION_KINDS = ("walk", "rect", "line", "diag", "lshape")
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _identity(arr: np.ndarray) -> np.ndarray:
    return arr


def _flip_both(arr: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(arr))


QUADRANT_FOLDS = (_identity, np.fliplr, np.flipud, _flip_both)
QUADRANT_SLICES = (
    lambda q: (slice(0, q), slice(0, q)),
    lambda q: (slice(0, q), slice(q, 2 * q)),
    lambda q: (slice(q, 2 * q), slice(0, q)),
    lambda q: (slice(q, 2 * q), slice(q, 2 * q)),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrants toward the center and merge aligned colored pixels.

    Input structure:
    - The input is an even-sized square grid with black background.
    - The horizontal and vertical midpoint boundaries split the grid into four equal quadrants.
    - Each quadrant contains sparse non-background pixels of a single foreground color, and the four quadrant colors are distinct.

    Rule:
    - Output a grid whose size matches one quadrant of the input.
    - Overlay the top-left quadrant as-is, the top-right quadrant after a left-right flip, the bottom-left quadrant after an up-down flip, and the bottom-right quadrant after both flips.
    - If exactly one folded quadrant contributes a non-background pixel at an output position, keep that color.
    - If two or more folded quadrants contribute at the same position, paint that output cell orange.
    - If no quadrant contributes at a position, leave it black.

    Key insight:
    - The four quadrants are different views of one hidden coordinate system; after folding them inward, only the overlap count matters for deciding whether a color survives or turns orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0

    q = h // 2
    folded_quadrants = (
        g[:q, :q],
        np.fliplr(g[:q, q:]),
        np.flipud(g[q:, :q]),
        np.flipud(np.fliplr(g[q:, q:])),
    )

    out = np.full((q, q), BACKGROUND, dtype=int)
    seen = np.zeros((q, q), dtype=bool)
    for quad in folded_quadrants:
        mask = quad != BACKGROUND
        overlap = seen & mask
        fresh = (~seen) & mask
        out[fresh] = quad[fresh]
        out[overlap] = ORANGE
        seen |= mask

    return out


def _owner_bits(owners: list[int]) -> int:
    bits = 0
    for owner in owners:
        bits |= 1 << int(owner)
    return bits


def _owner_count_map(owner_map: np.ndarray) -> np.ndarray:
    counts = np.zeros(owner_map.shape, dtype=int)
    for idx in range(4):
        counts += (owner_map >> idx) & 1
    return counts


def _expected_output(owner_map: np.ndarray, colors: list[int]) -> np.ndarray:
    q = owner_map.shape[0]
    out = np.full((q, q), BACKGROUND, dtype=int)
    counts = _owner_count_map(owner_map)
    singles = counts == 1
    overlaps = counts >= 2
    out[overlaps] = ORANGE
    for idx, color in enumerate(colors):
        out[singles & (owner_map == (1 << idx))] = int(color)
    return out


def _random_available_cell(available: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    coords = np.argwhere(available)
    if len(coords) == 0:
        return None
    r, c = coords[rng.randrange(len(coords))]
    return int(r), int(c)


def _coords_to_mask(shape: tuple[int, int], coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _walk_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    seed = _random_available_cell(available, rng)
    if seed is None:
        return None

    target = rng.randint(min_area, max_area)
    q = available.shape[0]
    region = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS_4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < q and 0 <= cc < q and available[rr, cc] and (rr, cc) not in region:
                frontier.add((rr, cc))

    add_frontier(*seed)
    while frontier and len(region) < target:
        cell = tuple(rng.choice(tuple(frontier)))
        frontier.discard(cell)
        region.add(cell)
        add_frontier(*cell)

    if len(region) < min_area:
        return None
    return _coords_to_mask(available.shape, region)


def _rect_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    q = available.shape[0]
    for _ in range(25):
        h = rng.randint(1, q)
        w = rng.randint(1, q)
        area = h * w
        if not (min_area <= area <= max_area):
            continue
        r0 = rng.randint(0, q - h)
        c0 = rng.randint(0, q - w)
        mask = np.zeros_like(available)
        mask[r0 : r0 + h, c0 : c0 + w] = True
        if np.all(available[mask]):
            return mask
    return None


def _line_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    q = available.shape[0]
    max_len = min(max_area, q)
    if max_len < min_area:
        return None
    for _ in range(25):
        length = rng.randint(min_area, max_len)
        horizontal = rng.random() < 0.5
        if horizontal:
            r = rng.randint(0, q - 1)
            c0 = rng.randint(0, q - length)
            mask = np.zeros_like(available)
            mask[r, c0 : c0 + length] = True
        else:
            r0 = rng.randint(0, q - length)
            c = rng.randint(0, q - 1)
            mask = np.zeros_like(available)
            mask[r0 : r0 + length, c] = True
        if np.all(available[mask]):
            return mask
    return None


def _diag_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    q = available.shape[0]
    max_len = min(max_area, q)
    if max_len < min_area:
        return None
    for _ in range(25):
        length = rng.randint(min_area, max_len)
        r0 = rng.randint(0, q - length)
        c0 = rng.randint(0, q - length)
        mask = np.zeros_like(available)
        if rng.random() < 0.5:
            for step in range(length):
                mask[r0 + step, c0 + step] = True
        else:
            c0 = rng.randint(length - 1, q - 1)
            for step in range(length):
                mask[r0 + step, c0 - step] = True
        if np.all(available[mask]):
            return mask
    return None


def _lshape_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    q = available.shape[0]
    if max_area < 3:
        return None
    for _ in range(25):
        v_len = rng.randint(2, q)
        h_len = rng.randint(2, q)
        area = v_len + h_len - 1
        if not (min_area <= area <= max_area):
            continue
        r0 = rng.randint(0, q - v_len)
        c0 = rng.randint(0, q - h_len)
        mask = np.zeros_like(available)
        orientation = rng.randrange(4)
        if orientation == 0:
            mask[r0 : r0 + v_len, c0] = True
            mask[r0, c0 : c0 + h_len] = True
        elif orientation == 1:
            mask[r0 : r0 + v_len, c0 + h_len - 1] = True
            mask[r0, c0 : c0 + h_len] = True
        elif orientation == 2:
            mask[r0 : r0 + v_len, c0] = True
            mask[r0 + v_len - 1, c0 : c0 + h_len] = True
        else:
            mask[r0 : r0 + v_len, c0 + h_len - 1] = True
            mask[r0 + v_len - 1, c0 : c0 + h_len] = True
        if np.all(available[mask]):
            return mask
    return None


def _sample_region(available: np.ndarray, rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    free_cells = int(np.count_nonzero(available))
    if free_cells == 0:
        return None

    max_area = min(max_area, free_cells)
    min_area = min(min_area, max_area)
    builders = {
        "walk": _walk_region,
        "rect": _rect_region,
        "line": _line_region,
        "diag": _diag_region,
        "lshape": _lshape_region,
    }
    for _ in range(3):
        kinds = list(REGION_KINDS)
        rng.shuffle(kinds)
        for kind in kinds:
            mask = builders[kind](available, rng, min_area, max_area)
            if mask is not None:
                return mask

    if min_area > 1:
        return None
    seed = _random_available_cell(available, rng)
    if seed is None:
        return None
    mask = np.zeros_like(available)
    mask[seed] = True
    return mask


def _place_region(owner_map: np.ndarray, mask: np.ndarray, bits: int) -> None:
    assert np.all(owner_map[mask] == 0)
    owner_map[mask] = bits


def _build_owner_map(q: int, rng: random.Random) -> np.ndarray | None:
    owner_map = np.zeros((q, q), dtype=np.uint8)

    overlap_max = max(2, min(q + 2, q * q // 4 + 1))
    overlap_mask = _sample_region(owner_map == 0, rng, max(2, q // 3), overlap_max)
    if overlap_mask is None:
        return None
    overlap_owners = rng.sample(range(4), rng.randint(2, 4))
    _place_region(owner_map, overlap_mask, _owner_bits(overlap_owners))

    unique_order = list(range(4))
    rng.shuffle(unique_order)
    for quad_idx in unique_order:
        unique_max = max(1, min(q, q * q // 8 + 2))
        unique_mask = _sample_region(owner_map == 0, rng, 1, unique_max)
        if unique_mask is None:
            return None
        _place_region(owner_map, unique_mask, 1 << quad_idx)

    extra_regions = rng.randint(0, min(4, q))
    for _ in range(extra_regions):
        available = owner_map == 0
        if np.count_nonzero(available) <= 1:
            break
        extra_max = max(1, min(q, q * q // 7 + 1))
        extra_mask = _sample_region(available, rng, 1, extra_max)
        if extra_mask is None:
            continue
        if rng.random() < 0.5:
            owners = [rng.randrange(4)]
        else:
            owners = rng.sample(range(4), rng.randint(2, 4))
        _place_region(owner_map, extra_mask, _owner_bits(owners))

    counts = _owner_count_map(owner_map)
    if not np.any(counts == 0):
        return None
    if not np.any(counts >= 2):
        return None
    for quad_idx in range(4):
        if not np.any(owner_map == (1 << quad_idx)):
            return None

    covered = int(np.count_nonzero(counts))
    if covered < max(4, q) or covered > q * q - 1:
        return None
    return owner_map


def _quadrant_from_canonical(mask: np.ndarray, quad_idx: int) -> np.ndarray:
    return QUADRANT_FOLDS[quad_idx](mask)


def _render_input(owner_map: np.ndarray, colors: list[int]) -> np.ndarray:
    q = owner_map.shape[0]
    grid = np.full((2 * q, 2 * q), BACKGROUND, dtype=int)
    for quad_idx, color in enumerate(colors):
        canonical = np.full((q, q), BACKGROUND, dtype=int)
        canonical[(owner_map & (1 << quad_idx)) != 0] = int(color)
        quadrant = _quadrant_from_canonical(canonical, quad_idx)
        rs, cs = QUADRANT_SLICES[quad_idx](q)
        grid[rs, cs] = quadrant
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h != w or h % 2 != 0 or h > 30:
        return False
    q = h // 2
    quadrants = (
        grid[:q, :q],
        grid[:q, q:],
        grid[q:, :q],
        grid[q:, q:],
    )
    seen_colors = []
    for quad in quadrants:
        colors = {int(v) for v in np.unique(quad) if int(v) != BACKGROUND}
        if len(colors) != 1:
            return False
        seen_colors.append(next(iter(colors)))
    return len(set(seen_colors)) == 4


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    q = rng.randint(4, 15)
    owner_map = _build_owner_map(q, rng)
    if owner_map is None:
        return None

    colors = rng.sample(list(INPUT_COLORS), 4)
    grid = _render_input(owner_map, colors)
    if not _valid_input(grid):
        return None

    expected = _expected_output(owner_map, colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not (np.any(actual == BACKGROUND) and np.any(actual == ORANGE)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-quadrant ARC input")
