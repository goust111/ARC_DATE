import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
TEAL = 8

OBJECT_COLORS = (BLUE, GREEN, ORANGE, TEAL)
FOREGROUND_COLORS = OBJECT_COLORS
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

PATTERN_LIBRARY = [
    ((0, 1), (1, 2), (0,)),
    ((0, 1), (1, 2), (2,)),
    ((0, 1), (1, 2), (1,)),
    ((0, 1), (1, 2), (0,), (2,)),
    ((0, 1), (1, 2), (0,), (1,)),
    ((0, 1, 2), (0, 1), (2,)),
    ((0, 1, 2), (1, 2), (0,)),
    ((0, 1, 2), (0, 1), (1,), (2,)),
    ((0, 1, 2), (1, 2), (0,), (2,)),
]

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == RED)]
    assert len(rows) == 2, "expected exactly two full-width red separator rows"
    return rows


def _band_ranges(grid: np.ndarray) -> list[tuple[int, int]]:
    s0, s1 = _find_separator_rows(grid)
    bands = [(0, s0), (s0 + 1, s1), (s1 + 1, grid.shape[0])]
    assert all(r1 - r0 >= 1 for r0, r1 in bands)
    return bands


def _band_index_for_row(row: int, separators: list[int]) -> int:
    if row < separators[0]:
        return 0
    if row > separators[1]:
        return 2
    return 1


def _gravity_band(band: np.ndarray) -> np.ndarray:
    h, w = band.shape
    out = np.zeros((h, w), dtype=int)
    for c in range(w):
        values = [int(v) for v in band[:, c] if int(v) not in (BLACK, RED)]
        if values:
            out[h - len(values) :, c] = values
    return out


def _post_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    separators = _find_separator_rows(g)
    out[separators[0], :] = RED
    out[separators[1], :] = RED
    for r0, r1 in _band_ranges(g):
        out[r0:r1, :] = _gravity_band(g[r0:r1, :])
    return out


def _band_foreground_count(grid: np.ndarray, band: tuple[int, int]) -> int:
    r0, r1 = band
    patch = grid[r0:r1, :]
    return int(np.count_nonzero((patch != BLACK) & (patch != RED)))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fallen = _post_gravity(g)
    bands = _band_ranges(g)
    counts = [_band_foreground_count(fallen, band) for band in bands]
    assert len(set(counts)) == 3, "band foreground counts must be distinct"

    winner = int(np.argmax(counts))
    out = fallen.copy()
    for i, (r0, r1) in enumerate(bands):
        band = out[r0:r1, :]
        fg_mask = (band != BLACK) & (band != RED)
        if i == winner:
            band[band == BLACK] = GREY
        else:
            band[fg_mask] = BLACK
        out[r0:r1, :] = band
    return out


def _component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _foreground_by_color(grid: np.ndarray) -> dict[int, np.ndarray]:
    colors = sorted(set(map(int, grid.ravel())) - {BLACK, RED})
    return {color: np.argwhere(grid == color) for color in colors}


def main(grid: np.ndarray) -> np.ndarray:
    """Split separator-crossing color objects into band fragments, drop each band, and keep only the densest band filled.

    Input structure:
    - The grid has black background and exactly two full-width red separator rows that partition it into three horizontal bands.
    - Foreground objects use distinct colors from {blue, green, orange, teal}; some colors appear in more than one band because a separator visually slices that object.
    - Separator rows contain no object pixels and remain fixed barriers.

    Rule:
    - Treat each band independently: every non-black, non-red pixel falls straight downward within its own column until it reaches the band floor or another settled pixel.
    - Count the total foreground pixels in each band after this band-wise gravity.
    - Keep the unique densest band, recolor every remaining black cell in that band to grey, and clear the foreground pixels from the other two bands to black.
    - Leave both red separator rows unchanged.

    Key insight:
    - The red lines do not move; they split any multi-band color object into separate band fragments, so the winner is determined only by the post-split per-band pixel counts.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sample_band_heights(rng: random.Random) -> tuple[int, int, int]:
    while True:
        heights = [rng.randint(4, 6) for _ in range(3)]
        if sum(heights) + 2 <= 20:
            return tuple(heights)


def _sample_centers(rng: random.Random, width: int, count: int) -> list[int] | None:
    candidates = list(range(2, width - 2))
    rng.shuffle(candidates)
    centers: list[int] = []
    for c in candidates:
        if all(abs(c - other) >= 3 for other in centers):
            centers.append(c)
            if len(centers) == count:
                return centers
    return None


def _allocate_sizes(patterns: list[tuple[int, ...]], rng: random.Random) -> list[dict[int, int]] | None:
    for _ in range(200):
        allocations: list[dict[int, int]] = []
        totals = [0, 0, 0]
        for pattern in patterns:
            bands = tuple(pattern)
            if len(bands) == 1:
                total = rng.randint(4, 8)
                split = {bands[0]: total}
            elif len(bands) == 2:
                candidates = [(a, b) for a in range(2, 5) for b in range(2, 5) if 4 <= a + b <= 8]
                a, b = rng.choice(candidates)
                split = {bands[0]: a, bands[1]: b}
            else:
                candidates = [
                    (a, b, c)
                    for a in range(1, 4)
                    for b in range(1, 4)
                    for c in range(1, 4)
                    if 4 <= a + b + c <= 8
                ]
                a, b, c = rng.choice(candidates)
                split = {bands[0]: a, bands[1]: b, bands[2]: c}

            for band, size in split.items():
                totals[band] += int(size)
            allocations.append(split)

        if min(totals) >= 1 and len(set(totals)) == 3:
            return allocations
    return None


def _random_connected_shape(area: int, max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    if area == 1:
        return np.array([[True]], dtype=bool)

    dims = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= area]
    rng.shuffle(dims)
    for h, w in dims:
        for _ in range(60):
            mask = np.zeros((h, w), dtype=bool)
            seed = (rng.randrange(h), rng.randrange(w))
            mask[seed] = True
            while int(mask.sum()) < area:
                frontier = []
                for rr, cc in np.argwhere(mask):
                    for dr, dc in NEIGHBORS4:
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                mask[rng.choice(frontier)] = True
            if int(mask.sum()) != area:
                continue
            box = _bbox(mask)
            assert box is not None
            r0, c0, r1, c1 = box
            return mask[r0 : r1 + 1, c0 : c1 + 1]
    return None


def _vertical_preferences(pattern: tuple[int, ...]) -> dict[int, str]:
    if pattern == (0, 1):
        return {0: "bottom", 1: "top"}
    if pattern == (1, 2):
        return {1: "bottom", 2: "top"}
    if pattern == (0, 1, 2):
        return {0: "bottom", 1: "middle", 2: "top"}
    return {pattern[0]: "free"}


def _preferred_rows(r0: int, r1: int, shape_h: int, mode: str) -> list[int]:
    positions = list(range(r0, r1 - shape_h + 1))
    if not positions:
        return []
    if mode == "top":
        k = min(3, len(positions))
        return positions[:k]
    if mode == "bottom":
        k = min(3, len(positions))
        return positions[-k:]
    if mode == "middle":
        mid = (r0 + r1 - shape_h) // 2
        ordered = sorted(positions, key=lambda x: abs(x - mid))
        return ordered[: min(4, len(ordered))]
    return positions


def _can_place(used: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = used.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if used[r, c]:
            return False
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and used[nr, nc]:
                return False
    return True


def _place_segments(
    grid: np.ndarray,
    used: np.ndarray,
    bands: list[tuple[int, int]],
    center: int,
    color: int,
    pattern: tuple[int, ...],
    allocation: dict[int, int],
    rng: random.Random,
) -> bool:
    prefs = _vertical_preferences(pattern)
    for band in pattern:
        area = int(allocation[band])
        band_r0, band_r1 = bands[band]
        band_h = band_r1 - band_r0
        max_h = min(band_h, 4 if area >= 5 else 3 if area >= 3 else 2)
        max_w = min(grid.shape[1], 4 if area >= 5 else 3 if area >= 3 else 2)
        shape = _random_connected_shape(area, max_h, max_w, rng)
        if shape is None:
            return False
        sh, sw = shape.shape

        row_choices = _preferred_rows(band_r0, band_r1, sh, prefs.get(band, "free"))
        if not row_choices:
            return False

        base_c = max(0, min(grid.shape[1] - sw, center - sw // 2))
        col_choices = []
        for delta in [0, -1, 1, -2, 2, -3, 3]:
            c0 = base_c + delta
            if 0 <= c0 <= grid.shape[1] - sw:
                col_choices.append(c0)
        col_choices = list(dict.fromkeys(col_choices))

        placed = False
        attempts = [(r0, c0) for r0 in row_choices for c0 in col_choices]
        rng.shuffle(attempts)
        for abs_r0, abs_c0 in attempts:
            coords = [(abs_r0 + rr, abs_c0 + cc) for rr, cc in np.argwhere(shape)]
            if not _can_place(used, coords):
                continue
            for r, c in coords:
                grid[r, c] = color
                used[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _semantic_validation(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if GREY in set(map(int, inp.ravel())):
        return False

    separators = _find_separator_rows(inp)
    bands = _band_ranges(inp)
    if any((r1 - r0) < 4 for r0, r1 in bands):
        return False

    colors = _foreground_by_color(inp)
    if not (3 <= len(colors) <= 4):
        return False
    if any(color not in FOREGROUND_COLORS for color in colors):
        return False

    crossers = 0
    for color, coords in colors.items():
        if not (4 <= len(coords) <= 8):
            return False
        occupied_bands = sorted({_band_index_for_row(int(r), separators) for r, _c in coords})
        if occupied_bands != list(range(occupied_bands[0], occupied_bands[-1] + 1)):
            return False
        if len(occupied_bands) > 1:
            crossers += 1
        for band in occupied_bands:
            r0, r1 = bands[band]
            mask = inp[r0:r1, :] == color
            if _component_count(mask) != 1:
                return False
    if crossers < 2:
        return False

    fallen = _post_gravity(inp)
    counts = [_band_foreground_count(fallen, band) for band in bands]
    if min(counts) < 1 or len(set(counts)) != 3:
        return False

    out = main(inp)
    if not np.array_equal(out[separators[0], :], inp[separators[0], :]):
        return False
    if not np.array_equal(out[separators[1], :], inp[separators[1], :]):
        return False

    winner = int(np.argmax(counts))
    for band_idx, (r0, r1) in enumerate(bands):
        before = fallen[r0:r1, :]
        after = out[r0:r1, :]
        fg = (before != BLACK) & (before != RED)
        if band_idx == winner:
            if not np.array_equal(after[fg], before[fg]):
                return False
            if np.any(after[~fg] != GREY):
                return False
        else:
            if np.any((after != BLACK) & (after != RED)):
                return False
    return True


def _generate_once(rng: random.Random) -> np.ndarray | None:
    band_heights = _sample_band_heights(rng)
    height = sum(band_heights) + 2
    width = rng.randint(14, 18)
    grid = np.zeros((height, width), dtype=int)

    sep0 = band_heights[0]
    sep1 = band_heights[0] + 1 + band_heights[1]
    grid[sep0, :] = RED
    grid[sep1, :] = RED
    bands = _band_ranges(grid)

    patterns = list(rng.choice(PATTERN_LIBRARY))
    rng.shuffle(patterns)
    colors = rng.sample(list(OBJECT_COLORS), k=len(patterns))
    centers = _sample_centers(rng, width, len(patterns))
    if centers is None:
        return None

    allocations = _allocate_sizes(patterns, rng)
    if allocations is None:
        return None

    order = sorted(
        range(len(patterns)),
        key=lambda i: (-len(patterns[i]), -sum(allocations[i].values()), centers[i]),
    )

    used = np.zeros_like(grid, dtype=bool)
    for i in order:
        ok = _place_segments(
            grid,
            used,
            bands,
            centers[i],
            colors[i],
            tuple(patterns[i]),
            allocations[i],
            rng,
        )
        if not ok:
            return None

    if not _semantic_validation(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
