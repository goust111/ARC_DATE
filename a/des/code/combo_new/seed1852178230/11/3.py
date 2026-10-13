from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MOTIF_COLORS = (5, 6, 7, 8, 9)

ARC_GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 300
MAX_TILE_ATTEMPTS = 200


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _is_nonperiodic(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for sub_h in _divisors(h):
        for sub_w in _divisors(w):
            if sub_h == h and sub_w == w:
                continue
            base = tile[:sub_h, :sub_w]
            tiled = np.tile(base, (h // sub_h, w // sub_w))
            if np.array_equal(tile, tiled):
                return False
    return True


def _recover_candidate(
    grid: np.ndarray, tile_h: int, tile_w: int
) -> tuple[np.ndarray, list[int]] | None:
    h, w = grid.shape
    if h % tile_h != 0 or w % tile_w != 0:
        return None

    tile_rows = h // tile_h
    tile_cols = w // tile_w
    if tile_rows * tile_cols < 2:
        return None

    pattern = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            votes = Counter(
                int(v) for v in grid[r::tile_h, c::tile_w].ravel() if int(v) != RED
            )
            if not votes:
                return None
            color, _count = votes.most_common(1)[0]
            if len(votes) != 1:
                return None
            pattern[r, c] = int(color)

    if not np.any(pattern != BG):
        return None

    noise_counts: list[int] = []
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            tile = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]
            red_mask = tile == RED
            if np.any(tile[~red_mask] != pattern[~red_mask]):
                return None
            noise_counts.append(int(red_mask.sum()))

    if min(noise_counts) != 0:
        return None

    return pattern, noise_counts


def _infer_tile_structure(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray, list[int]]] = []
    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            recovered = _recover_candidate(grid, tile_h, tile_w)
            if recovered is None:
                continue
            pattern, noise_counts = recovered
            candidates.append((tile_h * tile_w, tile_h, tile_w, pattern, noise_counts))

    assert candidates, "expected at least one valid tile factorization"

    # Prefer the smallest repeated unit; larger bundles of whole tiles also fit.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _area, tile_h, tile_w, pattern, noise_counts = candidates[0]
    return tile_h, tile_w, pattern, noise_counts


def _shade_pattern(pattern: np.ndarray, noise_count: int) -> np.ndarray:
    if noise_count == 0:
        return pattern.copy()

    if noise_count == 1:
        fill = GREEN
    elif noise_count == 2:
        fill = BLUE
    else:
        fill = YELLOW

    out = pattern.copy()
    out[pattern != BG] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a repeated tile pattern corrupted by red noise, then recolor each tile by noise count.

    Input structure:
    - Black background cells are part of the repeated tile pattern and also fill empty tile space.
    - The full grid is a tight rectangular tiling of identical tile copies with no separator lines.
    - Some cells inside each tile are replaced by red noise; red never belongs to the clean pattern.
    - At least one tile is fully clean, so every true tile position is visible somewhere in the grid.

    Rule:
    - Enumerate factorizations of the grid shape into equal rectangular tiles and choose the smallest one whose copies agree everywhere after ignoring red cells.
    - Recover the clean tile by taking the shared non-red color at each tile-relative position across all copies.
    - Count the red cells in each tile: keep the clean colors when N=0, replace every non-black tile cell by green when N=1, by blue when N=2, and by yellow when N>=3.
    - Reassemble the output by placing the restored and recolored tile back into every original tile slot.

    Key insight:
    - Red is not part of the motif; comparing many aligned copies reveals the smallest clean repeating unit, and the amount of red damage then serves as an independent class label for recoloring that whole unit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile_h, tile_w, pattern, noise_counts = _infer_tile_structure(g)
    tile_rows = g.shape[0] // tile_h
    tile_cols = g.shape[1] // tile_w

    out = np.empty_like(g)
    index = 0
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = _shade_pattern(
                pattern, noise_counts[index]
            )
            index += 1
    return out


def _draw_primitive(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    kind = int(np.random.randint(0, 5))

    if kind == 0:
        rect_h = int(np.random.randint(1, h + 1))
        rect_w = int(np.random.randint(1, w + 1))
        r0 = int(np.random.randint(0, h - rect_h + 1))
        c0 = int(np.random.randint(0, w - rect_w + 1))
        tile[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        return

    if kind == 1:
        row = int(np.random.randint(0, h))
        length = int(np.random.randint(2, w + 1))
        c0 = int(np.random.randint(0, w - length + 1))
        tile[row, c0 : c0 + length] = color
        return

    if kind == 2:
        col = int(np.random.randint(0, w))
        length = int(np.random.randint(2, h + 1))
        r0 = int(np.random.randint(0, h - length + 1))
        tile[r0 : r0 + length, col] = color
        return

    if kind == 3:
        row = int(np.random.randint(0, h))
        col = int(np.random.randint(0, w))
        max_up = row
        max_down = h - row - 1
        max_left = col
        max_right = w - col - 1
        if max(max_up, max_down, max_left, max_right) == 0:
            tile[row, col] = color
            return
        arm_v = int(np.random.randint(0, max(max_up, max_down) + 1))
        arm_h = int(np.random.randint(0, max(max_left, max_right) + 1))
        tile[max(0, row - arm_v) : min(h, row + arm_v + 1), col] = color
        tile[row, max(0, col - arm_h) : min(w, col + arm_h + 1)] = color
        return

    arm_h = int(np.random.randint(2, w + 1))
    arm_v = int(np.random.randint(2, h + 1))
    orient = int(np.random.randint(0, 4))
    if orient == 0:
        r0 = int(np.random.randint(0, h - arm_v + 1))
        c0 = int(np.random.randint(0, w - arm_h + 1))
        tile[r0, c0 : c0 + arm_h] = color
        tile[r0 : r0 + arm_v, c0] = color
    elif orient == 1:
        r0 = int(np.random.randint(0, h - arm_v + 1))
        c1 = int(np.random.randint(arm_h - 1, w))
        tile[r0, c1 - arm_h + 1 : c1 + 1] = color
        tile[r0 : r0 + arm_v, c1] = color
    elif orient == 2:
        r1 = int(np.random.randint(arm_v - 1, h))
        c0 = int(np.random.randint(0, w - arm_h + 1))
        tile[r1, c0 : c0 + arm_h] = color
        tile[r1 - arm_v + 1 : r1 + 1, c0] = color
    else:
        r1 = int(np.random.randint(arm_v - 1, h))
        c1 = int(np.random.randint(arm_h - 1, w))
        tile[r1, c1 - arm_h + 1 : c1 + 1] = color
        tile[r1 - arm_v + 1 : r1 + 1, c1] = color


def _generate_clean_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    area = tile_h * tile_w
    for _ in range(MAX_TILE_ATTEMPTS):
        tile = np.full((tile_h, tile_w), BG, dtype=int)
        palette = np.random.choice(
            MOTIF_COLORS, size=int(np.random.randint(2, 4)), replace=False
        )
        primitive_count = int(np.random.randint(3, 6))
        for _ in range(primitive_count):
            _draw_primitive(tile, int(palette[int(np.random.randint(0, len(palette)))]))

        filled = int(np.count_nonzero(tile != BG))
        colors = set(map(int, tile.ravel())) - {BG}
        if len(colors) < 2:
            continue
        if filled < max(4, area // 5):
            continue
        if filled > (3 * area) // 4:
            continue
        if not _is_nonperiodic(tile):
            continue
        return tile
    return None


def _sample_layout() -> tuple[int, int, int, int] | None:
    for _ in range(MAX_TILE_ATTEMPTS):
        tile_h = int(np.random.randint(3, 8))
        tile_w = int(np.random.randint(3, 8))
        max_rows = min(5, ARC_GRID_MAX // tile_h)
        max_cols = min(5, ARC_GRID_MAX // tile_w)
        if max_rows < 2 or max_cols < 2:
            continue
        tile_rows = int(np.random.randint(2, max_rows + 1))
        tile_cols = int(np.random.randint(2, max_cols + 1))
        if tile_rows * tile_cols < 4:
            continue
        return tile_h, tile_w, tile_rows, tile_cols
    return None


def _noise_count_for_category(category: int, area: int) -> int:
    if category == 0:
        return 0
    if category == 1:
        return 1
    if category == 2:
        return 2
    return int(np.random.randint(3, min(area, 5) + 1))


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    tile_h, tile_w, tile_rows, tile_cols = layout

    clean_tile = _generate_clean_tile(tile_h, tile_w)
    if clean_tile is None:
        return None

    grid = np.tile(clean_tile, (tile_rows, tile_cols))
    tile_count = tile_rows * tile_cols
    categories = [0, 1, 2, 3] + [
        int(np.random.randint(0, 4)) for _ in range(tile_count - 4)
    ]
    np.random.shuffle(categories)

    area = tile_h * tile_w
    for tile_index, category in enumerate(categories):
        noise_count = _noise_count_for_category(category, area)
        if noise_count == 0:
            continue
        positions = np.random.choice(area, size=noise_count, replace=False)
        tile_r = tile_index // tile_cols
        tile_c = tile_index % tile_cols
        r0 = tile_r * tile_h
        c0 = tile_c * tile_w
        for pos in positions:
            rr = int(pos) // tile_w
            cc = int(pos) % tile_w
            grid[r0 + rr, c0 + cc] = RED

    inferred_h, inferred_w, inferred_pattern, noise_counts = _infer_tile_structure(grid)
    if (inferred_h, inferred_w) != (tile_h, tile_w):
        return None
    if not np.array_equal(inferred_pattern, clean_tile):
        return None
    if sorted(set(noise_counts)) == [0]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-noise ARC input")
