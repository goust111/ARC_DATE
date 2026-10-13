import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
BASE_COLORS = (BLUE, GREEN, YELLOW, GREY)


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _translation_support(grid: np.ndarray, dr: int, dc: int) -> tuple[bool, int]:
    h, w = grid.shape
    if dr < 0 or dc < 0 or (dr == 0 and dc == 0) or dr >= h or dc >= w:
        return False, 0
    if dr > 0:
        src = grid[: h - dr, :]
        dst = grid[dr:, :]
    else:
        src = grid[:, : w - dc]
        dst = grid[:, dc:]
    known = (src != BLACK) & (dst != BLACK)
    support = int(known.sum())
    if support == 0:
        return False, 0
    return bool(np.all(src[known] == dst[known])), support


def _detect_periods(grid: np.ndarray) -> tuple[int | None, int | None]:
    h, w = grid.shape
    row_period = None
    col_period = None
    for dr in range(1, h):
        ok, support = _translation_support(grid, dr, 0)
        if ok and support >= w:
            row_period = dr
            break
    for dc in range(1, w):
        ok, support = _translation_support(grid, 0, dc)
        if ok and support >= h:
            col_period = dc
            break
    if row_period is None and col_period is None:
        raise AssertionError("failed to detect translational symmetry")
    return row_period, col_period


def _equivalent_values(grid: np.ndarray, r: int, c: int, row_period: int | None, col_period: int | None) -> list[int]:
    h, w = grid.shape
    rows = [r] if row_period is None else list(range(r % row_period, h, row_period))
    cols = [c] if col_period is None else list(range(c % col_period, w, col_period))
    values = []
    for rr in rows:
        for cc in cols:
            val = int(grid[rr, cc])
            if val != BLACK:
                values.append(val)
    return values


def _restore_grid(grid: np.ndarray, row_period: int | None, col_period: int | None) -> np.ndarray:
    restored = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(restored == BLACK):
        vals = _equivalent_values(restored, int(r), int(c), row_period, col_period)
        assert vals, "every occluded orbit must expose a visible sample"
        color, count = Counter(vals).most_common(1)[0]
        assert count == len(vals), "symmetry orbit should agree on a unique color"
        restored[int(r), int(c)] = int(color)
    return restored


def _every_black_has_visible_orbit_member(grid: np.ndarray, row_period: int | None, col_period: int | None) -> bool:
    for r, c in np.argwhere(grid == BLACK):
        vals = _equivalent_values(grid, int(r), int(c), row_period, col_period)
        if not vals:
            return False
    return True


def _find_seeds(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seeds = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            red_coords = np.argwhere(block == RED)
            if red_coords.shape[0] != 1:
                continue
            fill_vals = block[block != RED]
            if len(set(map(int, fill_vals))) != 1:
                continue
            color = int(fill_vals[0])
            if color == RED:
                continue
            seeds.append(
                {
                    "top_left": (int(r), int(c)),
                    "color": color,
                    "red_local": tuple(map(int, red_coords[0])),
                }
            )
    return seeds


def _paint_clipped_square(out: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = out.shape
    if r0 >= h or c0 >= w or r0 + 1 < 0 or c0 + 1 < 0:
        return False
    for dr in (0, 1):
        for dc in (0, 1):
            rr = r0 + dr
            cc = c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)
    return True


def _ray_step(red_local: tuple[int, int]) -> tuple[int, int]:
    lr, lc = red_local
    return (-1 if lr == 0 else 1, -1 if lc == 0 else 1)


def _apply_rays(restored: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    seeds = _find_seeds(restored)
    out = restored.copy()
    for seed in seeds:
        rr, cc = seed["top_left"]
        step_r, step_c = _ray_step(seed["red_local"])
        while _paint_clipped_square(out, rr, cc, seed["color"]):
            rr += step_r
            cc += step_c
    return out, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover an occluded translational pattern, then reveal diagonal seed rays only inside the original mask.

    Input structure:
    - The grid is fully filled by a translationally repeated non-black pattern; black is only an occluding rectangle.
    - Some repeated 2x2 seed blocks are embedded in the pattern; each seed has three same-color non-red cells and one red corner cell.
    - The black rectangle may hide part or all of one or more seeds, including the red direction marker.

    Rule:
    - Treat black as an occlusion mask and detect the pattern's translational period from the remaining visible cells.
    - Restore every black cell from a non-black cell in the same translation orbit.
    - In the restored grid, find every 2x2 block containing exactly one red cell and three equal non-red cells; each such block is a seed.
    - Replace each seed's red corner by the seed color and repeatedly stamp a filled 2x2 square along the matching diagonal direction until it leaves the grid.
    - Return only the region covered by the original black rectangle.

    Key insight:
    - The occlusion hides not just colors but seed identity and direction, so symmetry restoration must happen before the diagonal ray rule can be applied.
    """
    g = np.asarray(grid, dtype=int)
    occlusion = g == BLACK
    box = bbox(occlusion)
    assert box is not None, "expected one black occlusion rectangle"

    row_period, col_period = _detect_periods(g)
    restored = _restore_grid(g, row_period, col_period)
    final_grid, _ = _apply_rays(restored)
    return crop_bbox(final_grid, box)


GEN_RETRIES = 500


def _build_tile(size: int, palette: list[int]) -> np.ndarray:
    tile = np.random.choice(palette, size=(size, size), replace=True).astype(int)
    for color in palette:
        if color not in tile:
            rr = np.random.randint(0, size)
            cc = np.random.randint(0, size)
            tile[rr, cc] = color
    return tile


def _tile_has_smaller_period(tile: np.ndarray) -> bool:
    size = tile.shape[0]
    for shift in range(1, size):
        ok_r, support_r = _translation_support(tile, shift, 0)
        ok_c, support_c = _translation_support(tile, 0, shift)
        if ok_r and support_r > 0:
            return True
        if ok_c and support_c > 0:
            return True
    return False


def _hidden_tile_for_corner(red_local: tuple[int, int]) -> tuple[int, int]:
    lr, lc = red_local
    return (0 if lr == 0 else 1, 0 if lc == 0 else 1)


def _expand_mask(size: int, hidden_tile: tuple[int, int]) -> tuple[int, int, int, int]:
    tr, tc = hidden_tile
    r0 = tr * size
    c0 = tc * size
    r1 = r0 + size - 1
    c1 = c0 + size - 1

    extra_r = np.random.randint(0, 3)
    extra_c = np.random.randint(0, 3)
    if tr == 0:
        r1 = min(2 * size - 1, r1 + extra_r)
    else:
        r0 = max(0, r0 - extra_r)
    if tc == 0:
        c1 = min(2 * size - 1, c1 + extra_c)
    else:
        c0 = max(0, c0 - extra_c)
    return int(r0), int(c0), int(r1), int(c1)


def _seed_cells_mask(shape: tuple[int, int], seeds: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for seed in seeds:
        r0, c0 = seed["top_left"]
        mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return mask


def _valid_visible_seed(mask: np.ndarray, seed: dict) -> bool:
    r0, c0 = seed["top_left"]
    return not bool(mask[r0 : r0 + 2, c0 : c0 + 2].any())


def _hidden_seed_constraint(mask: np.ndarray, seed: dict) -> bool:
    r0, c0 = seed["top_left"]
    lr, lc = seed["red_local"]
    red_r = r0 + lr
    red_c = c0 + lc
    return bool(mask[red_r, red_c] or mask[r0 : r0 + 2, c0 : c0 + 2].all())


def _generate_candidate() -> np.ndarray | None:
    size = int(np.random.randint(5, 7))
    palette_size = int(np.random.randint(3, 5))
    palette = random.sample(list(BASE_COLORS), palette_size)

    tile = _build_tile(size, palette)
    if _tile_has_smaller_period(tile):
        return None

    seed_r = int(np.random.randint(1, size - 2))
    seed_c = int(np.random.randint(1, size - 2))
    red_local = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    paint = random.choice(palette)
    tile[seed_r : seed_r + 2, seed_c : seed_c + 2] = paint
    tile[seed_r + red_local[0], seed_c + red_local[1]] = RED

    restored = np.tile(tile, (2, 2))
    designed_seeds = _find_seeds(restored)
    if len(designed_seeds) != 4:
        return None

    hidden_tile = _hidden_tile_for_corner(red_local)
    box = _expand_mask(size, hidden_tile)
    r0, c0, r1, c1 = box
    mask = np.zeros_like(restored, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True

    visible_seeds = [seed for seed in designed_seeds if _valid_visible_seed(mask, seed)]
    if not visible_seeds:
        return None

    hidden_seeds = [seed for seed in designed_seeds if _hidden_seed_constraint(mask, seed)]
    if not hidden_seeds:
        return None

    masked = restored.copy()
    masked[mask] = BLACK

    try:
        row_period, col_period = _detect_periods(masked)
    except AssertionError:
        return None
    if row_period != size or col_period != size:
        return None

    if not _every_black_has_visible_orbit_member(masked, row_period, col_period):
        return None

    recovered = _restore_grid(masked, row_period, col_period)
    if not np.array_equal(recovered, restored):
        return None

    final_grid, recovered_seeds = _apply_rays(restored)
    if len(recovered_seeds) != 4:
        return None

    recovered_crop = crop_bbox(restored, box)
    final_crop = crop_bbox(final_grid, box)
    if np.array_equal(recovered_crop, final_crop):
        return None

    seed_mask = _seed_cells_mask(restored.shape, recovered_seeds)
    changed_non_seed = (final_grid != restored) & mask & (~seed_mask)
    if not changed_non_seed.any():
        return None

    if not np.array_equal(main(masked), final_crop):
        return None

    return masked


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
