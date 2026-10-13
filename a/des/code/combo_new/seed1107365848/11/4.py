from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 500
PAIR_PLACE_RETRIES = 160
FRAGMENT_RETRIES = 240

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _round_half_up(numer: int, denom: int) -> int:
    assert denom > 0
    if numer >= 0:
        return (2 * numer + denom) // (2 * denom)
    return -_round_half_up(-numer, denom)


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    out = coords.copy()
    out[:, 0] += int(top)
    out[:, 1] += int(left)
    return out.astype(int)


def _fits(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        coords.size
        and coords[:, 0].min() >= 0
        and coords[:, 1].min() >= 0
        and coords[:, 0].max() < height
        and coords[:, 1].max() < width
    )


def _conflicts(occupied: np.ndarray, coords: np.ndarray, pad: int = 1) -> bool:
    height, width = map(int, occupied.shape)
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(height, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(width, int(c) + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return True
    return False


def _mark(occupied: np.ndarray, coords: np.ndarray, pad: int = 1) -> None:
    height, width = map(int, occupied.shape)
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(height, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(width, int(c) + pad + 1)
        occupied[r0:r1, c0:c1] = True


def _touches_all_sides(mask: np.ndarray) -> bool:
    if not bool(mask[0, :].any()) or not bool(mask[-1, :].any()):
        return False
    if not bool(mask[:, 0].any()) or not bool(mask[:, -1].any()):
        return False
    return True


def _merged_is_connected(mask: np.ndarray) -> bool:
    merged = mask | np.rot90(mask, 2)
    comps = _components(merged)
    return len(comps) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-color fragment pair at the midpoint of their bounding-box centers.

    Input structure:
    - The background is black.
    - The grid contains several scattered monochrome sprites, each a small 4-connected non-black region.
    - Every non-black color appears in exactly two separate sprites.
    - The two sprites of the same color have shapes related by a 180-degree rotation.

    Rule:
    - For each color, find its two sprites and compute each sprite's bounding-box center.
    - Take the midpoint of those two centers and round it to the nearest grid cell.
    - Translate both sprites so their bounding-box centers coincide with that midpoint cell.
    - Keep the shared color where the translated sprites overlap, and clear the original positions to black.

    Key insight:
    - The pairing signal is color, but the landing spot is implicit: it is the midpoint jointly determined by the two same-color fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    out = np.full_like(g, BLACK)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BLACK)

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each non-black color must appear in exactly two components"

        boxes = [_bbox(comp) for comp in comps]
        target_r = _round_half_up(sum(int(r0 + r1) for r0, _c0, r1, _c1 in boxes), 4)
        target_c = _round_half_up(sum(int(c0 + c1) for _r0, c0, _r1, c1 in boxes), 4)

        for comp, (r0, c0, r1, c1) in zip(comps, boxes):
            center_r = _round_half_up(int(r0 + r1), 2)
            center_c = _round_half_up(int(c0 + c1), 2)
            shifted = comp.copy()
            shifted[:, 0] += int(target_r - center_r)
            shifted[:, 1] += int(target_c - center_c)
            assert _fits(shifted, height, width), "translated sprite must stay within the grid"
            for r, c in shifted:
                prev = int(out[r, c])
                assert prev in (BLACK, color), "different colors may not collide after merging"
                out[r, c] = int(color)

    return out


def _grow_fragment(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    return mask


def _random_fragment() -> np.ndarray:
    dims = [(3, 3), (3, 5), (5, 3)]
    areas = [5, 6]

    for _ in range(FRAGMENT_RETRIES):
        height, width = random.choice(dims)
        area = random.choice(areas)
        mask = _grow_fragment(height, width, area)
        if mask is None:
            continue
        if not _touches_all_sides(mask):
            continue
        if not _merged_is_connected(mask):
            continue
        rotated = np.rot90(mask, 2)
        if int((mask | rotated).sum()) <= int(mask.sum()):
            continue
        k = random.randrange(4)
        return np.rot90(mask, k)

    raise RuntimeError("failed to generate a valid fragment")


def _displacement_limits(center: int, radius: int, size: int) -> int:
    return min(center - radius - 1, size - center - radius - 2)


def _sample_displacement(max_dr: int, max_dc: int) -> tuple[int, int] | None:
    modes: list[str] = []
    if max_dc >= 2:
        modes.append("horizontal")
    if max_dr >= 2:
        modes.append("vertical")
    if max_dr >= 2 and max_dc >= 2:
        modes.append("diagonal")
        modes.append("oblique")
    if not modes:
        return None

    mode = random.choice(modes)
    if mode == "horizontal":
        return 0, random.choice([-1, 1]) * random.randint(2, max_dc)
    if mode == "vertical":
        return random.choice([-1, 1]) * random.randint(2, max_dr), 0
    if mode == "diagonal":
        return (
            random.choice([-1, 1]) * random.randint(2, max_dr),
            random.choice([-1, 1]) * random.randint(2, max_dc),
        )

    dr = random.choice([-1, 1]) * random.randint(2, max_dr)
    dc = random.choice([-1, 1]) * random.randint(2, max_dc)
    if abs(dr) == abs(dc):
        dc = -dc
    return dr, dc


def _place_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    color: int,
    mask: np.ndarray,
) -> bool:
    height, width = map(int, grid.shape)
    h, w = map(int, mask.shape)
    rotated = np.rot90(mask, 2)
    merged = mask | rotated

    row_min = h // 2 + 1
    row_max = height - h // 2 - 2
    col_min = w // 2 + 1
    col_max = width - w // 2 - 2
    if row_min > row_max or col_min > col_max:
        return False

    for _ in range(PAIR_PLACE_RETRIES):
        center_r = random.randint(row_min, row_max)
        center_c = random.randint(col_min, col_max)

        max_dr = _displacement_limits(center_r, h // 2, height)
        max_dc = _displacement_limits(center_c, w // 2, width)
        delta = _sample_displacement(max_dr, max_dc)
        if delta is None:
            continue
        dr, dc = delta

        out_top = center_r - h // 2
        out_left = center_c - w // 2
        a_top = center_r - dr - h // 2
        a_left = center_c - dc - w // 2
        b_top = center_r + dr - h // 2
        b_left = center_c + dc - w // 2

        out_coords = _coords_from_mask(merged, out_top, out_left)
        a_coords = _coords_from_mask(mask, a_top, a_left)
        b_coords = _coords_from_mask(rotated, b_top, b_left)

        if not (_fits(out_coords, height, width) and _fits(a_coords, height, width) and _fits(b_coords, height, width)):
            continue
        if _conflicts(output_occ, out_coords, pad=1):
            continue
        if _conflicts(input_occ, a_coords, pad=1):
            continue

        temp_occ = input_occ.copy()
        _mark(temp_occ, a_coords, pad=1)
        if _conflicts(temp_occ, b_coords, pad=1):
            continue

        for coords in (a_coords, b_coords):
            for r, c in coords:
                if int(grid[r, c]) != BLACK:
                    return False
                grid[r, c] = int(color)

        for r, c in out_coords:
            if int(expected[r, c]) not in (BLACK, color):
                return False
            expected[r, c] = int(color)

        _mark(input_occ, a_coords, pad=1)
        _mark(input_occ, b_coords, pad=1)
        _mark(output_occ, out_coords, pad=1)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    max_pairs = 5 if min(height, width) >= 24 else 4
    pair_count = random.randint(2, max_pairs)
    colors = random.sample(COLOR_POOL, pair_count)

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)
    input_occ = np.zeros((height, width), dtype=bool)
    output_occ = np.zeros((height, width), dtype=bool)

    for color in colors:
        try:
            mask = _random_fragment()
        except RuntimeError:
            return None
        if not _place_pair(grid, expected, input_occ, output_occ, color, mask):
            return None

    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 2:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
