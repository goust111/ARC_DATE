from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _array_key(a: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, a.shape)), a.astype(np.int16, copy=False).tobytes()


def _cross_background(g: np.ndarray) -> tuple[int, int]:
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be square"
    side = int(g.shape[0])
    assert side % 2 == 1 and side >= 7, "input side must be odd and large enough"
    mid = side // 2
    bg = int(g[mid, mid])
    assert np.all(g[mid, :] == bg), "center row must be the background separator"
    assert np.all(g[:, mid] == bg), "center column must be the background separator"
    assert bg != RED, "output marker red must not be the input background"
    return bg, mid


def _corner_slices(n: int) -> list[tuple[slice, slice]]:
    side = 2 * n + 1
    return [
        (slice(0, n), slice(0, n)),
        (slice(0, n), slice(n + 1, side)),
        (slice(n + 1, side), slice(0, n)),
        (slice(n + 1, side), slice(n + 1, side)),
    ]


def _extract_tiles(g: np.ndarray, n: int) -> list[np.ndarray]:
    return [g[rs, cs].copy() for rs, cs in _corner_slices(n)]


def _rotations(tile: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(tile, k).copy() for k in range(4)]


def _damage_mask(reference: np.ndarray, damaged: np.ndarray, bg: int) -> np.ndarray | None:
    ref_fg = reference != bg
    dam_fg = damaged != bg
    if not ref_fg.any() or not dam_fg.any():
        return None

    extra = dam_fg & ~ref_fg
    if extra.any():
        return None

    if not np.array_equal(reference[dam_fg], damaged[dam_fg]):
        return None

    missing = ref_fg & (damaged == bg)
    if not missing.any():
        return None
    return missing


def _solve_missing_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg, n = _cross_background(g)
    tiles = _extract_tiles(g, n)

    solutions: dict[tuple[int, tuple[tuple[int, int], bytes], tuple[tuple[int, int], bytes]], np.ndarray] = {}
    for damaged_idx in range(4):
        complete_idxs = [i for i in range(4) if i != damaged_idx]
        complete_tiles = [tiles[i] for i in complete_idxs]

        for base in complete_tiles:
            family = _rotations(base)
            if len({_array_key(tile) for tile in family}) != 4:
                continue

            seen_angles: list[int] = []
            ok = True
            for complete in complete_tiles:
                matches = [k for k, candidate in enumerate(family) if np.array_equal(complete, candidate)]
                if len(matches) != 1:
                    ok = False
                    break
                seen_angles.append(matches[0])
            if not ok or len(set(seen_angles)) != 3:
                continue

            missing_angles = set(range(4)) - set(seen_angles)
            assert len(missing_angles) == 1
            reference = family[missing_angles.pop()]
            missing = _damage_mask(reference, tiles[damaged_idx], bg)
            if missing is None:
                continue

            key = (damaged_idx, _array_key(reference), _array_key(missing.astype(np.uint8)))
            solutions[key] = missing

    assert len(solutions) == 1, "expected a unique damaged fourth rotation tile"
    return next(iter(solutions.values())), bg


def _mask_to_output(mask: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "missing mask must be nonempty"
    crop = _crop_bbox(mask, box)
    out = np.full(crop.shape, bg, dtype=int)
    out[crop] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the missing fourth rotation and output the cropped red damage mask.

    Input structure:
    - An odd-sided square grid is split by a one-pixel-wide background cross into
      four equal square corner tiles.
    - Three corner tiles are complete, distinct rotations of the same foreground
      pattern; the fourth is the missing rotation with some foreground cells
      erased to background.

    Rule:
    - Use the three complete tiles to form the four-member rotation family and
      identify the only rotation not already visible as a complete tile.
    - Align that constructed reference with the damaged tile and select exactly
      the cells where the reference is foreground and the damaged tile is
      background.
    - Crop the selected cells to their own bounding box, paint them red, and
      leave all other output cells as background.

    Key insight:
    - The tile used for comparison is not present in full; it must be constructed
      from the rotation family before subtracting the damaged copy.
    """
    missing, bg = _solve_missing_mask(grid)
    return _mask_to_output(missing, bg)


def _neighbors4(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def _grow_cells(n: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(n), random.randrange(n))
    cells = {start}
    while len(cells) < target:
        frontier = sorted({nb for cell in cells for nb in _neighbors4(cell, n) if nb not in cells})
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _grow_subset(
    allowed: set[tuple[int, int]],
    target: int,
    *,
    seed: tuple[int, int] | None = None,
) -> set[tuple[int, int]] | None:
    if target <= 0 or target > len(allowed):
        return None
    if seed is None:
        seed = random.choice(tuple(allowed))
    if seed not in allowed:
        return None

    cells = {seed}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nb = (r + dr, c + dc)
                if nb in allowed and nb not in cells:
                    frontier.add(nb)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _paint_accent(
    tile: np.ndarray,
    allowed: set[tuple[int, int]],
    color: int,
    max_cells: int,
) -> None:
    count = random.randint(1, max(1, max_cells))
    cells = _grow_subset(allowed, min(count, len(allowed)))
    if cells is None:
        return
    for r, c in cells:
        tile[r, c] = color


def _make_motif(n: int) -> np.ndarray | None:
    min_area = max(8, n + 2)
    max_area = min(n * n - 4, max(min_area, int(n * n * 0.45)))
    if min_area > max_area:
        return None

    target = random.randint(min_area, max_area)
    cells = _grow_cells(n, target)
    if len(cells) < min_area:
        return None

    colors = random.sample(INPUT_COLORS, random.choices([1, 2, 3], weights=[4, 4, 2])[0])
    tile = np.full((n, n), BACKGROUND, dtype=int)
    for r, c in cells:
        tile[r, c] = colors[0]

    for color in colors[1:]:
        _paint_accent(tile, cells, color, max(2, len(cells) // 5))

    if len({_array_key(rot) for rot in _rotations(tile)}) != 4:
        return None
    return tile


def _manhattan_to_set(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - rr) + abs(c - cc) for rr, cc in cells)


def _make_damage(reference: np.ndarray, bg: int) -> np.ndarray | None:
    fg = {tuple(map(int, rc)) for rc in np.argwhere(reference != bg)}
    area = len(fg)
    if area < 8:
        return None

    max_missing = min(12, max(2, area // 3))
    count = random.randint(2, max_missing)
    mode = random.choices(["single", "double", "scattered"], weights=[5, 4, 2])[0]

    for _ in range(80):
        cells: set[tuple[int, int]] | None = None

        if mode == "single":
            cells = _grow_subset(fg, count)
        elif mode == "double" and count >= 3:
            first_count = random.randint(1, count - 1)
            first = _grow_subset(fg, first_count)
            if first is None:
                continue
            remaining = fg - first
            far = [cell for cell in remaining if _manhattan_to_set(cell, first) >= 3]
            if not far:
                continue
            second_count = count - first_count
            second_seed = random.choice(far)
            second = _grow_subset(remaining, second_count, seed=second_seed)
            if second is None:
                continue
            cells = first | second
        else:
            cells = set(random.sample(tuple(fg), count))

        if cells is None or not (0 < len(cells) < area):
            continue

        mask = np.zeros(reference.shape, dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _bbox(mask) is not None:
            return mask
    return None


def _build_grid(
    n: int,
    motif: np.ndarray,
    missing_angle: int,
    damaged_idx: int,
    missing_mask: np.ndarray,
) -> np.ndarray:
    side = 2 * n + 1
    grid = np.full((side, side), BACKGROUND, dtype=int)
    rotations = _rotations(motif)

    complete_angles = [angle for angle in range(4) if angle != missing_angle]
    random.shuffle(complete_angles)
    angle_iter = iter(complete_angles)

    for idx, (rs, cs) in enumerate(_corner_slices(n)):
        if idx == damaged_idx:
            tile = rotations[missing_angle].copy()
            tile[missing_mask] = BACKGROUND
        else:
            tile = rotations[next(angle_iter)]
        grid[rs, cs] = tile
    return grid


def _validate_constructed(grid: np.ndarray, expected_mask: np.ndarray, expected_bg: int) -> bool:
    try:
        solved_mask, solved_bg = _solve_missing_mask(grid)
    except AssertionError:
        return False
    if solved_bg != expected_bg or not np.array_equal(solved_mask, expected_mask):
        return False
    return np.array_equal(main(grid), _mask_to_output(expected_mask, expected_bg))


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        n = random.randint(5, 14)
        motif = _make_motif(n)
        if motif is None:
            continue

        rotations = _rotations(motif)
        missing_angle = random.randrange(4)
        reference = rotations[missing_angle]
        missing_mask = _make_damage(reference, BACKGROUND)
        if missing_mask is None:
            continue

        damaged = reference.copy()
        damaged[missing_mask] = BACKGROUND
        if np.count_nonzero(damaged != BACKGROUND) < 3:
            continue

        damaged_idx = random.randrange(4)
        grid = _build_grid(n, motif, missing_angle, damaged_idx, missing_mask)

        if _validate_constructed(grid, missing_mask, BACKGROUND):
            return grid

    raise RuntimeError("failed to generate a valid damaged-rotation input")
