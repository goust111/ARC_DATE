from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ORANGE = 7
TEAL = 8
SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 9)

GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
SPRITE_BOX_HEIGHT_RANGE = (2, 5)
SPRITE_BOX_WIDTH_RANGE = (2, 5)
SPRITE_AREA_RANGE = (4, 9)
GENERATION_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        if seen[start_r, start_c]:
            continue
        stack = [(int(start_r), int(start_c))]
        seen[start_r, start_c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(cells, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _axis_ranges(coords: np.ndarray, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = _bbox(coords)
    row_range = ((r1 + 1) // 2, (height - 1 + r0) // 2)
    col_range = ((c1 + 1) // 2, (width - 1 + c0) // 2)
    return row_range, col_range


def _reflect_coords(coords: np.ndarray, pivot: tuple[int, int]) -> np.ndarray:
    pivot_r, pivot_c = map(int, pivot)
    reflected = coords.copy()
    reflected[:, 0] = 2 * pivot_r - coords[:, 0]
    reflected[:, 1] = 2 * pivot_c - coords[:, 1]
    return reflected


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return bool(
        np.all((0 <= coords[:, 0]) & (coords[:, 0] < height))
        and np.all((0 <= coords[:, 1]) & (coords[:, 1] < width))
    )


def _discover_scene(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    orange = np.argwhere(g == ORANGE)
    teal = np.argwhere(g == TEAL)
    assert orange.shape == (1, 2), "expected exactly one orange seed"
    assert teal.shape == (1, 2), "expected exactly one teal seed"

    orange_rc = tuple(map(int, orange[0]))
    teal_rc = tuple(map(int, teal[0]))
    assert orange_rc[0] != teal_rc[0] and orange_rc[1] != teal_rc[1], "seeds must be in different rows and columns"

    sprite_mask = (g != BACKGROUND) & (g != ORANGE) & (g != TEAL)
    components = _find_components(sprite_mask)
    assert len(components) == 1, "expected exactly one detached sprite"

    sprite_coords = components[0]
    sprite_colors = g[sprite_coords[:, 0], sprite_coords[:, 1]].astype(int).copy()
    return orange_rc, teal_rc, sprite_coords, sprite_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the unique sprite through the only feasible hidden row-column intersection defined by the orange and teal seeds.

    Input structure
    - A black background contains exactly one orange pixel and one teal pixel, in different rows and different columns.
    - All remaining non-black pixels form one detached 4-connected sprite whose colors are neither orange nor teal.
    - The intersection pivots themselves are invisible; only the two seed rows and seed columns matter.

    Rule
    - Form the two candidate pivots at (orange_row, teal_col) and (teal_row, orange_col).
    - Compute the sprite's 180-degree point reflection around each candidate pivot.
    - Exactly one candidate keeps the full reflected sprite inside the grid; erase the original sprite and redraw it there while leaving the orange and teal seeds unchanged.

    Key insight
    - The pivot is never drawn directly: it is one of the two hidden cross-intersections implied by the two isolated seeds, and full in-bounds feasibility selects the correct one.
    """
    g = np.asarray(grid, dtype=int)
    orange_rc, teal_rc, sprite_coords, sprite_colors = _discover_scene(g)

    candidates = [
        (orange_rc[0], teal_rc[1]),
        (teal_rc[0], orange_rc[1]),
    ]
    feasible: list[np.ndarray] = []
    for pivot in candidates:
        reflected = _reflect_coords(sprite_coords, pivot)
        if _coords_in_bounds(reflected, g.shape):
            feasible.append(reflected)

    assert len(feasible) == 1, "expected exactly one feasible hidden pivot"
    reflected = feasible[0]

    out = g.copy()
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = BACKGROUND
    out[reflected[:, 0], reflected[:, 1]] = sprite_colors
    out[orange_rc] = ORANGE
    out[teal_rc] = TEAL
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = sorted(
            {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
            }
        )
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1 or c0 == c1:
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    max_colors = min(3, int(coords.shape[0]))
    if max_colors == 1 or random.random() < 0.45:
        n_colors = 1
    else:
        n_colors = random.randint(2, max_colors)

    palette = random.sample(SPRITE_COLORS, n_colors)
    assigned = [palette[i % n_colors] for i in range(coords.shape[0])]
    random.shuffle(assigned)

    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    for (r, c), color in zip(coords.tolist(), assigned):
        patch[r, c] = int(color)
    return patch


def _valid_reflections(coords: np.ndarray, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], list[tuple[tuple[int, int], np.ndarray]]]:
    row_range, col_range = _axis_ranges(coords, shape)
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    source_set = _coords_set(coords)
    valid: list[tuple[tuple[int, int], np.ndarray]] = []

    for pivot_r in range(row_lo, row_hi + 1):
        for pivot_c in range(col_lo, col_hi + 1):
            reflected = _reflect_coords(coords, (pivot_r, pivot_c))
            reflected_set = _coords_set(reflected)
            if source_set & reflected_set:
                continue
            valid.append(((pivot_r, pivot_c), reflected))

    return row_range, col_range, valid


def _seed_blocked_cells(shape: tuple[int, int], *coord_arrays: np.ndarray) -> set[tuple[int, int]]:
    height, width = map(int, shape)
    blocked: set[tuple[int, int]] = set()
    for coords in coord_arrays:
        for r, c in coords.tolist():
            rr = int(r)
            cc = int(c)
            blocked.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))
    return blocked


def _seed_options(
    shape: tuple[int, int],
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    pivot: tuple[int, int],
    branch: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    height, width = map(int, shape)
    pivot_r, pivot_c = map(int, pivot)
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    blocked = _seed_blocked_cells(shape, source_coords, target_coords)
    options = {"row": [], "col": [], "both": []}

    if branch == 0:
        orange_candidates = [
            (pivot_r, col)
            for col in range(width)
            if col != pivot_c and (pivot_r, col) not in blocked
        ]
        teal_candidates = [
            (row, pivot_c)
            for row in range(height)
            if row != pivot_r and (row, pivot_c) not in blocked
        ]
        for orange_rc in orange_candidates:
            for teal_rc in teal_candidates:
                bad_pivot = (teal_rc[0], orange_rc[1])
                if _coords_in_bounds(_reflect_coords(source_coords, bad_pivot), shape):
                    continue
                row_ok = row_lo <= bad_pivot[0] <= row_hi
                col_ok = col_lo <= bad_pivot[1] <= col_hi
                reason = "both" if (not row_ok and not col_ok) else "row" if not row_ok else "col"
                options[reason].append((orange_rc, teal_rc))
        return options

    orange_candidates = [
        (row, pivot_c)
        for row in range(height)
        if row != pivot_r and (row, pivot_c) not in blocked
    ]
    teal_candidates = [
        (pivot_r, col)
        for col in range(width)
        if col != pivot_c and (pivot_r, col) not in blocked
    ]
    for orange_rc in orange_candidates:
        for teal_rc in teal_candidates:
            bad_pivot = (orange_rc[0], teal_rc[1])
            if _coords_in_bounds(_reflect_coords(source_coords, bad_pivot), shape):
                continue
            row_ok = row_lo <= bad_pivot[0] <= row_hi
            col_ok = col_lo <= bad_pivot[1] <= col_hi
            reason = "both" if (not row_ok and not col_ok) else "row" if not row_ok else "col"
            options[reason].append((orange_rc, teal_rc))
    return options


def _paint_scene(
    shape: tuple[int, int],
    sprite_coords: np.ndarray,
    sprite_colors: np.ndarray,
    orange_rc: tuple[int, int],
    teal_rc: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[sprite_coords[:, 0], sprite_coords[:, 1]] = sprite_colors
    grid[orange_rc] = ORANGE
    grid[teal_rc] = TEAL
    return grid


def _semantic_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        orange_rc, teal_rc, sprite_coords, _sprite_colors = _discover_scene(inp)
    except AssertionError:
        return False

    candidates = [
        (orange_rc[0], teal_rc[1]),
        (teal_rc[0], orange_rc[1]),
    ]
    feasible = [
        pivot
        for pivot in candidates
        if _coords_in_bounds(_reflect_coords(sprite_coords, pivot), inp.shape)
    ]
    if len(feasible) != 1:
        return False
    if np.array_equal(inp, expected):
        return False
    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    box_h = random.randint(*SPRITE_BOX_HEIGHT_RANGE)
    box_w = random.randint(*SPRITE_BOX_WIDTH_RANGE)
    area_lo = max(SPRITE_AREA_RANGE[0], max(box_h, box_w))
    area_hi = min(SPRITE_AREA_RANGE[1], box_h * box_w)
    if area_lo > area_hi or height < box_h or width < box_w:
        return None

    area = random.randint(area_lo, area_hi)
    mask = _random_connected_mask(box_h, box_w, area)
    if mask is None:
        return None

    sprite_patch = _colorize_mask(mask)
    local_coords = np.argwhere(mask)
    sprite_colors = sprite_patch[local_coords[:, 0], local_coords[:, 1]].astype(int)

    top = random.randint(0, height - box_h)
    left = random.randint(0, width - box_w)
    source_coords = local_coords + np.array([top, left], dtype=int)

    row_range, col_range, valid = _valid_reflections(source_coords, (height, width))
    if not valid:
        return None
    random.shuffle(valid)

    branch_order = [0, 1]
    random.shuffle(branch_order)
    for pivot, target_coords in valid:
        for branch in branch_order:
            by_reason = _seed_options(
                (height, width),
                source_coords,
                target_coords,
                pivot,
                branch,
                row_range,
                col_range,
            )
            reasons = [reason for reason, pairs in by_reason.items() if pairs]
            if not reasons:
                continue
            chosen_reason = random.choice(reasons)
            orange_rc, teal_rc = random.choice(by_reason[chosen_reason])

            inp = _paint_scene((height, width), source_coords, sprite_colors, orange_rc, teal_rc)
            expected = _paint_scene((height, width), target_coords, sprite_colors, orange_rc, teal_rc)
            if _semantic_ok(inp, expected):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-pivot reflection input")
