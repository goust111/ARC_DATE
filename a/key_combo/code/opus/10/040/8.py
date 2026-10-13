from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TEMPLATE_COLORS = (BLUE, GREEN, YELLOW)

GRID_SIZE = 9
TEMPLATE_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300
QUADRANT_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _corner_origin(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    dr, dc = direction
    return (0 if dr < 0 else h - TEMPLATE_SIZE, 0 if dc < 0 else w - TEMPLATE_SIZE)


def _corner_patch(grid: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    r0, c0 = _corner_origin(grid.shape, direction)
    return grid[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _noncenter_red_direction(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    center = (h // 2, w // 2)
    center_color = int(grid[center])
    assert center_color == RED, "the center pixel must be red"

    markers = []
    directions = set()
    for r_raw, c_raw in np.argwhere(grid == center_color):
        r = int(r_raw)
        c = int(c_raw)
        if (r, c) == center:
            continue
        dr = _sign(r - center[0])
        dc = _sign(c - center[1])
        assert dr != 0 and dc != 0, "off-center red markers must stay off the center row and column"
        markers.append((r, c))
        directions.add((dr, dc))

    assert markers, "expected off-center red markers"
    assert len(directions) == 1, "all off-center red markers must indicate one diagonal quadrant"
    return next(iter(directions))


def _template_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    candidates = []
    for direction in QUADRANT_DIRS:
        patch = _corner_patch(grid, direction)
        if np.all((patch != BLACK) & (patch != RED)) and len(np.unique(patch)) >= 2:
            assert set(map(int, np.unique(patch))) <= set(TEMPLATE_COLORS), "template colors must be blue/green/yellow"
            candidates.append((direction, patch.copy()))
        else:
            assert not np.any((patch != BLACK) & (patch != RED)), "non-template corners must stay black"
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the corner template into the corner pointed to by the center pixel's same-color quadrant markers.

    Input structure:
    - The grid is an odd square with a black background and a red center pixel.
    - Additional red pixels lie strictly in one diagonal quadrant relative to the center.
    - Exactly one non-target corner contains a 3x3 multicolor template made of blue, green, and/or yellow cells.

    Rule:
    - Use the off-center red pixels to determine one diagonal quadrant.
    - Read the 3x3 template from the unique non-black, non-red corner patch.
    - Copy that template into the indicated corner, leaving the source template, the red pixels, and every other cell unchanged.

    Key insight:
    - The red markers do not tell you how to draw a path; they only name which corner should receive the copied 3x3 template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1 and h >= 7, "expected an odd square grid"

    target_direction = _noncenter_red_direction(g)
    target_r0, target_c0 = _corner_origin(g.shape, target_direction)

    candidates = _template_candidates(g)
    assert len(candidates) == 1, "expected exactly one corner template"
    source_direction, template = candidates[0]
    assert source_direction != target_direction, "the template corner must differ from the indicated corner"
    assert np.all(g[target_r0 : target_r0 + TEMPLATE_SIZE, target_c0 : target_c0 + TEMPLATE_SIZE] == BLACK), (
        "the target corner must be blank before stamping"
    )

    out = g.copy()
    _paste(out, template, target_r0, target_c0)
    return out


def _all_transforms(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    cur = patch
    for _ in range(4):
        variants.append(cur)
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return variants


def _make_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        color_count = int(rng.integers(2, 4))
        palette = rng.choice(np.array(TEMPLATE_COLORS), size=color_count, replace=False).astype(int)
        cells = np.repeat(np.arange(color_count), 1)
        extra = rng.integers(0, color_count, size=9 - color_count)
        indices = np.concatenate([cells, extra])
        rng.shuffle(indices)
        patch = palette[indices].reshape(TEMPLATE_SIZE, TEMPLATE_SIZE)
        if len(np.unique(patch)) < 2:
            continue
        if any(np.array_equal(patch, transformed) for transformed in _all_transforms(patch)[1:]):
            continue
        return patch.astype(int)
    raise RuntimeError("failed to build an asymmetric template")


def _quadrant_cells_excluding_corner(shape: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    center_r, center_c = h // 2, w // 2
    rows = range(0, center_r) if direction[0] < 0 else range(center_r + 1, h)
    cols = range(0, center_c) if direction[1] < 0 else range(center_c + 1, w)

    corner_r0, corner_c0 = _corner_origin(shape, direction)
    forbidden = {
        (r, c)
        for r in range(corner_r0, corner_r0 + TEMPLATE_SIZE)
        for c in range(corner_c0, corner_c0 + TEMPLATE_SIZE)
    }
    return [(r, c) for r in rows for c in cols if (r, c) not in forbidden]


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    direction = _noncenter_red_direction(g)

    template_cells = np.argwhere((g != BLACK) & (g != RED))
    assert template_cells.size > 0, "expected template cells"
    r0, c0 = template_cells.min(axis=0)
    r1, c1 = template_cells.max(axis=0)
    assert (int(r1) - int(r0) + 1, int(c1) - int(c0) + 1) == (TEMPLATE_SIZE, TEMPLATE_SIZE)
    template = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    assert np.all((template != BLACK) & (template != RED))
    assert len(np.unique(template)) >= 2

    source_direction = None
    for candidate in QUADRANT_DIRS:
        cr0, cc0 = _corner_origin(g.shape, candidate)
        if int(r0) == cr0 and int(c0) == cc0:
            source_direction = candidate
            break
    assert source_direction is not None, "template must sit in a corner"
    assert source_direction != direction

    tr0, tc0 = _corner_origin(g.shape, direction)
    assert np.all(g[tr0 : tr0 + TEMPLATE_SIZE, tc0 : tc0 + TEMPLATE_SIZE] == BLACK)

    out = g.copy()
    out[tr0 : tr0 + TEMPLATE_SIZE, tc0 : tc0 + TEMPLATE_SIZE] = template
    return out


def _validate_example(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False

    center = GRID_SIZE // 2
    if int(grid[center, center]) != RED:
        return False

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED) if tuple(map(int, rc)) != (center, center)]
    if not (2 <= len(red_coords) <= 4):
        return False
    if any(r == center or c == center for r, c in red_coords):
        return False

    directions = {(_sign(r - center), _sign(c - center)) for r, c in red_coords}
    if len(directions) != 1:
        return False
    direction = next(iter(directions))

    source_candidates = []
    for candidate in QUADRANT_DIRS:
        patch = _corner_patch(grid, candidate)
        if np.all((patch != BLACK) & (patch != RED)) and len(np.unique(patch)) >= 2:
            source_candidates.append(candidate)
        elif np.any((patch != BLACK) & (patch != RED)):
            return False
    if len(source_candidates) != 1:
        return False
    if source_candidates[0] == direction:
        return False

    target_patch = _corner_patch(grid, direction)
    if not np.all(target_patch == BLACK):
        return False

    non_template_mask = (grid != BLACK) & (grid != RED)
    template_dir = source_candidates[0]
    sr0, sc0 = _corner_origin(grid.shape, template_dir)
    allowed = np.zeros_like(grid, dtype=bool)
    allowed[sr0 : sr0 + TEMPLATE_SIZE, sc0 : sc0 + TEMPLATE_SIZE] = True
    if np.any(non_template_mask & ~allowed):
        return False

    expected = _oracle_output(grid)
    if not np.array_equal(main(grid), expected):
        return False

    alternatives = []
    for candidate in QUADRANT_DIRS:
        tr0, tc0 = _corner_origin(grid.shape, candidate)
        alt = grid.copy()
        alt[tr0 : tr0 + TEMPLATE_SIZE, tc0 : tc0 + TEMPLATE_SIZE] = grid[sr0 : sr0 + TEMPLATE_SIZE, sc0 : sc0 + TEMPLATE_SIZE]
        alternatives.append(alt)
    if sum(np.array_equal(expected, alt) for alt in alternatives) != 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    center = GRID_SIZE // 2
    grid[center, center] = RED

    target_direction = QUADRANT_DIRS[int(rng.integers(0, len(QUADRANT_DIRS)))]
    source_direction = random.choice([d for d in QUADRANT_DIRS if d != target_direction])

    marker_cells = _quadrant_cells_excluding_corner(grid.shape, target_direction)
    marker_count = int(rng.integers(2, 5))
    chosen_indices = rng.choice(len(marker_cells), size=marker_count, replace=False)
    for idx in chosen_indices:
        r, c = marker_cells[int(idx)]
        grid[r, c] = RED

    template = _make_template(rng)
    sr0, sc0 = _corner_origin(grid.shape, source_direction)
    _paste(grid, template, sr0, sc0)

    return grid if _validate_example(grid) else None


def generate_input() -> np.ndarray:
    # 7x7 conflicts with the blank-target-corner constraint: a 3x3 corner equals the whole quadrant.
    # We therefore generate only 9x9 examples, which still cover all semantic branches.
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
