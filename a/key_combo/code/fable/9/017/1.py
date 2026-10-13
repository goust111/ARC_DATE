from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


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


def _orbit_positions(height: int, width: int, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r, c),
        (r, width - 1 - c),
        (height - 1 - r, c),
        (height - 1 - r, width - 1 - c),
    ]


def _quadrant_position(height: int, width: int, r: int, c: int, quadrant: int) -> tuple[int, int]:
    if quadrant == 0:
        return r, c
    if quadrant == 1:
        return r, width - 1 - c
    if quadrant == 2:
        return height - 1 - r, c
    if quadrant == 3:
        return height - 1 - r, width - 1 - c
    raise ValueError("quadrant must be in 0..3")


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected at least one non-black pixel"
    return _crop_bbox(grid, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold symmetry-breaking minority pixels into the upper-left quadrant and crop the recovered sprite.

    Input structure:
    - The grid has even height and width.
    - Ignoring intruders, the base picture is mirror-symmetric across both the horizontal and vertical midlines.
    - Each four-cell orbit under those two mirrors contains either four identical base colors, or exactly one overwritten intruder cell and three unchanged base cells.

    Rule:
    - Scan every four-cell mirror orbit using the upper-left member as the orbit representative.
    - If all four orbit cells match, ignore that orbit.
    - Otherwise the 3:1 majority color is the base color and the unique minority cell is the intruder; place the intruder color onto a black canvas at the representative position in the upper-left quadrant.
    - Crop that upper-left canvas to the minimum bounding box of its non-black pixels and return it.

    Key insight:
    The hidden figure never appears whole in any one quadrant; only orbit-wise majority voting both identifies the overwritten cell and tells where to fold it back.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "grid dimensions must be even"

    half_h, half_w = height // 2, width // 2
    canvas = np.zeros((half_h, half_w), dtype=int)
    intruder_count = 0

    for r in range(half_h):
        for c in range(half_w):
            orbit = _orbit_positions(height, width, r, c)
            values = [int(g[rr, cc]) for rr, cc in orbit]
            if values.count(values[0]) == 4:
                continue

            counts = Counter(values)
            freq = sorted(counts.values())
            assert freq == [1, 3], "each broken orbit must contain a 3:1 majority"

            intruder_color = next(color for color, count in counts.items() if count == 1)
            assert intruder_color != BLACK, "intruder pixels must be non-black"
            canvas[r, c] = intruder_color
            intruder_count += 1

    assert intruder_count > 0, "expected at least one intruder orbit"
    return _crop_nonblack(canvas)


def _random_partition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    pieces = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _make_base_quadrant(rng: np.random.Generator, half_h: int, half_w: int) -> tuple[np.ndarray, np.ndarray]:
    color_count = int(rng.integers(3, 6))
    nonblack = rng.choice(NON_BLACK_COLORS, size=color_count, replace=False).astype(int)
    palette = nonblack.tolist()
    if rng.random() < 0.5:
        palette.append(BLACK)

    coarse_h = int(rng.integers(2, min(half_h, 5) + 1))
    coarse_w = int(rng.integers(2, min(half_w, 5) + 1))
    row_sizes = _random_partition(rng, half_h, coarse_h)
    col_sizes = _random_partition(rng, half_w, coarse_w)

    coarse = rng.choice(palette, size=(coarse_h, coarse_w), replace=True).astype(int)
    if len(np.unique(coarse)) < 3:
        flat = coarse.reshape(-1)
        missing = [color for color in palette if color not in set(map(int, flat))]
        rng.shuffle(missing)
        for idx, color in enumerate(missing[: 3 - len(np.unique(flat))]):
            flat[idx] = color
        coarse = flat.reshape(coarse.shape)

    quadrant = np.zeros((half_h, half_w), dtype=int)
    r0 = 0
    for rr, hsize in enumerate(row_sizes):
        c0 = 0
        for cc, wsize in enumerate(col_sizes):
            quadrant[r0 : r0 + hsize, c0 : c0 + wsize] = int(coarse[rr, cc])
            c0 += wsize
        r0 += hsize

    for _ in range(int(rng.integers(1, 4))):
        color = int(rng.choice(palette))
        r0 = int(rng.integers(0, half_h))
        r1 = int(rng.integers(r0 + 1, half_h + 1))
        c0 = int(rng.integers(0, half_w))
        c1 = int(rng.integers(c0 + 1, half_w + 1))
        quadrant[r0:r1, c0:c1] = color

    for color in map(int, nonblack):
        if not np.any(quadrant == color):
            r = int(rng.integers(0, half_h))
            c = int(rng.integers(0, half_w))
            quadrant[r, c] = color

    assert len(np.unique(quadrant)) >= 3
    return quadrant, nonblack


def _grow_connected_shape(
    rng: np.random.Generator, box_h: int, box_w: int, target_area: int
) -> np.ndarray:
    cells = {(int(rng.integers(box_h)), int(rng.integers(box_w)))}
    while len(cells) < target_area:
        cr, cc = list(cells)[int(rng.integers(len(cells)))]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if not neighbors:
            continue
        cells.add(neighbors[int(rng.integers(len(neighbors)))])

    coords = np.array(sorted(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    return coords


def _make_hidden_sprite(
    rng: np.random.Generator, quadrant: np.ndarray, nonblack: np.ndarray
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    half_h, half_w = map(int, quadrant.shape)
    box_h = int(rng.integers(3, min(half_h - 1, 7) + 1))
    box_w = int(rng.integers(3, min(half_w - 1, 7) + 1))
    max_area = min(box_h * box_w - 1, 12)
    min_area = max(4, (box_h * box_w + 2) // 3)
    area = int(rng.integers(min_area, max_area + 1))
    local = _grow_connected_shape(rng, box_h, box_w, area)

    local_box = _bbox(np.ones((int(local[:, 0].max()) + 1, int(local[:, 1].max()) + 1), dtype=bool))
    assert local_box is not None
    sprite_h = local_box[2] - local_box[0] + 1
    sprite_w = local_box[3] - local_box[1] + 1
    assert sprite_h >= 2 and sprite_w >= 2
    assert sprite_h < half_h and sprite_w < half_w

    top = int(rng.integers(0, half_h - sprite_h))
    left = int(rng.integers(0, half_w - sprite_w))

    cells: list[tuple[int, int, int]] = []
    canvas = np.zeros((half_h, half_w), dtype=int)
    for dr, dc in map(tuple, local):
        r = top + int(dr)
        c = left + int(dc)
        base_color = int(quadrant[r, c])
        choices = [int(color) for color in nonblack if int(color) != base_color]
        color = int(rng.choice(choices))
        canvas[r, c] = color
        cells.append((r, c, color))

    expected = _crop_nonblack(canvas)
    return cells, expected


def _mirror_full(quadrant: np.ndarray) -> np.ndarray:
    top = np.concatenate([quadrant, quadrant[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def _place_intruders(
    rng: np.random.Generator, base_grid: np.ndarray, sprite_cells: list[tuple[int, int, int]]
) -> tuple[np.ndarray, list[int]]:
    grid = base_grid.copy()
    height, width = map(int, grid.shape)
    intruder_count = len(sprite_cells)
    used_count = int(rng.integers(2, 5))
    used_quadrants = rng.choice(4, size=used_count, replace=False).astype(int).tolist()
    assignments = used_quadrants.copy()
    while len(assignments) < intruder_count:
        assignments.append(int(rng.choice(used_quadrants)))
    rng.shuffle(assignments)

    quadrant_hist = Counter(assignments)
    assert len(quadrant_hist) >= 2
    assert max(quadrant_hist.values()) < intruder_count

    for (r, c, color), quadrant in zip(sprite_cells, assignments):
        rr, cc = _quadrant_position(height, width, r, c, quadrant)
        assert int(grid[rr, cc]) != color
        grid[rr, cc] = color

    return grid, assignments


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    sprite_cells: list[tuple[int, int, int]],
    assignments: list[int],
) -> bool:
    height, width = map(int, grid.shape)
    if grid.ndim != 2 or height % 2 != 0 or width % 2 != 0:
        return False
    if max(height, width) > 30 or max(expected.shape) > 30:
        return False
    if len(set(assignments)) < 2 or max(Counter(assignments).values()) >= len(assignments):
        return False

    half_h, half_w = height // 2, width // 2
    orbit_breaks = 0
    canvas = np.zeros((half_h, half_w), dtype=int)
    for r in range(half_h):
        for c in range(half_w):
            values = [int(grid[rr, cc]) for rr, cc in _orbit_positions(height, width, r, c)]
            counts = Counter(values)
            if len(counts) == 1:
                continue
            if sorted(counts.values()) != [1, 3]:
                return False
            color = next(col for col, count in counts.items() if count == 1)
            if color == BLACK:
                return False
            canvas[r, c] = color
            orbit_breaks += 1

    if orbit_breaks != len(sprite_cells):
        return False
    if not np.array_equal(_crop_nonblack(canvas), expected):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    try:
        half_h = int(rng.integers(6, 13))
        half_w = int(rng.integers(6, 13))
        quadrant, nonblack = _make_base_quadrant(rng, half_h, half_w)
        sprite_cells, expected = _make_hidden_sprite(rng, quadrant, nonblack)
        base_grid = _mirror_full(quadrant)
        grid, assignments = _place_intruders(rng, base_grid, sprite_cells)
        if not _validate_case(grid, expected, sprite_cells, assignments):
            return None

        signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if signature in _SEEN_OUTPUT_SIGNATURES:
            return None
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
        if attempt == MAX_GENERATION_ATTEMPTS // 2:
            _SEEN_OUTPUT_SIGNATURES.clear()
    raise RuntimeError("failed to generate a valid input for symmetric-fold extraction")
