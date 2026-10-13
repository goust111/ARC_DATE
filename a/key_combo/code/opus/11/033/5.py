import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GEN_ATTEMPTS = 400
GEN_CELL_SIZES = (3, 5)
GEN_BLUE_SIZE_RANGE = (3, 6)
GEN_OCCUPIED_COUNT_RANGE = (1, 3)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    starts = [0] + [sep + 1 for sep in separators]
    ends = separators + [limit]
    bands = list(zip(starts, ends))
    assert len(bands) == 3
    assert all(lo < hi for lo, hi in bands)
    return bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    hits = [i for i, (lo, hi) in enumerate(bands) if lo <= value < hi]
    assert len(hits) == 1
    return hits[0]


def _parse_partition(
    g: np.ndarray,
) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(row_seps) == len(col_seps) == 2
    row_bands = _bands(row_seps, g.shape[0])
    col_bands = _bands(col_seps, g.shape[1])
    row_sizes = {hi - lo for lo, hi in row_bands}
    col_sizes = {hi - lo for lo, hi in col_bands}
    assert len(row_sizes) == len(col_sizes) == 1
    assert row_sizes == col_sizes

    centers = np.argwhere(g == RED)
    assert centers.shape == (1, 2)
    cr, cc = map(int, centers[0])
    assert (cr, cc) == (g.shape[0] // 2, g.shape[1] // 2)
    assert row_bands[1][0] <= cr < row_bands[1][1]
    assert col_bands[1][0] <= cc < col_bands[1][1]
    return row_seps, col_seps, row_bands, col_bands, (cr, cc)


def _region_of(
    coord: tuple[int, int], row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> tuple[int, int]:
    r, c = coord
    return _band_index(r, row_bands), _band_index(c, col_bands)


def _component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _blue_component(g: np.ndarray) -> set[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert coords
    comp = _component(g == BLUE, coords[0])
    assert len(comp) == len(coords)
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the blue source pattern through the central red marker, but only keep reflected pixels whose destination region is not occupied.

    Input structure
    - A black grid is partitioned by two full grey separator rows and two full grey separator columns into a 3x3 array of equal regions.
    - A single red marker sits at the geometric center of the board, inside the middle region.
    - One 4-connected blue pattern lies entirely inside one non-central region, and 1-3 other regions may each contain one green occupancy marker.

    Rule
    - Keep the separators, the red center marker, the original blue pattern, and all green occupancy markers unchanged.
    - For each blue pixel at (r, c), compute its 180-degree point reflection through the red marker at (cr, cc): (2*cr-r, 2*cc-c).
    - Find which macro-region contains that reflected position; draw the reflected blue pixel only if that region has no green occupancy marker.

    Key insight
    - The output is determined by the intersection of two structures: geometric 180-degree reflection gives every candidate pixel, and region-level occupancy decides whether each candidate is allowed to survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    _row_seps, _col_seps, row_bands, col_bands, (cr, cc) = _parse_partition(g)

    blue = _blue_component(g)
    source_regions = {_region_of(coord, row_bands, col_bands) for coord in blue}
    assert len(source_regions) == 1
    source_region = next(iter(source_regions))
    assert source_region != (1, 1)

    green_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert 1 <= len(green_coords) <= 3
    occupied_regions = {_region_of(coord, row_bands, col_bands) for coord in green_coords}
    assert len(occupied_regions) == len(green_coords)
    assert source_region not in occupied_regions

    out = g.copy()
    for r, c in blue:
        rr = 2 * cr - r
        rc = 2 * cc - c
        assert 0 <= rr < g.shape[0] and 0 <= rc < g.shape[1]
        dst_region = _region_of((rr, rc), row_bands, col_bands)
        if dst_region not in occupied_regions:
            assert out[rr, rc] in (BLACK, BLUE)
            out[rr, rc] = BLUE
    return out


def _separator_positions(cell_size: int) -> tuple[int, int]:
    return cell_size, 2 * cell_size + 1


def _empty_board(cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    n = 3 * cell_size + 2
    seps = _separator_positions(cell_size)
    g = np.full((n, n), BLACK, dtype=int)
    g[list(seps), :] = GREY
    g[:, list(seps)] = GREY
    g[n // 2, n // 2] = RED
    row_bands = _bands(list(seps), n)
    col_bands = _bands(list(seps), n)
    return g, row_bands, col_bands


def _random_polyomino(cell_size: int) -> set[tuple[int, int]] | None:
    max_cells = min(GEN_BLUE_SIZE_RANGE[1], cell_size * cell_size - 1)
    size = random.randint(GEN_BLUE_SIZE_RANGE[0], max_cells)
    for _ in range(100):
        cells = {(random.randrange(cell_size), random.randrange(cell_size))}
        while len(cells) < size:
            r, c = random.choice(tuple(cells))
            options = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in cells
            ]
            if not options:
                break
            cells.add(random.choice(options))
        if len(cells) != size:
            continue
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        r0, r1 = min(rr), max(rr)
        c0, c1 = min(cc), max(cc)
        norm = {(r - r0, c - c0) for r, c in cells}
        rot180 = {(r1 - r, c1 - c) for r, c in cells}
        rot180_norm = {(r - min(r for r, _ in rot180), c - min(c for _, c in rot180)) for r, c in rot180}
        if norm == rot180_norm:
            continue
        return cells
    return None


def _region_origin(
    region: tuple[int, int], row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> tuple[int, int]:
    return row_bands[region[0]][0], col_bands[region[1]][0]


def _reflect_region(region: tuple[int, int]) -> tuple[int, int]:
    return 2 - region[0], 2 - region[1]


def _random_region_cell(
    region: tuple[int, int], row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> tuple[int, int]:
    r0, r1 = row_bands[region[0]]
    c0, c1 = col_bands[region[1]]
    return random.randrange(r0, r1), random.randrange(c0, c1)


def _added_blue_count(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero((inp != BLUE) & (out == BLUE)))


def _oracle_check(inp: np.ndarray) -> bool:
    out = main(inp)
    row_seps, col_seps, row_bands, col_bands, (cr, cc) = _parse_partition(inp)
    assert row_seps == col_seps
    blue = _blue_component(inp)
    source_region = next(iter({_region_of(coord, row_bands, col_bands) for coord in blue}))
    target_region = _reflect_region(source_region)
    occupied_regions = {_region_of(tuple(map(int, rc)), row_bands, col_bands) for rc in np.argwhere(inp == GREEN)}

    expected_added = 0
    for r, c in blue:
        rr = 2 * cr - r
        rc = 2 * cc - c
        assert _region_of((rr, rc), row_bands, col_bands) == target_region
        if target_region not in occupied_regions:
            expected_added += 1
            assert out[rr, rc] == BLUE
        else:
            assert (rr, rc) not in blue
            assert out[rr, rc] == inp[rr, rc]

    assert _added_blue_count(inp, out) == expected_added
    assert np.array_equal(out[row_seps, :], inp[row_seps, :])
    assert np.array_equal(out[:, col_seps], inp[:, col_seps])
    assert np.array_equal(out[inp == GREEN], inp[inp == GREEN])
    assert np.array_equal(out[inp == RED], inp[inp == RED])
    return True


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(GEN_CELL_SIZES)
    g, row_bands, col_bands = _empty_board(cell_size)

    source_region = random.choice(
        [(br, bc) for br in range(3) for bc in range(3) if (br, bc) != (1, 1)]
    )
    target_region = _reflect_region(source_region)

    poly = _random_polyomino(cell_size)
    if poly is None:
        return None
    src_r0, src_c0 = _region_origin(source_region, row_bands, col_bands)
    for dr, dc in poly:
        g[src_r0 + dr, src_c0 + dc] = BLUE

    occupied_count = random.randint(*GEN_OCCUPIED_COUNT_RANGE)
    other_regions = [
        (br, bc)
        for br in range(3)
        for bc in range(3)
        if (br, bc) not in {source_region, target_region}
    ]

    occupied_regions = []
    remaining = occupied_count
    if remaining > len(other_regions):
        return None
    occupied_regions.extend(random.sample(other_regions, remaining))
    random.shuffle(occupied_regions)

    for region in occupied_regions:
        rr, cc = _random_region_cell(region, row_bands, col_bands)
        if g[rr, cc] != BLACK:
            return None
        g[rr, cc] = GREEN

    if not _oracle_check(g):
        return None
    if _added_blue_count(g, main(g)) == 0:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        g = _gen_attempt()
        if g is None:
            continue
        return g
    raise RuntimeError("failed to generate valid input")
