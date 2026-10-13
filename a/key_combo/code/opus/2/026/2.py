from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# The prompt's 3x3 halo cannot create any visible cross-group overlap once a full
# separator row is excluded, so this task uses the smallest consistent repair:
# a clipped 5x5 halo (Chebyshev radius 2) while still leaving the separator row
# unchanged.
HALO_RADIUS = 2
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == YELLOW))]
    assert len(rows) == 1, "expected exactly one full yellow separator row"
    return rows[0]


def _group_markers(grid: np.ndarray, sep_row: int) -> tuple[list[Coord], list[Coord]]:
    red_markers = [(int(r), int(c)) for r, c in np.argwhere(grid == RED) if int(r) < sep_row]
    blue_markers = [(int(r), int(c)) for r, c in np.argwhere(grid == BLUE) if int(r) > sep_row]
    assert red_markers, "expected at least one red marker above the separator"
    assert blue_markers, "expected at least one blue marker below the separator"
    assert not bool(np.any(grid[:sep_row, :] == BLUE)), "blue markers must stay below the separator"
    assert not bool(np.any(grid[sep_row + 1 :, :] == RED)), "red markers must stay above the separator"
    return red_markers, blue_markers


def _build_halo(markers: list[Coord], shape: tuple[int, int], sep_row: int) -> np.ndarray:
    height, width = map(int, shape)
    halo = np.zeros((height, width), dtype=bool)
    for r, c in markers:
        r0 = max(0, int(r) - HALO_RADIUS)
        r1 = min(height - 1, int(r) + HALO_RADIUS)
        c0 = max(0, int(c) - HALO_RADIUS)
        c1 = min(width - 1, int(c) + HALO_RADIUS)
        halo[r0 : r1 + 1, c0 : c1 + 1] = True
    halo[int(sep_row), :] = False
    return halo


def _render_from_groups(shape: tuple[int, int], sep_row: int, red_markers: list[Coord], blue_markers: list[Coord]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[int(sep_row), :] = YELLOW

    red_halo = _build_halo(red_markers, shape, sep_row)
    blue_halo = _build_halo(blue_markers, shape, sep_row)

    out[red_halo & ~blue_halo] = RED
    out[blue_halo & ~red_halo] = BLUE
    out[red_halo & blue_halo] = GREEN

    for r, c in red_markers:
        out[int(r), int(c)] = RED
    for r, c in blue_markers:
        out[int(r), int(c)] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the red and blue marker groups around a separator row, then recolor exclusive and overlapping halo cells.

    Input structure:
    - A black background grid contains exactly one full horizontal yellow separator row.
    - Sparse red single-cell markers lie above the separator and sparse blue single-cell markers lie below it.
    - Some markers sit near the separator while others are farther away, so some expanded halos interact across groups and some remain isolated.

    Rule:
    - Use the yellow row as the group boundary and keep it unchanged.
    - Around every marker, mark its clipped 5x5 neighborhood (Chebyshev radius 2) as that group's halo, but skip every cell on the separator row.
    - Recolor cells covered only by the red-group halo to red, only by the blue-group halo to blue, and cells covered by both halos to green.
    - Finally restore the original red and blue marker pixels on top of those halo colors.

    Key insight:
    - The separator tells you which markers belong to which group, but the final color of any non-marker cell depends on whether the two independently expanded halos reach it alone or reach it together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_row = _find_separator_row(g)
    red_markers, blue_markers = _group_markers(g, sep_row)
    return _render_from_groups(g.shape, sep_row, red_markers, blue_markers)


def _chebyshev_nonadjacent(existing: list[Coord], pos: Coord) -> bool:
    r, c = pos
    return all(max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) > 1 for rr, cc in existing)


def _pair_halos_overlap(a: Coord, b: Coord, shape: tuple[int, int], sep_row: int) -> bool:
    return bool((_build_halo([a], shape, sep_row) & _build_halo([b], shape, sep_row)).any())


def _marker_crosses_group(marker: Coord, other_group: list[Coord], shape: tuple[int, int], sep_row: int) -> bool:
    mine = _build_halo([marker], shape, sep_row)
    other = _build_halo(other_group, shape, sep_row)
    return bool((mine & other).any())


def _place_specific_row(existing: list[Coord], row: int, width: int) -> Coord | None:
    cols = list(range(width))
    random.shuffle(cols)
    if width >= 2 and random.random() < 0.35:
        edge_cols = [0, width - 1]
        random.shuffle(edge_cols)
        cols = edge_cols + [c for c in cols if c not in edge_cols]
    for c in cols:
        pos = (int(row), int(c))
        if _chebyshev_nonadjacent(existing, pos):
            return pos
    return None


def _place_mandatory_pair(
    red_markers: list[Coord],
    blue_markers: list[Coord],
    shape: tuple[int, int],
    sep_row: int,
) -> bool:
    height, width = map(int, shape)
    del height
    near_red_rows = [sep_row - 1, sep_row - 2]
    near_blue_rows = [sep_row + 1, sep_row + 2]

    for _ in range(300):
        rr = int(random.choice(near_red_rows))
        rc = int(random.randrange(width))
        red_pos = (rr, rc)
        if not _chebyshev_nonadjacent(red_markers, red_pos):
            continue

        blue_rows = near_blue_rows[:]
        random.shuffle(blue_rows)
        for rb in blue_rows:
            col_lo = max(0, rc - 4)
            col_hi = min(width - 1, rc + 4)
            cols = list(range(col_lo, col_hi + 1))
            random.shuffle(cols)
            for bc in cols:
                blue_pos = (int(rb), int(bc))
                if not _chebyshev_nonadjacent(blue_markers, blue_pos):
                    continue
                if _pair_halos_overlap(red_pos, blue_pos, shape, sep_row):
                    red_markers.append(red_pos)
                    blue_markers.append(blue_pos)
                    return True
    return False


def _candidate_positions(
    rows: list[int],
    width: int,
    existing: list[Coord],
) -> list[Coord]:
    return [
        (int(r), int(c))
        for r in rows
        for c in range(width)
        if _chebyshev_nonadjacent(existing, (int(r), int(c)))
    ]


def _place_extra(
    own_markers: list[Coord],
    own_rows: list[int],
    other_markers: list[Coord],
    shape: tuple[int, int],
    sep_row: int,
    *,
    prefer_same_group_overlap: bool,
) -> bool:
    _, width = map(int, shape)
    candidates = _candidate_positions(own_rows, width, own_markers)
    if not candidates:
        return False

    if prefer_same_group_overlap and own_markers:
        overlap_candidates = []
        for pos in candidates:
            if any(_pair_halos_overlap(pos, other, shape, sep_row) for other in own_markers):
                overlap_candidates.append(pos)
        if overlap_candidates:
            candidates = overlap_candidates

    random.shuffle(candidates)
    for pos in candidates:
        own_markers.append(pos)
        return True
    return False


def _validate_scene(inp: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    sep_row = _find_separator_row(g)
    red_markers, blue_markers = _group_markers(g, sep_row)
    assert 3 <= len(red_markers) <= 5
    assert 3 <= len(blue_markers) <= 5
    allowed = {BG, BLUE, RED, YELLOW}
    assert set(map(int, np.unique(g))) <= allowed

    for group in (red_markers, blue_markers):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                assert _chebyshev_nonadjacent([a], b), "same-group markers must be non-adjacent"

    shape = g.shape
    pair_count = sum(
        1
        for red_pos in red_markers
        for blue_pos in blue_markers
        if _pair_halos_overlap(red_pos, blue_pos, shape, sep_row)
    )
    assert pair_count >= 2, "need at least two cross-group overlapping pairs"

    assert any(
        not _marker_crosses_group(red_pos, blue_markers, shape, sep_row)
        for red_pos in red_markers
    ), "need at least one red marker whose halo stays isolated from the blue group"
    assert any(
        not _marker_crosses_group(blue_pos, red_markers, shape, sep_row)
        for blue_pos in blue_markers
    ), "need at least one blue marker whose halo stays isolated from the red group"

    expected = _render_from_groups(shape, sep_row, red_markers, blue_markers)
    actual = main(g)
    assert np.array_equal(actual, expected)

    marker_mask = np.zeros(shape, dtype=bool)
    for r, c in red_markers + blue_markers:
        marker_mask[int(r), int(c)] = True

    red_halo = _build_halo(red_markers, shape, sep_row)
    blue_halo = _build_halo(blue_markers, shape, sep_row)
    green_cells = (red_halo & blue_halo) & ~marker_mask
    red_only = (red_halo & ~blue_halo) & ~marker_mask
    blue_only = (blue_halo & ~red_halo) & ~marker_mask

    assert bool(green_cells.any()), "output must contain visible conflict cells"
    assert bool(red_only.any()), "output must contain red-only halo cells"
    assert bool(blue_only.any()), "output must contain blue-only halo cells"
    assert bool(np.all(actual[sep_row, :] == YELLOW))
    assert int(np.count_nonzero(actual == GREEN)) == int(np.count_nonzero(green_cells))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(11, 16)
    width = random.randint(12, 18)
    sep_row = random.randint(4, height - 5)

    red_count = random.randint(3, 5)
    blue_count = random.randint(3, 5)

    red_markers: list[Coord] = []
    blue_markers: list[Coord] = []

    far_red = _place_specific_row(red_markers, sep_row - 4, width)
    far_blue = _place_specific_row(blue_markers, sep_row + 4, width)
    if far_red is None or far_blue is None:
        return None
    red_markers.append(far_red)
    blue_markers.append(far_blue)

    if not _place_mandatory_pair(red_markers, blue_markers, (height, width), sep_row):
        return None
    if not _place_mandatory_pair(red_markers, blue_markers, (height, width), sep_row):
        return None

    red_rows = list(range(0, sep_row))
    blue_rows = list(range(sep_row + 1, height))

    want_red_same_overlap = bool(red_count >= 4 and random.random() < 0.6)
    want_blue_same_overlap = bool(blue_count >= 4 and random.random() < 0.6)

    while len(red_markers) < red_count:
        if not _place_extra(
            red_markers,
            red_rows,
            blue_markers,
            (height, width),
            sep_row,
            prefer_same_group_overlap=want_red_same_overlap,
        ):
            return None
        want_red_same_overlap = False

    while len(blue_markers) < blue_count:
        if not _place_extra(
            blue_markers,
            blue_rows,
            red_markers,
            (height, width),
            sep_row,
            prefer_same_group_overlap=want_blue_same_overlap,
        ):
            return None
        want_blue_same_overlap = False

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = YELLOW
    for r, c in red_markers:
        grid[int(r), int(c)] = RED
    for r, c in blue_markers:
        grid[int(r), int(c)] = BLUE

    try:
        _validate_scene(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
