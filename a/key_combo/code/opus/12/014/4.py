import random

import numpy as np


BG = 0
TILE_COLORS = (1, 2, 3)
PLACEHOLDER = 5
MARKER = 7
GEN_RETRIES = 400
QUADRANTS = ("UL", "UR", "LL", "LR")


def _quadrant_slices(shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    height, width = shape
    half_h = height // 2
    half_w = width // 2
    return {
        "UL": (slice(0, half_h), slice(0, half_w)),
        "UR": (slice(0, half_h), slice(half_w, width)),
        "LL": (slice(half_h, height), slice(0, half_w)),
        "LR": (slice(half_h, height), slice(half_w, width)),
    }


def _count_markers(grid: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, (rs, cs) in _quadrant_slices(grid.shape).items():
        counts[name] = int(np.count_nonzero(grid[rs, cs] == MARKER))
    return counts


def _find_hidden_cuts(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    hidden = (grid == PLACEHOLDER) | (grid == MARKER)
    matches: list[tuple[int, int]] = []
    for r_cut in range(1, height + 1):
        for c_cut in range(1, width + 1):
            expected = np.zeros_like(hidden, dtype=bool)
            if r_cut < height:
                expected[r_cut:, :] = True
            if c_cut < width:
                expected[:, c_cut:] = True
            if np.array_equal(hidden, expected):
                matches.append((r_cut, c_cut))
    assert len(matches) == 1, "hidden region must be a unique suffix union"
    return matches[0]


def _minimal_periods(block: np.ndarray) -> tuple[int, int]:
    height, width = block.shape

    row_period = None
    for period in range(1, height + 1):
        if all(block[r, c] == block[r % period, c] for r in range(height) for c in range(width)):
            row_period = period
            break

    col_period = None
    for period in range(1, width + 1):
        if all(block[r, c] == block[r, c % period] for r in range(height) for c in range(width)):
            col_period = period
            break

    assert row_period is not None and col_period is not None
    return int(row_period), int(col_period)


def _tile_to_full(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _shift_quadrants(full: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    out = full.copy()
    for name, (rs, cs) in _quadrant_slices(full.shape).items():
        patch = full[rs, cs]
        steps = int(counts[name])
        if name == "UL":
            moved = np.roll(patch, steps, axis=1)
        elif name == "UR":
            moved = np.roll(patch, steps, axis=0)
        elif name == "LL":
            moved = np.roll(patch, -steps, axis=0)
        else:
            moved = np.roll(patch, -steps, axis=1)
        out[rs, cs] = moved
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the hidden periodic tiling, then shift each quadrant by its marker count.

    Input structure:
    - A black background grid contains a visible top-left rectangular block filled by a repeating tile using only blue, red, and green.
    - The rest of the grid is hidden by a suffix-union placeholder region made of grey cells, with isolated orange marker pixels embedded inside that hidden region.
    - The grid height and width are even, so the full grid can be split into equal upper-left, upper-right, lower-left, and lower-right quadrants.

    Rule:
    - Count the orange marker pixels separately in the four quadrants before any completion happens.
    - Infer the minimal repeating tile from the visible top-left non-placeholder block, then extend that tile periodically across the full grid, overwriting both grey placeholders and orange markers.
    - Shift the completed quadrants independently and cyclically: upper-left moves right by its count, upper-right moves down, lower-left moves up, and lower-right moves left.

    Key insight:
    - The orange markers live inside the region that gets overwritten by periodic completion, so their counts must be extracted first; quadrant position gives the shift direction and marker count gives the shift distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height % 2 == 0 and width % 2 == 0

    counts = _count_markers(g)
    r_cut, c_cut = _find_hidden_cuts(g)
    block = g[:r_cut, :c_cut]
    assert block.size > 0
    assert not np.any((block == PLACEHOLDER) | (block == MARKER))

    row_period, col_period = _minimal_periods(block)
    assert row_period < r_cut and col_period < c_cut
    tile = block[:row_period, :col_period]
    assert len(set(map(int, tile.flat))) >= 2

    full = _tile_to_full(tile, g.shape)
    return _shift_quadrants(full, counts)


def _random_tile(tile_h: int, tile_w: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.choice((2, 3))
        colors = random.sample(TILE_COLORS, n_colors)
        tile = np.array(
            [[random.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _minimal_periods(tile) != (tile_h, tile_w):
            continue
        return tile
    raise RuntimeError("failed to sample a minimal tile")


def _choose_geometry(tile_h: int, tile_w: int) -> tuple[int, int, int, int]:
    reps_h = random.randint(2, 5)
    reps_w = random.randint(2, 5)
    half_h = tile_h * reps_h
    half_w = tile_w * reps_w
    return 2 * half_h, 2 * half_w, half_h, half_w


def _hidden_quadrants(hidden: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    cells: dict[str, list[tuple[int, int]]] = {}
    for name, (rs, cs) in _quadrant_slices(hidden.shape).items():
        pts = np.argwhere(hidden[rs, cs])
        r0 = rs.start or 0
        c0 = cs.start or 0
        cells[name] = [(int(r + r0), int(c + c0)) for r, c in pts]
    return cells


def _choose_cuts(
    height: int,
    width: int,
    half_h: int,
    half_w: int,
    tile_h: int,
    tile_w: int,
    need_ul_hidden: bool,
) -> tuple[int, int]:
    row_lo = tile_h + 1
    col_lo = tile_w + 1
    options: list[tuple[int, int]] = []
    for r_cut in range(row_lo, height):
        for c_cut in range(col_lo, width):
            ul_hidden = r_cut < half_h or c_cut < half_w
            if ul_hidden != need_ul_hidden:
                continue
            hidden = np.zeros((height, width), dtype=bool)
            hidden[r_cut:, :] = True
            hidden[:, c_cut:] = True
            ul_cells = np.count_nonzero(hidden[:half_h, :half_w])
            if need_ul_hidden and ul_cells < 3:
                continue
            options.append((r_cut, c_cut))
    assert options
    return random.choice(options)


def _axis_for_quadrant(name: str) -> str:
    return "h" if name in ("UL", "LR") else "v"


def _choose_counts(tile_h: int, tile_w: int) -> dict[str, int]:
    max_steps = {"h": tile_w - 1, "v": tile_h - 1}
    support_axis = "v" if max_steps["v"] >= 2 else "h"
    support_quadrants = [q for q in QUADRANTS if _axis_for_quadrant(q) == support_axis]
    q2 = random.choice(support_quadrants)
    q1 = random.choice([q for q in QUADRANTS if q != q2])
    counts = {q: 0 for q in QUADRANTS}
    counts[q2] = 2
    counts[q1] = 1
    return counts


def _markers_non_adjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _place_markers(hidden: np.ndarray, counts: dict[str, int]) -> list[tuple[int, int]] | None:
    by_quad = _hidden_quadrants(hidden)
    active = [q for q in QUADRANTS if counts[q] > 0]
    for _ in range(200):
        placed: list[tuple[int, int]] = []
        ok = True
        random.shuffle(active)
        for name in active:
            need = counts[name]
            candidates = by_quad[name][:]
            random.shuffle(candidates)
            chosen: list[tuple[int, int]] = []
            for cell in candidates:
                if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in placed + chosen):
                    continue
                chosen.append(cell)
                if len(chosen) == need:
                    break
            if len(chosen) != need:
                ok = False
                break
            placed.extend(chosen)
        if ok and _markers_non_adjacent(placed):
            return placed
    return None


def _expected_from_plan(tile: np.ndarray, shape: tuple[int, int], counts: dict[str, int]) -> np.ndarray:
    full = _tile_to_full(tile, shape)
    return _shift_quadrants(full, counts)


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    counts: dict[str, int],
    r_cut: int,
    c_cut: int,
) -> bool:
    hidden = (grid == PLACEHOLDER) | (grid == MARKER)
    check_hidden = np.zeros_like(hidden, dtype=bool)
    check_hidden[r_cut:, :] = True
    check_hidden[:, c_cut:] = True
    if not np.array_equal(hidden, check_hidden):
        return False
    if _count_markers(grid) != counts:
        return False
    positives = sorted(v for v in counts.values() if v > 0)
    if len(positives) < 2 or len(set(positives)) != len(positives) or 0 not in counts.values():
        return False
    if not _markers_non_adjacent([tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]):
        return False
    block = grid[:r_cut, :c_cut]
    if _minimal_periods(block) != tile.shape:
        return False
    if np.array_equal(expected, _tile_to_full(tile, grid.shape)):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    tile_h, tile_w = random.choice(((2, 3), (3, 2)))
    height, width, half_h, half_w = _choose_geometry(tile_h, tile_w)
    tile = _random_tile(tile_h, tile_w)
    counts = _choose_counts(tile_h, tile_w)

    need_ul_hidden = counts["UL"] > 0 or random.random() < 0.45
    r_cut, c_cut = _choose_cuts(height, width, half_h, half_w, tile_h, tile_w, need_ul_hidden)

    full = _tile_to_full(tile, (height, width))
    grid = full.copy()
    hidden = np.zeros((height, width), dtype=bool)
    hidden[r_cut:, :] = True
    hidden[:, c_cut:] = True
    grid[hidden] = PLACEHOLDER

    markers = _place_markers(hidden, counts)
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = MARKER

    expected = _expected_from_plan(tile, (height, width), counts)
    if not _validate_instance(grid, expected, tile, counts, r_cut, c_cut):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
