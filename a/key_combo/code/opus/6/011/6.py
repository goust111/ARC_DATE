import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 2
SEED_COLORS = (1, 3, 4, 7)
GEN_ATTEMPTS = 500


def _all_full_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _bands_from_separators(seps: list[int], limit: int) -> list[tuple[int, int]]:
    assert seps[0] == 0 and seps[-1] == limit - 1, "separator frame must touch the grid border"
    bands = []
    for left, right in zip(seps, seps[1:]):
        gap = int(right - left - 1)
        assert gap in (2, 3), "encoded band size must be 2 or 3"
        bands.append((left + 1, right - 1))
    return bands


def _parse_cells(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == SEP), "expected separator lines"

    sep_rows = _all_full_indices(g, SEP, axis=0)
    sep_cols = _all_full_indices(g, SEP, axis=1)
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "expected full separator lattice"

    row_bands = _bands_from_separators(sep_rows, g.shape[0])
    col_bands = _bands_from_separators(sep_cols, g.shape[1])

    colors = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(cell == SEP), "separator color cannot appear inside a cell"
            payload = sorted(int(v) for v in np.unique(cell) if v != BG)
            assert all(v in SEED_COLORS for v in payload), "unexpected seed color"
            assert len(payload) <= 1, "each cell must be empty or contain one seed color"
            if payload:
                colors[ri, ci] = payload[0]
    return sep_rows, sep_cols, row_bands, col_bands, colors


def _render_from_cells(
    shape: tuple[int, int],
    sep_rows: list[int],
    sep_cols: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    colors: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, SEP, dtype=int)

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            color = int(colors[ri, ci])
            out[r0 : r1 + 1, c0 : c1 + 1] = color if color else BG

    for ri, (r0, r1) in enumerate(row_bands):
        for ci in range(colors.shape[1] - 1):
            color = int(colors[ri, ci])
            if color != 0 and color == int(colors[ri, ci + 1]):
                mid_c = sep_cols[ci + 1]
                out[r0 : r1 + 1, mid_c] = color

    for ri in range(colors.shape[0] - 1):
        mid_r = sep_rows[ri + 1]
        for ci, (c0, c1) in enumerate(col_bands):
            color = int(colors[ri, ci])
            if color != 0 and color == int(colors[ri + 1, ci]):
                out[mid_r, c0 : c1 + 1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand seeded separator-lattice cells and merge same-color neighbors by erasing only their shared inner separator segments.

    Input structure:
    - The grid is framed and partitioned by complete red separator rows and columns on a black background.
    - Adjacent separator lines are separated by 2 or 3 non-separator cells, so each lattice cell has an encoded height and width.
    - Each cell interior is either empty black or contains one seed color from blue, green, yellow, or orange.

    Rule:
    - Read each cell's height from its row-band gap and its width from its column-band gap.
    - Expand every non-empty seed to fill the entire non-separator interior of its own cell with that same color; empty cells stay black.
    - For every horizontally or vertically adjacent pair of filled cells, recolor the shared internal separator segment only when the two expanded colors are equal.
    - Keep all other separator pixels red, including the full outer border and every separator intersection.

    Key insight:
    - The separator lattice is both the codebook and the canvas: its uneven gaps determine each rectangle's growth size, then same-color adjacency selectively removes only the internal line segments between matching grown blocks.
    """
    sep_rows, sep_cols, row_bands, col_bands, colors = _parse_cells(grid)
    return _render_from_cells(np.asarray(grid).shape, sep_rows, sep_cols, row_bands, col_bands, colors)


def _band_sizes(count: int) -> list[int]:
    while True:
        sizes = [random.choice((2, 3)) for _ in range(count)]
        if 2 in sizes and 3 in sizes:
            return sizes


def _band_positions(sizes: Iterable[int]) -> tuple[list[int], list[tuple[int, int]]]:
    seps = [0]
    bands = []
    cursor = 0
    for size in sizes:
        r0 = cursor + 1
        r1 = r0 + int(size) - 1
        bands.append((r0, r1))
        cursor = r1 + 1
        seps.append(cursor)
    return seps, bands


def _seed_offset(height: int, width: int, ri: int, ci: int) -> tuple[int, int]:
    choices = {
        (2, 2): [(0, 0), (1, 1), (0, 1), (1, 0)],
        (2, 3): [(0, 1), (1, 2), (1, 0), (0, 2)],
        (3, 2): [(1, 0), (0, 1), (2, 1), (1, 1)],
        (3, 3): [(1, 1), (0, 0), (2, 2), (0, 2), (2, 0)],
    }[(height, width)]
    return choices[(ri * 5 + ci * 3) % len(choices)]


def _adjacent_pairs(colors: np.ndarray) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    horiz = []
    vert = []
    for ri in range(colors.shape[0]):
        for ci in range(colors.shape[1] - 1):
            horiz.append(((ri, ci), (ri, ci + 1)))
    for ri in range(colors.shape[0] - 1):
        for ci in range(colors.shape[1]):
            vert.append(((ri, ci), (ri + 1, ci)))
    return horiz, vert


def _has_chain(colors: np.ndarray) -> bool:
    for ri in range(colors.shape[0]):
        for ci in range(colors.shape[1] - 2):
            color = int(colors[ri, ci])
            if color != 0 and color == int(colors[ri, ci + 1]) == int(colors[ri, ci + 2]):
                return True
    for ri in range(colors.shape[0] - 2):
        for ci in range(colors.shape[1]):
            color = int(colors[ri, ci])
            if color != 0 and color == int(colors[ri + 1, ci]) == int(colors[ri + 2, ci]):
                return True
    return False


def _filled_in_rows(colors: np.ndarray, row_sizes: list[int], want: int) -> bool:
    return any(row_sizes[ri] == want and np.any(colors[ri, :] != 0) for ri in range(len(row_sizes)))


def _filled_in_cols(colors: np.ndarray, col_sizes: list[int], want: int) -> bool:
    return any(col_sizes[ci] == want and np.any(colors[:, ci] != 0) for ci in range(len(col_sizes)))


def _semantic_checks(colors: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> bool:
    total = int(colors.size)
    filled = int(np.count_nonzero(colors))
    if not (int(np.ceil(0.6 * total)) <= filled <= int(np.floor(0.8 * total))):
        return False
    used = {int(v) for v in colors.flat if v != 0}
    if len(used) < 3:
        return False
    if filled == total:
        return False
    if not (_filled_in_rows(colors, row_sizes, 2) and _filled_in_rows(colors, row_sizes, 3)):
        return False
    if not (_filled_in_cols(colors, col_sizes, 2) and _filled_in_cols(colors, col_sizes, 3)):
        return False

    horiz_pairs, vert_pairs = _adjacent_pairs(colors)
    same_h = 0
    same_v = 0
    diff_or_empty = 0
    has_empty_neighbor = False
    for a, b in horiz_pairs + vert_pairs:
        ca = int(colors[a])
        cb = int(colors[b])
        if ca != 0 and ca == cb:
            if a[0] == b[0]:
                same_h += 1
            else:
                same_v += 1
        else:
            diff_or_empty += 1
            if (ca == 0) != (cb == 0):
                has_empty_neighbor = True
    if same_h < 1 or same_v < 1:
        return False
    if same_h + same_v < 2:
        return False
    if diff_or_empty < 3 or not has_empty_neighbor:
        return False
    if not _has_chain(colors):
        return False
    return True


def _place_segment(colors: np.ndarray, orient: str, length: int, color: int, blocked: set[tuple[int, int]]) -> bool:
    nr, nc = colors.shape
    for _ in range(80):
        if orient == "h":
            ri = random.randrange(nr)
            ci = random.randrange(nc - length + 1)
            coords = [(ri, ci + k) for k in range(length)]
        else:
            ri = random.randrange(nr - length + 1)
            ci = random.randrange(nc)
            coords = [(ri + k, ci) for k in range(length)]
        if any(coord in blocked for coord in coords):
            continue
        for coord in coords:
            colors[coord] = color
            blocked.add(coord)
        return True
    return False


def _maybe_bias_color(colors: np.ndarray, r: int, c: int) -> int:
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < colors.shape[0] and 0 <= cc < colors.shape[1] and colors[rr, cc] != 0:
            neighbors.append(int(colors[rr, cc]))
    if neighbors and random.random() < 0.35:
        return random.choice(neighbors)
    if neighbors and random.random() < 0.55:
        options = [color for color in SEED_COLORS if color not in neighbors]
        if options:
            return random.choice(options)
    return random.choice(SEED_COLORS)


def _sample_colors(row_sizes: list[int], col_sizes: list[int]) -> np.ndarray | None:
    nr = len(row_sizes)
    nc = len(col_sizes)
    total = nr * nc
    min_filled = int(np.ceil(0.6 * total))
    max_filled = max(min_filled, int(np.floor(0.8 * total)))
    target_filled = random.randint(min_filled, max_filled)

    colors = np.zeros((nr, nc), dtype=int)
    blocked: set[tuple[int, int]] = set()
    chain_color, pair_color = random.sample(SEED_COLORS, 2)
    chain_orient = random.choice(("h", "v"))
    if not _place_segment(colors, chain_orient, 3, chain_color, blocked):
        return None
    if not _place_segment(colors, "v" if chain_orient == "h" else "h", 2, pair_color, blocked):
        return None

    required_positions = set(blocked)
    for want in (2, 3):
        if not _filled_in_rows(colors, row_sizes, want):
            candidates = [(ri, ci) for ri in range(nr) if row_sizes[ri] == want for ci in range(nc) if colors[ri, ci] == 0]
            if not candidates:
                return None
            r, c = random.choice(candidates)
            colors[r, c] = random.choice(SEED_COLORS)
            required_positions.add((r, c))
        if not _filled_in_cols(colors, col_sizes, want):
            candidates = [(ri, ci) for ci in range(nc) if col_sizes[ci] == want for ri in range(nr) if colors[ri, ci] == 0]
            if not candidates:
                return None
            r, c = random.choice(candidates)
            colors[r, c] = random.choice(SEED_COLORS)
            required_positions.add((r, c))

    empties_needed = total - target_filled
    if empties_needed < 1:
        return None

    while int(np.count_nonzero(colors)) < target_filled:
        empties = list(zip(*np.where(colors == 0)))
        if not empties:
            break
        r, c = random.choice(empties)
        colors[r, c] = _maybe_bias_color(colors, int(r), int(c))

    if int(np.count_nonzero(colors)) != target_filled:
        return None

    used = {int(v) for v in colors.flat if v != 0}
    if len(used) < 3:
        mutable = [(r, c) for r in range(nr) for c in range(nc) if colors[r, c] != 0 and (r, c) not in required_positions]
        random.shuffle(mutable)
        for r, c in mutable:
            missing = [color for color in SEED_COLORS if color not in used]
            if not missing:
                break
            colors[r, c] = missing[0]
            used.add(missing[0])

    if not _semantic_checks(colors, row_sizes, col_sizes):
        return None
    return colors


def _build_input(row_sizes: list[int], col_sizes: list[int], colors: np.ndarray) -> np.ndarray:
    sep_rows, row_bands = _band_positions(row_sizes)
    sep_cols, col_bands = _band_positions(col_sizes)
    h = sep_rows[-1] + 1
    w = sep_cols[-1] + 1
    grid = np.full((h, w), SEP, dtype=int)

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            grid[r0 : r1 + 1, c0 : c1 + 1] = BG
            color = int(colors[ri, ci])
            if color == 0:
                continue
            dr, dc = _seed_offset(r1 - r0 + 1, c1 - c0 + 1, ri, ci)
            grid[r0 + dr, c0 + dc] = color
    return grid


def _validate_input_structure(grid: np.ndarray, row_sizes: list[int], col_sizes: list[int], colors: np.ndarray) -> bool:
    try:
        sep_rows, sep_cols, row_bands, col_bands, parsed = _parse_cells(grid)
    except AssertionError:
        return False
    if parsed.shape != colors.shape or not np.array_equal(parsed, colors):
        return False
    if [r1 - r0 + 1 for r0, r1 in row_bands] != row_sizes:
        return False
    if [c1 - c0 + 1 for c0, c1 in col_bands] != col_sizes:
        return False
    if not np.all(grid[0, :] == SEP) or not np.all(grid[-1, :] == SEP):
        return False
    if not np.all(grid[:, 0] == SEP) or not np.all(grid[:, -1] == SEP):
        return False
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            payload = np.argwhere((cell != BG) & (cell != SEP))
            expected = int(colors[ri, ci])
            if expected == 0 and payload.size != 0:
                return False
            if expected != 0 and payload.shape != (1, 2):
                return False
    return True


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray, colors: np.ndarray) -> bool:
    sep_rows, sep_cols, row_bands, col_bands, _ = _parse_cells(inp)
    expected = _render_from_cells(inp.shape, sep_rows, sep_cols, row_bands, col_bands, colors)
    if not np.array_equal(out, expected):
        return False
    if not np.all(out[0, :] == SEP) or not np.all(out[-1, :] == SEP):
        return False
    if not np.all(out[:, 0] == SEP) or not np.all(out[:, -1] == SEP):
        return False
    for r in sep_rows[1:-1]:
        for c in sep_cols[1:-1]:
            if out[r, c] != SEP:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    nr = random.randint(4, 6)
    nc = random.randint(4, 6)
    row_sizes = _band_sizes(nr)
    col_sizes = _band_sizes(nc)
    if sum(row_sizes) + nr + 1 > 30 or sum(col_sizes) + nc + 1 > 30:
        return None

    colors = _sample_colors(row_sizes, col_sizes)
    if colors is None:
        return None
    grid = _build_input(row_sizes, col_sizes, colors)
    if max(grid.shape) > 30:
        return None
    if not _validate_input_structure(grid, row_sizes, col_sizes, colors):
        return None
    out = main(grid)
    if not _validate_output_semantics(grid, out, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid separator-growth input")
