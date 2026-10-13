from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = (1, 2, 3, 4)
PERIOD_CHOICES = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 800


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _majority_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    counts = np.zeros((row_period, col_period, 10), dtype=int)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color != 0:
                counts[r % row_period, c % col_period, color] += 1

    tile = np.full((row_period, col_period), -1, dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            phase_counts = counts[rr, cc]
            best = int(phase_counts.max())
            if best <= 0:
                return None
            winners = np.flatnonzero(phase_counts == best)
            if len(winners) != 1:
                return None
            tile[rr, cc] = int(winners[0])
    return tile


def _strict_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0:
                continue
            rr = r % row_period
            cc = c % col_period
            known = int(tile[rr, cc])
            if known == -1:
                tile[rr, cc] = color
            elif known != color:
                return None
    if np.any(tile == -1):
        return None
    return tile


def _find_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = grid.shape
    for row_period in range(2, h + 1):
        for col_period in range(2, w + 1):
            tile = _majority_tile(grid, row_period, col_period)
            if tile is not None:
                return row_period, col_period, tile
    raise AssertionError("no valid period pair found")


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.empty(shape, dtype=int)
    th, tw = tile.shape
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _noise_mask(grid: np.ndarray, row_period: int, col_period: int, majority_tile: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    noise = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color != 0 and color != int(majority_tile[r % row_period, c % col_period]):
                noise[r, c] = True
    return noise


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove wrong-color noise blocks from a periodic wallpaper and restore the missing cells.

    Input structure:
    - The grid is a repeated non-black wallpaper tile with row and column periods at least 2.
    - Some cells are black holes, while 1-2 small wrong-color noise blocks overwrite the wallpaper.
    - Noise uses the same visible palette as the wallpaper, so only phase-consistency across repeated positions distinguishes it.

    Rule:
    - Search row and column periods from 2 upward and take the first pair whose every phase has a unique non-black majority color.
    - Mark every non-black cell whose color disagrees with its phase majority as noise, erase those cells to black, and re-confirm the phase tile on the cleaned grid.
    - Fill every black cell, including original holes and erased noise, with the color of its phase in the recovered tile.

    Key insight:
    - Connectivity and palette alone cannot isolate the noise, because the noise copies wallpaper colors; the decisive cue is that true wallpaper cells agree with the repeated phase majority while noise is the minority at its phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_period, col_period, majority_tile = _find_period(g)
    noise = _noise_mask(g, row_period, col_period, majority_tile)

    cleaned = g.copy()
    cleaned[noise] = 0
    tile = _strict_tile(cleaned, row_period, col_period)
    assert tile is not None, "cleaned grid must define a unique periodic tile"

    out = _tile_grid(tile, g.shape)
    assert np.all((cleaned == 0) | (cleaned == out)), "cleaned non-black cells must match the recovered tile"
    return out


def _shape_variants() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ]

    variants: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for shape in base_shapes:
        pts = np.array(shape, dtype=int)
        for flip_r in (1, -1):
            for flip_c in (1, -1):
                work = pts.copy()
                work[:, 0] *= flip_r
                work[:, 1] *= flip_c
                for _ in range(4):
                    rr = work[:, 0]
                    cc = work[:, 1]
                    norm = np.stack([rr - rr.min(), cc - cc.min()], axis=1)
                    key = tuple(sorted((int(r), int(c)) for r, c in norm))
                    if key not in variants:
                        variants.add(key)
                        out.append(key)
                    work = np.stack([work[:, 1], -work[:, 0]], axis=1)
    return out


SHAPES = _shape_variants()


def _random_dimensions(row_period: int, col_period: int) -> tuple[int, int]:
    row_repeats = random.randint(3, 5)
    col_repeats = random.randint(3, 5)
    extra_rows = random.randint(0, row_period - 1)
    extra_cols = random.randint(0, col_period - 1)
    height = row_period * row_repeats + extra_rows
    width = col_period * col_repeats + extra_cols
    return height, width


def _random_tile(row_period: int, col_period: int, colors: tuple[int, ...]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.array(
            [[random.choice(colors) for _ in range(col_period)] for _ in range(row_period)],
            dtype=int,
        )
        if set(map(int, tile.flat)) != set(colors):
            continue
        row_patterns = {tuple(map(int, tile[r])) for r in range(row_period)}
        col_patterns = {tuple(map(int, tile[:, c])) for c in range(col_period)}
        if len(row_patterns) < 2 or len(col_patterns) < 2:
            continue
        wall = _tile_grid(tile, (row_period * 5, col_period * 5))
        found_row, found_col, _ = _find_period(wall)
        if (found_row, found_col) == (row_period, col_period):
            return tile
    return None


def _pick_holes(wall: np.ndarray, hole_count: int) -> list[tuple[int, int]]:
    h, w = wall.shape
    coords = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(100):
        random.shuffle(coords)
        holes: list[tuple[int, int]] = []
        for r, c in coords:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in holes):
                continue
            holes.append((r, c))
            if len(holes) == hole_count:
                return holes
    return random.sample(coords, hole_count)


def _phase_correct_counts(
    wall: np.ndarray,
    holes: set[tuple[int, int]],
    noise_positions: set[tuple[int, int]],
    row_period: int,
    col_period: int,
) -> np.ndarray:
    counts = np.zeros((row_period, col_period), dtype=int)
    h, w = wall.shape
    for r in range(h):
        for c in range(w):
            if (r, c) not in holes and (r, c) not in noise_positions:
                counts[r % row_period, c % col_period] += 1
    return counts


def _choose_noise_block(
    wall: np.ndarray,
    holes: set[tuple[int, int]],
    existing_noise: set[tuple[int, int]],
    colors: tuple[int, ...],
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = wall.shape
    candidates: list[tuple[list[tuple[int, int]], int]] = []
    for shape in SHAPES:
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        positions = [(r0, c0) for r0 in range(h - shape_h + 1) for c0 in range(w - shape_w + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:80]:
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(cell in holes or cell in existing_noise for cell in cells):
                continue
            if any(
                (nr, nc) in existing_noise
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
            ):
                continue
            forbidden = {int(wall[r, c]) for r, c in cells}
            color_options = [color for color in colors if color not in forbidden]
            if not color_options:
                continue
            candidates.append((cells, random.choice(color_options)))
    if not candidates:
        return None
    return random.choice(candidates)


def _iter_earlier_pairs(shape: tuple[int, int], row_period: int, col_period: int):
    h, w = shape
    for rr in range(2, h + 1):
        for cc in range(2, w + 1):
            if (rr, cc) == (row_period, col_period):
                return
            yield rr, cc


def _validate_instance(
    inp: np.ndarray,
    wall: np.ndarray,
    row_period: int,
    col_period: int,
    holes: list[tuple[int, int]],
    noise_blocks: list[tuple[list[tuple[int, int]], int]],
) -> bool:
    found_row, found_col, majority_tile = _find_period(inp)
    if (found_row, found_col) != (row_period, col_period):
        return False

    noise_truth = {cell for cells, _ in noise_blocks for cell in cells}
    noise_found = set(map(tuple, np.argwhere(_noise_mask(inp, found_row, found_col, majority_tile))))
    if noise_found != noise_truth:
        return False

    clean = inp.copy()
    clean[tuple(np.array(list(noise_truth)).T)] = 0 if noise_truth else clean[()]
    recovered = _strict_tile(clean, row_period, col_period)
    if recovered is None or not np.array_equal(_tile_grid(recovered, inp.shape), wall):
        return False

    if not np.array_equal(main(inp), wall):
        return False

    exact_row, exact_col = _find_period(wall)[:2]
    if (exact_row, exact_col) != (row_period, col_period):
        return False

    hole_set = set(holes)
    visible_counts = _phase_correct_counts(wall, hole_set, noise_truth, row_period, col_period)
    if np.any(visible_counts < 2):
        return False

    noise_comps = _components(_noise_mask(inp, row_period, col_period, majority_tile))
    if len(noise_comps) != len(noise_blocks):
        return False
    if any(len(comp) < 2 or len(comp) > 4 for comp in noise_comps):
        return False

    for cells, color in noise_blocks:
        if len(cells) < 2 or len(cells) > 4:
            return False
        if len(_components(_mask_from_cells(inp.shape, cells))) != 1:
            return False
        if len({int(inp[r, c]) for r, c in cells}) != 1 or int(inp[cells[0]]) != color:
            return False
        if any(int(wall[r, c]) == color for r, c in cells):
            return False

    for rr, cc in _iter_earlier_pairs(inp.shape, row_period, col_period):
        if _majority_tile(inp, rr, cc) is not None:
            return False

    return True


def _mask_from_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _attempt() -> np.ndarray | None:
    row_period = random.choice(PERIOD_CHOICES)
    col_period = random.choice(PERIOD_CHOICES)
    color_count = random.choice((3, 4))
    colors = tuple(sorted(random.sample(PALETTE, color_count)))

    tile = _random_tile(row_period, col_period, colors)
    if tile is None:
        return None

    shape = _random_dimensions(row_period, col_period)
    wall = _tile_grid(tile, shape)

    holes = _pick_holes(wall, random.randint(3, 6))
    inp = wall.copy()
    for r, c in holes:
        inp[r, c] = 0

    noise_blocks: list[tuple[list[tuple[int, int]], int]] = []
    noise_positions: set[tuple[int, int]] = set()
    block_count = random.choice((1, 2))
    for _ in range(block_count):
        picked = _choose_noise_block(wall, set(holes), noise_positions, colors)
        if picked is None:
            return None
        cells, color = picked
        for r, c in cells:
            inp[r, c] = color
        noise_blocks.append((cells, color))
        noise_positions.update(cells)

    if not _validate_instance(inp, wall, row_period, col_period, holes, noise_blocks):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-noise ARC input")
