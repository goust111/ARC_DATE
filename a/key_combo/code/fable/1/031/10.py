import random

import numpy as np


DEFECT_COLOR = 2
UNKNOWN_COLOR = -1
BASE_COLOR_PAIRS = [
    (0, 4),
    (0, 6),
    (0, 8),
    (1, 3),
    (1, 7),
    (3, 6),
    (4, 8),
    (5, 7),
    (6, 9),
]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _grow_connected_mask(height: int, width: int, target_cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < target_cells and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _has_smaller_square_period(tile: np.ndarray) -> bool:
    p = int(tile.shape[0])
    for q in range(1, p):
        if np.array_equal(tile, np.roll(tile, -q, axis=0)) and np.array_equal(tile, np.roll(tile, -q, axis=1)):
            return True
    return False


def _infer_period_and_tile(grid: np.ndarray) -> tuple[int, np.ndarray]:
    grid = np.asarray(grid, dtype=int)
    height, width = map(int, grid.shape)
    non_defect_colors = sorted(set(map(int, np.unique(grid))) - {DEFECT_COLOR})
    assert len(non_defect_colors) == 2, "expected exactly two non-red lattice colors"

    for period in range(2, min(height, width) + 1):
        tile = np.full((period, period), UNKNOWN_COLOR, dtype=int)
        ok = True
        for r in range(height):
            for c in range(width):
                color = int(grid[r, c])
                if color == DEFECT_COLOR:
                    continue
                pr = r % period
                pc = c % period
                if tile[pr, pc] == UNKNOWN_COLOR:
                    tile[pr, pc] = color
                elif tile[pr, pc] != color:
                    ok = False
                    break
            if not ok:
                break
        if not ok or np.any(tile == UNKNOWN_COLOR):
            continue
        if len(set(map(int, tile.flat))) != 2:
            continue
        return period, tile

    raise AssertionError("failed to infer a fully observed square period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the lattice period, fold every red defect back by modulo phase, and output one marked base tile.

    Input structure:
    - The whole grid is tiled by a hidden p-by-p two-color periodic lattice.
    - Some lattice cells are replaced by isolated red defect pixels.
    - Red may appear in several distant copies of the same modulo phase.

    Rule:
    - Recover the smallest square period whose non-red cells are consistent under modulo folding.
    - Reconstruct that p-by-p base tile from the clean observations.
    - For every red input cell, mark the corresponding (row mod p, col mod p) phase red in the output tile.
    - Leave every other phase at its recovered lattice color.

    Key insight:
    - The absolute red positions are noise; only their coordinates modulo the hidden lattice period matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    period, tile = _infer_period_and_tile(g)
    out = tile.copy()

    defect_rows, defect_cols = np.where(g == DEFECT_COLOR)
    out[defect_rows % period, defect_cols % period] = DEFECT_COLOR
    return out


def _make_base_tile(period: int) -> np.ndarray:
    color_a, color_b = random.choice(BASE_COLOR_PAIRS)
    min_cells = max(3, period)
    max_cells = min(period * period - 3, (period * period * 3) // 5)
    if max_cells < min_cells:
        max_cells = min_cells

    for _ in range(80):
        target = random.randint(min_cells, max_cells)
        motif = _grow_connected_mask(period, period, target)
        if not motif.any() or motif.all():
            continue

        tile = np.full((period, period), color_a, dtype=int)
        tile[motif] = color_b
        if len(set(map(int, tile.flat))) != 2:
            continue
        if _has_smaller_square_period(tile):
            continue
        return tile

    raise RuntimeError("failed to build a fundamental base tile")


def _make_defect_mask(period: int, max_cells: int) -> np.ndarray:
    upper = min(max_cells, period + 1, 6)
    lower = 1 if upper == 1 else 2
    count = random.randint(lower, upper)
    return _grow_connected_mask(period, period, count)


def _slot_options(nr: int, nc: int, needed: int) -> list[list[tuple[int, int]]]:
    options = []
    for row_offset in range(2):
        for col_offset in range(2):
            slots = [(tr, tc) for tr in range(row_offset, nr, 2) for tc in range(col_offset, nc, 2)]
            if len(slots) >= needed:
                options.append(slots)
    return options


def _defect_slots(nr: int, nc: int, needed: int) -> list[tuple[int, int]] | None:
    options = _slot_options(nr, nc, needed)
    if not options:
        return None
    return random.choice(options)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    period = random.randint(3, 8)
    max_repeat = 30 // period
    if max_repeat < 3:
        raise RuntimeError("period too large for ARC bounds")

    nr = random.randint(3, max_repeat)
    nc = random.randint(3, max_repeat)

    base_tile = _make_base_tile(period)
    grid = np.tile(base_tile, (nr, nc))

    slot_options = _slot_options(nr, nc, needed=1)
    if not slot_options:
        raise RuntimeError("no defect slots")

    max_marked = min(max(len(slots) for slots in slot_options), period + 1, 6)
    defect_mask = _make_defect_mask(period, max_marked)
    phase_coords = [tuple(map(int, rc)) for rc in np.argwhere(defect_mask)]
    if not phase_coords:
        raise RuntimeError("empty defect mask")

    eligible_slots = _slot_options(nr, nc, needed=len(phase_coords))
    if not eligible_slots:
        raise RuntimeError("not enough separated slots")
    slots = random.choice(eligible_slots)

    extra_budget = min(len(slots) - len(phase_coords), len(phase_coords))
    total_defects = len(phase_coords) + random.randint(0, max(0, extra_budget))
    chosen_slots = random.sample(slots, total_defects)
    random.shuffle(chosen_slots)

    assignments = {}
    for slot, phase in zip(chosen_slots, phase_coords):
        assignments[slot] = phase
    for slot in chosen_slots[len(phase_coords) :]:
        assignments[slot] = random.choice(phase_coords)

    for (tile_r, tile_c), (phase_r, phase_c) in assignments.items():
        grid[tile_r * period + phase_r, tile_c * period + phase_c] = DEFECT_COLOR

    expected = base_tile.copy()
    expected[defect_mask] = DEFECT_COLOR

    inferred_period, inferred_tile = _infer_period_and_tile(grid)
    if inferred_period != period:
        raise RuntimeError("ambiguous period")
    if not np.array_equal(main(grid), expected):
        raise RuntimeError("solver/output mismatch")
    if np.array_equal(inferred_tile, expected):
        raise RuntimeError("defects disappeared in folded tile")

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, _ = _build_case()
            return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid input")
