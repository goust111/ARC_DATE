from __future__ import annotations

import random

import numpy as np


SEED_SIZE = 3
BG = 0
SEED_FG = 3
GRAY = 5
OUTSIDE_PALETTE = (1, 2, 4, 6, 7, 8)
GRID_CHOICES = (9, 12)
HOLE_RATIO_RANGE = (0.20, 0.30)
MAX_GENERATION_ATTEMPTS = 400


def _is_seed_cell(r: int, c: int) -> bool:
    return r < SEED_SIZE and c < SEED_SIZE


def _seed_mask(grid: np.ndarray) -> np.ndarray:
    seed = np.asarray(grid[:SEED_SIZE, :SEED_SIZE], dtype=int)
    return seed != BG


def _outside_diagonal_colors(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    colors: dict[int, int] = {}
    for s in range(h + w - 1):
        seen = {
            int(grid[r, s - r])
            for r in range(max(0, s - (w - 1)), min(h - 1, s) + 1)
            if not _is_seed_cell(r, s - r) and int(grid[r, s - r]) != BG
        }
        if seen:
            assert len(seen) == 1, f"anti-diagonal {s} is ambiguous"
            colors[s] = next(iter(seen))
    return colors


def _fill_occlusions(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = _outside_diagonal_colors(g)
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if _is_seed_cell(r, c) or out[r, c] != BG:
                continue
            s = r + c
            assert s in colors, f"missing visible reference for anti-diagonal {s}"
            out[r, c] = colors[s]
    return out


def _apply_seed_mask(filled: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    keep = seed_mask[rows % SEED_SIZE, cols % SEED_SIZE]
    outside = np.ones((h, w), dtype=bool)
    outside[:SEED_SIZE, :SEED_SIZE] = False

    out = filled.copy()
    out[outside & ~keep] = GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill anti-diagonal holes, then tile the top-left seed mask to punch grey cells outside the seed.

    Input structure:
    - The top-left 3x3 seed region contains green foreground cells and black background cells forming a binary mask.
    - Outside the seed, non-black cells form an anti-diagonal color field: cells with the same row+col sum share one color.
    - Some outside field cells are black occlusions; the seed itself is never occluded.

    Rule:
    - Infer the unique visible non-black color on each anti-diagonal outside the seed.
    - Replace every black outside cell with that anti-diagonal color.
    - Treat the 3x3 seed as a binary mask repeated with period 3 across the rest of the grid: mask foreground keeps the filled color, mask background becomes grey.
    - Leave the 3x3 seed unchanged.

    Key insight:
    - The same top-left 3x3 patch plays two roles: it is the untampered legend for the periodic mask, while the outside area must first be restored by anti-diagonal color continuity before the grey punching step can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= SEED_SIZE and g.shape[1] >= SEED_SIZE

    seed_mask = _seed_mask(g)
    filled = _fill_occlusions(g)
    return _apply_seed_mask(filled, seed_mask)


def _random_seed_mask() -> np.ndarray:
    counts = (3, 4, 4, 5, 5)
    while True:
        n_fg = random.choice(counts)
        flat = np.zeros(SEED_SIZE * SEED_SIZE, dtype=bool)
        flat[random.sample(range(SEED_SIZE * SEED_SIZE), n_fg)] = True
        mask = flat.reshape(SEED_SIZE, SEED_SIZE)
        if len(set(np.where(mask)[0])) < 2:
            continue
        if len(set(np.where(mask)[1])) < 2:
            continue
        if len(set(np.where(~mask)[0])) < 2:
            continue
        if len(set(np.where(~mask)[1])) < 2:
            continue
        return mask


def _diagonal_sequence() -> list[int]:
    period = random.randint(4, len(OUTSIDE_PALETTE))
    seq = random.sample(list(OUTSIDE_PALETTE), period)
    shift = random.randrange(period)
    return seq[shift:] + seq[:shift]


def _build_filled_grid(shape: tuple[int, int], seed_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    seq = _diagonal_sequence()
    full = np.zeros((h, w), dtype=int)
    seed_patch = full[:SEED_SIZE, :SEED_SIZE]
    seed_patch[seed_mask] = SEED_FG

    for r in range(h):
        for c in range(w):
            if _is_seed_cell(r, c):
                continue
            full[r, c] = seq[(r + c - SEED_SIZE) % len(seq)]
    return full


def _outside_groups(shape: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    h, w = shape
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if _is_seed_cell(r, c):
                continue
            groups.setdefault(r + c, []).append((r, c))
    return groups


def _sample_hole_mask(shape: tuple[int, int]) -> np.ndarray | None:
    groups = _outside_groups(shape)
    total = sum(len(coords) for coords in groups.values())
    target_lo = max(1, int(np.ceil(total * HOLE_RATIO_RANGE[0])))
    target_hi = max(target_lo, int(np.floor(total * HOLE_RATIO_RANGE[1])))
    target = random.randint(target_lo, target_hi)

    capacities = {s: len(coords) - 1 for s, coords in groups.items()}
    available_sums = [s for s, cap in capacities.items() if cap > 0]
    if not available_sums:
        return None

    holes_by_sum = {s: 0 for s in groups}
    forced: list[int] = []
    special = [s for s in (3, 4) if capacities.get(s, 0) > 0]
    if special and target > 0:
        forced.append(random.choice(special))

    extra_count = min(len(available_sums), max(0, random.randint(2, 6) - len(forced)))
    other_choices = [s for s in available_sums if s not in forced]
    if extra_count > 0 and other_choices:
        max_extra = max(0, min(extra_count, len(other_choices), target - len(forced)))
        forced.extend(random.sample(other_choices, max_extra))

    for s in forced:
        if target == 0 or holes_by_sum[s] >= capacities[s]:
            continue
        holes_by_sum[s] += 1
        target -= 1

    while target > 0:
        choices = [s for s in available_sums if holes_by_sum[s] < capacities[s]]
        if not choices:
            return None
        weights = [capacities[s] - holes_by_sum[s] for s in choices]
        s = random.choices(choices, weights=weights, k=1)[0]
        holes_by_sum[s] += 1
        target -= 1

    mask = np.zeros(shape, dtype=bool)
    for s, count in holes_by_sum.items():
        if count == 0:
            continue
        coords = random.sample(groups[s], count)
        for r, c in coords:
            mask[r, c] = True

    if not (mask.sum() >= target_lo and mask.sum() <= target_hi):
        return None
    if not any(mask[r, c] for s in (3, 4) for r, c in groups.get(s, [])):
        return None
    return mask


def _expected_output(full: np.ndarray) -> np.ndarray:
    return _apply_seed_mask(full, _seed_mask(full))


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(GRID_CHOICES), random.choice(GRID_CHOICES))
    seed_mask = _random_seed_mask()
    full = _build_filled_grid(shape, seed_mask)
    holes = _sample_hole_mask(shape)
    if holes is None:
        return None

    inp = full.copy()
    inp[holes] = BG
    expected = _expected_output(full)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
