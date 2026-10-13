from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
PERIOD_TO_SIZES = {
    2: (8, 10),
    3: (9,),
    4: (8,),
}
VISIBLE_DENSITY_RANGE = (0.35, 0.70)
MAX_GENERATION_ATTEMPTS = 500


def _divisors(n: int, *, minimum: int = 2, maximum: int | None = None) -> list[int]:
    upper = n if maximum is None else min(n, maximum)
    return [d for d in range(minimum, upper + 1) if n % d == 0]


def _find_separator(grid: np.ndarray) -> int:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(sep_cols) == 1, "expected exactly one grey separator column"
    return int(sep_cols[0])


def _candidate_tile(
    samples: np.ndarray, period: int, *, require_all_phases: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    tile = np.zeros((period, period), dtype=int)
    seen = np.zeros((period, period), dtype=bool)

    for r, c in np.argwhere(samples != BG):
        pr, pc = int(r) % period, int(c) % period
        color = int(samples[r, c])
        if seen[pr, pc]:
            if int(tile[pr, pc]) != color:
                return None
        else:
            tile[pr, pc] = color
            seen[pr, pc] = True

    if require_all_phases and not np.all(seen):
        return None
    return tile, seen


def _render_periodic(tile: np.ndarray, size: int) -> np.ndarray:
    period = int(tile.shape[0])
    rows, cols = np.indices((size, size))
    return tile[rows % period, cols % period]


def _merge_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    overlap = (left != BG) & (right != BG)
    assert not np.any(overlap), "panel samples must be disjoint"
    merged = left.copy()
    mask = (merged == BG) & (right != BG)
    merged[mask] = right[mask]
    return merged


def _smallest_complete_period(samples: np.ndarray) -> tuple[int, np.ndarray]:
    size = int(samples.shape[0])
    for period in _divisors(size):
        candidate = _candidate_tile(samples, period, require_all_phases=True)
        if candidate is not None:
            tile, _seen = candidate
            return period, tile
    raise AssertionError("no complete period can be inferred from the merged samples")


def _primitive_period(tile: np.ndarray) -> int:
    period = int(tile.shape[0])
    for smaller in _divisors(period, maximum=period - 1):
        reduced = tile[:smaller, :smaller]
        if np.array_equal(tile, _render_periodic(reduced, period)):
            return smaller
    return period


def _phase_positions(size: int, period: int, pr: int, pc: int) -> list[tuple[int, int]]:
    step = size // period
    return [
        (pr + kr * period, pc + kc * period)
        for kr in range(step)
        for kc in range(step)
    ]


def _choose_tile(rng: np.random.Generator, period: int) -> np.ndarray:
    phase_count = period * period
    color_count = int(rng.integers(2, min(4, phase_count) + 1))
    colors = rng.choice(PALETTE, size=color_count, replace=False)

    for _ in range(200):
        tile = rng.choice(colors, size=(period, period), replace=True)
        for color in colors:
            if not np.any(tile == int(color)):
                rr, cc = divmod(int(rng.integers(phase_count)), period)
                tile[rr, cc] = int(color)
        if _primitive_period(tile) == period:
            return tile.astype(int)
    raise AssertionError("failed to build a primitive periodic tile")


def _choose_phase_owner(
    rng: np.random.Generator, period: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    phases = [(r, c) for r in range(period) for c in range(period)]
    if period == 4:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for phase in phases:
            groups.setdefault((phase[0] % 2, phase[1] % 2), []).append(phase)
        group_keys = list(groups)
        left_group_count = int(rng.integers(1, len(group_keys)))
        chosen_ids = set(
            int(idx)
            for idx in rng.choice(len(group_keys), size=left_group_count, replace=False)
        )
        left = {
            phase
            for idx, key in enumerate(group_keys)
            if idx in chosen_ids
            for phase in groups[key]
        }
    else:
        left_size = int(rng.integers(1, len(phases)))
        selected = rng.choice(len(phases), size=left_size, replace=False)
        left = {phases[int(idx)] for idx in selected}

    right = set(phases) - left
    assert left and right
    return left, right


def _visible_count_bounds(occurrences: int) -> tuple[int, int]:
    low = max(1, int(np.floor(occurrences * VISIBLE_DENSITY_RANGE[0])))
    high = min(occurrences - 1, int(np.ceil(occurrences * VISIBLE_DENSITY_RANGE[1])))
    if low > high:
        low = high = max(1, occurrences - 1)
    return low, high


def _sample_panels(
    rng: np.random.Generator,
    full: np.ndarray,
    period: int,
    left_phases: set[tuple[int, int]],
    right_phases: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    size = int(full.shape[0])
    left = np.zeros_like(full)
    right = np.zeros_like(full)

    for pr in range(period):
        for pc in range(period):
            positions = _phase_positions(size, period, pr, pc)
            owner = left if (pr, pc) in left_phases else right
            low, high = _visible_count_bounds(len(positions))
            keep_count = int(rng.integers(low, high + 1))
            keep_ids = rng.choice(len(positions), size=keep_count, replace=False)
            color = int(full[pr, pc])
            for idx in keep_ids:
                r, c = positions[int(idx)]
                owner[r, c] = color

    return left, right


def _panel_candidate_periods(panel: np.ndarray) -> list[int]:
    size = int(panel.shape[0])
    candidates: list[int] = []
    for period in _divisors(size):
        if _candidate_tile(panel, period, require_all_phases=True) is not None:
            candidates.append(period)
    return candidates


def _validate_instance(inp: np.ndarray, full: np.ndarray, actual_period: int) -> bool:
    if inp.ndim != 2 or full.ndim != 2:
        return False

    sep = _find_separator(inp)
    size = int(inp.shape[0])
    if sep != size or inp.shape[1] != 2 * size + 1:
        return False

    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    if np.any(left == SEP) or np.any(right == SEP):
        return False
    if np.any((left != BG) & (right != BG)):
        return False

    merged = _merge_panels(left, right)
    if not np.any(merged == BG):
        return False
    if np.any(full == BG) or np.any(full == SEP):
        return False

    try:
        inferred_period, tile = _smallest_complete_period(merged)
    except AssertionError:
        return False
    if inferred_period != actual_period:
        return False
    if not np.array_equal(_render_periodic(tile, size), full):
        return False

    if _panel_candidate_periods(left):
        return False
    if _panel_candidate_periods(right):
        return False

    visible_colors = sorted(set(int(v) for v in np.unique(full)))
    if not (2 <= len(visible_colors) <= 4):
        return False

    return np.array_equal(main(inp), full)


def main(grid: np.ndarray) -> np.ndarray:
    """Merge two complementary sample panels, infer the shared period tile, and fill the remaining gaps.

    Input structure:
    - The grid is split by one full grey separator column into left and right square panels of the same size.
    - Both panels sample the same hidden periodic color pattern using non-black, non-grey colors; unsampled cells are black.
    - The panels never show color at the same coordinate, and each panel alone omits some necessary phase information.

    Rule:
    - Find the grey separator and split the grid into the two aligned panels.
    - Take the cellwise union of the panels, preserving the visible colors.
    - Infer the smallest repeating period whose phase classes are fully represented and color-consistent in the merged samples.
    - Use the inferred period tile to fill every remaining black cell and return the completed single-panel pattern.

    Key insight:
    - The union is only an information-gathering step: neither panel alone determines the pattern, but together they reveal one sample for every phase of the hidden periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    sep = _find_separator(g)
    size = int(g.shape[0])
    assert sep == size and g.shape[1] == 2 * size + 1, "input must be N x (2N+1)"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape == (size, size), "panels must be square and aligned"
    assert not np.any(left == SEP) and not np.any(right == SEP), "grey is reserved for the separator"

    merged = _merge_panels(left, right)
    period, tile = _smallest_complete_period(merged)
    out = _render_periodic(tile, size)
    assert np.array_equal(out[merged != BG], merged[merged != BG]), "visible samples must match the inferred tile"
    assert period >= 2
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    period = random.choice(sorted(PERIOD_TO_SIZES))
    size = random.choice(PERIOD_TO_SIZES[period])

    tile = _choose_tile(rng, period)
    full = _render_periodic(tile, size)
    left_phases, right_phases = _choose_phase_owner(rng, period)
    left, right = _sample_panels(rng, full, period, left_phases, right_phases)

    if np.count_nonzero(left) < 3 or np.count_nonzero(right) < 3:
        return None

    inp = np.zeros((size, 2 * size + 1), dtype=int)
    inp[:, :size] = left
    inp[:, size] = SEP
    inp[:, size + 1 :] = right

    if not _validate_instance(inp, full, period):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel periodic-union puzzle")
