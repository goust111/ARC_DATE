from __future__ import annotations

import math
from typing import Iterable

import numpy as np


SEPARATOR = 0
BLUE = 1
RED = 2
PANEL_BG = 5
OUTPUT_ACTIVE = 2
PANEL_SIZE = 11
GRID_SHAPE = (PANEL_SIZE, PANEL_SIZE * 2 + 1)
PERIOD_VALUES = (2, 3, 4)
MIN_VISIBLE_RATIO = 0.7
MIN_HIDDEN_RATIO = 0.1
MAX_HIDDEN_RATIO = 0.3
MAX_ATTEMPTS = 500


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(1, w - 1)
        if np.all(grid[:, c] == SEPARATOR) and c == (w - 1) // 2 and c == w - c - 1
    ]
    assert len(candidates) == 1, "expected one centered separator column"
    return int(candidates[0])


def _foreground_mask(panel: np.ndarray) -> np.ndarray:
    return panel != PANEL_BG


def _residue_positions(shape: tuple[int, int], pr: int, pc: int, rr: int, cc: int) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(rr, h, pr) for c in range(cc, w, pc)]


def _period_support(mask: np.ndarray, pr: int, pc: int) -> frozenset[tuple[int, int]] | None:
    residues: set[tuple[int, int]] = set()
    for rr in range(pr):
        for cc in range(pc):
            cls = mask[rr::pr, cc::pc]
            total = int(cls.size)
            active = int(np.count_nonzero(cls))
            if active == 0:
                continue
            if active < math.ceil(MIN_VISIBLE_RATIO * total):
                return None
            residues.add((rr, cc))
    if not residues:
        return None
    return frozenset(residues)


def _infer_period(mask: np.ndarray) -> tuple[int, int, frozenset[tuple[int, int]]]:
    h, w = mask.shape
    candidates: list[tuple[int, int, int, frozenset[tuple[int, int]]]] = []
    for pr in PERIOD_VALUES:
        if pr > h:
            continue
        for pc in PERIOD_VALUES:
            if pc > w:
                continue
            residues = _period_support(mask, pr, pc)
            if residues is not None:
                candidates.append((pr * pc, pr, pc, residues))

    assert candidates, "no admissible period found"
    best_area = min(area for area, _pr, _pc, _res in candidates)
    best = [(pr, pc, res) for area, pr, pc, res in candidates if area == best_area]
    assert len(best) == 1, "expected a unique smallest admissible period"
    pr, pc, residues = best[0]
    return int(pr), int(pc), residues


def _expand_residues(
    shape: tuple[int, int], pr: int, pc: int, residues: Iterable[tuple[int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    residue_set = set((int(rr), int(cc)) for rr, cc in residues)
    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r % pr, c % pc) in residue_set:
                out[r, c] = True
    return out


def _is_exact_periodic(arr: np.ndarray, pr: int, pc: int) -> bool:
    for rr in range(pr):
        for cc in range(pc):
            cls = arr[rr::pr, cc::pc]
            if not np.all(cls == cls[0, 0]):
                return False
    return True


def _unique_exact_period(mask: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    candidates: list[tuple[int, int, int]] = []
    for pr in PERIOD_VALUES:
        if pr > h:
            continue
        for pc in PERIOD_VALUES:
            if pc > w:
                continue
            if _is_exact_periodic(mask, pr, pc):
                candidates.append((pr * pc, pr, pc))
    if not candidates:
        return None
    best_area = min(area for area, _pr, _pc in candidates)
    best = [(pr, pc) for area, pr, pc in candidates if area == best_area]
    if len(best) != 1:
        return None
    return best[0]


def _has_small_panel_period(panel: np.ndarray) -> bool:
    fg = _foreground_mask(panel)
    for pr in PERIOD_VALUES:
        for pc in PERIOD_VALUES:
            if _is_exact_periodic(panel, pr, pc) or _is_exact_periodic(fg, pr, pc):
                return True
    return False


def _xor_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.logical_xor(_foreground_mask(left), _foreground_mask(right))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two side panels by foreground XOR, infer the hidden smallest periodic residue pattern, and complete it.

    Input structure:
    - The grid has a centered black separator column, with one square panel on each side.
    - Each panel uses grey as background and blue/red as scattered foreground pixels.
    - A cell is relevant only through its foreground-vs-background status; the single panels themselves are not periodic.

    Rule:
    - Ignore the separator and align the two side panels cell by cell.
    - Mark a cell active when exactly one panel shows a foreground pixel there and the other shows grey background.
    - These active cells are incomplete samples of a rectangular period with height and width between 2 and 4; choose the unique smallest admissible period whose nonempty residue classes keep at least 70% visible support.
    - Activate every cell whose period residue class has any visible support, then output that repaired mask as red on black.

    Key insight:
    - The periodic structure is not visible in either panel directly; it emerges only after XORing foreground presence, and even that mask still needs period-based repair to restore hidden cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "panels must align"

    mask = _xor_mask(left, right)
    pr, pc, residues = _infer_period(mask)
    repaired = _expand_residues(mask.shape, pr, pc, residues)

    out = np.zeros(mask.shape, dtype=int)
    out[repaired] = OUTPUT_ACTIVE
    return out


def _sample_periodic_mask(
    rng: np.random.Generator,
) -> tuple[int, int, frozenset[tuple[int, int]], np.ndarray]:
    for _ in range(MAX_ATTEMPTS):
        pr = int(rng.choice(PERIOD_VALUES))
        pc = int(rng.choice(PERIOD_VALUES))
        residues = [(rr, cc) for rr in range(pr) for cc in range(pc)]

        min_count = max(2, math.ceil(0.25 * len(residues)))
        max_count = min(len(residues) - 1, max(min_count, math.floor(0.6 * len(residues))))
        if min_count > max_count:
            continue
        chosen = frozenset(
            residues[idx]
            for idx in rng.choice(len(residues), size=int(rng.integers(min_count, max_count + 1)), replace=False)
        )
        full = _expand_residues((PANEL_SIZE, PANEL_SIZE), pr, pc, chosen)
        active_count = int(np.count_nonzero(full))
        if not (18 <= active_count <= 70):
            continue
        if _unique_exact_period(full) != (pr, pc):
            continue
        return pr, pc, chosen, full
    raise RuntimeError("failed to sample a nondegenerate periodic mask")


def _sample_visible_mask(
    rng: np.random.Generator,
    pr: int,
    pc: int,
    residues: frozenset[tuple[int, int]],
    full: np.ndarray,
) -> np.ndarray:
    total_active = int(np.count_nonzero(full))
    by_residue = {
        residue: _residue_positions(full.shape, pr, pc, residue[0], residue[1]) for residue in residues
    }
    for _ in range(MAX_ATTEMPTS):
        hidden = np.zeros_like(full, dtype=bool)
        for residue, positions in by_residue.items():
            del residue
            n = len(positions)
            max_hide = min(n - 1, int(math.floor(MAX_HIDDEN_RATIO * n)))
            hide_count = int(rng.integers(0, max_hide + 1))
            if hide_count == 0:
                continue
            pick = rng.choice(n, size=hide_count, replace=False)
            for idx in np.atleast_1d(pick):
                r, c = positions[int(idx)]
                hidden[r, c] = True

        hidden_count = int(np.count_nonzero(hidden))
        hidden_ratio = hidden_count / total_active
        if hidden_count == 0 or not (MIN_HIDDEN_RATIO <= hidden_ratio <= MAX_HIDDEN_RATIO):
            continue

        visible = full & ~hidden
        try:
            got_pr, got_pc, got_residues = _infer_period(visible)
        except AssertionError:
            continue
        if (got_pr, got_pc) != (pr, pc) or got_residues != residues:
            continue
        if not np.array_equal(_expand_residues(full.shape, got_pr, got_pc, got_residues), full):
            continue
        return visible
    raise RuntimeError("failed to hide a valid subset of the periodic mask")


def _build_panels(
    rng: np.random.Generator,
    full: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.full((PANEL_SIZE, PANEL_SIZE), PANEL_BG, dtype=int)
    right = np.full((PANEL_SIZE, PANEL_SIZE), PANEL_BG, dtype=int)
    hidden = full & ~visible

    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            if visible[r, c]:
                fg = int(rng.choice((BLUE, RED)))
                if int(rng.integers(0, 2)) == 0:
                    left[r, c] = fg
                else:
                    right[r, c] = fg
            elif hidden[r, c]:
                if int(rng.integers(0, 2)) == 0:
                    left[r, c] = PANEL_BG
                    right[r, c] = PANEL_BG
                else:
                    fg = int(rng.choice((BLUE, RED)))
                    left[r, c] = fg
                    right[r, c] = fg
            else:
                value = int(rng.choice((PANEL_BG, PANEL_BG, PANEL_BG, BLUE, RED)))
                left[r, c] = value
                right[r, c] = value
    return left, right


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    grid = np.full(GRID_SHAPE, PANEL_BG, dtype=int)
    grid[:, :PANEL_SIZE] = left
    grid[:, PANEL_SIZE] = SEPARATOR
    grid[:, PANEL_SIZE + 1 :] = right
    return grid


def _validate_instance(
    grid: np.ndarray,
    pr: int,
    pc: int,
    residues: frozenset[tuple[int, int]],
    full: np.ndarray,
    visible: np.ndarray,
) -> bool:
    if grid.shape != GRID_SHAPE:
        return False
    if not np.all(grid[:, PANEL_SIZE] == SEPARATOR):
        return False

    left = grid[:, :PANEL_SIZE]
    right = grid[:, PANEL_SIZE + 1 :]
    if _has_small_panel_period(left) or _has_small_panel_period(right):
        return False

    xor = _xor_mask(left, right)
    if not np.array_equal(xor, visible):
        return False

    if _unique_exact_period(full) != (pr, pc):
        return False

    try:
        got_pr, got_pc, got_residues = _infer_period(xor)
    except AssertionError:
        return False
    if (got_pr, got_pc) != (pr, pc) or got_residues != residues:
        return False

    repaired = _expand_residues(full.shape, got_pr, got_pc, got_residues)
    if not np.array_equal(repaired, full):
        return False

    hidden_count = int(np.count_nonzero(full & ~visible))
    hidden_ratio = hidden_count / int(np.count_nonzero(full))
    if not (MIN_HIDDEN_RATIO <= hidden_ratio <= MAX_HIDDEN_RATIO):
        return False

    if np.count_nonzero(_foreground_mask(left) ^ full) < 8:
        return False
    if np.count_nonzero(_foreground_mask(right) ^ full) < 8:
        return False

    colors = set(map(int, np.unique(grid)))
    if not {BLUE, RED, PANEL_BG, SEPARATOR}.issubset(colors):
        return False

    expected = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    expected[full] = OUTPUT_ACTIVE
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        pr, pc, residues, full = _sample_periodic_mask(rng)
        visible = _sample_visible_mask(rng, pr, pc, residues, full)
        left, right = _build_panels(rng, full, visible)
        grid = _assemble_input(left, right)
        if _validate_instance(grid, pr, pc, residues, full, visible):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
