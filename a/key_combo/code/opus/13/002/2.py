from __future__ import annotations

from functools import lru_cache

import numpy as np


BG = 0
FG = 1
MASK = 5
GRID_SIZES = (11, 13)
MAX_GENERATION_ATTEMPTS = 400


def _symmetry_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = int(n) - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, r),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


@lru_cache(maxsize=None)
def _all_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(int(n)):
        for c in range(int(n)):
            orbit = _symmetry_orbit(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return tuple(orbits)


def _count_bands(separator_lines: np.ndarray) -> int:
    bands = 0
    in_band = False
    for is_separator in np.asarray(separator_lines, dtype=bool):
        if is_separator:
            in_band = False
        elif not in_band:
            bands += 1
            in_band = True
    return bands


def _restore_masks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, w = g.shape
    assert n == w, "expected a square grid"
    out = g.copy()
    for orbit in _all_orbits(n):
        values = [int(g[r, c]) for r, c in orbit]
        if MASK not in values:
            continue
        visible = {value for value in values if value != MASK}
        assert len(visible) == 1, "each masked orbit must expose one visible color"
        fill = visible.pop()
        for r, c in orbit:
            if int(out[r, c]) == MASK:
                out[r, c] = fill
    return out


def _separator_lines(restored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_rows = np.all(restored == FG, axis=1)
    full_cols = np.all(restored == FG, axis=0)
    assert np.any(full_rows), "expected at least one full horizontal separator"
    assert np.any(full_cols), "expected at least one full vertical separator"
    return full_rows, full_cols


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the masked symmetric pattern, detect the full blue separator lines that emerge, and output a black grid sized by the resulting row and column region counts.

    Input structure:
    - The grid is a square black background with blue pattern pixels and gray masked cells.
    - The underlying blue pattern is invariant under all eight symmetries of the square, but some blue cells are replaced by gray.
    - After restoration, exactly the complete all-blue rows and columns act as separator lines; other blue pixels are only decorative.

    Rule:
    - For each gray cell, inspect its eight-way symmetry orbit and replace it with the unique visible non-gray color from that orbit.
    - In the restored grid, find every row filled entirely with blue and every column filled entirely with blue.
    - Count the contiguous runs of non-separator rows and non-separator columns, then return an all-black grid with those run counts as height and width.

    Key insight:
    - The separator lines are intentionally broken only by gray masks, so they become recognizable as complete full-width or full-height lines only after eight-way symmetry completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    restored = _restore_masks(g)
    full_rows, full_cols = _separator_lines(restored)
    out_h = _count_bands(full_rows)
    out_w = _count_bands(full_cols)
    assert out_h > 0 and out_w > 0
    return np.full((out_h, out_w), BG, dtype=int)


def _expected_output_shape(sep_rows: list[int], sep_cols: list[int]) -> tuple[int, int]:
    return len(sep_rows) + 1, len(sep_cols) + 1


def _choose_separator_layout(rng: np.random.Generator, n: int) -> tuple[list[int], list[int], str]:
    mid = n // 2
    if float(rng.random()) < 0.5:
        return [mid], [mid], "center"

    offsets = list(range(2, mid))
    offset = int(rng.choice(offsets))
    rows = [mid - offset, mid + offset]
    cols = [mid - offset, mid + offset]
    return rows, cols, "double"


def _render_completed_grid(
    rng: np.random.Generator, n: int, sep_rows: list[int], sep_cols: list[int]
) -> np.ndarray | None:
    grid = np.full((n, n), BG, dtype=int)
    grid[sep_rows, :] = FG
    grid[:, sep_cols] = FG

    candidates = [
        orbit
        for orbit in _all_orbits(n)
        if all(r not in sep_rows and c not in sep_cols for r, c in orbit)
    ]
    if not candidates:
        return None

    if n == 11:
        deco_count = int(rng.integers(1, 4))
    else:
        deco_count = int(rng.integers(2, 5))
    deco_count = min(deco_count, len(candidates))

    order = rng.permutation(len(candidates))
    for idx in order[:deco_count]:
        orbit = candidates[int(idx)]
        for r, c in orbit:
            grid[r, c] = FG

    full_rows = np.flatnonzero(np.all(grid == FG, axis=1)).tolist()
    full_cols = np.flatnonzero(np.all(grid == FG, axis=0)).tolist()
    if full_rows != sep_rows or full_cols != sep_cols:
        return None

    non_line_blue = int(np.count_nonzero(grid == FG) - len(sep_rows) * n - len(sep_cols) * n + len(sep_rows) * len(sep_cols))
    if non_line_blue <= 0:
        return None
    return grid


def _apply_required_masks(
    rng: np.random.Generator, grid: np.ndarray, sep_rows: list[int], sep_cols: list[int], mode: str
) -> set[tuple[int, int]]:
    n = grid.shape[0]
    masked: set[tuple[int, int]] = set()
    if mode == "center":
        mid = n // 2
        options = [c for c in range(1, n - 1) if c != mid]
        pivot = int(rng.choice(options))
        masked.update({(mid, pivot), (pivot, mid)})
    else:
        r0, r1 = sep_rows
        if float(rng.random()) < 0.5:
            masked.update({(r0, r0), (r1, r1)})
        else:
            masked.update({(r0, r1), (r1, r0)})

    for r, c in masked:
        assert int(grid[r, c]) == FG
        grid[r, c] = MASK
    return masked


def _add_extra_masks(rng: np.random.Generator, completed: np.ndarray, inp: np.ndarray, masked: set[tuple[int, int]]) -> None:
    extra_steps = int(rng.integers(0, 3))
    blue_orbits = [
        orbit for orbit in _all_orbits(inp.shape[0]) if all(int(completed[r, c]) == FG for r, c in orbit) and len(orbit) >= 2
    ]
    if not blue_orbits:
        return

    order = rng.permutation(len(blue_orbits))
    for idx in order[:extra_steps]:
        orbit = blue_orbits[int(idx)]
        already_masked = [pos for pos in orbit if pos in masked]
        max_new = len(orbit) - 1 - len(already_masked)
        if max_new <= 0:
            continue
        available = [pos for pos in orbit if pos not in masked]
        take = int(rng.integers(1, min(2, max_new) + 1))
        chosen_ids = rng.choice(len(available), size=take, replace=False)
        for local_idx in np.atleast_1d(chosen_ids):
            r, c = available[int(local_idx)]
            inp[r, c] = MASK
            masked.add((r, c))


def _validate_input(
    inp: np.ndarray, completed: np.ndarray, sep_rows: list[int], sep_cols: list[int]
) -> bool:
    try:
        restored = _restore_masks(inp)
    except Exception:
        return False
    if not np.array_equal(restored, completed):
        return False

    if np.any(np.all(inp == FG, axis=1)) or np.any(np.all(inp == FG, axis=0)):
        return False

    full_rows, full_cols = _separator_lines(restored)
    if np.flatnonzero(full_rows).tolist() != sep_rows:
        return False
    if np.flatnonzero(full_cols).tolist() != sep_cols:
        return False

    expected = np.full(_expected_output_shape(sep_rows, sep_cols), BG, dtype=int)
    try:
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice(GRID_SIZES))
    sep_rows, sep_cols, mode = _choose_separator_layout(rng, n)
    completed = _render_completed_grid(rng, n, sep_rows, sep_cols)
    if completed is None:
        return None

    inp = completed.copy()
    masked = _apply_required_masks(rng, inp, sep_rows, sep_cols, mode)
    _add_extra_masks(rng, completed, inp, masked)

    if not _validate_input(inp, completed, sep_rows, sep_cols):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
