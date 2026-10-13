from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


SEPARATOR = 5
BLACK = 0
FILL_COLORS = tuple(color for color in range(1, 10) if color != SEPARATOR)
ROW_RUNS = ((0, 1), (3, 4), (6, 7))
COL_RUNS = ((0, 1), (3, 4), (6, 7))
GRID_SHAPE = (8, 8)
MAX_GENERATION_ATTEMPTS = 400
RECENT_OUTPUT_LIMIT = 24

_RNG = np.random.default_rng()
_RECENT_OUTPUTS: list[str] = []


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty array"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority"
    return int(ranked[0][0])


def _separator_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full_sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    full_sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    row_runs = _runs([r for r in range(grid.shape[0]) if r not in set(full_sep_rows)])
    col_runs = _runs([c for c in range(grid.shape[1]) if c not in set(full_sep_cols)])
    return row_runs, col_runs


def _mirror_spread(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _pixel_vote_in_block(block: np.ndarray, dr: int, dc: int) -> int:
    bh, bw = map(int, block.shape)
    mirrored = _mirror_spread(block)
    samples = np.array(
        [
            int(mirrored[dr, dc]),
            int(mirrored[dr, bw + dc]),
            int(mirrored[bh + dr, dc]),
            int(mirrored[bh + dr, bw + dc]),
        ],
        dtype=int,
    )
    return _unique_mode(samples)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-expand a separator lattice and compress each cell back by four-way majority vote.

    Input structure:
    - The grid contains complete grey separator rows and columns with no outer border.
    - The separators divide the non-grey area into a 3x3 lattice of equal-sized colored blocks.
    - Each block is mostly one non-black, non-grey color, with at most one off-color noise pixel.

    Rule:
    - Reflect the whole input rightward, then reflect that result downward to form a four-quadrant mirror spread.
    - For every non-separator input pixel, collect the four pixels at the same local position across the mirrored quadrants and keep their unique majority color.
    - Within each original lattice block, fill the whole block with the unique majority color obtained after that mirror vote, while preserving the grey separators.

    Key insight:
    - The mirror tiling creates four aligned copies of each noisy block, so an asymmetric one-pixel defect can be outvoted and removed before compressing back to the original lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > 0 and w > 0, "expected a non-empty grid"
    assert max(h, w) <= 30, "ARC grids must stay within 30x30"

    row_runs, col_runs = _separator_runs(g)
    assert row_runs and col_runs, "expected separator-defined row and column bands"
    assert len(row_runs) == 3 and len(col_runs) == 3, "expected a 3x3 lattice"
    assert all((r1 - r0 + 1) == 2 for r0, r1 in row_runs), "expected 2-pixel-tall row bands"
    assert all((c1 - c0 + 1) == 2 for c0, c1 in col_runs), "expected 2-pixel-wide column bands"

    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == BLACK), "blocks must avoid black"
            assert not np.any(block == SEPARATOR), "blocks must avoid the separator color"
            counts = Counter(map(int, block.reshape(-1)))
            ranked = counts.most_common()
            assert len(ranked) == 1 or ranked[0][1] >= 3, "each block needs a visible main color"

    voted = g.copy()
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    voted[r, c] = _pixel_vote_in_block(block, r - r0, c - c0)

    out = g.copy()
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            resolved_color = _unique_mode(voted[r0 : r1 + 1, c0 : c1 + 1])
            out[r0 : r1 + 1, c0 : c1 + 1] = resolved_color

    return out


def _expected_output(main_colors: np.ndarray) -> np.ndarray:
    out = np.full(GRID_SHAPE, SEPARATOR, dtype=int)
    for br, (r0, r1) in enumerate(ROW_RUNS):
        for bc, (c0, c1) in enumerate(COL_RUNS):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(main_colors[br, bc])
    return out


def _sample_main_colors() -> np.ndarray:
    palette_size = int(_RNG.integers(3, 5))
    palette = [int(color) for color in _RNG.choice(np.array(FILL_COLORS, dtype=int), size=palette_size, replace=False)]

    cells = palette[:]
    remaining = 9 - len(cells)
    extra = [int(color) for color in _RNG.choice(np.array(palette, dtype=int), size=remaining, replace=True)]
    cells.extend(extra)
    _RNG.shuffle(cells)

    colors = np.array(cells, dtype=int).reshape(3, 3)
    if np.all(colors == colors[0, 0]):
        swap_idx = int(_RNG.integers(1, 9))
        colors.reshape(-1)[swap_idx] = int(palette[-1])
    return colors


def _sample_noise_plan(main_colors: np.ndarray) -> tuple[list[tuple[int, int, int, int, int]], set[tuple[int, int]]]:
    block_order = [(br, bc) for br in range(3) for bc in range(3)]
    _RNG.shuffle(block_order)

    noisy_blocks = int(_RNG.integers(2, 7))
    chosen_blocks = block_order[:noisy_blocks]
    seen_noisy_positions: set[tuple[int, int]] = set()
    noise_plan: list[tuple[int, int, int, int, int]] = []

    for br, bc in chosen_blocks:
        main_color = int(main_colors[br, bc])
        candidates = [color for color in np.unique(main_colors) if int(color) != main_color]
        if not candidates:
            candidates = [color for color in FILL_COLORS if color != main_color]
        noise_color = int(candidates[int(_RNG.integers(len(candidates)))])
        nr = int(_RNG.integers(0, 2))
        nc = int(_RNG.integers(0, 2))
        seen_noisy_positions.add((nr, nc))
        noise_plan.append((br, bc, nr, nc, noise_color))

    return noise_plan, seen_noisy_positions


def _render_input(main_colors: np.ndarray, noise_plan: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    grid = _expected_output(main_colors)
    for br, bc, nr, nc, noise_color in noise_plan:
        r0, _r1 = ROW_RUNS[br]
        c0, _c1 = COL_RUNS[bc]
        grid[r0 + nr, c0 + nc] = int(noise_color)
    return grid


def _naive_identity_fails(inp: np.ndarray, expected: np.ndarray) -> bool:
    return not np.array_equal(inp, expected)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.shape != GRID_SHAPE or expected.shape != GRID_SHAPE:
        return False
    row_runs, col_runs = _separator_runs(inp)
    if row_runs != list(ROW_RUNS) or col_runs != list(COL_RUNS):
        return False

    block_majorities: list[int] = []
    noisy_blocks = 0
    noisy_positions: set[tuple[int, int]] = set()
    for br, (r0, r1) in enumerate(ROW_RUNS):
        for bc, (c0, c1) in enumerate(COL_RUNS):
            block = inp[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(block == BLACK) or np.any(block == SEPARATOR):
                return False
            counts = Counter(map(int, block.reshape(-1)))
            ranked = counts.most_common()
            if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
                return False
            if ranked[0][1] < 3:
                return False
            block_majorities.append(int(ranked[0][0]))
            if ranked[0][1] == 3:
                noisy_blocks += 1
                for rr in range(2):
                    for cc in range(2):
                        if int(block[rr, cc]) != int(ranked[0][0]):
                            noisy_positions.add((rr, cc))

    if not 3 <= len(set(block_majorities)) <= 4:
        return False
    if noisy_blocks < 2:
        return False
    if len(noisy_positions) < 2:
        return False
    if not _naive_identity_fails(inp, expected):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    changed_by_vote = False
    for r0, r1 in ROW_RUNS:
        for c0, c1 in COL_RUNS:
            block = inp[r0 : r1 + 1, c0 : c1 + 1]
            mirrored = _mirror_spread(block)
            if mirrored.shape != (4, 4):
                return False
            for rr in range(block.shape[0]):
                for cc in range(block.shape[1]):
                    voted = _pixel_vote_in_block(block, rr, cc)
                    if voted != int(block[rr, cc]):
                        changed_by_vote = True
    return changed_by_vote


def _output_signature(out: np.ndarray) -> str:
    return hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()


def _remember_output(sig: str) -> None:
    _RECENT_OUTPUTS.append(sig)
    if len(_RECENT_OUTPUTS) > RECENT_OUTPUT_LIMIT:
        del _RECENT_OUTPUTS[0]


def _gen_attempt() -> np.ndarray | None:
    main_colors = _sample_main_colors()
    expected = _expected_output(main_colors)
    out_sig = _output_signature(expected)
    if out_sig in _RECENT_OUTPUTS:
        return None

    noise_plan, noisy_positions = _sample_noise_plan(main_colors)
    if len(noisy_positions) < 2:
        return None

    inp = _render_input(main_colors, noise_plan)
    if not _semantic_checks(inp, expected):
        return None

    _remember_output(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-majority input")
