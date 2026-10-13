from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRID_WIDTH = 6
MIN_ROWS = 12
MAX_ROWS = 20
MIN_STRIPES = 3
MAX_STRIPES = 8
MIN_DISTINCT_COLORS = 3
MAX_DISTINCT_COLORS = 6
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800

_SEEN_OUTPUTS: set[bytes] = set()
_GENERATION_CALLS = 0


def _rows_are_uniform(grid: np.ndarray) -> bool:
    return bool(np.all(grid == grid[:, :1]))


def _compress_horizontal_stripes(grid: np.ndarray) -> list[int]:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0])]
    compressed: list[int] = []
    for color in row_colors:
        if not compressed or color != compressed[-1]:
            compressed.append(color)
    return compressed


def _stripe_thicknesses(grid: np.ndarray) -> list[int]:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0])]
    thicknesses: list[int] = []
    prev = None
    for color in row_colors:
        if color != prev:
            thicknesses.append(1)
            prev = color
        else:
            thicknesses[-1] += 1
    return thicknesses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress full-width horizontal color bands and encode the compressed color order on a diagonal.

    Input structure:
    - The grid is fully covered by horizontal full-width color bands with no exposed background.
    - Each row is uniform, adjacent bands have different colors, and band thickness may vary from 1 to 4 rows.
    - A color may reappear later after other bands, so the ordered compressed stripe sequence matters.

    Rule:
    - Read the band colors from top to bottom and collapse consecutive duplicate rows into one color per band.
    - Let N be the length of that compressed color sequence.
    - Return an N x N black grid whose main diagonal stores those N colors in the same top-to-bottom order.

    Key insight:
    - Band thickness is irrelevant after compression, but compression produces both pieces of needed information:
      the ordered color sequence and the diagonal grid size.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    if g.size == 0:
        raise ValueError("grid must be non-empty")
    if not _rows_are_uniform(g):
        raise ValueError("input is not composed of full-width horizontal stripes")

    sequence = _compress_horizontal_stripes(g)
    n = len(sequence)
    out = np.full((n, n), BLACK, dtype=int)
    for i, color in enumerate(sequence):
        out[i, i] = color
    return out


def _choose_mode() -> str:
    global _GENERATION_CALLS
    mode = "unique" if _GENERATION_CALLS % 4 == 1 else "repeat"
    _GENERATION_CALLS += 1
    return mode


def _sample_stripe_count(rng: random.Random, mode: str) -> int:
    if mode == "unique":
        return rng.randint(3, 6)
    return rng.randint(4, 8)


def _sample_sequence(rng: random.Random, n: int, mode: str) -> list[int]:
    if mode == "unique":
        return rng.sample(PALETTE, n)

    distinct = rng.randint(MIN_DISTINCT_COLORS, min(MAX_DISTINCT_COLORS, n - 1))
    colors = rng.sample(PALETTE, distinct)
    for _ in range(300):
        seq = [rng.choice(colors)]
        for _ in range(n - 1):
            choices = [color for color in colors if color != seq[-1]]
            seq.append(rng.choice(choices))
        if len(set(seq)) != distinct:
            continue
        if not any(seq.count(color) >= 2 for color in colors):
            continue
        return seq
    raise RuntimeError("failed to sample a repeated compressed sequence")


def _sample_thicknesses(rng: random.Random, n: int) -> list[int] | None:
    total_min = max(MIN_ROWS, n)
    total_max = min(MAX_ROWS, 4 * n)
    if total_min > total_max:
        return None

    totals = list(range(total_min, total_max + 1))
    rng.shuffle(totals)
    for total in totals:
        thicknesses: list[int] = []
        remaining = total
        for idx in range(n):
            slots_left = n - idx - 1
            low = max(1, remaining - 4 * slots_left)
            high = min(4, remaining - slots_left)
            if low > high:
                break
            thicknesses.append(rng.randint(low, high))
            remaining -= thicknesses[-1]
        if len(thicknesses) != n or remaining != 0:
            continue
        if max(thicknesses) == min(thicknesses):
            continue
        if max(thicknesses) <= 1:
            continue
        return thicknesses
    return None


def _render_grid(sequence: list[int], thicknesses: list[int]) -> np.ndarray:
    rows = sum(thicknesses)
    grid = np.empty((rows, GRID_WIDTH), dtype=int)
    r0 = 0
    for color, thickness in zip(sequence, thicknesses):
        grid[r0 : r0 + thickness, :] = color
        r0 += thickness
    return grid


def _has_nonconsecutive_repeat(sequence: list[int]) -> bool:
    first_seen: dict[int, int] = {}
    for idx, color in enumerate(sequence):
        if color in first_seen and idx - first_seen[color] > 1:
            return True
        first_seen.setdefault(color, idx)
    return False


def _validate_generated_instance(grid: np.ndarray, preferred_mode: str) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if grid.shape[1] != GRID_WIDTH:
        return False
    if not (MIN_ROWS <= grid.shape[0] <= MAX_ROWS):
        return False
    if not _rows_are_uniform(grid):
        return False

    sequence = _compress_horizontal_stripes(grid)
    thicknesses = _stripe_thicknesses(grid)
    if not (MIN_STRIPES <= len(sequence) <= MAX_STRIPES):
        return False
    if not all(1 <= t <= 4 for t in thicknesses):
        return False
    if len(set(sequence)) < MIN_DISTINCT_COLORS or len(set(sequence)) > MAX_DISTINCT_COLORS:
        return False
    if any(color == BLACK for color in sequence):
        return False
    if any(a == b for a, b in zip(sequence, sequence[1:])):
        return False
    if max(thicknesses) == min(thicknesses):
        return False

    if preferred_mode == "repeat":
        if len(set(sequence)) == len(sequence):
            return False
        if not _has_nonconsecutive_repeat(sequence):
            return False
    if preferred_mode == "unique" and len(set(sequence)) != len(sequence):
        return False

    out = main(grid)
    n = len(sequence)
    if out.shape != (n, n):
        return False
    if not np.array_equal(np.diag(out), np.array(sequence, dtype=int)):
        return False
    off_diag = out.copy()
    np.fill_diagonal(off_diag, BLACK)
    if np.any(off_diag != BLACK):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    mode = _choose_mode()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = _sample_stripe_count(rng, mode)
        sequence = _sample_sequence(rng, n, mode)
        thicknesses = _sample_thicknesses(rng, n)
        if thicknesses is None:
            continue
        grid = _render_grid(sequence, thicknesses)
        if not _validate_generated_instance(grid, mode):
            continue

        out = main(grid)
        out_sig = out.tobytes() + str(out.shape).encode()
        if out_sig in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
