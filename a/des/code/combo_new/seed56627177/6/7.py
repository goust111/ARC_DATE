from __future__ import annotations

import random
from collections import Counter

import numpy as np


CELL_SIZE = 3
MASK_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

MACRO_SIZE_MIN = 4
MACRO_SIZE_MAX = 9
GEN_RETRIES = 300
BOARD_RETRIES = 120
COLOR_POOL = tuple(range(10))


def _expand_macro_board(board: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(board, CELL_SIZE, axis=0), CELL_SIZE, axis=1)


def _parse_macro_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % CELL_SIZE == 0, "expected a square grid tiled by 3x3 macro-cells"

    macro_n = h // CELL_SIZE
    macro = np.empty((macro_n, macro_n), dtype=int)
    uniform_colors: set[int] = set()

    for r in range(macro_n):
        for c in range(macro_n):
            block = g[r * CELL_SIZE : (r + 1) * CELL_SIZE, c * CELL_SIZE : (c + 1) * CELL_SIZE]
            vals = np.unique(block)
            if (r, c) == (0, 0):
                continue
            assert vals.size == 1, "all non-corner macro-cells must be solid"
            color = int(vals[0])
            macro[r, c] = color
            uniform_colors.add(color)

    assert len(uniform_colors) == 2, "expected exactly two regular macro-cell colors"

    corner = g[:CELL_SIZE, :CELL_SIZE]
    corner_vals = [int(v) for v in np.unique(corner)]
    mask_candidates = [v for v in corner_vals if v not in uniform_colors]
    assert len(mask_candidates) == 1, "corner mask color must be unique to the corner block"
    mask_color = int(mask_candidates[0])

    base_candidates = [v for v in corner_vals if v != mask_color]
    assert len(base_candidates) == 1, "corner block must expose a single base color"
    base_color = int(base_candidates[0])
    assert corner[1, 1] == base_color, "the corner center pixel must remain the base color"
    assert np.all((corner == base_color) | (corner == mask_color)), "corner block can only contain base and mask colors"

    active_dirs: list[tuple[int, int]] = []
    for dr, dc in MASK_OFFSETS:
        if int(corner[dr + 1, dc + 1]) == mask_color:
            active_dirs.append((dr, dc))
    assert active_dirs, "the direction mask must activate at least one neighbor"

    macro[0, 0] = base_color
    return macro, tuple(active_dirs)


def _vote_macro_board(
    board: np.ndarray, active_dirs: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = map(int, board.shape)
    out = board.copy()
    meta = {"ties": 0, "no_votes": 0, "changed": 0, "corner_changed": 0}

    for r in range(h):
        for c in range(w):
            tally: Counter[int] = Counter()
            for dr, dc in active_dirs:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    tally[int(board[rr, cc])] += 1

            if not tally:
                meta["no_votes"] += 1
                continue

            best = max(tally.values())
            winners = [color for color, count in tally.items() if count == best]
            if len(winners) != 1:
                meta["ties"] += 1
                continue

            chosen = int(winners[0])
            out[r, c] = chosen
            if chosen != int(board[r, c]):
                meta["changed"] += 1
                if (r, c) == (0, 0):
                    meta["corner_changed"] = 1

    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 3x3 macro-cell by majority vote from the directions encoded in the special corner cell.

    Input structure:
    - The canvas is a square grid whose side length is a multiple of 3, so it splits into equal 3×3 macro-cells.
    - Every macro-cell except the top-left one is a solid 3×3 block of one of exactly two data colors.
    - The top-left macro-cell uses one of those data colors as its base fill, but some of its eight outer pixels are overpainted with a third color to mark active neighbor directions.

    Rule:
    - Read the top-left 3×3 block as a direction mask: each outer pixel painted with the third color activates the corresponding neighbor offset, while the center pixel is ignored.
    - Treat each macro-cell as carrying one data color; for the masked corner cell, that color is its visible base fill rather than the overpainted mask pixels.
    - For every macro-cell, count only the in-bounds neighbors in the active directions and replace the cell with the unique majority color among those votes.
    - If the active vote ties, or if no active neighbor stays in bounds, keep the cell's original data color; then repaint the whole output as solid 3×3 macro-cells.

    Key insight:
    - The non-uniform corner block is a hidden stencil: its yellow-like overpaint does not participate as cell content, but instead selects which neighbor directions are allowed to vote everywhere in the macro-grid.
    """
    macro, active_dirs = _parse_macro_input(np.asarray(grid, dtype=int))
    voted, _ = _vote_macro_board(macro, active_dirs)
    return _expand_macro_board(voted)


def _sample_active_dirs(rng: random.Random, scenario: str) -> tuple[tuple[int, int], ...]:
    want_even = scenario in {"tie", "mixed"} or (scenario == "flip" and rng.random() < 0.5)
    style = rng.choice(("orthogonal", "diagonal", "mixed"))

    if style == "orthogonal":
        pool = list(ORTHOGONAL_OFFSETS)
        sizes = [k for k in range(1, len(pool) + 1) if (k % 2 == 0) == want_even]
        count = rng.choice(sizes)
        dirs = rng.sample(pool, count)
    elif style == "diagonal":
        pool = list(DIAGONAL_OFFSETS)
        sizes = [k for k in range(1, len(pool) + 1) if (k % 2 == 0) == want_even]
        count = rng.choice(sizes)
        dirs = rng.sample(pool, count)
    else:
        pool = list(MASK_OFFSETS)
        sizes = [k for k in range(3, 7) if (k % 2 == 0) == want_even]
        count = rng.choice(sizes)
        while True:
            dirs = rng.sample(pool, count)
            if any(d in ORTHOGONAL_OFFSETS for d in dirs) and any(d in DIAGONAL_OFFSETS for d in dirs):
                break

    return tuple(sorted(dirs))


def _apply_random_symmetry(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_board(n: int, colors: tuple[int, int], rng: random.Random, scenario: str) -> np.ndarray:
    if scenario in {"tie", "mixed"}:
        mode = rng.choice(("noise", "patches", "checker"))
    else:
        mode = rng.choice(("noise", "patches", "checker", "stripes_h", "stripes_v"))

    if mode == "noise":
        p = rng.uniform(0.38, 0.62)
        mask = np.array([[rng.random() < p for _ in range(n)] for _ in range(n)], dtype=bool)
    elif mode == "checker":
        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        phase = rng.randrange(2)
        mask = ((rows + cols + phase) % 2 == 0)
        flips = rng.randint(max(1, n // 2), max(1, n * n // 4))
        for _ in range(flips):
            rr = rng.randrange(n)
            cc = rng.randrange(n)
            mask[rr, cc] = ~mask[rr, cc]
    elif mode == "stripes_h":
        mask = np.zeros((n, n), dtype=bool)
        current = bool(rng.randrange(2))
        idx = 0
        while idx < n:
            span = rng.randint(1, min(3, n - idx))
            mask[idx : idx + span, :] = current
            if rng.random() < 0.85:
                current = not current
            idx += span
        for _ in range(rng.randint(0, n)):
            mask[rng.randrange(n), rng.randrange(n)] = bool(rng.randrange(2))
    elif mode == "stripes_v":
        mask = np.zeros((n, n), dtype=bool)
        current = bool(rng.randrange(2))
        idx = 0
        while idx < n:
            span = rng.randint(1, min(3, n - idx))
            mask[:, idx : idx + span] = current
            if rng.random() < 0.85:
                current = not current
            idx += span
        for _ in range(rng.randint(0, n)):
            mask[rng.randrange(n), rng.randrange(n)] = bool(rng.randrange(2))
    else:
        mask = np.full((n, n), bool(rng.randrange(2)))
        for _ in range(rng.randint(2, 5)):
            r0 = rng.randrange(n)
            c0 = rng.randrange(n)
            r1 = rng.randint(r0, n - 1)
            c1 = rng.randint(c0, n - 1)
            mask[r0 : r1 + 1, c0 : c1 + 1] = bool(rng.randrange(2))
        for _ in range(rng.randint(0, n)):
            mask[rng.randrange(n), rng.randrange(n)] = bool(rng.randrange(2))

    mask = _apply_random_symmetry(mask, rng)
    return np.where(mask, colors[0], colors[1]).astype(int)


def _board_is_clear(board: np.ndarray) -> bool:
    flat = board.reshape(-1)
    other_cells = flat[1:]
    values, counts = np.unique(other_cells, return_counts=True)
    if len(values) != 2:
        return False
    min_count = max(2, int(board.shape[0]) - 1)
    return int(counts.min()) >= min_count


def _render_input(
    board: np.ndarray, active_dirs: tuple[tuple[int, int], ...], mask_color: int
) -> np.ndarray:
    grid = _expand_macro_board(board)
    for dr, dc in active_dirs:
        grid[dr + 1, dc + 1] = int(mask_color)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(MACRO_SIZE_MIN, MACRO_SIZE_MAX)
    color_a, color_b, mask_color = rng.sample(COLOR_POOL, 3)
    scenario = rng.choice(("tie", "flip", "mixed"))
    active_dirs = _sample_active_dirs(rng, scenario)

    for _ in range(BOARD_RETRIES):
        board = _sample_board(n, (color_a, color_b), rng, scenario)
        if not _board_is_clear(board):
            continue

        voted, meta = _vote_macro_board(board, active_dirs)
        if meta["changed"] == 0:
            continue
        if scenario in {"tie", "mixed"} and meta["ties"] == 0:
            continue
        if len(np.unique(voted)) != 2:
            continue

        grid = _render_input(board, active_dirs, mask_color)
        try:
            actual = main(grid)
        except Exception:
            continue
        expected = _expand_macro_board(voted)
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
