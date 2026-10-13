from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
MASK_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
_GEN_COUNTER = 0


def _runs(values) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.copy())
    return variants


def _render(board: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=int)
    k = int(mask_a.shape[0])
    assert mask_a.shape == mask_b.shape == (k, k)
    h, w = board.shape
    out = np.zeros((h * k, w * k), dtype=int)
    for r in range(h):
        for c in range(w):
            mask = mask_a if (r + c) % 2 == 0 else mask_b
            block = out[r * k : (r + 1) * k, c * k : (c + 1) * k]
            block[mask] = int(board[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the lower checkerboard by alternating two top stencil templates on checkerboard parity.

    Input structure:
    - The background is black (0).
    - The top legend contains two left-to-right square grey (5) stencil masks of the same size, separated by one black column.
    - One all-black separator row splits the legend from a lower tight rectangular checkerboard that uses exactly two non-black, non-grey colors.

    Rule:
    - Call the left stencil template A and the right stencil template B.
    - For each checkerboard cell at board-local coordinate (r, c), use template A when (r + c) mod 2 = 0 and template B when (r + c) mod 2 = 1.
    - Replace every grey stencil pixel with that checkerboard cell's color, keep stencil holes black, and tile these k×k blocks into the output.
    - Return only the expanded tiling, not the legend or separator row.

    Key insight:
    - The checkerboard colors do not select the shape; board parity selects which legend stencil to use, while the cell color only recolors that chosen stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_runs = _runs(np.any(g != BG, axis=1))
    assert len(row_runs) == 2
    (top_r0, top_r1), (board_r0, board_r1) = row_runs
    assert board_r0 == top_r1 + 2

    top = g[top_r0 : top_r1 + 1, :]
    k = int(top.shape[0])
    assert k >= 2

    col_runs = _runs(np.any(top == MASK_COLOR, axis=0))
    assert len(col_runs) == 2
    (a_c0, a_c1), (b_c0, b_c1) = col_runs
    assert b_c0 == a_c1 + 2
    assert a_c1 - a_c0 + 1 == k
    assert b_c1 - b_c0 + 1 == k

    mask_a = top[:, a_c0 : a_c1 + 1] == MASK_COLOR
    mask_b = top[:, b_c0 : b_c1 + 1] == MASK_COLOR
    assert np.all(np.any(mask_a, axis=0)) and np.all(np.any(mask_a, axis=1))
    assert np.all(np.any(mask_b, axis=0)) and np.all(np.any(mask_b, axis=1))

    lower = g[board_r0 : board_r1 + 1, :]
    board_box = _bbox(lower != BG)
    assert board_box is not None
    br0, bc0, br1, bc1 = board_box
    board = lower[br0 : br1 + 1, bc0 : bc1 + 1]

    colors = sorted(int(c) for c in np.unique(board))
    assert len(colors) == 2
    assert BG not in colors and MASK_COLOR not in colors

    rr, cc = np.indices(board.shape)
    even_colors = {int(v) for v in board[(rr + cc) % 2 == 0]}
    odd_colors = {int(v) for v in board[(rr + cc) % 2 == 1]}
    assert len(even_colors) == 1 and len(odd_colors) == 1 and even_colors != odd_colors

    return _render(board, mask_a, mask_b)


def _base_masks(k: int) -> list[np.ndarray]:
    rr, cc = np.indices((k, k))
    mid = k // 2
    masks = [
        rr == cc,
        rr + cc == k - 1,
        (rr == cc) | (rr + cc == k - 1),
        (rr == 0) | (cc == 0),
        (rr == 0) | (cc == mid),
        (rr == mid) | (cc == mid),
        (rr == 0) | (rr == k - 1) | (cc == 0) | (cc == k - 1),
        cc <= rr,
        cc >= rr,
        (rr + cc) % 2 == 0,
        (rr + cc) % 2 == 1,
    ]
    if k == 4:
        masks.extend(
            [
                (rr <= 1) | (cc >= 2),
                (rr >= 1) | (cc == 0),
                ((rr == 1) | (rr == 2) | (cc == 1)),
            ]
        )
    return [mask.astype(bool) for mask in masks]


def _valid_template(mask: np.ndarray) -> bool:
    k = int(mask.shape[0])
    if mask.shape != (k, k):
        return False
    count = int(mask.sum())
    if count < k or count >= k * k:
        return False
    if not np.all(np.any(mask, axis=0)):
        return False
    if not np.all(np.any(mask, axis=1)):
        return False
    return True


def _template_pool(k: int) -> list[np.ndarray]:
    pool = []
    seen = set()
    for base in _base_masks(k):
        for variant in _d4_variants(base):
            if not _valid_template(variant):
                continue
            key = variant.tobytes()
            if key in seen:
                continue
            seen.add(key)
            pool.append(variant.copy())
    return pool


def _pick_templates(rng: random.Random, k: int) -> tuple[np.ndarray, np.ndarray]:
    pool = _template_pool(k)
    pairs = [
        (a, b)
        for a, b in combinations(pool, 2)
        if int(a.sum()) != int(b.sum()) and np.count_nonzero(a != b) >= k
    ]
    assert pairs
    a, b = rng.choice(pairs)
    if rng.randrange(2):
        return a.copy(), b.copy()
    return b.copy(), a.copy()


def _make_checkerboard(
    h: int, w: int, color_even: int, color_odd: int
) -> np.ndarray:
    rr, cc = np.indices((h, w))
    return np.where((rr + cc) % 2 == 0, color_even, color_odd).astype(int)


def _paste_template(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    patch = np.where(mask, MASK_COLOR, BG)
    grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _build_input(
    rng: random.Random,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    board: np.ndarray,
) -> np.ndarray:
    k = int(mask_a.shape[0])
    m, n = map(int, board.shape)
    legend_w = 2 * k + 1

    left_margin = rng.randint(0, 4)
    right_margin = rng.randint(0, 4)
    top_margin = rng.randint(0, 3)
    bottom_margin = rng.randint(0, 4)
    width = max(legend_w, n) + left_margin + right_margin
    height = top_margin + k + 1 + m + bottom_margin

    grid = np.zeros((height, width), dtype=int)
    legend_left = rng.randint(left_margin, width - right_margin - legend_w)
    board_left = rng.randint(left_margin, width - right_margin - n)
    legend_top = top_margin
    board_top = top_margin + k + 1

    _paste_template(grid, mask_a, legend_top, legend_left)
    _paste_template(grid, mask_b, legend_top, legend_left + k + 1)
    grid[board_top : board_top + m, board_left : board_left + n] = board
    return grid


def _semantic_checks(
    grid: np.ndarray,
    board: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    expected: np.ndarray,
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if int(mask_a.sum()) == int(mask_b.sum()):
        return False
    if np.array_equal(mask_a, mask_b):
        return False
    if board.shape[0] < 3 or board.shape[1] < 3:
        return False

    rr, cc = np.indices(board.shape)
    if len({int(v) for v in board[(rr + cc) % 2 == 0]}) != 1:
        return False
    if len({int(v) for v in board[(rr + cc) % 2 == 1]}) != 1:
        return False

    try:
        parsed = main(grid)
    except Exception:
        return False
    if not np.array_equal(parsed, expected):
        return False
    if np.array_equal(expected, _render(board, mask_b, mask_a)):
        return False
    if np.array_equal(expected, _render(board, mask_a, mask_a)):
        return False
    if np.array_equal(expected, _render(board, mask_b, mask_b)):
        return False

    solid = np.kron(board, np.ones(mask_a.shape, dtype=int))
    if np.array_equal(expected, solid):
        return False

    used_colors = set(int(v) for v in np.unique(expected)) - {BG}
    board_colors = set(int(v) for v in np.unique(board))
    if used_colors != board_colors:
        return False
    return True


def _gen_attempt(
    rng: random.Random,
    forced_k: int | None,
    even_uses_first_color: bool,
) -> np.ndarray | None:
    k = int(forced_k if forced_k is not None else rng.choice([3, 4]))
    mask_a, mask_b = _pick_templates(rng, k)

    max_side = 30 // k
    m = rng.randint(3, min(5, max_side))
    n = rng.randint(3, min(5, max_side))
    color1, color2 = rng.sample(PALETTE, 2)
    even_color, odd_color = (color1, color2) if even_uses_first_color else (color2, color1)
    board = _make_checkerboard(m, n, even_color, odd_color)

    grid = _build_input(rng, mask_a, mask_b, board)
    expected = _render(board, mask_a, mask_b)
    if _semantic_checks(grid, board, mask_a, mask_b, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    _GEN_COUNTER += 1
    rng = random.Random()
    forced_k = 3 if _GEN_COUNTER % 2 else 4
    even_uses_first_color = (_GEN_COUNTER % 2) == 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, forced_k, even_uses_first_color)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
