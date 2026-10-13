from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _row_colors(row: np.ndarray) -> tuple[int, int | None]:
    counts = Counter(map(int, row))
    assert counts, "each row must contain at least one cell"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color in every row"

    body_color = int(ranked[0][0])
    if len(ranked) == 1:
        return body_color, None

    assert len(ranked) == 2, "a noisy row must contain exactly one pointer color"
    pointer_color = int(ranked[1][0])
    assert ranked[1][1] < ranked[0][1], "pointer pixels must be the row minority"
    return body_color, pointer_color


def _chain_order(body_colors: list[int], pointer_colors: list[int | None]) -> list[int]:
    n = len(body_colors)
    assert n >= 2, "expected at least two rows"
    assert len(set(body_colors)) == n, "row body colors must be distinct"

    color_to_row = {color: row for row, color in enumerate(body_colors)}
    successors: list[int | None] = [None] * n
    indegree = [0] * n
    tails = 0

    for row, pointer in enumerate(pointer_colors):
        if pointer is None:
            tails += 1
            continue
        assert pointer in color_to_row, "each pointer color must name another row body color"
        nxt = color_to_row[int(pointer)]
        assert nxt != row, "a row may not point to itself"
        successors[row] = nxt
        indegree[nxt] += 1

    assert tails == 1, "expected exactly one solid tail row"
    assert all(deg <= 1 for deg in indegree), "a row in a chain can have at most one predecessor"
    heads = [row for row, deg in enumerate(indegree) if deg == 0]
    assert len(heads) == 1, "expected exactly one chain head"

    order: list[int] = []
    seen: set[int] = set()
    row: int | None = heads[0]
    while row is not None:
        assert row not in seen, "pointer chain must not contain a cycle"
        seen.add(row)
        order.append(row)
        row = successors[row]

    assert len(order) == n, "the pointer relation must connect every row into one chain"
    assert successors[order[-1]] is None, "the final chain row must be the solid row"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder nearly solid rows by reading each row's minority color as a pointer to its successor.

    Input structure:
    - Each row is mostly filled with one body color, and all row body colors are distinct.
    - Exactly one row is completely solid; every other row has a small number of pixels in exactly one other color.
    - Each minority color equals the body color of another row, so the rows form one successor chain ending at the solid row.

    Rule:
    - For each row, take the majority color as the row body color and the sole minority color as the pointer color.
    - Follow pointers from the only row with no predecessor to the solid tail row.
    - Return a grid of the same shape whose rows are pure body-color stripes in that chain order.

    Key insight:
    - The minority pixels are not removable noise; their color names the next row, while their count and positions are irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 3, "expected at least two rows and enough columns for minority pixels"

    body_colors: list[int] = []
    pointer_colors: list[int | None] = []
    for r in range(h):
        body, pointer = _row_colors(g[r, :])
        body_colors.append(body)
        pointer_colors.append(pointer)

    order = _chain_order(body_colors, pointer_colors)
    out = np.empty_like(g)
    for out_r, src_r in enumerate(order):
        out[out_r, :] = int(body_colors[src_r])
    return out


# --------------- generator ---------------

_GEN_COLORS = tuple(range(10))
_GEN_ROW_COUNT_RANGE = (4, 9)
_GEN_WIDTH_RANGE = (8, 20)
_GEN_MAX_POINTER_PIXELS = 4
_GEN_RETRIES = 500


def _expected_from_metadata(body_colors: list[int], chain: list[int], width: int) -> np.ndarray:
    out = np.empty((len(body_colors), width), dtype=int)
    for r, src_r in enumerate(chain):
        out[r, :] = int(body_colors[src_r])
    return out


def _majority_denoise(grid: np.ndarray) -> np.ndarray:
    out = np.empty_like(grid)
    for r in range(grid.shape[0]):
        color = Counter(map(int, grid[r, :])).most_common(1)[0][0]
        out[r, :] = int(color)
    return out


def _valid_chain_shape(chain: list[int], body_colors: list[int]) -> bool:
    n = len(chain)
    if chain == list(range(n)):
        return False
    if chain[0] == 0 or chain[-1] == n - 1:
        return False

    deltas = [b - a for a, b in zip(chain, chain[1:])]
    if not any(delta > 0 for delta in deltas):
        return False
    if not any(delta < 0 for delta in deltas):
        return False

    by_color = sorted(range(n), key=lambda row: body_colors[row])
    if chain == by_color or chain == by_color[::-1]:
        return False
    return True


def _render_input(body_colors: list[int], chain: list[int], width: int) -> np.ndarray:
    n = len(body_colors)
    grid = np.empty((n, width), dtype=int)
    for r, color in enumerate(body_colors):
        grid[r, :] = int(color)

    max_pointer_count = min(_GEN_MAX_POINTER_PIXELS, max(1, (width - 1) // 3))
    pointer_counts = [0] * n
    for src, dst in zip(chain, chain[1:]):
        pointer_counts[src] = random.randint(1, max_pointer_count)

    if max_pointer_count >= 2:
        noisy_rows = [src for src in chain[:-1]]
        pointer_counts[random.choice(noisy_rows)] = random.randint(2, max_pointer_count)

    for src, dst in zip(chain, chain[1:]):
        count = pointer_counts[src]
        cols = random.sample(range(width), count)
        grid[src, cols] = int(body_colors[dst])

    return grid


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, body_colors: list[int], chain: list[int]) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(expected, _majority_denoise(grid)):
        return False
    if not _valid_chain_shape(chain, body_colors):
        return False

    rows_with_noise = 0
    multi_pixel_pointer = False
    for r in range(grid.shape[0]):
        counts = Counter(map(int, grid[r, :]))
        if len(counts) == 2:
            rows_with_noise += 1
            if min(counts.values()) > 1:
                multi_pixel_pointer = True
        elif len(counts) != 1:
            return False
    if rows_with_noise != grid.shape[0] - 1:
        return False
    if not multi_pixel_pointer:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_ROW_COUNT_RANGE)
    width = random.randint(*_GEN_WIDTH_RANGE)
    body_colors = random.sample(_GEN_COLORS, height)

    chain = random.sample(range(height), height)
    for _ in range(100):
        if _valid_chain_shape(chain, body_colors):
            break
        chain = random.sample(range(height), height)
    if not _valid_chain_shape(chain, body_colors):
        return None

    grid = _render_input(body_colors, chain, width)
    expected = _expected_from_metadata(body_colors, chain, width)
    if not _semantic_ok(grid, expected, body_colors, chain):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-chain rearrangement grid")
