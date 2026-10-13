from __future__ import annotations

from collections import Counter
from itertools import permutations
import random

import numpy as np


BACKGROUND = 0
LINE_KINDS = ("horizontal", "vertical", "diag_down", "diag_up")
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
LINE_COUNT_RANGE = (3, 5)
MIN_VISIBLE_PIXELS = 2
MIN_DIAGONAL_LENGTH = 4
GENERATION_ATTEMPTS = 600
CHAIN_BUILD_ATTEMPTS = 160


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    height, width = map(int, shape)
    if kind == "horizontal":
        return frozenset((int(value), c) for c in range(width))
    if kind == "vertical":
        return frozenset((r, int(value)) for r in range(height))
    if kind == "diag_down":
        return frozenset((r, r - int(value)) for r in range(height) if 0 <= r - int(value) < width)
    if kind == "diag_up":
        return frozenset((r, int(value) - r) for r in range(height) if 0 <= int(value) - r < width)
    raise AssertionError(f"unknown line kind: {kind}")


def _infer_signature(coords: np.ndarray) -> tuple[str, int]:
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert int(coords.shape[0]) >= MIN_VISIBLE_PIXELS

    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[tuple[str, int]] = []
    if np.all(rows == rows[0]):
        candidates.append(("horizontal", int(rows[0])))
    if np.all(cols == cols[0]):
        candidates.append(("vertical", int(cols[0])))
    diffs = rows - cols
    if np.all(diffs == diffs[0]):
        candidates.append(("diag_down", int(diffs[0])))
    sums = rows + cols
    if np.all(sums == sums[0]):
        candidates.append(("diag_up", int(sums[0])))
    assert len(candidates) == 1, "each color must reveal one recoverable straight line"
    return candidates[0]


def _recover_lines(grid: np.ndarray, bg: int) -> dict[int, frozenset[tuple[int, int]]]:
    lines: dict[int, frozenset[tuple[int, int]]] = {}
    seen_signatures: set[tuple[str, int]] = set()
    for color in _colors_present(grid, bg):
        coords = np.argwhere(grid == color)
        signature = _infer_signature(coords)
        assert signature not in seen_signatures, "two colors cannot encode the same line"
        seen_signatures.add(signature)
        cells = _line_cells(grid.shape, *signature)
        assert set(map(tuple, coords)).issubset(cells)
        lines[int(color)] = cells
    assert lines
    return lines


def _below_edges(
    grid: np.ndarray,
    lines: dict[int, frozenset[tuple[int, int]]],
    bg: int,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    items = list(lines.items())
    height, width = map(int, grid.shape)
    for r in range(height):
        for c in range(width):
            covering = [color for color, cells in items if (r, c) in cells]
            visible = int(grid[r, c])
            if visible == bg:
                assert not covering
                continue
            assert visible in covering
            for color in covering:
                if color != visible:
                    edges.add((int(color), visible))
    assert edges, "expected at least one informative crossing"
    return edges


def _infer_order(
    grid: np.ndarray,
    lines: dict[int, frozenset[tuple[int, int]]],
    bg: int,
) -> tuple[int, ...]:
    colors = list(lines)
    edges = _below_edges(grid, lines, bg)
    valid_orders: list[tuple[int, ...]] = []
    for order in permutations(colors):
        pos = {color: i for i, color in enumerate(order)}
        if all(pos[lower] < pos[upper] for lower, upper in edges):
            valid_orders.append(tuple(int(v) for v in order))
    assert len(valid_orders) == 1, "visible crossing colors must imply one total layer order"
    return valid_orders[0]


def _render_input(shape: tuple[int, int], ordered_lines: list[tuple[int, frozenset[tuple[int, int]]]], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for color, cells in ordered_lines:
        for r, c in cells:
            grid[int(r), int(c)] = int(color)
    return grid


def _render_shifted(
    shape: tuple[int, int],
    lines: dict[int, frozenset[tuple[int, int]]],
    order: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), int(bg), dtype=int)
    for depth, color in enumerate(order, start=1):
        for r, c in lines[int(color)]:
            rr = int(r) + depth
            cc = int(c)
            if 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden layer order of crossing full-span lines and redraw them lower by depth.

    Input structure:
    - The grid has a dominant black background and 3-5 distinct foreground colors.
    - Each foreground color is the visible part of one full one-cell-thick horizontal, vertical, or 45-degree diagonal line.
    - Lines may cross, and at each crossing only the highest visible line color remains on the input grid.

    Rule:
    - Recover every line's full geometric path from its visible collinear cells, including the hidden crossing cells.
    - Read every crossing as an occlusion constraint: the visible color there lies above every other line through that cell, and these constraints determine a unique bottom-to-top order.
    - Clear the grid, then redraw the recovered lines from bottom layer to top layer, shifting each line downward by K rows where K is its layer index starting from 1.
    - Clip shifted pixels that leave the bottom edge, and let later redraws overwrite earlier ones wherever the shifted lines cross again.

    Key insight:
    - The crossing colors do not directly specify the answer; they first reveal the unique occlusion order, and that hidden order becomes both the vertical displacement amount and the redraw priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    lines = _recover_lines(g, bg)
    order = _infer_order(g, lines, bg)
    return _render_shifted(g.shape, lines, order, bg)


def _candidate_lines(shape: tuple[int, int]) -> list[tuple[str, int, frozenset[tuple[int, int]]]]:
    height, width = map(int, shape)
    pool: list[tuple[str, int, frozenset[tuple[int, int]]]] = []
    for row in range(height):
        pool.append(("horizontal", row, _line_cells(shape, "horizontal", row)))
    for col in range(width):
        pool.append(("vertical", col, _line_cells(shape, "vertical", col)))
    for diff in range(-(width - 1), height):
        cells = _line_cells(shape, "diag_down", diff)
        if len(cells) >= MIN_DIAGONAL_LENGTH:
            pool.append(("diag_down", diff, cells))
    for total in range(height + width - 1):
        cells = _line_cells(shape, "diag_up", total)
        if len(cells) >= MIN_DIAGONAL_LENGTH:
            pool.append(("diag_up", total, cells))
    return pool


def _crosses(
    a: tuple[str, int, frozenset[tuple[int, int]]],
    b: tuple[str, int, frozenset[tuple[int, int]]],
) -> bool:
    return bool(a[2] & b[2])


def _build_line_path(shape: tuple[int, int], count: int) -> list[tuple[str, int, frozenset[tuple[int, int]]]] | None:
    pool = _candidate_lines(shape)

    for _ in range(CHAIN_BUILD_ATTEMPTS):
        first = random.choice(pool)
        order = [first]
        used = {(first[0], first[1])}

        def backtrack() -> bool:
            if len(order) == count:
                return True

            prev = order[-1]
            candidates: list[tuple[str, int, frozenset[tuple[int, int]]]] = []
            for cand in pool:
                key = (cand[0], cand[1])
                if key in used:
                    continue
                if not _crosses(prev, cand):
                    continue
                if any(_crosses(cand, earlier) for earlier in order[:-1]):
                    continue
                candidates.append(cand)

            random.shuffle(candidates)
            candidates.sort(
                key=lambda cand: (
                    cand[0] != prev[0],
                    len(cand[2]),
                    random.random(),
                ),
                reverse=True,
            )

            for cand in candidates:
                order.append(cand)
                used.add((cand[0], cand[1]))
                if backtrack():
                    return True
                used.remove((cand[0], cand[1]))
                order.pop()
            return False

        if backtrack():
            return order
    return None


def _has_clipping(shape: tuple[int, int], lines: dict[int, frozenset[tuple[int, int]]], order: tuple[int, ...]) -> bool:
    height = int(shape[0])
    for depth, color in enumerate(order, start=1):
        if any(int(r) + depth >= height for r, _ in lines[int(color)]):
            return True
    return False


def _all_shifted_visible(shape: tuple[int, int], lines: dict[int, frozenset[tuple[int, int]]], order: tuple[int, ...]) -> bool:
    height = int(shape[0])
    for depth, color in enumerate(order, start=1):
        if not any(int(r) + depth < height for r, _ in lines[int(color)]):
            return False
    return True


def _line_type_mix(signatures: list[tuple[str, int, frozenset[tuple[int, int]]]]) -> bool:
    kinds = {kind for kind, _value, _cells in signatures}
    has_diag = any(kind.startswith("diag") for kind in kinds)
    has_ortho = any(kind in {"horizontal", "vertical"} for kind in kinds)
    if random.random() < 0.65:
        return has_diag and has_ortho
    return True


def _semantic_checks(inp: np.ndarray, expected_clip: bool) -> bool:
    try:
        bg = _get_bg(inp)
        if bg != BACKGROUND:
            return False
        lines = _recover_lines(inp, bg)
        order = _infer_order(inp, lines, bg)
    except AssertionError:
        return False

    if any(int(np.count_nonzero(inp == color)) < MIN_VISIBLE_PIXELS for color in order):
        return False
    if not _all_shifted_visible(inp.shape, lines, order):
        return False
    if _has_clipping(inp.shape, lines, order) != expected_clip:
        return False

    out = _render_shifted(inp.shape, lines, order, bg)
    if np.array_equal(out, inp):
        return False
    if any(int(np.count_nonzero(out == color)) == 0 for color in order):
        return False

    reverse = _render_shifted(inp.shape, lines, tuple(reversed(order)), bg)
    if np.array_equal(reverse, out):
        return False

    uniform = np.full_like(inp, bg)
    for color in order:
        for r, c in lines[int(color)]:
            rr = int(r) + 1
            if rr < inp.shape[0]:
                uniform[rr, int(c)] = int(color)
    if np.array_equal(uniform, out):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*LINE_COUNT_RANGE)
    expected_clip = random.random() < 0.5
    shape = (height, width)

    signatures = _build_line_path(shape, count)
    if signatures is None:
        return None
    if not _line_type_mix(signatures):
        return None

    colors = random.sample([c for c in range(1, 10) if c != BACKGROUND], count)
    ordered_lines = [(int(color), cells) for color, (_kind, _value, cells) in zip(colors, signatures)]
    inp = _render_input(shape, ordered_lines, BACKGROUND)

    if not _semantic_checks(inp, expected_clip):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
