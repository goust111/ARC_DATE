from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PANEL_SIZE = 3
SEPARATOR_ROWS = 1
OPERATIONS_ROW0 = PANEL_SIZE + SEPARATOR_ROWS
STEP = 2


def _mask(rows: tuple[str, str, str]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(1 if ch == "1" else 0 for ch in row) for row in rows)


SYMBOL_TO_COLOR = {
    _mask(("010", "011", "011")): 1,
    _mask(("111", "110", "000")): 1,
    _mask(("100", "110", "110")): 2,
    _mask(("000", "100", "111")): 2,
    _mask(("010", "011", "001")): 3,
    _mask(("000", "110", "011")): 3,
    _mask(("000", "110", "100")): 4,
    _mask(("110", "011", "000")): 4,
}


def _direction_from_symbol(symbol: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
    coords = np.argwhere(np.array(symbol, dtype=int) == 1)
    assert 3 <= len(coords) <= 5, "symbol must contain 3-5 grey cells"
    dr, dc = coords.mean(axis=0) - np.array([1.0, 1.0])
    if abs(dc) >= abs(dr):
        return (0, 1) if dc > 0 else (0, -1)
    return (1, 0) if dr > 0 else (-1, 0)


COLOR_TO_SYMBOLS: dict[int, list[tuple[tuple[int, ...], ...]]] = {}
for _symbol, _color in SYMBOL_TO_COLOR.items():
    COLOR_TO_SYMBOLS.setdefault(int(_color), []).append(_symbol)


def _decode_panels(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    height, width = map(int, grid.shape)
    assert height >= OPERATIONS_ROW0 + 1, "expected symbol rows, a separator row, and an operation area"
    assert width % PANEL_SIZE == 0, "expected an integer number of 3x3 panels"

    instructions: list[tuple[int, tuple[int, int]]] = []
    for c0 in range(0, width, PANEL_SIZE):
        panel = grid[:PANEL_SIZE, c0 : c0 + PANEL_SIZE]
        assert panel.shape == (PANEL_SIZE, PANEL_SIZE), "every panel must be 3x3"
        symbol = tuple(tuple(int(v) for v in row) for row in (panel == GREY).astype(int))
        assert symbol in SYMBOL_TO_COLOR, "encountered an unknown symbol"
        instructions.append((int(SYMBOL_TO_COLOR[symbol]), _direction_from_symbol(symbol)))
    return instructions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each top 3x3 grey symbol into a color and move that color's bottom pixels in the direction encoded by the same symbol.

    Input structure:
    - The grid has a black background, with the top three rows partitioned into adjacent 3x3 black/grey symbol panels.
    - A single black separator row sits below the symbol band.
    - The lower operation area contains isolated single pixels whose colors match the symbol codebook.

    Rule:
    - Read each 3x3 panel as a binary grey-mask symbol and decode it with the task codebook to obtain a target color.
    - Compute the grey-pixel centroid inside that same 3x3 symbol; the dominant signed offset from the panel center chooses up, down, left, or right.
    - Move every operation-area pixel of the decoded color by two cells in that direction, clear its source cell to black, and keep the symbol band unchanged.
    - Apply panels from left to right; if multiple moved pixels land on one cell, the later move overwrites the earlier one.

    Key insight:
    - Each symbol carries two independent instructions at once: its exact binary shape identifies which color to move, while its grey-cell balance around the center identifies where that color should move.
    """
    g = np.asarray(grid, dtype=int)
    instructions = _decode_panels(g)

    out = g.copy()
    op = np.full_like(g[OPERATIONS_ROW0:, :], BLACK)
    original_op = g[OPERATIONS_ROW0:, :]

    moved_colors = {color for color, _direction in instructions}
    for r, c in np.argwhere((original_op != BLACK) & ~np.isin(original_op, list(moved_colors))):
        op[int(r), int(c)] = int(original_op[int(r), int(c)])

    for color, (dr, dc) in instructions:
        for r, c in np.argwhere(original_op == int(color)):
            nr = int(r + dr * STEP)
            nc = int(c + dc * STEP)
            assert 0 <= nr < op.shape[0] and 0 <= nc < op.shape[1], "translated pixel must stay in bounds"
            op[nr, nc] = int(color)

    out[OPERATIONS_ROW0:, :] = op
    return out


GEN_HEIGHT_RANGE = (8, 12)
GEN_PANEL_COUNT_RANGE = (2, 4)
GEN_PIXELS_PER_COLOR_RANGE = (1, 3)
GEN_RETRY_LIMIT = 400
GEN_COLLISION_PROBABILITY = 0.4


def _is_isolated(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _in_operation_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _sample_symbols(panel_colors: list[int]) -> list[tuple[tuple[int, ...], ...]]:
    for _ in range(80):
        symbols = [random.choice(COLOR_TO_SYMBOLS[color]) for color in panel_colors]
        directions = [_direction_from_symbol(symbol) for symbol in symbols]
        vertical = any(dr != 0 for dr, _dc in directions)
        horizontal = any(dc != 0 for _dr, dc in directions)
        if len(set(directions)) >= 2 and vertical and horizontal:
            return symbols
    raise RuntimeError("failed to sample diverse symbol directions")


def _collision_pair(
    op_shape: tuple[int, int],
    colors: list[int],
    directions: list[tuple[int, int]],
) -> tuple[dict[int, list[tuple[int, int]]], tuple[int, int] | None] | None:
    pairs = [
        (i, j)
        for i in range(len(colors))
        for j in range(i + 1, len(colors))
        if directions[i] != directions[j]
    ]
    random.shuffle(pairs)

    for i, j in pairs:
        dr1, dc1 = directions[i]
        dr2, dc2 = directions[j]
        candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        for r in range(op_shape[0]):
            for c in range(op_shape[1]):
                dst = (r, c)
                src1 = (r - dr1 * STEP, c - dc1 * STEP)
                src2 = (r - dr2 * STEP, c - dc2 * STEP)
                if not _in_operation_bounds(op_shape, src1) or not _in_operation_bounds(op_shape, src2):
                    continue
                if src1 == src2 or not _is_isolated(src1, {src2}):
                    continue
                candidates.append((src1, src2, dst))
        if candidates:
            src1, src2, dst = random.choice(candidates)
            return {int(colors[i]): [src1], int(colors[j]): [src2]}, dst
    return None


def _place_sources(
    colors: list[int],
    directions: list[tuple[int, int]],
    counts: dict[int, int],
    op_shape: tuple[int, int],
    want_collision: bool,
) -> dict[int, list[tuple[int, int]]] | None:
    placed: dict[int, list[tuple[int, int]]] = {int(color): [] for color in colors}
    used_sources: set[tuple[int, int]] = set()
    used_destinations: set[tuple[int, int]] = set()
    collision_slot: tuple[int, int] | None = None

    if want_collision:
        forced = _collision_pair(op_shape, colors, directions)
        if forced is not None:
            seed_sources, collision_slot = forced
            used_destinations.add(collision_slot)
            for color, srcs in seed_sources.items():
                for src in srcs:
                    placed[color].append(src)
                    used_sources.add(src)

    for color, direction in zip(colors, directions):
        dr, dc = direction
        while len(placed[color]) < counts[color]:
            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for r in range(op_shape[0]):
                for c in range(op_shape[1]):
                    src = (r, c)
                    dst = (r + dr * STEP, c + dc * STEP)
                    if not _in_operation_bounds(op_shape, dst):
                        continue
                    if src in used_sources or dst in used_sources:
                        continue
                    if collision_slot is not None and dst == collision_slot:
                        continue
                    if dst in used_destinations:
                        continue
                    if not _is_isolated(src, used_sources):
                        continue
                    if not _is_isolated(dst, used_destinations):
                        continue
                    candidates.append((src, dst))

            if not candidates:
                return None

            src, dst = random.choice(candidates)
            placed[color].append(src)
            used_sources.add(src)
            used_destinations.add(dst)

    return placed


def _render_input(
    height: int,
    panel_colors: list[int],
    symbols: list[tuple[tuple[int, ...], ...]],
    placed_sources: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    width = PANEL_SIZE * len(panel_colors)
    grid = np.full((height, width), BLACK, dtype=int)

    for panel_index, symbol in enumerate(symbols):
        c0 = panel_index * PANEL_SIZE
        panel = np.array(symbol, dtype=int)
        grid[:PANEL_SIZE, c0 : c0 + PANEL_SIZE] = np.where(panel == 1, GREY, BLACK)

    for color, srcs in placed_sources.items():
        for r, c in srcs:
            grid[OPERATIONS_ROW0 + int(r), int(c)] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*GEN_PANEL_COUNT_RANGE)
    panel_colors = random.sample(sorted(COLOR_TO_SYMBOLS), panel_count)
    symbols = _sample_symbols(panel_colors)
    directions = [_direction_from_symbol(symbol) for symbol in symbols]

    height = random.randint(*GEN_HEIGHT_RANGE)
    width = PANEL_SIZE * panel_count
    op_shape = (height - OPERATIONS_ROW0, width)

    counts = {
        int(color): random.randint(*GEN_PIXELS_PER_COLOR_RANGE)
        for color in panel_colors
    }
    want_collision = random.random() < GEN_COLLISION_PROBABILITY
    placed_sources = _place_sources(panel_colors, directions, counts, op_shape, want_collision)
    if placed_sources is None:
        return None

    grid = _render_input(height, panel_colors, symbols, placed_sources)
    if not np.array_equal(main(grid), main(grid.copy())):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
