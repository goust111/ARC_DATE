from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


GRID_PARTS = 3
BLOCK_SIDE_CHOICES = (4, 5)
ACTIVE_COUNT_CHOICES = (4, 5, 6)
RETRY_LIMIT = 400
PALETTE = tuple(Color.NOT_BLACK)


def _extract_blueprint(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % GRID_PARTS == 0 and w % GRID_PARTS == 0, "grid must split evenly into 3x3 blocks"

    bg = int(get_bg(g))
    block_h = h // GRID_PARTS
    block_w = w // GRID_PARTS
    blueprint = np.zeros((GRID_PARTS, GRID_PARTS), dtype=int)

    for br in range(GRID_PARTS):
        for bc in range(GRID_PARTS):
            block = g[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ]
            fg = block[block != bg]
            assert fg.size <= 1, "each partition block must contain at most one foreground pixel"
            blueprint[br, bc] = int(fg[0]) if fg.size == 1 else bg
    return blueprint


def _apply_column_gravity(blueprint: np.ndarray) -> np.ndarray:
    bp = np.asarray(blueprint, dtype=int)
    assert bp.shape == (GRID_PARTS, GRID_PARTS)
    out = np.zeros_like(bp)
    for c in range(GRID_PARTS):
        colors = [int(v) for v in bp[:, c] if int(v) != 0]
        if colors:
            out[GRID_PARTS - len(colors) :, c] = colors
    return out


def _expand_blueprint(blueprint: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    block_h = h // GRID_PARTS
    block_w = w // GRID_PARTS
    out = np.zeros(shape, dtype=int)
    for br in range(GRID_PARTS):
        for bc in range(GRID_PARTS):
            color = int(blueprint[br, bc])
            if color == 0:
                continue
            out[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a 3x3 partition blueprint, drop its colors by column gravity, then repaint the large blocks.

    Input structure:
    - The input is a black-background grid whose height and width are each divisible by 3.
    - The grid is implicitly partitioned into a 3x3 array of equal rectangular blocks.
    - Each block contains either no foreground pixel or exactly one sparse non-black pixel marking that block's blueprint color.

    Rule:
    - Read each partition block and compress it to a 3x3 color blueprint, using black for empty blocks.
    - In every blueprint column, move the non-black colors downward to the lowest available rows while preserving their top-to-bottom order.
    - Create a blank output grid with the original size, then fill every large partition block whose settled blueprint cell is non-black with that settled color.
    - Do not keep the original sparse pixels.

    Key insight:
    - The sparse pixels only encode a coarse 3x3 blueprint; the decisive transformation happens after compression, because gravity rearranges the blueprint itself before it is expanded back into full-size colored blocks.
    """
    g = np.asarray(grid, dtype=int)
    blueprint = _extract_blueprint(g)
    settled = _apply_column_gravity(blueprint)
    return _expand_blueprint(settled, g.shape)


def _column_patterns(count: int) -> list[tuple[int, ...]]:
    patterns = []
    for bits in range(1 << GRID_PARTS):
        rows = tuple(r for r in range(GRID_PARTS) if bits & (1 << r))
        if len(rows) == count:
            patterns.append(rows)
    return patterns


PATTERNS_BY_COUNT = {count: _column_patterns(count) for count in range(GRID_PARTS + 1)}
COUNT_LAYOUTS = [
    counts
    for counts in (
        (3, 1, 0),
        (3, 0, 1),
        (1, 3, 0),
        (0, 3, 1),
        (1, 0, 3),
        (0, 1, 3),
        (2, 2, 0),
        (2, 0, 2),
        (0, 2, 2),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (3, 2, 0),
        (3, 0, 2),
        (2, 3, 0),
        (0, 3, 2),
        (2, 0, 3),
        (0, 2, 3),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (3, 2, 1),
        (3, 1, 2),
        (2, 3, 1),
        (1, 3, 2),
        (2, 1, 3),
        (1, 2, 3),
        (2, 2, 2),
    )
    if sum(counts) in ACTIVE_COUNT_CHOICES
]


def _pick_column_positions(counts: tuple[int, int, int]) -> list[tuple[int, ...]] | None:
    choices: list[tuple[int, ...]] = []
    moved = False
    for count in counts:
        patterns = list(PATTERNS_BY_COUNT[count])
        random.shuffle(patterns)
        selected = None
        target = tuple(range(GRID_PARTS - count, GRID_PARTS))
        for rows in patterns:
            if count > 0 and rows != target:
                moved = True
            selected = rows
            if count == 0 or rows != target:
                break
        if selected is None:
            return None
        choices.append(selected)
    if not moved:
        return None
    return choices


def _assign_column_colors(counts: tuple[int, int, int], distinct_count: int) -> list[list[int]] | None:
    total = sum(counts)
    if distinct_count > total:
        return None

    slots = [c for c, count in enumerate(counts) for _ in range(count)]
    for _ in range(100):
        palette = random.sample(PALETTE, distinct_count)
        colors = palette.copy()
        while len(colors) < total:
            colors.append(random.choice(palette))
        random.shuffle(colors)

        per_column: list[list[int]] = [[] for _ in range(GRID_PARTS)]
        ok = True
        for col, color in zip(slots, colors):
            if color in per_column[col]:
                ok = False
                break
            per_column[col].append(int(color))
        if not ok:
            continue
        if len({color for column in per_column for color in column}) != distinct_count:
            continue
        if max(len(column) for column in per_column) < 2:
            continue
        return per_column
    return None


def _build_blueprint(
    counts: tuple[int, int, int],
    positions: list[tuple[int, ...]],
    column_colors: list[list[int]],
) -> np.ndarray:
    blueprint = np.zeros((GRID_PARTS, GRID_PARTS), dtype=int)
    for c in range(GRID_PARTS):
        rows = positions[c]
        colors = column_colors[c]
        assert len(rows) == len(colors)
        for r, color in zip(rows, colors):
            blueprint[r, c] = int(color)
    return blueprint


def _place_sparse_pixels(blueprint: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    grid = np.zeros((GRID_PARTS * block_h, GRID_PARTS * block_w), dtype=int)
    for br in range(GRID_PARTS):
        for bc in range(GRID_PARTS):
            color = int(blueprint[br, bc])
            if color == 0:
                continue
            rr = br * block_h + random.randrange(block_h)
            cc = bc * block_w + random.randrange(block_w)
            grid[rr, cc] = color
    return grid


def _validate_semantics(inp: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    blueprint = _extract_blueprint(g)
    settled = _apply_column_gravity(blueprint)
    out = main(g)
    no_gravity = _expand_blueprint(blueprint, g.shape)

    assert g.shape == out.shape
    assert g.shape[0] <= 30 and g.shape[1] <= 30
    assert 4 <= int(np.count_nonzero(blueprint)) <= 6
    distinct = {int(v) for v in blueprint.flatten() if int(v) != 0}
    assert 3 <= len(distinct) <= 5
    assert not np.array_equal(blueprint, settled), "gravity must change the blueprint"
    assert not np.array_equal(out, no_gravity), "gravity must affect the expanded output"

    occupied_columns = [c for c in range(GRID_PARTS) if np.any(blueprint[:, c] != 0)]
    assert len(occupied_columns) >= 2
    assert any(np.count_nonzero(blueprint[:, c]) >= 2 for c in occupied_columns)

    for c in range(GRID_PARTS):
        initial_column = [int(v) for v in blueprint[:, c] if int(v) != 0]
        final_column = [int(v) for v in settled[:, c] if int(v) != 0]
        assert initial_column == final_column
        assert len(final_column) == len(set(final_column))

    block_h = g.shape[0] // GRID_PARTS
    block_w = g.shape[1] // GRID_PARTS
    for br in range(GRID_PARTS):
        for bc in range(GRID_PARTS):
            block = g[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ]
            assert np.count_nonzero(block) <= 1
            out_block = out[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ]
            target = int(settled[br, bc])
            if target == 0:
                assert np.all(out_block == 0)
            else:
                assert np.all(out_block == target)


def _generate_case() -> np.ndarray | None:
    block_h = random.choice(BLOCK_SIDE_CHOICES)
    block_w = random.choice(BLOCK_SIDE_CHOICES)

    counts = random.choice(COUNT_LAYOUTS)
    distinct_count = random.randint(3, min(5, sum(counts)))

    positions = _pick_column_positions(counts)
    if positions is None:
        return None

    column_colors = _assign_column_colors(counts, distinct_count)
    if column_colors is None:
        return None

    blueprint = _build_blueprint(counts, positions, column_colors)
    settled = _apply_column_gravity(blueprint)
    if np.array_equal(blueprint, settled):
        return None

    grid = _place_sparse_pixels(blueprint, block_h, block_w)
    if not np.array_equal(_extract_blueprint(grid), blueprint):
        return None

    _validate_semantics(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
