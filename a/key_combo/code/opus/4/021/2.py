from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, colors_present, paste


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
GREY = int(Color.GREY)

SEPARATOR = BLACK
ERASED = GREY
START = GREEN

BLOCK_SIZE_RANGE = range(3, 6)
MACRO_ROWS_RANGE = range(2, 5)
MACRO_COLS_RANGE = range(3, 6)
MAX_GENERATION_ATTEMPTS = 500
SPRITE_ATTEMPTS = 100


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({SEPARATOR, ERASED})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    if not separators:
        return [(0, int(length))]
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty blocks"
    assert len(set(sizes)) == 1, "all blocks must have the same size"
    return intervals


def _find_sprite(
    g: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    payload_colors: list[int],
) -> np.ndarray:
    payload = set(payload_colors)
    candidates: list[np.ndarray] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            if np.any(block == ERASED) or np.any(block == START):
                continue
            if set(colors_present(block)) == payload:
                candidates.append(block.copy())
    assert candidates, "expected at least one fully visible mixed sprite block"
    sprite = candidates[0]
    for candidate in candidates[1:]:
        assert np.array_equal(candidate, sprite), "fully visible sprite blocks must agree"
    return sprite


def _background_and_foreground(sprite: np.ndarray, payload_colors: list[int]) -> tuple[int, int]:
    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload_colors}
    vals = sorted(counts.values())
    assert vals[0] != vals[1], "sprite colors need a strict majority/minority split"
    background = max(payload_colors, key=lambda color: counts[color])
    foreground = min(payload_colors, key=lambda color: counts[color])
    return int(background), int(foreground)


def _decode(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    row_separators = [r for r in range(H) if _is_separator_line(g[r, :])]
    col_separators = [c for c in range(W) if _is_separator_line(g[:, c])]
    row_intervals = _regular_intervals(row_separators, H)
    col_intervals = _regular_intervals(col_separators, W)

    block_heights = {r1 - r0 for r0, r1 in row_intervals}
    block_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert block_heights == block_widths, "macro cells must be square"

    payload_colors = colors_present(g, exclude={SEPARATOR, ERASED, START})
    assert len(payload_colors) == 2, "expected exactly two payload colors"
    sprite = _find_sprite(g, row_intervals, col_intervals, payload_colors)
    background, blank = _background_and_foreground(sprite, payload_colors)

    out = np.full_like(g, SEPARATOR)
    passable = np.zeros_like(g, dtype=bool)
    if row_separators:
        passable[row_separators, :] = True
    if col_separators:
        passable[:, col_separators] = True

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            known = ~np.isin(block, [ERASED, START])
            assert known.any(), "each block must expose at least one known pixel"
            matches_sprite = bool(np.all(block[known] == sprite[known]))
            matches_blank = bool(np.all(block[known] == blank))
            assert matches_sprite != matches_blank, "block classification must be unambiguous"
            if matches_sprite:
                out[r0:r1, c0:c1] = background
            else:
                out[r0:r1, c0:c1] = sprite
                passable[r0:r1, c0:c1] = sprite == background

    starts = [tuple(map(int, rc)) for rc in np.argwhere(g == START)]
    return out, passable, starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the swapped block-grid code and paint horizontal green flow lines through the revealed channel blocks.

    Input structure:
    - Black separator rows and columns split the grid into equal square blocks.
    - Each block is either one repeated two-color sprite or one uniform blank block of the sprite's minority color.
    - Grey pixels are damage/wildcards, and green pixels on the left edge mark horizontal flow start rows.

    Rule:
    - Infer the repeated sprite from fully visible mixed-color blocks, treating grey damage and green start pixels as non-code overlays.
    - Decode by swapping the block states: sprite blocks become solid majority-color wall blocks, blank blocks become sprite channel blocks, and separator lines are restored to black.
    - Inside decoded sprite blocks only the sprite background color is passable; the sprite foreground color and every solid wall block are blocked, while black separator pixels are passable.
    - From each green start pixel, move right one pixel at a time and recolor every passable cell green until the next cell would be blocked or the grid edge is reached.

    Key insight:
    - The traversable corridor is determined only after inverting the block code, because originally blank macro-cells become the sprite-shaped channels and originally sprite-coded macro-cells become solid walls.
    """
    decoded, passable, starts = _decode(grid)
    out = decoded.copy()
    _H, W = out.shape
    for r, c in starts:
        out[r, c] = START
        cc = int(c) + 1
        while cc < W and passable[r, cc]:
            out[r, cc] = START
            cc += 1
    return out


def _block_origin(block_r: int, block_c: int, block_size: int) -> tuple[int, int]:
    step = block_size + 1
    return int(block_r * step), int(block_c * step)


def _make_sprite(block_size: int) -> tuple[np.ndarray, list[int]]:
    for _ in range(SPRITE_ATTEMPTS):
        sprite = np.full((block_size, block_size), RED, dtype=int)
        corridor_count = random.randint(1, min(2, block_size - 1))
        corridor_rows = sorted(random.sample(range(block_size), corridor_count))
        spine_c = random.randrange(block_size)

        for r in range(block_size):
            if r in corridor_rows:
                continue
            cols = [c for c in range(block_size) if c != spine_c]
            blue_count = random.randint(1, max(1, len(cols) - 1))
            for c in random.sample(cols, blue_count):
                sprite[r, c] = BLUE

        if np.count_nonzero(sprite == BLUE) == 0:
            continue
        if np.count_nonzero(sprite == RED) <= np.count_nonzero(sprite == BLUE):
            continue
        if not any(np.any(row == BLUE) and np.any(row == RED) for row in sprite):
            continue
        return sprite, corridor_rows
    raise RuntimeError("failed to sample a valid sprite")


def _build_channel_mask(macro_rows: int, macro_cols: int) -> tuple[np.ndarray, int, int]:
    channel = np.zeros((macro_rows, macro_cols), dtype=bool)
    open_row = random.randrange(macro_rows)
    blocked_choices = [r for r in range(macro_rows) if r != open_row]
    blocked_row = random.choice(blocked_choices)

    channel[open_row, :] = True
    stop_col = random.randint(1, macro_cols - 1)
    channel[blocked_row, :stop_col] = True
    channel[blocked_row, stop_col] = False
    if stop_col + 1 < macro_cols:
        channel[blocked_row, stop_col + 1 :] = np.random.rand(macro_cols - stop_col - 1) < 0.5

    for r in range(macro_rows):
        if r in (open_row, blocked_row):
            continue
        row = np.random.rand(macro_cols) < random.choice((0.35, 0.5, 0.65))
        if row.all():
            row[random.randrange(macro_cols)] = False
        if not row.any():
            row[random.randrange(macro_cols)] = True
        channel[r] = row

    assert np.any(~channel), "need at least one wall block"
    return channel, open_row, blocked_row


def _render_input(
    sprite: np.ndarray,
    channel: np.ndarray,
    block_size: int,
) -> np.ndarray:
    macro_rows, macro_cols = channel.shape
    H = macro_rows * block_size + (macro_rows - 1)
    W = macro_cols * block_size + (macro_cols - 1)
    grid = np.full((H, W), SEPARATOR, dtype=int)
    blank = np.full((block_size, block_size), BLUE, dtype=int)
    for br in range(macro_rows):
        for bc in range(macro_cols):
            top, left = _block_origin(br, bc, block_size)
            patch = blank if channel[br, bc] else sprite
            paste(grid, patch, top, left)
    return grid


def _place_starts(
    grid: np.ndarray,
    block_size: int,
    corridor_rows: list[int],
    open_row: int,
    blocked_row: int,
    channel: np.ndarray,
) -> list[tuple[int, int]]:
    start_macro_rows = [open_row, blocked_row]
    extras = [r for r in range(channel.shape[0]) if r not in start_macro_rows]
    if extras and random.random() < 0.5:
        start_macro_rows.append(random.choice(extras))

    starts: list[tuple[int, int]] = []
    for macro_r in start_macro_rows:
        inner_r = random.choice(corridor_rows)
        rr = macro_r * (block_size + 1) + inner_r
        starts.append((int(rr), 0))

    for r, c in starts:
        grid[r, c] = START
    return starts


def _protect_visible_sprite_block(
    channel: np.ndarray,
    block_size: int,
    start_macro_rows: set[int],
) -> set[tuple[int, int]]:
    wall_blocks = [(int(r), int(c)) for r, c in np.argwhere(~channel)]
    preferred = [cell for cell in wall_blocks if cell[1] > 0 and cell[0] not in start_macro_rows]
    if not preferred:
        preferred = [cell for cell in wall_blocks if cell[1] > 0] or wall_blocks
    br, bc = random.choice(preferred)
    top, left = _block_origin(br, bc, block_size)
    return {(r, c) for r in range(top, top + block_size) for c in range(left, left + block_size)}


def _add_grey_damage(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
    starts: list[tuple[int, int]],
) -> None:
    H, W = grid.shape
    candidates = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if (r, c) not in protected and (r, c) not in starts
    ]
    if not candidates:
        return
    max_damage = min(len(candidates), max(2, (H * W) // 8))
    if max_damage <= 0:
        return
    damage_count = random.randint(0, max_damage)
    if damage_count == 0:
        return
    for r, c in random.sample(candidates, damage_count):
        grid[r, c] = ERASED


def _visible_mixed_block_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    row_separators = [r for r in range(H) if _is_separator_line(g[r, :])]
    col_separators = [c for c in range(W) if _is_separator_line(g[:, c])]
    row_intervals = _regular_intervals(row_separators, H)
    col_intervals = _regular_intervals(col_separators, W)
    payload = set(colors_present(g, exclude={SEPARATOR, ERASED, START}))
    count = 0
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            if np.any(block == ERASED) or np.any(block == START):
                continue
            if set(colors_present(block)) == payload:
                count += 1
    return count


def _has_partial_flow(output: np.ndarray, starts: list[tuple[int, int]]) -> bool:
    W = output.shape[1]
    for r, _c in starts:
        green_cols = np.flatnonzero(output[r] == START)
        if len(green_cols) > 1 and int(green_cols[-1]) < W - 1:
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        block_size = random.choice(tuple(BLOCK_SIZE_RANGE))
        macro_rows = random.choice(tuple(MACRO_ROWS_RANGE))
        macro_cols = random.choice(tuple(MACRO_COLS_RANGE))
        if macro_rows * block_size + (macro_rows - 1) > 30:
            continue
        if macro_cols * block_size + (macro_cols - 1) > 30:
            continue

        sprite, corridor_rows = _make_sprite(block_size)
        channel, open_row, blocked_row = _build_channel_mask(macro_rows, macro_cols)
        grid = _render_input(sprite, channel, block_size)
        starts = _place_starts(grid, block_size, corridor_rows, open_row, blocked_row, channel)
        protected = _protect_visible_sprite_block(channel, block_size, {open_row, blocked_row})
        _add_grey_damage(grid, protected, starts)

        try:
            out = main(grid)
        except Exception:
            continue

        if _visible_mixed_block_count(grid) < 1:
            continue
        if not np.any(out[:, -1] == START):
            continue
        if not _has_partial_flow(out, starts):
            continue
        if len(starts) < 2:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
