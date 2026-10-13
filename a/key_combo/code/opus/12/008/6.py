from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
GEN_ATTEMPTS = 400


def _intervals_from_separators(separators: list[int], length: int) -> list[tuple[int, int]]:
    if not separators:
        return [(0, length)]
    cuts = [-1, *sorted(separators), length]
    intervals = [(cuts[i] + 1, cuts[i + 1]) for i in range(len(cuts) - 1)]
    assert all(start < end for start, end in intervals), "separators must be single non-adjacent lines"
    return intervals


def _split_layers(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    separator_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAGENTA)]
    assert len(separator_rows) == 1, "expected exactly one full-width magenta separator row"
    sep = separator_rows[0]
    upper = grid[:sep, :]
    lower = grid[sep + 1 :, :]
    assert upper.shape == lower.shape and upper.size > 0, "separator must split two equal layers"
    assert not np.any(upper == MAGENTA) and not np.any(lower == MAGENTA), "magenta is reserved for the layer separator"
    return upper, lower


def _block_layout(layer: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_separators = [r for r in range(layer.shape[0]) if np.all(layer[r, :] == BLACK)]
    col_separators = [c for c in range(layer.shape[1]) if np.all(layer[:, c] == BLACK)]
    row_intervals = _intervals_from_separators(row_separators, layer.shape[0])
    col_intervals = _intervals_from_separators(col_separators, layer.shape[1])
    block_heights = {r1 - r0 for r0, r1 in row_intervals}
    block_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert len(block_heights) == 1 and len(block_widths) == 1, "blocks must be equally sized"
    assert block_heights == block_widths, "blocks must be square"
    block_size = next(iter(block_heights))
    return row_intervals, col_intervals, block_size


def _iter_blocks(layer: np.ndarray, row_intervals: list[tuple[int, int]], col_intervals: list[tuple[int, int]]):
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            yield br, bc, layer[r0:r1, c0:c1]


def _infer_sprite(layers: list[np.ndarray], row_intervals: list[tuple[int, int]], col_intervals: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    payload = sorted(set(map(int, np.unique(np.concatenate([layer.ravel() for layer in layers])))) - {BLACK, GREY, MAGENTA})
    assert len(payload) == 2, "expected exactly two payload colors"

    clean_sprite_blocks = []
    for layer in layers:
        for _br, _bc, block in _iter_blocks(layer, row_intervals, col_intervals):
            if np.any(block == GREY):
                continue
            colors = set(map(int, np.unique(block)))
            if colors == set(payload):
                clean_sprite_blocks.append(block.copy())

    assert clean_sprite_blocks, "need at least one clean mixed-color sprite block"
    sprite = clean_sprite_blocks[0]
    assert all(np.array_equal(block, sprite) for block in clean_sprite_blocks[1:]), "all clean sprite blocks must agree"

    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload}
    assert len(set(counts.values())) == 2, "sprite colors must have distinct counts"
    blank_color = min(counts, key=counts.get)
    foreground_color = max(counts, key=counts.get)
    return sprite, foreground_color, blank_color


def _classify_block(block: np.ndarray, sprite: np.ndarray, blank_color: int) -> bool:
    known = block != GREY
    assert known.any(), "each block must expose at least one non-grey cell"
    matches_sprite = bool(np.all(block[known] == sprite[known]))
    matches_blank = bool(np.all(block[known] == blank_color))
    assert matches_sprite != matches_blank, "each block must resolve uniquely after sprite inference"
    return matches_sprite


def _render_from_states(
    states: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    sprite: np.ndarray,
    blank_color: int,
) -> np.ndarray:
    height = row_intervals[-1][1]
    width = col_intervals[-1][1]
    out = np.full((height, width), BLACK, dtype=int)
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            out[r0:r1, c0:c1] = sprite if states[br, bc] else blank_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two separator-stacked block grids, repair wildcarded blocks, and XOR their block types into one layer.

    Input structure:
    - A single full-width magenta row splits the canvas into equal-size upper and lower block grids.
    - Inside each layer, black separator rows and columns partition the layer into equal square blocks.
    - Every block is either one repeated two-color sprite or a uniform fill of the sprite's minority color; some block cells may be grey wildcards.

    Rule:
    - Remove the magenta row and parse the matching upper and lower block layouts.
    - Infer the sprite template from any clean mixed-color blocks, then classify each wildcarded block by comparing its non-grey cells against the sprite or the uniform minority-color blank.
    - For each aligned upper/lower block pair, output the full sprite when exactly one block is a sprite and output the uniform blank when the pair has the same type; keep the black block separators and drop the magenta row.

    Key insight:
    - Grey cells erase local evidence, so the puzzle must be solved in layers: first recover the shared sprite template, then repair block identities, then apply a block-wise XOR between the two layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    upper, lower = _split_layers(g)
    row_intervals, col_intervals, _block_size = _block_layout(upper)
    lower_rows, lower_cols, _ = _block_layout(lower)
    assert row_intervals == lower_rows and col_intervals == lower_cols, "upper and lower layers must share one block layout"

    sprite, _foreground_color, blank_color = _infer_sprite([upper, lower], row_intervals, col_intervals)

    upper_states = np.zeros((len(row_intervals), len(col_intervals)), dtype=bool)
    lower_states = np.zeros_like(upper_states)
    for br, bc, block in _iter_blocks(upper, row_intervals, col_intervals):
        upper_states[br, bc] = _classify_block(block, sprite, blank_color)
    for br, bc, block in _iter_blocks(lower, row_intervals, col_intervals):
        lower_states[br, bc] = _classify_block(block, sprite, blank_color)

    return _render_from_states(upper_states ^ lower_states, row_intervals, col_intervals, sprite, blank_color)


def _has_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _make_sprite(block_size: int, foreground: int, blank_color: int) -> np.ndarray | None:
    area = block_size * block_size
    fg_min = max(block_size + 1, math.ceil(area * 0.55))
    fg_max = min(area - 2, math.floor(area * 0.72))
    if fg_min > fg_max:
        return None

    for _ in range(120):
        fg_count = random.randint(fg_min, fg_max)
        indices = random.sample(range(area), fg_count)
        mask = np.zeros((block_size, block_size), dtype=bool)
        for idx in indices:
            mask[idx // block_size, idx % block_size] = True
        if any(np.all(mask[r, :]) or not np.any(mask[r, :]) for r in range(block_size)):
            continue
        if any(np.all(mask[:, c]) or not np.any(mask[:, c]) for c in range(block_size)):
            continue
        if _has_symmetry(mask):
            continue
        return np.where(mask, foreground, blank_color).astype(int)
    return None


def _sample_states(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = shape
    count = rows * cols
    for _ in range(120):
        upper = np.array([random.random() < 0.5 for _ in range(count)], dtype=bool).reshape(shape)
        lower = np.array([random.random() < 0.5 for _ in range(count)], dtype=bool).reshape(shape)
        if upper.all() or (~upper).all() or lower.all() or (~lower).all():
            continue
        xor = upper ^ lower
        if xor.all() or (~xor).all():
            continue
        return upper, lower
    return None


def _render_layer(
    states: np.ndarray,
    sprite: np.ndarray,
    blank_color: int,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> np.ndarray:
    return _render_from_states(states, row_intervals, col_intervals, sprite, blank_color)


def _majority_positions(sprite: np.ndarray, foreground_color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(sprite == foreground_color)]


def _dirty_block(
    clean_block: np.ndarray,
    sprite: np.ndarray,
    foreground_positions: list[tuple[int, int]],
    is_sprite: bool,
) -> np.ndarray:
    block = clean_block.copy()
    area = block.shape[0] * block.shape[1]
    keep_count = random.randint(1, min(area - 1, 4))
    must_keep = random.choice(foreground_positions)
    all_positions = [(r, c) for r in range(block.shape[0]) for c in range(block.shape[1])]
    remaining = [pos for pos in all_positions if pos != must_keep]
    keep = {must_keep}
    extra_count = min(len(remaining), keep_count - 1)
    if extra_count > 0:
        keep.update(random.sample(remaining, extra_count))
    for r, c in all_positions:
        if (r, c) not in keep:
            block[r, c] = GREY
    if not is_sprite:
        assert any(sprite[r, c] != clean_block[r, c] for r, c in keep), "dirty blank must visibly disagree with the sprite"
    return block


def _case_layout(block_rows: int, block_cols: int, block_size: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_intervals = [(r * (block_size + 1), r * (block_size + 1) + block_size) for r in range(block_rows)]
    col_intervals = [(c * (block_size + 1), c * (block_size + 1) + block_size) for c in range(block_cols)]
    return row_intervals, col_intervals


def _assemble_input(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    separator = np.full((1, upper.shape[1]), MAGENTA, dtype=int)
    return np.vstack([upper, separator, lower])


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    block_rows = random.randint(2, 3)
    block_cols = random.randint(2, 5)
    block_size = random.randint(3, 4)
    row_intervals, col_intervals = _case_layout(block_rows, block_cols, block_size)

    payload = random.sample(PALETTE, 2)
    foreground_color, blank_color = payload[0], payload[1]
    sprite = _make_sprite(block_size, foreground_color, blank_color)
    assert sprite is not None

    sampled_states = _sample_states((block_rows, block_cols))
    assert sampled_states is not None
    upper_states, lower_states = sampled_states

    upper = _render_layer(upper_states, sprite, blank_color, row_intervals, col_intervals)
    lower = _render_layer(lower_states, sprite, blank_color, row_intervals, col_intervals)

    sprite_positions = [(layer, br, bc) for layer, states in enumerate((upper_states, lower_states)) for br, bc in np.argwhere(states)]
    blank_positions = [(layer, br, bc) for layer, states in enumerate((upper_states, lower_states)) for br, bc in np.argwhere(~states)]
    clean_sprite = random.choice(sprite_positions)
    dirty_sprite_choices = [pos for pos in sprite_positions if pos != clean_sprite]
    dirty_sprite = random.choice(dirty_sprite_choices)
    dirty_blank = random.choice(blank_positions)

    dirty_blocks = {dirty_sprite, dirty_blank}
    all_positions = [(layer, br, bc) for layer in (0, 1) for br in range(block_rows) for bc in range(block_cols)]
    eligible_extra = [pos for pos in all_positions if pos != clean_sprite and pos not in dirty_blocks]
    extra_dirty = random.sample(eligible_extra, random.randint(0, min(len(eligible_extra), max(2, len(all_positions) // 2))))
    dirty_blocks.update(extra_dirty)

    foreground_positions = _majority_positions(sprite, foreground_color)
    layers = [upper, lower]
    layer_states = [upper_states, lower_states]

    for layer, br, bc in dirty_blocks:
        r0, r1 = row_intervals[br]
        c0, c1 = col_intervals[bc]
        clean_block = layers[layer][r0:r1, c0:c1].copy()
        layers[layer][r0:r1, c0:c1] = _dirty_block(clean_block, sprite, foreground_positions, bool(layer_states[layer][br, bc]))

    inp = _assemble_input(layers[0], layers[1])
    expected = _render_layer(upper_states ^ lower_states, sprite, blank_color, row_intervals, col_intervals)
    meta = {
        "sprite": sprite,
        "foreground_color": foreground_color,
        "blank_color": blank_color,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "upper_states": upper_states,
        "lower_states": lower_states,
        "dirty_blocks": dirty_blocks,
        "clean_sprite": clean_sprite,
        "input": inp,
        "expected": expected,
    }
    return inp, expected, meta


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not meta["dirty_blocks"]:
        return False

    upper, lower = _split_layers(inp)
    row_intervals, col_intervals, _ = _block_layout(upper)
    if row_intervals != meta["row_intervals"] or col_intervals != meta["col_intervals"]:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    dirty_sprite_seen = False
    dirty_blank_seen = False
    for layer_idx, layer in enumerate((upper, lower)):
        states = meta["upper_states"] if layer_idx == 0 else meta["lower_states"]
        for br, bc, block in _iter_blocks(layer, row_intervals, col_intervals):
            is_sprite = bool(states[br, bc])
            if np.any(block == GREY):
                resolved = _classify_block(block, meta["sprite"], meta["blank_color"])
                if resolved != is_sprite:
                    return False
                dirty_sprite_seen |= resolved
                dirty_blank_seen |= not resolved
    if not dirty_sprite_seen or not dirty_blank_seen:
        return False

    xor = meta["upper_states"] ^ meta["lower_states"]
    if xor.all() or (~xor).all():
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected, meta = _generate_case()
        if _validate_case(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
