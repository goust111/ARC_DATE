from __future__ import annotations

import random

import numpy as np


CONTROL = 6
SEPARATOR = 0
ERASED = 5
BLANK = 8
SPRITE = 9

BLOCK_SIZE_OPTIONS = (3, 4)
GRID_ROW_OPTIONS = (2, 3, 4)
GRID_COL_OPTIONS = (2, 3, 4)
CONTROL_LENGTH_OPTIONS = (2, 3, 4)
GENERATION_ATTEMPTS = 500


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({SEPARATOR, ERASED})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *map(int, separators), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert intervals and min(sizes) > 0
    assert len(set(sizes)) == 1
    return intervals


def _control_length(grid: np.ndarray) -> int:
    row = np.asarray(grid[0], dtype=int)
    n = 0
    while n < row.size and int(row[n]) == CONTROL:
        n += 1
    assert 2 <= n <= 4
    assert np.all(row[n:] != CONTROL)
    return int(n)


def _rise_fall(n: int) -> list[int]:
    return list(range(1, n + 1)) + list(range(n - 1, 0, -1))


def _body_structure(body: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_seps = [r for r in range(body.shape[0]) if _is_separator_line(body[r])]
    col_seps = [c for c in range(body.shape[1]) if _is_separator_line(body[:, c])]
    row_intervals = _regular_intervals(row_seps, body.shape[0])
    col_intervals = _regular_intervals(col_seps, body.shape[1])
    heights = {r1 - r0 for r0, r1 in row_intervals}
    widths = {c1 - c0 for c0, c1 in col_intervals}
    assert heights == widths
    block_size = heights.pop()
    assert block_size in BLOCK_SIZE_OPTIONS
    return row_intervals, col_intervals, int(block_size)


def _blocks_in_order(
    body: np.ndarray, row_intervals: list[tuple[int, int]], col_intervals: list[tuple[int, int]]
) -> list[np.ndarray]:
    return [body[r0:r1, c0:c1] for r0, r1 in row_intervals for c0, c1 in col_intervals]


def _infer_sprite_template(blocks: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    clean_sprite_ids = []
    clean_sprites = []
    for idx, block in enumerate(blocks):
        colors = set(map(int, np.unique(block)))
        if ERASED in colors:
            continue
        if colors == {BLANK, SPRITE}:
            clean_sprites.append(block.copy())
            clean_sprite_ids.append(idx)

    assert len(clean_sprites) >= 2
    stack = np.stack(clean_sprites, axis=0)
    maroon_votes = np.count_nonzero(stack == SPRITE, axis=0)
    template = np.where(maroon_votes * 2 >= stack.shape[0], SPRITE, BLANK).astype(int)
    assert set(map(int, np.unique(template))) == {BLANK, SPRITE}
    for block in clean_sprites:
        assert np.array_equal(block, template)
    return template, clean_sprite_ids


def _classify_block(block: np.ndarray, template: np.ndarray) -> bool:
    known = block != ERASED
    assert known.any()
    matches_sprite = bool(np.all(block[known] == template[known]))
    matches_blank = bool(np.all(block[known] == BLANK))
    assert matches_sprite != matches_blank
    return matches_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top control bar into a rise-fall parity prefix, then flip matching blocks between a shared sprite and a solid blank.

    Input structure:
    - Row 0 contains a left-aligned magenta control bar whose length n is 2-4.
    - Rows 1.. form an equal-size square block grid separated by single black rows and columns.
    - Each block is either one repeated 8/9 sprite or a solid cyan block; grey cells are wildcard noise and may also appear on separators.

    Rule:
    - Read n from the control bar and build the sequence [1, 2, ..., n, n-1, ..., 1].
    - Infer the shared sprite template from grey-free mixed-color blocks.
    - Scan blocks in row-major order; for the first 2n-1 blocks, flip blocks whose paired sequence value is odd and keep blocks whose value is even.
    - Flipping means sprite -> solid cyan blank and solid cyan blank -> sprite template; later blocks stay unchanged.
    - Remove the control row in the output and clean every separator/noise cell back to black.

    Key insight:
    - The magenta bar does not directly mark blocks: it only gives n, which must be expanded into a rise-fall sequence whose parity decides which block states to invert.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    n = _control_length(g)
    body = g[1:]
    row_intervals, col_intervals, _block_size = _body_structure(body)
    blocks = _blocks_in_order(body, row_intervals, col_intervals)
    template, _ = _infer_sprite_template(blocks)
    sequence = _rise_fall(n)

    out = np.zeros_like(body)
    for idx, ((r0, r1), (c0, c1)) in enumerate(
        [(rs, cs) for rs in row_intervals for cs in col_intervals]
    ):
        is_sprite = _classify_block(body[r0:r1, c0:c1], template)
        if idx < len(sequence) and sequence[idx] % 2 == 1:
            is_sprite = not is_sprite
        out[r0:r1, c0:c1] = template if is_sprite else BLANK

    return out


def _shape_for_layout(block_rows: int, block_cols: int, block_size: int) -> tuple[int, int]:
    return (
        block_rows * block_size + (block_rows - 1),
        block_cols * block_size + (block_cols - 1),
    )


def _sample_layout() -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for n in CONTROL_LENGTH_OPTIONS:
        prefix_len = 2 * n - 1
        for block_size in BLOCK_SIZE_OPTIONS:
            for block_rows in GRID_ROW_OPTIONS:
                for block_cols in GRID_COL_OPTIONS:
                    total = block_rows * block_cols
                    body_h, body_w = _shape_for_layout(block_rows, block_cols, block_size)
                    if total <= prefix_len:
                        continue
                    if max(body_h + 1, body_w) > 30:
                        continue
                    candidates.append((n, block_rows, block_cols, block_size))
    assert candidates
    return random.choice(candidates)


def _sample_template(block_size: int) -> np.ndarray:
    area = block_size * block_size
    min_maroon = area // 2
    max_maroon = area - block_size

    for _ in range(200):
        maroon_count = random.randint(min_maroon, max_maroon)
        mask = np.zeros((block_size, block_size), dtype=bool)
        coords = [(r, c) for r in range(block_size) for c in range(block_size)]
        for r, c in random.sample(coords, maroon_count):
            mask[r, c] = True
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if np.any(mask.sum(axis=0) == block_size) or np.any(mask.sum(axis=1) == block_size):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return np.where(mask, SPRITE, BLANK).astype(int)
    raise RuntimeError("failed to sample an asymmetric sprite template")


def _sample_states(total_blocks: int, n: int) -> list[bool]:
    prefix_len = 2 * n - 1
    flip_positions = list(range(0, prefix_len, 2))
    stay_positions = [idx for idx in range(total_blocks) if idx not in flip_positions]

    for _ in range(300):
        states = [bool(random.getrandbits(1)) for _ in range(total_blocks)]
        sprite_count = sum(states)
        blank_count = total_blocks - sprite_count
        if sprite_count < 2 or blank_count < 2:
            continue
        flip_values = {states[idx] for idx in flip_positions}
        stay_values = {states[idx] for idx in stay_positions}
        if len(flip_positions) >= 2 and len(flip_values) != 2:
            continue
        if len(stay_positions) >= 2 and len(stay_values) != 2:
            continue
        return states
    raise RuntimeError("failed to sample a non-degenerate block state map")


def _block_origin(index: int, block_cols: int, block_size: int) -> tuple[int, int]:
    br, bc = divmod(index, block_cols)
    stride = block_size + 1
    return br * stride, bc * stride


def _render_body(
    block_rows: int, block_cols: int, block_size: int, template: np.ndarray, states: list[bool]
) -> np.ndarray:
    body_h, body_w = _shape_for_layout(block_rows, block_cols, block_size)
    body = np.zeros((body_h, body_w), dtype=int)
    for idx, is_sprite in enumerate(states):
        r0, c0 = _block_origin(idx, block_cols, block_size)
        body[r0 : r0 + block_size, c0 : c0 + block_size] = template if is_sprite else BLANK
    return body


def _separator_coords(block_rows: int, block_cols: int, block_size: int) -> list[tuple[int, int]]:
    body_h, body_w = _shape_for_layout(block_rows, block_cols, block_size)
    coords: list[tuple[int, int]] = []
    for r in range(body_h):
        for c in range(body_w):
            if r % (block_size + 1) == block_size or c % (block_size + 1) == block_size:
                coords.append((r, c))
    return coords


def _pick_block_noise_indices(states: list[bool], n: int) -> tuple[set[int], list[int]]:
    total_blocks = len(states)
    sprite_indices = [idx for idx, is_sprite in enumerate(states) if is_sprite]
    if len(sprite_indices) == 2:
        clean_count = 2
    else:
        clean_count = random.randint(2, min(3, len(sprite_indices) - 1))
    clean_sprite_ids = set(random.sample(sprite_indices, clean_count))

    noisy_blocks: list[int] = []
    noisy_sprite_pool = [idx for idx in sprite_indices if idx not in clean_sprite_ids]
    noisy_blank_pool = [idx for idx, is_sprite in enumerate(states) if not is_sprite]

    prefix = list(range(2 * n - 1))
    if noisy_sprite_pool:
        preferred = [idx for idx in noisy_sprite_pool if idx in prefix]
        noisy_blocks.append(random.choice(preferred or noisy_sprite_pool))
    if noisy_blank_pool:
        preferred = [idx for idx in noisy_blank_pool if idx in prefix]
        pick = random.choice(preferred or noisy_blank_pool)
        if pick not in noisy_blocks:
            noisy_blocks.append(pick)

    remainder = [
        idx
        for idx in range(total_blocks)
        if idx not in clean_sprite_ids and idx not in noisy_blocks
    ]
    if not noisy_blocks:
        noisy_blocks.append(random.choice(remainder or list(range(total_blocks))))
        if noisy_blocks[0] in clean_sprite_ids:
            clean_sprite_ids.remove(noisy_blocks[0])
    remainder = [idx for idx in remainder if idx not in noisy_blocks]
    extra = random.randint(0, min(2, len(remainder)))
    noisy_blocks.extend(random.sample(remainder, extra))
    assert noisy_blocks
    return clean_sprite_ids, noisy_blocks


def _apply_block_noise(
    grid: np.ndarray,
    block_index: int,
    block_cols: int,
    block_size: int,
    template: np.ndarray,
) -> None:
    r0, c0 = _block_origin(block_index, block_cols, block_size)
    block = grid[1 + r0 : 1 + r0 + block_size, c0 : c0 + block_size]
    maroon_cells = [tuple(map(int, rc)) for rc in np.argwhere(template == SPRITE)]
    protected = {random.choice(maroon_cells)}
    visible_budget = 2 if block_size == 3 else 3
    remaining = [(r, c) for r in range(block_size) for c in range(block_size) if (r, c) not in protected]
    extra_keep = random.sample(remaining, k=min(visible_budget - 1, len(remaining)))
    protected.update(extra_keep)

    candidates = [(r, c) for r in range(block_size) for c in range(block_size) if (r, c) not in protected]
    grey_count = random.randint(1, min(block_size, len(candidates)))
    for rr, cc in random.sample(candidates, grey_count):
        block[rr, cc] = ERASED


def _apply_separator_noise(
    grid: np.ndarray, block_rows: int, block_cols: int, block_size: int
) -> None:
    separator_cells = _separator_coords(block_rows, block_cols, block_size)
    if not separator_cells:
        return
    grey_count = random.randint(1, min(block_size + 1, len(separator_cells)))
    for rr, cc in random.sample(separator_cells, grey_count):
        grid[1 + rr, cc] = ERASED


def _expected_output_from_states(
    block_rows: int,
    block_cols: int,
    block_size: int,
    template: np.ndarray,
    states: list[bool],
    n: int,
) -> np.ndarray:
    out_states = states[:]
    for idx, value in enumerate(_rise_fall(n)):
        if value % 2 == 1:
            out_states[idx] = not out_states[idx]
    return _render_body(block_rows, block_cols, block_size, template, out_states)


def _validate_generated(
    grid: np.ndarray,
    block_rows: int,
    block_cols: int,
    block_size: int,
    template: np.ndarray,
    source_states: list[bool],
    clean_sprite_ids: set[int],
    noisy_blocks: list[int],
) -> None:
    n = _control_length(grid)
    prefix_len = 2 * n - 1
    flip_positions = list(range(0, prefix_len, 2))
    stay_positions = [idx for idx in range(len(source_states)) if idx not in flip_positions]

    body = grid[1:]
    row_intervals, col_intervals, parsed_block_size = _body_structure(body)
    assert (len(row_intervals), len(col_intervals), parsed_block_size) == (
        block_rows,
        block_cols,
        block_size,
    )
    blocks = _blocks_in_order(body, row_intervals, col_intervals)
    inferred_template, inferred_clean = _infer_sprite_template(blocks)
    assert np.array_equal(inferred_template, template)
    assert set(clean_sprite_ids).issubset(inferred_clean)

    inferred_states = [_classify_block(block, template) for block in blocks]
    assert inferred_states == source_states

    changed = [source_states[idx] for idx in flip_positions]
    assert len(set(changed)) == 2
    if len(stay_positions) >= 2:
        assert len({source_states[idx] for idx in stay_positions}) == 2
    assert len(clean_sprite_ids) >= 2
    assert noisy_blocks
    assert any(ERASED in blocks[idx] for idx in noisy_blocks)
    assert sum(ERASED in block for block in blocks) >= 1
    assert np.count_nonzero(body == ERASED) >= len(noisy_blocks)
    assert len(source_states) > prefix_len

    expected = _expected_output_from_states(
        block_rows, block_cols, block_size, template, source_states, n
    )
    actual = main(grid)
    assert np.array_equal(actual, expected)

    out_blocks = _blocks_in_order(actual, row_intervals, col_intervals)
    for idx, block in enumerate(out_blocks):
        is_sprite = bool(np.array_equal(block, template))
        if idx in flip_positions:
            assert is_sprite != source_states[idx]
        else:
            assert is_sprite == source_states[idx]
    assert np.all(actual[np.where(body == ERASED)] != ERASED)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        n, block_rows, block_cols, block_size = _sample_layout()
        total_blocks = block_rows * block_cols
        template = _sample_template(block_size)
        states = _sample_states(total_blocks, n)

        body = _render_body(block_rows, block_cols, block_size, template, states)
        control_row = np.zeros((1, body.shape[1]), dtype=int)
        control_row[0, :n] = CONTROL
        grid = np.vstack([control_row, body]).astype(int)

        clean_sprite_ids, noisy_blocks = _pick_block_noise_indices(states, n)
        for idx in noisy_blocks:
            _apply_block_noise(grid, idx, block_cols, block_size, template)
        if random.random() < 0.7:
            _apply_separator_noise(grid, block_rows, block_cols, block_size)

        _validate_generated(
            grid,
            block_rows,
            block_cols,
            block_size,
            template,
            states,
            clean_sprite_ids,
            noisy_blocks,
        )
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
