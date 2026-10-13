from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _consecutive_groups(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(v) for v in indices]
    if not values:
        return []
    groups = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            groups.append((start, end))
            start = end = value
    groups.append((start, end))
    return groups


def _extract_sprite_and_legend(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    occupied_cols = np.flatnonzero(np.any(grid != 0, axis=0))
    col_groups = _consecutive_groups(occupied_cols)
    assert len(col_groups) == 2, "expected exactly two separated non-background column groups"

    (sprite_c0, sprite_c1), (legend_c0, legend_c1) = col_groups
    assert legend_c0 - sprite_c1 >= 3, "sprite and legend must be separated by at least two bg columns"
    assert sprite_c1 - sprite_c0 + 1 == 3, "expected a 3-column sprite footprint"
    assert legend_c1 - legend_c0 + 1 in (1, 2), "legend footprint must be 1 or 2 visible columns"

    sprite_rows = np.flatnonzero(np.any(grid[:, sprite_c0 : sprite_c1 + 1] != 0, axis=1))
    assert len(sprite_rows) > 0, "sprite must contain non-background pixels"
    sprite_row_groups = _consecutive_groups(sprite_rows)
    assert len(sprite_row_groups) == 1, "sprite rows must form one contiguous band"
    sprite_r0, sprite_r1 = sprite_row_groups[0]
    assert sprite_r1 - sprite_r0 + 1 == 3, "expected a 3-row sprite footprint"
    sprite = grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3].copy()

    legend_src_col = legend_c0
    legend_rows = np.flatnonzero(grid[:, legend_src_col] != 0)
    assert len(legend_rows) > 0, "legend source column must be visible"
    legend_row_groups = _consecutive_groups(legend_rows)
    assert len(legend_row_groups) == 1, "legend rows must form one contiguous band"
    legend_r0, legend_r1 = legend_row_groups[0]
    assert 1 <= legend_r1 - legend_r0 + 1 <= 3, "legend height must be 1..3"
    assert np.all(grid[legend_r0 : legend_r1 + 1, legend_src_col] != 0), "legend source cells must be non-background"

    dst_col = legend_src_col + 1
    assert dst_col < grid.shape[1], "legend target column must fit in the grid"
    pairs = [
        (int(grid[r, legend_src_col]), int(grid[r, dst_col]))
        for r in range(legend_r0, legend_r1 + 1)
    ]
    return sprite, pairs


def _apply_chain(sprite: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(sprite, dtype=int, copy=True)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            color = int(out[r, c])
            for src, dst in pairs:
                if color != 0 and color == src:
                    color = dst
            out[r, c] = color
    return out


def _apply_without_chain(sprite: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(sprite, dtype=int, copy=True)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            color = int(out[r, c])
            if color == 0:
                continue
            for src, dst in pairs:
                if color == src:
                    out[r, c] = dst
                    break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the original sprite pixels, chain-recolor the sprite by the legend, then scale the recolored sprite by the frozen count.

    Input structure:
    - A black background contains one separated 3x3 multicolor sprite on the left and one ordered legend on the right.
    - The legend is arranged as 1-3 rows, conceptually two columns wide, with a visible non-black source cell in the left column of every row.
    - Some legend target cells may be black, so parts of the right legend column can visually disappear into the background.

    Rule:
    - Count the sprite's original non-black pixels; this count S is the output scale factor.
    - Read legend rows from top to bottom and apply them as chained substitutions on the sprite: whenever a current non-black pixel color matches a row's left color, replace it with that row's right color and continue scanning lower rows.
    - After all recoloring is finished, enlarge the resulting 3x3 sprite by factor S in both directions, turning each sprite cell into an S x S block.

    Key insight:
    - The scale is frozen before recoloring, while the pixels being enlarged come from the post-legend sprite, so counting after recoloring gives the wrong output size whenever the legend turns some pixels black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprite, pairs = _extract_sprite_and_legend(g)
    scale = int(np.count_nonzero(sprite))
    assert 4 <= scale <= 7, "sprite foreground count must define a valid scale"

    recolored = _apply_chain(sprite, pairs)
    return np.kron(recolored, np.ones((scale, scale), dtype=int))


_MASKS: list[np.ndarray] = []
for bits in range(1 << 9):
    if not 4 <= bits.bit_count() <= 7:
        continue
    mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
        _MASKS.append(mask)


def _random_sprite(mask: np.ndarray) -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    n_colors = random.choice((2, 3))
    colors = random.sample(PALETTE, n_colors)

    random.shuffle(occupied)
    assigned = occupied[:]
    for color, (r, c) in zip(colors, assigned):
        sprite[r, c] = color
    for r, c in assigned[n_colors:]:
        sprite[r, c] = random.choice(colors)
    return sprite


def _build_pairs(sprite: np.ndarray) -> tuple[list[tuple[int, int]], bool]:
    present = sorted({int(v) for v in sprite.flat if int(v) != 0})
    absent = [c for c in PALETTE if c not in present]
    mode = random.choices(
        ["double_chain", "triple_chain", "double_direct"],
        weights=[4, 4, 3],
        k=1,
    )[0]

    if mode == "double_direct":
        vanish = random.choice(present)
        recolor_src = random.choice([c for c in present if c != vanish])
        recolor_dst_pool = [c for c in absent if c != recolor_src]
        recolor_dst = random.choice(recolor_dst_pool or [c for c in present if c != recolor_src])
        return [(vanish, 0), (recolor_src, recolor_dst)], False

    if mode == "double_chain":
        a, b = random.sample(present, 2)
        return [(a, b), (b, 0)], True

    if len(present) >= 3 and random.random() < 0.5:
        a, b, c = random.sample(present, 3)
        recolor_dst = random.choice(absent or [x for x in present if x != c])
        return [(a, b), (b, 0), (c, recolor_dst)], True

    a, b = random.sample(present, 2)
    mid = random.choice(absent)
    recolor_dst = random.choice([c for c in PALETTE if c not in {b, mid}])
    return [(a, mid), (mid, 0), (b, recolor_dst)], True


def _render_input(sprite: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    k = len(pairs)
    height = random.randint(10, 14)
    gap = random.randint(2, 5)
    min_width = gap + 7
    width = random.randint(max(10, min_width), 15)

    sprite_r0 = random.randint(1, height - 4)
    legend_r0 = random.randint(1, height - k - 1)
    max_legend_c0 = width - 3
    sprite_c0_max = max_legend_c0 - gap - 3
    assert sprite_c0_max >= 1
    sprite_c0 = random.randint(1, sprite_c0_max)
    legend_c0 = sprite_c0 + 3 + gap

    grid = np.zeros((height, width), dtype=int)
    grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3] = sprite
    for offset, (src, dst) in enumerate(pairs):
        row = legend_r0 + offset
        grid[row, legend_c0] = src
        grid[row, legend_c0 + 1] = dst
    return grid


def _validate_instance(grid: np.ndarray, pairs: list[tuple[int, int]], chain_expected: bool) -> bool:
    try:
        sprite, parsed_pairs = _extract_sprite_and_legend(grid)
    except AssertionError:
        return False

    if parsed_pairs != pairs:
        return False

    original_count = int(np.count_nonzero(sprite))
    if not 4 <= original_count <= 7:
        return False

    colors = {int(v) for v in sprite.flat if int(v) != 0}
    if len(colors) not in (2, 3):
        return False

    if not any(dst == 0 for _src, dst in pairs):
        return False

    recolored = _apply_chain(sprite, pairs)
    recolored_count = int(np.count_nonzero(recolored))
    if not (1 <= recolored_count < original_count):
        return False

    if np.array_equal(recolored, sprite):
        return False

    if chain_expected and np.array_equal(recolored, _apply_without_chain(sprite, pairs)):
        return False

    expected = np.kron(recolored, np.ones((original_count, original_count), dtype=int))
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False

    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    mask = random.choice(_MASKS)
    sprite = _random_sprite(mask)
    pairs, chain_expected = _build_pairs(sprite)
    grid = _render_input(sprite, pairs)
    if _validate_instance(grid, pairs, chain_expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
