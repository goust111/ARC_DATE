from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Toolbox-style helper: most common color in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.unique(grid)))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, np.unique(line))).issubset({BLACK, GREY})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [b - a for a, b in spans]
    assert spans and min(sizes) > 0
    assert len(set(sizes)) == 1
    return spans


def _grid_structure(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    row_seps = [r for r in range(h) if _is_separator_line(g[r, :])]
    col_seps = [c for c in range(w) if _is_separator_line(g[:, c])]
    row_spans = _regular_intervals(row_seps, h)
    col_spans = _regular_intervals(col_seps, w)
    assert len(row_spans) == 3 and len(col_spans) == 3
    block_h = {r1 - r0 for r0, r1 in row_spans}
    block_w = {c1 - c0 for c0, c1 in col_spans}
    assert block_h == block_w
    return row_spans, col_spans


def _infer_sprite(g: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    payload = colors_present(g, exclude={BLACK, GREY})
    assert len(payload) == 2
    sprite = np.full((row_spans[0][1] - row_spans[0][0], col_spans[0][1] - col_spans[0][0]), -1, dtype=int)

    mixed_blocks = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            block = g[r0:r1, c0:c1]
            visible = block != GREY
            visible_colors = set(map(int, np.unique(block[visible])))
            if visible_colors == set(payload):
                mixed_blocks += 1
                for rr, cc in np.argwhere(visible):
                    rr = int(rr)
                    cc = int(cc)
                    val = int(block[rr, cc])
                    if int(sprite[rr, cc]) == -1:
                        sprite[rr, cc] = val
                    else:
                        assert int(sprite[rr, cc]) == val

    assert mixed_blocks >= 1
    assert np.all(sprite >= 0)

    a, b = payload
    count_a = int(np.count_nonzero(sprite == a))
    count_b = int(np.count_nonzero(sprite == b))
    assert count_a != count_b
    majority = int(a if count_a > count_b else b)
    minority = int(b if majority == a else a)
    return sprite, majority, minority


def _classify_block(block: np.ndarray, sprite: np.ndarray, minority: int) -> str:
    visible = block != GREY
    assert visible.any()
    matches_sprite = bool(np.all(block[visible] == sprite[visible]))
    matches_blank = bool(np.all(block[visible] == int(minority)))
    assert matches_sprite != matches_blank
    return "sprite" if matches_sprite else "blank"


def _ring_block(size: int, majority: int, minority: int) -> np.ndarray:
    out = np.full((size, size), int(majority), dtype=int)
    if size >= 3:
        out[1:-1, 1:-1] = int(minority)
    return out


def _render_output_from_state(
    states: np.ndarray,
    sprite: np.ndarray,
    majority: int,
    minority: int,
) -> np.ndarray:
    block_size = int(sprite.shape[0])
    out = np.zeros((block_size * 3 + 2, block_size * 3 + 2), dtype=int)
    ring = _ring_block(block_size, majority, minority)
    for br in range(3):
        for bc in range(3):
            r0 = br * (block_size + 1)
            c0 = bc * (block_size + 1)
            patch = ring if states[br, bc] else sprite
            out[r0 : r0 + block_size, c0 : c0 + block_size] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse a sprite-vs-blank block code, then separate the interiors of the newly solid blocks.

    Input structure:
    - Black separator rows and columns split the grid into a 3x3 array of equal square blocks.
    - Each block is either one repeated two-color sprite or a uniform block of the sprite's minority color.
    - Up to a few grey pixels may erase cells inside sprite blocks or on separator lines; grey acts as a wildcard.

    Rule:
    - Infer the repeated sprite from the mixed-color blocks, treating grey cells as unknown.
    - Determine which sprite color is the majority color and which is the minority color.
    - Reverse every block's code: sprite blocks become solid majority-color squares, while uniform minority-color blocks become the sprite.
    - Clean every separator cell to black, then recolor each newly solid block so its one-cell border stays majority-colored and only its strict interior becomes the minority color.

    Key insight:
    - The minority color must be inferred from the repeated sprite first, because that same color identifies the blank-code blocks and later becomes the interior fill of the inverted solid blocks.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _grid_structure(g)
    sprite, majority, minority = _infer_sprite(g, row_spans, col_spans)
    block_size = int(sprite.shape[0])
    ring = _ring_block(block_size, majority, minority)

    out = np.zeros_like(g)
    for br, (r0, r1) in enumerate(row_spans):
        for bc, (c0, c1) in enumerate(col_spans):
            block = g[r0:r1, c0:c1]
            kind = _classify_block(block, sprite, minority)
            out[r0:r1, c0:c1] = ring if kind == "sprite" else sprite
    return out


def _make_sprite(block_size: int, majority: int, minority: int) -> np.ndarray | None:
    area = block_size * block_size
    min_minority = max((area + 4) // 5, block_size)
    max_minority = (area - 1) // 2
    if min_minority > max_minority:
        return None

    ring_mask = np.zeros((block_size, block_size), dtype=bool)
    ring_mask[1:-1, 1:-1] = True

    for _ in range(200):
        target = random.randint(min_minority, max_minority)
        mask = np.zeros((block_size, block_size), dtype=bool)

        cols = list(range(block_size))
        random.shuffle(cols)
        for r, c in enumerate(cols):
            mask[r, c] = True

        remaining = target - block_size
        if remaining > 0:
            pool = [(r, c) for r in range(block_size) for c in range(block_size) if not mask[r, c]]
            extra = random.sample(pool, remaining)
            for r, c in extra:
                mask[r, c] = True

        if any(mask[r].all() for r in range(block_size)):
            continue
        if any(mask[:, c].all() for c in range(block_size)):
            continue
        if not mask[1:-1, 1:-1].any():
            continue
        border = mask.copy()
        border[1:-1, 1:-1] = False
        if not border.any():
            continue
        if np.array_equal(mask, ring_mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue

        sprite = np.full((block_size, block_size), int(majority), dtype=int)
        sprite[mask] = int(minority)
        return sprite
    return None


def _choose_states() -> np.ndarray:
    sprite_count = random.randint(2, 7)
    flat = [True] * sprite_count + [False] * (9 - sprite_count)
    random.shuffle(flat)
    states = np.array(flat, dtype=bool).reshape(3, 3)
    assert 2 <= int(states.sum()) <= 7
    return states


def _place_greys(
    grid: np.ndarray,
    states: np.ndarray,
    sprite: np.ndarray,
    block_size: int,
    majority: int,
    minority: int,
) -> None:
    grey_count = random.choice([0, 1, 1, 2, 2])
    if grey_count == 0:
        return

    sprite_blocks = [(br, bc) for br in range(3) for bc in range(3) if bool(states[br, bc])]
    separator_cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r % (block_size + 1) == block_size) or (c % (block_size + 1) == block_size)
    ]

    random.shuffle(separator_cells)
    used: set[tuple[int, int]] = set()
    touched_sprite_blocks: set[tuple[int, int]] = set()

    for _ in range(grey_count):
        options: list[tuple[int, int, tuple[int, int] | None]] = []
        if separator_cells:
            r, c = separator_cells[-1]
            options.append((r, c, None))

        untouched = [p for p in sprite_blocks if p not in touched_sprite_blocks]
        touched = [p for p in sprite_blocks if p in touched_sprite_blocks]
        if len(untouched) >= 2:
            allowed_blocks = list(sprite_blocks)
        elif len(untouched) == 1:
            allowed_blocks = touched
        else:
            allowed_blocks = list(sprite_blocks)
        random.shuffle(allowed_blocks)
        for br, bc in allowed_blocks:
            majority_cells = [(r, c) for r in range(block_size) for c in range(block_size) if int(sprite[r, c]) == int(majority)]
            # `sprite` is two-color and minority occupies the lexically sampled mask,
            # so the visible-majority guarantee only needs one majority cell left.
            if len(majority_cells) <= 1:
                continue
            rr, cc = random.choice(majority_cells + [(r, c) for r in range(block_size) for c in range(block_size)])
            gr = br * (block_size + 1) + rr
            gc = bc * (block_size + 1) + cc
            options.append((gr, gc, (br, bc)))
            break

        if not options:
            break
        gr, gc, owner = random.choice(options)
        if (gr, gc) in used:
            continue
        used.add((gr, gc))
        grid[gr, gc] = GREY
        if owner is None:
            separator_cells.pop()
        else:
            touched_sprite_blocks.add(owner)


def _expected_from_generator(states: np.ndarray, sprite: np.ndarray, majority: int, minority: int) -> np.ndarray:
    return _render_output_from_state(states, sprite, majority, minority)


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    states: np.ndarray,
    sprite: np.ndarray,
    majority: int,
    minority: int,
) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    block_size = int(sprite.shape[0])
    ring = _ring_block(block_size, majority, minority)
    if np.array_equal(sprite, ring):
        return False

    if int(states.sum()) < 2 or int((~states).sum()) < 2:
        return False

    row_spans, col_spans = _grid_structure(inp)
    inferred_sprite, inferred_majority, inferred_minority = _infer_sprite(inp, row_spans, col_spans)
    if not np.array_equal(inferred_sprite, sprite):
        return False
    if (int(inferred_majority), int(inferred_minority)) != (int(majority), int(minority)):
        return False

    seen_sprite = False
    seen_blank = False
    for br, (r0, r1) in enumerate(row_spans):
        for bc, (c0, c1) in enumerate(col_spans):
            block = inp[r0:r1, c0:c1]
            kind = _classify_block(block, sprite, minority)
            seen_sprite |= kind == "sprite"
            seen_blank |= kind == "blank"
    if not (seen_sprite and seen_blank):
        return False

    no_inner_recolor = np.zeros_like(expected)
    solid = np.full_like(sprite, int(majority))
    for br in range(3):
        for bc in range(3):
            r0 = br * (block_size + 1)
            c0 = bc * (block_size + 1)
            patch = solid if bool(states[br, bc]) else sprite
            no_inner_recolor[r0 : r0 + block_size, c0 : c0 + block_size] = patch
    if np.array_equal(no_inner_recolor, expected):
        return False

    recolor_all_blocks = np.zeros_like(expected)
    for br in range(3):
        for bc in range(3):
            r0 = br * (block_size + 1)
            c0 = bc * (block_size + 1)
            recolor_all_blocks[r0 : r0 + block_size, c0 : c0 + block_size] = ring
    if np.array_equal(recolor_all_blocks, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    block_size = random.randint(3, 6)
    majority, minority = random.sample(PALETTE, 2)
    sprite = _make_sprite(block_size, majority, minority)
    if sprite is None:
        return None

    states = _choose_states()
    grid = np.zeros((block_size * 3 + 2, block_size * 3 + 2), dtype=int)
    blank = np.full((block_size, block_size), int(minority), dtype=int)

    for br in range(3):
        for bc in range(3):
            r0 = br * (block_size + 1)
            c0 = bc * (block_size + 1)
            patch = sprite if bool(states[br, bc]) else blank
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = patch

    _place_greys(grid, states, sprite, block_size, majority, minority)
    expected = _expected_from_generator(states, sprite, majority, minority)
    return grid if _validate_semantics(grid, expected, states, sprite, majority, minority) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
