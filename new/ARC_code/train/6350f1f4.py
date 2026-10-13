from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, components_mask
import math
import random


ERASED = Color.GREY
SEPARATOR = Color.BLACK


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({SEPARATOR, ERASED})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected at least one separator line"
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty blocks"
    assert len(set(sizes)) == 1, "all encoded blocks must have the same size"
    return intervals


def _find_sprite(
    g: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    payload_colors: list[int],
) -> np.ndarray:
    payload = set(payload_colors)
    sprites = []

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            block_colors = set(colors_present(block, exclude={ERASED}))
            assert SEPARATOR not in block_colors, "block interiors cannot contain separator color"
            if ERASED not in set(map(int, block.flat)) and block_colors == payload:
                sprites.append(block.copy())

    assert sprites, "expected at least one fully visible sprite block"
    sprite = sprites[0]
    for candidate in sprites:
        assert np.array_equal(candidate, sprite), "fully visible sprite blocks must match"
    return sprite


def _foreground_and_blank(sprite: np.ndarray, payload_colors: list[int]) -> tuple[int, int]:
    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload_colors}
    assert counts[payload_colors[0]] != counts[payload_colors[1]], "sprite colors must have a majority/minority split"
    foreground = max(payload_colors, key=lambda color: counts[color])
    blank_colors = [color for color in payload_colors if color != foreground]
    assert len(blank_colors) == 1
    return int(foreground), int(blank_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a block-grid image by swapping each sprite-coded cell with a solid cell and each blank cell with the sprite.

    Input structure
    The grid is partitioned by black separator rows and columns into equal-size
    square blocks. Each block is either a copy of one two-color sprite or a
    uniform blank block of the sprite's minority color; grey pixels are erased
    cells that act as wildcards.

    Rule
    Infer the repeated sprite from fully visible mixed-color blocks. Replace a
    block matching that sprite, ignoring grey erasures, with a solid block of
    the sprite's majority color. Replace a uniform blank block with the sprite.
    Separator rows and columns, including grey erasures on them, become black.

    Key insight
    The small sprite is a block-level code for one macro-cell state, and the
    uniform minority-color block is the other state; the output renders the
    opposite code in each macro-cell while cleaning erased pixels.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    row_separators = [r for r in range(H) if _is_separator_line(g[r, :])]
    col_separators = [c for c in range(W) if _is_separator_line(g[:, c])]
    row_intervals = _regular_intervals(row_separators, H)
    col_intervals = _regular_intervals(col_separators, W)
    block_heights = {r1 - r0 for r0, r1 in row_intervals}
    block_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert block_heights == block_widths, "encoded blocks must be square"

    payload_colors = colors_present(g, exclude={SEPARATOR, ERASED})
    assert len(payload_colors) == 2, "expected exactly two non-separator payload colors"
    sprite = _find_sprite(g, row_intervals, col_intervals, payload_colors)
    foreground, blank = _foreground_and_blank(sprite, payload_colors)

    out = np.zeros_like(g)
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            known = block != ERASED
            assert known.any(), "each block must expose at least one non-grey cell"
            matches_sprite = bool(np.all(block[known] == sprite[known]))
            matches_blank = bool(np.all(block[known] == blank))
            assert matches_sprite != matches_blank, "each block must unambiguously match the sprite or blank state"
            out[r0:r1, c0:c1] = foreground if matches_sprite else sprite

    return out

# --------------- generator (att2) ---------------

# Canvas and palette bounds.
GRID_COUNT_RANGE, BLOCK_SIZE_RANGE = (2, 4), (2, 4)
PAYLOAD_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAYLOAD_COLOR_COUNT = 2
GEN_SEPARATOR_COLOR, GEN_ERASED_COLOR = int(SEPARATOR), int(ERASED)

# Sprite composition constraints.
SPRITE_FG_MIN_RATIO, SPRITE_FG_MAX_RATIO = (2, 3), (3, 4)
SPRITE_ATTEMPTS, TINY_BLOCK_SIZE = 80, 2

# Macro-cell state constraints.
MIN_BLANK_BLOCKS, MAX_BLANK_BLOCK_DENOMINATOR = 1, 2

# Grey overlay constraints.
GREY_MIN_FRACTION, GREY_MAX_FRACTION = 0.05, 0.25
GREY_TWO_COMPONENT_PROBABILITY, GREY_TWO_COMPONENT_ATTEMPTS = 0.5, 30
GREY_CONNECTIVITY, GENERATION_ATTEMPTS = 4, 300

NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _separator_indices(n: int, block_size: int) -> set[int]:
    return {(i + 1) * block_size + i for i in range(n - 1)}


def _block_origin(br: int, bc: int, block_size: int) -> tuple[int, int]:
    return br * (block_size + 1), bc * (block_size + 1)


def _has_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _make_sprite(block_size: int, foreground: int, blank: int) -> np.ndarray | None:
    area = block_size * block_size
    min_num, min_den = SPRITE_FG_MIN_RATIO
    max_num, max_den = SPRITE_FG_MAX_RATIO
    fg_min = math.ceil(area * min_num / min_den)
    fg_max = math.floor(area * max_num / max_den)
    if fg_min > fg_max:
        fg_min = fg_max = round(area * min_num / min_den)
    fg_count = random.randint(fg_min, fg_max)
    blank_count = area - fg_count

    for _ in range(SPRITE_ATTEMPTS):
        mask = np.ones((block_size, block_size), dtype=bool)
        if block_size == TINY_BLOCK_SIZE:
            blanks = [random.randrange(area)]
        else:
            cols = list(range(block_size))
            random.shuffle(cols)
            blanks = [r * block_size + cols[r] for r in range(block_size)]
            extra = blank_count - block_size
            if extra > 0:
                pool = [i for i in range(area) if i not in blanks]
                blanks.extend(random.sample(pool, extra))

        for idx in blanks:
            mask[idx // block_size, idx % block_size] = False

        if block_size > TINY_BLOCK_SIZE:
            row_mixed = all(0 < np.count_nonzero(mask[r]) < block_size for r in range(block_size))
            col_mixed = all(0 < np.count_nonzero(mask[:, c]) < block_size for c in range(block_size))
            if not row_mixed or not col_mixed:
                continue
        if _has_symmetry(mask):
            continue

        sprite = np.where(mask, foreground, blank).astype(int)
        return sprite
    return None


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBOR_OFFSETS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _grow_blob(allowed: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    comps = [
        [tuple(map(int, rc)) for rc in coords]
        for coords in components_mask(allowed, connectivity=GREY_CONNECTIVITY)
        if len(coords) >= size
    ]
    if not comps:
        return None

    comp = random.choice(comps)
    comp_set = set(comp)
    seed = random.choice(comp)
    blob = {seed}
    frontier = {nb for nb in _neighbors4(*seed, *allowed.shape) if nb in comp_set}

    while len(blob) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for nb in _neighbors4(*cell, *allowed.shape):
            if nb in comp_set and nb not in blob:
                frontier.add(nb)

    return blob


def _make_grey_mask(allowed: np.ndarray, target: int) -> np.ndarray | None:
    if target <= 0:
        return np.zeros_like(allowed, dtype=bool)

    want_two = target >= 2 and random.random() < GREY_TWO_COMPONENT_PROBABILITY
    if want_two:
        for _ in range(GREY_TWO_COMPONENT_ATTEMPTS):
            first_size = random.randint(1, target - 1)
            first = _grow_blob(allowed, first_size)
            if first is None:
                continue

            blocked = set(first)
            for r, c in tuple(first):
                blocked.update(_neighbors4(r, c, *allowed.shape))

            allowed_second = allowed.copy()
            for r, c in blocked:
                allowed_second[r, c] = False

            second = _grow_blob(allowed_second, target - first_size)
            if second is None:
                continue

            mask = np.zeros_like(allowed, dtype=bool)
            for r, c in first | second:
                mask[r, c] = True
            return mask

    one = _grow_blob(allowed, target)
    if one is None:
        return None

    mask = np.zeros_like(allowed, dtype=bool)
    for r, c in one:
        mask[r, c] = True
    return mask


def _sample_block_states(n: int) -> tuple[np.ndarray, int]:
    block_count = n * n
    max_blanks = max(MIN_BLANK_BLOCKS, block_count // MAX_BLANK_BLOCK_DENOMINATOR)
    blank_blocks = random.randint(MIN_BLANK_BLOCKS, max_blanks)
    state = np.ones((n, n), dtype=bool)
    for idx in random.sample(range(block_count), blank_blocks):
        state[idx // n, idx % n] = False
    return state, blank_blocks


def _render_clean_grids(n, block_size, side, state, sprite, foreground, blank) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((side, side), GEN_SEPARATOR_COLOR, dtype=int)
    inp = np.full((side, side), GEN_SEPARATOR_COLOR, dtype=int)
    for br in range(n):
        for bc in range(n):
            r0, c0 = _block_origin(br, bc, block_size)
            if state[br, bc]:
                out[r0 : r0 + block_size, c0 : c0 + block_size] = foreground
                inp[r0 : r0 + block_size, c0 : c0 + block_size] = sprite
            else:
                out[r0 : r0 + block_size, c0 : c0 + block_size] = sprite
                inp[r0 : r0 + block_size, c0 : c0 + block_size] = blank
    return inp, out


def _build_protected_mask(n, block_size, side, clean_block, sprite, foreground) -> np.ndarray:
    """Reserve enough evidence for the solver to infer every block and separator."""
    protected = np.zeros((side, side), dtype=bool)
    r0, c0 = _block_origin(*clean_block, block_size)
    protected[r0 : r0 + block_size, c0 : c0 + block_size] = True

    fg_positions = list(zip(*np.nonzero(sprite == foreground)))
    for br in range(n):
        for bc in range(n):
            rr, cc = random.choice(fg_positions)
            r0, c0 = _block_origin(br, bc, block_size)
            protected[r0 + rr, c0 + cc] = True

    row_seps = _separator_indices(n, block_size)
    col_seps = _separator_indices(n, block_size)
    non_sep_rows = [r for r in range(side) if r not in row_seps]
    non_sep_cols = [c for c in range(side) if c not in col_seps]
    for r in non_sep_rows:
        protected[r, random.choice(non_sep_cols)] = True
    for c in non_sep_cols:
        protected[random.choice(non_sep_rows), c] = True
    return protected


def _add_grey_overlay(inp: np.ndarray, protected: np.ndarray) -> bool:
    allowed = ~protected
    side = inp.shape[0]
    min_grey = max(1, math.ceil(side * side * GREY_MIN_FRACTION))
    max_grey = max(min_grey, math.floor(side * side * GREY_MAX_FRACTION))
    available_sizes = [len(comp) for comp in components_mask(allowed, connectivity=GREY_CONNECTIVITY)]
    largest_available = max(available_sizes, default=0)
    if largest_available < min_grey:
        return False
    target = random.randint(min_grey, min(max_grey, largest_available))
    grey = _make_grey_mask(allowed, target)
    if grey is None:
        return False

    inp[grey] = GEN_ERASED_COLOR
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_COUNT_RANGE)
    block_size = random.randint(*BLOCK_SIZE_RANGE)
    side = n * block_size + (n - 1)

    foreground, blank = random.sample(PAYLOAD_COLORS, PAYLOAD_COLOR_COUNT)
    sprite = _make_sprite(block_size, foreground, blank)
    if sprite is None:
        return None

    state, blank_blocks = _sample_block_states(n)
    clean_sprite_blocks = [(r, c) for r in range(n) for c in range(n) if state[r, c]]
    if not clean_sprite_blocks or blank_blocks == 0:
        return None
    clean_block = random.choice(clean_sprite_blocks)

    inp, out = _render_clean_grids(n, block_size, side, state, sprite, foreground, blank)
    protected = _build_protected_mask(n, block_size, side, clean_block, sprite, foreground)
    if not _add_grey_overlay(inp, protected):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
