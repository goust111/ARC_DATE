from __future__ import annotations

import math
import random

import numpy as np


SEPARATOR = 0
ERASED = 5
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({SEPARATOR, ERASED})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected at least one separator line"
    bounds = [-1, *map(int, separators), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty blocks"
    assert len(set(sizes)) == 1, "encoded blocks must have equal size"
    return intervals


def _split_panels(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = g.shape
    assert w % 2 == 1, "input must have one center divider column"
    divider = w // 2
    assert _is_separator_line(g[:, divider]), "center divider must be a separator column"
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.shape == right.shape and left.shape[1] > 0
    return left, right


def _panel_layout(panel: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = panel.shape
    row_seps = [r for r in range(h) if _is_separator_line(panel[r, :])]
    col_seps = [c for c in range(w) if _is_separator_line(panel[:, c])]
    row_intervals = _regular_intervals(row_seps, h)
    col_intervals = _regular_intervals(col_seps, w)
    block_heights = {r1 - r0 for r0, r1 in row_intervals}
    block_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert block_heights == block_widths, "panel blocks must be equal squares"
    return row_intervals, col_intervals


def _find_sprite(
    panels: tuple[np.ndarray, np.ndarray],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    payload_colors: list[int],
) -> np.ndarray:
    payload = set(map(int, payload_colors))
    sprites: list[np.ndarray] = []
    for panel in panels:
        for r0, r1 in row_intervals:
            for c0, c1 in col_intervals:
                block = panel[r0:r1, c0:c1]
                visible_colors = set(_colors_present(block, exclude={ERASED}))
                assert SEPARATOR not in visible_colors, "block interiors cannot contain separator color"
                if ERASED not in set(map(int, block.ravel())) and visible_colors == payload:
                    sprites.append(block.copy())

    assert sprites, "expected at least one fully visible sprite block"
    sprite = sprites[0]
    for candidate in sprites:
        assert np.array_equal(candidate, sprite), "fully visible sprite blocks must match"
    return sprite


def _sprite_colors(sprite: np.ndarray, payload_colors: list[int]) -> tuple[int, int]:
    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload_colors}
    assert counts[payload_colors[0]] != counts[payload_colors[1]], "sprite needs a minority color"
    foreground = max(payload_colors, key=lambda color: counts[color])
    blank = min(payload_colors, key=lambda color: counts[color])
    return int(foreground), int(blank)


def _classify_block(block: np.ndarray, sprite: np.ndarray, blank: int) -> bool:
    known = block != ERASED
    assert known.any(), "each block must expose at least one non-grey cell"
    known_colors = set(map(int, block[known].ravel()))
    assert SEPARATOR not in known_colors, "block interiors cannot contain separator color"

    matches_sprite = bool(np.all(block[known] == sprite[known]))
    matches_blank = bool(np.all(block[known] == int(blank)))
    assert matches_sprite != matches_blank, "block must unambiguously decode to sprite or blank"
    return matches_sprite


def _decode_panel(
    panel: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    sprite: np.ndarray,
    blank: int,
) -> np.ndarray:
    state = np.zeros((len(row_intervals), len(col_intervals)), dtype=bool)
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            state[br, bc] = _classify_block(panel[r0:r1, c0:c1], sprite, blank)
    return state


def _render_panel(
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    state: np.ndarray,
    sprite: np.ndarray,
    blank: int,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, SEPARATOR, dtype=int)
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            out[r0:r1, c0:c1] = sprite if bool(state[br, bc]) else int(blank)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two matching block panels, XOR their macro-cell states, and redraw one clean encoded panel.

    Input structure:
    - A solid separator column splits the input into equal left and right panels.
    - Each panel has the same black-line block grid of equal square cells.
    - Every macro-cell is either a copy of one shared two-color sprite or a uniform block of that sprite's minority color.
    - Grey pixels are erasures and act as wildcards when reading blocks.

    Rule:
    - Infer the shared sprite from fully visible two-color sprite blocks.
    - Decode each left/right corresponding macro-cell as sprite-state or blank-state, ignoring grey erasures.
    - Render the XOR of the two decoded state grids: different states become a full sprite, equal states become a minority-color blank block.
    - Return one clean panel with all separator lines black and no grey pixels.

    Key insight:
    - The comparison is at the macro-cell bit layer, not the pixel layer; erasures must be ignored before XOR and the result is re-encoded as whole blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    left, right = _split_panels(g)

    left_rows, left_cols = _panel_layout(left)
    right_rows, right_cols = _panel_layout(right)
    assert left_rows == right_rows and left_cols == right_cols, "panels must share a layout"

    payload_colors = _colors_present(g, exclude={SEPARATOR, ERASED})
    assert len(payload_colors) == 2, "expected exactly two sprite payload colors"
    sprite = _find_sprite((left, right), left_rows, left_cols, payload_colors)
    _foreground, blank = _sprite_colors(sprite, payload_colors)

    left_state = _decode_panel(left, left_rows, left_cols, sprite, blank)
    right_state = _decode_panel(right, left_rows, left_cols, sprite, blank)
    xor_state = left_state != right_state
    return _render_panel(left_rows, left_cols, xor_state, sprite, blank, left.shape)


# ---------------- generator ----------------

GRID_COUNT_RANGE = (2, 4)
BLOCK_SIZE_RANGE = (2, 5)
MAX_GRID_SIDE = 30
MAX_ATTEMPTS = 500
GREY_FRACTION_RANGE = (0.06, 0.18)


def _block_origin(br: int, bc: int, block_size: int) -> tuple[int, int]:
    return br * (block_size + 1), bc * (block_size + 1)


def _panel_shape(rows: int, cols: int, block_size: int) -> tuple[int, int]:
    return rows * block_size + rows - 1, cols * block_size + cols - 1


def _valid_layouts() -> list[tuple[int, int, int]]:
    layouts = []
    for rows in range(GRID_COUNT_RANGE[0], GRID_COUNT_RANGE[1] + 1):
        for cols in range(GRID_COUNT_RANGE[0], GRID_COUNT_RANGE[1] + 1):
            for block_size in range(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1] + 1):
                h, w = _panel_shape(rows, cols, block_size)
                if h <= MAX_GRID_SIDE and 2 * w + 1 <= MAX_GRID_SIDE:
                    layouts.append((rows, cols, block_size))
    return layouts


LAYOUTS = _valid_layouts()


def _has_simple_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _make_sprite(block_size: int, foreground: int, blank: int) -> np.ndarray | None:
    area = block_size * block_size
    if block_size == 2:
        minority_choices = [1]
    else:
        minority_choices = list(range(block_size, max(block_size, (area - 1) // 2) + 1))

    for _ in range(120):
        minority_count = random.choice(minority_choices)
        mask = np.ones((block_size, block_size), dtype=bool)
        blank_cells: set[int] = set()
        if block_size == 2:
            blank_cells.add(random.randrange(area))
        else:
            cols = list(range(block_size))
            random.shuffle(cols)
            for r, c in enumerate(cols):
                blank_cells.add(r * block_size + c)
            pool = [idx for idx in range(area) if idx not in blank_cells]
            blank_cells.update(random.sample(pool, minority_count - len(blank_cells)))

        for idx in blank_cells:
            mask[idx // block_size, idx % block_size] = False

        if block_size > 2:
            row_mixed = all(0 < np.count_nonzero(mask[r, :]) < block_size for r in range(block_size))
            col_mixed = all(0 < np.count_nonzero(mask[:, c]) < block_size for c in range(block_size))
            if not row_mixed or not col_mixed:
                continue
        if _has_simple_symmetry(mask):
            continue
        return np.where(mask, int(foreground), int(blank)).astype(int)
    return None


def _sample_state_pair(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    cells = list(range(rows * cols))
    random.shuffle(cells)
    combos = [(False, False), (False, True), (True, False), (True, True)]
    random.shuffle(combos)

    left = np.zeros((rows, cols), dtype=bool)
    right = np.zeros((rows, cols), dtype=bool)
    for idx, pair in zip(cells[:4], combos):
        r, c = divmod(idx, cols)
        left[r, c], right[r, c] = pair
    for idx in cells[4:]:
        r, c = divmod(idx, cols)
        left[r, c], right[r, c] = random.choice(combos)
    return left, right


def _render_clean_panel(
    rows: int,
    cols: int,
    block_size: int,
    state: np.ndarray,
    sprite: np.ndarray,
    blank: int,
) -> np.ndarray:
    h, w = _panel_shape(rows, cols, block_size)
    panel = np.full((h, w), SEPARATOR, dtype=int)
    for br in range(rows):
        for bc in range(cols):
            r0, c0 = _block_origin(br, bc, block_size)
            panel[r0 : r0 + block_size, c0 : c0 + block_size] = (
                sprite if bool(state[br, bc]) else int(blank)
            )
    return panel


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    out = np.full((h, 2 * w + 1), SEPARATOR, dtype=int)
    out[:, :w] = left
    out[:, w + 1 :] = right
    return out


def _protect_evidence(
    protected: np.ndarray,
    panel_width: int,
    rows: int,
    cols: int,
    block_size: int,
    left_state: np.ndarray,
    right_state: np.ndarray,
    sprite: np.ndarray,
    foreground: int,
) -> None:
    h = protected.shape[0]
    protected[:, panel_width] = True
    fg_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite == int(foreground))]

    sprite_blocks = []
    for side, states in enumerate((left_state, right_state)):
        offset = 0 if side == 0 else panel_width + 1
        for br in range(rows):
            for bc in range(cols):
                r0, c0 = _block_origin(br, bc, block_size)
                rr, cc = random.choice(fg_positions)
                protected[r0 + rr, offset + c0 + cc] = True
                if bool(states[br, bc]):
                    sprite_blocks.append((offset, r0, c0))

    assert sprite_blocks
    offset, r0, c0 = random.choice(sprite_blocks)
    protected[r0 : r0 + block_size, offset + c0 : offset + c0 + block_size] = True

    row_seps = {(i + 1) * block_size + i for i in range(rows - 1)}
    col_seps = {(i + 1) * block_size + i for i in range(cols - 1)}
    non_sep_rows = [r for r in range(h) if r not in row_seps]
    non_sep_cols = [c for c in range(panel_width) if c not in col_seps]
    for offset in (0, panel_width + 1):
        for r in non_sep_rows:
            protected[r, offset + random.choice(non_sep_cols)] = True
        for c in non_sep_cols:
            protected[random.choice(non_sep_rows), offset + c] = True


def _add_grey_overlay(
    inp: np.ndarray,
    protected: np.ndarray,
    rows: int,
    cols: int,
    block_size: int,
) -> bool:
    h, total_w = inp.shape
    panel_width = (total_w - 1) // 2
    row_seps = {(i + 1) * block_size + i for i in range(rows - 1)}
    col_seps = {(i + 1) * block_size + i for i in range(cols - 1)}

    payload_mask = np.zeros_like(inp, dtype=bool)
    separator_mask = np.zeros_like(inp, dtype=bool)
    for offset in (0, panel_width + 1):
        for r in range(h):
            for c in range(panel_width):
                target = payload_mask if r not in row_seps and c not in col_seps else separator_mask
                target[r, offset + c] = True

    allowed = ~protected
    required: set[tuple[int, int]] = set()
    left_payload = np.argwhere(allowed & payload_mask & (np.indices(inp.shape)[1] < panel_width))
    right_payload = np.argwhere(allowed & payload_mask & (np.indices(inp.shape)[1] > panel_width))
    sep_coords = np.argwhere(allowed & separator_mask)
    for pool in (left_payload, right_payload, sep_coords):
        if len(pool):
            required.add(tuple(map(int, random.choice(pool))))

    candidates = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(candidates) < 3:
        return False
    lo = max(len(required), math.ceil(inp.size * GREY_FRACTION_RANGE[0]))
    hi = max(lo, math.floor(inp.size * GREY_FRACTION_RANGE[1]))
    if lo > len(candidates):
        return False
    target = random.randint(lo, min(hi, len(candidates)))

    chosen = set(required)
    remaining = [cell for cell in candidates if cell not in chosen]
    if target > len(chosen):
        chosen.update(random.sample(remaining, target - len(chosen)))
    for r, c in chosen:
        inp[r, c] = ERASED
    return bool(chosen)


def _semantic_accepts(
    inp: np.ndarray,
    expected: np.ndarray,
    rows: int,
    cols: int,
    block_size: int,
    left_state: np.ndarray,
    right_state: np.ndarray,
    sprite: np.ndarray,
    blank: int,
) -> bool:
    if max(inp.shape) > MAX_GRID_SIDE or max(expected.shape) > MAX_GRID_SIDE:
        return False
    if not np.any(inp == ERASED):
        return False
    xor = left_state != right_state
    if not (np.any(xor) and np.any(~xor)):
        return False
    observed_pairs = {(bool(left_state[r, c]), bool(right_state[r, c])) for r in range(rows) for c in range(cols)}
    if observed_pairs != {(False, False), (False, True), (True, False), (True, True)}:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == ERASED):
        return False

    left, right = _split_panels(inp)
    row_intervals, col_intervals = _panel_layout(left)
    if _panel_layout(right) != (row_intervals, col_intervals):
        return False
    decoded_left = _decode_panel(left, row_intervals, col_intervals, sprite, blank)
    decoded_right = _decode_panel(right, row_intervals, col_intervals, sprite, blank)
    return np.array_equal(decoded_left, left_state) and np.array_equal(decoded_right, right_state)


def _gen_attempt() -> np.ndarray | None:
    rows, cols, block_size = random.choice(LAYOUTS)
    foreground, blank = random.sample(PAYLOAD_COLORS, 2)
    sprite = _make_sprite(block_size, foreground, blank)
    if sprite is None:
        return None

    left_state, right_state = _sample_state_pair(rows, cols)
    expected_state = left_state != right_state
    left_clean = _render_clean_panel(rows, cols, block_size, left_state, sprite, blank)
    right_clean = _render_clean_panel(rows, cols, block_size, right_state, sprite, blank)
    expected = _render_clean_panel(rows, cols, block_size, expected_state, sprite, blank)
    inp = _assemble_input(left_clean, right_clean)

    protected = np.zeros_like(inp, dtype=bool)
    _protect_evidence(
        protected,
        left_clean.shape[1],
        rows,
        cols,
        block_size,
        left_state,
        right_state,
        sprite,
        foreground,
    )
    if not _add_grey_overlay(inp, protected, rows, cols, block_size):
        return None
    if not _semantic_accepts(inp, expected, rows, cols, block_size, left_state, right_state, sprite, blank):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid macro-grid XOR input")
