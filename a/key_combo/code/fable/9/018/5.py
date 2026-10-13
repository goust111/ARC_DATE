from __future__ import annotations

import numpy as np


SEPARATOR = 0
ERASED = 5
UNKNOWN = -1
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_ATTEMPTS = 400


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({SEPARATOR, ERASED})


def _layout_candidates(shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    height, width = map(int, shape)
    candidates: list[tuple[int, int, int]] = []
    for block_size in range(2, min(height, width) + 1):
        step = block_size + 1
        if (height + 1) % step or (width + 1) % step:
            continue
        block_rows = (height + 1) // step
        block_cols = (width + 1) // step
        if block_rows < 2 or block_cols < 2:
            continue
        candidates.append((block_size, block_rows, block_cols))
    return candidates


def _block_origin(block_row: int, block_col: int, block_size: int) -> tuple[int, int]:
    step = block_size + 1
    return block_row * step, block_col * step


def _extract_blocks(
    grid: np.ndarray, block_size: int, block_rows: int, block_cols: int
) -> list[list[np.ndarray]]:
    blocks: list[list[np.ndarray]] = []
    for br in range(block_rows):
        row: list[np.ndarray] = []
        for bc in range(block_cols):
            r0, c0 = _block_origin(br, bc, block_size)
            row.append(grid[r0 : r0 + block_size, c0 : c0 + block_size].copy())
        blocks.append(row)
    return blocks


def _layout_has_valid_separators(
    grid: np.ndarray, block_size: int, block_rows: int, block_cols: int
) -> bool:
    step = block_size + 1
    row_separators = [k * step - 1 for k in range(1, block_rows)]
    col_separators = [k * step - 1 for k in range(1, block_cols)]
    return all(_is_separator_line(grid[r, :]) for r in row_separators) and all(
        _is_separator_line(grid[:, c]) for c in col_separators
    )


def _infer_sprite(blocks: list[list[np.ndarray]], payload: tuple[int, int]) -> np.ndarray:
    mixed_clean: list[np.ndarray] = []
    mixed_partial: list[np.ndarray] = []
    for row in blocks:
        for block in row:
            visible = block[block != ERASED]
            if visible.size == 0:
                continue
            colors = set(map(int, np.unique(visible)))
            if colors == set(payload):
                if np.any(block == ERASED):
                    mixed_partial.append(block)
                else:
                    mixed_clean.append(block)

    assert mixed_clean or mixed_partial, "expected at least one sprite block exposing both colors"

    if mixed_clean:
        sprite = mixed_clean[0].copy()
        for block in mixed_clean[1:]:
            assert np.array_equal(block, sprite), "clean sprite blocks must match exactly"
    else:
        shape = mixed_partial[0].shape
        sprite = np.full(shape, UNKNOWN, dtype=int)
        for block in mixed_partial:
            known = block != ERASED
            both_known = known & (sprite != UNKNOWN)
            assert np.all(sprite[both_known] == block[both_known]), "sprite evidence conflicts"
            sprite[known] = block[known]
        assert not np.any(sprite == UNKNOWN), "partial sprite evidence must cover the full sprite"

    for block in mixed_partial:
        known = block != ERASED
        assert np.all(block[known] == sprite[known]), "mixed sprite evidence must fit the inferred sprite"
    return sprite


def _classify_blocks(
    blocks: list[list[np.ndarray]], sprite: np.ndarray, majority: int, minority: int
) -> tuple[np.ndarray, np.ndarray]:
    block_rows = len(blocks)
    block_cols = len(blocks[0])
    macro = np.full((block_rows, block_cols), UNKNOWN, dtype=int)
    ambiguous = np.zeros((block_rows, block_cols), dtype=bool)

    for br in range(block_rows):
        for bc in range(block_cols):
            block = blocks[br][bc]
            known = block != ERASED
            assert known.any(), "every block must keep at least one visible cell"
            matches_sprite = bool(np.all(block[known] == sprite[known]))
            matches_blank = bool(np.all(block[known] == minority))
            assert matches_sprite or matches_blank, "block must fit sprite or blank"
            if matches_sprite and matches_blank:
                ambiguous[br, bc] = True
            else:
                macro[br, bc] = majority if matches_sprite else minority

    assert np.any(ambiguous), "expected at least one ambiguous block"
    return macro, ambiguous


def _mirror_coord(
    row: int, col: int, shape: tuple[int, int], axis: str
) -> tuple[int, int]:
    height, width = map(int, shape)
    if axis == "h":
        return height - 1 - row, col
    if axis == "v":
        return row, width - 1 - col
    raise ValueError(axis)


def _axis_is_valid(macro: np.ndarray, ambiguous: np.ndarray, axis: str) -> bool:
    known = macro != UNKNOWN
    for row in range(macro.shape[0]):
        for col in range(macro.shape[1]):
            rr, cc = _mirror_coord(row, col, macro.shape, axis)
            if known[row, col] and known[rr, cc] and macro[row, col] != macro[rr, cc]:
                return False
            if ambiguous[row, col]:
                if (rr, cc) == (row, col):
                    return False
                if not known[rr, cc]:
                    return False
    return True


def _resolve_axis(macro: np.ndarray, ambiguous: np.ndarray) -> str:
    axes = [axis for axis in ("h", "v") if _axis_is_valid(macro, ambiguous, axis)]
    assert len(axes) == 1, "known macro-cells must imply a unique mirror axis"
    return axes[0]


def _fill_ambiguous(macro: np.ndarray, ambiguous: np.ndarray, axis: str) -> np.ndarray:
    out = macro.copy()
    for row, col in np.argwhere(ambiguous):
        row = int(row)
        col = int(col)
        rr, cc = _mirror_coord(row, col, out.shape, axis)
        out[row, col] = out[rr, cc]
    return out


def _solve_with_layout(
    grid: np.ndarray, block_size: int, block_rows: int, block_cols: int
) -> np.ndarray:
    assert _layout_has_valid_separators(grid, block_size, block_rows, block_cols)
    payload = sorted(set(map(int, np.unique(grid))) - {SEPARATOR, ERASED})
    assert len(payload) == 2, "expected exactly two payload colors"

    blocks = _extract_blocks(grid, block_size, block_rows, block_cols)
    sprite = _infer_sprite(blocks, (payload[0], payload[1]))

    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload}
    assert counts[payload[0]] != counts[payload[1]], "sprite must have majority and minority colors"
    majority = max(payload, key=lambda color: counts[color])
    minority = min(payload, key=lambda color: counts[color])

    macro, ambiguous = _classify_blocks(blocks, sprite, int(majority), int(minority))
    axis = _resolve_axis(macro, ambiguous)
    out = _fill_ambiguous(macro, ambiguous, axis)
    assert np.array_equal(out, np.flipud(out)) if axis == "h" else np.array_equal(out, np.fliplr(out))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the block grid into a macro image, letting whole-image mirror symmetry decide locally ambiguous blocks.

    Input structure:
    - Black separator rows and columns split the grid into equal square blocks.
    - Each block is either a repeated two-color sprite or a solid block of the sprite's minority color.
    - Grey pixels are erasures that act as wildcards, and a few blocks are erased so heavily that local evidence alone is ambiguous.

    Rule:
    - Infer the repeated sprite from the determinable sprite blocks and decode each locally determinable block into one macro-pixel.
    - Sprite blocks decode to the sprite's majority color, while solid blank blocks decode to the minority color.
    - The already decoded macro-pixels exhibit exactly one whole-image mirror symmetry, horizontal or vertical; copy each ambiguous block's value from its mirror mate.

    Key insight:
    - The hard blocks are not repaired locally: their identity is forced only after the block code is compressed into a mirror-symmetric macro image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    outputs: list[np.ndarray] = []
    signatures: set[tuple[int, int, bytes]] = set()
    for block_size, block_rows, block_cols in _layout_candidates(g.shape):
        try:
            out = _solve_with_layout(g, block_size, block_rows, block_cols)
        except AssertionError:
            continue
        sig = (out.shape[0], out.shape[1], out.tobytes())
        if sig not in signatures:
            signatures.add(sig)
            outputs.append(out)

    assert len(outputs) == 1, "input must decode to a unique macro image"
    return outputs[0]


def _make_sprite(rng: np.random.Generator, block_size: int, majority: int, minority: int) -> np.ndarray:
    area = block_size * block_size
    min_majority = (3 * area + 4) // 5
    max_majority = area - 1
    for _ in range(200):
        majority_count = int(rng.integers(min_majority, max_majority + 1))
        minority_count = area - majority_count
        if minority_count < 1:
            continue
        if area >= 6 and minority_count < 2:
            continue
        flat = np.array([majority] * majority_count + [minority] * minority_count, dtype=int)
        rng.shuffle(flat)
        sprite = flat.reshape(block_size, block_size)
        if np.array_equal(sprite, np.flipud(sprite)):
            continue
        if np.array_equal(sprite, np.fliplr(sprite)):
            continue
        if block_size >= 3:
            if sum(len(set(map(int, row))) == 2 for row in sprite) < 2:
                continue
            if sum(len(set(map(int, sprite[:, c]))) == 2 for c in range(block_size)) < 2:
                continue
        return sprite
    raise RuntimeError("failed to build a usable sprite")


def _make_macro(
    rng: np.random.Generator, rows: int, cols: int, axis: str, majority: int, minority: int
) -> np.ndarray:
    for _ in range(300):
        macro = np.full((rows, cols), minority, dtype=int)
        if axis == "h":
            half = (rows + 1) // 2
            seed = rng.choice([majority, minority], size=(half, cols))
            macro[:half, :] = seed
            for r in range(rows):
                macro[rows - 1 - r, :] = macro[r, :]
        else:
            half = (cols + 1) // 2
            seed = rng.choice([majority, minority], size=(rows, half))
            macro[:, :half] = seed
            for c in range(cols):
                macro[:, cols - 1 - c] = macro[:, c]
        if len(set(map(int, macro.flat))) != 2:
            continue
        if int(np.count_nonzero(macro == majority)) < 2:
            continue
        if int(np.count_nonzero(macro == minority)) < 2:
            continue
        if axis == "h" and np.array_equal(macro, np.fliplr(macro)):
            continue
        if axis == "v" and np.array_equal(macro, np.flipud(macro)):
            continue
        return macro
    raise RuntimeError("failed to build a uniquely symmetric macro image")


def _pair_representatives(rows: int, cols: int, axis: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for row in range(rows):
        for col in range(cols):
            if (row, col) in seen:
                continue
            rr, cc = _mirror_coord(row, col, (rows, cols), axis)
            seen.add((row, col))
            seen.add((rr, cc))
            if (rr, cc) != (row, col):
                pairs.append(((row, col), (rr, cc)))
    return pairs


def _pick_ambiguous_positions(
    rng: np.random.Generator, macro: np.ndarray, axis: str
) -> set[tuple[int, int]]:
    pairs = _pair_representatives(*macro.shape, axis)
    for _ in range(100):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        count = int(rng.integers(1, min(4, len(shuffled) + 1)))
        ambiguous: set[tuple[int, int]] = set()
        for a, b in shuffled[:count]:
            choose = a if int(rng.integers(0, 2)) == 0 else b
            ambiguous.add(choose)
        partial, ambiguous_mask = _macro_known_from_construction(macro, ambiguous)
        try:
            if _resolve_axis(partial, ambiguous_mask) == axis:
                return ambiguous
        except AssertionError:
            continue
    raise RuntimeError("failed to pick ambiguity positions with a unique implied axis")


def _random_subset_mask(
    rng: np.random.Generator, shape: tuple[int, int], required: list[tuple[int, int]], pool: list[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in required:
        mask[r, c] = True
    extras = [coord for coord in pool if not mask[coord]]
    if extras:
        extra_count = int(rng.integers(0, len(extras) + 1))
        if extra_count:
            chosen = rng.choice(len(extras), size=extra_count, replace=False)
            for idx in np.atleast_1d(chosen):
                r, c = extras[int(idx)]
                mask[r, c] = True
    return mask


def _render_block(
    rng: np.random.Generator,
    clean_block: np.ndarray,
    sprite: np.ndarray,
    majority_positions: list[tuple[int, int]],
    minority_positions: list[tuple[int, int]],
    kind: str,
    keep_clean: bool,
) -> np.ndarray:
    if keep_clean:
        return clean_block.copy()

    if kind == "ambiguous":
        keep_count = int(rng.integers(1, len(minority_positions) + 1))
        chosen = rng.choice(len(minority_positions), size=keep_count, replace=False)
        visible = np.zeros(sprite.shape, dtype=bool)
        for idx in np.atleast_1d(chosen):
            r, c = minority_positions[int(idx)]
            visible[r, c] = True
    elif kind == "sprite":
        required = [majority_positions[int(rng.integers(0, len(majority_positions)))]]
        all_positions = majority_positions + minority_positions
        visible = _random_subset_mask(rng, sprite.shape, required, all_positions)
    elif kind == "blank":
        required = [majority_positions[int(rng.integers(0, len(majority_positions)))]]
        visible = _random_subset_mask(rng, sprite.shape, required, majority_positions + minority_positions)
    else:
        raise ValueError(kind)

    out = clean_block.copy()
    out[~visible] = ERASED
    return out


def _build_input_grid(
    rng: np.random.Generator,
    macro: np.ndarray,
    sprite: np.ndarray,
    ambiguous_positions: set[tuple[int, int]],
    clean_anchor: tuple[int, int],
) -> np.ndarray:
    block_rows, block_cols = macro.shape
    block_size = sprite.shape[0]
    height = block_rows * block_size + (block_rows - 1)
    width = block_cols * block_size + (block_cols - 1)
    grid = np.full((height, width), SEPARATOR, dtype=int)

    majority = int(max(set(map(int, np.unique(sprite))), key=lambda color: int(np.count_nonzero(sprite == color))))
    minority = int(min(set(map(int, np.unique(sprite))), key=lambda color: int(np.count_nonzero(sprite == color))))
    majority_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite == majority)]
    minority_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite == minority)]

    for br in range(block_rows):
        for bc in range(block_cols):
            r0, c0 = _block_origin(br, bc, block_size)
            is_sprite = macro[br, bc] == majority
            clean_block = sprite.copy() if is_sprite else np.full(sprite.shape, minority, dtype=int)
            if (br, bc) in ambiguous_positions:
                kind = "ambiguous"
                keep_clean = False
            elif is_sprite:
                kind = "sprite"
                keep_clean = (br, bc) == clean_anchor or bool(rng.integers(0, 3) == 0)
            else:
                kind = "blank"
                keep_clean = bool(rng.integers(0, 5) == 0)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = _render_block(
                rng, clean_block, sprite, majority_positions, minority_positions, kind, keep_clean
            )

    sep_mask = grid == SEPARATOR
    sep_coords = np.argwhere(sep_mask)
    if len(sep_coords):
        grey_count = int(rng.integers(max(1, len(sep_coords) // 8), max(2, len(sep_coords) // 3 + 1)))
        grey_count = min(grey_count, len(sep_coords))
        chosen = rng.choice(len(sep_coords), size=grey_count, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = sep_coords[int(idx)]
            grid[r, c] = ERASED

    return grid


def _macro_known_from_construction(
    macro: np.ndarray, ambiguous_positions: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    partial = macro.copy()
    ambiguous = np.zeros_like(macro, dtype=bool)
    for row, col in ambiguous_positions:
        partial[row, col] = UNKNOWN
        ambiguous[row, col] = True
    return partial, ambiguous


def _construction_semantics_ok(
    grid: np.ndarray,
    macro: np.ndarray,
    sprite: np.ndarray,
    axis: str,
    ambiguous_positions: set[tuple[int, int]],
) -> bool:
    majority = int(max(set(map(int, np.unique(sprite))), key=lambda color: int(np.count_nonzero(sprite == color))))
    minority = int(min(set(map(int, np.unique(sprite))), key=lambda color: int(np.count_nonzero(sprite == color))))
    majority_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite == majority)]
    minority_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite == minority)]

    block_size = sprite.shape[0]
    blocks = _extract_blocks(grid, block_size, macro.shape[0], macro.shape[1])
    saw_clean_sprite = False

    for br in range(macro.shape[0]):
        for bc in range(macro.shape[1]):
            block = blocks[br][bc]
            known = block != ERASED
            if not known.any():
                return False
            matches_sprite = bool(np.all(block[known] == sprite[known]))
            matches_blank = bool(np.all(block[known] == minority))
            if (br, bc) in ambiguous_positions:
                if not (matches_sprite and matches_blank):
                    return False
                known_coords = {tuple(map(int, rc)) for rc in np.argwhere(known)}
                if not known_coords.issubset(set(minority_positions)):
                    return False
                if not any(known[r, c] for r, c in minority_positions):
                    return False
            else:
                if matches_sprite == matches_blank:
                    return False
            if np.array_equal(block, sprite):
                saw_clean_sprite = True

    if not saw_clean_sprite:
        return False

    partial, ambiguous = _macro_known_from_construction(macro, ambiguous_positions)
    try:
        if _resolve_axis(partial, ambiguous) != axis:
            return False
    except AssertionError:
        return False
    if axis == "h" and not np.array_equal(macro, np.flipud(macro)):
        return False
    if axis == "v" and not np.array_equal(macro, np.fliplr(macro)):
        return False
    if axis == "h" and np.array_equal(macro, np.fliplr(macro)):
        return False
    if axis == "v" and np.array_equal(macro, np.flipud(macro)):
        return False
    if int(np.count_nonzero(grid == ERASED)) == 0:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    block_size = int(rng.integers(2, 5))
    max_blocks = (30 + 1) // (block_size + 1)
    rows = int(rng.integers(3, min(6, max_blocks + 1)))
    cols = int(rng.integers(3, min(6, max_blocks + 1)))

    height = rows * block_size + (rows - 1)
    width = cols * block_size + (cols - 1)
    if max(height, width) > 30:
        return None

    colors = rng.choice(PALETTE, size=2, replace=False)
    majority, minority = int(colors[0]), int(colors[1])
    sprite = _make_sprite(rng, block_size, majority, minority)

    axis = "h" if bool(rng.integers(0, 2) == 0) else "v"
    macro = _make_macro(rng, rows, cols, axis, majority, minority)
    ambiguous_positions = _pick_ambiguous_positions(rng, macro, axis)

    majority_cells = [tuple(map(int, rc)) for rc in np.argwhere(macro == majority) if tuple(map(int, rc)) not in ambiguous_positions]
    if not majority_cells:
        return None
    clean_anchor = majority_cells[int(rng.integers(0, len(majority_cells)))]

    grid = _build_input_grid(rng, macro, sprite, ambiguous_positions, clean_anchor)
    if not _construction_semantics_ok(grid, macro, sprite, axis, ambiguous_positions):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, macro):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric arbitration puzzle")
