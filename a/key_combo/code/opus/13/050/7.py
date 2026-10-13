from __future__ import annotations

import numpy as np


INPUT_SIZE = 11
QUAD_SIZE = 5
CROSS_INDEX = 5
MACRO_SIZE = 10
OUTPUT_SIZE = 30

BG_COLOR = 0
SOURCE_COLOR = 2
PLACEHOLDER_COLOR = 7

GEN_ATTEMPTS = 1000
AREA_MIN = 5
AREA_MAX = 11

_PLACEHOLDER_CYCLE: list[int] = []
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    return [
        grid[:QUAD_SIZE, :QUAD_SIZE],
        grid[:QUAD_SIZE, CROSS_INDEX + 1 :],
        grid[CROSS_INDEX + 1 :, :QUAD_SIZE],
        grid[CROSS_INDEX + 1 :, CROSS_INDEX + 1 :],
    ]


def _rotation_family(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _bbox_hw(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _render_tile(mask: np.ndarray, source_color: int) -> np.ndarray:
    tile = np.full(mask.shape, BG_COLOR, dtype=int)
    tile[mask] = int(source_color)
    return tile


def _assemble_macro(masks: list[np.ndarray], source_color: int) -> np.ndarray:
    assert len(masks) == 4
    tiles = [_render_tile(mask, source_color) for mask in masks]
    top = np.hstack((tiles[0], tiles[1]))
    bottom = np.hstack((tiles[2], tiles[3]))
    return np.vstack((top, bottom))


def _solve_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_SIZE, INPUT_SIZE), "expected an 11x11 grid"
    assert np.all(g[CROSS_INDEX, :] == BG_COLOR), "center row must be the black divider"
    assert np.all(g[:, CROSS_INDEX] == BG_COLOR), "center column must be the black divider"

    blocks = _quadrants(g)
    color_block_counts: dict[int, int] = {}
    for block in blocks:
        for color in np.unique(block):
            color = int(color)
            if color != BG_COLOR:
                color_block_counts[color] = color_block_counts.get(color, 0) + 1

    source_candidates = [color for color, count in color_block_counts.items() if count == 3]
    assert len(source_candidates) == 1, "expected one source color shared by the three pattern quadrants"
    source_color = int(source_candidates[0])

    visible_masks: list[np.ndarray] = []
    completed_masks: list[np.ndarray | None] = [None, None, None, None]
    placeholder_index: int | None = None

    for idx, block in enumerate(blocks):
        colors = set(int(v) for v in np.unique(block))
        if colors <= {BG_COLOR, source_color} and source_color in colors:
            mask = block == source_color
            visible_masks.append(mask)
            completed_masks[idx] = mask
        else:
            assert len(colors) == 1, "placeholder quadrant must be solid"
            fill_color = next(iter(colors))
            assert fill_color != BG_COLOR and fill_color != source_color
            assert placeholder_index is None, "expected exactly one placeholder quadrant"
            placeholder_index = idx

    assert placeholder_index is not None, "missing placeholder quadrant"
    assert len(visible_masks) == 3, "expected exactly three visible pattern quadrants"
    assert len({_mask_key(mask) for mask in visible_masks}) == 3, "visible rotations must be distinct"

    family = _rotation_family(visible_masks[0])
    family_keys = {_mask_key(mask) for mask in family}
    assert len(family_keys) == 4, "rotation family must have four distinct members"

    seen = {_mask_key(mask) for mask in visible_masks}
    missing = [mask for mask in family if _mask_key(mask) not in seen]
    assert len(missing) == 1, "expected exactly one missing rotation"
    completed_masks[placeholder_index] = missing[0]

    full_masks = [mask for mask in completed_masks if mask is not None]
    assert len(full_masks) == 4
    assert all(_mask_key(mask) in family_keys for mask in full_masks)
    return [mask for mask in completed_masks if mask is not None], placeholder_index, source_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing rotated corner tile, assemble the 2x2 macro tile, then tile it with a mirrored middle row.

    Input structure:
    - The input is an 11x11 grid split by a one-cell black cross at row 5 and column 5.
    - Each corner quadrant is a 5x5 tile.
    - Three quadrants contain red-on-black masks that are distinct 90-degree rotations of one another.
    - The fourth quadrant is a solid orange placeholder tile.

    Rule:
    - Ignore the black cross and extract the four 5x5 corner quadrants.
    - Recover the missing fourth rotation from the three visible red masks and replace the placeholder with it.
    - Reassemble the completed quadrants in their original spatial order into a 10x10 macro tile.
    - Tile that macro tile into a 3x3 array, but use the left-right mirror of the macro tile for the entire middle tile row.

    Key insight:
    - The placeholder does not specify a fixed corner; the three visible corner masks identify a unique missing rotation, and that completed 2x2 macro tile is the unit that gets tiled and mirrored.
    """
    completed_masks, _placeholder_index, source_color = _solve_quadrants(grid)
    macro = _assemble_macro(completed_masks, source_color)
    mirrored_macro = np.fliplr(macro)

    top_band = np.hstack((macro, macro, macro))
    middle_band = np.hstack((mirrored_macro, mirrored_macro, mirrored_macro))
    out = np.vstack((top_band, middle_band, top_band))
    assert out.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < QUAD_SIZE and 0 <= cc < QUAD_SIZE and not mask[rr, cc]:
                frontier.add((rr, cc))
    return sorted(frontier)


def _grow_connected_mask(rng: np.random.Generator) -> np.ndarray:
    target_area = int(rng.integers(AREA_MIN, AREA_MAX + 1))
    mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
    seed_r = int(rng.integers(0, QUAD_SIZE))
    seed_c = int(rng.integers(0, QUAD_SIZE))
    mask[seed_r, seed_c] = True

    while int(mask.sum()) < target_area:
        frontier = _frontier(mask)
        if not frontier:
            break
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        mask[rr, cc] = True
        if rng.random() < 0.18 and int(mask.sum()) < target_area:
            frontier = _frontier(mask)
            if frontier:
                rr2, cc2 = frontier[int(rng.integers(0, len(frontier)))]
                mask[rr2, cc2] = True

    mask = np.rot90(mask, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask)
    return mask


def _valid_base_mask(mask: np.ndarray) -> bool:
    if int(mask.sum()) < AREA_MIN or int(mask.sum()) > AREA_MAX:
        return False
    height, width = _bbox_hw(mask)
    if height < 3 or width < 3:
        return False
    family = _rotation_family(mask)
    if len({_mask_key(rot) for rot in family}) != 4:
        return False
    return True


def _next_placeholder_index(rng: np.random.Generator) -> int:
    global _PLACEHOLDER_CYCLE
    if not _PLACEHOLDER_CYCLE:
        _PLACEHOLDER_CYCLE = rng.permutation(4).astype(int).tolist()
    return int(_PLACEHOLDER_CYCLE.pop())


def _build_input(placed_masks: list[np.ndarray], placeholder_index: int) -> np.ndarray:
    grid = np.full((INPUT_SIZE, INPUT_SIZE), BG_COLOR, dtype=int)
    rows = (0, 0, CROSS_INDEX + 1, CROSS_INDEX + 1)
    cols = (0, CROSS_INDEX + 1, 0, CROSS_INDEX + 1)

    for idx, mask in enumerate(placed_masks):
        r0 = rows[idx]
        c0 = cols[idx]
        if idx == placeholder_index:
            tile = np.full((QUAD_SIZE, QUAD_SIZE), PLACEHOLDER_COLOR, dtype=int)
        else:
            tile = _render_tile(mask, SOURCE_COLOR)
        grid[r0 : r0 + QUAD_SIZE, c0 : c0 + QUAD_SIZE] = tile
    return grid


def _validate_generated_case(grid: np.ndarray) -> np.ndarray | None:
    try:
        completed_masks, placeholder_index, source_color = _solve_quadrants(grid)
        if source_color != SOURCE_COLOR:
            return None
        if placeholder_index not in range(4):
            return None
        if not any(np.all(block == PLACEHOLDER_COLOR) for block in _quadrants(grid)):
            return None

        macro = _assemble_macro(completed_masks, source_color)
        if np.array_equal(macro, np.fliplr(macro)):
            return None

        out = main(grid)
        top_band = np.hstack((macro, macro, macro))
        mid_band = np.hstack((np.fliplr(macro),) * 3)
        if out.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
            return None
        if not np.array_equal(out[:MACRO_SIZE], top_band):
            return None
        if not np.array_equal(out[MACRO_SIZE : 2 * MACRO_SIZE], mid_band):
            return None
        if not np.array_equal(out[2 * MACRO_SIZE :], top_band):
            return None
        if len({_mask_key(mask) for mask in completed_masks}) != 4:
            return None
        return out
    except AssertionError:
        return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mask = _grow_connected_mask(rng)
    if not _valid_base_mask(mask):
        return None

    family = _rotation_family(mask)
    perm = rng.permutation(4).astype(int).tolist()
    placed_masks = [family[idx] for idx in perm]
    placeholder_index = _next_placeholder_index(rng)
    grid = _build_input(placed_masks, placeholder_index)

    out = _validate_generated_case(grid)
    if out is None:
        return None

    signature = out.tobytes()
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
