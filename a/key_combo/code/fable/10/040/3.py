from __future__ import annotations

import random

import numpy as np


INPUT_SHAPE = (3, 3)
SLOT_SIZE = 3
OUTPUT_SHAPE = (9, 9)
BLACK = 0
FOREGROUND_COLORS = tuple(range(1, 10))

_CORNERS = frozenset({(0, 0), (0, 2), (2, 0), (2, 2)})
_VERTICAL_EDGES = frozenset({(0, 1), (2, 1)})
_HORIZONTAL_EDGES = frozenset({(1, 0), (1, 2)})
_MODE_REQUIREMENTS = (
    {"center": True, "v_edge": True, "h_edge": True, "corner": True},
    {"center": False, "v_edge": True, "h_edge": True, "corner": True},
    {"center": True, "v_edge": True, "h_edge": False, "corner": True},
    {"center": True, "v_edge": False, "h_edge": True, "corner": True},
)
_GEN_COUNTER = 0


def _single_foreground_color(grid: np.ndarray) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != BLACK]
    assert len(colors) == 1, "expected exactly one non-black foreground color"
    return colors[0]


def _slot_transform(tile: np.ndarray, slot_r: int, slot_c: int) -> np.ndarray:
    transformed = tile
    if int(slot_r) != 1:
        transformed = np.flipud(transformed)
    if int(slot_c) != 1:
        transformed = np.fliplr(transformed)
    return transformed


def _mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid != BLACK, dtype=bool)


def _is_puzzle_valid_mask(mask: np.ndarray) -> bool:
    if mask.shape != INPUT_SHAPE:
        return False
    fg = int(mask.sum())
    if fg == 0 or fg == mask.size:
        return False
    black_mask = ~mask
    return not (
        np.array_equal(black_mask, np.fliplr(black_mask))
        or np.array_equal(black_mask, np.flipud(black_mask))
    )


def _is_generation_ready_mask(mask: np.ndarray) -> bool:
    if not _is_puzzle_valid_mask(mask):
        return False
    fg = int(mask.sum())
    if fg < 4 or fg > 6:
        return False
    return not (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.flipud(np.fliplr(mask)))
    )


def _slot_categories(mask: np.ndarray) -> dict[str, bool]:
    pts = {tuple(cell) for cell in np.argwhere(mask)}
    return {
        "center": (1, 1) in pts,
        "v_edge": bool(pts & _VERTICAL_EDGES),
        "h_edge": bool(pts & _HORIZONTAL_EDGES),
        "corner": bool(pts & _CORNERS),
    }


def _matches_mode(mask: np.ndarray, requirement: dict[str, bool]) -> bool:
    categories = _slot_categories(mask)
    return all(categories[key] == value for key, value in requirement.items())


def _render_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    tile = mask.astype(int) * int(color)
    for slot_r, slot_c in np.argwhere(mask):
        block = _slot_transform(tile, int(slot_r), int(slot_c))
        r0 = int(slot_r) * SLOT_SIZE
        c0 = int(slot_c) * SLOT_SIZE
        out[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = block
    return out


def _recover_mask_from_output(out: np.ndarray) -> np.ndarray:
    coarse = np.zeros(INPUT_SHAPE, dtype=bool)
    for slot_r in range(INPUT_SHAPE[0]):
        for slot_c in range(INPUT_SHAPE[1]):
            r0 = slot_r * SLOT_SIZE
            c0 = slot_c * SLOT_SIZE
            coarse[slot_r, slot_c] = np.any(out[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] != BLACK)
    return coarse


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 pattern into 9x9 by stamping the complement of the black mask with slot-dependent reflections.

    Input structure:
    - The input is a 3x3 grid with black background cells and exactly one non-black foreground color.
    - Black cells and foreground cells are both present.
    - The black-cell mask is neither left-right symmetric nor up-down symmetric.

    Rule:
    - Split the 9x9 output into nine 3x3 slots aligned with the input cells.
    - Every black input cell maps to an all-black output slot.
    - Let the stamp tile be the complement of the black mask, namely the input foreground mask itself.
    - For each foreground input cell, stamp that 3x3 tile in the corresponding slot using the foreground color.
    - Slots above or below the center flip the tile vertically, slots left or right of the center flip it horizontally, corners apply both flips, and the center applies none.

    Key insight:
    - The input foreground mask serves twice: it chooses which output slots receive stamps, and it is also the stamp whose orientation depends on slot position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, "expected a 3x3 input grid"

    color = _single_foreground_color(g)
    mask = _mask_from_grid(g)
    assert _is_puzzle_valid_mask(mask), "input mask must satisfy the puzzle constraints"

    return _render_from_mask(mask, color)


def _build_mask_catalog() -> tuple[list[np.ndarray], ...]:
    signatures: dict[bytes, list[np.ndarray]] = {}
    valid_masks: list[np.ndarray] = []
    for bits in range(1 << (INPUT_SHAPE[0] * INPUT_SHAPE[1])):
        flat = np.array([(bits >> idx) & 1 for idx in range(9)], dtype=bool)
        mask = flat.reshape(INPUT_SHAPE)
        if not _is_puzzle_valid_mask(mask):
            continue
        valid_masks.append(mask.copy())
        sig = _render_from_mask(mask, 1).tobytes()
        signatures.setdefault(sig, []).append(mask.copy())

    assert all(len(candidates) == 1 for candidates in signatures.values()), "valid inputs must remain uniquely decodable"

    catalogs: list[list[np.ndarray]] = [[] for _ in _MODE_REQUIREMENTS]
    for mask in valid_masks:
        if not _is_generation_ready_mask(mask):
            continue
        sig = _render_from_mask(mask, 1).tobytes()
        assert len(signatures[sig]) == 1
        for mode_idx, requirement in enumerate(_MODE_REQUIREMENTS):
            if _matches_mode(mask, requirement):
                catalogs[mode_idx].append(mask.copy())

    assert all(catalogs), "each generation mode needs at least one candidate mask"
    return tuple(catalogs)


_MODE_MASKS = _build_mask_catalog()


def _validate_example(inp: np.ndarray, requirement: dict[str, bool]) -> None:
    g = np.asarray(inp, dtype=int)
    assert g.shape == INPUT_SHAPE
    color = _single_foreground_color(g)
    mask = _mask_from_grid(g)
    assert _is_generation_ready_mask(mask)

    out = main(g)
    assert out.shape == OUTPUT_SHAPE
    recovered_mask = _recover_mask_from_output(out)
    assert np.array_equal(recovered_mask, mask), "output slot occupancy must recover the input foreground mask"

    tile = mask.astype(int) * color
    for slot_r in range(3):
        for slot_c in range(3):
            r0 = slot_r * SLOT_SIZE
            c0 = slot_c * SLOT_SIZE
            block = out[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE]
            if not mask[slot_r, slot_c]:
                assert np.all(block == BLACK), "background slots must stay black"
                continue
            expected = _slot_transform(tile, slot_r, slot_c)
            assert np.array_equal(block, expected), "foreground slots must carry the correctly reflected stamp"
            unflipped = _slot_transform(block, slot_r, slot_c)
            assert np.array_equal(unflipped, tile), "slot reflections must be invertible back to the source tile"

    assert _matches_mode(mask, requirement), "generator must cover the intended reflection-branch combination"


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    mode_idx = _GEN_COUNTER % len(_MODE_MASKS)
    _GEN_COUNTER += 1

    mask = random.choice(_MODE_MASKS[mode_idx]).copy()
    color = random.choice(FOREGROUND_COLORS)
    grid = mask.astype(int) * color

    _validate_example(grid, _MODE_REQUIREMENTS[mode_idx])
    return grid
