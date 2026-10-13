from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

BLOCK_SIZE = 6
SPRITE_SIZE = 4
CELL_SPAN = BLOCK_SIZE + 1  # 6 interior cells + 1 grid line step

MISSING_TO_ROT90_CCW = {
    (0, 0): 0,  # top-left missing -> 0 degrees
    (0, 1): 3,  # top-right missing -> clockwise 90
    (1, 1): 2,  # bottom-right missing -> 180
    (1, 0): 1,  # bottom-left missing -> clockwise 270
}

_USED_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _line_indices(body: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(body.shape[0]) if np.all(body[r, :] == GREY)]
    return [c for c in range(body.shape[1]) if np.all(body[:, c] == GREY)]


def _extract_body(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[int], list[int]]:
    body_box = _bbox(grid == GREY)
    assert body_box is not None, "expected one grey block grid"
    body = _crop(grid, body_box)
    line_rows = _line_indices(body, axis=0)
    line_cols = _line_indices(body, axis=1)

    assert len(line_rows) >= 3 and len(line_cols) >= 3, "expected at least a 2x2 block lattice"
    assert line_rows[0] == 0 and line_rows[-1] == body.shape[0] - 1
    assert line_cols[0] == 0 and line_cols[-1] == body.shape[1] - 1
    assert all(b - a == CELL_SPAN for a, b in zip(line_rows, line_rows[1:])), "unexpected row spacing"
    assert all(b - a == CELL_SPAN for a, b in zip(line_cols, line_cols[1:])), "unexpected col spacing"
    return body, body_box, line_rows, line_cols


def _extract_control_sprite(grid: np.ndarray, body_box: tuple[int, int, int, int]) -> np.ndarray:
    mask = (grid == RED) | (grid == TEAL)
    box = _bbox(mask)
    assert box is not None, "expected one red/teal control sprite"
    r0, c0, r1, c1 = box
    br0, bc0, br1, bc1 = body_box
    assert r1 < br0 or r0 > br1 or c1 < bc0 or c0 > bc1, "control sprite must be outside the block grid"

    sprite = _crop(grid, box)
    assert sprite.shape == (SPRITE_SIZE, SPRITE_SIZE), "control sprite must be 4x4"
    assert np.all((sprite == RED) | (sprite == TEAL)), "control sprite must contain only red/teal pixels"
    assert np.count_nonzero(mask) == SPRITE_SIZE * SPRITE_SIZE, "control sprite must fill its 4x4 box"
    return sprite


def _block_anchor(line_rows: list[int], line_cols: list[int], block_r: int, block_c: int) -> tuple[int, int]:
    return line_rows[block_r] + 1, line_cols[block_c] + 1


def _missing_corner_from_block(block: np.ndarray) -> tuple[int, int]:
    patch = block[:2, :2]
    assert np.count_nonzero(patch == ORANGE) == 3, "each block must contain one orange L in its top-left 2x2"
    assert np.count_nonzero(patch == BLACK) == 1, "each orange L must have exactly one missing corner"
    assert np.all(np.isin(patch, [BLACK, ORANGE])), "unexpected colors inside the 2x2 L area"
    missing = np.argwhere(patch == BLACK)
    return int(missing[0, 0]), int(missing[0, 1])


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.astype(np.int8, copy=False).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each orange missing-corner L and stamp the rotated control sprite into its block.

    Input structure:
    - The background is black, with one grey line lattice partitioning a rectangular region into equal 6x6 blocks.
    - In every block, the top-left 2x2 area contains a 3-cell orange L with one black missing corner.
    - A separate 4x4 control sprite made only of red and teal pixels sits outside the grey block grid.

    Rule:
    - Recover the 6x6 blocks from the grey lattice and locate the unique black corner inside each block's orange 2x2 L.
    - Recolor that missing corner blue, completing a 2x2 square while preserving the three original orange pixels.
    - Map the missing-corner position to a control-sprite rotation: top-left -> 0 degrees, top-right -> clockwise 90,
      bottom-right -> 180, bottom-left -> clockwise 270.
    - Paste the rotated 4x4 control sprite into the block's bottom-right 4x4 region and keep the grey grid lines unchanged.

    Key insight:
    - The missing-corner position is a double code: it marks the blue repair pixel and simultaneously encodes the
      rotation used to stamp the same control sprite into that block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    body, body_box, line_rows, line_cols = _extract_body(g)
    sprite = _extract_control_sprite(g, body_box)

    out = g.copy()
    br0, bc0, _br1, _bc1 = body_box
    block_rows = len(line_rows) - 1
    block_cols = len(line_cols) - 1

    for block_r in range(block_rows):
        for block_c in range(block_cols):
            r0, c0 = _block_anchor(line_rows, line_cols, block_r, block_c)
            block = body[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
            assert block.shape == (BLOCK_SIZE, BLOCK_SIZE)
            assert np.all(block[2:, 2:] == BLACK), "stamp target area must be empty before overlay"

            miss = _missing_corner_from_block(block)
            out[br0 + r0 + miss[0], bc0 + c0 + miss[1]] = BLUE

            rotated = np.rot90(sprite, k=MISSING_TO_ROT90_CCW[miss])
            _paste(out, rotated, br0 + r0 + 2, bc0 + c0 + 2)

    return out


def _build_body(block_rows: int, block_cols: int) -> np.ndarray:
    body_h = block_rows * CELL_SPAN + 1
    body_w = block_cols * CELL_SPAN + 1
    body = np.full((body_h, body_w), BLACK, dtype=int)
    body[::CELL_SPAN, :] = GREY
    body[:, ::CELL_SPAN] = GREY
    return body


def _rotations_are_distinct(sprite: np.ndarray) -> bool:
    variants = {np.rot90(sprite, k).tobytes() for k in range(4)}
    return len(variants) == 4


def _sample_sprite() -> np.ndarray:
    for _ in range(200):
        sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), RED, dtype=int)
        teal_count = random.randint(5, 11)
        teal_idx = random.sample(range(SPRITE_SIZE * SPRITE_SIZE), teal_count)
        sprite.flat[teal_idx] = TEAL
        if _rotations_are_distinct(sprite):
            return sprite
    raise RuntimeError("failed to sample an asymmetric 4x4 control sprite")


def _sample_missing_codes(count: int) -> list[tuple[int, int]]:
    assert count >= 4, "generator expects at least four blocks"
    orientations = list(MISSING_TO_ROT90_CCW)
    use_count = random.choice([2, 3, 4])
    chosen = random.sample(orientations, use_count)
    codes = chosen.copy()
    while len(codes) < count:
        codes.append(random.choice(chosen))
    random.shuffle(codes)
    return codes


def _render_l(block: np.ndarray, missing: tuple[int, int]) -> None:
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                block[dr, dc] = ORANGE


def _assemble_input(body: np.ndarray, sprite: np.ndarray, placement: str) -> np.ndarray:
    top_pad = random.randint(0, 1)
    left_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 1)
    right_pad = random.randint(0, 1)
    gap = 1
    body_h, body_w = body.shape

    if placement == "top":
        total_h = top_pad + SPRITE_SIZE + gap + body_h + bottom_pad
        total_w = left_pad + body_w + right_pad
        assert total_h <= 30 and total_w <= 30
        grid = np.full((total_h, total_w), BLACK, dtype=int)
        body_r0 = top_pad + SPRITE_SIZE + gap
        body_c0 = left_pad
        sprite_r0 = top_pad
        sprite_c0 = left_pad + random.randint(0, body_w - SPRITE_SIZE)
    else:
        total_h = top_pad + body_h + bottom_pad
        total_w = left_pad + SPRITE_SIZE + gap + body_w + right_pad
        assert total_h <= 30 and total_w <= 30
        grid = np.full((total_h, total_w), BLACK, dtype=int)
        body_r0 = top_pad
        body_c0 = left_pad + SPRITE_SIZE + gap
        sprite_r0 = top_pad + random.randint(0, body_h - SPRITE_SIZE)
        sprite_c0 = left_pad

    _paste(grid, body, body_r0, body_c0)
    _paste(grid, sprite, sprite_r0, sprite_c0)
    return grid


def _expected_output_from_metadata(inp: np.ndarray, body_box: tuple[int, int, int, int], sprite: np.ndarray, missing_codes: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    body = _crop(inp, body_box)
    line_rows = _line_indices(body, axis=0)
    line_cols = _line_indices(body, axis=1)
    idx = 0
    br0, bc0, _br1, _bc1 = body_box
    for block_r in range(len(line_rows) - 1):
        for block_c in range(len(line_cols) - 1):
            r0, c0 = _block_anchor(line_rows, line_cols, block_r, block_c)
            miss = missing_codes[idx]
            idx += 1
            out[br0 + r0 + miss[0], bc0 + c0 + miss[1]] = BLUE
            rotated = np.rot90(sprite, k=MISSING_TO_ROT90_CCW[miss])
            _paste(out, rotated, br0 + r0 + 2, bc0 + c0 + 2)
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, sprite: np.ndarray, missing_codes: list[tuple[int, int]]) -> bool:
    try:
        body, body_box, line_rows, line_cols = _extract_body(inp)
        recovered_sprite = _extract_control_sprite(inp, body_box)
        assert np.array_equal(recovered_sprite, sprite)
        assert _rotations_are_distinct(sprite)
        assert len(set(missing_codes)) >= 2

        block_total = (len(line_rows) - 1) * (len(line_cols) - 1)
        assert block_total == len(missing_codes)

        solved = main(inp)
        assert np.array_equal(solved, expected), "solver output mismatch"

        fixed_stamp = inp.copy()
        no_completion = inp.copy()
        idx = 0
        br0, bc0, _br1, _bc1 = body_box
        used_rotations: set[bytes] = set()
        for block_r in range(len(line_rows) - 1):
            for block_c in range(len(line_cols) - 1):
                r0, c0 = _block_anchor(line_rows, line_cols, block_r, block_c)
                block = body[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
                miss = _missing_corner_from_block(block)
                assert miss == missing_codes[idx]
                idx += 1

                patch = solved[br0 + r0 : br0 + r0 + 2, bc0 + c0 : bc0 + c0 + 2]
                assert np.count_nonzero(patch == BLUE) == 1
                assert np.count_nonzero(patch == ORANGE) == 3
                assert patch[miss] == BLUE

                rotated = np.rot90(sprite, k=MISSING_TO_ROT90_CCW[miss])
                used_rotations.add(rotated.tobytes())
                assert np.array_equal(
                    solved[br0 + r0 + 2 : br0 + r0 + 6, bc0 + c0 + 2 : bc0 + c0 + 6],
                    rotated,
                )

                _paste(fixed_stamp, sprite, br0 + r0 + 2, bc0 + c0 + 2)
                _paste(no_completion, rotated, br0 + r0 + 2, bc0 + c0 + 2)

        assert len(used_rotations) >= 2, "at least two distinct stamp rotations must appear"
        assert not np.array_equal(fixed_stamp, expected), "case degenerates to a fixed stamp shortcut"
        assert not np.array_equal(no_completion, expected), "case degenerates without the blue completion step"
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    block_rows = random.randint(2, 3)
    block_cols = random.randint(2, 3)
    body = _build_body(block_rows, block_cols)
    sprite = _sample_sprite()

    missing_codes = _sample_missing_codes(block_rows * block_cols)
    idx = 0
    for block_r in range(block_rows):
        for block_c in range(block_cols):
            r0 = block_r * CELL_SPAN + 1
            c0 = block_c * CELL_SPAN + 1
            _render_l(body[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE], missing_codes[idx])
            idx += 1

    placement = random.choice(["top", "left"])
    inp = _assemble_input(body, sprite, placement)

    body_box = _bbox(inp == GREY)
    assert body_box is not None
    expected = _expected_output_from_metadata(inp, body_box, sprite, missing_codes)

    if _output_signature(expected) in _USED_OUTPUT_SIGNATURES:
        return None
    if not _validate_case(inp, expected, sprite, missing_codes):
        return None

    _USED_OUTPUT_SIGNATURES.add(_output_signature(expected))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner stamp case")
