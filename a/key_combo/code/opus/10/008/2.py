from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
PINK = 6
INSTRUCTION_SIZE = 4
MACRO_SCALE = 2
ACTIVE_HEIGHT = 2
PERIOD = 2
MAX_ATTEMPTS = 400


def _decode_mask(block: np.ndarray) -> np.ndarray:
    assert block.shape == (INSTRUCTION_SIZE, INSTRUCTION_SIZE)
    mask = np.zeros((2, 2), dtype=bool)
    for mr in range(2):
        for mc in range(2):
            cell = block[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            if np.all(cell == PINK):
                mask[mr, mc] = True
            else:
                assert np.all(cell == BLACK), "instruction blocks must be solid 2x2 black or pink"
    assert mask.any() and (~mask).any(), "mask must contain both ON and OFF positions"
    return mask


def _encode_mask(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(int), np.full((MACRO_SCALE, MACRO_SCALE), PINK, dtype=int))


def _find_active_band(main_area: np.ndarray) -> tuple[int, np.ndarray]:
    coords = np.argwhere(main_area == BLUE)
    assert coords.size > 0, "expected visible blue pixels in the main area"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    assert r1 - r0 + 1 == ACTIVE_HEIGHT, "this puzzle uses a 2-row periodic band"
    band = main_area[r0 : r1 + 1, :]
    return r0, band


def _visible_cutoff(band: np.ndarray) -> int:
    cols = np.flatnonzero(np.any(band == BLUE, axis=0))
    assert cols.size > 0, "expected a visible periodic prefix"
    cutoff = int(cols[-1]) + 1
    assert cutoff < band.shape[1], "input must hide a suffix that needs completion"
    return cutoff


def _period_candidates(visible: np.ndarray, full_width: int) -> list[int]:
    candidates: list[int] = []
    for p in range(1, full_width + 1):
        if full_width % p != 0 or p > visible.shape[1]:
            continue
        if all(np.array_equal(visible[:, c], visible[:, c % p]) for c in range(visible.shape[1])):
            candidates.append(p)
    return candidates


def _infer_period_and_tile(band: np.ndarray) -> tuple[int, np.ndarray]:
    cutoff = _visible_cutoff(band)
    visible = band[:, :cutoff]
    candidates = _period_candidates(visible, band.shape[1])
    assert candidates, "expected at least one valid period"
    period = candidates[0]
    tile = visible[:, :period].copy()
    for c in range(cutoff):
        assert np.array_equal(visible[:, c], tile[:, c % period]), "visible prefix must repeat exactly"
    return period, tile


def _complete_main(main_area: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    band_r0, band = _find_active_band(main_area)
    period, tile = _infer_period_and_tile(band)
    completed = np.zeros_like(main_area)
    for c in range(main_area.shape[1]):
        completed[band_r0 : band_r0 + tile.shape[0], c] = tile[:, c % period]
    return band_r0, tile, completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partially visible horizontal blue motif, then keep and recolor only the mask-selected tile positions.

    Input structure:
    - The grid has a black background, an upper main area, and a 4x4 instruction block in the lower-left corner.
    - The main area contains a 2-row blue band whose left prefix is visible while the right suffix is hidden as black.
    - The instruction block is a 2x2 macro-grid encoded by solid 2x2 pink blocks for ON and solid 2x2 black blocks for OFF.

    Rule:
    - Infer the smallest column period of the visible main-band prefix and repeat that tile across the full main-area width.
    - Decode the lower-left 4x4 instruction block into a 2x2 binary mask.
    - In every completed period tile, keep only the blue pixels at mask-ON positions and recolor them pink; set all other main-area cells to black.
    - Preserve the instruction block unchanged.

    Key insight:
    - The instruction mask applies to the completed periodic tile, not to the truncated input prefix, so both period completion and mask decoding are required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] > INSTRUCTION_SIZE and g.shape[1] >= INSTRUCTION_SIZE

    main_area = g[:-INSTRUCTION_SIZE, :]
    instruction = g[-INSTRUCTION_SIZE:, :INSTRUCTION_SIZE]
    mask = _decode_mask(instruction)

    band_r0, tile, completed = _complete_main(main_area)
    assert tile.shape == mask.shape, "instruction mask must match the inferred periodic tile"

    out = np.zeros_like(g)
    kept = np.zeros_like(main_area)
    for c0 in range(0, main_area.shape[1], tile.shape[1]):
        block = completed[band_r0 : band_r0 + tile.shape[0], c0 : c0 + tile.shape[1]]
        kept_block = np.where(mask & (block == BLUE), PINK, BLACK)
        kept[band_r0 : band_r0 + tile.shape[0], c0 : c0 + tile.shape[1]] = kept_block

    out[:-INSTRUCTION_SIZE, :] = kept
    out[-INSTRUCTION_SIZE:, :INSTRUCTION_SIZE] = instruction
    return out


def _sample_tile_and_mask() -> tuple[np.ndarray, np.ndarray]:
    while True:
        tile = np.array(
            [[random.randint(0, 1), random.randint(0, 1)], [random.randint(0, 1), random.randint(0, 1)]],
            dtype=bool,
        )
        if not tile.any():
            continue
        if not tile[:, 0].any() or not tile[:, 1].any():
            continue
        if np.array_equal(tile[:, 0], tile[:, 1]):
            continue

        mask = np.array(
            [[random.randint(0, 1), random.randint(0, 1)], [random.randint(0, 1), random.randint(0, 1)]],
            dtype=bool,
        )
        if not mask.any() or np.all(mask):
            continue
        if not np.any(tile & mask):
            continue
        if not np.any(tile & ~mask):
            continue
        return tile.astype(int), mask


def _build_input_and_output(
    total_rows: int,
    width: int,
    band_r0: int,
    cutoff: int,
    tile: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    main_rows = total_rows - INSTRUCTION_SIZE
    inp = np.zeros((total_rows, width), dtype=int)
    out = np.zeros_like(inp)

    for c in range(cutoff):
        inp[band_r0 : band_r0 + ACTIVE_HEIGHT, c] = tile[:, c % PERIOD]
    for c in range(width):
        col = c % PERIOD
        kept_col = np.where(mask[:, col] & (tile[:, col] == 1), PINK, BLACK)
        out[band_r0 : band_r0 + ACTIVE_HEIGHT, c] = kept_col

    instruction = _encode_mask(mask)
    inp[-INSTRUCTION_SIZE:, :INSTRUCTION_SIZE] = instruction
    out[-INSTRUCTION_SIZE:, :INSTRUCTION_SIZE] = instruction
    return inp, out


def _semantic_checks(inp: np.ndarray, tile: np.ndarray, mask: np.ndarray, cutoff: int) -> None:
    main_area = inp[:-INSTRUCTION_SIZE, :]
    band_r0, band = _find_active_band(main_area)
    assert band_r0 >= 0
    assert cutoff == _visible_cutoff(band)

    candidates = _period_candidates(band[:, :cutoff], band.shape[1])
    assert candidates == [PERIOD], f"expected a unique minimal period of {PERIOD}, got {candidates}"
    assert np.any(tile & mask), "mask must keep at least one blue cell"
    assert np.any(tile & ~mask), "mask must remove at least one blue cell"

    hidden_kept = False
    for c in range(cutoff, band.shape[1]):
        if np.any(tile[:, c % PERIOD] & mask[:, c % PERIOD]):
            hidden_kept = True
            break
    assert hidden_kept, "completion must create at least one new visible output pixel"


def _attempt_generate() -> np.ndarray | None:
    total_rows = random.randint(14, 17)
    width = random.choice((10, 12, 14, 16))
    main_rows = total_rows - INSTRUCTION_SIZE
    band_r0 = random.randint(1, main_rows - ACTIVE_HEIGHT - 1)
    cutoff = random.randint(4, min(8, width - 1))
    tile, mask = _sample_tile_and_mask()
    inp, expected = _build_input_and_output(total_rows, width, band_r0, cutoff, tile, mask)

    try:
        _semantic_checks(inp, tile.astype(bool), mask, cutoff)
        pred = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(pred, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-mask input")
