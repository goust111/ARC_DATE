from __future__ import annotations

import random

import numpy as np


INPUT_SIZE = 4
ORBIT_STEP = 2
OUTPUT_SIZE = INPUT_SIZE * INPUT_SIZE
BACKGROUND = 0
OCCLUSION = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 400


def _orbit_coords(rr: int, cc: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(rr, INPUT_SIZE, ORBIT_STEP)
        for c in range(cc, INPUT_SIZE, ORBIT_STEP)
    ]


def _repair_seed(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (INPUT_SIZE, INPUT_SIZE), "expected a 4x4 grid"

    repaired = g.copy()
    for rr in range(ORBIT_STEP):
        for cc in range(ORBIT_STEP):
            coords = _orbit_coords(rr, cc)
            visible = {int(g[r, c]) for r, c in coords if int(g[r, c]) != OCCLUSION}
            assert visible, "every symmetry orbit must expose at least one visible reference"
            assert len(visible) == 1, "visible cells in one orbit must agree"
            color = next(iter(visible))
            for r, c in coords:
                if int(repaired[r, c]) == OCCLUSION:
                    repaired[r, c] = color

    assert not np.any(repaired == OCCLUSION), "all occlusions should be repaired"
    return repaired


def _expand_seed(seed: np.ndarray) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BACKGROUND, dtype=int)
    for r in range(INPUT_SIZE):
        for c in range(INPUT_SIZE):
            if int(seed[r, c]) != BACKGROUND:
                r0 = r * INPUT_SIZE
                c0 = c * INPUT_SIZE
                out[r0 : r0 + INPUT_SIZE, c0 : c0 + INPUT_SIZE] = seed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a partially occluded 4x4 translational seed, then expand it by self-copy block stamping.

    Input structure:
    - The input is a 4x4 grid with black background, two foreground colors, and grey occlusion cells.
    - The 4x4 grid is composed of four repeated 2x2 sub-blocks, so cells with the same (row mod 2, col mod 2)
      belong to one translational symmetry orbit.
    - Exactly some cells are replaced by grey, and each grey cell still has at least one visible partner in its orbit.

    Rule:
    - Treat grey as an occlusion marker rather than a real color.
    - Repair each grey cell by copying the unique visible color from its translational orbit.
    - Use the repaired 4x4 grid as a seed for a 16x16 output.
    - For every non-background seed cell at (r, c), copy the entire repaired 4x4 seed into output block
      with top-left corner at (4r, 4c); blocks for background seed cells stay black.

    Key insight:
    - The occluded cells must be resolved first, because whether a repaired seed cell becomes foreground or
      background decides whether its whole 4x4 output block is stamped or left empty.
    """
    seed = _repair_seed(grid)
    return _expand_seed(seed)


def _make_template() -> np.ndarray:
    fg1, fg2 = random.sample(PALETTE, 2)
    duplicate = random.choice((BACKGROUND, fg1, fg2))
    cells = [BACKGROUND, fg1, fg2, duplicate]
    random.shuffle(cells)
    template = np.array(cells, dtype=int).reshape(ORBIT_STEP, ORBIT_STEP)
    assert BACKGROUND in template
    assert fg1 in template
    assert fg2 in template
    return template


def _tile_template(template: np.ndarray) -> np.ndarray:
    seed = np.tile(template, (ORBIT_STEP, ORBIT_STEP))
    assert seed.shape == (INPUT_SIZE, INPUT_SIZE)
    return seed


def _block_positions() -> list[tuple[int, int]]:
    return [(0, 0), (0, 2), (2, 0), (2, 2)]


def _local_positions() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def _choose_occlusions(seed: np.ndarray) -> list[tuple[int, int]] | None:
    blocks = _block_positions()
    locals_ = _local_positions()
    grey_count = random.choice((2, 3))

    for _ in range(80):
        chosen_blocks = random.sample(range(len(blocks)), grey_count)
        coords: list[tuple[int, int]] = []
        repaired_values: list[int] = []
        orbits: set[tuple[int, int]] = set()

        for block_index in chosen_blocks:
            br, bc = blocks[block_index]
            lr, lc = random.choice(locals_)
            r, c = br + lr, bc + lc
            coords.append((r, c))
            repaired_values.append(int(seed[r, c]))
            orbits.add((r % ORBIT_STEP, c % ORBIT_STEP))

        if not any(value == BACKGROUND for value in repaired_values):
            continue
        if not any(value != BACKGROUND for value in repaired_values):
            continue
        if len(orbits) < 2:
            continue
        return coords
    return None


def _valid_generated_input(inp: np.ndarray, repaired_seed: np.ndarray, occluded: list[tuple[int, int]]) -> bool:
    if inp.shape != (INPUT_SIZE, INPUT_SIZE):
        return False
    if np.count_nonzero(inp == OCCLUSION) != len(occluded):
        return False
    if len(occluded) not in (2, 3):
        return False

    blocks = {(r // ORBIT_STEP, c // ORBIT_STEP) for r, c in occluded}
    if len(blocks) != len(occluded):
        return False

    repaired = _repair_seed(inp)
    if not np.array_equal(repaired, repaired_seed):
        return False

    if np.count_nonzero(repaired != BACKGROUND) < 4:
        return False

    restored_values = [int(repaired[r, c]) for r, c in occluded]
    if not any(value == BACKGROUND for value in restored_values):
        return False
    if not any(value != BACKGROUND for value in restored_values):
        return False

    visible_colors = set(int(x) for x in inp.flat if int(x) not in (BACKGROUND, OCCLUSION))
    if len(visible_colors) != 2:
        return False

    expected = _expand_seed(repaired_seed)
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template = _make_template()
    repaired_seed = _tile_template(template)
    occluded = _choose_occlusions(repaired_seed)
    if occluded is None:
        return None

    inp = repaired_seed.copy()
    for r, c in occluded:
        inp[r, c] = OCCLUSION

    if not _valid_generated_input(inp, repaired_seed, occluded):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
