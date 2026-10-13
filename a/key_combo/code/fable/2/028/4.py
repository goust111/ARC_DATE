import numpy as np


BLACK = 0
INPUT_SIZE = 3
SLOT_SIZE = 3
OUTPUT_SIZE = INPUT_SIZE * SLOT_SIZE
PALETTE = np.arange(1, 10, dtype=int)
GEN_BRANCH_CYCLE = (1, 2, 3)
GEN_RETRY_LIMIT = 200

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Minimal opaque top-left paste aligned with arc_utils' (r, c) convention."""
    h, w = patch.shape
    dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch
    return dst


def _defect_mask(grid: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(grid)
    return grid != mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the left-right mirror of the 3x3 input into every output slot whose source cell disagrees with its mirror mate.

    Input structure:
    - The input is a 3x3 grid with no black cells.
    - The pattern is not fully left-right symmetric, so some cells match their mirror
      counterparts and some do not.

    Rule:
    - Reflect the input across the vertical centerline to obtain a mirrored 3x3 pattern.
    - Create a 9x9 black output divided into a 3x3 lattice of 3x3 block slots.
    - Compare the input and mirrored pattern cell by cell.
    - Whenever a cell differs from its mirrored value, stamp the full mirrored 3x3
      pattern into that cell's output slot; matching cells leave their slots black.

    Key insight:
    - The slot mask comes from self-comparison against the mirror rather than from any
      color class, and the stamped motif is the mirrored pattern itself rather than the
      original input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_SIZE, INPUT_SIZE), "input must be 3x3"
    assert np.all(g != BLACK), "input must be fully non-black"

    mirrored = np.fliplr(g)
    defect = _defect_mask(g)
    assert defect.any(), "input must contain at least one symmetry defect"
    assert (~defect).any(), "input must contain at least one symmetric cell"

    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BLACK, dtype=int)
    for r, c in np.argwhere(defect):
        paste(out, mirrored, int(r) * SLOT_SIZE, int(c) * SLOT_SIZE)
    return out


def _block(output: np.ndarray, r: int, c: int) -> np.ndarray:
    r0 = int(r) * SLOT_SIZE
    c0 = int(c) * SLOT_SIZE
    return output[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE]


def _recover_input_from_output(output: np.ndarray) -> np.ndarray:
    assert output.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    mirrored = None
    nonblack_slots = 0
    for r in range(INPUT_SIZE):
        for c in range(INPUT_SIZE):
            block = _block(output, r, c)
            if np.any(block != BLACK):
                nonblack_slots += 1
                if mirrored is None:
                    mirrored = block.copy()
                    assert np.all(mirrored != BLACK), "stamped motif must be fully non-black"
                else:
                    assert np.array_equal(block, mirrored), "all stamped slots must match"
            else:
                assert np.all(block == BLACK), "unstamped slots must stay black"
    assert nonblack_slots > 0, "at least one slot must be stamped"
    assert mirrored is not None
    recovered = np.fliplr(mirrored)
    assert np.array_equal(main(recovered), output), "output must uniquely recover the input"
    return recovered


def _validate_pair(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.shape == (INPUT_SIZE, INPUT_SIZE)
    assert out.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    assert np.all(inp != BLACK)

    mirrored = np.fliplr(inp)
    defect = _defect_mask(inp)
    assert defect.any()
    assert (~defect).any()
    assert not np.array_equal(inp, mirrored), "stamp must differ from the original pattern"
    assert len(np.unique(inp)) >= 3, "use enough visible color variety to avoid bland cases"

    defect_colors = set(map(int, inp[defect]))
    intact_colors = set(map(int, inp[~defect]))
    assert defect_colors & intact_colors, "at least one color should appear in both mask classes"
    for color in np.unique(inp):
        mask = (inp == int(color))
        assert not np.array_equal(mask, defect), "defect mask must not be reducible to one color"
        assert not np.array_equal(mask, ~defect), "intact mask must not be reducible to one color"

    stamped_slots = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    for r in range(INPUT_SIZE):
        for c in range(INPUT_SIZE):
            block = _block(out, r, c)
            if np.any(block != BLACK):
                stamped_slots[r, c] = True
                assert np.array_equal(block, mirrored), "stamped slot must contain the mirrored input"
            else:
                assert np.all(block == BLACK), "unstamped slot must stay black"
    assert np.array_equal(stamped_slots, defect), "slot selection must equal the self-comparison defect mask"

    recovered = _recover_input_from_output(out)
    assert np.array_equal(recovered, inp), "output should determine the input uniquely"


def _branch_count() -> int:
    global _GEN_CALL_COUNT
    count = int(GEN_BRANCH_CYCLE[_GEN_CALL_COUNT % len(GEN_BRANCH_CYCLE)])
    _GEN_CALL_COUNT += 1
    return count


def _pick_distinct(palette: np.ndarray, avoid: int | None = None) -> tuple[int, int]:
    if avoid is None:
        pair = _GEN_RNG.choice(palette, size=2, replace=False)
        return int(pair[0]), int(pair[1])
    options = palette[palette != int(avoid)]
    other = int(_GEN_RNG.choice(options))
    if _GEN_RNG.random() < 0.5:
        return int(avoid), other
    return other, int(avoid)


def _generate_candidate(mismatch_row_count: int) -> np.ndarray:
    palette_size = int(_GEN_RNG.integers(3, 6))
    palette = _GEN_RNG.choice(PALETTE, size=palette_size, replace=False)
    bridge_color = int(_GEN_RNG.choice(palette))
    mismatch_rows = set(map(int, _GEN_RNG.choice(INPUT_SIZE, size=mismatch_row_count, replace=False)))
    matched_rows = [r for r in range(INPUT_SIZE) if r not in mismatch_rows]
    bridge_defect_row = int(_GEN_RNG.choice(sorted(mismatch_rows)))
    bridge_nondefect_row = (
        int(_GEN_RNG.choice(matched_rows)) if matched_rows else int(_GEN_RNG.integers(INPUT_SIZE))
    )

    grid = np.empty((INPUT_SIZE, INPUT_SIZE), dtype=int)
    for r in range(INPUT_SIZE):
        center = int(_GEN_RNG.choice(palette))
        if r in mismatch_rows:
            if r == bridge_defect_row:
                left, right = _pick_distinct(palette, avoid=bridge_color)
            else:
                left, right = _pick_distinct(palette)
        else:
            side = bridge_color if r == bridge_nondefect_row else int(_GEN_RNG.choice(palette))
            left = right = side
        grid[r] = (left, center, right)

    if not matched_rows:
        grid[bridge_nondefect_row, 1] = bridge_color

    if _GEN_RNG.random() < 0.5:
        grid = np.flipud(grid)
    return grid


def generate_input() -> np.ndarray:
    mismatch_row_count = _branch_count()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_candidate(mismatch_row_count)
        try:
            _validate_pair(grid, main(grid))
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid symmetry-defect fractal input")
