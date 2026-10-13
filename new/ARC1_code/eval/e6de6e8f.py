import numpy as np

from arc_utils import Color, crop_bbox, find_components
import random


def _glyphs(g: np.ndarray) -> list[int | None]:
    comps = find_components(g, bg=Color.BLACK, colors=[Color.RED], connectivity=4)
    comps = sorted(comps, key=lambda comp: comp["bbox"][1])

    glyphs: list[int | None] = []
    for comp in comps:
        r0, _c0, r1, _c1 = comp["bbox"]
        assert (r0, r1) == (0, 1)

        mask = crop_bbox(g == Color.RED, comp["bbox"])
        assert mask.shape[0] == 2
        assert mask.shape[1] in (1, 2)

        top_cols = np.flatnonzero(mask[0])
        bottom_cols = np.flatnonzero(mask[1])
        assert len(top_cols) == 1
        assert len(bottom_cols) in (1, 2)

        entry = int(top_cols[0])
        bottom = [int(c) for c in bottom_cols]
        assert entry in bottom

        if len(bottom) == 1:
            glyphs.append(None)
            continue

        exit_cols = [c for c in bottom if c != entry]
        assert len(exit_cols) == 1
        step = exit_cols[0] - entry
        assert abs(step) == 1
        glyphs.append(step)

    assert glyphs
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the two-row red glyph strip into a downward red trace starting from a centered green marker.

    Input structure
    - The input is a black 2-row strip containing 4-connected red glyphs, read left to right.
    - Each glyph spans both rows and has exactly one top red entry cell; its bottom row is either the same cell only or a two-cell elbow.
    Rule
    - Make a black output whose width is one more than half the input width and whose height is one marker row plus one row for each elbow and two rows for each vertical glyph.
    - Put a green marker in the center of the top row, then trace the glyphs downward: a vertical glyph paints two red cells in the current column; an elbow paints the two bottom cells as a horizontal red segment and moves the current column left or right.
    Key insight
    - The missing top cell of each 2x2 elbow tells whether the trace exits left or right; the glyphs are pipe instructions rather than shapes to copy literally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] == 2
    assert g.shape[1] % 2 == 0
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.RED}

    glyphs = _glyphs(g)
    out_h = 1 + sum(2 if step is None else 1 for step in glyphs)
    out_w = g.shape[1] // 2 + 1
    out = np.full((out_h, out_w), Color.BLACK, dtype=int)

    r = 0
    c = out_w // 2
    out[r, c] = Color.GREEN
    r += 1

    for step in glyphs:
        if step is None:
            assert r + 1 < out_h
            out[r : r + 2, c] = Color.RED
            r += 2
        else:
            next_c = c + step
            assert 0 <= next_c < out_w
            out[r, c] = Color.RED
            out[r, next_c] = Color.RED
            r += 1
            c = next_c

    assert r == out_h
    return out

# --------------- generator (o5x) ---------------
# Glyph sequence, canvas, and retry configuration.
GEN_GLYPH_COUNT = 5
GEN_NUM_VERTICAL = 2
GEN_STEP_CHOICES = (-1, 1)
GEN_INPUT_ROWS = 2
GEN_INPUT_WIDTH = 3 * GEN_GLYPH_COUNT - GEN_NUM_VERTICAL - 1
GEN_OUTPUT_WIDTH = GEN_INPUT_WIDTH // 2 + 1
GEN_RETRY_LIMIT = 300


def _sample_steps() -> list[int | None] | None:
    """Sample glyph instructions and reject paths that leave the output."""
    col = GEN_OUTPUT_WIDTH // 2
    vertical_positions = set(random.sample(range(GEN_GLYPH_COUNT), GEN_NUM_VERTICAL))
    steps: list[int | None] = []

    for pos in range(GEN_GLYPH_COUNT):
        if pos in vertical_positions:
            steps.append(None)
            continue

        choices = list(GEN_STEP_CHOICES)
        random.shuffle(choices)
        valid = [step for step in choices if 0 <= col + step < GEN_OUTPUT_WIDTH]
        if not valid:
            return None
        step = random.choice(valid)
        steps.append(step)
        col += step
    return steps


def _encode_input(steps: list[int | None]) -> np.ndarray:
    """Encode sampled instructions into the two-row glyph strip."""
    grid = np.full((GEN_INPUT_ROWS, GEN_INPUT_WIDTH), Color.BLACK, dtype=int)
    c = 0
    for step in steps:
        if step is None:
            grid[:, c] = Color.RED
            c += 1
        else:
            top_offset = 0 if step == 1 else 1
            grid[0, c + top_offset] = Color.RED
            grid[1, c : c + 2] = Color.RED
            c += 2
        if c < GEN_INPUT_WIDTH:
            c += 1
    return grid


def _render_expected(steps: list[int | None]) -> np.ndarray:
    """Render the decoded output used to validate the generated strip."""
    out_h = 1 + sum(2 if step is None else 1 for step in steps)
    expected = np.full((out_h, GEN_OUTPUT_WIDTH), Color.BLACK, dtype=int)
    r = 0
    col = GEN_OUTPUT_WIDTH // 2
    expected[r, col] = Color.GREEN
    r += 1
    for step in steps:
        if step is None:
            expected[r : r + 2, col] = Color.RED
            r += 2
        else:
            expected[r, col] = Color.RED
            expected[r, col + step] = Color.RED
            col += step
            r += 1
    return expected


def _gen_attempt() -> np.ndarray | None:
    steps = _sample_steps()
    if steps is None:
        return None
    grid = _encode_input(steps)
    return grid if np.array_equal(main(grid), _render_expected(steps)) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
