import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEMPLATE_COLOR_POOL = (3, 4, 5, 6, 7, 8, 9)

MIN_HEIGHT = 7
MAX_HEIGHT = 18
MIN_WIDTH = 6
MAX_WIDTH = 20
MAX_TEMPLATE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 200

BLANK_STATE = None
FORWARD_0 = (0, RED)
FLIPPED_0 = (0, BLUE)
FORWARD_1 = (1, RED)
FLIPPED_1 = (1, BLUE)
ALL_STATES = (BLANK_STATE, FORWARD_0, FLIPPED_0, FORWARD_1, FLIPPED_1)


def _run_count(mask: np.ndarray) -> int:
    count = 0
    prev = False
    for value in mask.tolist():
        current = bool(value)
        if current and not prev:
            count += 1
        prev = current
    return count


def _make_template_row(length: int, color: int) -> np.ndarray:
    assert length >= 4
    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        density = random.uniform(0.25, 0.65)
        mask = np.array([random.random() < density for _ in range(length)], dtype=bool)
        if int(mask.sum()) < 2 or int(mask.sum()) > length - 1:
            continue
        if np.array_equal(mask, mask[::-1]):
            continue
        if _run_count(mask) > 3:
            continue
        row = np.full(length, BLACK, dtype=int)
        row[mask] = int(color)
        return row
    raise RuntimeError("failed to create a non-symmetric template row")


def _decode_instruction(signal_cells: np.ndarray) -> tuple[int | None, bool]:
    non_black = np.flatnonzero(signal_cells != BLACK)
    assert len(non_black) <= 1, "instruction zone rows must have at most one colored signal"
    if len(non_black) == 0:
        return None, False
    col = int(non_black[0])
    color = int(signal_cells[col])
    assert color in (RED, BLUE), "signals must be red or blue"
    return col, color == BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read two top-row templates and stamp them into lower rows from red/blue column signals.

    Input structure:
    - Black is the background.
    - The top 2 rows form the template zone, with a black 2x2 overlap at the top-left corner.
    - The left 2 columns form the instruction zone; each lower row is either all black there or contains one red/blue signal pixel.
    - The lower-right area starts black and is the output area to be filled row by row.

    Rule:
    - Template row 0 is selected by a signal in instruction column 0, and template row 1 is selected by a signal in column 1.
    - For each row below the template zone, keep the output area black when both instruction cells are black.
    - If the selected signal is red, copy that template row into the output area unchanged.
    - If the selected signal is blue, copy the selected template row reversed left-to-right.
    - Leave the original template rows and instruction columns unchanged.

    Key insight:
    - The signal column chooses which template to use, while the red/blue signal color chooses normal versus mirrored copying.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 3 and width >= 3
    assert np.all(g[:2, :2] == BLACK), "template/instruction overlap must be black"

    templates = [g[0, 2:].copy(), g[1, 2:].copy()]
    out = g.copy()
    for r in range(2, height):
        template_idx, flipped = _decode_instruction(g[r, :2])
        out[r, 2:] = BLACK
        if template_idx is None:
            continue
        template = templates[template_idx]
        out[r, 2:] = template[::-1] if flipped else template
    return out


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    height, width = grid.shape
    assert 0 < height <= 30 and 0 < width <= 30
    assert height >= 7 and width >= 6
    assert np.all(grid[:2, :2] == BLACK)

    seen_states = {state: 0 for state in ALL_STATES}
    for template_row in (grid[0, 2:], grid[1, 2:]):
        colors = {int(v) for v in template_row.tolist() if int(v) != BLACK}
        assert len(colors) == 1, "each template row should use one visible color"
        assert colors.isdisjoint({RED, BLUE}), "template rows must avoid signal colors"
        assert not np.array_equal(template_row, template_row[::-1]), "template rows must distinguish mirrored copies"

    for r in range(2, height):
        signal = grid[r, :2]
        template_idx, flipped = _decode_instruction(signal)
        assert np.all(grid[r, 2:] == BLACK), "output area must start black"
        if template_idx is None:
            seen_states[BLANK_STATE] += 1
        elif template_idx == 0 and not flipped:
            seen_states[FORWARD_0] += 1
        elif template_idx == 0 and flipped:
            seen_states[FLIPPED_0] += 1
        elif template_idx == 1 and not flipped:
            seen_states[FORWARD_1] += 1
        else:
            seen_states[FLIPPED_1] += 1

    assert all(count >= 1 for count in seen_states.values()), "every semantic branch should appear"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
        width = random.randint(MIN_WIDTH, MAX_WIDTH)
        template_width = width - 2

        color0, color1 = random.sample(TEMPLATE_COLOR_POOL, 2)
        template0 = _make_template_row(template_width, color0)
        template1 = _make_template_row(template_width, color1)

        grid = np.full((height, width), BLACK, dtype=int)
        grid[0, 2:] = template0
        grid[1, 2:] = template1

        output_rows = height - 2
        states = list(ALL_STATES)
        states.extend(random.choice(ALL_STATES) for _ in range(output_rows - len(ALL_STATES)))
        random.shuffle(states)

        for r, state in enumerate(states, start=2):
            if state is BLANK_STATE:
                continue
            col, color = state
            grid[r, col] = int(color)

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
