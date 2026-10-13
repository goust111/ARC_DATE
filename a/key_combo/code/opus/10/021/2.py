import random

import numpy as np


BACKGROUND = 0
STRIPE_COLOR = 5
SHIFT_BY_COLOR = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

MIN_STRIPE_HEIGHT = 9
MAX_STRIPE_HEIGHT = 29
MIN_WIDTH = 7
MAX_WIDTH = 15
GENERATION_RETRIES = 300


def _roll_column_manual(column: np.ndarray, shift: int) -> np.ndarray:
    h = int(column.shape[0])
    out = np.empty_like(column)
    for r in range(h):
        out[(r + shift) % h] = column[r]
    return out


def _make_stripe_body(height: int, width: int) -> np.ndarray:
    body = np.full((height, width), BACKGROUND, dtype=int)
    body[1::2, :] = STRIPE_COLOR
    return body


def _sample_control_row(width: int) -> np.ndarray:
    values = [0, 1, 2, 3, 4]
    extras = [random.choice(list(SHIFT_BY_COLOR)) for _ in range(width - len(values))]
    if extras and all(value == 0 for value in extras):
        extras[random.randrange(len(extras))] = random.choice([1, 2, 3, 4])
    values.extend(extras)

    for _ in range(20):
        random.shuffle(values)
        if values != sorted(values) and values != sorted(values, reverse=True):
            break

    if values[0] == 0 and values[-1] == 0:
        nonzero_positions = [idx for idx, value in enumerate(values) if value != 0]
        swap_idx = random.choice(nonzero_positions)
        edge_idx = random.choice([0, len(values) - 1])
        values[swap_idx], values[edge_idx] = values[edge_idx], values[swap_idx]

    return np.array(values, dtype=int)


def _manual_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    body = g[:-1]
    control = g[-1]
    out = np.empty_like(body)
    for c, color in enumerate(control):
        shift = SHIFT_BY_COLOR[int(color)]
        out[:, c] = _roll_column_manual(body[:, c], shift)
    return out


def _check_stripe_body(body: np.ndarray) -> None:
    expected = _make_stripe_body(body.shape[0], body.shape[1])
    assert np.array_equal(body, expected), "body must be alternating horizontal stripes"


def _validate_semantics(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 10 <= h <= 30
    assert 1 <= w <= 30

    body = g[:-1]
    control = g[-1]
    stripe_h = body.shape[0]
    assert stripe_h >= MIN_STRIPE_HEIGHT
    assert stripe_h % 2 == 1, "odd stripe height keeps all five shifts distinguishable"
    _check_stripe_body(body)

    assert set(np.unique(control)).issubset(set(SHIFT_BY_COLOR))
    nonzero = [int(value) for value in control if int(value) != 0]
    assert len(nonzero) >= 3
    assert len(set(nonzero)) >= 2
    assert set(control).issuperset({0, 1, 2, 3, 4})

    out = main(g)
    expected = _manual_expected(g)
    assert np.array_equal(out, expected)

    base_column = body[:, 0]
    signatures = {
        shift: tuple(int(v) for v in _roll_column_manual(base_column, shift))
        for shift in range(5)
    }
    assert len(set(signatures.values())) == 5

    for c, color in enumerate(control):
        sig = tuple(int(v) for v in out[:, c])
        assert sig == signatures[SHIFT_BY_COLOR[int(color)]]

    color_to_sig = {}
    for c, color in enumerate(control):
        color = int(color)
        sig = tuple(int(v) for v in out[:, c])
        if color in color_to_sig:
            assert color_to_sig[color] == sig
        else:
            color_to_sig[color] = sig
    assert len(set(color_to_sig.values())) == len(color_to_sig)


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the bottom control row and roll each stripe column downward by its marker-coded shift.

    Input structure:
    - The grid is black(0) and grey(5) horizontal stripes, plus a one-cell-tall control row at the bottom.
    - In the stripe area, even-indexed rows are black and odd-indexed rows are grey across the full width.
    - Each bottom-row cell is black(0) or one marker color among blue(1), red(2), green(3), yellow(4).

    Rule:
    - Remove the bottom control row from the output.
    - Decode each control-row color as a column-local cyclic downward shift: 0->0, 1->1, 2->2, 3->3, 4->4.
    - Independently roll every remaining column downward by its own shift amount, wrapping overflow to the top.

    Key insight:
    - The bottom markers do not select or recolor stripes; they numerically encode a separate vertical phase offset for each column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 2

    body = g[:-1]
    control = g[-1]
    out = np.empty_like(body)
    for c, color in enumerate(control):
        shift = SHIFT_BY_COLOR.get(int(color))
        assert shift is not None, "control row must use only colors 0..4"
        out[:, c] = np.roll(body[:, c], shift=shift)
    return out


def _generate_one() -> np.ndarray:
    stripe_height = random.choice(list(range(MIN_STRIPE_HEIGHT, MAX_STRIPE_HEIGHT + 1, 2)))
    width = random.randint(MIN_WIDTH, MAX_WIDTH)

    body = _make_stripe_body(stripe_height, width)
    control = _sample_control_row(width)
    grid = np.vstack([body, control[np.newaxis, :]])

    _validate_semantics(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            return _generate_one()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid stripe-shift ARC input")
