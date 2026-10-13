from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


MARKER_COLOR = Color.RED
CONTROL_COLOR = Color.PINK
PALETTE = tuple(color for color in Color.ALL_COLORS if color not in (MARKER_COLOR, CONTROL_COLOR))
MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (8, 17)
WIDTH_RANGE = (8, 17)


def _marker_row(grid: np.ndarray) -> int:
    control = np.asarray(grid[:, 0], dtype=int)
    rows = np.flatnonzero(control == int(MARKER_COLOR))
    assert rows.size == 1, "expected exactly one red marker in the control column"
    marker_row = int(rows[0])
    mask = np.ones(control.shape[0], dtype=bool)
    mask[marker_row] = False
    assert bool(np.all(control[mask] == int(CONTROL_COLOR))), "non-marker control cells must be pink"
    return marker_row


def _single_foreground_color(row: np.ndarray, bg: int) -> tuple[int, int]:
    fg = np.asarray(row, dtype=int)
    fg = fg[fg != int(bg)]
    assert fg.size > 0, "expected at least one foreground pixel"
    colors = np.unique(fg)
    assert colors.size == 1, "expected a single foreground color on the selected row"
    return int(colors[0]), int(fg.size)


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int | np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected a control column and an image area"

    marker_row = _marker_row(g)
    image = g[:, 1:].copy()
    bg = int(get_bg(image))

    _shift_color, shift = _single_foreground_color(image[marker_row], bg)
    rolled = np.roll(image, shift=shift, axis=0)
    erase_color, _ = _single_foreground_color(rolled[marker_row], bg)

    out = rolled.copy()
    out[out == erase_color] = bg
    details: dict[str, int | np.ndarray] = {
        "marker_row": int(marker_row),
        "bg": int(bg),
        "shift": int(shift),
        "erase_color": int(erase_color),
        "rolled": rolled,
    }
    return out, details


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Drop the control column, roll the image downward by the marked row's foreground count,
    then erase the color occupying that same row after the roll.
    """
    return _solve_details(np.asarray(input_grid, dtype=int))[0]


def _paint_run(
    image: np.ndarray,
    row: int,
    start: int,
    length: int,
    color: int,
) -> None:
    image[int(row), int(start) : int(start) + int(length)] = int(color)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))

    bg = int(rng.choice([color for color in PALETTE if color != int(CONTROL_COLOR)]))
    usable_colors = [color for color in PALETTE if color != bg]
    shift_color = int(rng.choice(usable_colors))
    erase_color = int(rng.choice([color for color in usable_colors if color != shift_color]))

    marker_row = int(rng.integers(0, height))
    max_shift = min(width - 1, height - 1)
    shift = int(rng.integers(1, max_shift + 1))
    source_row = (marker_row - shift) % height

    image = np.full((height, width), bg, dtype=int)

    shift_cols = rng.choice(width, size=shift, replace=False)
    image[marker_row, shift_cols] = shift_color

    erase_count = int(rng.integers(1, max(2, width // 2) + 1))
    erase_cols = rng.choice(width, size=erase_count, replace=False)
    image[source_row, erase_cols] = erase_color

    other_rows = [row for row in range(height) if row not in (marker_row, source_row)]
    other_colors = [color for color in usable_colors if color != shift_color]
    for row in other_rows:
        if float(rng.random()) < 0.45:
            continue
        run_count = int(rng.integers(1, 3))
        for _ in range(run_count):
            run_len = int(rng.integers(1, min(4, width - 1) + 1))
            start = int(rng.integers(0, width - run_len + 1))
            color = int(rng.choice(other_colors))
            _paint_run(image, row, start, run_len, color)

    if other_rows:
        extra_row = int(rng.choice(other_rows))
        extra_len = int(rng.integers(1, min(3, width - 1) + 1))
        extra_start = int(rng.integers(0, width - extra_len + 1))
        _paint_run(image, extra_row, extra_start, extra_len, erase_color)

    grid = np.full((height, width + 1), CONTROL_COLOR, dtype=int)
    grid[:, 1:] = image
    grid[marker_row, 0] = int(MARKER_COLOR)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] > 30 or g.shape[1] > 30 or g.shape[1] < 2:
        return False

    try:
        out, details = _solve_details(g)
        marker_row = int(details["marker_row"])
        bg = int(details["bg"])
        shift = int(details["shift"])
        erase_color = int(details["erase_color"])
        rolled = np.asarray(details["rolled"], dtype=int)
        image = g[:, 1:]
        shift_color, counted = _single_foreground_color(image[marker_row], bg)
        landed_color, _ = _single_foreground_color(rolled[marker_row], bg)
    except AssertionError:
        return False

    if counted != shift:
        return False
    if (marker_row - shift) % g.shape[0] == marker_row:
        return False
    if shift_color == landed_color:
        return False
    if landed_color != erase_color:
        return False
    if int(np.count_nonzero(rolled == erase_color)) <= 1:
        return False

    expected = rolled.copy()
    expected[expected == erase_color] = bg
    if not np.array_equal(out, expected):
        return False
    if out.ndim != 2 or out.shape[0] > 30 or out.shape[1] > 30:
        return False
    if not np.any(out != bg):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if _validate_generated(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input grid")
