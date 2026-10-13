import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = (1, 2, 3, 4)
ORDER_POSITIONS = ((0, 0), (0, 1), (1, 1), (1, 0))  # TL, TR, BR, BL
GEN_ATTEMPTS = 300


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """ARC-style inclusive bbox helper using the shared (r, c) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def rotate_order_clockwise(order: Iterable[int]) -> tuple[int, int, int, int]:
    """Move each color one clockwise step around (TL, TR, BR, BL)."""
    tl, tr, br, bl = tuple(int(v) for v in order)
    return bl, tl, tr, br


def extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 4
    assert np.all(g[0, :] == FRAME) and np.all(g[-1, :] == FRAME), "top/bottom borders must be gray"

    separators = [c for c in range(w) if np.all(g[:, c] == FRAME)]
    assert len(separators) >= 4, "need at least three shared-border panels"
    assert separators[0] == 0 and separators[-1] == w - 1, "shared borders must include both outer edges"

    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert strides and len(set(strides)) == 1, "panel separators must be equally spaced"
    stride = int(strides[0])
    panel_w = stride + 1

    panels = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[:, 0] == FRAME) and np.all(panel[:, -1] == FRAME), "panel side borders must be gray"
        panels.append(panel)
    return panels, panel_w


def extract_sprite(panel: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]]:
    mask = (panel != BG) & (panel != FRAME)
    coords = np.argwhere(mask)
    assert coords.shape == (4, 2), "sprite must contain exactly four colored pixels"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "sprite must occupy a 2x2 box"
    expected = {(r0 + dr, c0 + dc) for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))}
    assert expected == {tuple(map(int, rc)) for rc in coords}, "sprite bbox must be fully filled"
    order = (
        int(panel[r0, c0]),
        int(panel[r0, c1]),
        int(panel[r1, c1]),
        int(panel[r1, c0]),
    )
    assert set(order) == set(SPRITE_COLORS), "sprite must use blue/red/green/yellow exactly once"
    return r0 - 1, c0 - 1, order


def build_output(panel_h: int, panel_w: int, row_off: int, col_off: int, order: tuple[int, int, int, int]) -> np.ndarray:
    out = np.full((panel_h, panel_w), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME

    top = 1 + int(row_off)
    left = 1 + int(col_off)
    for color, (dr, dc) in zip(order, ORDER_POSITIONS):
        out[top + dr, left + dc] = int(color)
    return out


def parse_sequence(grid: np.ndarray) -> tuple[int, int, int, int, tuple[int, int, int, int]]:
    panels, panel_w = extract_panels(grid)
    panel_h = int(panels[0].shape[0])
    interior_h = panel_h - 2
    interior_w = panel_w - 2

    rows: list[int] = []
    cols: list[int] = []
    orders: list[tuple[int, int, int, int]] = []
    for panel in panels:
        row_off, col_off, order = extract_sprite(panel)
        rows.append(int(row_off))
        cols.append(int(col_off))
        orders.append(order)

    assert len(set(cols)) == 1, "sprite column offset must stay fixed across frames"
    assert all(0 <= r <= interior_h - 2 for r in rows)
    assert all(0 <= c <= interior_w - 2 for c in cols)

    deltas = [b - a for a, b in zip(rows, rows[1:])]
    assert deltas and len(set(deltas)) == 1, "row offsets must follow constant motion"
    delta_r = int(deltas[0])
    assert delta_r != 0, "motion must be non-zero"

    for prev, curr in zip(orders, orders[1:]):
        assert curr == rotate_order_clockwise(prev), "colors must rotate one clockwise step per frame"

    next_row = rows[-1] + delta_r
    next_col = cols[0]
    assert 0 <= next_row <= interior_h - 2, "extrapolated sprite must stay inside the output panel"
    next_order = rotate_order_clockwise(orders[-1])
    return panel_h, panel_w, next_row, next_col, next_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate a moving 2x2 color-cycling sprite from shared-border frame panels into the next standalone panel.

    Input structure:
    - The grid is a horizontal row of 3-4 equal-height gray framed panels sharing their vertical border columns.
    - Each panel has black interior cells and contains one fully visible 2x2 sprite using blue, red, green, and yellow once each.
    - The sprite keeps the same local columns in every panel while its local top row changes by a constant amount.

    Rule:
    - Read the shared-border panel sequence and recover both the constant row step and the clockwise color-cycle order around the 2x2 sprite positions (TL, TR, BR, BL).
    - Build one standalone output panel with the same size, gray border, and black interior.
    - Place the sprite one frame later: advance the row by the same constant step, keep the local columns unchanged, and rotate the four colors one more clockwise step.

    Key insight:
    - The panel sequence simultaneously encodes two independent regularities, motion and color cycling; the next frame is correct only when both are extrapolated together.
    """
    g = np.asarray(grid, dtype=int)
    panel_h, panel_w, next_row, next_col, next_order = parse_sequence(g)
    return build_output(panel_h, panel_w, next_row, next_col, next_order)


def _render_input(panel_h: int, interior_w: int, panel_count: int, start_row: int, delta_r: int, col_off: int, start_order: tuple[int, int, int, int]) -> np.ndarray:
    stride = interior_w + 1
    grid_w = panel_count * stride + 1
    grid = np.full((panel_h, grid_w), BG, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME

    for k in range(panel_count + 1):
        c = k * stride
        grid[:, c] = FRAME

    order = tuple(int(v) for v in start_order)
    for k in range(panel_count):
        row_off = start_row + k * delta_r
        top = 1 + row_off
        left = 1 + k * stride + col_off
        for color, (dr, dc) in zip(order, ORDER_POSITIONS):
            grid[top + dr, left + dc] = int(color)
        order = rotate_order_clockwise(order)
    return grid


def validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert isinstance(g, np.ndarray) and g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BG, FRAME, *SPRITE_COLORS})

    panel_h, panel_w, next_row, next_col, next_order = parse_sequence(g)
    out = main(g)
    assert out.shape == (panel_h, panel_w)
    assert np.all(out[0, :] == FRAME) and np.all(out[-1, :] == FRAME)
    assert np.all(out[:, 0] == FRAME) and np.all(out[:, -1] == FRAME)

    rows = []
    orders = []
    for panel in extract_panels(g)[0]:
        row_off, col_off, order = extract_sprite(panel)
        rows.append(row_off)
        orders.append(order)
        assert col_off == next_col

    assert next_row != rows[-1], "position extrapolation must change the output"
    assert next_order != orders[-1], "color extrapolation must change the output"
    assert len(set(rows)) == len(rows), "monotone non-zero motion should prevent repeated frame positions"

    expected = build_output(panel_h, panel_w, next_row, next_col, next_order)
    assert np.array_equal(out, expected)

    last_panel = extract_panels(g)[0][-1]
    assert not np.array_equal(out, last_panel), "output must not duplicate the last observed frame"


def _valid_motion_choices(interior_h: int, panel_count: int) -> list[tuple[int, int]]:
    max_row = interior_h - 2
    choices: list[tuple[int, int]] = []
    for delta_r in (-2, -1, 1, 2):
        for start_row in range(max_row + 1):
            rows = [start_row + k * delta_r for k in range(panel_count + 1)]
            if all(0 <= row <= max_row for row in rows):
                choices.append((start_row, delta_r))
    return choices


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice((3, 4))
    panel_h = random.randint(8, 10)
    interior_h = panel_h - 2

    max_interior_w = min(6, (30 - 1) // panel_count - 1)
    interior_w = random.randint(4, max_interior_w)

    motion_choices = _valid_motion_choices(interior_h, panel_count)
    if not motion_choices:
        return None

    start_row, delta_r = random.choice(motion_choices)
    col_off = random.randint(0, interior_w - 2)
    start_order = tuple(random.sample(SPRITE_COLORS, len(SPRITE_COLORS)))

    grid = _render_input(panel_h, interior_w, panel_count, start_row, delta_r, col_off, start_order)
    try:
        validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
