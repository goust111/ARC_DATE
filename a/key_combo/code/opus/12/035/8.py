from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

ABSTRACT_ROWS = (0, 1)
ABSTRACT_COLS = (0, 1, 2)
ABSTRACT_STATES = [(r, c) for r in ABSTRACT_ROWS for c in ABSTRACT_COLS]
ABSTRACT_VELOCITIES = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]

MAX_GENERATION_ATTEMPTS = 300
TYPE_WEIGHTS = {
    "straight": 2,
    "observed_row": 2,
    "observed_col": 2,
    "bounce5_row": 2,
    "bounce5_col": 2,
}


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _panel_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    seps = _separator_columns(grid)
    assert len(seps) == 5, "expected 4 shared-border panels"
    strides = [b - a for a, b in zip(seps, seps[1:])]
    assert len(set(strides)) == 1, "panel spacing must be regular"
    return [(seps[i], seps[i + 1]) for i in range(4)]


def _find_bar(panel: np.ndarray) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    h, w = panel.shape
    for r in range(h):
        for c in range(w - 4):
            if np.all(panel[r, c : c + 5] == BLUE):
                matches.append((r, c))
    assert len(matches) == 1, "expected one horizontal blue bar"
    return matches[0]


def _clean_panel(panel: np.ndarray) -> np.ndarray:
    cleaned = panel.copy()
    cleaned[cleaned == RED] = BLACK
    return cleaned


def _actual_to_state(actual: tuple[int, int], bar_row: int, bar_start: int) -> tuple[int, int]:
    r, c = actual
    if r == bar_row - 1:
        sr = 0
    else:
        assert r == bar_row + 1, "red sprite must sit on one of the two interior motion rows"
        sr = 1
    sc = c - (bar_start + 1)
    assert sc in ABSTRACT_COLS, "red sprite must sit inside the diamond interior"
    return sr, int(sc)


def _state_to_actual(state: tuple[int, int], bar_row: int, bar_start: int) -> tuple[int, int]:
    sr, sc = state
    return bar_row - 1 + 2 * sr, bar_start + 1 + sc


def _extract_state(panel: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(panel == RED)
    assert coords.shape == (1, 2), "expected exactly one red sprite per panel"
    bar_row, bar_start = _find_bar(panel)
    actual = tuple(map(int, coords[0]))
    return _actual_to_state(actual, bar_row, bar_start), (bar_row, bar_start)


def _step_state(
    state: tuple[int, int], velocity: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], str | None] | None:
    sr, sc = state
    dr, dc = velocity
    nr = sr + dr
    nc = sc + dc
    hit_row = nr not in ABSTRACT_ROWS
    hit_col = nc not in ABSTRACT_COLS
    if hit_row and hit_col:
        return None
    if hit_row:
        dr = -dr
    if hit_col:
        dc = -dc
    nxt = (sr + dr, sc + dc)
    if nxt[0] not in ABSTRACT_ROWS or nxt[1] not in ABSTRACT_COLS:
        return None
    if (dr, dc) == (0, 0):
        return None
    axis = "row" if hit_row else "col" if hit_col else None
    return nxt, (dr, dc), axis


def _build_panel(
    height: int,
    panel_w: int,
    state: tuple[int, int] | None = None,
    *,
    bar_row: int | None = None,
    bar_start: int | None = None,
) -> np.ndarray:
    panel = np.full((height, panel_w), BLACK, dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY

    interior_w = panel_w - 2
    if bar_row is None:
        bar_row = 1 + (height - 2) // 2
    if bar_start is None:
        bar_start = 1 + (interior_w - 5) // 2
    bar_end = bar_start + 4

    panel[bar_row, bar_start : bar_end + 1] = BLUE
    panel[bar_row - 2, bar_start + 2] = TEAL
    panel[bar_row - 1, bar_start] = TEAL
    panel[bar_row - 1, bar_end] = TEAL
    panel[bar_row + 1, bar_start] = TEAL
    panel[bar_row + 1, bar_end] = TEAL
    panel[bar_row + 2, bar_start + 2] = TEAL

    if state is not None:
        rr, cc = _state_to_actual(state, bar_row, bar_start)
        panel[rr, cc] = RED
    return panel


def _compose_input(
    height: int,
    panel_w: int,
    states: list[tuple[int, int]],
    *,
    bar_row: int,
    bar_start: int,
) -> np.ndarray:
    stride = panel_w - 1
    width = 4 * panel_w - 3
    grid = np.full((height, width), BLACK, dtype=int)
    for i, state in enumerate(states):
        start = i * stride
        grid[:, start : start + panel_w] = _build_panel(
            height,
            panel_w,
            state,
            bar_row=bar_row,
            bar_start=bar_start,
        )
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the red sprite one frame forward inside a static diamond envelope with mirror reflections.

    Input structure:
    - The grid is a row of 4 equal-width panels sharing gray vertical border columns.
    - Every panel contains the same static background: a gray frame, a central blue horizontal bar, and a teal diamond outline around that bar.
    - Each panel also contains one red single-pixel sprite at a background location inside the diamond interior.

    Rule:
    - Read the red sprite position in each panel using panel-local coordinates.
    - Treat the 4 panels as consecutive frames of constant-velocity motion on the diamond's 2x3 interior lattice.
    - When a step would leave the top or bottom of that lattice, flip the row-velocity component; when it would leave the left or right side, flip the column-velocity component; then complete the reflected step.
    - Output a single standalone panel with the same static background and the extrapolated 5th-frame red sprite position.

    Key insight:
    - The teal-and-blue diamond is not decoration: it defines which velocity component reflects, so the next-frame position can only be inferred by combining motion extrapolation with the envelope geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g[0, :] == GREY) and np.all(g[-1, :] == GREY), "expected gray top and bottom frame rows"

    panels = [g[:, c0 : c1 + 1] for c0, c1 in _panel_slices(g)]
    cleaned = [_clean_panel(panel) for panel in panels]
    template = cleaned[0]
    for panel in cleaned[1:]:
        assert np.array_equal(panel, template), "static panel background must match across all frames"

    states: list[tuple[int, int]] = []
    bar_info: tuple[int, int] | None = None
    for panel in panels:
        state, info = _extract_state(panel)
        states.append(state)
        if bar_info is None:
            bar_info = info
        else:
            assert bar_info == info, "all panels must share the same bar geometry"

    deltas = [
        (states[i + 1][0] - states[i][0], states[i + 1][1] - states[i][1])
        for i in range(3)
    ]
    assert all(delta != (0, 0) for delta in deltas), "sprite must move between frames"

    change_count = sum(deltas[i] != deltas[i - 1] for i in range(1, len(deltas)))
    assert change_count <= 1, "at most one reflection may appear in the observed frames"
    velocity = deltas[-1]

    stepped = _step_state(states[-1], velocity)
    assert stepped is not None, "final extrapolation step must be well-defined"
    next_state, _next_velocity, _axis = stepped

    out = template.copy()
    assert bar_info is not None
    rr, cc = _state_to_actual(next_state, *bar_info)
    out[rr, cc] = RED
    return out


def _simulate_trajectory(
    start: tuple[int, int], velocity: tuple[int, int], steps: int = 4
) -> tuple[list[tuple[int, int]], list[str | None]] | None:
    states = [start]
    axes: list[str | None] = []
    cur = start
    vel = velocity
    for _ in range(steps):
        stepped = _step_state(cur, vel)
        if stepped is None:
            return None
        cur, vel, axis = stepped
        states.append(cur)
        axes.append(axis)
    return states, axes


def _trajectory_catalog() -> dict[str, list[list[tuple[int, int]]]]:
    catalog = {name: [] for name in TYPE_WEIGHTS}
    for start in ABSTRACT_STATES:
        for velocity in ABSTRACT_VELOCITIES:
            simulated = _simulate_trajectory(start, velocity, steps=4)
            if simulated is None:
                continue
            states, axes = simulated
            observed = [axis for axis in axes[:3] if axis is not None]
            total_reflections = [axis for axis in axes if axis is not None]
            if len(observed) > 1:
                continue
            if len(total_reflections) > 1:
                continue

            if not total_reflections:
                catalog["straight"].append(states)
            elif len(observed) == 1:
                catalog[f"observed_{observed[0]}"].append(states)
            else:
                assert axes[3] is not None
                catalog[f"bounce5_{axes[3]}"].append(states)
    return catalog


TRAJECTORY_CATALOG = _trajectory_catalog()


def _choose_trajectory_type() -> str:
    available = [(name, weight) for name, weight in TYPE_WEIGHTS.items() if TRAJECTORY_CATALOG[name]]
    labels = [name for name, _ in available]
    weights = [weight for _, weight in available]
    return random.choices(labels, weights=weights, k=1)[0]


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(9, 20)
    panel_w = random.choice((7, 8))
    bar_row = random.randint(3, panel_h - 4)
    bar_start = random.randint(1, panel_w - 6)
    kind = _choose_trajectory_type()
    states = random.choice(TRAJECTORY_CATALOG[kind])

    inp = _compose_input(
        panel_h,
        panel_w,
        states[:4],
        bar_row=bar_row,
        bar_start=bar_start,
    )
    expected = _build_panel(
        panel_h,
        panel_w,
        states[4],
        bar_row=bar_row,
        bar_start=bar_start,
    )
    try:
        observed = main(inp)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
