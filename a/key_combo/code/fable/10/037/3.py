import random
from typing import Iterable

import numpy as np


SEPARATOR_THICKNESS = 1
TOTAL_PANELS = 4  # three visible history frames + one blank template slot
BG_COLOR = 0
GENERATION_RETRIES = 600
DIRS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MODE_CHOICES = ("bounce_forecast", "collision_history", "collision_forecast")


def _full_line_rows(grid: np.ndarray, color: int) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]


def _full_line_cols(grid: np.ndarray, color: int) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _segment_lengths(cuts: list[int], total: int) -> list[int]:
    lengths: list[int] = []
    start = 0
    for cut in cuts:
        lengths.append(cut - start)
        start = cut + 1
    lengths.append(total - start)
    return lengths


def _valid_panel_cuts(cuts: list[int], total: int) -> bool:
    if len(cuts) != TOTAL_PANELS - 1:
        return False
    if any(b - a <= 1 for a, b in zip(cuts, cuts[1:])):
        return False
    lengths = _segment_lengths(cuts, total)
    return min(lengths) >= 3 and len(set(lengths)) == 1


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], str, int]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g)]
    candidates: list[tuple[str, int, list[np.ndarray]]] = []

    for color in colors:
        row_cuts = _full_line_rows(g, color)
        if _valid_panel_cuts(row_cuts, g.shape[0]):
            parts = np.split(g, row_cuts, axis=0)
            panels = [part if i == 0 else part[1:, :] for i, part in enumerate(parts)]
            if all(panel.shape == panels[0].shape for panel in panels) and not any(np.any(panel == color) for panel in panels):
                candidates.append(("vertical", color, panels))

        col_cuts = _full_line_cols(g, color)
        if _valid_panel_cuts(col_cuts, g.shape[1]):
            parts = np.split(g, col_cuts, axis=1)
            panels = [part if i == 0 else part[:, 1:] for i, part in enumerate(parts)]
            if all(panel.shape == panels[0].shape for panel in panels) and not any(np.any(panel == color) for panel in panels):
                candidates.append(("horizontal", color, panels))

    assert len(candidates) == 1, "expected exactly one separator scheme"
    orientation, sep_color, panels = candidates[0]
    return panels, orientation, int(sep_color)


def _template_palette(template: np.ndarray) -> set[int]:
    return {int(c) for c in np.unique(template)}


def _dot_position(panel: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(panel == color)
    assert coords.shape[0] == 1, "each moving color must occupy exactly one cell"
    r, c = map(int, coords[0])
    return r - 1, c - 1


def _bounce_velocity(pos: tuple[int, int], vel: tuple[int, int], shape: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    r, c = pos
    dr, dc = vel
    h, w = shape
    bounced = False
    if not (0 <= r + dr < h):
        dr = -dr
        bounced = True
    if not (0 <= c + dc < w):
        dc = -dc
        bounced = True
    return (dr, dc), bounced


def _advance_state(
    positions: list[tuple[int, int]], velocities: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    bounced_vels: list[tuple[int, int]] = []
    bounce_flags: list[bool] = []
    for pos, vel in zip(positions, velocities):
        bounced, did_bounce = _bounce_velocity(pos, vel, shape)
        bounced_vels.append(bounced)
        bounce_flags.append(did_bounce)

    candidates = [
        (positions[i][0] + bounced_vels[i][0], positions[i][1] + bounced_vels[i][1]) for i in range(2)
    ]
    collision = candidates[0] == candidates[1]
    if collision:
        swapped_vels = [bounced_vels[1], bounced_vels[0]]
        bounced_vels = []
        for i in range(2):
            rebounced, did_rebounce = _bounce_velocity(positions[i], swapped_vels[i], shape)
            bounced_vels.append(rebounced)
            bounce_flags[i] = bounce_flags[i] or did_rebounce
        candidates = [
            (positions[i][0] + bounced_vels[i][0], positions[i][1] + bounced_vels[i][1]) for i in range(2)
        ]
        assert candidates[0] != candidates[1], "swap-before-move should separate the two dots"

    for r, c in candidates:
        assert 0 <= r < shape[0] and 0 <= c < shape[1]

    return candidates, bounced_vels, {"bounce": tuple(bounce_flags), "collision": collision}


def _build_template(panel_shape: tuple[int, int], border: int) -> np.ndarray:
    panel = np.full(panel_shape, BG_COLOR, dtype=int)
    panel[0, :] = border
    panel[-1, :] = border
    panel[:, 0] = border
    panel[:, -1] = border
    return panel


def _render_panel(template: np.ndarray, colors: list[int], positions: list[tuple[int, int]]) -> np.ndarray:
    panel = template.copy()
    for color, (r, c) in zip(colors, positions):
        panel[r + 1, c + 1] = color
    return panel


def _assemble_panels(panels: Iterable[np.ndarray], orientation: str, sep_color: int) -> np.ndarray:
    panel_list = [np.asarray(panel, dtype=int) for panel in panels]
    if orientation == "horizontal":
        pieces = []
        for i, panel in enumerate(panel_list):
            if i:
                pieces.append(np.full((panel.shape[0], SEPARATOR_THICKNESS), sep_color, dtype=int))
            pieces.append(panel)
        return np.concatenate(pieces, axis=1)

    pieces = []
    for i, panel in enumerate(panel_list):
        if i:
            pieces.append(np.full((SEPARATOR_THICKNESS, panel.shape[1]), sep_color, dtype=int))
        pieces.append(panel)
    return np.concatenate(pieces, axis=0)


def _naive_next_positions(history: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    prev_pos, last_pos = history[-2], history[-1]
    out = []
    for i in range(2):
        dr = last_pos[i][0] - prev_pos[i][0]
        dc = last_pos[i][1] - prev_pos[i][1]
        out.append((last_pos[i][0] + dr, last_pos[i][1] + dc))
    return out


def _state_signature(positions: list[tuple[int, int]], velocities: list[tuple[int, int]]) -> tuple:
    return tuple(positions + velocities)


def _simulate_candidate(
    inner_shape: tuple[int, int], mode: str
) -> tuple[list[list[tuple[int, int]]], list[dict], list[int], int] | None:
    h, w = inner_shape
    border = random.choice([c for c in range(1, 10)])
    sep_choices = [c for c in range(1, 10) if c != border]
    sep_color = random.choice(sep_choices)
    dot_colors = random.sample([c for c in range(1, 10) if c not in (border, sep_color)], 2)

    positions = random.sample([(r, c) for r in range(h) for c in range(w)], 2)
    velocities = [random.choice(DIRS_8), random.choice(DIRS_8)]

    history = [list(positions)]
    velocities_now = list(velocities)
    events: list[dict] = []
    seen = {_state_signature(history[0], velocities_now)}

    for _ in range(TOTAL_PANELS - 1):
        try:
            positions, velocities_now, event = _advance_state(history[-1], velocities_now, inner_shape)
        except AssertionError:
            return None
        sig = _state_signature(positions, velocities_now)
        if sig in seen:
            return None
        seen.add(sig)
        history.append(list(positions))
        events.append(event)

    history_bounce = any(any(step["bounce"]) for step in events[:-1])
    history_collision = any(step["collision"] for step in events[:-1])
    forecast_collision = bool(events[-1]["collision"])
    forecast_bounce = any(events[-1]["bounce"])
    naive_next = _naive_next_positions(history[:-1])
    forecast_non_linear = any(naive_next[i] != history[-1][i] for i in range(2))

    if not history_bounce or not forecast_non_linear:
        return None

    if mode == "bounce_forecast":
        if not forecast_bounce or forecast_collision or history_collision:
            return None
    elif mode == "collision_history":
        if not history_collision or forecast_collision or not forecast_bounce:
            return None
    elif mode == "collision_forecast":
        if not forecast_collision:
            return None
    else:
        raise AssertionError("unknown generation mode")

    if len({tuple(frame) for frame in history}) != len(history):
        return None

    return history, events, dot_colors, sep_color


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(("horizontal", "vertical"))
    mode = random.choice(MODE_CHOICES)

    if orientation == "horizontal":
        panel_h = random.randint(6, 9)
        panel_w = 6
    else:
        panel_h = 6
        panel_w = random.randint(6, 9)

    inner_shape = (panel_h - 2, panel_w - 2)
    sim = _simulate_candidate(inner_shape, mode)
    if sim is None:
        return None
    history, events, dot_colors, sep_color = sim

    border_colors = [c for c in range(1, 10) if c not in dot_colors and c != sep_color]
    border = random.choice(border_colors)
    template = _build_template((panel_h, panel_w), border)
    input_panels = [_render_panel(template, dot_colors, positions) for positions in history[:-1]]
    expected = _render_panel(template, dot_colors, history[-1])
    grid = _assemble_panels(input_panels + [template], orientation, sep_color)

    predicted = main(grid)
    if not np.array_equal(predicted, expected):
        return None

    panels, parsed_orientation, parsed_sep = _split_panels(grid)
    if parsed_orientation != orientation or parsed_sep != sep_color:
        return None

    template_colors = _template_palette(panels[-1])
    dynamic_colors = sorted({int(c) for c in np.unique(grid)} - template_colors - {sep_color})
    if len(dynamic_colors) != 2:
        return None

    last_visible = panels[-2]
    prev_visible = panels[-3]
    last_positions = [_dot_position(last_visible, color) for color in dynamic_colors]
    prev_positions = [_dot_position(prev_visible, color) for color in dynamic_colors]
    linear = [(2 * last_positions[i][0] - prev_positions[i][0], 2 * last_positions[i][1] - prev_positions[i][1]) for i in range(2)]
    true_next = [_dot_position(expected, color) for color in dynamic_colors]
    if all(linear[i] == true_next[i] for i in range(2)):
        return None

    assert any(any(step["bounce"]) for step in events[:-1])
    if mode == "collision_history":
        assert any(step["collision"] for step in events[:-1])
    if mode == "collision_forecast":
        assert events[-1]["collision"]
    if mode == "bounce_forecast":
        assert any(events[-1]["bounce"]) and not events[-1]["collision"]

    return grid, expected


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the missing next panel by continuing two moving pixels under wall-bounce and swap-on-collision dynamics.

    Input structure:
    - The grid is split by solid separator lines into one row or one column of equal-size panels.
    - Every panel shares the same static frame: a one-cell border around a plain background.
    - The first panels each contain exactly two differently colored single-pixel dots, while the final panel is the empty template.

    Rule:
    - Use the final template panel as the output canvas and identify the two moving colors absent from that template.
    - Read each dot's current velocity from its displacement between the last two visible panels.
    - Advance one more step inside the frame interior: bounce any velocity component that would cross the border, then if both dots would enter the same cell, swap their velocities before moving.
    - Draw the two dots at those predicted positions on the unchanged template.

    Key insight:
    - The last slot is not a past frame but a blank prediction canvas; simple linear extrapolation fails because the next step may be altered by a wall reflection or by the two dots exchanging velocities.
    """
    panels, _orientation, sep_color = _split_panels(grid)
    assert len(panels) == TOTAL_PANELS

    template = panels[-1]
    template_colors = _template_palette(template)
    dynamic_colors = sorted({int(c) for c in np.unique(grid)} - template_colors - {sep_color})
    assert len(dynamic_colors) == 2, "expected exactly two moving colors"

    prev_panel = panels[-2]
    prev_prev_panel = panels[-3]
    positions = [_dot_position(prev_panel, color) for color in dynamic_colors]
    prev_positions = [_dot_position(prev_prev_panel, color) for color in dynamic_colors]
    velocities = [
        (positions[i][0] - prev_positions[i][0], positions[i][1] - prev_positions[i][1]) for i in range(2)
    ]
    assert all(vel != (0, 0) for vel in velocities)
    assert all(max(abs(v[0]), abs(v[1])) == 1 for v in velocities)

    inner_shape = (template.shape[0] - 2, template.shape[1] - 2)
    next_positions, _next_velocities, _event = _advance_state(positions, velocities, inner_shape)
    return _render_panel(template, dynamic_colors, next_positions)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid bounce-extrapolation input")
