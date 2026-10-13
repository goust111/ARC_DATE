from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 800

Coord = tuple[int, int]
Velocity = tuple[int, int]


def _wall_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 9

    wall_cols = [c for c in range(w) if bool(np.all(g[:, c] != BLACK))]
    assert len(wall_cols) >= 3
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1

    steps = np.diff(wall_cols)
    assert len(set(map(int, steps))) == 1
    step = int(steps[0])
    assert step >= 4
    assert wall_cols == list(range(0, w, step))

    panels = [g[:, wall_cols[i] : wall_cols[i + 1] + 1].copy() for i in range(len(wall_cols) - 1)]
    return panels, wall_cols


def _detect_ball_color(grid: np.ndarray, wall_cols: list[int]) -> int:
    mask = np.ones(grid.shape, dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, wall_cols] = False
    colors = sorted(set(map(int, grid[mask])) - {BLACK})
    assert len(colors) == 1
    return int(colors[0])


def _detect_border_color(panel: np.ndarray, ball_color: int) -> int:
    border_pixels = panel[_wall_mask(panel.shape)]
    colors = sorted(set(map(int, border_pixels)) - {ball_color})
    assert len(colors) == 1
    return int(colors[0])


def _panel_ball(panel: np.ndarray, ball_color: int) -> Coord:
    coords = np.argwhere(panel[1:-1, 1:-1] == ball_color)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    return r + 1, c + 1


def _panel_dents(panel: np.ndarray, ball_color: int) -> set[Coord]:
    wall = _wall_mask(panel.shape)
    coords = np.argwhere((panel == ball_color) & wall)
    return {tuple(map(int, coord)) for coord in coords}


def _draw_panel(
    shape: tuple[int, int],
    border_color: int,
    ball_color: int,
    ball_pos: Coord,
    dents: set[Coord],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    out[0, :] = border_color
    out[-1, :] = border_color
    out[:, 0] = border_color
    out[:, -1] = border_color
    for r, c in dents:
        out[r, c] = ball_color
    out[ball_pos] = ball_color
    return out


def _advance(
    ball_pos: Coord,
    velocity: Velocity,
    shape: tuple[int, int],
    dents: set[Coord],
) -> tuple[Coord, Velocity, set[Coord], bool, bool]:
    h, w = shape
    r, c = ball_pos
    dr, dc = velocity
    next_dents = set(dents)

    hit_vertical = (dr < 0 and r == 1) or (dr > 0 and r == h - 2)
    hit_horizontal = (dc < 0 and c == 1) or (dc > 0 and c == w - 2)
    if hit_vertical and hit_horizontal:
        raise ValueError("corner hits are intentionally excluded")

    if hit_vertical:
        dent = (0, c) if dr < 0 else (h - 1, c)
        if dent in next_dents:
            raise ValueError("repeated dent")
        next_dents.add(dent)
        dr *= -1

    if hit_horizontal:
        dent = (r, 0) if dc < 0 else (r, w - 1)
        if dent in next_dents:
            raise ValueError("repeated dent")
        next_dents.add(dent)
        dc *= -1

    nr = r + dr
    nc = c + dc
    assert 1 <= nr < h - 1 and 1 <= nc < w - 1
    return (nr, nc), (dr, dc), next_dents, hit_vertical, hit_horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next standalone frame of a bouncing ball whose wall dents accumulate over time.

    Input structure:
    - The input is a left-to-right sequence of equal-size framed panels that share one border column.
    - Every panel has the same black interior, the same frame color, one interior ball pixel, and
      any number of permanent ball-colored dents on the walls.
    - Later panels keep every earlier dent and show the ball after one more motion step.

    Rule:
    - Split the strip into panels by the full-height wall columns and identify the single ball color
      from the interior of the panels.
    - Read the ball positions frame by frame; the difference between the last two positions is the
      current velocity after any earlier bounce.
    - Starting from the last frame, advance one more step. If the next move would enter a wall,
      reflect that velocity component first, add a new dent on the struck wall pixel, and then move
      the ball one cell with the reflected velocity.
    - Return one complete panel containing the new ball position and the full accumulated dent set.

    Key insight:
    - The panels are a time series with persistent wall history: dents tell you which wall state to
      preserve, while the last two ball positions already encode the post-bounce direction needed
      for the next extrapolated frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels, wall_cols = _split_panels(g)
    assert len(panels) >= 2

    ball_color = _detect_ball_color(g, wall_cols)
    border_color = _detect_border_color(panels[0], ball_color)

    positions: list[Coord] = []
    dent_sets: list[set[Coord]] = []
    for panel in panels:
        positions.append(_panel_ball(panel, ball_color))
        dents = _panel_dents(panel, ball_color)
        dent_sets.append(dents)
        assert _detect_border_color(panel, ball_color) == border_color

    for prev, cur in zip(dent_sets, dent_sets[1:]):
        assert prev.issubset(cur)
        assert len(cur - prev) <= 1

    dr = positions[-1][0] - positions[-2][0]
    dc = positions[-1][1] - positions[-2][1]
    assert (dr, dc) != (0, 0)

    next_pos, _next_vel, next_dents, _hit_vertical, _hit_horizontal = _advance(
        positions[-1],
        (dr, dc),
        panels[-1].shape,
        dent_sets[-1],
    )
    return _draw_panel(panels[-1].shape, border_color, ball_color, next_pos, next_dents)


def _assemble_sequence(panels: list[np.ndarray]) -> np.ndarray:
    assert panels
    h, w = panels[0].shape
    out = np.full((h, len(panels) * (w - 1) + 1), BLACK, dtype=int)
    for i, panel in enumerate(panels):
        start = i * (w - 1)
        out[:, start : start + w] = panel
    return out


def _sample_dimensions() -> tuple[int, int, int]:
    num_frames = int(np.random.randint(4, 7))
    max_panel_w = (29 // num_frames) + 1
    panel_w = int(np.random.randint(5, max_panel_w + 1))
    panel_h = int(np.random.randint(6, 11))
    return num_frames, panel_h, panel_w


def _sample_colors() -> tuple[int, int]:
    border_color, ball_color = np.random.choice(PALETTE, size=2, replace=False)
    return int(border_color), int(ball_color)


def _sample_horizontal_motion(num_frames: int, panel_w: int) -> tuple[int, int]:
    obs_steps = num_frames - 1
    if num_frames == 4 and panel_w >= 6 and int(np.random.randint(3)) > 0:
        dc = int(np.random.choice([-1, 1]))
        max_hit_step = panel_w - 3
        if panel_w >= 7 and int(np.random.randint(2)) == 0:
            hit_step = int(np.random.randint(obs_steps, max_hit_step + 1))
        else:
            hit_step = obs_steps
        start_c = (panel_w - 2 - hit_step) if dc > 0 else (1 + hit_step)
        if not (1 <= start_c < panel_w - 1):
            raise ValueError("invalid horizontal start")
        return dc, int(start_c)

    start_c = int(np.random.randint(1, panel_w - 1))
    return 0, start_c


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    num_frames, panel_h, panel_w = _sample_dimensions()
    obs_steps = num_frames - 1
    border_color, ball_color = _sample_colors()

    dr = int(np.random.choice([-1, 1]))
    max_vhit = min(obs_steps - 1, panel_h - 3)
    if max_vhit < 0:
        return None
    vhit_step = int(np.random.randint(0, max_vhit + 1))
    start_r = (panel_h - 2 - vhit_step) if dr > 0 else (1 + vhit_step)
    if not (1 <= start_r < panel_h - 1):
        return None

    try:
        dc, start_c = _sample_horizontal_motion(num_frames, panel_w)
    except ValueError:
        return None

    ball_pos: Coord = (int(start_r), int(start_c))
    velocity: Velocity = (dr, dc)
    dents: set[Coord] = set()

    panels: list[np.ndarray] = []
    observed_vertical_hits = 0
    observed_collision_free_steps = 0

    for step in range(num_frames):
        panels.append(_draw_panel((panel_h, panel_w), border_color, ball_color, ball_pos, dents))
        if step == num_frames - 1:
            break
        try:
            ball_pos, velocity, dents, hit_vertical, hit_horizontal = _advance(
                ball_pos,
                velocity,
                (panel_h, panel_w),
                dents,
            )
        except ValueError:
            return None
        if hit_horizontal:
            return None
        observed_vertical_hits += int(hit_vertical)
        observed_collision_free_steps += int(not hit_vertical)

    if observed_vertical_hits == 0 or not dents:
        return None

    try:
        next_pos, _next_velocity, next_dents, _next_hit_vertical, _next_hit_horizontal = _advance(
            ball_pos,
            velocity,
            (panel_h, panel_w),
            dents,
        )
    except ValueError:
        return None

    # Keep some straight-motion examples so "always add a dent next" is a visible counterexample.
    if observed_collision_free_steps == 0 and next_dents != dents:
        return None

    inp = _assemble_sequence(panels)
    expected = _draw_panel((panel_h, panel_w), border_color, ball_color, next_pos, next_dents)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        inp, expected = attempt
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid dented-bounce sequence")
