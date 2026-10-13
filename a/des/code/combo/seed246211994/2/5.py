from __future__ import annotations

import random

import numpy as np


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP_VECTORS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_HEIGHT = 29
MAX_GRID_WIDTH = 16


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_runs(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    h = grid.shape[0]
    r = 0
    while r < h:
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            color = int(row[0])
            start = r
            r += 1
            while r < h and np.all(grid[r] == color):
                r += 1
            runs.append((start, r, color))
        else:
            r += 1
    assert runs, "expected one or more full-width separator rows"
    assert len({color for _, _, color in runs}) == 1, "separator rows must share one color"
    return runs


def _panel_slices(height: int, runs: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    panels: list[tuple[int, int]] = []
    prev = 0
    for start, stop, _color in runs:
        assert prev < start, "separator rows cannot touch each other without a panel between them"
        panels.append((prev, start))
        prev = stop
    assert prev < height, "expected a panel below the last separator"
    panels.append((prev, height))
    panel_heights = {stop - start for start, stop in panels}
    assert len(panel_heights) == 1, "panels must all have equal height"
    return panels


def _panel_state(panel: np.ndarray, bg: int) -> dict[str, object] | None:
    coords = np.argwhere(panel != bg)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    patch = panel[r0:r1, c0:c1].copy()
    return {
        "origin": (int(r0), int(c0)),
        "patch": patch,
    }


def _infer_step(states: list[dict[str, object]]) -> tuple[int, int]:
    assert len(states) >= 2, "need at least two observed frames"
    ref_patch = np.asarray(states[0]["patch"], dtype=int)
    prev_origin = states[0]["origin"]
    assert isinstance(prev_origin, tuple)

    step: tuple[int, int] | None = None
    for state in states[1:]:
        patch = np.asarray(state["patch"], dtype=int)
        origin = state["origin"]
        assert isinstance(origin, tuple)
        assert patch.shape == ref_patch.shape, "animation must preserve sprite size"
        assert np.array_equal(patch, ref_patch), "animation must be a rigid translation"
        dr = int(origin[0] - prev_origin[0])
        dc = int(origin[1] - prev_origin[1])
        if step is None:
            step = (dr, dc)
            assert step != (0, 0), "translation step must be non-zero"
        else:
            assert step == (dr, dc), "all observed frame-to-frame steps must match"
        prev_origin = origin
    assert step is not None
    return step


def _stamp_patch(panel: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> np.ndarray:
    out = panel.copy()
    coords = np.argwhere(patch != bg)
    for rr, cc in coords:
        tr = top + int(rr)
        tc = left + int(cc)
        assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1], "next frame must stay within the panel"
        out[tr, tc] = int(patch[rr, cc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat horizontal bands as animation frames and draw the translated next frame into the blank band.

    Input structure:
    - One or more full-width separator rows split the grid into equal-height horizontal panels.
    - Each non-empty panel contains the same two-color sprite in a different position.
    - The final panel is blank.

    Rule:
    - Compare consecutive non-empty panels to infer the constant translation vector.
    - Apply that same one-step translation to the final observed panel.
    - Draw the predicted next frame into the blank last panel.

    Key insight:
    - The panels are time-ordered animation frames; the motion is inferred from
      repeated sprite positions, not from a fixed absolute template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    runs = _separator_runs(g, bg)
    panels = _panel_slices(g.shape[0], runs)
    panel_arrays = [g[r0:r1, :] for r0, r1 in panels]
    states = [_panel_state(panel, bg) for panel in panel_arrays]

    assert all(state is not None for state in states[:-1]), "all panels except the last must be non-empty"
    assert states[-1] is None, "the final panel must be blank"

    observed = [state for state in states[:-1] if state is not None]
    step_r, step_c = _infer_step(observed)

    last_state = observed[-1]
    origin = last_state["origin"]
    assert isinstance(origin, tuple)
    patch = np.asarray(last_state["patch"], dtype=int)

    next_top = int(origin[0] + step_r)
    next_left = int(origin[1] + step_c)
    filled_last_panel = _stamp_patch(panel_arrays[-1], patch, next_top, next_left, bg)

    out = g.copy()
    last_r0, last_r1 = panels[-1]
    out[last_r0:last_r1, :] = filled_last_panel
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _grow_polyomino(rng: random.Random, size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ORTHO_DIRS:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(rng.choice(frontier))
    return _normalize(cells)


def _sprite_shape(rng: random.Random) -> tuple[set[tuple[int, int]], tuple[int, int], int, int]:
    for _ in range(100):
        body = _grow_polyomino(rng, rng.randint(3, 6))
        body_h, body_w = _bbox_dims(body)
        if body_h < 2 or body_w < 2 or body_h > 3 or body_w > 5:
            continue

        candidates = set()
        for r, c in body:
            for dr, dc in ORTHO_DIRS:
                nxt = (r + dr, c + dc)
                if nxt not in body:
                    candidates.add(nxt)
        if not candidates:
            continue

        marker = rng.choice(sorted(candidates))
        combined = set(body) | {marker}
        min_r = min(r for r, _ in combined)
        min_c = min(c for _, c in combined)
        body_norm = {(r - min_r, c - min_c) for r, c in body}
        marker_norm = (marker[0] - min_r, marker[1] - min_c)
        sprite_cells = body_norm | {marker_norm}
        sprite_h, sprite_w = _bbox_dims(sprite_cells)
        if sprite_h > 4 or sprite_w > 6:
            continue

        if sprite_h < 2 or sprite_w < 3:
            continue
        return body_norm, marker_norm, sprite_h, sprite_w
    raise RuntimeError("failed to build a valid two-color sprite")


def _render_patch(
    body_cells: set[tuple[int, int]],
    marker: tuple[int, int],
    height: int,
    width: int,
    bg: int,
    body_color: int,
    marker_color: int,
) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    for r, c in body_cells:
        patch[r, c] = body_color
    patch[marker[0], marker[1]] = marker_color
    return patch


def _build_input_and_output(
    panel_h: int,
    panel_w: int,
    observed_frames: int,
    sep_color: int,
    bg: int,
    patch: np.ndarray,
    top0: int,
    left0: int,
    step: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    total_panels = observed_frames + 1
    panels_in = []
    panels_out = []
    for frame_idx in range(total_panels):
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        if frame_idx < observed_frames:
            top = top0 + frame_idx * step[0]
            left = left0 + frame_idx * step[1]
            panel = _stamp_patch(panel, patch, top, left, bg)
        panels_in.append(panel)

        panel_target = np.full((panel_h, panel_w), bg, dtype=int)
        top_t = top0 + frame_idx * step[0]
        left_t = left0 + frame_idx * step[1]
        panel_target = _stamp_patch(panel_target, patch, top_t, left_t, bg)
        panels_out.append(panel_target)

    rows_in: list[np.ndarray] = []
    rows_out: list[np.ndarray] = []
    for idx in range(total_panels):
        rows_in.append(panels_in[idx])
        rows_out.append(panels_out[idx])
        if idx != total_panels - 1:
            rows_in.append(np.full((1, panel_w), sep_color, dtype=int))
            rows_out.append(np.full((1, panel_w), sep_color, dtype=int))
    return np.vstack(rows_in), np.vstack(rows_out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = rng.sample(range(10), 4)
    bg, sep_color, body_color, marker_color = colors

    step = rng.choice(STEP_VECTORS)
    body_cells, marker, sprite_h, sprite_w = _sprite_shape(rng)

    allowed_frames = []
    for observed_frames in (2, 3):
        travel_r = abs(step[0]) * observed_frames
        travel_c = abs(step[1]) * observed_frames
        min_panel_h = sprite_h + travel_r + 2
        min_panel_w = sprite_w + travel_c + 2
        total_h = (observed_frames + 1) * min_panel_h + observed_frames
        if total_h <= MAX_GRID_HEIGHT and min_panel_w <= MAX_GRID_WIDTH:
            allowed_frames.append((observed_frames, min_panel_h, min_panel_w))
    if not allowed_frames:
        return None

    observed_frames, min_panel_h, min_panel_w = rng.choice(allowed_frames)
    panel_h_cap = (MAX_GRID_HEIGHT - observed_frames) // (observed_frames + 1)
    panel_h = rng.randint(min_panel_h, min(panel_h_cap, min_panel_h + 2))
    panel_w = rng.randint(min_panel_w, min(MAX_GRID_WIDTH, min_panel_w + 4))

    patch = _render_patch(body_cells, marker, sprite_h, sprite_w, bg, body_color, marker_color)

    future_steps = observed_frames
    min_shift_r = min(0, step[0] * future_steps)
    max_shift_r = max(0, step[0] * future_steps)
    min_shift_c = min(0, step[1] * future_steps)
    max_shift_c = max(0, step[1] * future_steps)

    top_low = 1 - min_shift_r
    top_high = panel_h - sprite_h - 1 - max_shift_r
    left_low = 1 - min_shift_c
    left_high = panel_w - sprite_w - 1 - max_shift_c
    if top_low > top_high or left_low > left_high:
        return None

    top0 = rng.randint(top_low, top_high)
    left0 = rng.randint(left_low, left_high)
    inp, expected = _build_input_and_output(
        panel_h=panel_h,
        panel_w=panel_w,
        observed_frames=observed_frames,
        sep_color=sep_color,
        bg=bg,
        patch=patch,
        top0=top0,
        left0=left0,
        step=step,
    )

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
