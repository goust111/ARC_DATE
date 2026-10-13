from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
COLOR_A = 2
COLOR_B = 8

PANEL_COUNTS = (2, 3, 4, 5)
PANEL_SIZES = (7, 9, 11, 13)
SHAPE_KINDS = ("L", "arc")
MAX_GENERATION_ATTEMPTS = 400


def _panel_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEP))]
    if not sep_cols:
        assert h == w
        return [(0, w)]

    panels: list[tuple[int, int]] = []
    start = 0
    for c in sep_cols:
        if c > start:
            panels.append((start, c))
        start = c + 1
    if start < w:
        panels.append((start, w))

    assert panels, "expected at least one panel"
    widths = {c1 - c0 for c0, c1 in panels}
    assert len(widths) == 1, "panels must have equal width"
    panel_size = widths.pop()
    assert panel_size == h, "panels must be square"
    return panels


def _mean_radius2(coords: np.ndarray, center: tuple[float, float]) -> float:
    dr = coords[:, 0].astype(float) - center[0]
    dc = coords[:, 1].astype(float) - center[1]
    return float(np.mean(dr * dr + dc * dc))


def _panel_turn_is_clockwise(panel: np.ndarray) -> bool:
    coords_a = np.argwhere(panel == COLOR_A)
    coords_b = np.argwhere(panel == COLOR_B)
    assert coords_a.size and coords_b.size, "expected both cue colors in every panel"

    center = ((panel.shape[0] - 1) / 2.0, (panel.shape[1] - 1) / 2.0)
    score_a = _mean_radius2(coords_a, center)
    score_b = _mean_radius2(coords_b, center)
    assert abs(score_a - score_b) > 1e-6, "outer/inner cue must be unambiguous"
    return score_a > score_b


def _rotate_panel(panel: np.ndarray, clockwise: bool) -> np.ndarray:
    return np.rot90(panel, -1 if clockwise else 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel's two-color turning object according to its outer/inner color cue.

    The background is black, separator columns are grey, and each panel is square.
    Color A on the outer side of the turn means "rotate clockwise"; color B on the
    outer side means "rotate counterclockwise".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for c0, c1 in _panel_slices(g):
        panel = g[:, c0:c1]
        clockwise = _panel_turn_is_clockwise(panel)
        out[:, c0:c1] = _rotate_panel(panel, clockwise)
    return out


def _base_l_masks(size: int, arm: int) -> tuple[np.ndarray, np.ndarray]:
    m = size // 2
    outer = np.zeros((size, size), dtype=bool)
    inner = np.zeros((size, size), dtype=bool)

    outer[m - arm : m, m + 1] = True
    outer[m - 1, m + 1 : m + arm + 1] = True

    inner[m - arm : m + 1, m] = True
    inner[m, m : m + arm + 1] = True
    return outer, inner


def _base_arc_masks(size: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    m = size // 2
    outer = np.zeros((size, size), dtype=bool)
    inner = np.zeros((size, size), dtype=bool)

    r2_min = radius * radius
    r2_max = (radius + 1) * (radius + 1)
    split = r2_min + radius

    for r in range(size):
        for c in range(size):
            dr = m - r
            dc = c - m
            if dr < 0 or dc < 0:
                continue
            d2 = dr * dr + dc * dc
            if not (r2_min <= d2 <= r2_max):
                continue
            if d2 <= split:
                inner[r, c] = True
            else:
                outer[r, c] = True

    return outer, inner


def _render_panel(
    size: int,
    shape_kind: str,
    turn_k: int,
    clockwise: bool,
    shape_param: int,
) -> np.ndarray:
    if shape_kind == "L":
        outer, inner = _base_l_masks(size, shape_param)
    elif shape_kind == "arc":
        outer, inner = _base_arc_masks(size, shape_param)
    else:
        raise AssertionError(shape_kind)

    outer = np.rot90(outer, turn_k)
    inner = np.rot90(inner, turn_k)
    assert not bool((outer & inner).any())
    assert bool((outer | inner).any())

    panel = np.full((size, size), BG, dtype=int)
    outer_color = COLOR_A if clockwise else COLOR_B
    inner_color = COLOR_B if clockwise else COLOR_A
    panel[outer] = outer_color
    panel[inner] = inner_color
    return panel


def _sample_shape_param(size: int, shape_kind: str) -> int:
    m = size // 2
    if shape_kind == "L":
        return random.randint(2, m)
    return random.randint(2, m - 1)


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    size = panels[0].shape[0]
    count = len(panels)
    w = count * size + (count - 1)
    grid = np.full((size, w), BG, dtype=int)

    col = 0
    for i, panel in enumerate(panels):
        grid[:, col : col + size] = panel
        col += size
        if i != count - 1:
            grid[:, col] = SEP
            col += 1
    return grid


def _sample_directions(count: int) -> list[bool]:
    directions = [True, False]
    while len(directions) < count:
        directions.append(bool(random.getrandbits(1)))
    random.shuffle(directions)
    return directions


def _sample_shapes(count: int) -> list[str]:
    shapes = ["L", "arc"]
    while len(shapes) < count:
        shapes.append(random.choice(SHAPE_KINDS))
    random.shuffle(shapes)
    return shapes


def _expected_output(panels: list[np.ndarray], directions: list[bool]) -> np.ndarray:
    rotated = [_rotate_panel(panel, clockwise) for panel, clockwise in zip(panels, directions)]
    return _assemble_grid(rotated)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice(PANEL_COUNTS)
    size = random.choice(PANEL_SIZES)

    directions = _sample_directions(panel_count)
    shapes = _sample_shapes(panel_count)

    panels: list[np.ndarray] = []
    for clockwise, shape_kind in zip(directions, shapes):
        shape_param = _sample_shape_param(size, shape_kind)
        turn_k = random.randrange(4)
        panel = _render_panel(size, shape_kind, turn_k, clockwise, shape_param)

        # Reject weak cues or accidental solver mismatches before assembling the full grid.
        if _panel_turn_is_clockwise(panel) != clockwise:
            return None
        if np.array_equal(panel, _rotate_panel(panel, clockwise)):
            return None
        panels.append(panel)

    inp = _assemble_grid(panels)
    expected = _expected_output(panels, directions)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
