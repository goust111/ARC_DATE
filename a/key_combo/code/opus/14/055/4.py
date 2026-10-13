from __future__ import annotations

from collections import Counter
import random

import numpy as np


FRAME = 5
FILL = 0
RING_COLORS = (1, 2, 3)
RING_SIZE = 5
PANEL_COUNT = 3
PANEL_INNER_W = 8
PANEL_W = PANEL_INNER_W + 2
MOTION_VECTORS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
ROTATION_STEPS = (1, 2)  # right-rotate by one layer or by two layers
MAX_GENERATION_ATTEMPTS = 400


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_layers(layers: tuple[int, int, int], step: int) -> tuple[int, int, int]:
    k = int(step) % len(layers)
    if k == 0:
        return tuple(layers)
    return tuple(layers[-k:] + layers[:-k])


def _ring_patch(layers: tuple[int, int, int]) -> np.ndarray:
    outer, middle, core = map(int, layers)
    patch = np.full((RING_SIZE, RING_SIZE), outer, dtype=int)
    patch[1:4, 1:4] = middle
    patch[2, 2] = core
    return patch


def _blank_panel(height: int, width: int) -> np.ndarray:
    panel = np.full((height, width), FILL, dtype=int)
    panel[0, :] = FRAME
    panel[-1, :] = FRAME
    panel[:, 0] = FRAME
    panel[:, -1] = FRAME
    return panel


def _stamp_ring(panel: np.ndarray, top: int, left: int, layers: tuple[int, int, int]) -> None:
    patch = _ring_patch(layers)
    h, w = patch.shape
    panel[top : top + h, left : left + w] = patch


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    frame = int(g[0, 0])
    separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == frame)]
    assert len(separators) == PANEL_COUNT + 1
    assert separators[0] == 0 and separators[-1] == g.shape[1] - 1

    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1

    panels = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (g.shape[0], panel_w)
        assert np.all(panel[0, :] == frame)
        assert np.all(panel[-1, :] == frame)
        assert np.all(panel[:, 0] == frame)
        assert np.all(panel[:, -1] == frame)
        panels.append(panel)
    return panels


def _read_panel(panel: np.ndarray) -> tuple[int, int, tuple[int, int, int]]:
    interior = panel[1:-1, 1:-1]
    fill = _most_common(interior)
    mask = interior != fill
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (RING_SIZE, RING_SIZE)

    crop = interior[r0 : r1 + 1, c0 : c1 + 1]
    rows = np.arange(RING_SIZE)[:, None]
    cols = np.arange(RING_SIZE)[None, :]
    depth = np.minimum(
        np.minimum(rows, RING_SIZE - 1 - rows),
        np.minimum(cols, RING_SIZE - 1 - cols),
    )
    layers = []
    for d in range(3):
        colors = np.unique(crop[depth == d])
        assert colors.size == 1
        layers.append(int(colors[0]))
    assert tuple(sorted(layers)) == tuple(sorted(RING_COLORS))

    top = int(r0 + 1)
    left = int(c0 + 1)
    return top, left, tuple(layers)


def _rotation_between(
    before: tuple[int, int, int], after: tuple[int, int, int]
) -> int:
    matches = [step for step in ROTATION_STEPS if _rotate_layers(before, step) == after]
    assert len(matches) == 1
    return int(matches[0])


def _build_output_panel(height: int, top: int, left: int, layers: tuple[int, int, int]) -> np.ndarray:
    out = _blank_panel(height, PANEL_W)
    _stamp_ring(out, top, left, layers)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the next standalone panel by continuing both the ring's motion and its layer-color cycle.

    Input structure:
    - The grid is a horizontal strip of three equal-width framed panels sharing vertical border columns.
    - Each panel has black fill, grey frame borders, and one 5x5 concentric three-layer ring inside.
    - The same ring moves by a constant local row/column offset from panel to panel, while the outer/middle/core colors rotate cyclically across the three layers.

    Rule:
    - Split the shared-border strip into its three panels and locate the ring inside each panel.
    - Measure the constant local displacement between consecutive ring positions.
    - Measure the constant one-step cyclic rotation of the ordered layer colors `(outer, middle, core)`.
    - Return one standalone panel of the same size as each input panel, placing the ring one more displacement step after panel 3 and recoloring its layers with one more cyclic rotation.

    Key insight:
    - The spatial trajectory and the layer-color cycle are independent regularities carried by the same object, so the next frame is correct only when both extrapolations are applied together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panels = _split_panels(g)
    assert len(panels) == PANEL_COUNT

    specs = [_read_panel(panel) for panel in panels]
    tops = [spec[0] for spec in specs]
    lefts = [spec[1] for spec in specs]
    layers = [spec[2] for spec in specs]

    drs = [b - a for a, b in zip(tops, tops[1:])]
    dcs = [b - a for a, b in zip(lefts, lefts[1:])]
    assert len(set(drs)) == 1 and len(set(dcs)) == 1
    dr = int(drs[0])
    dc = int(dcs[0])
    assert (dr, dc) != (0, 0)

    k1 = _rotation_between(layers[0], layers[1])
    k2 = _rotation_between(layers[1], layers[2])
    assert k1 == k2

    next_top = int(tops[-1] + dr)
    next_left = int(lefts[-1] + dc)
    assert 1 <= next_top <= g.shape[0] - RING_SIZE - 1
    assert 1 <= next_left <= PANEL_W - RING_SIZE - 1
    next_layers = _rotate_layers(layers[-1], k1)
    return _build_output_panel(g.shape[0], next_top, next_left, next_layers)


def _feasible_starts(limit: int, delta: int) -> list[int]:
    max_start = limit - RING_SIZE
    return [
        start
        for start in range(max_start + 1)
        if all(0 <= start + i * delta <= max_start for i in range(PANEL_COUNT + 1))
    ]


def _single_panel_from_local(
    height: int, top_local: int, left_local: int, layers: tuple[int, int, int]
) -> np.ndarray:
    panel = _blank_panel(height, PANEL_W)
    _stamp_ring(panel, top_local + 1, left_local + 1, layers)
    return panel


def _expected_output(
    height: int,
    start_top: int,
    start_left: int,
    dr: int,
    dc: int,
    base_layers: tuple[int, int, int],
    rotation: int,
) -> np.ndarray:
    top_local = start_top + PANEL_COUNT * dr
    left_local = start_left + PANEL_COUNT * dc
    layers = _rotate_layers(base_layers, PANEL_COUNT * rotation)
    return _single_panel_from_local(height, top_local, left_local, layers)


def _build_input(
    height: int,
    start_top: int,
    start_left: int,
    dr: int,
    dc: int,
    base_layers: tuple[int, int, int],
    rotation: int,
) -> np.ndarray:
    width = PANEL_COUNT * (PANEL_W - 1) + 1
    grid = np.full((height, width), FILL, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    for idx in range(PANEL_COUNT + 1):
        grid[:, idx * (PANEL_W - 1)] = FRAME

    for idx in range(PANEL_COUNT):
        panel_left = idx * (PANEL_W - 1)
        top_local = start_top + idx * dr
        left_local = start_left + idx * dc
        layers = _rotate_layers(base_layers, idx * rotation)
        _stamp_ring(grid, top_local + 1, panel_left + left_local + 1, layers)
    return grid


def _validate_attempt(
    inp: np.ndarray,
    expected: np.ndarray,
    dr: int,
    dc: int,
    rotation: int,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    panels = _split_panels(inp)
    specs = [_read_panel(panel) for panel in panels]
    tops = [spec[0] for spec in specs]
    lefts = [spec[1] for spec in specs]
    layers = [spec[2] for spec in specs]

    if [b - a for a, b in zip(tops, tops[1:])] != [dr, dr]:
        return False
    if [b - a for a, b in zip(lefts, lefts[1:])] != [dc, dc]:
        return False
    if _rotation_between(layers[0], layers[1]) != rotation:
        return False
    if _rotation_between(layers[1], layers[2]) != rotation:
        return False
    if any(np.array_equal(expected, panel) for panel in panels):
        return False

    top3_local = tops[-1] - 1
    left3_local = lefts[-1] - 1
    move_only = _single_panel_from_local(
        inp.shape[0], top3_local + dr, left3_local + dc, layers[-1]
    )
    color_only = _single_panel_from_local(
        inp.shape[0], top3_local, left3_local, _rotate_layers(layers[-1], rotation)
    )
    if np.array_equal(move_only, expected):
        return False
    if np.array_equal(color_only, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(8, 12)
    height = inner_h + 2
    dr, dc = random.choice(MOTION_VECTORS)
    rotation = random.choice(ROTATION_STEPS)
    base_layers = tuple(random.sample(RING_COLORS, k=3))

    top_choices = _feasible_starts(inner_h, dr)
    left_choices = _feasible_starts(PANEL_INNER_W, dc)
    if not top_choices or not left_choices:
        return None

    start_top = random.choice(top_choices)
    start_left = random.choice(left_choices)
    inp = _build_input(height, start_top, start_left, dr, dc, base_layers, rotation)
    expected = _expected_output(height, start_top, start_left, dr, dc, base_layers, rotation)

    if not _validate_attempt(inp, expected, dr, dc, rotation):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
