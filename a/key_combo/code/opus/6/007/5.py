from __future__ import annotations

import random
from collections import Counter

import numpy as np


PANEL_SIZE = 9
MIN_RADIUS = 1
MAX_RADIUS = 3
GEN_PANEL_COUNT = 3  # 4 panels would exceed ARC's 30-column limit: 4 * 8 + 1 = 33.
GEN_MAX_ATTEMPTS = 300
ARC_COLORS = tuple(range(10))


def _advance_radius(radius: int, delta: int) -> int:
    """Advance one step in the 1..3 cycle with wrap-at-boundary semantics."""
    nxt = int(radius) + int(delta)
    if nxt <= 0:
        return MAX_RADIUS
    if nxt >= MAX_RADIUS + 1:
        return MIN_RADIUS
    return nxt


def _would_wrap(radius: int, delta: int) -> bool:
    nxt = int(radius) + int(delta)
    return nxt <= 0 or nxt >= MAX_RADIUS + 1


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and h % 2 == 1

    frame = int(g[0, 0])
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)

    separators = [c for c in range(w) if np.all(g[:, c] == frame)]
    assert len(separators) >= 4
    assert separators[0] == 0 and separators[-1] == w - 1

    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert strides and len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1
    assert panel_w == h

    panels: list[np.ndarray] = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[0, :] == frame)
        assert np.all(panel[-1, :] == frame)
        assert np.all(panel[:, 0] == frame)
        assert np.all(panel[:, -1] == frame)
        panels.append(panel.copy())
    return panels, frame


def _infer_fill(panels: list[np.ndarray]) -> int:
    interior_values = []
    for panel in panels:
        interior_values.extend(map(int, panel[1:-1, 1:-1].ravel()))
    return int(Counter(interior_values).most_common(1)[0][0])


def _extract_radius(panel: np.ndarray, fill: int) -> tuple[int, int]:
    interior = panel[1:-1, 1:-1]
    center_r = interior.shape[0] // 2
    center_c = interior.shape[1] // 2

    coords = np.argwhere(interior != fill)
    assert coords.size > 0

    colors = {int(interior[r, c]) for r, c in coords}
    assert len(colors) == 1
    diamond_color = next(iter(colors))

    distances = {abs(int(r) - center_r) + abs(int(c) - center_c) for r, c in coords}
    assert len(distances) == 1
    radius = next(iter(distances))
    assert MIN_RADIUS <= radius <= MAX_RADIUS

    expected = {
        (r, c)
        for r in range(interior.shape[0])
        for c in range(interior.shape[1])
        if abs(r - center_r) + abs(c - center_c) == radius
    }
    actual = {(int(r), int(c)) for r, c in coords}
    assert actual == expected
    return int(radius), int(diamond_color)


def _infer_delta(radii: list[int]) -> int:
    raw_diffs = [b - a for a, b in zip(radii, radii[1:])]
    unit_diffs = {diff for diff in raw_diffs if diff in (-1, 1)}
    assert len(unit_diffs) == 1
    delta = next(iter(unit_diffs))

    for prev, nxt in zip(radii, radii[1:]):
        assert _advance_radius(prev, delta) == nxt
    return int(delta)


def _draw_panel(size: int, frame: int, fill: int, diamond: int, radius: int) -> np.ndarray:
    panel = np.full((size, size), int(fill), dtype=int)
    panel[0, :] = int(frame)
    panel[-1, :] = int(frame)
    panel[:, 0] = int(frame)
    panel[:, -1] = int(frame)

    interior = panel[1:-1, 1:-1]
    center_r = interior.shape[0] // 2
    center_c = interior.shape[1] // 2
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            if abs(r - center_r) + abs(c - center_c) == int(radius):
                interior[r, c] = int(diamond)
    return panel


def _sequence_with_wrap(start: int, delta: int, count: int) -> list[int]:
    radii = [int(start)]
    for _ in range(count - 1):
        radii.append(_advance_radius(radii[-1], delta))
    return radii


def _validate_semantics(grid: np.ndarray) -> tuple[list[int], int]:
    panels, _frame = _split_panels(grid)
    assert len(panels) == GEN_PANEL_COUNT

    fill = _infer_fill(panels)
    radii = []
    diamond_colors = set()
    for panel in panels:
        radius, diamond = _extract_radius(panel, fill)
        radii.append(radius)
        diamond_colors.add(diamond)

    assert len(diamond_colors) == 1
    delta = _infer_delta(radii)
    assert any(_would_wrap(radius, delta) for radius in radii[:-1])
    assert any((b - a) == delta for a, b in zip(radii, radii[1:]))
    return radii, delta


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the next framed diamond panel from a shared-border frame sequence.

    Input structure:
    - The grid is a horizontal row of equal odd-sized square panels that share vertical border columns.
    - Every panel has the same border color, the same fill color, and one centered Manhattan-diamond outline in a third color.
    - The diamond radius changes from panel to panel while the center position stays fixed.

    Rule:
    - Split the shared-border strip into panels and read the centered diamond radius in each one.
    - Infer whether the radius changes by +1 or -1 from the non-wrapping adjacent pair, then verify the whole sequence under cyclic bounds 1..3.
    - Advance one more step with the same delta, wrapping 0 to 3 and 4 to 1, and draw that radius in a standalone panel with the same colors.

    Key insight:
    - The frames encode a time series of a geometric parameter: the diamond does not move, only its radius follows a constant-rate cycle.
    """
    panels, frame = _split_panels(grid)
    fill = _infer_fill(panels)

    radii = []
    diamond_colors = set()
    for panel in panels:
        radius, diamond = _extract_radius(panel, fill)
        radii.append(radius)
        diamond_colors.add(diamond)

    assert len(diamond_colors) == 1
    diamond = next(iter(diamond_colors))
    delta = _infer_delta(radii)
    next_radius = _advance_radius(radii[-1], delta)
    return _draw_panel(panels[0].shape[0], frame, fill, diamond, next_radius)


def _gen_palette() -> tuple[int, int, int]:
    frame, fill, diamond = random.sample(ARC_COLORS, 3)
    return int(frame), int(fill), int(diamond)


def _gen_radii() -> tuple[list[int], int]:
    delta = random.choice((-1, 1))
    starts = []
    for start in range(MIN_RADIUS, MAX_RADIUS + 1):
        radii = _sequence_with_wrap(start, delta, GEN_PANEL_COUNT)
        if any(_would_wrap(radius, delta) for radius in radii[:-1]) and any(
            (b - a) == delta for a, b in zip(radii, radii[1:])
        ):
            starts.append((start, radii))
    assert starts
    _start, radii = random.choice(starts)
    return radii, int(delta)


def _make_input(frame: int, fill: int, diamond: int, radii: list[int]) -> np.ndarray:
    width = GEN_PANEL_COUNT * (PANEL_SIZE - 1) + 1
    grid = np.full((PANEL_SIZE, width), int(fill), dtype=int)
    for idx, radius in enumerate(radii):
        panel = _draw_panel(PANEL_SIZE, frame, fill, diamond, radius)
        c0 = idx * (PANEL_SIZE - 1)
        grid[:, c0 : c0 + PANEL_SIZE] = panel
    return grid


def _expected_output(frame: int, fill: int, diamond: int, radii: list[int], delta: int) -> np.ndarray:
    return _draw_panel(PANEL_SIZE, frame, fill, diamond, _advance_radius(radii[-1], delta))


def _gen_attempt() -> np.ndarray | None:
    frame, fill, diamond = _gen_palette()
    radii, delta = _gen_radii()
    inp = _make_input(frame, fill, diamond, radii)
    expected = _expected_output(frame, fill, diamond, radii, delta)

    parsed_radii, parsed_delta = _validate_semantics(inp)
    if parsed_radii != radii or parsed_delta != delta:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-sequence diamond input")
