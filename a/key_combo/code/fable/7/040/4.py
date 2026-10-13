"""ARC task: scattered marker count controls a frame-anchored stripe period."""

from __future__ import annotations

from collections import deque

import numpy as np


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(counts.max())
    assert int(np.count_nonzero(counts == largest)) == 1, "expected a unique dominant background"
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.astype(bool, copy=True)
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(unseen):
        start = (int(start_r), int(start_c))
        if not unseen[start]:
            continue
        unseen[start] = False
        queue = deque([start])
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    queue.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _is_hollow_frame(grid: np.ndarray, bg: int, color: int, coords: np.ndarray):
    """Return the inclusive frame bbox, or None when the component is not a frame."""
    if len(coords) < 8:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height < 3 or width < 3 or height == width:
        return None
    border = np.zeros((height, width), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if len(coords) != int(border.sum()):
        return None
    if not np.all(sub[border] == color) or not np.all(sub[~border] == bg):
        return None
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Replace a hollow frame and scattered markers with count-periodic full-canvas stripes.

    Input structure:
    - A uniquely dominant background contains one monochrome, one-cell-thick hollow
      rectangular frame whose unequal height and width are at least three.
    - Outside the frame are one or more isolated pixels of one other color; marker
      color and marker positions carry no meaning.

    Rule:
    - Count the isolated marker pixels as n and erase all original foreground.
    - If the frame is wider than tall, draw full rows in the frame color at every
      row congruent to its top edge modulo n.  If it is taller than wide, draw full
      columns congruent to its left edge modulo n.  Extend in both directions.

    Key insight:
    - The frame supplies stripe color, orientation, and phase, while only the
      number—not the locations or color—of scattered pixels supplies the period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    foreground_colors = [int(x) for x in np.unique(g) if int(x) != bg]
    assert len(foreground_colors) == 2, "expected exactly a frame color and a marker color"

    frame_matches: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    components_by_color: dict[int, list[np.ndarray]] = {}
    for color in foreground_colors:
        comps = _components(g == color)
        components_by_color[color] = comps
        for comp in comps:
            box = _is_hollow_frame(g, bg, color, comp)
            if box is not None:
                frame_matches.append((color, box, comp))
    assert len(frame_matches) == 1, "expected exactly one complete hollow rectangular frame"

    frame_color, (r0, c0, r1, c1), frame_component = frame_matches[0]
    marker_color = next(color for color in foreground_colors if color != frame_color)
    marker_components = components_by_color[marker_color]
    assert marker_components and all(len(comp) == 1 for comp in marker_components), (
        "expected all marker-color components to be singleton pixels"
    )
    assert len(components_by_color[frame_color]) == 1, "expected no detached frame-color pixels"
    assert len(frame_component) == int(np.count_nonzero(g == frame_color))

    markers = np.argwhere(g == marker_color)
    assert len(markers) > 0, "stripe period must be positive"
    assert np.all(
        (markers[:, 0] < r0) | (markers[:, 0] > r1) |
        (markers[:, 1] < c0) | (markers[:, 1] > c1)
    ), "markers must lie outside the frame"

    period = int(len(markers))
    height, width = r1 - r0 + 1, c1 - c0 + 1
    out = np.full(g.shape, bg, dtype=int)
    if width > height:
        rows = np.flatnonzero((np.arange(g.shape[0]) - r0) % period == 0)
        out[rows, :] = frame_color
    else:
        cols = np.flatnonzero((np.arange(g.shape[1]) - c0) % period == 0)
        out[:, cols] = frame_color
    return out


_RNG = np.random.default_rng()
_GEN_CALLS = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _oracle(shape: tuple[int, int], bg: int, frame_color: int,
            horizontal: bool, anchor: int, period: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    axis_size = shape[0] if horizontal else shape[1]
    indices = np.flatnonzero((np.arange(axis_size) - anchor) % period == 0)
    if horizontal:
        out[indices, :] = frame_color
    else:
        out[:, indices] = frame_color
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _generate_attempt(horizontal: bool):
    h = int(_RNG.integers(15, 26))
    w = int(_RNG.integers(15, 26))
    period = int(_RNG.integers(3, 8))

    if horizontal:
        frame_h = int(_RNG.integers(3, 7))
        max_w = min(11, w - 2)
        if frame_h + 1 > max_w:
            return None
        frame_w = int(_RNG.integers(frame_h + 1, max_w + 1))
        valid_top = [r for r in range(1, h - frame_h) if r >= period and h - 1 - r >= period]
        if not valid_top:
            return None
        r0 = int(_RNG.choice(valid_top))
        c0 = int(_RNG.integers(1, w - frame_w))
    else:
        frame_w = int(_RNG.integers(3, 7))
        max_h = min(11, h - 2)
        if frame_w + 1 > max_h:
            return None
        frame_h = int(_RNG.integers(frame_w + 1, max_h + 1))
        valid_left = [c for c in range(1, w - frame_w) if c >= period and w - 1 - c >= period]
        if not valid_left:
            return None
        c0 = int(_RNG.choice(valid_left))
        r0 = int(_RNG.integers(1, h - frame_h))

    if period in (frame_h, frame_w):
        return None
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    palette = list(range(10))
    bg, frame_color, marker_color = map(int, _RNG.choice(palette, size=3, replace=False))
    grid = np.full((h, w), bg, dtype=int)
    _draw_frame(grid, (r0, c0, r1, c1), frame_color)

    # Keep markers visually separate from the frame and from one another (8-neighborhood).
    candidates = [
        (r, c) for r in range(h) for c in range(w)
        if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)
    ]
    _RNG.shuffle(candidates)
    markers: list[tuple[int, int]] = []
    for candidate in candidates:
        if all(max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 2 for r, c in markers):
            markers.append(candidate)
            if len(markers) == period:
                break
    if len(markers) != period:
        return None
    for r, c in markers:
        grid[r, c] = marker_color

    anchor = r0 if horizontal else c0
    expected = _oracle(grid.shape, bg, frame_color, horizontal, anchor, period)
    signature = (expected.shape, expected.tobytes())
    if signature in _SEEN_OUTPUTS:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic checks: both directions, background gaps, erasure, and every control
    # dependency must be observable rather than merely self-consistent.
    stripe_axis = np.flatnonzero(np.any(expected == frame_color, axis=1 if horizontal else 0))
    if len(stripe_axis) < 3 or not np.any(expected == bg):
        return None
    reduced = grid.copy()
    reduced[markers[-1]] = bg
    if np.array_equal(main(reduced), expected):
        return None
    replacement = next(color for color in palette if color not in (bg, frame_color, marker_color))
    recolored = grid.copy()
    recolored[recolored == frame_color] = replacement
    if np.array_equal(main(recolored), expected):
        return None
    return grid, signature


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples with alternating orientation coverage."""
    global _GEN_CALLS
    horizontal = (_GEN_CALLS % 2 == 0)
    for _ in range(500):
        result = _generate_attempt(horizontal)
        if result is not None:
            grid, output_signature = result
            _SEEN_OUTPUTS.add(output_signature)
            _GEN_CALLS += 1
            return grid
    raise RuntimeError("failed to generate a valid periodic-stripe input")
