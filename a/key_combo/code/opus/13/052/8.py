from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
SEP = 5

GEN_ATTEMPTS = 1200
PANEL_SIZE = 6  # 4 * 6 + 3 separators = 27 <= 30

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)
VECTORS: tuple[tuple[int, int], ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _sorted_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _connected4(coords: tuple[tuple[int, int], ...]) -> bool:
    if not coords:
        return False
    cells = set(coords)
    seen = set()
    stack = [coords[0]]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return seen == cells


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    rel = _sorted_coords((r - r0, c - c0) for r, c in coords)
    return (r0, c0), rel


def _component(panel: np.ndarray, color: int) -> tuple[tuple[int, int], ...]:
    coords = _sorted_coords(map(tuple, np.argwhere(panel == color)))
    assert 2 <= len(coords) <= 3, f"expected a 2-3 pixel sprite for color {color}"
    assert _connected4(coords), f"sprite color {color} must be one 4-connected component"
    return coords


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 3, "expected exactly three separator columns"
    assert all(b - a >= 2 for a, b in zip(sep_cols, sep_cols[1:])), "separator columns must be isolated"

    panels: list[np.ndarray] = []
    start = 0
    for stop in sep_cols + [w]:
        panels.append(g[:, start:stop])
        start = stop + 1

    assert len(panels) == 4, "expected four panels"
    shape = panels[0].shape
    assert shape[0] == shape[1], "panels must be square"
    assert all(panel.shape == shape for panel in panels), "panel sizes must match"
    for panel in panels:
        assert set(map(int, np.unique(panel))) <= {BG, BLUE, RED}
    return panels


def _infer_motion(
    panels: list[np.ndarray], color: int
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int], tuple[tuple[int, int], ...], list[tuple[int, int]]]:
    components = [_component(panel, color) for panel in panels]
    anchors: list[tuple[int, int]] = []
    rel0: tuple[tuple[int, int], ...] | None = None
    for comp in components:
        anchor, rel = _normalize(comp)
        anchors.append(anchor)
        if rel0 is None:
            rel0 = rel
        else:
            assert rel == rel0, f"shape for color {color} must remain constant"
    assert rel0 is not None

    deltas = [
        (anchors[i + 1][0] - anchors[i][0], anchors[i + 1][1] - anchors[i][1])
        for i in range(3)
    ]
    assert deltas[0] == deltas[1] == deltas[2], f"motion for color {color} must be constant"
    return components[-1], deltas[0], rel0, anchors


def _predict(coords: tuple[tuple[int, int], ...], delta: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in coords}


def _visible(cells: Iterable[tuple[int, int]], size: int) -> set[tuple[int, int]]:
    return {(r, c) for r, c in cells if 0 <= r < size and 0 <= c < size}


def _render_output(size: int, red_cells: Iterable[tuple[int, int]], blue_cells: Iterable[tuple[int, int]]) -> np.ndarray:
    red_vis = _visible(red_cells, size)
    blue_vis = _visible(blue_cells, size)
    overlap = red_vis & blue_vis
    out = np.zeros((size, size), dtype=int)
    for r, c in red_vis - overlap:
        out[r, c] = RED
    for r, c in blue_vis - overlap:
        out[r, c] = BLUE
    for r, c in overlap:
        out[r, c] = GREEN
    return out


def _render_panel(size: int, red_cells: Iterable[tuple[int, int]], blue_cells: Iterable[tuple[int, int]]) -> np.ndarray:
    red_vis = _visible(red_cells, size)
    blue_vis = _visible(blue_cells, size)
    assert not (red_vis & blue_vis)
    out = np.zeros((size, size), dtype=int)
    for r, c in red_vis:
        out[r, c] = RED
    for r, c in blue_vis:
        out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the fifth panel by extrapolating two independent sprite motions and recoloring their overlaps green.

    Input structure:
    - The input is a row of four equal square panels separated by one-column gray dividers.
    - Each panel has black background plus exactly one red sprite and one blue sprite.
    - For each color, the sprite shape is fixed across panels and only its position changes.

    Rule:
    - Split the strip into four panels and independently track the red and blue sprite coordinates.
    - For each color, verify that the per-panel displacement vector is constant, then apply that same vector once more to the fourth panel.
    - Produce one panel of the same size: red-only predicted cells stay red, blue-only predicted cells stay blue, overlapping predicted cells become green, and out-of-bounds predicted cells are discarded.

    Key insight:
    - The output is the superposition of two separately extrapolated motions; green cells appear only where the two predicted next-frame positions coincide.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    size = panels[0].shape[0]

    red_last, red_delta, _red_shape, _red_anchors = _infer_motion(panels, RED)
    blue_last, blue_delta, _blue_shape, _blue_anchors = _infer_motion(panels, BLUE)

    red_next = _predict(red_last, red_delta)
    blue_next = _predict(blue_last, blue_delta)
    return _render_output(size, red_next, blue_next)


def _shape_extent(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape), max(c for _, c in shape)


def _place_shape(anchor: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in shape}


def _valid_start_anchors(
    size: int, shape: tuple[tuple[int, int], ...], delta: tuple[int, int]
) -> list[tuple[int, int]]:
    max_r, max_c = _shape_extent(shape)
    dr, dc = delta
    valid: list[tuple[int, int]] = []
    for r0 in range(size):
        for c0 in range(size):
            ok = True
            for t in range(4):
                rr = r0 + t * dr
                cc = c0 + t * dc
                if rr < 0 or cc < 0 or rr + max_r >= size or cc + max_c >= size:
                    ok = False
                    break
            if ok:
                valid.append((r0, c0))
    random.shuffle(valid)
    return valid


def _trajectory(
    anchor0: tuple[int, int], shape: tuple[tuple[int, int], ...], delta: tuple[int, int]
) -> list[set[tuple[int, int]]]:
    dr, dc = delta
    return [
        _place_shape((anchor0[0] + t * dr, anchor0[1] + t * dc), shape)
        for t in range(5)
    ]


def _min_l1_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _find_case(
    size: int,
    red_shape: tuple[tuple[int, int], ...],
    blue_shape: tuple[tuple[int, int], ...],
    red_delta: tuple[int, int],
    blue_delta: tuple[int, int],
    want_collision: bool,
    want_clip: bool,
    want_near_miss: bool,
) -> tuple[list[set[tuple[int, int]]], list[set[tuple[int, int]]], bool] | None:
    red_starts = _valid_start_anchors(size, red_shape, red_delta)
    blue_starts = _valid_start_anchors(size, blue_shape, blue_delta)
    if not red_starts or not blue_starts:
        return None

    for red_anchor in red_starts:
        red_traj = _trajectory(red_anchor, red_shape, red_delta)
        for blue_anchor in blue_starts:
            blue_traj = _trajectory(blue_anchor, blue_shape, blue_delta)
            if any(red_traj[t] & blue_traj[t] for t in range(4)):
                continue

            red5 = _visible(red_traj[4], size)
            blue5 = _visible(blue_traj[4], size)
            if not red5 or not blue5:
                continue

            overlap = red5 & blue5
            if want_collision:
                if not overlap:
                    continue
                if len(overlap) == len(red5) or len(overlap) == len(blue5):
                    continue
            else:
                if overlap:
                    continue
                if want_near_miss and _min_l1_distance(red5, blue5) != 1:
                    continue

            clipped = len(red5) != len(red_traj[4]) or len(blue5) != len(blue_traj[4])
            if want_clip and not clipped:
                continue

            out = _render_output(size, red_traj[4], blue_traj[4])
            last = _render_panel(size, red_traj[3], blue_traj[3])
            red_static = _render_output(size, red_traj[3], blue_traj[4])
            blue_static = _render_output(size, red_traj[4], blue_traj[3])
            if np.array_equal(out, last):
                continue
            if np.array_equal(out, red_static) or np.array_equal(out, blue_static):
                continue
            if any(np.array_equal(out, _render_panel(size, red_traj[t], blue_traj[t])) for t in range(4)):
                continue

            green_count = int(np.sum(out == GREEN))
            if want_collision and green_count < 1:
                continue
            if not want_collision and green_count != 0:
                continue

            return red_traj, blue_traj, clipped
    return None


def _assemble_grid(size: int, red_traj: list[set[tuple[int, int]]], blue_traj: list[set[tuple[int, int]]]) -> np.ndarray:
    panels = [_render_panel(size, red_traj[t], blue_traj[t]) for t in range(4)]
    sep = np.full((size, 1), SEP, dtype=int)
    row = [panels[0]]
    for panel in panels[1:]:
        row.append(sep.copy())
        row.append(panel)
    return np.hstack(row)


def _validate_generated_case(
    grid: np.ndarray, *, want_collision: bool, want_near_miss: bool
) -> None:
    panels = _split_panels(grid)
    size = panels[0].shape[0]
    red_last, red_delta, _red_shape, _red_anchors = _infer_motion(panels, RED)
    blue_last, blue_delta, _blue_shape, _blue_anchors = _infer_motion(panels, BLUE)

    expected = _render_output(size, _predict(red_last, red_delta), _predict(blue_last, blue_delta))
    actual = main(grid)
    assert np.array_equal(actual, expected)

    overlap = set(map(tuple, np.argwhere(actual == GREEN)))
    assert bool(overlap) == want_collision

    if want_collision:
        assert np.any(actual == RED) and np.any(actual == BLUE)
    if want_near_miss:
        red_next = _visible(_predict(red_last, red_delta), size)
        blue_next = _visible(_predict(blue_last, blue_delta), size)
        assert not (red_next & blue_next)
        assert _min_l1_distance(red_next, blue_next) == 1

    for panel in panels:
        assert not np.array_equal(panel, actual), "output should not duplicate an observed panel"


def _gen_attempt(want_collision: bool) -> np.ndarray | None:
    want_clip = random.random() < 0.35
    want_near_miss = (not want_collision) and (random.random() < 0.6)

    red_shape, blue_shape = random.sample(SHAPES, 2)
    red_delta, blue_delta = random.sample(VECTORS, 2)

    case = _find_case(
        PANEL_SIZE,
        red_shape,
        blue_shape,
        red_delta,
        blue_delta,
        want_collision=want_collision,
        want_clip=want_clip,
        want_near_miss=want_near_miss,
    )
    if case is None and want_clip:
        case = _find_case(
            PANEL_SIZE,
            red_shape,
            blue_shape,
            red_delta,
            blue_delta,
            want_collision=want_collision,
            want_clip=False,
            want_near_miss=want_near_miss,
        )
    if case is None:
        return None

    red_traj, blue_traj, _clipped = case
    grid = _assemble_grid(PANEL_SIZE, red_traj, blue_traj)
    _validate_generated_case(grid, want_collision=want_collision, want_near_miss=want_near_miss)
    return grid


def generate_input() -> np.ndarray:
    want_collision = random.random() < 0.5
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(want_collision)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-trajectory panel sequence")
