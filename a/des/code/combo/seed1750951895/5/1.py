from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _interior_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _is_frame_color(g: np.ndarray, color: int) -> bool:
    coords = np.argwhere(g == color)
    if coords.shape[0] < 4:
        return False
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    side_hits = [False, False, False, False]
    for r, c in coords:
        r = int(r)
        c = int(c)
        on_top = r == r0
        on_bottom = r == r1
        on_left = c == c0
        on_right = c == c1
        if not (on_top or on_bottom or on_left or on_right):
            return False
        side_hits[0] |= on_top
        side_hits[1] |= on_bottom
        side_hits[2] |= on_left
        side_hits[3] |= on_right
    return all(side_hits)


def _recover_frames(g: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        if not _is_frame_color(g, color):
            continue
        coords = np.argwhere(g == color)
        box = _bbox_from_coords(coords)
        frames.append({"color": color, "box": box, "border": _border_cells(box)})
    assert frames, "expected at least one recoverable frame"
    return frames


def _topological_bottom_to_top(frames: list[dict], g: np.ndarray) -> list[dict]:
    by_color = {frame["color"]: frame for frame in frames}
    outgoing: dict[int, set[int]] = defaultdict(set)
    indegree = {frame["color"]: 0 for frame in frames}

    for i, fa in enumerate(frames):
        for fb in frames[i + 1 :]:
            inter = fa["border"] & fb["border"]
            if not inter:
                continue
            visible = {int(g[r, c]) for r, c in inter if int(g[r, c]) in (fa["color"], fb["color"])}
            assert len(visible) == 1, "pairwise overlap must expose one top frame color"
            top = next(iter(visible))
            bottom = fb["color"] if top == fa["color"] else fa["color"]
            if top not in outgoing[bottom]:
                outgoing[bottom].add(top)
                indegree[top] += 1

    order: list[int] = []
    ready = [color for color, deg in indegree.items() if deg == 0]
    while ready:
        assert len(ready) == 1, "frame order must be uniquely determined by occlusion"
        color = ready.pop()
        order.append(color)
        for nxt in sorted(outgoing[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == len(frames), "occlusion graph must define a full stacking order"
    return [by_color[color] for color in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapping rectangle frames, infer their stacking order, extract each
    frame's unique marker color, and redraw those marker colors as concentric rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _recover_frames(g)
    frame_colors = {frame["color"] for frame in frames}
    ordered = _topological_bottom_to_top(frames, g)

    marker_colors = []
    for frame in ordered:
        r0, c0, r1, c1 = frame["box"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_pos = np.argwhere((inner != 0) & (~np.isin(inner, list(frame_colors))))
        assert marker_pos.shape[0] == 1, "each frame interior must contain exactly one visible marker"
        rr, cc = map(int, marker_pos[0])
        marker_colors.append(int(inner[rr, cc]))

    n = len(marker_colors)
    out = np.zeros((2 * n - 1, 2 * n - 1), dtype=int)
    for layer, color in enumerate(marker_colors):
        r0 = layer
        c0 = layer
        r1 = out.shape[0] - 1 - layer
        c1 = out.shape[1] - 1 - layer
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 26
FRAME_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 6
RECT_SIDE_MAX = 13
MAX_RECT_TRIES = 120
MAX_GEN_ATTEMPTS = 500


def _sample_rect(h: int, w: int) -> tuple[int, int, int, int]:
    rh = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, h - 2))
    rw = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, w - 2))
    r0 = random.randint(1, h - rh - 1)
    c0 = random.randint(1, w - rw - 1)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _draw_frame(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _build_expected(marker_colors: list[int]) -> np.ndarray:
    n = len(marker_colors)
    out = np.zeros((2 * n - 1, 2 * n - 1), dtype=int)
    for layer, color in enumerate(marker_colors):
        r0 = layer
        c0 = layer
        r1 = out.shape[0] - 1 - layer
        c1 = out.shape[1] - 1 - layer
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _pick_marker_positions(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]] | None:
    interiors = [_interior_cells(box) for box in rects]
    all_borders = set().union(*(_border_cells(box) for box in rects))
    positions = []
    used = set()
    for i, cells in enumerate(interiors):
        exclusive = [
            cell
            for cell in cells
            if cell not in all_borders
            and cell not in used
            and sum(cell in other for other in interiors) == 1
        ]
        if not exclusive:
            return None
        choice = random.choice(exclusive)
        positions.append(choice)
        used.add(choice)
    return positions


def _sequence_boxes(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for i in range(n):
        success = False
        for _ in range(MAX_RECT_TRIES):
            box = _sample_rect(h, w)
            if box in rects:
                continue
            if i > 0 and not (_border_cells(box) & _border_cells(rects[-1])):
                continue
            rects.append(box)
            success = True
            break
        if not success:
            return None
    return rects


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n = random.choice(FRAME_COUNT_CHOICES)

    colors = random.sample(range(1, 10), 2 * n)
    frame_colors = colors[:n]
    marker_colors = colors[n:]

    rects = _sequence_boxes(h, w, n)
    if rects is None:
        return None
    marker_positions = _pick_marker_positions(rects)
    if marker_positions is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for box, color in zip(rects, frame_colors):
        _draw_frame(grid, box, color)
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = color

    expected = _build_expected(marker_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
