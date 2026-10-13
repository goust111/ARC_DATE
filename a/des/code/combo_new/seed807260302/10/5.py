from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
SIDES = ("top", "bottom", "left", "right")
COLORS = tuple(range(1, 10))
MARKER_LENGTHS = (3, 5, 7)
MAX_GENERATION_ATTEMPTS = 500

ARROW_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "top": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    "bottom": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "left": ((0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
    "right": ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0)),
}
SHAPE_TO_SIDE = {frozenset(v): k for k, v in ARROW_SHAPES.items()}


def _same_color_runs(line: np.ndarray, min_len: int = 2) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    i = 0
    n = int(line.size)
    while i < n:
        color = int(line[i])
        if color == BLACK:
            i += 1
            continue
        start = i
        while i < n and int(line[i]) == color:
            i += 1
        end = i - 1
        if end - start + 1 >= min_len:
            runs.append({"color": color, "start": int(start), "end": int(end)})
    return runs


def _marker_cells(side: str, start: int, end: int, height: int, width: int) -> set[tuple[int, int]]:
    if side == "top":
        return {(0, c) for c in range(start, end + 1)}
    if side == "bottom":
        return {(height - 1, c) for c in range(start, end + 1)}
    if side == "left":
        return {(r, 0) for r in range(start, end + 1)}
    if side == "right":
        return {(r, width - 1) for r in range(start, end + 1)}
    raise AssertionError("unknown side")


def _edge_markers(g: np.ndarray) -> dict[str, dict[str, int]]:
    height, width = map(int, g.shape)
    raw = {
        "top": _same_color_runs(g[0, :]),
        "bottom": _same_color_runs(g[height - 1, :]),
        "left": _same_color_runs(g[:, 0]),
        "right": _same_color_runs(g[:, width - 1]),
    }

    markers: dict[str, dict[str, int]] = {}
    for side, runs in raw.items():
        assert len(runs) <= 1, "each side has at most one marker strip"
        if not runs:
            continue
        run = runs[0]
        start, end = int(run["start"]), int(run["end"])
        length = end - start + 1
        assert length % 2 == 1, "marker strip center must be a grid cell"
        markers[side] = {
            "color": int(run["color"]),
            "start": start,
            "end": end,
            "center": (start + end) // 2,
        }

    colors = [m["color"] for m in markers.values()]
    assert len(colors) == len(set(colors)), "edge marker colors must be distinct"
    return markers


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(height):
        for sc in range(width):
            if bool(seen[sr, sc]) or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            queue: deque[tuple[int, int]] = deque([(sr, sc)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if bool(seen[rr, cc]) or not bool(mask[rr, cc]):
                        continue
                    seen[rr, cc] = True
                    queue.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _arrow_side(coords: np.ndarray) -> tuple[str, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each arrow has a 3x3 bbox"
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert rel in SHAPE_TO_SIDE, "unknown 5-pixel arrow shape"
    return SHAPE_TO_SIDE[rel], r0, c0


def _target_top_left(side: str, marker: dict[str, int], height: int, width: int) -> tuple[int, int]:
    center = int(marker["center"])
    if side == "top":
        return 0, center - 1
    if side == "bottom":
        return height - 3, center - 1
    if side == "left":
        return center - 1, 0
    if side == "right":
        return center - 1, width - 3
    raise AssertionError("unknown side")


def _arrow_cells(side: str, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + dr), int(c0 + dc)) for dr, dc in ARROW_SHAPES[side]]


def _paint_arrow(g: np.ndarray, side: str, r0: int, c0: int, color: int) -> None:
    height, width = map(int, g.shape)
    for r, c in _arrow_cells(side, r0, c0):
        assert 0 <= r < height and 0 <= c < width, "arrow placement must fit"
        g[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored triangular arrow sprites to the edge they point at, centered on matching color strips.

    Input structure:
    - The background is black.
    - Each grid side has zero or one odd-length, one-pixel-thick colored marker strip on the boundary.
    - Interior foreground components are monochrome 5-pixel triangular arrow sprites pointing top, bottom, left, or right.
    - Each arrow color equals the marker-strip color on the edge it points toward.

    Rule:
    - Remove every arrow from its scattered interior source position.
    - Keep the arrow's color and orientation unchanged.
    - Place it against the edge indicated by its tip, with its front pixel on the boundary and its centerline aligned to the center of the same-colored marker strip on that edge.
    - All marker-strip cells and unrelated background cells remain unchanged.

    Key insight:
    The arrow geometry encodes the destination side, while the same-colored edge strip independently encodes the landing coordinate along that side.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    markers = _edge_markers(g)

    marker_cells: set[tuple[int, int]] = set()
    for side, marker in markers.items():
        marker_cells |= _marker_cells(side, int(marker["start"]), int(marker["end"]), height, width)

    movable = g != BLACK
    for r, c in marker_cells:
        movable[r, c] = False

    out = g.copy()
    arrows: list[dict[str, int | str | np.ndarray]] = []
    for coords in _components(movable):
        assert int(coords.shape[0]) == 5, "each movable component must be a 5-pixel arrow"
        vals = [int(g[r, c]) for r, c in coords]
        color = vals[0]
        assert all(v == color for v in vals), "each arrow must be monochrome"
        side, _r0, _c0 = _arrow_side(coords)
        assert side in markers, "arrow must point to a marked side"
        assert int(markers[side]["color"]) == color, "arrow color must match its destination marker"
        arrows.append({"coords": coords, "side": side, "color": color})
        out[coords[:, 0], coords[:, 1]] = BLACK

    for arrow in arrows:
        side = str(arrow["side"])
        color = int(arrow["color"])
        tr, tc = _target_top_left(side, markers[side], height, width)
        for r, c in _arrow_cells(side, tr, tc):
            assert int(out[r, c]) in (BLACK, color), "moved arrows must not cover other colors"
        _paint_arrow(out, side, tr, tc, color)

    return out


def _paint_marker(g: np.ndarray, side: str, start: int, end: int, color: int) -> None:
    for r, c in _marker_cells(side, start, end, int(g.shape[0]), int(g.shape[1])):
        g[r, c] = int(color)


def _sample_marker(side: str, height: int, width: int) -> tuple[int, int]:
    span = width if side in ("top", "bottom") else height
    lengths = [length for length in MARKER_LENGTHS if length <= span - 6]
    if not lengths:
        return -1, -1
    length = random.choice(lengths)
    half = length // 2
    center = random.randint(2 + half, span - 3 - half)
    return center - half, center + half


def _can_paint_output_arrow(g: np.ndarray, side: str, r0: int, c0: int, color: int) -> bool:
    height, width = map(int, g.shape)
    for r, c in _arrow_cells(side, r0, c0):
        if not (0 <= r < height and 0 <= c < width):
            return False
        if int(g[r, c]) not in (BLACK, int(color)):
            return False
    return True


def _can_place_source(g: np.ndarray, side: str, r0: int, c0: int) -> bool:
    height, width = map(int, g.shape)
    cells = set(_arrow_cells(side, r0, c0))
    for r, c in cells:
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            return False
        if int(g[r, c]) != BLACK:
            return False
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                if (rr, cc) not in cells and int(g[rr, cc]) != BLACK:
                    return False
    return True


def _place_source_arrow(g: np.ndarray, side: str, color: int) -> bool:
    height, width = map(int, g.shape)
    candidates = [(r, c) for r in range(1, height - 3) for c in range(1, width - 3)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place_source(g, side, r0, c0):
            continue
        _paint_arrow(g, side, r0, c0, color)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 30)
    width = random.randint(12, 30)
    n_arrows = random.randint(2, 4)
    sides = random.sample(list(SIDES), n_arrows)
    colors = random.sample(list(COLORS), n_arrows)

    marker_specs: dict[str, dict[str, int]] = {}
    markers_only = np.full((height, width), BLACK, dtype=int)
    for side, color in zip(sides, colors):
        start, end = _sample_marker(side, height, width)
        if start < 0:
            return None
        marker_specs[side] = {
            "color": int(color),
            "start": int(start),
            "end": int(end),
            "center": int((start + end) // 2),
        }
        _paint_marker(markers_only, side, start, end, color)

    expected = markers_only.copy()
    for side in sides:
        marker = marker_specs[side]
        color = int(marker["color"])
        r0, c0 = _target_top_left(side, marker, height, width)
        if not _can_paint_output_arrow(expected, side, r0, c0, color):
            return None
        _paint_arrow(expected, side, r0, c0, color)

    inp = markers_only.copy()
    order = list(sides)
    random.shuffle(order)
    for side in order:
        if not _place_source_arrow(inp, side, int(marker_specs[side]["color"])):
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/10/5 input")
