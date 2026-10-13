from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

RING_COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA)


def _ring_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "ring must have a nonempty interior"
    path: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        path.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    assert len(path) == len(set(path))
    return path


def _pointer_direction(points: list[tuple[int, int]]) -> tuple[int, int]:
    rows = sorted({r for r, _c in points})
    cols = sorted({c for _r, c in points})
    n = len(points)
    assert 1 <= n <= 4, "pointer length must be one to four pixels"

    if n == 1:
        # A single pixel has no visible orientation; generated one-pixel cases are
        # limited to 3x3 rings where a one-step shift is direction-independent.
        return n, 1

    if len(rows) == 1:
        assert len(cols) == n and cols == list(range(cols[0], cols[0] + n))
        return n, 1

    if len(cols) == 1:
        assert len(rows) == n and rows == list(range(rows[0], rows[0] + n))
        return n, -1

    raise AssertionError("pointer must be a straight horizontal or vertical line")


def _detect_rings(g: np.ndarray) -> list[dict]:
    rings: list[dict] = []
    structural = {BLACK, AZURE, MAROON}
    for raw_color in np.unique(g):
        color = int(raw_color)
        if color in structural:
            continue

        coords = np.argwhere(g == color)
        assert coords.size, "ring color must be present"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        path = _ring_path(r0, c0, r1, c1)
        corners = {(r0, c0), (r0, c1), (r1, c1), (r1, c0)}

        expected_body = set(path) - corners
        actual_body = {(int(r), int(c)) for r, c in coords}
        assert actual_body == expected_body, "ring body must be exactly one hollow rectangle"
        for r, c in corners:
            assert int(g[r, c]) == MAROON, "each ring corner must be maroon"

        rings.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "path": path,
                "corners": corners,
            }
        )

    assert rings, "expected at least one rectangular ring"
    rings.sort(key=lambda ring: ring["bbox"])
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the maroon corner markers around each rectangular ring according to its interior pointer.

    Input structure:
    - The background is black.
    - There are one or more hollow rectangular rings with distinct body colors.
    - The four corner cells of each ring are maroon, replacing the ring body color at those positions.
    - Inside each ring there is exactly one azure straight pointer segment of length one to four.

    Rule:
    - Unwrap each rectangular ring clockwise, starting at its top-left corner.
    - The pointer length N is the cyclic shift distance.
    - A horizontal pointer shifts every ring-path color N steps clockwise; a vertical pointer shifts every ring-path color N steps counterclockwise.
    - Clear the azure pointer cells to black after applying the shift.

    Key insight:
    - The colored ring body is uniform, so the visible motion is the maroon corner markers sliding along the perimeter cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rings = _detect_rings(g)

    azure_points = {(int(r), int(c)) for r, c in np.argwhere(g == AZURE)}
    assigned: set[tuple[int, int]] = set()
    out = g.copy()
    for r, c in azure_points:
        out[r, c] = BLACK

    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        pointer = sorted(
            (r, c)
            for r, c in azure_points
            if r0 < r < r1 and c0 < c < c1
        )
        assert pointer, "each ring must contain one pointer"
        assert not (assigned & set(pointer)), "pointers must belong to only one ring"
        assigned.update(pointer)

        steps, direction = _pointer_direction(pointer)
        path = ring["path"]
        period = len(path)
        colors = [int(g[r, c]) for r, c in path]
        for i, value in enumerate(colors):
            nr, nc = path[(i + direction * steps) % period]
            out[nr, nc] = value

    assert assigned == azure_points, "every azure pointer pixel must be inside exactly one ring"
    return out


# --------------- generator ---------------

GRID_MIN_SIDE = 14
GRID_MAX_SIDE = 30
RING_COUNT_RANGE = (2, 5)
RING_MAX_SIDE = 9
PLACEMENT_MARGIN = 1
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 500
PLACE_ATTEMPTS_PER_RING = 150


def _expanded_slice(
    top: int, left: int, height: int, width: int, rows: int, cols: int
) -> tuple[slice, slice]:
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(rows, top + height + PLACEMENT_PADDING)
    c1 = min(cols, left + width + PLACEMENT_PADDING)
    return slice(r0, r1), slice(c0, c1)


def _make_specs(count: int) -> list[dict]:
    orientations = ["h", "v"]
    orientations.extend(random.choice(["h", "v"]) for _ in range(count - 2))
    random.shuffle(orientations)

    use_single = random.random() < 0.35
    single_index = random.randrange(count) if use_single else -1
    specs: list[dict] = []
    for i, orientation in enumerate(orientations):
        if i == single_index:
            length = 1
            height = width = 3
        else:
            length = random.randint(2, 4)
            min_side = length + 2
            height = random.randint(min_side, RING_MAX_SIDE)
            width = random.randint(min_side, RING_MAX_SIDE)

        specs.append(
            {
                "orientation": orientation,
                "length": length,
                "height": height,
                "width": width,
            }
        )

    random.shuffle(specs)
    return specs


def _place_specs(rows: int, cols: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        height = spec["height"]
        width = spec["width"]
        if rows - height <= 2 * PLACEMENT_MARGIN or cols - width <= 2 * PLACEMENT_MARGIN:
            return None

        for _ in range(PLACE_ATTEMPTS_PER_RING):
            top = random.randint(PLACEMENT_MARGIN, rows - height - PLACEMENT_MARGIN)
            left = random.randint(PLACEMENT_MARGIN, cols - width - PLACEMENT_MARGIN)
            rr, cc = _expanded_slice(top, left, height, width, rows, cols)
            if occupied[rr, cc].any():
                continue

            occupied[rr, cc] = True
            item = dict(spec)
            item["top"] = top
            item["left"] = left
            placed.append(item)
            break
        else:
            return None

    return placed


def _paint_ring(grid: np.ndarray, spec: dict, color: int) -> None:
    top = spec["top"]
    left = spec["left"]
    height = spec["height"]
    width = spec["width"]
    bottom = top + height - 1
    right = left + width - 1

    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    for r, c in ((top, left), (top, right), (bottom, right), (bottom, left)):
        grid[r, c] = MAROON

    length = spec["length"]
    if spec["orientation"] == "h":
        row = random.randint(top + 1, bottom - 1)
        start = random.randint(left + 1, right - length)
        grid[row, start : start + length] = AZURE
    else:
        col = random.randint(left + 1, right - 1)
        start = random.randint(top + 1, bottom - length)
        grid[start : start + length, col] = AZURE


def _shifted_marker_cells(spec: dict, clockwise: bool) -> set[tuple[int, int]]:
    top = spec["top"]
    left = spec["left"]
    bottom = top + spec["height"] - 1
    right = left + spec["width"] - 1
    path = _ring_path(top, left, bottom, right)
    corners = {(top, left), (top, right), (bottom, right), (bottom, left)}
    corner_indices = [i for i, cell in enumerate(path) if cell in corners]
    direction = 1 if clockwise else -1
    return {path[(i + direction * spec["length"]) % len(path)] for i in corner_indices}


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    cols = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    count = random.randint(*RING_COUNT_RANGE)
    count = min(count, len(RING_COLORS))

    specs = _make_specs(count)
    placed = _place_specs(rows, cols, specs)
    if placed is None:
        return None

    colors = random.sample(RING_COLORS, len(placed))
    grid = np.full((rows, cols), BLACK, dtype=int)
    for spec, color in zip(placed, colors):
        marker_cells = _shifted_marker_cells(spec, spec["orientation"] == "h")
        top = spec["top"]
        left = spec["left"]
        bottom = top + spec["height"] - 1
        right = left + spec["width"] - 1
        corners = {(top, left), (top, right), (bottom, right), (bottom, left)}
        if marker_cells & corners:
            return None
        _paint_ring(grid, spec, color)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.any(out == AZURE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
