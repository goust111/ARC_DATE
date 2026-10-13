from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GRAY = 5
TEAL = 8
BAR_COLORS = (1, 2, 3, 6, 7, 9)

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components from a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers = []
    for coords in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 4 and w >= 4, "container must have visible interior"

        expected = np.zeros((h, w), dtype=bool)
        expected[:, 0] = True
        expected[:, -1] = True
        expected[-1, :] = True
        assert np.array_equal(patch == GRAY, expected), "gray component must be a U container"

        interior = patch[:-1, 1:-1]
        assert interior.size > 0
        assert np.all((interior == BG) | (interior == YELLOW)), "container interior must be empty or liquid"

        liquid = interior == YELLOW
        liquid_rows = np.flatnonzero(liquid.any(axis=1))
        assert liquid_rows.size > 0, "container must contain liquid"
        top_liquid = int(liquid_rows[0])
        assert np.all(liquid[top_liquid:, :]), "liquid must fill full rows"
        assert not liquid[:top_liquid, :].any(), "liquid must be bottom-aligned"

        inner_height = h - 1
        liquid_height = inner_height - top_liquid
        headspace = inner_height - liquid_height
        assert 1 <= headspace < inner_height, "generator keeps motion distance positive"

        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "row_lo": r0,
                "row_hi": r1 - 1,
                "headspace": headspace,
            }
        )

    containers.sort(key=lambda item: item["row_lo"])
    for prev, cur in zip(containers, containers[1:]):
        assert prev["row_hi"] < cur["row_lo"], "container row ranges must not overlap"
    assert containers, "expected at least one container"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside marker horizontally toward the nearer edge bar by its container headspace, then repaint it with that bar color.

    Input structure:
    - The leftmost and rightmost columns are solid vertical edge bars with different non-background colors.
    - Between the bars, several gray U-shaped containers sit on a black background, each holding yellow liquid that fills full bottom-aligned rows.
    - Each container contributes one teal single-pixel marker outside the container on one of the same interior rows.

    Rule:
    - For each teal marker, find the unique gray container whose interior row span includes the marker row.
    - Count the number of empty interior rows above that container's yellow liquid.
    - Move the marker horizontally by that many cells toward the nearer edge bar, clear the original marker cell, and paint the landing cell with the nearer bar's color.

    Key insight:
    - Direction comes from the marker's nearer full-height edge bar, but the travel distance comes from the associated container's liquid headspace instead of from the marker position itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 8

    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])
    assert left_color != BG and right_color != BG and left_color != right_color
    assert np.all(g[:, 0] == left_color), "left edge must be a full bar"
    assert np.all(g[:, w - 1] == right_color), "right edge must be a full bar"

    containers = _find_containers(g)
    markers = np.argwhere(g == TEAL)
    assert markers.shape[0] == len(containers), "expected one marker per container"

    out = g.copy()
    for r0, c0 in markers:
        r = int(r0)
        c = int(c0)
        owners = [box for box in containers if box["row_lo"] <= r <= box["row_hi"]]
        assert len(owners) == 1, "marker row must belong to exactly one container"
        headspace = int(owners[0]["headspace"])

        left_dist = c
        right_dist = (w - 1) - c
        assert left_dist != right_dist, "nearer bar must be unique"

        if left_dist < right_dist:
            target_c = c - headspace
            assert target_c >= 1
            assert np.all(g[r, target_c:c] == BG), "leftward path must be clear"
            out[r, c] = BG
            out[r, target_c] = left_color
        else:
            target_c = c + headspace
            assert target_c <= w - 2
            assert np.all(g[r, c + 1 : target_c + 1] == BG), "rightward path must be clear"
            out[r, c] = BG
            out[r, target_c] = right_color

    return out


def _draw_container(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, liquid_height: int) -> int:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    liquid_top = r1 - liquid_height
    grid[liquid_top:r1, c0 + 1 : c1] = YELLOW
    return outer_h - 1 - liquid_height


def _valid_marker_columns(width: int, c0: int, c1: int, headspace: int, side: str) -> list[int]:
    mid = (width - 1) / 2.0
    if side == "left":
        return [c for c in range(headspace + 1, c0) if c < mid]
    return [c for c in range(c1 + 1, width - 1 - headspace) if c > mid]


def _expected_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int, str]],
    left_color: int,
    right_color: int,
) -> np.ndarray:
    out = grid.copy()
    for r, c, headspace, side in markers:
        out[r, c] = BG
        if side == "left":
            out[r, c - headspace] = left_color
        else:
            out[r, c + headspace] = right_color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_containers = random.randint(2, 4)
    outer_heights = [random.randint(4, 7) for _ in range(num_containers)]
    outer_widths = [random.randint(4, 7) for _ in range(num_containers)]
    gaps = [random.randint(1, 3) for _ in range(num_containers - 1)]

    total_height = 1 + sum(outer_heights) + sum(gaps) + 1
    if total_height > 30:
        return None

    min_width = max(max(outer_widths) + 8, 18)
    width = random.randint(min_width, 30)
    left_color, right_color = random.sample(BAR_COLORS, 2)

    grid = np.full((total_height, width), BG, dtype=int)
    grid[:, 0] = left_color
    grid[:, width - 1] = right_color

    if num_containers == 1:
        sides = [random.choice(("left", "right"))]
    else:
        sides = ["left", "right"]
        while len(sides) < num_containers:
            sides.append(random.choice(("left", "right")))
        random.shuffle(sides)

    markers: list[tuple[int, int, int, str]] = []
    r = 1
    for outer_h, outer_w, side in zip(outer_heights, outer_widths, sides):
        inner_h = outer_h - 1
        liquid_height = random.randint(1, inner_h - 1)
        headspace = inner_h - liquid_height

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            c0 = random.randint(2, width - outer_w - 3)
            c1 = c0 + outer_w - 1
            marker_cols = _valid_marker_columns(width, c0, c1, headspace, side)
            if not marker_cols:
                continue

            marker_row = random.randint(r, r + outer_h - 2)
            marker_col = random.choice(marker_cols)

            _draw_container(grid, r, c0, outer_h, outer_w, liquid_height)
            if grid[marker_row, marker_col] != BG:
                grid[r : r + outer_h, c0 : c1 + 1] = BG
                continue

            if side == "left":
                target_col = marker_col - headspace
                if not np.all(grid[marker_row, target_col:marker_col] == BG):
                    grid[r : r + outer_h, c0 : c1 + 1] = BG
                    continue
            else:
                target_col = marker_col + headspace
                if not np.all(grid[marker_row, marker_col + 1 : target_col + 1] == BG):
                    grid[r : r + outer_h, c0 : c1 + 1] = BG
                    continue

            grid[marker_row, marker_col] = TEAL
            markers.append((marker_row, marker_col, headspace, side))
            placed = True
            break

        if not placed:
            return None
        r += outer_h
        if r < total_height - 1:
            r += gaps[len(markers) - 1] if len(markers) - 1 < len(gaps) else 0

    expected = _expected_output(grid, markers, left_color, right_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid liquid-shift input")
