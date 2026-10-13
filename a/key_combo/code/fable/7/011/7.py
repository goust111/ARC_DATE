from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBSTACLE_COLOR = 5

# The fixed pairings are demonstrated directly by every generated example.
# They are categorical ARC color pairings, not arithmetic on color ids.
MARKER_TO_OUTLINE = {1: 8, 2: 4}
DIRECTIONS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the common ARC convention: coordinates are always (row, column)."""
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _landing_cells(
    grid: np.ndarray,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return (r, c, color, marker_r, marker_c, distance) for all survivors."""
    occupied = grid != BACKGROUND
    landings: list[tuple[int, int, int, int, int, int]] = []

    for marker_color, outline_color in MARKER_TO_OUTLINE.items():
        for marker_r, marker_c in np.argwhere(grid == marker_color):
            mr, mc = int(marker_r), int(marker_c)
            for dr, dc in DIRECTIONS:
                distance = 1
                r, c = mr + dr, mc + dc
                while _in_bounds(grid.shape, r, c) and occupied[r, c]:
                    distance += 1
                    r += dr
                    c += dc
                if _in_bounds(grid.shape, r, c):
                    landings.append((r, c, outline_color, mr, mc, distance))
    return landings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a paired-color hollow 3x3 outline whose blocked cells slide outward.

    Input structure:
    - A black background contains isolated blue (1) and/or red (2) marker pixels.
    - Solid non-marker obstacle objects may occupy cells beside the markers; no
      outline colors occur in the input (the generator uses grey obstacles).

    Rule:
    - Blue markers pair with azure (8), and red markers pair with yellow (4).
    - Treat the eight neighbors of each marker as compass-directed outline
      units.  A unit stays on an empty neighbor, but if input foreground blocks
      it, move it repeatedly in that same compass direction to the first input
      background cell; discard it if it leaves the grid.
    - Outline units do not block one another.  Preserve every input pixel.

    Key insight:
    - The outline is eight independent directional sliders, so obstacles push
      individual frame cells radially outward instead of moving the whole frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 cells on either axis"

    landings = _landing_cells(g)
    colors_by_cell: dict[tuple[int, int], int] = {}
    for r, c, color, _mr, _mc, _distance in landings:
        old = colors_by_cell.get((r, c))
        assert old is None or old == color, "different outline colors cannot share a landing"
        colors_by_cell[r, c] = color

    out = g.copy()
    for (r, c), color in colors_by_cell.items():
        out[r, c] = color
    return out


def _obstacle_rectangle(
    marker: tuple[int, int], direction: tuple[int, int], length: int
) -> set[tuple[int, int]]:
    """Make a filled rectangle that blocks one compass ray for `length` cells."""
    r, c = marker
    dr, dc = direction
    if dr == 0 or dc == 0:
        return {(r + step * dr, c + step * dc) for step in range(1, length + 1)}
    return {
        (r + row_step * dr, c + col_step * dc)
        for row_step in range(1, length + 1)
        for col_step in range(1, length + 1)
    }


def _oracle(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Independent generator oracle; records (marker_color, distance, survives)."""
    height, width = grid.shape
    original_foreground = {
        (int(r), int(c)) for r, c in np.argwhere(grid != BACKGROUND)
    }
    expected = grid.copy()
    records: list[tuple[int, int, int]] = []
    targets: dict[tuple[int, int], int] = {}

    for marker_color, paint_color in MARKER_TO_OUTLINE.items():
        markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == marker_color)]
        for mr, mc in markers:
            for dr, dc in DIRECTIONS:
                step = 1
                while True:
                    r, c = mr + step * dr, mc + step * dc
                    if not (0 <= r < height and 0 <= c < width):
                        records.append((marker_color, step, 0))
                        break
                    if (r, c) not in original_foreground:
                        previous = targets.get((r, c))
                        if previous is not None and previous != paint_color:
                            return grid.copy(), []
                        targets[r, c] = paint_color
                        records.append((marker_color, step, 1))
                        break
                    step += 1

    for (r, c), color in targets.items():
        expected[r, c] = color
    return expected, records


def _try_generate() -> np.ndarray | None:
    height = random.randint(15, 23)
    width = random.randint(15, 23)
    marker_count = random.randint(2, 3)

    # One edge marker guarantees that the out-of-grid discard branch is shown.
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        edge = (0 if side == "top" else height - 1, random.randint(3, width - 4))
    else:
        edge = (random.randint(3, height - 4), 0 if side == "left" else width - 1)
    markers = [edge]

    candidates = [
        (r, c)
        for r in range(3, height - 3)
        for c in range(3, width - 3)
        if all(max(abs(r - rr), abs(c - cc)) >= 7 for rr, cc in markers)
    ]
    random.shuffle(candidates)
    for candidate in candidates:
        if all(
            max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 7
            for r, c in markers
        ):
            markers.append(candidate)
            if len(markers) == marker_count:
                break
    if len(markers) != marker_count:
        return None

    marker_colors = [1, 2] + [random.choice((1, 2)) for _ in range(marker_count - 2)]
    random.shuffle(marker_colors)
    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in zip(markers, marker_colors):
        grid[r, c] = color

    # Give every marker one or more relevant solid obstacles.  The first
    # interior marker receives a length-2/3 obstacle, forcing repeated sliding.
    rectangles: list[set[tuple[int, int]]] = []
    for marker_index, marker in enumerate(markers):
        direction_order = list(DIRECTIONS)
        random.shuffle(direction_order)
        desired = random.randint(1, 3)
        placed = 0
        for direction in direction_order:
            if marker_index == 1 and placed == 0:
                length = random.randint(2, 3)
            else:
                length = random.randint(1, 3)
            cells = _obstacle_rectangle(marker, direction, length)
            if not cells or not all(_in_bounds(grid.shape, r, c) for r, c in cells):
                continue
            if any(grid[r, c] != BACKGROUND for r, c in cells):
                continue
            # Keep separate obstacle bodies visually readable.
            if any(cells & old for old in rectangles):
                continue
            for r, c in cells:
                grid[r, c] = OBSTACLE_COLOR
            rectangles.append(cells)
            placed += 1
            if placed == desired:
                break
        if placed == 0:
            return None

    expected, records = _oracle(grid)
    if not records:
        return None
    immediate = sum(survives and distance == 1 for _color, distance, survives in records)
    slid = sum(survives and distance > 1 for _color, distance, survives in records)
    long_slide = sum(survives and distance >= 3 for _color, distance, survives in records)
    discarded = sum(not survives for _color, _distance, survives in records)
    if min(immediate, slid, long_slide, discarded) < 1:
        return None
    if any(
        not any(color == marker_color and survives for color, _distance, survives in records)
        for marker_color in MARKER_TO_OUTLINE
    ):
        return None

    # Semantic checks: all input foreground is preserved, obstacle displacement
    # matters, and the result is not the naive clipped 3x3 frame.
    if not np.array_equal(expected[grid != BACKGROUND], grid[grid != BACKGROUND]):
        return None
    without_obstacles = grid.copy()
    without_obstacles[without_obstacles == OBSTACLE_COLOR] = BACKGROUND
    obstacle_free_expected, _ = _oracle(without_obstacles)
    expected_outline = np.isin(expected, tuple(MARKER_TO_OUTLINE.values()))
    obstacle_free_outline = np.isin(
        obstacle_free_expected, tuple(MARKER_TO_OUTLINE.values())
    )
    if np.array_equal(
        np.where(expected_outline, expected, BACKGROUND),
        np.where(obstacle_free_outline, obstacle_free_expected, BACKGROUND),
    ):
        return None
    if np.array_equal(expected, grid) or not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate inputs covering all semantic branches."""
    for _ in range(500):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-outline input")
