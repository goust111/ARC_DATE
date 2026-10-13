from __future__ import annotations

import random

import numpy as np


GRID_SIZE = 15
BG = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

MIN_GAP = 4
MAX_GAP = 8
MAX_GENERATION_ATTEMPTS = 300


def _find_unique_cell(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one {color} marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_centers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 15x15 grid"
    allowed = {BG, BLUE, RED}
    assert set(map(int, np.unique(g))) <= allowed, "input may contain only background plus one blue and one red marker"

    red = _find_unique_cell(g, RED)
    blue = _find_unique_cell(g, BLUE)

    same_row = red[0] == blue[0]
    same_col = red[1] == blue[1]
    assert same_row != same_col, "markers must share exactly one axis"

    separation = abs(red[1] - blue[1]) if same_row else abs(red[0] - blue[0])
    gap = separation - 1
    assert MIN_GAP <= gap <= MAX_GAP, "marker gap must be between 4 and 8 cells"
    return red, blue


def _distance_fields(
    shape: tuple[int, int],
    red: tuple[int, int],
    blue: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.indices(shape)
    d_red = np.maximum(np.abs(rr - red[0]), np.abs(cc - red[1]))
    d_blue = np.maximum(np.abs(rr - blue[0]), np.abs(cc - blue[1]))
    return d_red, d_blue


def _render_output(shape: tuple[int, int], red: tuple[int, int], blue: tuple[int, int]) -> np.ndarray:
    d_red, d_blue = _distance_fields(shape, red, blue)
    out = np.zeros(shape, dtype=int)

    red_owned = d_red < d_blue
    blue_owned = d_blue < d_red

    out[red_owned & (d_red % 2 == 0)] = ORANGE
    out[red_owned & (d_red % 2 == 1)] = RED
    out[blue_owned & (d_blue % 2 == 0)] = GREEN
    out[blue_owned & (d_blue % 2 == 1)] = BLUE

    out[red] = RED
    out[blue] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the overlap between two concentric square-ring systems by nearest-center Chebyshev distance and color each claimed ring by parity.

    Input structure:
    - A 15x15 black grid contains exactly one red marker pixel and one blue marker pixel.
    - The two markers lie on the same row or the same column.
    - Their gap is 4 to 8 background cells, and no other foreground pixels are present.

    Rule:
    - Treat the red marker as the center of a red/orange concentric square-ring system: odd Chebyshev distance is red, even distance is orange.
    - Treat the blue marker as the center of a blue/green concentric square-ring system: odd Chebyshev distance is blue, even distance is green.
    - For every non-center cell, compare its Chebyshev distance to the two centers.
    - Color the cell from the nearer center's ring system; if the distances are equal, leave the cell black.
    - Keep the red and blue center pixels unchanged.

    Key insight:
    - A cell's final color depends on both which center wins the Chebyshev-distance competition and the parity of that winning distance, while exact ties form a black equidistant seam.
    """
    red, blue = _extract_centers(grid)
    return _render_output((GRID_SIZE, GRID_SIZE), red, blue)


def _build_input(
    orientation: str,
    anchor: int,
    start: int,
    separation: int,
    red_first: bool,
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    if orientation == "horizontal":
        red = (anchor, start) if red_first else (anchor, start + separation)
        blue = (anchor, start + separation) if red_first else (anchor, start)
    else:
        red = (start, anchor) if red_first else (start + separation, anchor)
        blue = (start + separation, anchor) if red_first else (start, anchor)
    grid[red] = RED
    grid[blue] = BLUE
    return grid


def _choose_anchor_coords(kind: str) -> list[int]:
    if kind == "edge":
        return [0, 1, 2, 12, 13, 14]
    if kind == "center":
        return [4, 5, 6, 7, 8, 9, 10]
    return list(range(GRID_SIZE))


def _case_signature(grid: np.ndarray) -> tuple[int, ...]:
    red, blue = _extract_centers(grid)
    d_red, d_blue = _distance_fields(grid.shape, red, blue)
    separation = abs(red[1] - blue[1]) if red[0] == blue[0] else abs(red[0] - blue[0])
    orientation = 0 if red[0] == blue[0] else 1
    return (
        orientation,
        separation % 2,
        min(red[0], blue[0]),
        min(red[1], blue[1]),
        int(np.sum(d_red == d_blue)),
        int(np.sum(np.minimum(d_red, d_blue) == 1)),
    )


def _validate_generated_case(grid: np.ndarray) -> bool:
    red, blue = _extract_centers(grid)
    d_red, d_blue = _distance_fields(grid.shape, red, blue)
    out = _render_output(grid.shape, red, blue)

    red_owned = d_red < d_blue
    blue_owned = d_blue < d_red
    eq = d_red == d_blue

    if not np.any(red_owned) or not np.any(blue_owned) or not np.any(eq):
        return False

    center_mask = np.zeros(grid.shape, dtype=bool)
    center_mask[red] = True
    center_mask[blue] = True

    required_masks = (
        red_owned & ~center_mask & (d_red % 2 == 0),
        red_owned & ~center_mask & (d_red % 2 == 1),
        blue_owned & ~center_mask & (d_blue % 2 == 0),
        blue_owned & ~center_mask & (d_blue % 2 == 1),
    )
    if not all(np.any(mask) for mask in required_masks):
        return False

    if len(set(map(int, np.unique(out)))) < 5:
        return False

    # Avoid overly symmetric placements that make the equidistant seam collapse into an easy midpoint shortcut.
    if red[0] == blue[0]:
        if red[0] in {GRID_SIZE // 2} and {red[1], blue[1]} == {3, 11}:
            return False
    else:
        if red[1] in {GRID_SIZE // 2} and {red[0], blue[0]} == {3, 11}:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    gap = random.randint(MIN_GAP, MAX_GAP)
    separation = gap + 1
    placement_kind = random.choices(
        ("edge", "center", "random"),
        weights=(3, 3, 4),
        k=1,
    )[0]

    anchor = random.choice(_choose_anchor_coords(placement_kind))
    start_choices = list(range(GRID_SIZE - separation))
    if placement_kind == "edge":
        start_choices = [s for s in start_choices if s <= 2 or s >= GRID_SIZE - separation - 2]
    elif placement_kind == "center":
        start_choices = [s for s in start_choices if 2 <= s <= GRID_SIZE - separation - 2]
    if not start_choices:
        return None

    start = random.choice(start_choices)
    grid = _build_input(
        orientation=orientation,
        anchor=anchor,
        start=start,
        separation=separation,
        red_first=bool(random.getrandbits(1)),
    )
    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seen: set[tuple[int, ...]] = set()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        sig = _case_signature(grid)
        if sig in seen and random.random() < 0.7:
            continue
        seen.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid dual-center ring input")
