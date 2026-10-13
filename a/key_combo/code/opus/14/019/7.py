from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
ALL_COLORS = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _reflect_color_coords(coords: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    reflected: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        mirrored = (
            (rr, cc),
            (2 * center_r - rr, cc),
            (rr, 2 * center_c - cc),
            (2 * center_r - rr, 2 * center_c - cc),
        )
        reflected.update(mirrored)
    out = np.array(sorted(reflected), dtype=int)
    assert out.ndim == 2 and out.shape[1] == 2
    return out


def _square_bounds_from_reflection(coords: np.ndarray, center_r: int, center_c: int) -> tuple[int, int, int, int]:
    reflected = _reflect_color_coords(coords, center_r, center_c)
    r0, c0, r1, c1 = _bbox_from_coords(reflected)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h % 2 == 1
    assert (r0 + r1) // 2 == center_r and (c0 + c1) // 2 == center_c
    return r0, c0, r1, c1


def _visible_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BLACK)


def _layer_specs(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    assert h == w and h % 2 == 1, "input must be an odd square"
    center_r = h // 2
    center_c = w // 2

    specs: list[tuple[int, int]] = []
    for color in _visible_colors(grid):
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _square_bounds_from_reflection(coords, center_r, center_c)
        side = r1 - r0 + 1
        specs.append((side, color))

    specs.sort(reverse=True)
    assert 3 <= len(specs) <= 5
    assert len({side for side, _ in specs}) == len(specs)
    return specs


def _compose_completed_squares(specs: Iterable[tuple[int, int]], grid_side: int) -> np.ndarray:
    center = grid_side // 2
    out = np.full((grid_side, grid_side), BLACK, dtype=int)
    for side, color in sorted(specs, reverse=True):
        radius = side // 2
        r0 = center - radius
        r1 = center + radius
        c0 = center - radius
        c1 = center + radius
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _stack_blocks_from_inner_to_outer(specs: Iterable[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(specs)
    out = np.full((2 * len(ordered), 2), BLACK, dtype=int)
    for i, (_side, color) in enumerate(ordered):
        out[2 * i : 2 * i + 2, :] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete symmetric corner clues into centered nested squares, then unfold the layers into a vertical strip of color blocks.

    Input structure:
    - The input is an odd square grid with black background.
    - Each non-black color appears as one or more corner-L fragments positioned on the corners of one hidden centered square layer.
    - Different colors correspond to different nested layer sizes; at least one corner fragment is visible for every layer color.

    Rule:
    - For each color, reflect every visible fragment across the horizontal and vertical axes through the grid center to recover the full four-corner clue set.
    - Use the completed clue set to determine that color's centered square layer, then fill the entire implied square; smaller squares are inner layers and would cover larger ones where they overlap.
    - Order the reconstructed layers from inner to outer, and replace each layer by a 2x2 monochrome block of that layer color.
    - Stack those 2x2 blocks vertically so the innermost layer is on top and the outermost layer is on the bottom.

    Key insight:
    - The visible pixels are only partial corner witnesses; only after four-way reflection around the true grid center do the hidden layer sizes become unambiguous, which then exposes the nested-square order needed for the inverse unfold step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    specs = _layer_specs(g)
    _compose_completed_squares(specs, g.shape[0])  # semantic reconstruction guard
    return _stack_blocks_from_inner_to_outer(specs)


def _layer_bounds(center: int, radius: int) -> tuple[int, int, int, int]:
    return center - radius, center - radius, center + radius, center + radius


def _corner_cells(bounds: tuple[int, int, int, int], corner: str, arm: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    cells: set[tuple[int, int]] = set()
    if corner == "tl":
        for c in range(c0, c0 + arm):
            cells.add((r0, c))
        for r in range(r0, r0 + arm):
            cells.add((r, c0))
    elif corner == "tr":
        for c in range(c1 - arm + 1, c1 + 1):
            cells.add((r0, c))
        for r in range(r0, r0 + arm):
            cells.add((r, c1))
    elif corner == "bl":
        for c in range(c0, c0 + arm):
            cells.add((r1, c))
        for r in range(r1 - arm + 1, r1 + 1):
            cells.add((r, c0))
    else:
        assert corner == "br"
        for c in range(c1 - arm + 1, c1 + 1):
            cells.add((r1, c))
        for r in range(r1 - arm + 1, r1 + 1):
            cells.add((r, c1))
    return cells


def _random_corner_subset(rng: np.random.Generator) -> tuple[str, ...]:
    patterns = (
        ("tl",),
        ("tr",),
        ("bl",),
        ("br",),
        ("tl", "tr"),
        ("tl", "bl"),
        ("tr", "br"),
        ("bl", "br"),
        ("tl", "br"),
        ("tr", "bl"),
        ("tl", "tr", "bl"),
        ("tl", "tr", "br"),
        ("tl", "bl", "br"),
        ("tr", "bl", "br"),
        ("tl", "tr", "bl", "br"),
    )
    return patterns[int(rng.integers(len(patterns)))]


def _choose_arms(radii: list[int], rng: np.random.Generator) -> list[int]:
    if len(radii) == 3:
        max_arm = min(4, min(radii) + 1)
        base = int(rng.integers(2, max_arm + 1))
        arms = [base, base]
        top = int(rng.integers(2, min(4, max(radii) + 1) + 1))
        arms.append(top)
        rng.shuffle(arms)
        return [min(arm, radius + 1) for arm, radius in zip(arms, radii)]

    pool = [2, 3, 4]
    repeated = int(rng.choice(pool))
    arms = [repeated, repeated]
    while len(arms) < len(radii):
        arms.append(int(rng.choice(pool)))
    rng.shuffle(arms)
    return [min(arm, radius + 1) for arm, radius in zip(arms, radii)]


def _build_expected_output(layer_specs: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(((spec["side"], spec["color"]) for spec in layer_specs))
    out = np.full((2 * len(ordered), 2), BLACK, dtype=int)
    for i, (_side, color) in enumerate(ordered):
        out[2 * i : 2 * i + 2, :] = color
    return out


def _semantic_checks(grid: np.ndarray, layer_specs: list[dict[str, int]]) -> bool:
    expected = _build_expected_output(layer_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    sides = [int(spec["side"]) for spec in layer_specs]
    arms = [int(spec["arm"]) for spec in layer_specs]
    corner_counts = [len(spec["corners"]) for spec in layer_specs]

    if len(set(sides)) != len(sides):
        return False
    if len(set(arms)) == len(arms):
        return False
    if min(corner_counts) > 2:
        return False
    if max(corner_counts) < 3:
        return False
    if len({corner for spec in layer_specs for corner in spec["corners"]}) < 3:
        return False

    recovered_specs = _layer_specs(grid)
    expected_specs = sorted((int(spec["side"]), int(spec["color"])) for spec in layer_specs)
    if sorted(recovered_specs) != expected_specs:
        return False

    reconstructed = _compose_completed_squares(expected_specs, grid.shape[0])
    center = grid.shape[0] // 2
    for spec in layer_specs:
        color = int(spec["color"])
        radius = int(spec["radius"])
        r0, c0, r1, c1 = _layer_bounds(center, radius)
        corner_cells = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
        if any(int(reconstructed[r, c]) != color for r, c in corner_cells):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_side = int(rng.choice((9, 11, 13)))
    max_radius = grid_side // 2 - int(rng.integers(0, 2))
    layer_count = int(rng.integers(3, min(5, max_radius) + 1))
    if max_radius < layer_count:
        return None

    radii = list(range(max_radius, max_radius - layer_count, -1))
    sides = [2 * radius + 1 for radius in radii]
    colors = rng.choice(ALL_COLORS, size=layer_count, replace=False).tolist()
    arms = _choose_arms(radii, rng)

    grid = np.full((grid_side, grid_side), BLACK, dtype=int)
    center = grid_side // 2
    layer_specs: list[dict[str, int | tuple[str, ...]]] = []

    has_small_subset = False
    has_large_subset = False

    for radius, side, color, arm in zip(radii, sides, colors, arms):
        bounds = _layer_bounds(center, radius)
        corners = _random_corner_subset(rng)
        has_small_subset |= len(corners) <= 2
        has_large_subset |= len(corners) >= 3
        for corner in corners:
            for r, c in _corner_cells(bounds, corner, arm):
                if grid[r, c] != BLACK:
                    return None
                grid[r, c] = int(color)
        layer_specs.append(
            {
                "radius": int(radius),
                "side": int(side),
                "color": int(color),
                "arm": int(arm),
                "corners": corners,
            }
        )

    if not has_small_subset or not has_large_subset:
        return None
    if not _semantic_checks(grid, layer_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
