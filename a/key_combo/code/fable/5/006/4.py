from __future__ import annotations

import random

import numpy as np


# Self-contained equivalents of the relevant arc_utils conventions are used here:
# coordinates are (row, column), bboxes are inclusive, and components are 4-connected.
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_SOLID_LOCAL = {
    "top_left": (0, 0),
    "top_right": (0, 1),
    "bottom_left": (1, 0),
    "bottom_right": (1, 1),
}


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    values, counts = np.unique(grid, return_counts=True)
    maximum = int(counts.max())
    winners = values[counts == maximum]
    assert winners.size == 1, "the background color must be uniquely dominant"
    return int(winners[0])


def _components_by_color(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """Find monochromatic 4-connected foreground components as (r,c) arrays."""
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or int(grid[start_r, start_c]) == int(bg):
                continue
            color = int(grid[start_r, start_c])
            seen[start_r, start_c] = True
            stack = [(start_r, start_c)]
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _decode_component(
    grid: np.ndarray, bg: int, coords: np.ndarray
) -> tuple[int, tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    """Return color, bbox, hole-window origin, and its solid-cell corner."""
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0
    color = int(grid[tuple(coords[0])])
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = sub.shape
    assert height >= 4 and width >= 4, "objects must have an interior"
    assert np.all((sub == color) | (sub == int(bg))), "object bboxes may not contain other colors"

    holes = np.argwhere(sub == int(bg))
    assert holes.shape[0] == 3, "a solid object must contain exactly one three-cell bite"
    hr0, hc0 = holes.min(axis=0)
    hr1, hc1 = holes.max(axis=0)
    hr0, hc0, hr1, hc1 = map(int, (hr0, hc0, hr1, hc1))
    assert (hr1 - hr0, hc1 - hc0) == (1, 1), "the bite must lie in one 2x2 window"
    assert 1 <= hr0 and hr1 < height - 1 and 1 <= hc0 and hc1 < width - 1, "the bite must be internal"

    window = sub[hr0 : hr0 + 2, hc0 : hc0 + 2]
    assert int(np.sum(window == int(bg))) == 3
    solid = np.argwhere(window == color)
    assert solid.shape == (1, 2), "the 2x2 instruction must have one solid object cell"
    solid_local = tuple(map(int, solid[0]))

    candidates = []
    for r in range(height - 1):
        for c in range(width - 1):
            block = sub[r : r + 2, c : c + 2]
            if int(np.sum(block == int(bg))) == 3 and int(np.sum(block == color)) == 1:
                candidates.append((r, c))
    assert candidates == [(hr0, hc0)], "the instruction window must be unique"
    assert coords.shape[0] == height * width - 3, "the rest of the bbox must be solid"
    return color, (r0, c0, r1, c1), (hr0, hc0), solid_local


def _corner_mask(shape: tuple[int, int], solid_local: tuple[int, int]) -> np.ndarray:
    """Mask the outer two rows and columns at the instructed bbox corner."""
    height, width = shape
    row_side, col_side = solid_local
    mask = np.zeros(shape, dtype=bool)
    if row_side == 0:
        mask[:2, :] = True
    else:
        mask[height - 2 :, :] = True
    if col_side == 0:
        mask[:, :2] = True
    else:
        mask[:, width - 2 :] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each object's internal L-bite instruction, erase its indicated bbox corner bands, and repair the bite.

    Input structure:
    - The grid has one dominant background and separated monochromatic 4-connected solid objects.
    - Each object has exactly one internal 2x2 window containing three background cells and one object-colored cell.

    Rule:
    - The surviving cell's corner inside the 2x2 window selects the same-named corner of that object's own bbox.
    - Erase object pixels in the outermost two rows and outermost two columns at that corner, then recolor the three bite cells with the object's color.
    - Preserve every other pixel and keep the output at the input size.

    Key insight:
    - The solid cell, not the three missing cells or the window's position, encodes direction; all erasure coordinates are local to each object's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _background(g)
    components = _components_by_color(g, bg)
    assert components, "expected at least one foreground object"

    decoded = [_decode_component(g, bg, coords) for coords in components]
    out = g.copy()
    for color, (r0, c0, r1, c1), (hole_r, hole_c), solid_local in decoded:
        source = g[r0 : r1 + 1, c0 : c1 + 1]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        erase = _corner_mask(source.shape, solid_local) & (source == color)
        target[erase] = int(bg)

        hole = source[hole_r : hole_r + 2, hole_c : hole_c + 2] == int(bg)
        repaired = target[hole_r : hole_r + 2, hole_c : hole_c + 2]
        repaired[hole] = color
    return out


# ------------------------------ generator ------------------------------

_COLOR_COUNT = 10
_OBJECT_COUNT_RANGE = (4, 6)
_GAP = 2
_GENERATION_ATTEMPTS = 100


def _new_spec(corner: str, height: int, width: int, hole_r: int, hole_c: int) -> dict[str, object]:
    return {
        "corner": corner,
        "height": int(height),
        "width": int(width),
        "hole": (int(hole_r), int(hole_c)),
        "solid": _SOLID_LOCAL[corner],
    }


def _sample_specs(count: int, bg: int) -> list[dict[str, object]]:
    """Cover all directions and embed a same-position/different-direction counterexample."""
    max_side = 9 if count == 4 else 8
    pair_corners = random.sample(list(_CORNERS), 2)
    other_corners = [corner for corner in _CORNERS if corner not in pair_corners]
    corners = pair_corners + other_corners
    corners += [random.choice(_CORNERS) for _ in range(count - 4)]

    pair_h = random.randint(7, max_side)
    pair_w = random.randint(7, max_side)
    pair_hole = (random.randint(2, pair_h - 4), random.randint(2, pair_w - 4))
    specs = [
        _new_spec(pair_corners[0], pair_h, pair_w, *pair_hole),
        _new_spec(pair_corners[1], pair_h, pair_w, *pair_hole),
    ]
    specs[0]["paired"] = True
    specs[1]["paired"] = True

    for corner in corners[2:]:
        height = random.randint(7, max_side)
        width = random.randint(7, max_side)
        hole_r = random.randint(2, height - 4)
        hole_c = random.randint(2, width - 4)
        spec = _new_spec(corner, height, width, hole_r, hole_c)
        spec["paired"] = False
        specs.append(spec)

    colors = random.sample([color for color in range(_COLOR_COUNT) if color != int(bg)], count)
    random.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
    return specs


def _place_in_bands(specs: list[dict[str, object]]) -> tuple[int, int]:
    """Assign objects to a jittered 2xN or Nx2 layout with two-cell gaps."""
    count = len(specs)
    if count == 4:
        n_rows, n_cols = 2, 2
    elif random.random() < 0.5:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 2

    slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    chosen_slots = random.sample(slots, count)
    shuffled_specs = random.sample(specs, count)
    assignment = list(zip(chosen_slots, shuffled_specs))

    row_heights = [
        max(int(spec["height"]) for (r, _c), spec in assignment if r == row)
        for row in range(n_rows)
    ]
    col_widths = [
        max(int(spec["width"]) for (_r, c), spec in assignment if c == col)
        for col in range(n_cols)
    ]
    min_height = sum(row_heights) + _GAP * (n_rows - 1) + 2
    min_width = sum(col_widths) + _GAP * (n_cols - 1) + 2
    assert min_height <= 30 and min_width <= 30
    grid_h = random.randint(min_height, 30)
    grid_w = random.randint(min_width, 30)

    vertical_margin = grid_h - sum(row_heights) - _GAP * (n_rows - 1)
    horizontal_margin = grid_w - sum(col_widths) - _GAP * (n_cols - 1)
    top = random.randint(1, vertical_margin - 1)
    left = random.randint(1, horizontal_margin - 1)

    row_starts: list[int] = []
    cursor = top
    for band_h in row_heights:
        row_starts.append(cursor)
        cursor += band_h + _GAP
    col_starts: list[int] = []
    cursor = left
    for band_w in col_widths:
        col_starts.append(cursor)
        cursor += band_w + _GAP

    for (row, col), spec in assignment:
        r0 = row_starts[row] + random.randint(0, row_heights[row] - int(spec["height"]))
        c0 = col_starts[col] + random.randint(0, col_widths[col] - int(spec["width"]))
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
    return grid_h, grid_w


def _render_input(
    shape: tuple[int, int], bg: int, specs: list[dict[str, object]]
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        height, width = int(spec["height"]), int(spec["width"])
        color = int(spec["color"])
        grid[r0 : r0 + height, c0 : c0 + width] = color
        hole_r, hole_c = spec["hole"]
        solid_r, solid_c = spec["solid"]
        for dr in range(2):
            for dc in range(2):
                if (dr, dc) != (solid_r, solid_c):
                    grid[r0 + hole_r + dr, c0 + hole_c + dc] = int(bg)
    return grid


def _semantic_oracle(
    grid: np.ndarray, bg: int, specs: list[dict[str, object]]
) -> np.ndarray:
    """Build the answer directly from sampled semantic roles, independently of decoding."""
    out = grid.copy()
    for spec in specs:
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        height, width = int(spec["height"]), int(spec["width"])
        color = int(spec["color"])
        solid = tuple(spec["solid"])
        source = grid[r0 : r0 + height, c0 : c0 + width]
        target = out[r0 : r0 + height, c0 : c0 + width]
        target[_corner_mask((height, width), solid) & (source == color)] = int(bg)
        hole_r, hole_c = spec["hole"]
        target[hole_r : hole_r + 2, hole_c : hole_c + 2] = color
    return out


def _validate_generated(grid: np.ndarray, bg: int, specs: list[dict[str, object]]) -> bool:
    """Check puzzle semantics, uniqueness, counterexamples, and both necessary steps."""
    if grid.ndim != 2 or max(grid.shape) > 30 or _background(grid) != int(bg):
        return False
    if {str(spec["corner"]) for spec in specs} != set(_CORNERS):
        return False

    paired = [spec for spec in specs if bool(spec.get("paired", False))]
    if len(paired) != 2:
        return False
    if not (
        paired[0]["height"] == paired[1]["height"]
        and paired[0]["width"] == paired[1]["width"]
        and paired[0]["hole"] == paired[1]["hole"]
        and paired[0]["corner"] != paired[1]["corner"]
    ):
        return False

    components = _components_by_color(grid, bg)
    if len(components) != len(specs):
        return False
    decoded_by_key: dict[tuple[int, int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    try:
        for coords in components:
            color, (r0, c0, _r1, _c1), hole, solid = _decode_component(grid, bg, coords)
            decoded_by_key[(color, r0, c0)] = (hole, solid)
    except AssertionError:
        return False

    for spec in specs:
        key = (int(spec["color"]), int(spec["r0"]), int(spec["c0"]))
        if decoded_by_key.get(key) != (tuple(spec["hole"]), tuple(spec["solid"])):
            return False

    expected = _semantic_oracle(grid, bg, specs)
    fill_only = grid.copy()
    erase_only = grid.copy()
    for spec in specs:
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        height, width = int(spec["height"]), int(spec["width"])
        color = int(spec["color"])
        hole_r, hole_c = spec["hole"]
        fill_only[r0 + hole_r : r0 + hole_r + 2, c0 + hole_c : c0 + hole_c + 2] = color
        source = grid[r0 : r0 + height, c0 : c0 + width]
        target = erase_only[r0 : r0 + height, c0 : c0 + width]
        target[_corner_mask((height, width), tuple(spec["solid"])) & (source == color)] = int(bg)

    if np.array_equal(expected, grid) or np.array_equal(expected, fill_only) or np.array_equal(expected, erase_only):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with all four instruction directions visible."""
    for _ in range(_GENERATION_ATTEMPTS):
        bg = random.randrange(_COLOR_COUNT)
        count = random.randint(*_OBJECT_COUNT_RANGE)
        specs = _sample_specs(count, bg)
        shape = _place_in_bands(specs)
        grid = _render_input(shape, bg, specs)
        if _validate_generated(grid, bg, specs):
            return grid
    raise RuntimeError("failed to generate a valid background-bite corner-erasure input")
