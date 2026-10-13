from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _on_border(r: int, c: int, shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[set[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(comp)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each small colored object toward its same-colored border marker.

    Input structure:
    - Black background with several small 4-connected objects of different colors.
    - Each object color appears once more as a singleton marker on the grid edge.
    - The marker shares either a row or a column with its object.

    Rule:
    - Pair each interior object with the edge marker of the same color.
    - If the marker shares a row with the object, fill the background cells on
      that row between the marker and the nearest object edge.
    - If the marker shares a column with the object, fill the background cells
      on that column between the marker and the nearest object edge.
    - Keep the original object pixels and the marker pixel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = g.shape

    bg = _background_color(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    for color in colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        border_pts = [(r, c) for r, c in coords if _on_border(r, c, g.shape)]
        interior_pts = [(r, c) for r, c in coords if not _on_border(r, c, g.shape)]

        assert len(border_pts) == 1, f"color {color} must have exactly one border marker"
        assert interior_pts, f"color {color} must have an interior object"

        mask = np.zeros_like(g, dtype=bool)
        for r, c in interior_pts:
            mask[r, c] = True
        comps = _components(mask)
        assert len(comps) == 1, f"color {color} must form a single 4-connected interior object"

        marker_r, marker_c = border_pts[0]
        object_rows = {r for r, _ in interior_pts}
        object_cols = {c for _, c in interior_pts}
        row_aligned = marker_r in object_rows
        col_aligned = marker_c in object_cols
        assert row_aligned != col_aligned, f"color {color} must align in exactly one axis"

        if row_aligned:
            assert marker_c in (0, cols - 1), "horizontal markers must sit on the left or right edge"
            row_cols = [c for r, c in interior_pts if r == marker_r]
            if marker_c < min(row_cols):
                start = marker_c + 1
                stop = min(row_cols) - 1
            else:
                start = max(row_cols) + 1
                stop = marker_c - 1
            if start <= stop:
                segment = out[marker_r, start : stop + 1]
                segment[segment == bg] = color
        else:
            assert marker_r in (0, rows - 1), "vertical markers must sit on the top or bottom edge"
            col_rows = [r for r, c in interior_pts if c == marker_c]
            if marker_r < min(col_rows):
                start = marker_r + 1
                stop = min(col_rows) - 1
            else:
                start = max(col_rows) + 1
                stop = marker_r - 1
            if start <= stop:
                segment = out[start : stop + 1, marker_c]
                segment[segment == bg] = color

    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_MIN = 10
GRID_MAX = 18
MIN_OBJECTS = 2
MAX_OBJECTS = 5
MIN_PATH_LEN = 2
SCENE_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200

BASE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in cells)


def _mirror_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, -c) for r, c in cells)


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    current = tuple(cells)
    for _ in range(4):
        for candidate in (current, _mirror_shape(current)):
            normalized = _normalize_shape(candidate)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
        current = _rotate_shape(current)
    return variants


SHAPE_VARIANTS = tuple(_shape_variants(shape) for shape in BASE_SHAPES)


def _random_shape() -> tuple[tuple[int, int], ...]:
    return random.choice(random.choice(SHAPE_VARIANTS))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _cells_touch(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], diagonals: bool) -> bool:
    directions = DIRS8 if diagonals else DIRS4
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for dr, dc in directions:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _horizontal_fill_cells(
    object_cells: set[tuple[int, int]], marker: tuple[int, int], cols: int
) -> set[tuple[int, int]]:
    row = marker[0]
    row_cols = [c for r, c in object_cells if r == row]
    assert row_cols, "horizontal placement must align with at least one object row"
    if marker[1] == 0:
        return {(row, c) for c in range(1, min(row_cols))}
    return {(row, c) for c in range(max(row_cols) + 1, cols - 1)}


def _vertical_fill_cells(
    object_cells: set[tuple[int, int]], marker: tuple[int, int], rows: int
) -> set[tuple[int, int]]:
    col = marker[1]
    col_rows = [r for r, c in object_cells if c == col]
    assert col_rows, "vertical placement must align with at least one object column"
    if marker[0] == 0:
        return {(r, col) for r in range(1, min(col_rows))}
    return {(r, col) for r in range(max(col_rows) + 1, rows - 1)}


def _placement_clear(
    object_cells: set[tuple[int, int]],
    marker: tuple[int, int],
    fill_cells: set[tuple[int, int]],
    used_fg: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
) -> bool:
    new_fg = set(object_cells)
    new_fg.add(marker)

    occupied_output = used_fg | reserved_fill
    if new_fg & occupied_output:
        return False
    if fill_cells & occupied_output:
        return False
    if fill_cells & new_fg:
        return False

    if _cells_touch(new_fg, used_fg, diagonals=True):
        return False
    if _cells_touch(new_fg, reserved_fill, diagonals=True):
        return False
    if _cells_touch(fill_cells, used_fg, diagonals=False):
        return False

    return True


def _sample_placement(
    rows: int,
    cols: int,
    orientation: str,
    used_fg: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _random_shape()
        shape_h, shape_w = _shape_dims(shape)
        if shape_h > rows - 2 or shape_w > cols - 2:
            continue

        if orientation == "H":
            local_row = random.choice(sorted({r for r, _ in shape}))
            side = random.choice(("left", "right"))
            row_cols = [c for r, c in shape if r == local_row]
            row_min = min(row_cols)
            row_max = max(row_cols)
            r_min = 1
            r_max = rows - shape_h - 1
            if side == "left":
                c_min = max(1, MIN_PATH_LEN + 1 - row_min)
                c_max = cols - shape_w - 1
            else:
                c_min = 1
                c_max = min(cols - shape_w - 1, cols - 2 - MIN_PATH_LEN - row_max)
            if r_min > r_max or c_min > c_max:
                continue

            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            object_cells = {(r0 + r, c0 + c) for r, c in shape}
            marker = (r0 + local_row, 0 if side == "left" else cols - 1)
            fill_cells = _horizontal_fill_cells(object_cells, marker, cols)
        else:
            local_col = random.choice(sorted({c for _, c in shape}))
            side = random.choice(("top", "bottom"))
            col_rows = [r for r, c in shape if c == local_col]
            col_min = min(col_rows)
            col_max = max(col_rows)
            c_min = 1
            c_max = cols - shape_w - 1
            if side == "top":
                r_min = max(1, MIN_PATH_LEN + 1 - col_min)
                r_max = rows - shape_h - 1
            else:
                r_min = 1
                r_max = min(rows - shape_h - 1, rows - 2 - MIN_PATH_LEN - col_max)
            if r_min > r_max or c_min > c_max:
                continue

            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            object_cells = {(r0 + r, c0 + c) for r, c in shape}
            marker = (0 if side == "top" else rows - 1, c0 + local_col)
            fill_cells = _vertical_fill_cells(object_cells, marker, rows)

        if len(fill_cells) < MIN_PATH_LEN:
            continue
        if not _placement_clear(object_cells, marker, fill_cells, used_fg, reserved_fill):
            continue

        return {
            "object_cells": object_cells,
            "marker": marker,
            "fill_cells": fill_cells,
            "orientation": orientation,
        }

    return None


def _render_scene(
    rows: int, cols: int, placements: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BG, dtype=int)
    out = inp.copy()

    for placement in placements:
        color = int(placement["color"])
        object_cells = placement["object_cells"]
        marker = placement["marker"]
        fill_cells = placement["fill_cells"]

        for r, c in object_cells:
            inp[r, c] = color
            out[r, c] = color
        mr, mc = marker
        inp[mr, mc] = color
        out[mr, mc] = color
        for r, c in fill_cells:
            out[r, c] = color

    return inp, out


def _orientation_mix(count: int) -> list[str]:
    orientations = ["H", "V"]
    while len(orientations) < count:
        orientations.append(random.choice(("H", "V")))
    random.shuffle(orientations)
    return orientations


def _sample_scene() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    max_objects = min(MAX_OBJECTS, max(MIN_OBJECTS, (rows + cols) // 6))
    count = random.randint(MIN_OBJECTS, max_objects)

    colors = random.sample(COLOR_POOL, count)
    orientations = _orientation_mix(count)

    used_fg: set[tuple[int, int]] = set()
    reserved_fill: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for color, orientation in zip(colors, orientations, strict=False):
        placement = _sample_placement(rows, cols, orientation, used_fg, reserved_fill)
        if placement is None:
            return None
        placement["color"] = color
        placements.append(placement)

        used_fg |= set(placement["object_cells"])
        used_fg.add(placement["marker"])
        reserved_fill |= set(placement["fill_cells"])

    inp, expected = _render_scene(rows, cols, placements)
    if np.count_nonzero(inp != BG) >= inp.size // 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        sample = _sample_scene()
        if sample is not None:
            return sample
    raise RuntimeError(f"failed to generate a valid input after {SCENE_ATTEMPTS} attempts")
