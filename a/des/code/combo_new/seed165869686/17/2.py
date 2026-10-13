from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import random

import numpy as np


_BG = 0
_GRID_MIN = 14
_GRID_MAX = 30
_MASK_MAX_DIM = 4
_PAIR_COUNT_OPTIONS = (1, 2, 3)
_PAIR_PLACEMENT_TRIES = 4000
_GRID_GENERATION_TRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _mean_fraction(values: np.ndarray) -> Fraction:
    return Fraction(int(values.sum()), int(values.size))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group each color's two fragments and mirror them across their shared hidden axis to complete a symmetric figure.

    Input structure:
    - The background is black and fills the empty cells.
    - Every non-background color appears in exactly two separate 4-connected fragments.
    - For a given color, the two fragments are small scattered pieces whose centroids line up
      either horizontally or vertically.

    Rule:
    - Process each non-background color independently and find its two 4-connected fragments.
    - Compute the perpendicular bisector of the segment joining the two fragment centroids;
      this hidden symmetry axis is guaranteed to be horizontal or vertical.
    - Keep both fragments fixed, then reflect every pixel of both fragments across that axis
      using the same color.
    - The output is the input grid plus all of these mirrored copies, with different colors
      never overlapping.

    Key insight:
    - The spacing between the two same-color fragments encodes the missing symmetry axis, so the
      task is to infer that axis from the fragment centroids and then grow each fragment by reflection.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _background_color(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each color must form exactly two fragments"

        a, b = comps
        a_row = _mean_fraction(a[:, 0])
        b_row = _mean_fraction(b[:, 0])
        a_col = _mean_fraction(a[:, 1])
        b_col = _mean_fraction(b[:, 1])

        if a_row == b_row and a_col != b_col:
            axis2 = a_col + b_col
            assert axis2.denominator == 1, "vertical reflection axis must land on cell centers or cell gaps"
            axis2_int = int(axis2)
            coords = np.vstack((a, b))
            for rr, cc in coords:
                mr = int(rr)
                mc = axis2_int - int(cc)
                assert 0 <= mr < h and 0 <= mc < w
                assert int(out[mr, mc]) in (int(bg), int(color))
                out[mr, mc] = int(color)
        elif a_col == b_col and a_row != b_row:
            axis2 = a_row + b_row
            assert axis2.denominator == 1, "horizontal reflection axis must land on cell centers or cell gaps"
            axis2_int = int(axis2)
            coords = np.vstack((a, b))
            for rr, cc in coords:
                mr = axis2_int - int(rr)
                mc = int(cc)
                assert 0 <= mr < h and 0 <= mc < w
                assert int(out[mr, mc]) in (int(bg), int(color))
                out[mr, mc] = int(color)
        else:
            raise AssertionError("fragment centroids must align horizontally or vertically")

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if not coords:
        return False

    cells = set(coords)
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (int(r + dr), int(c + dc))
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _enumerate_masks() -> list[dict]:
    masks: list[dict] = []
    for mh in range(1, _MASK_MAX_DIM + 1):
        for mw in range(1, _MASK_MAX_DIM + 1):
            cell_count = mh * mw
            for area in range(3, min(8, cell_count) + 1):
                for cells in combinations(range(cell_count), area):
                    mask = np.zeros((mh, mw), dtype=bool)
                    for cell in cells:
                        mask[cell // mw, cell % mw] = True
                    if not _is_connected(mask):
                        continue
                    rows, cols = np.where(mask)
                    if int(rows.min()) != 0 or int(rows.max()) != mh - 1:
                        continue
                    if int(cols.min()) != 0 or int(cols.max()) != mw - 1:
                        continue
                    masks.append(
                        {
                            "coords": tuple(tuple(map(int, coord)) for coord in np.argwhere(mask)),
                            "h": int(mh),
                            "w": int(mw),
                            "area": int(area),
                            "row_centroid": Fraction(int(rows.sum()), int(area)),
                            "col_centroid": Fraction(int(cols.sum()), int(area)),
                        }
                    )
    return masks


_MASKS = _enumerate_masks()
_VERTICAL_PAIRS = [
    (left, right)
    for left in _MASKS
    for right in _MASKS
    if left["row_centroid"] == right["row_centroid"]
    and (left["col_centroid"] + right["col_centroid"]).denominator == 1
]
_HORIZONTAL_PAIRS = [
    (top, bottom)
    for top in _MASKS
    for bottom in _MASKS
    if top["col_centroid"] == bottom["col_centroid"]
    and (top["row_centroid"] + bottom["row_centroid"]).denominator == 1
]

assert _VERTICAL_PAIRS and _HORIZONTAL_PAIRS


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _reflect_vertical(cells: set[tuple[int, int]], axis2: int) -> set[tuple[int, int]]:
    return {(int(r), int(axis2 - c)) for r, c in cells}


def _reflect_horizontal(cells: set[tuple[int, int]], axis2: int) -> set[tuple[int, int]]:
    return {(int(axis2 - r), int(c)) for r, c in cells}


def _place_pair(
    h: int,
    w: int,
    reserved_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    orientation: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pair_pool = _VERTICAL_PAIRS if orientation == "vertical" else _HORIZONTAL_PAIRS

    for _ in range(_PAIR_PLACEMENT_TRIES):
        first, second = random.choice(pair_pool)

        if orientation == "vertical":
            top = random.randint(0, h - max(int(first["h"]), int(second["h"])))
            centroid_sum = int(first["col_centroid"] + second["col_centroid"])

            left_min = max(0, int(second["w"]) - 1 - centroid_sum)
            left_max = min(w - int(first["w"]), w - 1 - centroid_sum)
            if left_min > left_max:
                continue
            left_col = random.randint(left_min, left_max)

            right_min = max(
                0,
                int(first["w"]) - 1 - centroid_sum,
                left_col + int(first["w"]) + 1,
                left_col + centroid_sum + 1,
                left_col + 2 * (int(first["w"]) - 1) - centroid_sum + 1,
            )
            right_max = min(w - int(second["w"]), w - 1 - centroid_sum)
            if right_min > right_max:
                continue
            right_col = random.randint(right_min, right_max)

            first_cells = {(top + r, left_col + c) for r, c in first["coords"]}
            second_cells = {(top + r, right_col + c) for r, c in second["coords"]}
            axis2 = left_col + right_col + centroid_sum
            reflected = _reflect_vertical(first_cells, axis2) | _reflect_vertical(second_cells, axis2)
        else:
            left = random.randint(0, w - max(int(first["w"]), int(second["w"])))
            centroid_sum = int(first["row_centroid"] + second["row_centroid"])

            top_min = max(0, int(second["h"]) - 1 - centroid_sum)
            top_max = min(h - int(first["h"]), h - 1 - centroid_sum)
            if top_min > top_max:
                continue
            top_row = random.randint(top_min, top_max)

            bottom_min = max(
                0,
                int(first["h"]) - 1 - centroid_sum,
                top_row + int(first["h"]) + 1,
                top_row + centroid_sum + 1,
                top_row + 2 * (int(first["h"]) - 1) - centroid_sum + 1,
            )
            bottom_max = min(h - int(second["h"]), h - 1 - centroid_sum)
            if bottom_min > bottom_max:
                continue
            bottom_row = random.randint(bottom_min, bottom_max)

            first_cells = {(top_row + r, left + c) for r, c in first["coords"]}
            second_cells = {(bottom_row + r, left + c) for r, c in second["coords"]}
            axis2 = top_row + bottom_row + centroid_sum
            reflected = _reflect_horizontal(first_cells, axis2) | _reflect_horizontal(second_cells, axis2)

        originals = first_cells | second_cells
        union = originals | reflected
        if len(union) <= len(originals):
            continue
        if originals & reserved_input:
            continue
        if union & occupied_output:
            continue
        return first_cells, second_cells, union

    return None


def _candidate_grid() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    pair_count = random.choice(_PAIR_COUNT_OPTIONS)
    colors = random.sample(list(range(1, 10)), pair_count)

    inp = np.full((h, w), _BG, dtype=int)
    expected = np.full((h, w), _BG, dtype=int)
    reserved_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()

    for color in colors:
        placed = None
        for _ in range(40):
            orientation = "vertical" if random.choice([False, True]) else "horizontal"
            placed = _place_pair(h, w, reserved_input, occupied_output, orientation)
            if placed is not None:
                break
        if placed is None:
            return None

        first_cells, second_cells, union = placed
        for r, c in first_cells | second_cells:
            inp[r, c] = int(color)
        for r, c in union:
            expected[r, c] = int(color)

        reserved_input |= _dilate(first_cells | second_cells, h, w)
        occupied_output |= union

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GRID_GENERATION_TRIES):
        candidate = _candidate_grid()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid mirror-completion grid")
