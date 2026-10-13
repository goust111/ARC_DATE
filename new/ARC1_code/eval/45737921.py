import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two foreground colors inside each separated 3x3 object.

    Input structure:
    - The grid has a dominant background color.
    - Every foreground object is a solid 3x3 rectangle connected by 4-connectivity.
    - Each object contains exactly two non-background colors.

    Rule:
    - Preserve the background and object positions.
    - Within each object, replace every pixel of either foreground color with the
      other foreground color from the same object.

    Key insight:
    - The transformation is local to each object: the two colors in a 3x3 motif
      are complements that exchange roles, independent of the colors used by
      other objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    out = g.copy()
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert objects, "expected at least one foreground object"

    for obj in objects:
        coords = obj["coords"]
        box = obj["bbox"]
        assert bbox_h(box) == 3 and bbox_w(box) == 3, "expected 3x3 foreground objects"
        assert int(obj["area"]) == 9, "expected each object to fill its 3x3 bbox"

        values = g[coords[:, 0], coords[:, 1]]
        colors = sorted(set(map(int, values)) - {int(bg)})
        assert len(colors) == 2, "expected exactly two colors per object"

        a, b = colors
        a_coords = coords[values == a]
        b_coords = coords[values == b]
        out[a_coords[:, 0], a_coords[:, 1]] = b
        out[b_coords[:, 0], b_coords[:, 1]] = a

    return out

# --------------- generator (o5x) ---------------

# Canvas, object count, and retry bounds.
_GEN_BACKGROUND_45737921 = 0
_GEN_ROW_RANGE_45737921 = range(7, 14)
_GEN_COL_RANGE_45737921 = range(7, 13)
_GEN_OBJECT_COUNT_RANGE_45737921 = (1, 4)
_GEN_MAX_ATTEMPTS_45737921 = 300

# 3x3 object geometry and top-left spacing for L-inf distance 2.
_GEN_PATCH_SIZE_45737921 = 3
_GEN_PATCH_AREA_45737921 = _GEN_PATCH_SIZE_45737921 ** 2
_GEN_BORDER_PADDING_45737921 = 1
_GEN_PLACEMENT_STRIDE_45737921 = 4

# Color and two-color split sampling.
_GEN_COLORS_45737921 = tuple(range(1, 10))
_GEN_MINORITY_COUNT_RANGE_45737921 = (2, 4)
_GEN_FIRST_COLOR_MINORITY_PROB_45737921 = 0.5


def _gen_capacity_45737921(rows: int, cols: int) -> int:
    row_slots = (int(rows) - _GEN_BORDER_PADDING_45737921) // _GEN_PLACEMENT_STRIDE_45737921
    col_slots = (int(cols) - _GEN_BORDER_PADDING_45737921) // _GEN_PLACEMENT_STRIDE_45737921
    return int(row_slots * col_slots)


def _gen_choose_dimensions_45737921(n_objects: int) -> tuple[int, int]:
    dims = [
        (rows, cols)
        for rows in _GEN_ROW_RANGE_45737921
        for cols in _GEN_COL_RANGE_45737921
        if _gen_capacity_45737921(rows, cols) >= int(n_objects)
    ]
    return random.choice(dims)


def _gen_locations_45737921(rows: int, cols: int, n_objects: int) -> list[tuple[int, int]] | None:
    row_range = range(_GEN_BORDER_PADDING_45737921, rows - _GEN_BORDER_PADDING_45737921 - _GEN_PATCH_SIZE_45737921 + 1)
    col_range = range(_GEN_BORDER_PADDING_45737921, cols - _GEN_BORDER_PADDING_45737921 - _GEN_PATCH_SIZE_45737921 + 1)
    candidates = [(r, c) for r in row_range for c in col_range]
    random.shuffle(candidates)

    def separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ar, ac = a
        br, bc = b
        return abs(ar - br) >= _GEN_PLACEMENT_STRIDE_45737921 or abs(ac - bc) >= _GEN_PLACEMENT_STRIDE_45737921

    def search(chosen: list[tuple[int, int]], available: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == n_objects:
            return chosen
        if len(chosen) + len(available) < n_objects:
            return None

        order = available[:]
        random.shuffle(order)
        for loc in order:
            next_available = [other for other in available if other != loc and separated(other, loc)]
            if len(chosen) + 1 + len(next_available) < n_objects:
                continue
            found = search(chosen + [loc], next_available)
            if found is not None:
                return found
        return None

    return search([], candidates)


def _gen_patch_pair_45737921() -> tuple[np.ndarray, np.ndarray]:
    a, b = random.sample(_GEN_COLORS_45737921, 2)
    minority_count = random.randint(*_GEN_MINORITY_COUNT_RANGE_45737921)
    minority, majority = (a, b) if random.random() < _GEN_FIRST_COLOR_MINORITY_PROB_45737921 else (b, a)

    flat = [majority] * _GEN_PATCH_AREA_45737921
    for idx in random.sample(range(_GEN_PATCH_AREA_45737921), minority_count):
        flat[idx] = minority

    patch = np.array(flat, dtype=int).reshape(_GEN_PATCH_SIZE_45737921, _GEN_PATCH_SIZE_45737921)
    swapped = np.where(patch == a, b, a).astype(int)
    return patch, swapped


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*_GEN_OBJECT_COUNT_RANGE_45737921)
    rows, cols = _gen_choose_dimensions_45737921(n_objects)
    locations = _gen_locations_45737921(rows, cols, n_objects)
    if locations is None:
        return None

    inp = np.full((rows, cols), _GEN_BACKGROUND_45737921, dtype=int)
    expected = np.full((rows, cols), _GEN_BACKGROUND_45737921, dtype=int)

    for r, c in locations:
        patch, swapped = _gen_patch_pair_45737921()
        paste(inp, patch, r, c)
        paste(expected, swapped, r, c)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS_45737921):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 45737921 input")
