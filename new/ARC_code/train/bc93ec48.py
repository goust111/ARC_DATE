from __future__ import annotations

import numpy as np

from arc_utils import bbox, find_components, get_bg, paste
import random


Corner = str

_NEXT_CORNER: dict[Corner, Corner] = {
    "TL": "TR",
    "TR": "BR",
    "BR": "BL",
    "BL": "TL",
}


def _bbox_corner(bbox: tuple[int, int, int, int], height: int, width: int) -> Corner | None:
    r0, c0, r1, c1 = bbox
    corners = []
    if r0 == 0 and c0 == 0:
        corners.append("TL")
    if r0 == 0 and c1 == width - 1:
        corners.append("TR")
    if r1 == height - 1 and c1 == width - 1:
        corners.append("BR")
    if r1 == height - 1 and c0 == 0:
        corners.append("BL")

    assert len(corners) <= 1, "expected each corner object to touch at most one corner"
    return corners[0] if corners else None


def _corner_anchor(corner: Corner, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(f"unknown corner: {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each foreground object anchored in a true grid corner clockwise into the next corner.

    Input structure
    - The dominant color is the background.
    - Foreground objects are monochromatic 4-connected components.
    - Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

    Rule
    - Start from the input unchanged.
    - For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
    - Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

    Key insight
    - Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = get_bg(g)

    by_corner: dict[Corner, dict] = {}
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        corner = _bbox_corner(comp["bbox"], height, width)
        if corner is None:
            continue
        assert corner not in by_corner, "expected at most one source object per corner"
        by_corner[corner] = comp

    assert by_corner, "expected at least one corner-anchored object"

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}
    for src_corner, comp in by_corner.items():
        r0, c0, r1, c1 = comp["bbox"]
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        dst_corner = _NEXT_CORNER[src_corner]
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, obj_h, obj_w)
        color = int(comp["color"])

        for r_raw, c_raw in comp["coords"]:
            r = int(dst_r0 + int(r_raw) - r0)
            c = int(dst_c0 + int(c_raw) - c0)
            assert 0 <= r < height and 0 <= c < width, "corner stamp must remain in bounds"
            assert (r, c) not in planned, "expected non-overlapping corner stamps"
            planned[(r, c)] = color

    for (r, c), color in planned.items():
        out[r, c] = color

    return out

# --------------- generator (att2) ---------------

_GEN_BG = 7
_GEN_MIN_SIZE, _GEN_MAX_SIZE = 8, 16
_GEN_CORNERS = ("TL", "TR", "BR", "BL")
_GEN_NEXT = {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"}
_GEN_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

_GEN_CORNER_COUNT_MIN, _GEN_CORNER_COUNT_MAX = 3, 4
_GEN_CORNER_DIM_MIN, _GEN_CORNER_DIM_MAX = 1, 4
_GEN_CORNER_MIN_PIXELS, _GEN_CORNER_MAX_PIXELS = 2, 12
_GEN_CORNER_MIN_DENSITY = 0.62
_GEN_SMALL_FULL_AREA_MAX, _GEN_SMALL_FULL_PROB = 12, 0.45
_GEN_CORNER_DIMS = tuple(
    (h, w)
    for h in range(_GEN_CORNER_DIM_MIN, _GEN_CORNER_DIM_MAX + 1)
    for w in range(_GEN_CORNER_DIM_MIN, _GEN_CORNER_DIM_MAX + 1)
    if not (h == 1 and w == 1)
)

_GEN_BLOB_PIXELS_MIN, _GEN_BLOB_PIXELS_MAX = 1, 5
_GEN_BLOB_H_MIN, _GEN_BLOB_H_MAX = 1, 5
_GEN_BLOB_W_MIN, _GEN_BLOB_W_MAX = 1, 4
_GEN_DISTRACTOR_BASE_COUNT = 4
_GEN_DISTRACTOR_SIZE_SPAN = 8
_GEN_DISTRACTOR_EXTRA_MAX, _GEN_DISTRACTOR_MIN_GAP = 10, 6

_GEN_RETRY_LIMIT = 300
_GEN_NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _gen_anchor(corner: str, n: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, n - w
    if corner == "BR":
        return n - h, n - w
    if corner == "BL":
        return n - h, 0
    raise AssertionError(corner)


def _gen_source_local_corner(corner: str, h: int, w: int) -> tuple[int, int]:
    r = 0 if corner in ("TL", "TR") else h - 1
    c = 0 if corner in ("TL", "BL") else w - 1
    return r, c


def _gen_step_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _gen_frontier(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for dr, dc in _GEN_NEIGHBORS_4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return frontier


def _gen_grow_mask(mask: np.ndarray, target: int) -> bool:
    while int(mask.sum()) < target:
        frontier = _gen_frontier(mask)
        if not frontier:
            return False
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True
    return True


def _gen_connected_corner_mask(corner: str, h: int, w: int) -> np.ndarray | None:
    sr, sc = _gen_source_local_corner(corner, h, w)
    er, ec = h - 1 - sr, w - 1 - sc
    mask = np.zeros((h, w), dtype=bool)

    if random.random() < 0.5:
        for r in _gen_step_range(sr, er):
            mask[r, sc] = True
        for c in _gen_step_range(sc, ec):
            mask[er, c] = True
    else:
        for c in _gen_step_range(sc, ec):
            mask[sr, c] = True
        for r in _gen_step_range(sr, er):
            mask[r, ec] = True

    area = h * w
    min_pixels = max(
        _GEN_CORNER_MIN_PIXELS,
        int(np.ceil(area * _GEN_CORNER_MIN_DENSITY)),
        int(mask.sum()),
    )
    max_pixels = min(_GEN_CORNER_MAX_PIXELS, area)
    if min_pixels > max_pixels:
        return None

    if area <= _GEN_SMALL_FULL_AREA_MAX and random.random() < _GEN_SMALL_FULL_PROB:
        target = area
    else:
        target = random.randint(min_pixels, max_pixels)

    return mask if _gen_grow_mask(mask, target) else None


def _gen_colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, _GEN_BG, dtype=int)
    patch[mask] = color
    return patch


def _gen_stamp_cells(
    corner: str, mask: np.ndarray, color: int, n: int
) -> dict[tuple[int, int], int]:
    h, w = mask.shape
    r0, c0 = _gen_anchor(_GEN_NEXT[corner], n, h, w)
    cells = {}
    for rr, cc in np.argwhere(mask):
        cells[(r0 + int(rr), c0 + int(cc))] = color
    return cells


def _gen_touches_true_corner(r0: int, c0: int, h: int, w: int, n: int) -> bool:
    top = r0 == 0
    bottom = r0 + h - 1 == n - 1
    left = c0 == 0
    right = c0 + w - 1 == n - 1
    return (top and left) or (top and right) or (bottom and right) or (bottom and left)


def _gen_blob_patch(color: int) -> np.ndarray:
    pixels = random.randint(_GEN_BLOB_PIXELS_MIN, _GEN_BLOB_PIXELS_MAX)
    dims = [
        (h, w)
        for h in range(_GEN_BLOB_H_MIN, _GEN_BLOB_H_MAX + 1)
        for w in range(_GEN_BLOB_W_MIN, _GEN_BLOB_W_MAX + 1)
        if h * w >= pixels
    ]
    h, w = random.choice(dims)
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    assert _gen_grow_mask(mask, pixels)

    bb = bbox(mask)
    assert bb is not None
    r0, c0, r1, c1 = bb
    return _gen_colored_patch(mask[r0 : r1 + 1, c0 : c1 + 1], color)


def _gen_adjacent_existing(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    cells: list[tuple[int, int]],
    cell_set: set[tuple[int, int]],
) -> bool:
    n = grid_a.shape[0]
    for r, c in cells:
        for dr, dc in _GEN_NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n) or (nr, nc) in cell_set:
                continue
            if grid_a[nr, nc] != _GEN_BG or grid_b[nr, nc] != _GEN_BG:
                return True
    return False


def _gen_place_distractor(inp: np.ndarray, out: np.ndarray) -> bool:
    n = inp.shape[0]
    color = random.choice(_GEN_COLORS)
    patch = _gen_blob_patch(color)
    h, w = patch.shape
    local_cells = [(int(r), int(c)) for r, c in np.argwhere(patch != _GEN_BG)]
    occupied = (inp != _GEN_BG) | (out != _GEN_BG)
    choices = []

    for r0 in range(n - h + 1):
        for c0 in range(n - w + 1):
            if _gen_touches_true_corner(r0, c0, h, w, n):
                continue
            cells = [(r0 + rr, c0 + cc) for rr, cc in local_cells]
            if any(occupied[r, c] for r, c in cells):
                continue
            cell_set = set(cells)
            if _gen_adjacent_existing(inp, out, cells, cell_set):
                continue
            choices.append((r0, c0))

    if not choices:
        return False

    r0, c0 = random.choice(choices)
    paste(inp, patch, r0, c0, transparent=_GEN_BG)
    paste(out, patch, r0, c0, transparent=_GEN_BG)
    return True


def _gen_bg_is_most_frequent(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = freq.get(_GEN_BG, 0)
    return all(bg_count > count for value, count in freq.items() if value != _GEN_BG)


def _gen_distractor_count(n: int) -> int:
    max_count = _GEN_DISTRACTOR_BASE_COUNT + round(
        (n - _GEN_MIN_SIZE) * _GEN_DISTRACTOR_EXTRA_MAX / _GEN_DISTRACTOR_SIZE_SPAN
    )
    min_count = max(_GEN_DISTRACTOR_BASE_COUNT, max_count - _GEN_DISTRACTOR_MIN_GAP)
    return random.randint(min_count, max_count)


def _gen_place_corner_sources(
    source_grid: np.ndarray, out: np.ndarray, occupied_corners: list[str]
) -> dict[str, tuple[np.ndarray, int]] | None:
    n = source_grid.shape[0]
    corner_objects: dict[str, tuple[np.ndarray, int]] = {}
    for corner in occupied_corners:
        h, w = random.choice(_GEN_CORNER_DIMS)
        mask = _gen_connected_corner_mask(corner, h, w)
        if mask is None:
            return None
        color = random.choice(_GEN_COLORS)
        corner_objects[corner] = (mask, color)
        patch = _gen_colored_patch(mask, color)
        r0, c0 = _gen_anchor(corner, n, h, w)
        if np.any(source_grid[r0 : r0 + h, c0 : c0 + w][mask] != _GEN_BG):
            return None
        paste(source_grid, patch, r0, c0, transparent=_GEN_BG)
        paste(out, patch, r0, c0, transparent=_GEN_BG)
    return corner_objects


def _gen_planned_stamps(
    corner_objects: dict[str, tuple[np.ndarray, int]], n: int
) -> dict[tuple[int, int], int] | None:
    planned: dict[tuple[int, int], int] = {}
    for corner, (mask, color) in corner_objects.items():
        for cell, stamp_color in _gen_stamp_cells(corner, mask, color, n).items():
            if cell in planned:
                return None
            planned[cell] = stamp_color
    return planned


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    source_grid = np.full((n, n), _GEN_BG, dtype=int)
    out = source_grid.copy()

    corner_count = random.randint(_GEN_CORNER_COUNT_MIN, _GEN_CORNER_COUNT_MAX)
    occupied_corners = random.sample(_GEN_CORNERS, corner_count)
    corner_objects = _gen_place_corner_sources(source_grid, out, occupied_corners)
    if corner_objects is None:
        return None

    planned = _gen_planned_stamps(corner_objects, n)
    if planned is None:
        return None

    for (r, c), color in planned.items():
        out[r, c] = color

    inp = out.copy()
    for r, c in planned:
        inp[r, c] = source_grid[r, c]

    for _ in range(_gen_distractor_count(n)):
        if not _gen_place_distractor(inp, out):
            return None

    if not _gen_bg_is_most_frequent(inp):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bc93ec48 input")
