from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, *, exclude_colors: set[int] | None = None) -> list[dict]:
    exclude = set() if exclude_colors is None else set(map(int, exclude_colors))
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in exclude)
    comps: list[dict] = []
    for color in colors:
        for coords in _mask_components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_set(component: dict) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in component["coords"]}


def _is_corner_bracket(component: dict) -> bool:
    coords = _component_set(component)
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or component["area"] != h + w - 1:
        return False
    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(coords == candidate for candidate in candidates)


def _find_bracket_and_marker(grid: np.ndarray) -> tuple[dict, tuple[int, int], int]:
    grey_components = [comp for comp in _find_monochrome_components(grid, exclude_colors={BLACK}) if comp["color"] == GREY]
    assert grey_components, "expected at least one grey component"
    bracket = max(grey_components, key=lambda comp: (int(comp["area"]),) + tuple(-int(v) for v in comp["bbox"]))
    assert _is_corner_bracket(bracket), "largest grey component must be the bracket"

    bracket_coords = _component_set(bracket)
    r0, c0, r1, c1 = bracket["bbox"]
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_coords and int(grid[r, c]) != BLACK
    ]
    assert len(markers) == 1, "expected exactly one marker inside the bracket"
    marker = markers[0]
    marker_color = int(grid[marker])
    assert marker_color not in (BLACK, GREY)
    return bracket, marker, marker_color


def _erase_phase(grid: np.ndarray) -> tuple[np.ndarray, dict, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"
    bracket, marker, erase_color = _find_bracket_and_marker(g)
    out = g.copy()
    out[out == erase_color] = bg
    out[marker] = erase_color
    return out, bracket, marker, erase_color


def _gather_movable_components(
    grid: np.ndarray,
    marker: tuple[int, int],
    *,
    immobile_colors: set[int] | None = None,
) -> list[dict]:
    immobile = set() if immobile_colors is None else set(map(int, immobile_colors))
    colors = [
        int(v)
        for v in sorted(np.unique(grid))
        if int(v) not in {BLACK, GREY} | immobile
    ]
    comps: list[dict] = []
    mr, mc = marker
    for color in colors:
        mask = grid == color
        if 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1] and color == int(grid[marker]):
            mask = mask.copy()
            mask[mr, mc] = False
        for coords in _mask_components(mask):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _drop_order(component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (-r1, -r0, c0, c1)


def _apply_gravity(
    grid: np.ndarray,
    marker: tuple[int, int],
    *,
    immobile_colors: set[int] | None = None,
    reverse: bool = False,
    return_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    immobile = set() if immobile_colors is None else set(map(int, immobile_colors))

    fixed_mask = g == GREY
    fixed_mask[marker] = True
    if immobile:
        fixed_mask |= np.isin(g, list(immobile))

    out = np.full_like(g, BLACK)
    out[fixed_mask] = g[fixed_mask]
    occupied = fixed_mask.copy()
    fixed_only = fixed_mask.copy()

    components = _gather_movable_components(g, marker, immobile_colors=immobile)
    components = sorted(components, key=_drop_order, reverse=reverse)

    meta = []
    for component in components:
        coords = np.asarray(component["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        reason = "wall"
        while True:
            next_rows = rows + shift + 1
            if np.any(next_rows >= h):
                reason = "wall"
                break
            blocked = occupied[next_rows, cols]
            if np.any(blocked):
                block_rows = next_rows[blocked]
                block_cols = cols[blocked]
                if np.any(fixed_only[block_rows, block_cols]):
                    reason = "fixed"
                else:
                    reason = "object"
                break
            shift += 1

        dest_rows = rows + shift
        assert not np.any(occupied[dest_rows, cols])
        out[dest_rows, cols] = int(component["color"])
        occupied[dest_rows, cols] = True
        if return_meta:
            meta.append(
                {
                    "color": int(component["color"]),
                    "reason": reason,
                    "shift": int(shift),
                    "source_bbox": tuple(map(int, component["bbox"])),
                    "dest_bbox": _bbox(np.column_stack([dest_rows, cols])),
                }
            )

    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the bracket-selected obstacle color, then let the remaining objects fall straight downward.

    Input structure
    - The background is black.
    - The largest grey 4-connected component is a corner bracket whose bbox interior contains exactly one non-black marker pixel.
    - Outside the bracket there are several monochromatic 4-connected objects, and one marker-matching color forms the obstacle layer in the middle-lower rows.

    Rule
    - Read the unique marker color inside the grey corner bracket.
    - Recolor every other pixel of that color to black, preserving the bracket and the marker itself.
    - On the erased grid, treat the grey bracket and marker as fixed, then process every remaining non-black, non-grey object from lower bbox bottom to higher bbox bottom, dropping each object straight down until the next step would leave the grid or hit an already settled object.

    Key insight
    - The bracket is not selecting the final answer directly: it first removes one obstacle color, and that terrain change opens new vertical fall paths whose final layout depends on gravity order and collision blocking.
    """
    erased, _bracket, marker, _erase_color = _erase_phase(np.asarray(grid, dtype=int))
    return _apply_gravity(erased, marker)


SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
)


def _transform_shape(shape: np.ndarray) -> np.ndarray:
    out = np.array(shape, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    coords = np.argwhere(out != 0)
    r0, c0, r1, c1 = _bbox(coords)
    return out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_cells(shape: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(shape != 0):
        coords.add((top + int(r), left + int(c)))
    return coords


def _touches(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    if existing & cells:
        return True
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing:
                return True
    return False


def _render_bracket(
    grid: np.ndarray,
    side: str,
    size: int,
    orientation: str,
    marker_color: int,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = grid.shape
    top = h - size
    left = 0 if side == "left" else w - size
    r0, c0, r1, c1 = top, left, h - 1, left + size - 1
    if orientation == "tl":
        cells = {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    elif orientation == "tr":
        cells = {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    elif orientation == "bl":
        cells = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    else:
        cells = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}

    interior = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cells]
    marker = random.choice(interior)
    for r, c in cells:
        grid[r, c] = GREY
    grid[marker] = marker_color
    return cells, marker


def _make_shelf(span_left: int, span_right: int, shelf_row: int, h: int) -> set[tuple[int, int]]:
    width = span_right - span_left + 1
    if width < 3:
        span_right = span_left + 2
        width = 3
    if width > 4:
        if random.random() < 0.5:
            span_left = span_right - 3
        else:
            span_right = span_left + 3
        width = 4
    cells = {(shelf_row, c) for c in range(span_left, span_right + 1)}
    nub_col = span_left if random.random() < 0.5 else span_right
    nub_row = shelf_row + 1 if shelf_row + 1 < h - 2 else shelf_row - 1
    if 0 <= nub_row < h:
        cells.add((nub_row, nub_col))
    return cells


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.asarray(sorted(cells), dtype=int)
    return _bbox(coords)


def _sample_object(shape: np.ndarray, top: int, left: int, color: int) -> dict:
    cells = _shape_cells(shape, top, left)
    bbox = _bbox_of_cells(cells)
    return {"shape": shape, "top": top, "left": left, "cells": cells, "bbox": bbox, "color": color}


def _place_other_object(
    placed_cells: set[tuple[int, int]],
    shelf_row: int,
    usable_left: int,
    usable_right: int,
    shape: np.ndarray,
    *,
    prefer_left: bool,
    hard_left: int,
    hard_right: int,
    min_bottom_gap: int,
) -> dict | None:
    h, w = shape.shape
    top_candidates = list(range(0, shelf_row - h - min_bottom_gap + 2))
    if not top_candidates:
        return None
    col_candidates = list(range(hard_left, hard_right - w + 2))
    if prefer_left:
        col_candidates.sort()
    else:
        col_candidates.sort(reverse=True)
    random.shuffle(top_candidates)
    for top in top_candidates:
        for left in col_candidates:
            if left < usable_left or left + w - 1 > usable_right:
                continue
            cells = _shape_cells(shape, top, left)
            if max(r for r, _ in cells) >= shelf_row:
                continue
            if _touches(placed_cells, cells):
                continue
            return {"shape": shape, "top": top, "left": left, "cells": cells, "bbox": _bbox_of_cells(cells)}
    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(16, 20)
    grid = np.zeros((h, w), dtype=int)

    erase_color = random.choice(PALETTE)
    side = random.choice(("left", "right"))
    bracket_size = random.randint(3, 4)
    orientation = random.choice(("tl", "tr", "bl", "br"))
    bracket_cells, marker = _render_bracket(grid, side, bracket_size, orientation, erase_color)

    usable_left = bracket_size + 2 if side == "left" else 1
    usable_right = w - bracket_size - 3 if side == "right" else w - 2
    if usable_right - usable_left < 8:
        return None

    shelf_row = random.randint(h // 2, h // 2 + 2)
    total_objects = random.randint(5, 6)
    mobile_count = total_objects - 2
    mobile_colors = random.sample([c for c in PALETTE if c != erase_color], mobile_count)

    shapes = [_transform_shape(random.choice(SHAPES)) for _ in range(mobile_count)]
    lower_shape, upper_shape = shapes[0], shapes[1]
    lower_color, upper_color = mobile_colors[0], mobile_colors[1]
    lw = lower_shape.shape[1]
    uw = upper_shape.shape[1]
    chain_width = max(lw, uw)

    group2_shapes = shapes[2:]
    group2_width = max(shape.shape[1] for shape in group2_shapes)

    if usable_right - usable_left + 1 < chain_width + group2_width + 4:
        return None

    if random.random() < 0.5:
        chain_left = random.randint(usable_left + group2_width + 3, usable_right - chain_width + 1)
        other_band_left = usable_left
        other_band_right = chain_left - 3
        other_prefer_left = True
    else:
        chain_left = random.randint(usable_left, usable_right - chain_width - group2_width - 2)
        other_band_left = chain_left + chain_width + 3
        other_band_right = usable_right
        other_prefer_left = False
    if other_band_right - other_band_left + 1 < group2_width:
        return None

    placed: list[dict] = []
    placed_cells: set[tuple[int, int]] = set(bracket_cells) | {marker}

    lower_bottom = shelf_row - random.randint(1, 2)
    lower_top = lower_bottom - lower_shape.shape[0] + 1
    upper_gap = random.randint(2, 3)
    upper_bottom = lower_top - upper_gap
    upper_top = upper_bottom - upper_shape.shape[0] + 1
    if lower_top < 1 or upper_top < 0:
        return None

    lower_obj = _sample_object(lower_shape, lower_top, chain_left, lower_color)
    upper_obj = _sample_object(upper_shape, upper_top, chain_left, upper_color)
    if _touches(placed_cells, lower_obj["cells"]):
        return None
    placed_cells |= lower_obj["cells"]
    placed.append(lower_obj)
    if _touches(placed_cells, upper_obj["cells"]):
        return None
    placed_cells |= upper_obj["cells"]
    placed.append(upper_obj)

    extra_bases = [shelf_row - 1, shelf_row - 2, shelf_row - 3]
    for idx, shape in enumerate(group2_shapes, start=2):
        obj = _place_other_object(
            placed_cells,
            shelf_row,
            usable_left,
            usable_right,
            shape,
            prefer_left=other_prefer_left,
            hard_left=other_band_left,
            hard_right=other_band_right,
            min_bottom_gap=random.choice(extra_bases),
        )
        if obj is None:
            return None
        obj["color"] = mobile_colors[idx]
        placed_cells |= obj["cells"]
        placed.append(obj)

    chain_span = (
        min(c for _r, c in lower_obj["cells"] | upper_obj["cells"]),
        max(c for _r, c in lower_obj["cells"] | upper_obj["cells"]),
    )
    other_cells = set().union(*(obj["cells"] for obj in placed[2:]))
    other_span = (
        min(c for _r, c in other_cells),
        max(c for _r, c in other_cells),
    )
    shelf1 = _make_shelf(chain_span[0], chain_span[1], shelf_row, h)
    shelf2 = _make_shelf(other_span[0], other_span[1], shelf_row, h)
    if _touches(bracket_cells | {marker}, shelf1 | shelf2):
        return None
    if _touches(shelf1, shelf2):
        return None
    if any(_touches(shelf1 | shelf2, obj["cells"]) for obj in placed):
        return None

    _paint_cells(grid, shelf1, erase_color)
    _paint_cells(grid, shelf2, erase_color)
    for obj in placed:
        _paint_cells(grid, obj["cells"], obj["color"])

    return grid


def _terrain_without_erasing(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bracket, marker, erase_color = _find_bracket_and_marker(g)
    return _apply_gravity(g, marker, immobile_colors={erase_color})


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        erased, _bracket, marker, erase_color = _erase_phase(grid)
        out, meta = _apply_gravity(erased, marker, return_meta=True)
        main_out = main(grid)
        if not np.array_equal(out, main_out):
            return False
        if np.array_equal(out, grid):
            return False
        if np.array_equal(out, erased):
            return False
        wrong_terrain, wrong_meta = _apply_gravity(grid, marker, immobile_colors={erase_color}, return_meta=True)
        if np.array_equal(wrong_terrain, out):
            return False
        reverse = _apply_gravity(erased, marker, reverse=True)
        if np.array_equal(reverse, out):
            return False
        reasons = {item["reason"] for item in meta}
        if "wall" not in reasons or "object" not in reasons:
            return False
        if not any(item["reason"] == "fixed" for item in wrong_meta):
            return False
        if sum(item["shift"] > 0 for item in meta) < 2:
            return False
        outside_marker = int(np.count_nonzero(grid == erase_color)) - 1
        if outside_marker < 6:
            return False
        if len(_gather_movable_components(erased, marker)) < 3:
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _build_candidate()
        if grid is None:
            continue
        if _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
