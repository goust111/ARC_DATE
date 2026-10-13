from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", utils_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
find_components = _ARC_UTILS.find_components
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


BAR_ORIENTATIONS = ("H", "V")
BAR_LENGTHS = (3, 5, 7)
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
RECT_H_RANGE = (6, 8)
RECT_W_RANGE = (8, 10)
RECT_MARGIN = 4
SELECTOR_MIN_GAP = 2
BAR_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 500
BAR_PLACEMENT_ATTEMPTS = 140


def _bar_pixels(center_r: int, center_c: int, length: int, orient: str) -> tuple[tuple[int, int], ...]:
    half = int(length) // 2
    if orient == "H":
        return tuple((int(center_r), int(center_c + d)) for d in range(-half, half + 1))
    assert orient == "V"
    return tuple((int(center_r + d), int(center_c)) for d in range(-half, half + 1))


def _rotated_bar_pixels(center_r: int, center_c: int, length: int, orient: str) -> tuple[tuple[int, int], ...]:
    if length == 1:
        return ((int(center_r), int(center_c)),)
    return _bar_pixels(center_r, center_c, length, "V" if orient == "H" else "H")


def _bar_geometry(coords: np.ndarray) -> tuple[int, int, int, str]:
    r0, c0, r1, c1 = bbox_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    area = int(coords.shape[0])
    assert height == 1 or width == 1
    length = max(height, width)
    assert area == length and length % 2 == 1
    center_r = int((r0 + r1) // 2)
    center_c = int((c0 + c1) // 2)
    orient = "H" if height == 1 else "V"
    return center_r, center_c, length, orient


def _all_in_bounds(shape: tuple[int, int], coords: tuple[tuple[int, int], ...]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in coords)


def _outside_rect_with_gap(
    coords: tuple[tuple[int, int], ...],
    rect: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = rect
    for r, c in coords:
        if r0 - gap <= r <= r1 + gap and c0 - gap <= c <= c1 + gap:
            return False
    return True


def _line_policy_ok(
    coords: tuple[tuple[int, int], ...],
    kind: str,
    center: tuple[int, int],
    selected_rows: set[int],
    selected_cols: set[int],
) -> bool:
    center_r, center_c = center
    if kind == "row":
        if center_r not in selected_rows or center_c in selected_cols:
            return False
        for r, c in coords:
            if c in selected_cols:
                return False
            if r in selected_rows and r != center_r:
                return False
        return True

    if kind == "col":
        if center_c not in selected_cols or center_r in selected_rows:
            return False
        for r, c in coords:
            if r in selected_rows:
                return False
            if c in selected_cols and c != center_c:
                return False
        return True

    assert kind == "off"
    if center_r in selected_rows or center_c in selected_cols:
        return False
    return all(r not in selected_rows and c not in selected_cols for r, c in coords)


def _reserve_ok(
    footprint: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
) -> bool:
    return all(coord not in occupied for coord in footprint)


def _expanded4(
    coords: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        expanded.add((r, c))
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc):
                expanded.add((nr, nc))
    return expanded


def _selector_options(values: list[int], min_count: int, max_count: int) -> list[tuple[int, ...]]:
    options: list[tuple[int, ...]] = []
    for count in range(min_count, max_count + 1):
        for comb in itertools.combinations(values, count):
            if all(b - a >= SELECTOR_MIN_GAP for a, b in zip(comb, comb[1:])):
                options.append(tuple(int(v) for v in comb))
    return options


def _find_rectangle_and_selectors(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    mask = np.isin(grid, [Color.BLUE, Color.RED])
    rect_components = components_mask(mask, connectivity=4)
    assert len(rect_components) == 1, "expected one blue/red rectangle component"
    rect_coords = rect_components[0]
    rect = bbox_coords(rect_coords)
    r0, c0, r1, c1 = rect
    assert int(rect_coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1), "rectangle must be solid"

    selected_rows: set[int] = set()
    selected_cols: set[int] = set()
    border_cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border_cells.add((r0, c))
        border_cells.add((r1, c))
    for r in range(r0, r1 + 1):
        border_cells.add((r, c0))
        border_cells.add((r, c1))

    for r, c in border_cells:
        if int(grid[r, c]) != Color.RED:
            continue
        red_neighbors = []
        for nr, nc in neighbors4(r, c):
            if r0 <= nr <= r1 and c0 <= nc <= c1 and int(grid[nr, nc]) == Color.RED:
                red_neighbors.append((nr, nc))
        assert len(red_neighbors) == 1, "each protrusion must have exactly one red 4-neighbor"
        nr, nc = red_neighbors[0]
        if nr != r:
            selected_cols.add(int(c))
        else:
            selected_rows.add(int(r))

    assert selected_rows and selected_cols, "expected both selected rows and selected columns"
    return rect, sorted(selected_rows), sorted(selected_cols)


def _extract_bars(
    grid: np.ndarray,
    selected_rows: list[int],
    selected_cols: list[int],
) -> list[dict]:
    row_set = set(map(int, selected_rows))
    col_set = set(map(int, selected_cols))
    bars = []
    for comp in find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    ):
        center_r, center_c, length, orient = _bar_geometry(comp["coords"])
        bars.append(
            {
                "center": (center_r, center_c),
                "length": int(length),
                "orient": orient,
                "selected": bool(center_r in row_set or center_c in col_set),
                "input_coords": _bar_pixels(center_r, center_c, length, orient),
                "rotated_coords": _rotated_bar_pixels(center_r, center_c, length, orient),
            }
        )
    assert bars, "expected grey bars"
    return bars


def _render_output(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    selected_rows: list[int],
    selected_cols: list[int],
    bars: list[dict],
    *,
    rotate_selected: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == Color.GREY] = Color.BLACK
    bar_mask = np.zeros_like(out, dtype=bool)

    for bar in bars:
        coords = bar["rotated_coords"] if rotate_selected and bar["selected"] else bar["input_coords"]
        for r, c in coords:
            out[r, c] = Color.GREY
            bar_mask[r, c] = True

    r0, c0, r1, c1 = rect
    for r in selected_rows:
        out[int(r), c0 : c1 + 1] = Color.BLUE
    for c in selected_cols:
        out[r0 : r1 + 1, int(c)] = Color.BLUE

    for r in selected_rows:
        rr = int(r)
        for c in range(c0 - 1, -1, -1):
            if bar_mask[rr, c]:
                break
            out[rr, c] = Color.RED
        for c in range(c1 + 1, out.shape[1]):
            if bar_mask[rr, c]:
                break
            out[rr, c] = Color.RED

    for c in selected_cols:
        cc = int(c)
        for r in range(r0 - 1, -1, -1):
            if bar_mask[r, cc]:
                break
            out[r, cc] = Color.RED
        for r in range(r1 + 1, out.shape[0]):
            if bar_mask[r, cc]:
                break
            out[r, cc] = Color.RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use red leaf-like protrusions to choose cut lines, rotate matching grey bars, then extend obstacle-gated rays.

    Input structure:
    - Black is the background. A single central solid rectangle is made only of blue border cells and red interior cells.
    - Some border cells of that rectangle are red one-pixel protrusions: top/bottom protrusions touch the red interior only vertically, and left/right protrusions touch it only horizontally.
    - Outside the rectangle, several grey odd-length one-cell-thick bars lie in the black background.

    Rule:
    - Read each red protrusion on the rectangle border. If its only red 4-neighbor is above or below it, select that protrusion's column; if the only red 4-neighbor is left or right, select that protrusion's row.
    - Rotate every grey bar whose center lies on any selected row or selected column by 90 degrees about its own center; leave every other grey bar unchanged.
    - Inside the rectangle, repaint every selected row and selected column blue, creating blue cut lines through the red interior.
    - From the rectangle edges, extend red rays outward along every selected row and selected column. A ray stops immediately before the first grey pixel in its path after the bar rotations; if there is no grey obstacle, it continues to the grid boundary.

    Key insight:
    - The protrusions provide the same row/column set for two coupled actions: they choose which bars rotate and they choose where rays travel, so the post-rotation bar footprints determine the final ray stopping points.
    """
    g = np.asarray(grid, dtype=int)
    rect, selected_rows, selected_cols = _find_rectangle_and_selectors(g)
    bars = _extract_bars(g, selected_rows, selected_cols)
    return _render_output(g, rect, selected_rows, selected_cols, bars, rotate_selected=True)


def _try_place_bar(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    selected_rows: set[int],
    selected_cols: set[int],
    kind: str,
    line_value: int | None,
) -> dict | None:
    height, width = shape
    r0, c0, r1, c1 = rect

    for _ in range(BAR_PLACEMENT_ATTEMPTS):
        orient = random.choice(BAR_ORIENTATIONS)
        length = random.choice(BAR_LENGTHS)
        half = length // 2

        if kind == "row":
            center_r = int(line_value)
            side = random.choice(("left", "right"))
            lo = half
            hi = c0 - 2 - half
            if side == "right":
                lo = c1 + 2 + half
                hi = width - 1 - half
            if lo > hi:
                continue
            center_c = random.randint(lo, hi)
        elif kind == "col":
            center_c = int(line_value)
            side = random.choice(("top", "bottom"))
            lo = half
            hi = r0 - 2 - half
            if side == "bottom":
                lo = r1 + 2 + half
                hi = height - 1 - half
            if lo > hi:
                continue
            center_r = random.randint(lo, hi)
        else:
            assert kind == "off"
            if orient == "H":
                center_r = random.randint(0, height - 1)
                center_c = random.randint(half, width - 1 - half)
            else:
                center_r = random.randint(half, height - 1 - half)
                center_c = random.randint(0, width - 1)

        input_coords = _bar_pixels(center_r, center_c, length, orient)
        rotated_coords = _rotated_bar_pixels(center_r, center_c, length, orient)
        reserve = tuple(sorted(set(input_coords) | set(rotated_coords)))

        if not _all_in_bounds(shape, reserve):
            continue
        if not _outside_rect_with_gap(reserve, rect, gap=1):
            continue
        if not _line_policy_ok(input_coords, kind, (center_r, center_c), selected_rows, selected_cols):
            continue
        if not _reserve_ok(reserve, occupied):
            continue

        return {
            "kind": kind,
            "center": (center_r, center_c),
            "length": int(length),
            "orient": orient,
            "selected": kind in {"row", "col"},
            "input_coords": input_coords,
            "rotated_coords": rotated_coords,
            "reserve": reserve,
        }

    return None


def _ray_side_blocked(
    bar_mask: np.ndarray,
    rect: tuple[int, int, int, int],
    axis: str,
    index: int,
    side: str,
) -> bool:
    r0, c0, r1, c1 = rect
    if axis == "row":
        r = int(index)
        if side == "left":
            return any(bar_mask[r, c] for c in range(c0 - 1, -1, -1))
        return any(bar_mask[r, c] for c in range(c1 + 1, bar_mask.shape[1]))

    c = int(index)
    if side == "up":
        return any(bar_mask[r, c] for r in range(r0 - 1, -1, -1))
    return any(bar_mask[r, c] for r in range(r1 + 1, bar_mask.shape[0]))


def _bar_mask_from_meta(shape: tuple[int, int], bars: list[dict], rotate_selected: bool) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for bar in bars:
        coords = bar["rotated_coords"] if rotate_selected and bar["selected"] else bar["input_coords"]
        for r, c in coords:
            mask[r, c] = True
    return mask


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    if height < rect_h + 2 * RECT_MARGIN or width < rect_w + 2 * RECT_MARGIN:
        return None

    top = random.randint(RECT_MARGIN, height - rect_h - RECT_MARGIN)
    left = random.randint(RECT_MARGIN, width - rect_w - RECT_MARGIN)
    rect = (top, left, top + rect_h - 1, left + rect_w - 1)
    r0, c0, r1, c1 = rect

    row_values = list(range(r0 + 1, r1))
    col_values = list(range(c0 + 1, c1))
    row_options = _selector_options(row_values, 1, min(3, len(row_values)))
    col_options = _selector_options(col_values, 1, min(3, len(col_values)))
    if not row_options or not col_options:
        return None

    selected_rows = list(random.choice(row_options))
    selected_cols = list(random.choice(col_options))
    selected_row_set = set(selected_rows)
    selected_col_set = set(selected_cols)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = Color.BLUE
    grid[r0 + 1 : r1, c0 + 1 : c1] = Color.RED

    row_protrusions = []
    for row in selected_rows:
        col = c0 if random.random() < 0.5 else c1
        grid[row, col] = Color.RED
        row_protrusions.append((int(row), int(col)))

    col_protrusions = []
    for col in selected_cols:
        row = r0 if random.random() < 0.5 else r1
        grid[row, col] = Color.RED
        col_protrusions.append((int(row), int(col)))

    occupied: set[tuple[int, int]] = set()
    bars: list[dict] = []
    required = [
        ("row", random.choice(selected_rows)),
        ("col", random.choice(selected_cols)),
        ("off", None),
    ]
    extra = random.randint(BAR_COUNT_RANGE[0] - len(required), BAR_COUNT_RANGE[1] - len(required))
    for _ in range(extra):
        kind = random.choice(("row", "col", "off"))
        if kind == "row":
            required.append((kind, random.choice(selected_rows)))
        elif kind == "col":
            required.append((kind, random.choice(selected_cols)))
        else:
            required.append((kind, None))

    for kind, line_value in required:
        bar = _try_place_bar(
            grid.shape,
            rect,
            occupied,
            selected_row_set,
            selected_col_set,
            kind,
            line_value,
        )
        if bar is None:
            return None
        bars.append(bar)
        occupied.update(_expanded4(bar["reserve"], grid.shape))

    for bar in bars:
        for r, c in bar["input_coords"]:
            grid[r, c] = Color.GREY

    meta = {
        "rect": rect,
        "selected_rows": selected_rows,
        "selected_cols": selected_cols,
        "row_protrusions": row_protrusions,
        "col_protrusions": col_protrusions,
        "bars": bars,
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    rect = meta["rect"]
    selected_rows = meta["selected_rows"]
    selected_cols = meta["selected_cols"]
    bars = meta["bars"]

    expected = _render_output(grid, rect, selected_rows, selected_cols, bars, rotate_selected=True)
    detected = main(grid)
    if not np.array_equal(expected, detected):
        return False

    no_rotation = _render_output(grid, rect, selected_rows, selected_cols, bars, rotate_selected=False)
    if np.array_equal(expected, no_rotation):
        return False

    detected_rect, detected_rows, detected_cols = _find_rectangle_and_selectors(grid)
    if detected_rect != rect or detected_rows != selected_rows or detected_cols != selected_cols:
        return False

    extracted_bars = _extract_bars(grid, selected_rows, selected_cols)
    if len(extracted_bars) != len(bars):
        return False

    selected_kinds = {bar["kind"] for bar in bars if bar["selected"]}
    if selected_kinds != {"row", "col"}:
        return False
    if not any(bar["kind"] == "off" for bar in bars):
        return False

    r0, c0, r1, c1 = rect
    internal_changes = int(np.count_nonzero((grid[r0 : r1 + 1, c0 : c1 + 1] == Color.RED) & (expected[r0 : r1 + 1, c0 : c1 + 1] == Color.BLUE)))
    if internal_changes == 0:
        return False

    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    external_red = int(np.count_nonzero((expected == Color.RED) & outside))
    if external_red == 0:
        return False

    bar_mask = _bar_mask_from_meta(grid.shape, bars, rotate_selected=True)
    side_hits = []
    for row in selected_rows:
        side_hits.append(_ray_side_blocked(bar_mask, rect, "row", row, "left"))
        side_hits.append(_ray_side_blocked(bar_mask, rect, "row", row, "right"))
    for col in selected_cols:
        side_hits.append(_ray_side_blocked(bar_mask, rect, "col", col, "up"))
        side_hits.append(_ray_side_blocked(bar_mask, rect, "col", col, "down"))
    if not any(side_hits):
        return False
    if all(side_hits):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, meta = built
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
