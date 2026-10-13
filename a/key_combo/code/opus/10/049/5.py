from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
LINE = 3
WALL = 8
PAIR_COLORS = (2, 4, 6)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PAIR_PATTERNS = (
    ("LT", "TR", "RB"),
    ("LT", "BL", "RB"),
    ("TR", "RB", "BL"),
    ("LT", "TR", "BL"),
)


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _full_bar_positions(mask: np.ndarray, box: BBox) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = box
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    rows = [r0 + i for i in range(patch.shape[0]) if np.all(patch[i, :])]
    cols = [c0 + j for j in range(patch.shape[1]) if np.all(patch[:, j])]
    assert rows[0] == r0 and rows[-1] == r1
    assert cols[0] == c0 and cols[-1] == c1

    rebuilt = np.zeros_like(patch, dtype=bool)
    rebuilt[[r - r0 for r in rows], :] = True
    rebuilt[:, [c - c0 for c in cols]] = True
    assert np.array_equal(patch, rebuilt), "green drawing must be a union of full bars"
    return rows, cols


def _marker_sides(marker: Coord, box: BBox) -> list[str]:
    r, c = marker
    r0, c0, r1, c1 = box
    sides: list[str] = []
    if r == r0 - 1 and c0 <= c <= c1:
        sides.append("top")
    if r == r1 + 1 and c0 <= c <= c1:
        sides.append("bottom")
    if c == c0 - 1 and r0 <= r <= r1:
        sides.append("left")
    if c == c1 + 1 and r0 <= r <= r1:
        sides.append("right")
    return sides


def _infer_target_bbox(shape: tuple[int, int], original: BBox, markers: list[Coord]) -> BBox:
    h, w = map(int, shape)
    r0, c0, r1, c1 = original

    cand_r0 = {r0} | {r + 1 for r, _c in markers if 0 <= r + 1 < h}
    cand_r1 = {r1} | {r - 1 for r, _c in markers if 0 <= r - 1 < h}
    cand_c0 = {c0} | {c + 1 for _r, c in markers if 0 <= c + 1 < w}
    cand_c1 = {c1} | {c - 1 for _r, c in markers if 0 <= c - 1 < w}

    candidates: list[BBox] = []
    for nr0 in cand_r0:
        for nc0 in cand_c0:
            for nr1 in cand_r1:
                for nc1 in cand_c1:
                    if not (0 <= nr0 <= r0 <= r1 <= nr1 < h and 0 <= nc0 <= c0 <= c1 <= nc1 < w):
                        continue
                    box = (nr0, nc0, nr1, nc1)
                    if all(len(_marker_sides(marker, box)) == 1 for marker in markers):
                        candidates.append(box)
    assert len(candidates) == 1, "control pixels must define a unique stretched bbox"
    return candidates[0]


def _round_half_up(num: int, den: int) -> int:
    return (2 * num + den) // (2 * den)


def _scale_positions(old_positions: list[int], old_lo: int, old_hi: int, new_lo: int, new_hi: int) -> list[int]:
    old_span = old_hi - old_lo
    new_span = new_hi - new_lo
    assert old_span > 0
    mapped = []
    for pos in old_positions:
        local = pos - old_lo
        mapped.append(new_lo + _round_half_up(local * new_span, old_span))
    assert mapped[0] == new_lo and mapped[-1] == new_hi
    assert len(mapped) == len(set(mapped)), "scaled bars must stay distinct"
    return mapped


def _draw_rectilinear(grid: np.ndarray, box: BBox, rows: list[int], cols: list[int], color: int) -> None:
    r0, c0, r1, c1 = box
    for r in rows:
        grid[r, c0 : c1 + 1] = color
    for c in cols:
        grid[r0 : r1 + 1, c] = color


def _scan_control_colors(grid: np.ndarray) -> list[int]:
    seen = {BG, LINE, WALL}
    order: list[int] = []
    for raw in grid.ravel():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _path_cells(start: Coord, stop: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    pts: list[Coord] = []

    if r0 == r1 or c0 == c1:
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                pts.append((r0, c))
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                pts.append((r, c0))
        return pts

    if first_axis == "h":
        c_step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + c_step, c_step):
            pts.append((r0, c))
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            pts.append((r, c1))
        return pts

    assert first_axis == "v"
    r_step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + r_step, r_step):
        pts.append((r, c0))
    c_step = 1 if c1 >= c0 else -1
    for c in range(c0 + c_step, c1 + c_step, c_step):
        pts.append((r1, c))
    return pts


def _path_clear(grid: np.ndarray, path: list[Coord]) -> bool:
    if len(path) <= 2:
        return True
    return all(int(grid[r, c]) == BG for r, c in path[1:-1])


def _axis_legality(grid: np.ndarray, start: Coord, stop: Coord) -> tuple[bool, bool]:
    if start[0] == stop[0] or start[1] == stop[1]:
        direct = _path_clear(grid, _path_cells(start, stop, "h"))
        return direct, direct
    return (
        _path_clear(grid, _path_cells(start, stop, "h")),
        _path_clear(grid, _path_cells(start, stop, "v")),
    )


def _choose_path(grid: np.ndarray, start: Coord, stop: Coord) -> tuple[list[Coord], str]:
    if start[0] == stop[0] or start[1] == stop[1]:
        path = _path_cells(start, stop, "h")
        assert _path_clear(grid, path)
        return path, "h"

    h_path = _path_cells(start, stop, "h")
    v_path = _path_cells(start, stop, "v")
    h_ok = _path_clear(grid, h_path)
    v_ok = _path_clear(grid, v_path)
    assert h_ok or v_ok, "each color pair needs a valid one-bend route"
    if h_ok:
        return h_path, "h"
    return v_path, "v"


def _connect_controls(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    out = grid.copy()
    axes: list[str] = []
    for color in _scan_control_colors(grid):
        pts = sorted((tuple(map(int, coord)) for coord in np.argwhere(grid == color)))
        assert len(pts) == 2
        path, axis = _choose_path(out, pts[0], pts[1])
        axes.append(axis)
        for r, c in path[1:-1]:
            out[r, c] = color
    return out, axes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the green rectilinear drawing to the control-defined bbox, then connect same-colored control pixels around the stretched obstacle.

    Input structure:
    - Black is the background, one green 4-connected rectilinear line drawing is the unique stretchable object, and teal cells are fixed wall segments.
    - Three other colors appear as isolated control pixels, exactly two pixels per color.
    - The green drawing is made only from full horizontal and vertical bars inside its bbox, while each control pixel sits exactly one cell outside one side of the final stretched bbox.

    Rule:
    - Find the green drawing bbox and infer the unique larger target bbox for which every control pixel is exactly one cell outside exactly one side.
    - Move the outer green bars to the target edges and map every interior full row or column proportionally inside the stretched bbox, redrawing each bar across the full new width or height.
    - Then process the control colors in first-appearance scan order and connect each same-colored pair by the shortest orthogonal path with at most one bend, using only black cells and treating the stretched green drawing, teal walls, other control pixels, and earlier connections as blockers.

    Key insight:
    - The same colored pixels first act as geometric constraints that enlarge the green obstacle and then immediately become the endpoints that must route around the obstacle they created.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_mask = g == LINE
    original = _bbox(line_mask)
    assert original is not None
    rows, cols = _full_bar_positions(line_mask, original)

    markers = [tuple(map(int, coord)) for coord in np.argwhere(~np.isin(g, (BG, LINE, WALL)))]
    assert len(markers) == 6
    target = _infer_target_bbox(g.shape, original, markers)

    out = g.copy()
    out[line_mask] = BG
    new_rows = _scale_positions(rows, original[0], original[2], target[0], target[2])
    new_cols = _scale_positions(cols, original[1], original[3], target[1], target[3])
    _draw_rectilinear(out, target, new_rows, new_cols, LINE)

    out, _axes = _connect_controls(out)
    return out


def _rand_inclusive(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _pick_from_ranges(ranges: list[tuple[int, int]]) -> int:
    valid = [(lo, hi) for lo, hi in ranges if lo <= hi]
    assert valid
    lengths = [hi - lo + 1 for lo, hi in valid]
    total = sum(lengths)
    ticket = random.randrange(total)
    acc = 0
    for (lo, hi), length in zip(valid, lengths):
        if ticket < acc + length:
            return lo + (ticket - acc)
        acc += length
    raise AssertionError("unreachable")


def _sample_box_geometry() -> tuple[int, int, int, int, int, int]:
    while True:
        old_h = _rand_inclusive(4, 6)
        old_w = _rand_inclusive(4, 6)
        top = _rand_inclusive(0, 3)
        bottom = _rand_inclusive(0, 3)
        left = _rand_inclusive(0, 3)
        right = _rand_inclusive(0, 3)
        if top + bottom == 0 or left + right == 0:
            continue
        new_h = old_h + top + bottom
        new_w = old_w + left + right
        if 8 <= new_h <= 12 and 8 <= new_w <= 12:
            return old_h, old_w, top, bottom, left, right


def _sample_bar_offsets(span: int) -> list[int]:
    count = 3 if random.random() < 0.75 else 2
    if count == 2:
        return [0, span - 1]
    interior = list(range(1, span - 1))
    return [0, random.choice(interior), span - 1]


def _target_slots(target: BBox, original: BBox) -> dict[str, int]:
    tr0, tc0, tr1, tc1 = target
    or0, oc0, or1, oc1 = original
    target_h = tr1 - tr0 + 1
    target_w = tc1 - tc0 + 1

    low_col_end = tc0 + max(1, target_w // 3) - 1
    high_col_start = tc1 - max(1, target_w // 3) + 1
    low_row_end = tr0 + max(1, target_h // 3) - 1
    high_row_start = tr1 - max(1, target_h // 3) + 1

    return {
        "top_left": _pick_from_ranges([(tc0, oc0 - 1), (tc0, low_col_end)]),
        "top_right": _pick_from_ranges([(oc1 + 1, tc1), (high_col_start, tc1)]),
        "bottom_left": _pick_from_ranges([(tc0, oc0 - 1), (tc0, low_col_end)]),
        "bottom_right": _pick_from_ranges([(oc1 + 1, tc1), (high_col_start, tc1)]),
        "left_top": _pick_from_ranges([(tr0, or0 - 1), (tr0, low_row_end)]),
        "right_top": _pick_from_ranges([(tr0, or0 - 1), (tr0, low_row_end)]),
        "left_bottom": _pick_from_ranges([(or1 + 1, tr1), (high_row_start, tr1)]),
        "right_bottom": _pick_from_ranges([(or1 + 1, tr1), (high_row_start, tr1)]),
    }


def _pair_endpoints(pair_type: str, target: BBox, slots: dict[str, int]) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = target
    if pair_type == "LT":
        return (r0 - 1, slots["top_left"]), (slots["left_top"], c0 - 1)
    if pair_type == "TR":
        return (r0 - 1, slots["top_right"]), (slots["right_top"], c1 + 1)
    if pair_type == "RB":
        return (slots["right_bottom"], c1 + 1), (r1 + 1, slots["bottom_right"])
    assert pair_type == "BL"
    return (r1 + 1, slots["bottom_left"]), (slots["left_bottom"], c0 - 1)


def _segment_cells(start: Coord, orient: str, length: int) -> list[Coord]:
    r, c = start
    if orient == "h":
        return [(r, c + i) for i in range(length)]
    if orient == "v":
        return [(r + i, c) for i in range(length)]
    if orient == "rh":
        return [(r, c - i) for i in range(length)]
    assert orient == "uv"
    return [(r - i, c) for i in range(length)]


def _safe_wall_segment(
    cells: list[Coord],
    shape: tuple[int, int],
    occupied: set[Coord],
    line_cells: set[Coord],
    controls: set[Coord],
) -> bool:
    h, w = shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied or (r, c) in controls or (r, c) in line_cells:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in line_cells:
                return False
    return min(abs(r - cr) + abs(c - cc) for r, c in cells for cr, cc in controls) <= 4


def _place_walls(
    base: np.ndarray,
    route_cells: set[Coord],
    line_cells: set[Coord],
    controls: set[Coord],
) -> np.ndarray | None:
    grid = base.copy()
    occupied = set(route_cells) | {(int(r), int(c)) for r, c in np.argwhere(grid == LINE)}
    target_count = _rand_inclusive(2, 4)
    placed = 0

    candidates: list[list[Coord]] = []
    for _ in range(240):
        length = _rand_inclusive(2, 3)
        orient = random.choice(("h", "rh", "v", "uv"))
        cr, cc = random.choice(sorted(controls))
        dr = _rand_inclusive(-3, 3)
        dc = _rand_inclusive(-3, 3)
        seg = _segment_cells((cr + dr, cc + dc), orient, length)
        if _safe_wall_segment(seg, grid.shape, occupied, line_cells, controls):
            candidates.append(seg)

    random.shuffle(candidates)
    for seg in candidates:
        if placed == target_count:
            break
        if any((r, c) in occupied for r, c in seg):
            continue
        for r, c in seg:
            grid[r, c] = WALL
            occupied.add((r, c))
        placed += 1

    if placed < 2:
        return None
    return grid


def _stage1_changes_route(input_grid: np.ndarray, stretched_grid: np.ndarray) -> bool:
    for color in _scan_control_colors(input_grid):
        pts = sorted((tuple(map(int, coord)) for coord in np.argwhere(input_grid == color)))
        if _axis_legality(input_grid, pts[0], pts[1]) != _axis_legality(stretched_grid, pts[0], pts[1]):
            return True
    return False


def _validate_instance(
    inp: np.ndarray,
    stretched: np.ndarray,
    expected: np.ndarray,
    original: BBox,
    target: BBox,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if _infer_target_bbox(inp.shape, original, [tuple(map(int, x)) for x in np.argwhere(~np.isin(inp, (BG, LINE, WALL)))]) != target:
        return False
    if not _stage1_changes_route(inp, stretched):
        return False

    simulated = stretched.copy()
    axes: list[str] = []
    for color in _scan_control_colors(stretched):
        pts = sorted((tuple(map(int, coord)) for coord in np.argwhere(stretched == color)))
        h_ok, v_ok = _axis_legality(simulated, pts[0], pts[1])
        if not (h_ok or v_ok):
            return False
        path, axis = _choose_path(simulated, pts[0], pts[1])
        axes.append(axis)
        for r, c in path[1:-1]:
            simulated[r, c] = color
    if not np.array_equal(simulated, expected):
        return False
    return "h" in axes and "v" in axes


def _generate_attempt() -> np.ndarray | None:
    old_h, old_w, top, bottom, left, right = _sample_box_geometry()
    new_h = old_h + top + bottom
    new_w = old_w + left + right

    H = _rand_inclusive(max(16, new_h + 6), 22)
    W = _rand_inclusive(max(16, new_w + 6), 22)
    target_r0 = _rand_inclusive(2, H - new_h - 3)
    target_c0 = _rand_inclusive(2, W - new_w - 3)
    target = (target_r0, target_c0, target_r0 + new_h - 1, target_c0 + new_w - 1)
    original = (target[0] + top, target[1] + left, target[2] - bottom, target[3] - right)

    row_offsets = _sample_bar_offsets(old_h)
    col_offsets = _sample_bar_offsets(old_w)
    old_rows = [original[0] + offset for offset in row_offsets]
    old_cols = [original[1] + offset for offset in col_offsets]
    new_rows = _scale_positions(old_rows, original[0], original[2], target[0], target[2])
    new_cols = _scale_positions(old_cols, original[1], original[3], target[1], target[3])

    stretched = np.full((H, W), BG, dtype=int)
    _draw_rectilinear(stretched, target, new_rows, new_cols, LINE)

    pair_types = list(random.choice(PAIR_PATTERNS))
    random.shuffle(pair_types)
    slots = _target_slots(target, original)
    colors = list(PAIR_COLORS)
    random.shuffle(colors)

    controls: dict[int, tuple[Coord, Coord]] = {}
    for color, pair_type in zip(colors, pair_types):
        a, b = _pair_endpoints(pair_type, target, slots)
        if a == b:
            return None
        controls[color] = (a, b)
        if pair_type in ("LT", "TR") and a[1] == target[1] - 1:
            return None
        if pair_type in ("RB", "BL") and a[0] == target[0] - 1:
            return None

    marker_cells = [cell for pair in controls.values() for cell in pair]
    if len(marker_cells) != len(set(marker_cells)):
        return None
    if any(stretched[r, c] != BG for r, c in marker_cells):
        return None
    for color, (a, b) in controls.items():
        if set(_marker_sides(a, target)) == set() or set(_marker_sides(b, target)) == set():
            return None
        if _marker_sides(a, target)[0] == _marker_sides(b, target)[0]:
            return None
        stretched[a] = color
        stretched[b] = color

    prelim_expected, prelim_axes = _connect_controls(stretched)
    if "h" not in prelim_axes or "v" not in prelim_axes:
        return None
    route_cells = {
        (int(r), int(c))
        for color in _scan_control_colors(stretched)
        for r, c in np.argwhere(prelim_expected == color)
        if stretched[r, c] == BG
    }
    line_cells = {(int(r), int(c)) for r, c in np.argwhere(stretched == LINE)}
    stretched_with_walls = _place_walls(stretched, route_cells, line_cells, set(marker_cells))
    if stretched_with_walls is None:
        return None

    expected, axes = _connect_controls(stretched_with_walls)
    if "h" not in axes or "v" not in axes:
        return None

    inp = stretched_with_walls.copy()
    inp[inp == LINE] = BG
    _draw_rectilinear(inp, original, old_rows, old_cols, LINE)

    if not _validate_instance(inp, stretched_with_walls, expected, original, target):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
