from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
ORANGE = 7

QUADRANT_ORDER = ("TL", "TR", "BL", "BR")
FILL_START_EDGE = {
    "TL": "top",
    "TR": "right",
    "BL": "left",
    "BR": "bottom",
}

_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_RECENT_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append({"coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _full_yellow_cross(grid: np.ndarray) -> tuple[int, int]:
    yellow_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == YELLOW)]
    yellow_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == YELLOW)]
    assert len(yellow_rows) == 1 and len(yellow_cols) == 1, "expected one full yellow row and one full yellow column"
    return yellow_rows[0], yellow_cols[0]


def _quadrant_of_bbox(bbox: tuple[int, int, int, int], mid_r: int, mid_c: int) -> str:
    r0, c0, r1, c1 = bbox
    if r1 < mid_r and c1 < mid_c:
        return "TL"
    if r1 < mid_r and c0 > mid_c:
        return "TR"
    if r0 > mid_r and c1 < mid_c:
        return "BL"
    if r0 > mid_r and c0 > mid_c:
        return "BR"
    raise AssertionError("rectangle must lie entirely within one quadrant")


def _adjacent_gray_cells(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    touched: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            assert int(grid[r, c]) == ORANGE
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GRAY:
                    touched.add((nr, nc))
    return touched


def _scene_info(grid: np.ndarray) -> tuple[int, int, dict[str, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mid_r, mid_c = _full_yellow_cross(g)

    orange_components = _find_color_components(g, ORANGE)
    assert len(orange_components) == 4, "expected exactly four orange rectangles"

    by_quadrant: dict[str, dict] = {}
    for comp in orange_components:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == ORANGE), "orange component must be a solid rectangle"
        quadrant = _quadrant_of_bbox(bbox, mid_r, mid_c)
        assert quadrant not in by_quadrant, "expected one orange rectangle per quadrant"
        gray_cells = _adjacent_gray_cells(g, bbox)
        by_quadrant[quadrant] = {
            "bbox": bbox,
            "count": len(gray_cells),
            "gray_cells": gray_cells,
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
        }

    assert set(by_quadrant) == set(QUADRANT_ORDER), "missing orange rectangle in some quadrant"
    return mid_r, mid_c, by_quadrant


def _paint_stripes(out: np.ndarray, quadrant: str, bbox: tuple[int, int, int, int], count: int) -> None:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if quadrant == "TL":
        assert 0 <= count <= height
        out[r0 : r0 + count, c0 : c1 + 1] = GRAY
    elif quadrant == "TR":
        assert 0 <= count <= width
        out[r0 : r1 + 1, c1 - count + 1 : c1 + 1] = GRAY
    elif quadrant == "BL":
        assert 0 <= count <= width
        out[r0 : r1 + 1, c0 : c0 + count] = GRAY
    else:
        assert quadrant == "BR"
        assert 0 <= count <= height
        out[r1 - count + 1 : r1 + 1, c0 : c1 + 1] = GRAY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer the counts of exterior gray bars into quadrant-directed stripes inside the orange rectangles.

    Input structure:
    - The grid has a black background with one full yellow horizontal line and one full yellow vertical line crossing at the center.
    - Each of the four quadrants contains exactly one solid orange rectangle that stays fully inside that quadrant.
    - Some rectangles have one-cell-thick gray bar segments 4-adjacent outside the rectangle; other rectangles have no gray contact.

    Rule:
    - Use the yellow cross to determine the four quadrants.
    - For each orange rectangle, count all exterior gray pixels that are 4-adjacent to that rectangle.
    - Fill that many complete rows or columns inside the rectangle, with the start edge determined only by the rectangle's quadrant: top for top-left, right for top-right, left for bottom-left, bottom for bottom-right.
    - Erase every original exterior gray bar to black and keep the yellow cross unchanged.

    Key insight:
    - The gray pixels encode only the amount of fill; the fill direction is hidden in the rectangle's quadrant, not in the bars' physical contact side.
    """
    g = np.asarray(grid, dtype=int)
    _mid_r, _mid_c, by_quadrant = _scene_info(g)

    out = g.copy()
    out[g == GRAY] = BLACK
    for quadrant in QUADRANT_ORDER:
        info = by_quadrant[quadrant]
        _paint_stripes(out, quadrant, info["bbox"], info["count"])
    return out


def _quadrant_bounds(shape: tuple[int, int], quadrant: str) -> tuple[int, int, int, int]:
    h, w = shape
    mid_r, mid_c = h // 2, w // 2
    if quadrant == "TL":
        return 0, mid_r - 1, 0, mid_c - 1
    if quadrant == "TR":
        return 0, mid_r - 1, mid_c + 1, w - 1
    if quadrant == "BL":
        return mid_r + 1, h - 1, 0, mid_c - 1
    return mid_r + 1, h - 1, mid_c + 1, w - 1


def _fill_capacity(quadrant: str, height: int, width: int) -> int:
    return height if quadrant in {"TL", "BR"} else width


def _bars_compatible_with(quadrant: str, active: bool, theme_kind: str) -> tuple[str | None, str | None]:
    if not active:
        return None, None

    start_edge = FILL_START_EDGE[quadrant]
    if theme_kind == "match_only":
        if start_edge in {"top", "bottom"}:
            return "horizontal", start_edge
        return "vertical", start_edge

    if theme_kind == "anti_orientation":
        if quadrant in {"TL", "BR"}:
            side_choices = ("left", "right")
            return "vertical", side_choices[int(_RNG.integers(0, len(side_choices)))]
        side_choices = ("top", "bottom")
        return "horizontal", side_choices[int(_RNG.integers(0, len(side_choices)))]

    if theme_kind == "forced_mismatch":
        options = {
            "horizontal": [s for s in ("top", "bottom") if s != start_edge],
            "vertical": [s for s in ("left", "right") if s != start_edge],
        }
        orientations = [ori for ori, sides in options.items() if sides]
        orientation = orientations[int(_RNG.integers(0, len(orientations)))]
        sides = options[orientation]
        return orientation, sides[int(_RNG.integers(0, len(sides)))]

    assert theme_kind == "free"
    orientation = ("horizontal", "vertical")[int(_RNG.integers(0, 2))]
    sides = ("top", "bottom") if orientation == "horizontal" else ("left", "right")
    return orientation, sides[int(_RNG.integers(0, len(sides)))]


def _sample_rectangle(
    quadrant_bounds: tuple[int, int, int, int], contact_side: str | None
) -> tuple[int, int, int, int] | None:
    rlo, rhi, clo, chi = quadrant_bounds
    qh = rhi - rlo + 1
    qw = chi - clo + 1
    max_h = min(5, qh)
    max_w = min(5, qw)
    if max_h < 3 or max_w < 3:
        return None

    for _ in range(200):
        height = int(_RNG.integers(3, max_h + 1))
        width = int(_RNG.integers(3, max_w + 1))
        r_min = rlo + (1 if contact_side == "top" else 0)
        r_max = rhi - height + 1 - (1 if contact_side == "bottom" else 0)
        c_min = clo + (1 if contact_side == "left" else 0)
        c_max = chi - width + 1 - (1 if contact_side == "right" else 0)
        if r_min > r_max or c_min > c_max:
            continue
        r0 = int(_RNG.integers(r_min, r_max + 1))
        c0 = int(_RNG.integers(c_min, c_max + 1))
        return r0, c0, r0 + height - 1, c0 + width - 1
    return None


def _segment_layout(span: int, total: int) -> list[tuple[int, int]] | None:
    if total <= 0 or total > span:
        return None
    layouts: list[list[tuple[int, int]]] = []

    for start in range(span - total + 1):
        layouts.append([(start, start + total - 1)])

    if total >= 2 and span >= total + 1:
        for left_len in range(1, total):
            right_len = total - left_len
            for s1 in range(span):
                e1 = s1 + left_len - 1
                if e1 >= span:
                    break
                min_s2 = e1 + 2
                max_s2 = span - right_len
                for s2 in range(min_s2, max_s2 + 1):
                    layouts.append([(s1, e1), (s2, s2 + right_len - 1)])

    if not layouts:
        return None
    choice = layouts[int(_RNG.integers(0, len(layouts)))]
    return [(int(a), int(b)) for a, b in choice]


def _bar_cells(
    bbox: tuple[int, int, int, int], orientation: str, side: str, count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]] | tuple[None, None]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if orientation == "horizontal":
        spans = _segment_layout(width, count)
        if spans is None:
            return None, None
        row = r0 - 1 if side == "top" else r1 + 1
        cells: list[tuple[int, int]] = []
        seg_boxes: list[tuple[int, int, int, int]] = []
        for s0, s1 in spans:
            seg_boxes.append((row, c0 + s0, row, c0 + s1))
            for c in range(c0 + s0, c0 + s1 + 1):
                cells.append((row, c))
        return cells, seg_boxes

    assert orientation == "vertical"
    spans = _segment_layout(height, count)
    if spans is None:
        return None, None
    col = c0 - 1 if side == "left" else c1 + 1
    cells = []
    seg_boxes = []
    for s0, s1 in spans:
        seg_boxes.append((r0 + s0, col, r0 + s1, col))
        for r in range(r0 + s0, r0 + s1 + 1):
            cells.append((r, col))
    return cells, seg_boxes


def _theme_plan(call_index: int) -> dict[str, dict]:
    phase = call_index % 3
    if phase == 0:
        return {
            "TL": {"active": True, "kind": "anti_orientation"},
            "TR": {"active": True, "kind": "anti_orientation"},
            "BL": {"active": True, "kind": "forced_mismatch"},
            "BR": {"active": True, "kind": "forced_mismatch"},
        }
    if phase == 1:
        inactive = QUADRANT_ORDER[int(_RNG.integers(0, 4))]
        plan = {}
        for quadrant in QUADRANT_ORDER:
            if quadrant == inactive:
                plan[quadrant] = {"active": False, "kind": "free"}
            elif quadrant in {"TL", "TR"}:
                plan[quadrant] = {"active": True, "kind": "match_only"}
            else:
                plan[quadrant] = {"active": True, "kind": "forced_mismatch"}
        return plan

    row_fill_quadrant = ("TL", "BR")[int(_RNG.integers(0, 2))]
    col_fill_quadrant = ("TR", "BL")[int(_RNG.integers(0, 2))]
    plan = {quadrant: {"active": False, "kind": "free"} for quadrant in QUADRANT_ORDER}
    plan[row_fill_quadrant] = {"active": True, "kind": "anti_orientation"}
    plan[col_fill_quadrant] = {"active": True, "kind": "anti_orientation"}
    extra = [q for q in QUADRANT_ORDER if q not in {row_fill_quadrant, col_fill_quadrant}]
    if int(_RNG.integers(0, 2)) == 1:
        q = extra[int(_RNG.integers(0, len(extra)))]
        plan[q] = {"active": True, "kind": ("match_only", "forced_mismatch")[int(_RNG.integers(0, 2))]}
    return plan


def _expected_output_from_info(grid: np.ndarray, rect_info: dict[str, dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GRAY] = BLACK
    for quadrant in QUADRANT_ORDER:
        _paint_stripes(out, quadrant, rect_info[quadrant]["bbox"], rect_info[quadrant]["count"])
    return out


def _signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _metadata_ok(grid: np.ndarray, meta: dict[str, dict], require_mismatch: bool) -> bool:
    try:
        mid_r, mid_c, observed = _scene_info(grid)
    except Exception:
        return False

    if mid_r != grid.shape[0] // 2 or mid_c != grid.shape[1] // 2:
        return False

    has_inactive = False
    mismatch_count = 0
    row_fill_active = 0
    col_fill_active = 0
    horizontal_bars = 0
    vertical_bars = 0

    for quadrant in QUADRANT_ORDER:
        expected = meta[quadrant]
        seen = observed[quadrant]
        if expected["bbox"] != seen["bbox"] or expected["count"] != seen["count"]:
            return False

        r0, c0, r1, c1 = expected["bbox"]
        if quadrant in {"TL", "TR"} and r1 >= mid_r:
            return False
        if quadrant in {"BL", "BR"} and r0 <= mid_r:
            return False
        if quadrant in {"TL", "BL"} and c1 >= mid_c:
            return False
        if quadrant in {"TR", "BR"} and c0 <= mid_c:
            return False

        if expected["count"] == 0:
            has_inactive = True
            if expected["orientation"] is not None or expected["contact_side"] is not None:
                return False
            continue

        if quadrant in {"TL", "BR"}:
            row_fill_active += 1
        else:
            col_fill_active += 1

        if expected["orientation"] == "horizontal":
            horizontal_bars += 1
            if expected["count"] > (c1 - c0 + 1):
                return False
        elif expected["orientation"] == "vertical":
            vertical_bars += 1
            if expected["count"] > (r1 - r0 + 1):
                return False
        else:
            return False

        if expected["contact_side"] != FILL_START_EDGE[quadrant]:
            mismatch_count += 1

    if row_fill_active == 0 or col_fill_active == 0:
        return False
    if horizontal_bars == 0 or vertical_bars == 0:
        return False
    if require_mismatch and mismatch_count == 0:
        return False

    out = main(grid)
    expected_out = _expected_output_from_info(grid, meta)
    if not np.array_equal(out, expected_out):
        return False
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == GRAY) <= np.count_nonzero(grid == GRAY):
        return False
    if not has_inactive and not require_mismatch:
        return False
    return True


def _generate_attempt(call_index: int) -> np.ndarray | None:
    h = int(_RNG.choice(np.array([14, 16, 18, 20], dtype=int)))
    w = int(_RNG.choice(np.array([14, 16, 18, 20], dtype=int)))
    grid = np.full((h, w), BLACK, dtype=int)
    mid_r, mid_c = h // 2, w // 2
    grid[mid_r, :] = YELLOW
    grid[:, mid_c] = YELLOW

    plan = _theme_plan(call_index)
    meta: dict[str, dict] = {}

    for quadrant in QUADRANT_ORDER:
        active = bool(plan[quadrant]["active"])
        orientation, side = _bars_compatible_with(quadrant, active, plan[quadrant]["kind"])
        bbox = _sample_rectangle(_quadrant_bounds((h, w), quadrant), side)
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        count = 0
        bar_cells: list[tuple[int, int]] = []
        if active:
            max_count = min(_fill_capacity(quadrant, height, width), width if orientation == "horizontal" else height)
            if max_count <= 0:
                return None
            count = int(_RNG.integers(1, max_count + 1))
            cells, _segments = _bar_cells(bbox, orientation, side, count)
            if cells is None:
                return None
            bar_cells = cells
            for br, bc in bar_cells:
                if not (0 <= br < h and 0 <= bc < w):
                    return None
                if br == mid_r or bc == mid_c:
                    return None
                if int(grid[br, bc]) != BLACK:
                    return None
                grid[br, bc] = GRAY

        meta[quadrant] = {
            "bbox": bbox,
            "count": count,
            "orientation": orientation,
            "contact_side": side,
            "bar_cells": tuple(bar_cells),
        }

    require_mismatch = (call_index % 3) != 1 or any(
        meta[q]["count"] > 0 and meta[q]["contact_side"] != FILL_START_EDGE[q] for q in QUADRANT_ORDER
    )
    if not _metadata_ok(grid, meta, require_mismatch=require_mismatch):
        return None

    out = main(grid)
    out_sig = _signature(out)
    if out_sig in _RECENT_OUTPUT_SIGNATURES:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(400):
        grid = _generate_attempt(_GEN_CALL_COUNT)
        if grid is None:
            continue
        out_sig = _signature(main(grid))
        _RECENT_OUTPUT_SIGNATURES.add(out_sig)
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid input for quadrant stripe transfer")
