from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GRID_MIN = 20
GRID_MAX = 28
RECT_SIDE_MIN = 5
RECT_SIDE_MAX = 8
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _make_rect(
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap_side: str,
    gap_line: int | None = None,
) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if gap_side == "top":
        if gap_line is None:
            gap_line = random.randint(c0 + 1, c1 - 1)
        gap = (r0, gap_line)
    elif gap_side == "bottom":
        if gap_line is None:
            gap_line = random.randint(c0 + 1, c1 - 1)
        gap = (r1, gap_line)
    elif gap_side == "left":
        if gap_line is None:
            gap_line = random.randint(r0 + 1, r1 - 1)
        gap = (gap_line, c0)
    else:
        if gap_line is None:
            gap_line = random.randint(r0 + 1, r1 - 1)
        gap = (gap_line, c1)
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "gap_side": gap_side,
    }


def _draw_rect(grid: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    gr, gc = rect["gap"]
    grid[gr, gc] = BG


def _extract_grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = grid == WALL
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(coords)
    return components


def _rect_from_component(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3

    wall_cells = set(coords)
    perimeter = _perimeter_cells(bbox)
    assert wall_cells <= perimeter
    missing = sorted(perimeter - wall_cells)
    assert len(missing) == 1
    gap = missing[0]

    if gap[0] == r0:
        gap_side = "top"
    elif gap[0] == r1:
        gap_side = "bottom"
    elif gap[1] == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {"bbox": bbox, "gap": gap, "gap_side": gap_side}


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects = [_rect_from_component(coords) for coords in _extract_grey_components(grid)]
    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1]))
    return rects


def _extract_markers(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    markers = []
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (bg, WALL):
                continue
            assert r in (0, h - 1) or c in (0, w - 1), "non-wall colors must be border markers"
            assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "corner markers are ambiguous"
            if r == 0:
                side = "top"
            elif r == h - 1:
                side = "bottom"
            elif c == 0:
                side = "left"
            else:
                side = "right"
            markers.append({"pos": (r, c), "color": val, "side": side})
    markers.sort(key=lambda marker: marker["pos"])
    return markers


def _gap_entry_map(rects: list[dict]) -> dict[tuple[int, int, int, int], int]:
    entries = {}
    for idx, rect in enumerate(rects):
        gr, gc = rect["gap"]
        dr, dc = DIR_BY_SIDE[rect["gap_side"]]
        entries[(gr, gc, dr, dc)] = idx
    return entries


def _wall_owner_map(rects: list[dict]) -> dict[tuple[int, int], int]:
    owner = {}
    for idx, rect in enumerate(rects):
        for cell in _perimeter_cells(rect["bbox"]):
            if cell != rect["gap"]:
                owner[cell] = idx
    return owner


def _trace_marker(
    grid: np.ndarray,
    marker: dict,
    gap_entries: dict[tuple[int, int, int, int], int],
) -> tuple[str, int | None]:
    (r, c) = marker["pos"]
    dr, dc = DIR_BY_SIDE[marker["side"]]
    rr = r + dr
    cc = c + dc
    while _in_bounds(grid.shape, rr, cc):
        val = int(grid[rr, cc])
        if val == BG:
            hit = gap_entries.get((rr, cc, dr, dc))
            if hit is not None:
                return "hit", hit
            rr += dr
            cc += dc
            continue
        return "blocked", None
    return "miss", None


def _trace_marker_verbose(
    grid: np.ndarray,
    marker_pos: tuple[int, int],
    side: str,
    gap_entries: dict[tuple[int, int, int, int], int],
    wall_owner: dict[tuple[int, int], int],
) -> dict:
    dr, dc = DIR_BY_SIDE[side]
    rr = marker_pos[0] + dr
    cc = marker_pos[1] + dc
    while _in_bounds(grid.shape, rr, cc):
        val = int(grid[rr, cc])
        if val == BG:
            hit = gap_entries.get((rr, cc, dr, dc))
            if hit is not None:
                return {"kind": "hit", "rect": hit}
            rr += dr
            cc += dc
            continue
        blocked_by = wall_owner.get((rr, cc))
        far_hits = set()
        rr2 = rr + dr
        cc2 = cc + dc
        while _in_bounds(grid.shape, rr2, cc2):
            hit = gap_entries.get((rr2, cc2, dr, dc))
            if hit is not None and hit != blocked_by:
                far_hits.add(hit)
            rr2 += dr
            cc2 += dc
        return {"kind": "blocked", "rect": blocked_by, "has_far_gap": bool(far_hits)}
    return {"kind": "miss"}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each border marker's line of sight through a single wall gap to fill one hollow rectangle, then seal every unlit gap.

    Input structure:
    - The background is black.
    - Several disjoint grey hollow rectangles appear away from the border; each rectangle has exactly one one-cell gap in its wall.
    - Additional single-pixel colored markers sit on the outer border and look inward perpendicular to that border side.

    Rule:
    - For every marker, trace its inward ray through background cells only.
    - If the ray first reaches a rectangle by passing exactly through that rectangle's gap, fill the whole interior of that rectangle with the marker color and stop that ray.
    - If a ray hits any wall before entering through a gap, or reaches no rectangle, it does nothing; after all rays, every rectangle that was never entered has its gap sealed with grey.

    Key insight:
    - Alignment alone is insufficient: a marker matters only when its first non-background encounter is a matching gap entry, otherwise a nearer wall blocks it and the untouched rectangle is sealed instead.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG

    rects = _extract_rectangles(g)
    assert rects
    markers = _extract_markers(g, bg)
    assert markers

    gap_entries = _gap_entry_map(rects)
    lit_colors: dict[int, int] = {}
    for marker in markers:
        kind, hit = _trace_marker(g, marker, gap_entries)
        if kind != "hit":
            continue
        assert hit is not None
        prev = lit_colors.get(hit)
        if prev is None:
            lit_colors[hit] = int(marker["color"])
        else:
            assert prev == int(marker["color"]), "multiple markers cannot assign different colors to one rectangle"

    out = g.copy()
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        gr, gc = rect["gap"]
        if idx in lit_colors:
            out[r0 + 1 : r1, c0 + 1 : c1] = lit_colors[idx]
        else:
            out[gr, gc] = WALL
    return out


def _sample_dims() -> tuple[int, int]:
    return random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX), random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)


def _can_place(rects: list[dict], bbox: tuple[int, int, int, int]) -> bool:
    return all(not _boxes_touch(rect["bbox"], bbox, pad=1) for rect in rects)


def _choose_anchor_range(line: int, span: int, limit: int) -> tuple[int, int] | None:
    lo = max(2, line - span + 2)
    hi = min(line - 1, limit - span - 2)
    if lo > hi:
        return None
    return lo, hi


def _sample_rect_on_side(
    h: int,
    w: int,
    shape: tuple[int, int],
    side: str,
    line: int,
    dist: int,
    *,
    gap_side: str | None = None,
) -> dict | None:
    grid_h, grid_w = shape
    if side in ("left", "right"):
        row_range = _choose_anchor_range(line, h, grid_h)
        if row_range is None:
            return None
        r0 = random.randint(*row_range)
        if side == "left":
            c0 = dist
        else:
            c0 = grid_w - dist - w
        if not (2 <= c0 <= grid_w - w - 3):
            return None
    else:
        col_range = _choose_anchor_range(line, w, grid_w)
        if col_range is None:
            return None
        c0 = random.randint(*col_range)
        if side == "top":
            r0 = dist
        else:
            r0 = grid_h - dist - h
        if not (2 <= r0 <= grid_h - h - 3):
            return None
    return _make_rect(r0, c0, h, w, gap_side or side, gap_line=line if (gap_side or side) == side else None)


def _corridor_clear(rects: list[dict], shape: tuple[int, int], side: str, line: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    for rect in rects:
        rr0, cc0, rr1, cc1 = rect["bbox"]
        if side == "left" and rr0 <= line <= rr1 and cc0 < c0:
            return False
        if side == "right" and rr0 <= line <= rr1 and cc1 > c1:
            return False
        if side == "top" and cc0 <= line <= cc1 and rr0 < r0:
            return False
        if side == "bottom" and cc0 <= line <= cc1 and rr1 > r1:
            return False
    return True


def _marker_pos_for_rect(rect: dict, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    gr, gc = rect["gap"]
    if rect["gap_side"] == "top":
        return (0, gc)
    if rect["gap_side"] == "bottom":
        return (h - 1, gc)
    if rect["gap_side"] == "left":
        return (gr, 0)
    return (gr, w - 1)


def _place_blocked_pair(shape: tuple[int, int], rects: list[dict], used_markers: set[tuple[int, int]]) -> tuple[str, int] | None:
    grid_h, grid_w = shape
    for _ in range(120):
        side = random.choice(SIDES)
        line_limit = grid_h if side in ("left", "right") else grid_w
        line = random.randint(3, line_limit - 4)
        bh, bw = _sample_dims()
        ah, aw = _sample_dims()
        near_dist = random.randint(3, 7)
        far_min = near_dist + max(aw if side in ("left", "right") else ah, 2) + 2
        far_max = (grid_w - bw - 3) if side in ("left", "right") else (grid_h - bh - 3)
        if side in ("right", "bottom"):
            far_max = far_max
        if far_min > far_max:
            continue
        far_dist = random.randint(far_min, far_max)
        far_rect = _sample_rect_on_side(bh, bw, shape, side, line, far_dist, gap_side=side)
        if far_rect is None or not _can_place(rects, far_rect["bbox"]):
            continue

        blocker_gap_side = random.choice([s for s in SIDES if s != side])
        blocker_rect = _sample_rect_on_side(ah, aw, shape, side, line, near_dist, gap_side=blocker_gap_side)
        if blocker_rect is None:
            continue
        if not _can_place(rects + [far_rect], blocker_rect["bbox"]):
            continue

        marker_pos = _marker_pos_for_rect(far_rect, shape)
        if marker_pos in used_markers:
            continue

        rects.extend([blocker_rect, far_rect])
        used_markers.add(marker_pos)
        return side, line
    return None


def _place_accessible_rect(
    shape: tuple[int, int],
    rects: list[dict],
    used_markers: set[tuple[int, int]],
) -> dict | None:
    grid_h, grid_w = shape
    for _ in range(200):
        side = random.choice(SIDES)
        line_limit = grid_h if side in ("left", "right") else grid_w
        line = random.randint(3, line_limit - 4)
        h, w = _sample_dims()
        max_dist = (grid_w - w - 3) if side in ("left", "right") else (grid_h - h - 3)
        if max_dist < 3:
            continue
        dist = random.randint(3, max_dist)
        rect = _sample_rect_on_side(h, w, shape, side, line, dist, gap_side=side)
        if rect is None:
            continue
        marker_pos = _marker_pos_for_rect(rect, shape)
        if marker_pos in used_markers:
            continue
        if not _can_place(rects, rect["bbox"]):
            continue
        if not _corridor_clear(rects, shape, side, line, rect["bbox"]):
            continue
        rects.append(rect)
        used_markers.add(marker_pos)
        return rect
    return None


def _build_input(shape: tuple[int, int], rects: list[dict], markers: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _draw_rect(grid, rect)
    for (r, c), color in markers:
        grid[r, c] = int(color)
    return grid


def _expected_output(shape: tuple[int, int], rects: list[dict], markers: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    inp = _build_input(shape, rects, markers)
    gap_entries = _gap_entry_map(rects)
    marker_info = []
    for pos, color in markers:
        r, c = pos
        if r == 0:
            side = "top"
        elif r == shape[0] - 1:
            side = "bottom"
        elif c == 0:
            side = "left"
        else:
            side = "right"
        marker_info.append({"pos": pos, "color": color, "side": side})

    lit: dict[int, int] = {}
    for marker in marker_info:
        kind, hit = _trace_marker(inp, marker, gap_entries)
        if kind == "hit":
            lit[int(hit)] = int(marker["color"])

    out = inp.copy()
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        gr, gc = rect["gap"]
        if idx in lit:
            out[r0 + 1 : r1, c0 + 1 : c1] = lit[idx]
        else:
            out[gr, gc] = WALL
    return out


def _validate_instance(shape: tuple[int, int], rects: list[dict], markers: list[tuple[tuple[int, int], int]]) -> bool:
    inp = _build_input(shape, rects, markers)
    out = main(inp)
    expected = _expected_output(shape, rects, markers)
    if not np.array_equal(out, expected):
        return False

    rects_from_input = _extract_rectangles(inp)
    if len(rects_from_input) != len(rects):
        return False
    markers_from_input = _extract_markers(inp, BG)
    if len(markers_from_input) != len(markers):
        return False

    gap_entries = _gap_entry_map(rects_from_input)
    wall_owner = _wall_owner_map(rects_from_input)
    lit_rects: dict[int, int] = {}
    blocked_with_far_gap = 0
    for marker in markers_from_input:
        verbose = _trace_marker_verbose(inp, marker["pos"], marker["side"], gap_entries, wall_owner)
        if verbose["kind"] == "hit":
            lit_rects[int(verbose["rect"])] = int(marker["color"])
        elif verbose["kind"] == "blocked" and bool(verbose["has_far_gap"]):
            blocked_with_far_gap += 1

    if not lit_rects:
        return False
    if blocked_with_far_gap < 1:
        return False
    if len(lit_rects) >= len(rects_from_input):
        return False

    for idx, rect in enumerate(rects_from_input):
        r0, c0, r1, c1 = rect["bbox"]
        gr, gc = rect["gap"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        if idx in lit_rects:
            if inner.size == 0 or not np.all(inner == lit_rects[idx]):
                return False
            if int(out[gr, gc]) != BG:
                return False
        else:
            if inner.size == 0 or not np.all(inner == BG):
                return False
            if int(out[gr, gc]) != WALL:
                return False

    for marker in markers_from_input:
        r, c = marker["pos"]
        if int(out[r, c]) != int(marker["color"]):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    rects: list[dict] = []
    used_markers: set[tuple[int, int]] = set()

    blocked_info = _place_blocked_pair(shape, rects, used_markers)
    if blocked_info is None:
        return None

    lit_count = random.randint(1, 2)
    lit_rects = []
    for _ in range(lit_count):
        rect = _place_accessible_rect(shape, rects, used_markers)
        if rect is None:
            return None
        lit_rects.append(rect)

    marker_count = 1 + len(lit_rects)
    colors = random.sample(MARKER_COLORS, marker_count)
    markers: list[tuple[tuple[int, int], int]] = []

    blocked_marker_pos = _marker_pos_for_rect(rects[1], shape)
    markers.append((blocked_marker_pos, colors[0]))
    for idx, rect in enumerate(lit_rects, start=1):
        markers.append((_marker_pos_for_rect(rect, shape), colors[idx]))

    if not _validate_instance(shape, rects, markers):
        return None
    return _build_input(shape, rects, markers)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
