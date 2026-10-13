from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 3
MARKER_COLORS = (1, 2, 4, 6, 7, 9)
GRID_HEIGHT_RANGE = (14, 19)
GRID_WIDTH_RANGE = (14, 19)
MARKER_COUNT_RANGE = (4, 6)
GENERATION_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    points = set(coords)
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for start in coords:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in points and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    points = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not points:
        return False
    seen = {points[0]}
    queue = deque([points[0]])
    point_set = set(points)
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in point_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(points)


def _build_pattern_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {"horizontal": [], "vertical": [], "both": [], "none": []}
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not (3 <= area <= 7):
            continue
        if area == 9 or not _mask_connected(mask):
            continue
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        horizontal = bool(np.array_equal(mask[0], mask[2]))
        vertical = bool(np.array_equal(mask[:, 0], mask[:, 2]))
        if horizontal and vertical:
            catalog["both"].append(mask.copy())
        elif horizontal:
            catalog["horizontal"].append(mask.copy())
        elif vertical:
            catalog["vertical"].append(mask.copy())
        else:
            catalog["none"].append(mask.copy())
    return catalog


PATTERN_CATALOG = _build_pattern_catalog()


def _anchor_adjacent_endpoint(
    r: int, c: int, anchor_bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = anchor_bbox
    if r0 <= r <= r1:
        if c < c0:
            return r, c0 - 1
        if c > c1:
            return r, c1 + 1
    if c0 <= c <= c1:
        if r < r0:
            return r0 - 1, c
        if r > r1:
            return r1 + 1, c
    return None


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _paint_line(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gate each aligned marker's ray extension by the mirror symmetry type of its nearby 3x3 same-color pattern.

    Input structure:
    - The grid is mostly black and contains exactly one green 2x2 anchor square.
    - Every other used color appears in exactly two monochrome components: one singleton marker and one nearby 3x3 pattern of the same color.
    - Each marker shares exactly one anchor row or one anchor column, while each 3x3 pattern is a connected non-full shape on black background.

    Rule:
    - Pair each marker with the unique larger component of the same color and read that component inside its 3x3 bounding box.
    - A row-aligned marker is allowed to fire only if its pattern is horizontally symmetric; a column-aligned marker is allowed to fire only if its pattern is vertically symmetric.
    - Allowed markers keep their pixel and extend a straight ray toward the anchor up to the cell adjacent to it; rejected markers are erased, and every 3x3 pattern is erased in all cases.

    Key insight:
    - Symmetry is not the output itself: it is a gate whose accepted symmetry axis must match the marker's alignment direction before the usual anchor-directed ray is drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    anchor_components = _find_color_components(g, ANCHOR_COLOR)
    anchors = [
        comp
        for comp in anchor_components
        if len(comp) == 4 and _bbox(comp)[2] - _bbox(comp)[0] == 1 and _bbox(comp)[3] - _bbox(comp)[1] == 1
    ]
    assert len(anchors) == 1, "expected exactly one 2x2 green anchor"
    anchor_bbox = _bbox(anchors[0])
    ar0, ac0, ar1, ac1 = anchor_bbox

    out = g.copy()
    active_colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BACKGROUND, ANCHOR_COLOR))
    for color in active_colors:
        comps = _find_color_components(g, color)
        assert len(comps) == 2, f"color {color} must have one marker and one pattern"
        comps.sort(key=len)
        marker_comp, pattern_comp = comps
        assert len(marker_comp) == 1, f"color {color} marker must be a singleton"

        pr0, pc0, pr1, pc1 = _bbox(pattern_comp)
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (3, 3), f"color {color} pattern must occupy a 3x3 box"
        pattern_mask = np.zeros((3, 3), dtype=bool)
        for r, c in pattern_comp:
            pattern_mask[r - pr0, c - pc0] = True

        horizontal = bool(np.array_equal(pattern_mask[0], pattern_mask[2]))
        vertical = bool(np.array_equal(pattern_mask[:, 0], pattern_mask[:, 2]))

        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BACKGROUND

        mr, mc = marker_comp[0]
        row_aligned = ar0 <= mr <= ar1
        col_aligned = ac0 <= mc <= ac1
        assert row_aligned ^ col_aligned, f"marker {color} must align by exactly one axis"

        qualified = horizontal if row_aligned else vertical
        if not qualified:
            out[mr, mc] = BACKGROUND
            continue

        endpoint = _anchor_adjacent_endpoint(mr, mc, anchor_bbox)
        assert endpoint is not None, f"marker {color} must point toward the anchor"
        _paint_line(out, _line_cells(mr, mc, endpoint[0], endpoint[1]), color)

    return out


def _choose_anchor(H: int, W: int) -> tuple[int, int]:
    row_choices = list(range(4, H - 5))
    assert row_choices
    ar = random.choice(row_choices)

    left_band = [c for c in range(max(4, W // 2 - 3), min(W - 5, W // 2))]
    right_band = [c for c in range(max(4, W // 2 + 1), min(W - 5, W // 2 + 4))]
    bands = [band for band in (left_band, right_band) if band]
    assert bands
    ac = random.choice(random.choice(bands))
    return ar, ac


def _slot_pool(anchor_bbox: tuple[int, int, int, int]) -> tuple[list[dict], list[dict]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    row_slots = [
        {"kind": "row", "side": "left", "line": ar0},
        {"kind": "row", "side": "right", "line": ar0},
        {"kind": "row", "side": "left", "line": ar1},
        {"kind": "row", "side": "right", "line": ar1},
    ]
    col_slots = [
        {"kind": "col", "side": "top", "line": ac0},
        {"kind": "col", "side": "bottom", "line": ac0},
        {"kind": "col", "side": "top", "line": ac1},
        {"kind": "col", "side": "bottom", "line": ac1},
    ]
    return row_slots, col_slots


def _slot_positions(
    slot: dict, anchor_bbox: tuple[int, int, int, int], H: int, W: int
) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    positions: list[tuple[int, int]] = []
    if slot["kind"] == "row":
        r = int(slot["line"])
        if slot["side"] == "left":
            positions = [(r, c) for c in range(0, ac0 - 1)]
        else:
            positions = [(r, c) for c in range(ac1 + 2, W)]
    else:
        c = int(slot["line"])
        if slot["side"] == "top":
            positions = [(r, c) for r in range(0, ar0 - 1)]
        else:
            positions = [(r, c) for r in range(ar1 + 2, H)]

    positions.sort(key=lambda rc: abs(rc[0] - (ar0 + ar1) / 2.0) + abs(rc[1] - (ac0 + ac1) / 2.0))
    random.shuffle(positions)
    return positions


def _choose_slot_outcomes(count: int) -> list[bool]:
    outcomes = [True, False]
    outcomes.extend(random.choice([True, False]) for _ in range(count - 2))
    random.shuffle(outcomes)
    return outcomes


def _choose_pattern_kind(kind: str, passed: bool) -> str:
    if kind == "row":
        choices = (["horizontal"] * 3 + ["both"]) if passed else (["vertical"] * 3 + ["none"])
    else:
        choices = (["vertical"] * 3 + ["both"]) if passed else (["horizontal"] * 3 + ["none"])
    return random.choice(choices)


def _marker_box_distance(marker: tuple[int, int], top_left: tuple[int, int]) -> int:
    mr, mc = marker
    tr, tc = top_left
    distances = [
        abs(mr - r) + abs(mc - c)
        for r in range(tr, tr + 3)
        for c in range(tc, tc + 3)
    ]
    return min(distances)


def _pattern_score(
    slot: dict, marker: tuple[int, int], top_left: tuple[int, int]
) -> tuple[int, int, int]:
    mr, mc = marker
    tr, tc = top_left
    rows = range(tr, tr + 3)
    cols = range(tc, tc + 3)
    outward = 0
    avoids_axis = 0
    if slot["kind"] == "row":
        if slot["side"] == "left" and tc + 2 <= mc - 2:
            outward = 1
        if slot["side"] == "right" and tc >= mc + 2:
            outward = 1
        avoids_axis = int(mr not in rows)
    else:
        if slot["side"] == "top" and tr + 2 <= mr - 2:
            outward = 1
        if slot["side"] == "bottom" and tr >= mr + 2:
            outward = 1
        avoids_axis = int(mc not in cols)
    center_bias = -abs((tr + 1) - mr) - abs((tc + 1) - mc)
    return outward, avoids_axis, center_bias


def _choose_pattern_box(
    slot: dict,
    marker: tuple[int, int],
    H: int,
    W: int,
    reserved: np.ndarray,
    path_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for tr in range(H - 2):
        for tc in range(W - 2):
            box_cells = {(r, c) for r in range(tr, tr + 3) for c in range(tc, tc + 3)}
            if reserved[tr : tr + 3, tc : tc + 3].any():
                continue
            if box_cells & path_cells:
                continue
            if _marker_box_distance(marker, (tr, tc)) != 2:
                continue
            score = _pattern_score(slot, marker, (tr, tc))
            candidates.append((score, (tr, tc)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    best = [pos for score, pos in candidates if score == best_score]
    return random.choice(best)


def _choose_markers(
    H: int, W: int, anchor_bbox: tuple[int, int, int, int], n_markers: int
) -> list[dict] | None:
    row_slots, col_slots = _slot_pool(anchor_bbox)
    max_rows = min(4, n_markers - 2)
    row_count = random.randint(2, max_rows)
    col_count = n_markers - row_count
    if not (2 <= col_count <= 4):
        return None

    selected_slots = random.sample(row_slots, row_count) + random.sample(col_slots, col_count)
    colors = random.sample(list(MARKER_COLORS), n_markers)
    random.shuffle(selected_slots)

    row_outcomes = _choose_slot_outcomes(row_count)
    col_outcomes = _choose_slot_outcomes(col_count)
    random.shuffle(colors)

    occupied = np.zeros((H, W), dtype=bool)
    ar0, ac0, ar1, ac1 = anchor_bbox
    occupied[ar0 : ar1 + 1, ac0 : ac1 + 1] = True

    row_i = 0
    col_i = 0
    records: list[dict] = []
    for slot, color in zip(selected_slots, colors):
        positions = _slot_positions(slot, anchor_bbox, H, W)
        chosen = None
        for marker in positions:
            mr, mc = marker
            if occupied[mr, mc]:
                continue
            chosen = marker
            break
        if chosen is None:
            return None
        occupied[chosen] = True
        passed = row_outcomes[row_i] if slot["kind"] == "row" else col_outcomes[col_i]
        if slot["kind"] == "row":
            row_i += 1
        else:
            col_i += 1
        records.append(
            {
                "color": int(color),
                "slot": slot,
                "marker": chosen,
                "passes": bool(passed),
                "pattern_kind": _choose_pattern_kind(slot["kind"], bool(passed)),
            }
        )
    return records


def _expected_output(
    grid: np.ndarray,
    records: list[dict],
    anchor_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        tr, tc = rec["pattern_top_left"]
        out[tr : tr + 3, tc : tc + 3] = BACKGROUND
        mr, mc = rec["marker"]
        if not rec["passes"]:
            out[mr, mc] = BACKGROUND
            continue
        endpoint = _anchor_adjacent_endpoint(mr, mc, anchor_bbox)
        assert endpoint is not None
        _paint_line(out, _line_cells(mr, mc, endpoint[0], endpoint[1]), rec["color"])
    return out


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((H, W), dtype=int)

    ar, ac = _choose_anchor(H, W)
    anchor_bbox = (ar, ac, ar + 1, ac + 1)
    grid[ar : ar + 2, ac : ac + 2] = ANCHOR_COLOR

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    records = _choose_markers(H, W, anchor_bbox, n_markers)
    if records is None:
        return None

    path_cells: set[tuple[int, int]] = set()
    for rec in records:
        mr, mc = rec["marker"]
        endpoint = _anchor_adjacent_endpoint(mr, mc, anchor_bbox)
        if endpoint is None:
            return None
        rec["endpoint"] = endpoint
        rec["path_cells"] = _line_cells(mr, mc, endpoint[0], endpoint[1])
        path_cells.update(rec["path_cells"][1:])
        grid[mr, mc] = rec["color"]

    reserved = np.zeros((H, W), dtype=bool)
    reserved[ar : ar + 2, ac : ac + 2] = True
    for rec in records:
        mr, mc = rec["marker"]
        reserved[mr, mc] = True

    for rec in records:
        box = _choose_pattern_box(rec["slot"], rec["marker"], H, W, reserved, path_cells)
        if box is None:
            return None
        tr, tc = box
        rec["pattern_top_left"] = box
        reserved[tr : tr + 3, tc : tc + 3] = True
        mask = random.choice(PATTERN_CATALOG[rec["pattern_kind"]])
        rec["pattern_mask"] = mask
        for rr in range(3):
            for cc in range(3):
                if mask[rr, cc]:
                    grid[tr + rr, tc + cc] = rec["color"]

    row_pass = sum(rec["slot"]["kind"] == "row" and rec["passes"] for rec in records)
    row_fail = sum(rec["slot"]["kind"] == "row" and not rec["passes"] for rec in records)
    col_pass = sum(rec["slot"]["kind"] == "col" and rec["passes"] for rec in records)
    col_fail = sum(rec["slot"]["kind"] == "col" and not rec["passes"] for rec in records)
    if not all(count >= 1 for count in (row_pass, row_fail, col_pass, col_fail)):
        return None

    expected = _expected_output(grid, records, anchor_bbox)
    try:
        observed = main(grid)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    if np.array_equal(observed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-gated ray input")
