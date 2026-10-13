from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5


SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _frame_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    grey_rows = [r for r in range(grid.shape[0]) if int(np.count_nonzero(grid[r] == GREY)) > 4]
    grey_cols = [c for c in range(grid.shape[1]) if int(np.count_nonzero(grid[:, c] == GREY)) > 4]
    assert len(grey_rows) == 4 and len(grey_cols) == 4
    row_steps = [grey_rows[i + 1] - grey_rows[i] for i in range(3)]
    col_steps = [grey_cols[i + 1] - grey_cols[i] for i in range(3)]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1
    assert row_steps[0] >= 2 and col_steps[0] >= 2
    return grey_rows, grey_cols


def _slot_bounds(rows: list[int], cols: list[int], sr: int, sc: int) -> tuple[int, int, int, int]:
    return rows[sr] + 1, cols[sc] + 1, rows[sr + 1] - 1, cols[sc + 1] - 1


def _slot_index(rows: list[int], cols: list[int], coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    for sr in range(3):
        for sc in range(3):
            rr0, cc0, rr1, cc1 = _slot_bounds(rows, cols, sr, sc)
            if rr0 <= r0 <= r1 <= rr1 and cc0 <= c0 <= c1 <= cc1:
                return sr, sc
    raise AssertionError("sprite component does not fit exactly one slot")


def _parse_slots(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], dict[int, tuple[tuple[int, int], ...]], list[int], list[int]]:
    rows, cols = _frame_lines(grid)

    comps: list[dict] = []
    proto_map: dict[int, set[tuple[tuple[int, int], ...]]] = {BLUE: set(), RED: set()}
    for color in (BLUE, RED):
        for comp in _components_of_color(grid, color):
            slot = _slot_index(rows, cols, comp)
            assert slot not in {item["slot"] for item in comps}
            r0, c0, _r1, _c1 = _bbox_coords(comp)
            key = _shape_key(comp)
            proto_map[color].add(key)
            comps.append(
                {
                    "slot": slot,
                    "color": int(color),
                    "coords": comp,
                    "top_left": (int(r0), int(c0)),
                    "shape": key,
                    "rows": {int(r) for r, _c in comp},
                    "cols": {int(c) for _r, c in comp},
                }
            )

    assert len(comps) == 9
    assert len(proto_map[BLUE]) == 1 and len(proto_map[RED]) == 1
    blue_shape = next(iter(proto_map[BLUE]))
    red_shape = next(iter(proto_map[RED]))
    assert blue_shape != red_shape

    slot_map = {item["slot"]: item for item in comps}
    assert len(slot_map) == 9
    return slot_map, {BLUE: blue_shape, RED: red_shape}, rows, cols


def _marker_target(rows: list[int], cols: list[int], r: int, c: int) -> tuple[tuple[int, int], str]:
    if c == cols[0] - 1 and rows[0] < r < rows[-1]:
        sr = next(i for i in range(3) if rows[i] < r < rows[i + 1])
        return (sr, 0), "row"
    if c == cols[-1] + 1 and rows[0] < r < rows[-1]:
        sr = next(i for i in range(3) if rows[i] < r < rows[i + 1])
        return (sr, 2), "row"
    if r == rows[0] - 1 and cols[0] < c < cols[-1]:
        sc = next(i for i in range(3) if cols[i] < c < cols[i + 1])
        return (0, sc), "col"
    if r == rows[-1] + 1 and cols[0] < c < cols[-1]:
        sc = next(i for i in range(3) if cols[i] < c < cols[i + 1])
        return (2, sc), "col"
    raise AssertionError("marker is not adjacent to a valid outer slot side")


def _count_hits_from_grid(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    slot_map, _protos, rows, cols = _parse_slots(grid)
    hits = {slot: 0 for slot in slot_map}
    axis_hits = {slot: [0, 0] for slot in slot_map}
    for r, c in np.argwhere(grid == GREEN):
        slot, mode = _marker_target(rows, cols, int(r), int(c))
        sprite = slot_map[slot]
        if mode == "row":
            if int(r) in sprite["rows"]:
                hits[slot] += 1
                axis_hits[slot][0] += 1
        else:
            if int(c) in sprite["cols"]:
                hits[slot] += 1
                axis_hits[slot][1] += 1
    return hits, {slot: (vals[0], vals[1]) for slot, vals in axis_hits.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip edge-reachable slot sprites to the opposite prototype when side-ray hits have odd parity.

    Input structure:
    - A black background contains a grey single-cell 3x3 frame grid with 9 equal rectangular slots.
    - Each slot contains one 4-connected monochrome sprite in blue or red; all blue sprites share one normalized prototype and all red sprites share another.
    - Green single-pixel side markers sit exactly one cell outside the outer frame, aligned to slot interior rows or columns.

    Rule:
    - A left/right marker sends a horizontal ray through the adjacent edge slot; a top/bottom marker sends a vertical ray through the adjacent edge slot.
    - Count one hit for a marker if its ray crosses any pixel of that slot's sprite; sum all hits per sprite.
    - Odd total hits swap the sprite to the opposite color/prototype, keeping the original sprite bbox top-left fixed; even totals keep it unchanged.
    - Background, frame lines, and markers remain unchanged.

    Key insight:
    - The frame identifies both the slot partition and the ray stop boundaries, so hit parity is recovered by matching marker rows or columns against the occupied rows or columns of each edge-slot sprite.
    """
    g = np.asarray(grid, dtype=int)
    slot_map, protos, _rows, _cols = _parse_slots(g)
    hits, _axis_hits = _count_hits_from_grid(g)

    out = g.copy()
    for slot, sprite in slot_map.items():
        if hits[slot] % 2 == 0:
            continue
        color = int(sprite["color"])
        other = RED if color == BLUE else BLUE
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = BG
        top, left = sprite["top_left"]
        for dr, dc in protos[other]:
            rr = int(top + dr)
            cc = int(left + dc)
            assert out[rr, cc] in (BG, color)
            out[rr, cc] = other
    return out


def _draw_frame(slot_h: int, slot_w: int) -> tuple[np.ndarray, list[int], list[int]]:
    h = 3 * slot_h + 6
    w = 3 * slot_w + 6
    rows = [1, slot_h + 2, 2 * slot_h + 3, 3 * slot_h + 4]
    cols = [1, slot_w + 2, 2 * slot_w + 3, 3 * slot_w + 4]
    grid = np.full((h, w), BG, dtype=int)
    for r in rows:
        grid[r, cols[0] : cols[-1] + 1] = GREY
    for c in cols:
        grid[rows[0] : rows[-1] + 1, c] = GREY
    return grid, rows, cols


def _paint_shape(grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    pts = []
    for dr, dc in shape:
        rr = int(top + dr)
        cc = int(left + dc)
        grid[rr, cc] = int(color)
        pts.append((rr, cc))
    return np.array(pts, dtype=int)


def _expected_output(
    grid: np.ndarray,
    slot_layout: dict[tuple[int, int], dict],
    protos: dict[int, tuple[tuple[int, int], ...]],
    hits: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for slot, rec in slot_layout.items():
        if hits[slot] % 2 == 0:
            continue
        color = int(rec["color"])
        other = RED if color == BLUE else BLUE
        for r, c in rec["coords"]:
            out[int(r), int(c)] = BG
        top, left = rec["top_left"]
        for dr, dc in protos[other]:
            out[int(top + dr), int(left + dc)] = other
    return out


def _sample_marker_set(values: list[int], allow_zero: bool = False) -> set[int]:
    max_take = min(5, len(values))
    min_take = 0 if allow_zero else 1
    take = random.randint(min_take, max_take)
    if take == 0:
        return set()
    return set(random.sample(values, take))


def _truth_hits(
    slot_layout: dict[tuple[int, int], dict],
    left_rows: set[int],
    right_rows: set[int],
    top_cols: set[int],
    bottom_cols: set[int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    hits = {slot: 0 for slot in slot_layout}
    axis_hits = {slot: [0, 0] for slot in slot_layout}
    for slot, rec in slot_layout.items():
        sr, sc = slot
        if sc == 0:
            for r in left_rows:
                if r in rec["rows"]:
                    hits[slot] += 1
                    axis_hits[slot][0] += 1
        if sc == 2:
            for r in right_rows:
                if r in rec["rows"]:
                    hits[slot] += 1
                    axis_hits[slot][0] += 1
        if sr == 0:
            for c in top_cols:
                if c in rec["cols"]:
                    hits[slot] += 1
                    axis_hits[slot][1] += 1
        if sr == 2:
            for c in bottom_cols:
                if c in rec["cols"]:
                    hits[slot] += 1
                    axis_hits[slot][1] += 1
    return hits, {slot: (vals[0], vals[1]) for slot, vals in axis_hits.items()}


def _naive_any_hit_output(
    grid: np.ndarray,
    slot_layout: dict[tuple[int, int], dict],
    protos: dict[int, tuple[tuple[int, int], ...]],
    hits: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for slot, rec in slot_layout.items():
        if hits[slot] == 0:
            continue
        color = int(rec["color"])
        other = RED if color == BLUE else BLUE
        for r, c in rec["coords"]:
            out[int(r), int(c)] = BG
        top, left = rec["top_left"]
        for dr, dc in protos[other]:
            out[int(top + dr), int(left + dc)] = other
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(500):
        slot_h = random.randint(4, 6)
        slot_w = random.randint(4, 6)
        grid, rows, cols = _draw_frame(slot_h, slot_w)

        candidates = [shape for shape in SHAPES if _shape_dims(shape)[0] <= slot_h and _shape_dims(shape)[1] <= slot_w]
        shape_blue, shape_red = random.sample(candidates, 2)
        max_h = max(_shape_dims(shape_blue)[0], _shape_dims(shape_red)[0])
        max_w = max(_shape_dims(shape_blue)[1], _shape_dims(shape_red)[1])

        blue_count = random.randint(3, 6)
        slot_order = [(sr, sc) for sr in range(3) for sc in range(3)]
        blue_slots = set(random.sample(slot_order, blue_count))

        slot_layout: dict[tuple[int, int], dict] = {}
        for sr, sc in slot_order:
            rr0, cc0, rr1, cc1 = _slot_bounds(rows, cols, sr, sc)
            top = random.randint(rr0, rr1 - max_h + 1)
            left = random.randint(cc0, cc1 - max_w + 1)
            color = BLUE if (sr, sc) in blue_slots else RED
            shape = shape_blue if color == BLUE else shape_red
            coords = _paint_shape(grid, top, left, shape, color)
            slot_layout[(sr, sc)] = {
                "color": int(color),
                "shape": shape,
                "coords": coords,
                "top_left": (int(top), int(left)),
                "rows": {int(r) for r, _c in coords},
                "cols": {int(c) for _r, c in coords},
            }

        usable_left = [r for r in range(rows[0] + 1, rows[-1]) if r not in rows]
        usable_right = [r for r in range(rows[0] + 1, rows[-1]) if r not in rows]
        usable_top = [c for c in range(cols[0] + 1, cols[-1]) if c not in cols]
        usable_bottom = [c for c in range(cols[0] + 1, cols[-1]) if c not in cols]

        marker_spec = None
        for _marker_attempt in range(300):
            left_rows = _sample_marker_set(usable_left, allow_zero=True)
            right_rows = _sample_marker_set(usable_right, allow_zero=True)
            top_cols = _sample_marker_set(usable_top, allow_zero=True)
            bottom_cols = _sample_marker_set(usable_bottom, allow_zero=True)
            used_sides = sum(bool(x) for x in (left_rows, right_rows, top_cols, bottom_cols))
            total_markers = len(left_rows) + len(right_rows) + len(top_cols) + len(bottom_cols)
            if used_sides < 3 or total_markers < 5 or total_markers > 14:
                continue

            hits, axis_hits = _truth_hits(slot_layout, left_rows, right_rows, top_cols, bottom_cols)
            odd_slots = [slot for slot, cnt in hits.items() if cnt % 2 == 1]
            even_positive = [slot for slot, cnt in hits.items() if cnt > 0 and cnt % 2 == 0]
            if len(odd_slots) < 2 or len(even_positive) < 2:
                continue
            if max(hits.values()) < 2:
                continue
            if not any(horiz > 0 and vert > 0 for slot, (horiz, vert) in axis_hits.items() if slot in ((0, 0), (0, 2), (2, 0), (2, 2))):
                continue
            if hits[(1, 1)] != 0:
                continue

            trial = grid.copy()
            trial[list(left_rows), cols[0] - 1] = GREEN
            trial[list(right_rows), cols[-1] + 1] = GREEN
            trial[rows[0] - 1, list(top_cols)] = GREEN
            trial[rows[-1] + 1, list(bottom_cols)] = GREEN

            expected = _expected_output(
                trial,
                slot_layout,
                {BLUE: tuple(sorted(shape_blue)), RED: tuple(sorted(shape_red))},
                hits,
            )
            if np.array_equal(expected, trial):
                continue
            if np.array_equal(expected, _naive_any_hit_output(trial, slot_layout, {BLUE: tuple(sorted(shape_blue)), RED: tuple(sorted(shape_red))}, hits)):
                continue

            solved = main(trial)
            parsed_hits, parsed_axis_hits = _count_hits_from_grid(trial)
            if parsed_hits != hits or parsed_axis_hits != axis_hits:
                continue
            if not np.array_equal(solved, expected):
                continue

            marker_spec = (left_rows, right_rows, top_cols, bottom_cols)
            grid = trial
            break

        if marker_spec is None:
            continue

        slot_map, protos, parsed_rows, parsed_cols = _parse_slots(grid)
        if parsed_rows != rows or parsed_cols != cols:
            continue
        blue_seen = sum(1 for rec in slot_map.values() if rec["color"] == BLUE)
        red_seen = sum(1 for rec in slot_map.values() if rec["color"] == RED)
        if blue_seen < 3 or red_seen < 3:
            continue
        if protos[BLUE] == protos[RED]:
            continue

        return grid

    raise RuntimeError("failed to generate a valid ray-parity puzzle")
