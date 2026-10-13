from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, in_bounds, neighbors4


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
PINK = Color.PINK

GEN_ATTEMPTS = 500
GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (16, 25)
TRACK_STEP_CHOICES = (2, 3)
TRACK_SLOT_RANGE = (6, 10)

OBJECT_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),
)

DECOR_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
)


def _first_coord(coords: set[tuple[int, int]]) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in coords)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift_coords(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _transpose_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(c), int(r)) for r, c in coords}


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == PINK)
    assert pts.shape == (1, 2), "expected one pink corner marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return r, c


def _blue_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    structural = np.zeros_like(grid, dtype=bool)
    decorative = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape
    for comp in components_mask(grid == BLUE, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp}
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)
        mask = structural if touches_border else decorative
        for r, c in cells:
            mask[r, c] = True
    return structural, decorative


def _flood_background(grid: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    mr, mc = marker
    for nr, nc in neighbors4(mr, mc):
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK and not reach[nr, nc]:
            reach[nr, nc] = True
            q.append((nr, nc))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if reach[nr, nc] or int(grid[nr, nc]) != BLACK:
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return reach


def _parse_track_from_grid(
    grid: np.ndarray, red_coords: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], str]:
    greens = np.argwhere(grid == GREEN)
    assert greens.shape[0] >= 2, "expected visible green track points"

    rows = sorted({int(r) for r in greens[:, 0]})
    cols = sorted({int(c) for c in greens[:, 1]})
    anchor_r, anchor_c = _first_coord(red_coords)

    if len(rows) == 1:
        axis = "h"
        line = rows[0]
        assert anchor_r == line, "first red pixel must lie on the track row"
        positions = sorted({int(c) for c in greens[:, 1]} | {anchor_c})
        mk = lambda p: (line, p)
    elif len(cols) == 1:
        axis = "v"
        line = cols[0]
        assert anchor_c == line, "first red pixel must lie on the track column"
        positions = sorted({int(r) for r in greens[:, 0]} | {anchor_r})
        mk = lambda p: (p, line)
    else:
        raise AssertionError("track must be horizontal or vertical")

    assert len(positions) >= 3, "need enough track slots to infer spacing"
    step = 0
    for a, b in zip(positions, positions[1:]):
        diff = int(b - a)
        assert diff > 0
        step = diff if step == 0 else math.gcd(step, diff)
    assert step > 0, "track spacing must be positive"

    start = positions[0]
    end = positions[-1]
    assert all((p - start) % step == 0 for p in positions)
    full_positions = list(range(start, end + 1, step))
    full_slots = [mk(pos) for pos in full_positions]
    return full_slots, axis


def _effective_track_slots(
    grid_shape: tuple[int, int],
    marker: tuple[int, int],
    reach: np.ndarray,
    track_slots: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in track_slots:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid_shape, nr, nc):
                continue
            if (nr, nc) == marker or bool(reach[nr, nc]):
                out.add((int(r), int(c)))
                break
    return out


def _move_delta(
    track_slots: list[tuple[int, int]],
    effective_slots: set[tuple[int, int]],
    red_coords: set[tuple[int, int]],
) -> tuple[int, int]:
    anchor_r, anchor_c = _first_coord(red_coords)
    if len({r for r, _ in track_slots}) == 1:
        anchor_pos = anchor_c
        positions = sorted(c for _r, c in effective_slots)
        before = sum(pos < anchor_pos for pos in positions)
        after = sum(pos > anchor_pos for pos in positions)
        direction = 1 if after >= before else -1
        candidates = [pos for pos in positions if direction * (pos - anchor_pos) > 0]
        assert candidates, "no effective slot to move into"
        next_pos = min(candidates) if direction == 1 else max(candidates)
        return 0, int(next_pos - anchor_pos)

    anchor_pos = anchor_r
    positions = sorted(r for r, _c in effective_slots)
    before = sum(pos < anchor_pos for pos in positions)
    after = sum(pos > anchor_pos for pos in positions)
    direction = 1 if after >= before else -1
    candidates = [pos for pos in positions if direction * (pos - anchor_pos) > 0]
    assert candidates, "no effective slot to move into"
    next_pos = min(candidates) if direction == 1 else max(candidates)
    return int(next_pos - anchor_pos), 0


def _render_output(
    shape: tuple[int, int],
    structural: np.ndarray,
    track_slots: list[tuple[int, int]],
    effective_slots: set[tuple[int, int]],
    red_coords: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    out[structural] = BLUE

    old_hidden = {slot for slot in track_slots if slot in red_coords}
    new_red = _shift_coords(red_coords, dr, dc)
    for r, c in new_red:
        assert 0 <= r < h and 0 <= c < w, "shifted object moved out of bounds"
        assert not structural[r, c], "shifted object hit a structural wall"

    new_hidden = {slot for slot in effective_slots if slot in new_red}
    for slot in effective_slots:
        if slot not in old_hidden and slot not in new_hidden:
            out[slot] = GREEN
    for r, c in new_red:
        out[r, c] = RED
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker = _find_marker(g)
    structural, _decorative = _blue_masks(g)

    red_components = components_mask(g == RED, connectivity=4)
    assert len(red_components) == 1, "expected one red foreground object"
    red_coords = {(int(r), int(c)) for r, c in red_components[0]}

    reach = _flood_background(g, marker)
    track_slots, _axis = _parse_track_from_grid(g, red_coords)
    effective_slots = _effective_track_slots(g.shape, marker, reach, track_slots)
    assert len(effective_slots) >= 3, "need at least three effective track slots"

    dr, dc = _move_delta(track_slots, effective_slots, red_coords)
    return _render_output(g.shape, structural, track_slots, effective_slots, red_coords, dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the dotted rail by corner reachability, erase decorative walls, then slide the red object one valid slot.

    Input structure:
    - Black background with blue wall pixels, one pink corner marker, one straight green dotted track, and one red 4-connected object.
    - Blue wall components that touch the grid border are structural; border-disconnected blue fragments are decorative.
    - The green track lies on one row or one column with equal spacing, and the red object hides at least its anchor slot on that track.

    Rule:
    - Flood-fill only black background cells starting from the pink corner marker, with every non-black cell acting as an obstacle.
    - Erase every blue component that does not touch the border, and keep only track slots that border the flood-reachable background (or the marker itself).
    - Infer the full equal-spacing track including the slot hidden under the row-major first red pixel.
    - Count effective track slots on both sides of that anchor slot; move toward the side with more effective slots, breaking ties rightward on horizontal tracks and downward on vertical tracks.
    - Translate the full red object so that its first pixel lands on the next effective slot, erase the old red footprint to black, and erase the pink marker.

    Key insight:
    - Reachability is not used for painting a region; it acts as a boolean filter that decides which dotted slots still belong to the usable rail segment.
    """
    return _solve(np.asarray(grid, dtype=int))


def _shape_to_coords(anchor: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc in shape}


def _classify_anchor(positions: list[int], anchor_pos: int) -> str | None:
    before = sum(pos < anchor_pos for pos in positions)
    after = sum(pos > anchor_pos for pos in positions)
    if after == 0 and before == 0:
        return None
    if after == before:
        return "tie"
    return "pos" if after > before else "neg"


def _case_input(case: dict) -> np.ndarray:
    h, w = case["shape"]
    grid = np.full((h, w), BLACK, dtype=int)
    grid[case["structural"]] = BLUE
    grid[case["decorative"]] = BLUE
    mr, mc = case["marker"]
    grid[mr, mc] = PINK
    red_coords = case["red_coords"]
    for r, c in case["track_slots"]:
        if (r, c) not in red_coords:
            grid[r, c] = GREEN
    for r, c in red_coords:
        grid[r, c] = RED
    return grid


def _case_semantics(case: dict) -> dict:
    inp = _case_input(case)
    reach = _flood_background(inp, case["marker"])
    effective = _effective_track_slots(inp.shape, case["marker"], reach, case["track_slots"])
    dr, dc = _move_delta(case["track_slots"], effective, case["red_coords"])
    out = _render_output(inp.shape, case["structural"], case["track_slots"], effective, case["red_coords"], dr, dc)
    return {
        "input": inp,
        "reach": reach,
        "effective": effective,
        "delta": (dr, dc),
        "output": out,
    }


def _decor_allowed_mask(
    grid: np.ndarray,
    region_mask: np.ndarray,
    blocked_coords: set[tuple[int, int]],
) -> np.ndarray:
    allowed = region_mask & (grid == BLACK)
    allowed[0, :] = False
    allowed[-1, :] = False
    allowed[:, 0] = False
    allowed[:, -1] = False

    crowded = np.zeros_like(allowed, dtype=bool)
    for r, c in blocked_coords:
        if in_bounds(allowed.shape, r, c):
            crowded[r, c] = True
        for nr, nc in neighbors4(r, c):
            if in_bounds(allowed.shape, nr, nc):
                crowded[nr, nc] = True
    allowed &= ~crowded
    return allowed


def _try_place_fragment(
    allowed: np.ndarray,
    occupied_blue: np.ndarray,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    shapes = list(DECOR_SHAPES)
    rng.shuffle(shapes)
    h, w = allowed.shape
    for shape in shapes:
        sh, sw = _shape_size(shape)
        origins = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        rng.shuffle(origins)
        for r0, c0 in origins:
            cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if not all(allowed[r, c] for r, c in cells):
                continue
            ok = True
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if not in_bounds((h, w), nr, nc):
                        continue
                    if occupied_blue[nr, nc] and (nr, nc) not in cells:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return {(int(r), int(c)) for r, c in cells}
    return None


def _positions_from_slots(track_slots: list[tuple[int, int]]) -> list[int]:
    if len({r for r, _ in track_slots}) == 1:
        return sorted(c for _r, c in track_slots)
    return sorted(r for r, _c in track_slots)


def _canonical_attempt() -> dict | None:
    rng = random
    shape = rng.choice(OBJECT_SHAPES)
    obj_h, obj_w = _shape_size(shape)

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    if h <= obj_h + 3 or w <= obj_w + 8:
        return None

    step = rng.choice(TRACK_STEP_CHOICES)
    max_slots = min(TRACK_SLOT_RANGE[1], 1 + (w - 3) // step)
    if max_slots < TRACK_SLOT_RANGE[0]:
        return None
    n_slots = rng.randint(TRACK_SLOT_RANGE[0], max_slots)
    span = step * (n_slots - 1)
    start_c = rng.randint(1, w - 2 - span)
    track_cols = [start_c + step * k for k in range(n_slots)]

    split_idx = rng.randint(2, n_slots - 3)
    split_c = rng.randint(track_cols[split_idx] + 1, track_cols[split_idx + 1] - 1)

    marker_left = rng.choice([True, False])
    marker_top = rng.choice([True, False])
    marker = (0 if marker_top else h - 1, 0 if marker_left else w - 1)
    reachable_left = marker_left

    use_pocket = rng.random() < 0.7
    pocket_side_top = rng.choice([True, False])

    if use_pocket and reachable_left:
        if split_idx < 3:
            return None
        outer_last = rng.randint(2, split_idx - 1)
        pocket_c = rng.randint(track_cols[outer_last] + 1, track_cols[outer_last + 1] - 1)
    elif use_pocket and not reachable_left:
        if n_slots - (split_idx + 1) < 4:
            return None
        boundary_idx = rng.randint(split_idx + 1, n_slots - 4)
        pocket_c = rng.randint(track_cols[boundary_idx] + 1, track_cols[boundary_idx + 1] - 1)
    else:
        pocket_c = None

    if use_pocket and pocket_side_top:
        max_row = h - obj_h - 2
        if max_row < 1:
            return None
        track_row = rng.randint(1, max_row)
        if track_row + 2 > h - 2:
            return None
        branch_r = rng.randint(track_row + 2, h - 2)
    elif use_pocket and not pocket_side_top:
        min_row = 2
        max_row = h - obj_h - 1
        if max_row < min_row:
            return None
        track_row = rng.randint(min_row, max_row)
        branch_r = rng.randint(1, track_row - 1)
    else:
        max_row = h - obj_h - 1
        if max_row < 1:
            return None
        track_row = rng.randint(1, max_row)
        branch_r = None

    structural = np.zeros((h, w), dtype=bool)
    structural[:, split_c] = True
    if use_pocket and pocket_c is not None and branch_r is not None:
        if pocket_side_top:
            structural[: branch_r + 1, pocket_c] = True
            c0, c1 = sorted((pocket_c, split_c))
            structural[branch_r, c0:c1 + 1] = True
        else:
            structural[branch_r:, pocket_c] = True
            c0, c1 = sorted((pocket_c, split_c))
            structural[branch_r, c0:c1 + 1] = True

    track_slots = [(track_row, c) for c in track_cols]
    provisional = np.full((h, w), BLACK, dtype=int)
    provisional[structural] = BLUE
    provisional[marker] = PINK
    for r, c in track_slots:
        provisional[r, c] = GREEN

    reach = _flood_background(provisional, marker)
    effective = _effective_track_slots(provisional.shape, marker, reach, track_slots)
    positions = _positions_from_slots(sorted(effective))
    if len(effective) < 3:
        return None

    pos_candidates: dict[str, list[int]] = {"tie": [], "pos": [], "neg": []}
    for r, c in sorted(effective):
        label = _classify_anchor(positions, c)
        if label is not None:
            pos_candidates[label].append(c)

    available = [label for label, vals in pos_candidates.items() if vals]
    if not available:
        return None
    weights = {"tie": 3, "pos": 2, "neg": 2}
    desired = rng.choices(available, weights=[weights[label] for label in available], k=1)[0]
    anchor_c = rng.choice(pos_candidates[desired])
    red_coords = _shape_to_coords((track_row, anchor_c), shape)

    if any(not in_bounds((h, w), r, c) for r, c in red_coords):
        return None
    if any(structural[r, c] or (r, c) == marker for r, c in red_coords):
        return None

    base_case = {
        "shape": (h, w),
        "structural": structural,
        "decorative": np.zeros((h, w), dtype=bool),
        "marker": marker,
        "track_slots": track_slots,
        "red_coords": red_coords,
        "desired": desired,
    }
    try:
        base_sem = _case_semantics(base_case)
    except AssertionError:
        return None
    eff_positions = _positions_from_slots(sorted(base_sem["effective"]))
    anchor_label = _classify_anchor(eff_positions, anchor_c)
    if anchor_label != desired:
        return None
    if len(base_sem["effective"]) < 3:
        return None
    if not any(slot in base_sem["effective"] and slot in red_coords for slot in track_slots):
        return None

    dr, dc = base_sem["delta"]
    shifted = _shift_coords(red_coords, dr, dc)
    if any(not in_bounds((h, w), r, c) or structural[r, c] or (r, c) == marker for r, c in shifted):
        return None
    for r, c in red_coords:
        if not any((nr, nc) == marker or base_sem["reach"][nr, nc] for nr, nc in neighbors4(r, c) if in_bounds((h, w), nr, nc)):
            return None

    blocked = set(track_slots) | red_coords | {marker}
    base_grid = base_sem["input"]
    allowed_reach = _decor_allowed_mask(base_grid, base_sem["reach"], blocked)
    allowed_unreach = _decor_allowed_mask(base_grid, (~base_sem["reach"]) & (base_grid == BLACK), blocked)

    decorative = np.zeros((h, w), dtype=bool)
    frag_reach = _try_place_fragment(allowed_reach, structural | decorative, rng)
    if frag_reach is None:
        return None
    for r, c in frag_reach:
        decorative[r, c] = True

    if allowed_unreach.any() and rng.random() < 0.8:
        frag_unreach = _try_place_fragment(allowed_unreach, structural | decorative, rng)
        if frag_unreach is not None:
            for r, c in frag_unreach:
                decorative[r, c] = True

    if rng.random() < 0.35:
        allowed_extra = _decor_allowed_mask(base_grid, base_grid == BLACK, blocked)
        frag_extra = _try_place_fragment(allowed_extra, structural | decorative, rng)
        if frag_extra is not None:
            for r, c in frag_extra:
                decorative[r, c] = True

    case = {
        "shape": (h, w),
        "structural": structural,
        "decorative": decorative,
        "marker": marker,
        "track_slots": track_slots,
        "red_coords": red_coords,
        "desired": desired,
    }
    try:
        sem = _case_semantics(case)
    except AssertionError:
        return None
    if sem["effective"] != base_sem["effective"]:
        return None
    if not decorative.any():
        return None
    return case


def _transpose_case(case: dict) -> dict:
    return {
        "shape": (case["shape"][1], case["shape"][0]),
        "structural": case["structural"].T.copy(),
        "decorative": case["decorative"].T.copy(),
        "marker": (case["marker"][1], case["marker"][0]),
        "track_slots": [(c, r) for r, c in case["track_slots"]],
        "red_coords": _transpose_coords(case["red_coords"]),
        "desired": case["desired"],
    }


def _shortcut_keep_decor(case: dict, sem: dict) -> np.ndarray:
    out = sem["output"].copy()
    out[case["decorative"]] = BLUE
    return out


def _shortcut_keep_all_track(case: dict, sem: dict) -> np.ndarray:
    dr, dc = sem["delta"]
    out = np.full(case["shape"], BLACK, dtype=int)
    out[case["structural"]] = BLUE
    old_hidden = {slot for slot in case["track_slots"] if slot in case["red_coords"]}
    new_red = _shift_coords(case["red_coords"], dr, dc)
    new_hidden = {slot for slot in case["track_slots"] if slot in new_red}
    for slot in case["track_slots"]:
        if slot not in old_hidden and slot not in new_hidden:
            out[slot] = GREEN
    for r, c in new_red:
        out[r, c] = RED
    return out


def _shortcut_restore_old_slots(case: dict, sem: dict) -> np.ndarray:
    out = sem["output"].copy()
    old_hidden = {slot for slot in case["track_slots"] if slot in case["red_coords"] and slot in sem["effective"]}
    for r, c in old_hidden:
        out[r, c] = GREEN
    return out


def _shortcut_hide_only_unreachable_decor(case: dict, sem: dict) -> np.ndarray:
    out = sem["output"].copy()
    undecorated = {
        "shape": case["shape"],
        "structural": case["structural"],
        "decorative": np.zeros(case["shape"], dtype=bool),
        "marker": case["marker"],
        "track_slots": case["track_slots"],
        "red_coords": case["red_coords"],
        "desired": case["desired"],
    }
    grid = _case_input(undecorated)
    reach = _flood_background(grid, case["marker"])
    for r, c in np.argwhere(case["decorative"]):
        rr = int(r)
        cc = int(c)
        near_reach = False
        for nr, nc in neighbors4(rr, cc):
            if not in_bounds(case["shape"], nr, nc):
                continue
            if (nr, nc) == case["marker"] or reach[nr, nc]:
                near_reach = True
                break
        out[rr, cc] = BLUE if near_reach else BLACK
    return out


def _valid_case(case: dict) -> bool:
    try:
        sem = _case_semantics(case)
    except AssertionError:
        return False
    inp = sem["input"]
    expected = sem["output"]

    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    decorative_components = components_mask(case["decorative"], connectivity=4)
    if not decorative_components:
        return False
    if len(sem["effective"]) < 3:
        return False

    invalid_visible = [slot for slot in case["track_slots"] if slot not in sem["effective"] and slot not in case["red_coords"]]
    if not invalid_visible:
        return False

    if np.array_equal(_shortcut_keep_decor(case, sem), expected):
        return False
    if np.array_equal(_shortcut_keep_all_track(case, sem), expected):
        return False
    if np.array_equal(_shortcut_restore_old_slots(case, sem), expected):
        return False
    if np.array_equal(_shortcut_hide_only_unreachable_decor(case, sem), expected):
        return False

    if np.count_nonzero(inp == BLUE) == np.count_nonzero(expected == BLUE):
        return False
    if np.count_nonzero(inp == GREEN) == np.count_nonzero(expected == GREEN):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _canonical_attempt()
        if case is None:
            continue
        if random.random() < 0.5:
            case = _transpose_case(case)
        if not _valid_case(case):
            continue
        return _case_input(case)
    raise RuntimeError("failed to generate a valid reachability-filtered rail puzzle")
