from __future__ import annotations

import itertools
import random

import numpy as np


GREY = 5
BG = 0
MARKER_COLORS = (1, 2, 3, 4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {"top": (1, 0), "right": (0, -1), "bottom": (-1, 0), "left": (0, 1)}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIRS.index(direction) + 1) % 4]


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIRS.index(direction) - 1) % 4]


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-int(direction[0]), -int(direction[1]))


def _line_indices(sub: np.ndarray, axis: int) -> list[int]:
    counts = np.sum(sub == GREY, axis=axis)
    threshold = (sub.shape[1 - axis] // 2) + 1
    return [int(i) for i, count in enumerate(counts) if int(count) >= threshold]


def _slot_index(lines: list[int], value: int) -> int:
    for idx in range(len(lines) - 1):
        if int(lines[idx]) < int(value) < int(lines[idx + 1]):
            return int(idx)
    raise AssertionError("door does not fall inside a slot segment")


def _fill_slot(out: np.ndarray, row_lines: list[int], col_lines: list[int], slot: tuple[int, int], color: int) -> None:
    sr, sc = map(int, slot)
    out[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[sc] + 1 : col_lines[sc + 1]] = int(color)


def _slot_anchor(row_lines: list[int], col_lines: list[int], slot: tuple[int, int], side: tuple[int, int]) -> tuple[int, int]:
    sr, sc = map(int, slot)
    r0 = int(row_lines[sr] + 1)
    r1 = int(row_lines[sr + 1] - 1)
    c0 = int(col_lines[sc] + 1)
    c1 = int(col_lines[sc + 1] - 1)
    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    if side == (-1, 0):
        return int(r0), int(mc)
    if side == (1, 0):
        return int(r1), int(mc)
    if side == (0, -1):
        return int(mr), int(c0)
    if side == (0, 1):
        return int(mr), int(c1)
    raise AssertionError("invalid side")


def _paint_segment(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        out[r0, lo : hi + 1] = int(color)
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        out[lo : hi + 1, c0] = int(color)
        return
    raise AssertionError("path segments must be orthogonal")


def _paint_slot_path(
    out: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    slot: tuple[int, int],
    entry_side: tuple[int, int],
    exit_side: tuple[int, int] | None,
    color: int,
) -> None:
    center = (
        (row_lines[slot[0]] + row_lines[slot[0] + 1]) // 2,
        (col_lines[slot[1]] + col_lines[slot[1] + 1]) // 2,
    )
    start = _slot_anchor(row_lines, col_lines, slot, entry_side)
    _paint_segment(out, start, center, color)
    if exit_side is not None:
        end = _slot_anchor(row_lines, col_lines, slot, exit_side)
        _paint_segment(out, center, end, color)


def _trace_path(
    n_rows: int, n_cols: int, start_slot: tuple[int, int], direction: tuple[int, int]
) -> tuple[dict[tuple[int, int], int], list[dict], bool, bool]:
    slot = (int(start_slot[0]), int(start_slot[1]))
    direction = (int(direction[0]), int(direction[1]))
    entry_side = _opposite(direction)
    length = 1
    arrivals: dict[tuple[int, int], int] = {}
    steps: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    saw_right = False
    saw_left = False

    while True:
        state = (slot, direction)
        if state in seen:
            break
        seen.add(state)
        arrivals.setdefault(slot, int(length))

        nr = int(slot[0] + direction[0])
        nc = int(slot[1] + direction[1])
        exit_side: tuple[int, int] | None
        next_slot: tuple[int, int] | None
        next_dir: tuple[int, int] | None

        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            exit_side = direction
            next_slot = (int(nr), int(nc))
            next_dir = direction
        else:
            right_dir = _right(direction)
            rr = int(slot[0] + right_dir[0])
            rc = int(slot[1] + right_dir[1])
            if 0 <= rr < n_rows and 0 <= rc < n_cols:
                saw_right = True
                exit_side = right_dir
                next_slot = (int(rr), int(rc))
                next_dir = right_dir
            else:
                left_dir = _left(direction)
                lr = int(slot[0] + left_dir[0])
                lc = int(slot[1] + left_dir[1])
                if 0 <= lr < n_rows and 0 <= lc < n_cols:
                    saw_left = True
                    exit_side = left_dir
                    next_slot = (int(lr), int(lc))
                    next_dir = left_dir
                else:
                    exit_side = None
                    next_slot = None
                    next_dir = None

        steps.append({"slot": slot, "entry": entry_side, "exit": exit_side})
        if next_slot is None or next_dir is None:
            break

        slot = next_slot
        direction = next_dir
        entry_side = _opposite(direction)
        length += 1

    return arrivals, steps, saw_right, saw_left


def _parse_grid(grid: np.ndarray) -> tuple[dict, list[dict]]:
    grey_mask = grid == GREY
    fr0, fc0, fr1, fc1 = _bbox(grey_mask)
    frame = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    row_lines = [fr0 + idx for idx in _line_indices(frame, axis=1)]
    col_lines = [fc0 + idx for idx in _line_indices(frame, axis=0)]
    if len(row_lines) < 4 or len(col_lines) < 4:
        raise AssertionError("expected a rectangular lattice frame")

    slot_heights = [row_lines[i + 1] - row_lines[i] - 1 for i in range(len(row_lines) - 1)]
    slot_widths = [col_lines[i + 1] - col_lines[i] - 1 for i in range(len(col_lines) - 1)]
    if min(slot_heights) < 1 or min(slot_widths) < 1:
        raise AssertionError("invalid slot size")

    for wall_r in row_lines[1:-1]:
        for sc in range(len(col_lines) - 1):
            segment = grid[int(wall_r), col_lines[sc] + 1 : col_lines[sc + 1]]
            if int(np.sum(segment == BG)) != 1:
                raise AssertionError("each internal horizontal wall segment needs one door")
    for wall_c in col_lines[1:-1]:
        for sr in range(len(row_lines) - 1):
            segment = grid[row_lines[sr] + 1 : row_lines[sr + 1], int(wall_c)]
            if int(np.sum(segment == BG)) != 1:
                raise AssertionError("each internal vertical wall segment needs one door")

    markers: list[dict] = []
    for r, c in np.argwhere((grid != BG) & (grid != GREY)):
        r = int(r)
        c = int(c)
        doors = []
        if r + 1 == fr0 and fc0 <= c <= fc1 and int(grid[fr0, c]) == BG:
            doors.append(("top", (int(fr0), int(c))))
        if r - 1 == fr1 and fc0 <= c <= fc1 and int(grid[fr1, c]) == BG:
            doors.append(("bottom", (int(fr1), int(c))))
        if c + 1 == fc0 and fr0 <= r <= fr1 and int(grid[r, fc0]) == BG:
            doors.append(("left", (int(r), int(fc0))))
        if c - 1 == fc1 and fr0 <= r <= fr1 and int(grid[r, fc1]) == BG:
            doors.append(("right", (int(r), int(fc1))))
        if len(doors) != 1:
            raise AssertionError("each marker must touch exactly one outer door")

        side, door = doors[0]
        if side in {"top", "bottom"}:
            slot = (0 if side == "top" else len(row_lines) - 2, _slot_index(col_lines, int(door[1])))
        else:
            slot = (_slot_index(row_lines, int(door[0])), 0 if side == "left" else len(col_lines) - 2)
        markers.append(
            {
                "color": int(grid[r, c]),
                "coord": (int(r), int(c)),
                "door": door,
                "side": side,
                "start_slot": (int(slot[0]), int(slot[1])),
                "direction": SIDE_TO_DIR[side],
                "tie_dist": abs(int(r) - int(fr0)) + abs(int(c) - int(fc0)),
            }
        )

    if not (3 <= len(markers) <= 4):
        raise AssertionError("expected 3 to 4 markers")
    if len({m["color"] for m in markers}) != len(markers):
        raise AssertionError("marker colors must be distinct")

    side_counts = {side: 0 for side in SIDE_TO_DIR}
    outer_doors = 0
    for sc in range(len(col_lines) - 1):
        outer_doors += int(np.sum(grid[row_lines[0], col_lines[sc] + 1 : col_lines[sc + 1]] == BG))
        outer_doors += int(np.sum(grid[row_lines[-1], col_lines[sc] + 1 : col_lines[sc + 1]] == BG))
    for sr in range(len(row_lines) - 1):
        outer_doors += int(np.sum(grid[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[0]] == BG))
        outer_doors += int(np.sum(grid[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[-1]] == BG))
    for marker in markers:
        side_counts[str(marker["side"])] += 1
    if any(count > 2 for count in side_counts.values()):
        raise AssertionError("at most two markers per side")
    if int(outer_doors) != len(markers):
        raise AssertionError("outer doors should match marker count")

    return {
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "row_lines": [int(x) for x in row_lines],
        "col_lines": [int(x) for x in col_lines],
        "n_rows": int(len(row_lines) - 1),
        "n_cols": int(len(col_lines) - 1),
    }, markers


def _solve_from_parsed(grid: np.ndarray, parsed: dict, markers: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    row_lines = parsed["row_lines"]
    col_lines = parsed["col_lines"]
    n_rows = int(parsed["n_rows"])
    n_cols = int(parsed["n_cols"])

    path_infos = []
    saw_right_any = False
    saw_left_any = False
    for idx, marker in enumerate(markers):
        arrivals, steps, saw_right, saw_left = _trace_path(n_rows, n_cols, marker["start_slot"], marker["direction"])
        saw_right_any |= bool(saw_right)
        saw_left_any |= bool(saw_left)
        path_infos.append(
            {
                "marker_id": int(idx),
                "color": int(marker["color"]),
                "tie_dist": int(marker["tie_dist"]),
                "arrivals": arrivals,
                "steps": steps,
            }
        )

    slot_winner: dict[tuple[int, int], int] = {}
    for sr in range(n_rows):
        for sc in range(n_cols):
            slot = (int(sr), int(sc))
            candidates = []
            for info in path_infos:
                if slot in info["arrivals"]:
                    candidates.append((int(info["arrivals"][slot]), int(info["tie_dist"]), int(info["marker_id"])))
            if not candidates:
                continue
            candidates.sort()
            if len(candidates) >= 2 and candidates[0][:2] == candidates[1][:2]:
                raise AssertionError("tie must be resolved by marker distance")
            slot_winner[slot] = int(candidates[0][2])
            _fill_slot(out, row_lines, col_lines, slot, path_infos[int(candidates[0][2])]["color"])

    for info in path_infos:
        for step in info["steps"]:
            if slot_winner.get(step["slot"]) != info["marker_id"]:
                continue
            _paint_slot_path(out, row_lines, col_lines, step["slot"], step["entry"], step["exit"], info["color"])

    if not saw_right_any or not saw_left_any:
        raise AssertionError("valid puzzles should exercise both right and left turns")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each outer marker through the lattice doors and color every reached slot by the winning path.

    Input structure:
    - The background is black, and a one-pixel-thick grey rectangular lattice divides the interior into slots.
    - Every internal wall segment between neighboring slots has exactly one one-pixel door, while 3 to 4 colored singleton markers sit just outside matching outer-wall doors.
    - Marker colors are distinct and come from the visible ARC palette, not from hidden numeric ordering.

    Rule:
    - Parse the lattice rows, columns, slots, internal doors, and marker-adjacent outer doors.
    - For each marker, enter the adjacent boundary slot and move orthogonally across slots: continue straight through the opposite internal door when possible; otherwise turn right, else left, and stop only if no adjacent slot exists.
    - Record for every reached slot the number of crossed doors needed to enter it. Fill each slot with the color of the marker that reaches it with the shortest path; if two paths reach a slot with the same length, choose the marker whose pixel is closer in Manhattan distance to the frame's top-left corner.
    - Keep the grey frame and the marker pixels unchanged, and draw the winner's path only inside the slots it wins.

    Key insight:
    - The output colors are decided dynamically by path competition on the slot graph, not by a fixed slot position or by marker color ids.
    """
    g = np.asarray(grid, dtype=int)
    parsed, markers = _parse_grid(g)
    return _solve_from_parsed(g, parsed, markers)


def _geometry(n_rows: int, n_cols: int, slot_h: int = 3, slot_w: int = 3) -> tuple[int, int, list[int], list[int]]:
    fr0 = fc0 = 1
    row_lines = [int(fr0 + i * (slot_h + 1)) for i in range(n_rows + 1)]
    col_lines = [int(fc0 + j * (slot_w + 1)) for j in range(n_cols + 1)]
    height = int(row_lines[-1] + 2)
    width = int(col_lines[-1] + 2)
    return height, width, row_lines, col_lines


def _door_center(row_lines: list[int], col_lines: list[int], side: str, index: int) -> tuple[int, int]:
    if side == "top":
        return int(row_lines[0]), int((col_lines[index] + col_lines[index + 1]) // 2)
    if side == "bottom":
        return int(row_lines[-1]), int((col_lines[index] + col_lines[index + 1]) // 2)
    if side == "left":
        return int((row_lines[index] + row_lines[index + 1]) // 2), int(col_lines[0])
    if side == "right":
        return int((row_lines[index] + row_lines[index + 1]) // 2), int(col_lines[-1])
    raise AssertionError("invalid side")


def _marker_coord(row_lines: list[int], col_lines: list[int], side: str, index: int) -> tuple[int, int]:
    r, c = _door_center(row_lines, col_lines, side, index)
    if side == "top":
        return int(r - 1), int(c)
    if side == "bottom":
        return int(r + 1), int(c)
    if side == "left":
        return int(r), int(c - 1)
    return int(r), int(c + 1)


def _boundary_starts(n_rows: int, n_cols: int) -> list[dict]:
    starts = []
    for sc in range(n_cols):
        starts.append({"side": "top", "index": int(sc), "start_slot": (0, int(sc)), "direction": (1, 0)})
    for sr in range(n_rows):
        starts.append({"side": "right", "index": int(sr), "start_slot": (int(sr), n_cols - 1), "direction": (0, -1)})
    for sc in range(n_cols):
        starts.append({"side": "bottom", "index": int(sc), "start_slot": (n_rows - 1, int(sc)), "direction": (-1, 0)})
    for sr in range(n_rows):
        starts.append({"side": "left", "index": int(sr), "start_slot": (int(sr), 0), "direction": (0, 1)})
    return starts


def _abstract_outcome(n_rows: int, n_cols: int, starts: list[dict], tie_dists: list[int]) -> dict:
    slots = {(int(r), int(c)) for r in range(n_rows) for c in range(n_cols)}
    path_infos = []
    saw_right = False
    saw_left = False
    for idx, (start, tie_dist) in enumerate(zip(starts, tie_dists)):
        arrivals, steps, saw_r, saw_l = _trace_path(n_rows, n_cols, start["start_slot"], start["direction"])
        saw_right |= bool(saw_r)
        saw_left |= bool(saw_l)
        path_infos.append(
            {
                "marker_id": int(idx),
                "arrivals": arrivals,
                "steps": steps,
                "tie_dist": int(tie_dist),
                "start": start,
            }
        )

    union = set().union(*[set(info["arrivals"]) for info in path_infos])
    contested = [slot for slot in slots if sum(slot in info["arrivals"] for info in path_infos) >= 2]
    tie_slots = []
    winners: dict[tuple[int, int], int] = {}
    for slot in slots:
        candidates = []
        for info in path_infos:
            if slot in info["arrivals"]:
                candidates.append((int(info["arrivals"][slot]), int(info["tie_dist"]), int(info["marker_id"])))
        if not candidates:
            continue
        candidates.sort()
        if len(candidates) >= 2 and candidates[0][:2] == candidates[1][:2]:
            return {"ok": False}
        if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
            tie_slots.append(slot)
        winners[slot] = int(candidates[0][2])

    wins_per_marker = {info["marker_id"]: 0 for info in path_infos}
    for marker_id in winners.values():
        wins_per_marker[int(marker_id)] += 1

    ok = (
        all(len(info["arrivals"]) >= 2 for info in path_infos)
        and union == slots
        and bool(contested)
        and bool(tie_slots)
        and all(count > 0 for count in wins_per_marker.values())
        and saw_right
        and saw_left
    )
    return {
        "ok": bool(ok),
        "paths": path_infos,
        "winners": winners,
        "tie_slots": tie_slots,
        "contested": contested,
    }


def _render_input(n_rows: int, n_cols: int, marker_specs: list[dict]) -> tuple[np.ndarray, dict]:
    height, width, row_lines, col_lines = _geometry(n_rows, n_cols)
    grid = np.zeros((height, width), dtype=int)
    fr0, fc0 = int(row_lines[0]), int(col_lines[0])
    fr1, fc1 = int(row_lines[-1]), int(col_lines[-1])

    grid[fr0 : fr1 + 1, col_lines] = GREY
    grid[row_lines, fc0 : fc1 + 1] = GREY

    for sr in range(n_rows - 1):
        wall_r = int(row_lines[sr + 1])
        for sc in range(n_cols):
            door_c = int((col_lines[sc] + col_lines[sc + 1]) // 2)
            grid[wall_r, door_c] = BG
    for sc in range(n_cols - 1):
        wall_c = int(col_lines[sc + 1])
        for sr in range(n_rows):
            door_r = int((row_lines[sr] + row_lines[sr + 1]) // 2)
            grid[door_r, wall_c] = BG

    for spec in marker_specs:
        door_r, door_c = _door_center(row_lines, col_lines, spec["side"], spec["index"])
        mark_r, mark_c = _marker_coord(row_lines, col_lines, spec["side"], spec["index"])
        grid[door_r, door_c] = BG
        grid[mark_r, mark_c] = int(spec["color"])
        spec["coord"] = (int(mark_r), int(mark_c))
        spec["tie_dist"] = abs(int(mark_r) - int(fr0)) + abs(int(mark_c) - int(fc0))

    meta = {
        "row_lines": [int(x) for x in row_lines],
        "col_lines": [int(x) for x in col_lines],
        "frame_topleft": (int(fr0), int(fc0)),
    }
    return grid, meta


def _render_output(inp: np.ndarray, n_rows: int, n_cols: int, marker_specs: list[dict], outcome: dict) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    _, _, row_lines, col_lines = _geometry(n_rows, n_cols)
    for slot, marker_id in outcome["winners"].items():
        color = int(marker_specs[int(marker_id)]["color"])
        _fill_slot(out, row_lines, col_lines, slot, color)
    for info in outcome["paths"]:
        color = int(marker_specs[int(info["marker_id"])]["color"])
        for step in info["steps"]:
            if outcome["winners"].get(step["slot"]) != info["marker_id"]:
                continue
            _paint_slot_path(out, row_lines, col_lines, step["slot"], step["entry"], step["exit"], color)
    return out


def _valid_combinations(n_rows: int, n_cols: int, count: int) -> list[tuple[dict, ...]]:
    starts = _boundary_starts(n_rows, n_cols)
    combos = []
    for combo in itertools.combinations(starts, count):
        side_counts = {side: 0 for side in SIDE_TO_DIR}
        for item in combo:
            side_counts[item["side"]] += 1
        if any(v > 2 for v in side_counts.values()):
            continue
        combos.append(combo)
    random.shuffle(combos)
    return combos


def _choose_marker_specs(n_rows: int, n_cols: int) -> tuple[list[dict], dict]:
    counts = [3, 4]
    random.shuffle(counts)
    for count in counts:
        for combo in _valid_combinations(n_rows, n_cols, count):
            colors = list(random.sample(MARKER_COLORS, count))
            random.shuffle(colors)
            marker_specs = [dict(item) | {"color": int(color)} for item, color in zip(combo, colors)]
            preview_specs = [dict(spec) for spec in marker_specs]
            _render_input(n_rows, n_cols, preview_specs)
            tie_dists = [int(spec["tie_dist"]) for spec in preview_specs]
            if len(set(tie_dists)) != len(tie_dists):
                continue
            outcome = _abstract_outcome(n_rows, n_cols, preview_specs, tie_dists)
            if outcome["ok"]:
                return marker_specs, outcome
    raise RuntimeError("failed to sample a non-degenerate marker layout")


def _attempt_generate() -> np.ndarray | None:
    n_rows, n_cols = random.choice([(3, 3), (4, 3)])
    marker_specs, _ = _choose_marker_specs(n_rows, n_cols)
    inp, _meta = _render_input(n_rows, n_cols, marker_specs)
    tie_dists = [int(spec["tie_dist"]) for spec in marker_specs]
    outcome = _abstract_outcome(n_rows, n_cols, marker_specs, tie_dists)
    if not outcome["ok"]:
        return None
    expected = _render_output(inp, n_rows, n_cols, marker_specs, outcome)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid puzzle")
