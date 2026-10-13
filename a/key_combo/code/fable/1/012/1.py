from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MIN_OCCLUDER_H = 4
MAX_OCCLUDER_H = 7
MAX_GENERATION_ATTEMPTS = 400
BRANCH_CYCLE = ("hidden", "straight", "wall", "hidden", "straight", "wall", "hidden")
_GENERATE_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _inside_box(box: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _blue_outside_box(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(grid == BLUE)
        if not _inside_box(box, (int(r), int(c)))
    }


def _red_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid == RED)}


def _simulate_path(
    shape: tuple[int, int],
    start_col: int,
    start_dc: int,
    red_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    h, w = shape
    r = h - 1
    c = int(start_col)
    dc = int(start_dc)
    path: list[tuple[int, int]] = []
    states: list[dict[str, object]] = []

    while True:
        path.append((r, c))
        if r == 0:
            break

        dc_before = dc
        forward: tuple[int, int] | None = None
        event = "none"
        blocker: tuple[int, int] | None = None

        if c + dc_before < 0 or c + dc_before >= w:
            event = "wall"
            dc = -dc_before
        else:
            forward = (r - 1, c + dc_before)
            if forward in red_cells:
                event = "red"
                blocker = forward
                dc = -dc_before

        next_cell = (r - 1, c + dc)
        states.append(
            {
                "row": r,
                "col": c,
                "dc": dc_before,
                "event": event,
                "forward": forward,
                "blocker": blocker,
                "next": next_cell,
            }
        )
        r, c = next_cell

    return path, states


def _build_output(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    path: Iterable[tuple[int, int]],
    hidden_red: tuple[int, int] | None,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for r, c in path:
        if _inside_box(box, (r, c)):
            out[r, c] = BLUE
    if hidden_red is not None:
        out[hidden_red] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the gray occluder, restore the hidden diagonal trajectory, and redraw the concealed red bumper if one caused the bend.

    Input structure:
    - A black background contains a blue ray that starts from one cell on the bottom row and rises one row per step.
    - Sparse red single-cell bumpers may redirect the ray, and a solid gray rectangle hides one middle segment.
    - Outside the gray box, every visible blue cell and visible red bumper is already correct.

    Rule:
    - Detect the gray rectangle and keep everything outside it unchanged.
    - Re-simulate the ray from the visible bottom start, trying both initial horizontal directions and either no hidden bumper or one hidden red bumper inside the gray box.
    - Accept the unique candidate whose simulated blue cells outside the gray box match the input exactly, then replace gray cells with the true hidden content: blue on the traversed segment, black elsewhere, and the inferred red bumper if it was the hidden cause of the bend.

    Key insight:
    - The hidden region is solved by replaying the trajectory physics, not by interpolation: the visible before/after path uniquely determines whether the occluded segment stayed straight, bounced off a wall, or was bent by one concealed red blocker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    occluder = g == GREY
    box = _bbox(occluder)
    assert box is not None, "expected one gray occluder"
    r0, c0, r1, c1 = box
    assert np.all(occluder[r0 : r1 + 1, c0 : c1 + 1]), "gray cells must form a solid rectangle"
    assert int(np.count_nonzero(occluder)) == (r1 - r0 + 1) * (c1 - c0 + 1), "occluder must be rectangular"

    starts = [
        int(c)
        for c in np.where(g[h - 1] == BLUE)[0].tolist()
        if not _inside_box(box, (h - 1, int(c)))
    ]
    assert starts, "expected a visible blue start on the bottom row"

    visible_blues = _blue_outside_box(g, box)
    visible_reds = _red_cells(g)
    hidden_candidates = [None]
    hidden_candidates.extend(
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    )

    outputs: dict[bytes, np.ndarray] = {}
    for start_col in starts:
        for start_dc in (-1, 1):
            for hidden_red in hidden_candidates:
                reds = set(visible_reds)
                if hidden_red is not None:
                    reds.add(hidden_red)

                path, states = _simulate_path((h, w), start_col, start_dc, reds)
                if hidden_red is not None and not any(
                    state["blocker"] == hidden_red for state in states
                ):
                    continue

                outside_path = {cell for cell in path if not _inside_box(box, cell)}
                if outside_path != visible_blues:
                    continue

                out = _build_output(g, box, path, hidden_red)
                outputs[out.tobytes()] = out

    assert outputs, "no trajectory reconstruction matched the visible evidence"
    assert len(outputs) == 1, "trajectory reconstruction must be unique"
    return next(iter(outputs.values()))


def _make_forced_trajectory(
    h: int,
    w: int,
    start_col: int,
    start_dc: int,
    forced_red_rows: set[int],
) -> tuple[list[tuple[int, int]], list[dict[str, object]], set[tuple[int, int]]] | None:
    red_cells: set[tuple[int, int]] = set()
    r = h - 1
    c = int(start_col)
    dc = int(start_dc)
    path: list[tuple[int, int]] = []
    states: list[dict[str, object]] = []

    while True:
        path.append((r, c))
        if r == 0:
            break

        dc_before = dc
        forward: tuple[int, int] | None = None
        event = "none"
        blocker: tuple[int, int] | None = None

        if c + dc_before < 0 or c + dc_before >= w:
            if r in forced_red_rows:
                return None
            event = "wall"
            dc = -dc_before
        else:
            forward = (r - 1, c + dc_before)
            if r in forced_red_rows:
                if c in (0, w - 1):
                    return None
                red_cells.add(forward)
                event = "red"
                blocker = forward
                dc = -dc_before

        next_cell = (r - 1, c + dc)
        states.append(
            {
                "row": r,
                "col": c,
                "dc": dc_before,
                "event": event,
                "forward": forward,
                "blocker": blocker,
                "next": next_cell,
            }
        )
        r, c = next_cell

    return path, states, red_cells


def _interval_candidates(
    h: int,
    w: int,
    states: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_type: dict[str, list[dict[str, object]]] = {"hidden": [], "wall": [], "straight": []}
    red_states = [state for state in states if state["event"] == "red"]

    for rt in range(2, h - MIN_OCCLUDER_H):
        max_rb = min(h - 3, rt + MAX_OCCLUDER_H - 1)
        for rb in range(rt + MIN_OCCLUDER_H - 1, max_rb + 1):
            events_inside = [
                state
                for state in states
                if rt <= int(state["row"]) <= rb and state["event"] != "none"
            ]

            balanced_reds = True
            for state in red_states:
                row_inside = rt <= int(state["row"]) <= rb
                blocker = state["blocker"]
                assert blocker is not None
                blocker_inside = rt <= int(blocker[0]) <= rb
                if row_inside != blocker_inside:
                    balanced_reds = False
                    break
            if not balanced_reds:
                continue

            if not events_inside:
                by_type["straight"].append({"rt": rt, "rb": rb})
                continue

            if len(events_inside) != 1:
                continue

            event_state = events_inside[0]
            if event_state["event"] == "wall":
                by_type["wall"].append({"rt": rt, "rb": rb, "state": event_state})
            elif event_state["event"] == "red" and int(event_state["col"]) not in (0, w - 1):
                by_type["hidden"].append({"rt": rt, "rb": rb, "state": event_state})

    return by_type


def _choose_box(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    interval: dict[str, object],
    branch: str,
) -> tuple[int, int, int, int]:
    h, w = shape
    rt = int(interval["rt"])
    rb = int(interval["rb"])
    cells = [(r, c) for r, c in path if rt <= r <= rb]
    if branch == "hidden":
        blocker = interval["state"]["blocker"]
        assert blocker is not None
        cells.append((int(blocker[0]), int(blocker[1])))

    min_c = min(c for _r, c in cells)
    max_c = max(c for _r, c in cells)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    c0 = max(0, min_c - left_pad)
    c1 = min(w - 1, max_c + right_pad)

    while c1 - c0 + 1 < 3:
        if c0 > 0:
            c0 -= 1
        elif c1 < w - 1:
            c1 += 1
        else:
            break

    assert 0 <= rt <= rb < h
    assert 0 <= c0 <= c1 < w
    return rt, c0, rb, c1


def _paint_truth(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    red_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in path:
        grid[r, c] = BLUE
    for r, c in red_cells:
        grid[r, c] = RED
    return grid


def _add_distractors(
    truth: np.ndarray,
    path: list[tuple[int, int]],
    states: list[dict[str, object]],
    red_cells: set[tuple[int, int]],
    box: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    h, w = truth.shape
    path_set = set(path)
    sensitive = {
        (int(state["forward"][0]), int(state["forward"][1]))
        for state in states
        if state["forward"] is not None
    }

    def far_from_path(cell: tuple[int, int]) -> bool:
        r, c = cell
        return all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in path_set)

    def far_from_reds(cell: tuple[int, int], placed: set[tuple[int, int]]) -> bool:
        r, c = cell
        return all(max(abs(r - rr), abs(c - rc)) > 1 for rr, rc in placed)

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in path_set
        and (r, c) not in red_cells
        and (r, c) not in sensitive
        and not _inside_box(box, (r, c))
        and far_from_path((r, c))
    ]
    random.shuffle(candidates)

    visible_reds = {cell for cell in red_cells if not _inside_box(box, cell)}
    min_needed = 0 if visible_reds else 1
    target = random.randint(min_needed, min(4, len(candidates)))

    distractors: set[tuple[int, int]] = set()
    for cell in candidates:
        if len(distractors) >= target:
            break
        if far_from_reds(cell, red_cells | distractors):
            distractors.add(cell)

    return distractors


def _generate_scene(branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 24)
    w = random.randint(8, 16)
    start_col = random.randint(1, w - 2)
    start_dc = random.choice((-1, 1))

    base = _make_forced_trajectory(h, w, start_col, start_dc, set())
    if base is None:
        return None
    _, base_states, _ = base

    forced_rows: set[int] = set()
    hidden_row: int | None = None

    if branch == "hidden":
        hidden_candidates = [
            int(state["row"])
            for state in base_states
            if state["event"] == "none"
            and 5 <= int(state["row"]) <= h - 6
            and int(state["col"]) not in (0, w - 1)
        ]
        if not hidden_candidates:
            return None
        hidden_row = random.choice(hidden_candidates)
        forced_rows.add(hidden_row)

    current = _make_forced_trajectory(h, w, start_col, start_dc, forced_rows)
    if current is None:
        return None
    path, states, red_cells = current

    if branch == "hidden" and random.random() < 0.5:
        visible_candidates = [
            int(state["row"])
            for state in states
            if state["event"] == "none"
            and hidden_row is not None
            and int(state["row"]) <= hidden_row - 4
            and 3 <= int(state["row"]) <= h - 4
        ]
        if visible_candidates:
            extra_row = random.choice(visible_candidates)
            current = _make_forced_trajectory(h, w, start_col, start_dc, forced_rows | {extra_row})
            if current is None:
                return None
            path, states, red_cells = current
    elif branch in {"straight", "wall"} and random.random() < 0.45:
        visible_candidates = [
            int(state["row"])
            for state in states
            if state["event"] == "none" and 4 <= int(state["row"]) <= h - 5
        ]
        if visible_candidates:
            extra_row = random.choice(visible_candidates)
            current = _make_forced_trajectory(h, w, start_col, start_dc, {extra_row})
            if current is None:
                return None
            path, states, red_cells = current

    intervals = _interval_candidates(h, w, states)
    if not intervals[branch]:
        return None
    interval = random.choice(intervals[branch])
    box = _choose_box((h, w), path, interval, branch)

    distractors = _add_distractors(
        np.full((h, w), BLACK, dtype=int),
        path,
        states,
        red_cells,
        box,
    )
    all_reds = red_cells | distractors
    truth = _paint_truth((h, w), path, all_reds)
    inp = truth.copy()
    r0, c0, r1, c1 = box
    inp[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return inp, truth


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    preferred = BRANCH_CYCLE[_GENERATE_CALL_COUNT % len(BRANCH_CYCLE)]
    _GENERATE_CALL_COUNT += 1
    branch_order = [preferred] + [b for b in ("hidden", "straight", "wall") if b != preferred]

    for branch in branch_order:
        for _ in range(MAX_GENERATION_ATTEMPTS):
            scene = _generate_scene(branch)
            if scene is None:
                continue
            inp, truth = scene
            try:
                rebuilt = main(inp)
            except Exception:
                continue
            if np.array_equal(rebuilt, truth):
                return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
