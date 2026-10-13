from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_MIN = 13
GRID_MAX = 24
FRAME_OUTER_MIN = 7
FRAME_OUTER_MAX = 16
MIN_INTERIOR = 5
MAX_SEEDS = 5
GEN_RETRIES = 500
MODES = ("boundary", "collision", "collision", "collision", "collision", "midline")


@dataclass(frozen=True)
class Candidate:
    pos: tuple[int, int]
    direction: tuple[int, int]
    steps_to_boundary: int
    trace: frozenset[tuple[int, int]]
    on_midline: bool


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == GREY)
    if pts.size == 0:
        raise ValueError("missing grey frame")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _interior_bbox(frame_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _inside_interior(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    return ir0 <= r <= ir1 and ic0 <= c <= ic1


def _midline_flags(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> tuple[bool, bool]:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    row_on = (ir0 + ir1) % 2 == 0 and r == (ir0 + ir1) // 2
    col_on = (ic0 + ic1) % 2 == 0 and c == (ic0 + ic1) // 2
    return row_on, col_on


def _on_midline(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> bool:
    row_on, col_on = _midline_flags(r, c, frame_bbox)
    return row_on or col_on


def _direction_for_cell(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    if _on_midline(r, c, frame_bbox):
        return 1, 1
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    row_mid = (ir0 + ir1) / 2.0
    col_mid = (ic0 + ic1) / 2.0
    dr = -1 if r < row_mid else 1
    dc = -1 if c < col_mid else 1
    return dr, dc


def _simulate_growth(
    grid: np.ndarray, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_bbox = _frame_bbox(g)
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)

    records: list[dict[str, object]] = []
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            value = int(g[r, c])
            if value in (BLACK, GREY):
                continue
            records.append(
                {
                    "color": value,
                    "start": (r, c),
                    "current": (r, c),
                    "direction": _direction_for_cell(r, c, frame_bbox),
                    "steps": 0,
                    "branches": [],
                    "stop_reason": None,
                    "on_midline": _on_midline(r, c, frame_bbox),
                }
            )

    active = list(range(len(records)))
    while active:
        proposals: dict[tuple[int, int], list[int]] = {}
        blocked: dict[int, str] = {}

        for idx in active:
            rec = records[idx]
            r, c = rec["current"]
            dr, dc = rec["direction"]
            nr, nc = r + dr, c + dc
            if not _inside_interior(nr, nc, frame_bbox):
                blocked[idx] = "boundary"
            elif int(out[nr, nc]) != BLACK:
                blocked[idx] = "occupied"
            else:
                proposals.setdefault((nr, nc), []).append(idx)

        moved: list[tuple[int, tuple[int, int]]] = []
        for target, indices in proposals.items():
            if len(indices) == 1:
                moved.append((indices[0], target))
            else:
                for idx in indices:
                    blocked[idx] = "occupied"

        for idx, reason in blocked.items():
            rec = records[idx]
            r, c = rec["current"]
            out[r, c] = YELLOW
            rec["stop_reason"] = reason

        next_active: list[int] = []
        for idx, (nr, nc) in sorted(moved, key=lambda item: (item[0], item[1][0], item[1][1])):
            rec = records[idx]
            rec["current"] = (nr, nc)
            rec["steps"] = int(rec["steps"]) + 1
            out[nr, nc] = int(rec["color"])
            next_active.append(idx)

        branch_proposals: dict[tuple[int, int], list[int]] = {}
        for idx in next_active:
            rec = records[idx]
            if int(rec["steps"]) % 2 != 0:
                continue
            r, c = rec["current"]
            dr, dc = rec["direction"]
            for br, bc in ((r + dr, c), (r, c + dc)):
                if _inside_interior(br, bc, frame_bbox) and int(out[br, bc]) == BLACK:
                    branch_proposals.setdefault((br, bc), []).append(idx)

        for (br, bc), indices in branch_proposals.items():
            if len(indices) != 1:
                continue
            idx = indices[0]
            out[br, bc] = int(records[idx]["color"])
            records[idx]["branches"].append((br, bc))

        active = next_active

    if return_meta:
        return out, records
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate_growth(grid)


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _predict_trace(start: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> Candidate:
    dr, dc = _direction_for_cell(start[0], start[1], frame_bbox)
    trace = {start}
    r, c = start
    steps = 0
    while True:
        nr, nc = r + dr, c + dc
        if not _inside_interior(nr, nc, frame_bbox):
            break
        r, c = nr, nc
        steps += 1
        trace.add((r, c))
        if steps % 2 == 0:
            for br, bc in ((r + dr, c), (r, c + dc)):
                if _inside_interior(br, bc, frame_bbox):
                    trace.add((br, bc))
    return Candidate(
        pos=start,
        direction=(dr, dc),
        steps_to_boundary=steps,
        trace=frozenset(trace),
        on_midline=_on_midline(start[0], start[1], frame_bbox),
    )


def _enumerate_candidates(frame_bbox: tuple[int, int, int, int]) -> list[Candidate]:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    candidates: list[Candidate] = []
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            cand = _predict_trace((r, c), frame_bbox)
            if cand.steps_to_boundary >= 1:
                candidates.append(cand)
    return candidates


def _surprising_midline(pos: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    if not _on_midline(pos[0], pos[1], frame_bbox):
        return False
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    row_mid = (ir0 + ir1) / 2.0
    col_mid = (ic0 + ic1) / 2.0
    return pos[0] < row_mid or pos[1] < col_mid


def _collision_options(candidates: list[Candidate]) -> list[tuple[Candidate, Candidate]]:
    by_pos = {cand.pos: cand for cand in candidates if not cand.on_midline}
    options: list[tuple[Candidate, Candidate]] = []
    for cand in candidates:
        if cand.on_midline or cand.steps_to_boundary < 3:
            continue
        dr, dc = cand.direction
        for gap in range(3, min(4, cand.steps_to_boundary) + 1):
            other = by_pos.get((cand.pos[0] + gap * dr, cand.pos[1] + gap * dc))
            if other is not None and other.direction == cand.direction:
                options.append((cand, other))
    return options


def _spaced_enough(pos: tuple[int, int], chosen_positions: set[tuple[int, int]]) -> bool:
    for other in chosen_positions:
        if max(abs(pos[0] - other[0]), abs(pos[1] - other[1])) < 2:
            return False
    return True


def _candidate_ok(
    cand: Candidate, chosen_positions: set[tuple[int, int]], reserved_cells: set[tuple[int, int]]
) -> bool:
    return cand.pos not in chosen_positions and _spaced_enough(cand.pos, chosen_positions) and cand.trace.isdisjoint(
        reserved_cells
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    max_frame_h = min(FRAME_OUTER_MAX, height - 2)
    max_frame_w = min(FRAME_OUTER_MAX, width - 2)
    if max_frame_h < FRAME_OUTER_MIN or max_frame_w < FRAME_OUTER_MIN:
        return None

    frame_h = rng.randint(FRAME_OUTER_MIN, max_frame_h)
    frame_w = rng.randint(FRAME_OUTER_MIN, max_frame_w)
    if frame_h - 2 < MIN_INTERIOR or frame_w - 2 < MIN_INTERIOR:
        return None

    r0 = rng.randint(1, height - frame_h - 1)
    c0 = rng.randint(1, width - frame_w - 1)
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    interior_area = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)

    candidates = _enumerate_candidates(frame_bbox)
    if len(candidates) < 2:
        return None

    mode = rng.choice(MODES)
    chosen: list[Candidate] = []
    chosen_positions: set[tuple[int, int]] = set()
    reserved_cells: set[tuple[int, int]] = set()

    if mode == "collision":
        options = _collision_options(candidates)
        if not options:
            return None
        first, second = rng.choice(options)
        chosen.extend([first, second])
        chosen_positions.update((first.pos, second.pos))
        reserved_cells.update(first.trace)
        reserved_cells.update(second.trace)
    elif mode == "midline":
        midline_pool = [cand for cand in candidates if cand.on_midline and cand.steps_to_boundary >= 2]
        special_pool = [cand for cand in midline_pool if _surprising_midline(cand.pos, frame_bbox)]
        if not midline_pool:
            return None
        lead = rng.choice(special_pool or midline_pool)
        chosen.append(lead)
        chosen_positions.add(lead.pos)
        reserved_cells.update(lead.trace)

    max_seed_count = min(MAX_SEEDS, len(SEED_COLORS), max(2, 2 + interior_area // 30))
    target_count = rng.randint(2, max_seed_count)
    if mode != "boundary":
        target_count = max(target_count, 2)
    if len(chosen) > target_count:
        target_count = len(chosen)

    used_dirs = {cand.direction for cand in chosen}
    while len(chosen) < target_count:
        valid = [cand for cand in candidates if _candidate_ok(cand, chosen_positions, reserved_cells)]
        if mode != "midline":
            valid = [cand for cand in valid if not cand.on_midline]
        if not valid:
            return None
        fresh = [cand for cand in valid if cand.direction not in used_dirs]
        long = [cand for cand in (fresh or valid) if cand.steps_to_boundary >= 2]
        pool = long or fresh or valid
        cand = rng.choice(pool)
        chosen.append(cand)
        chosen_positions.add(cand.pos)
        reserved_cells.update(cand.trace)
        used_dirs.add(cand.direction)

    if not any(cand.steps_to_boundary >= 2 for cand in chosen):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_frame(grid, frame_bbox)
    colors = list(rng.sample(SEED_COLORS, len(chosen)))
    rng.shuffle(colors)
    for cand, color in zip(chosen, colors):
        grid[cand.pos] = color

    out, meta = _simulate_growth(grid, return_meta=True)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == YELLOW) != len(chosen):
        return None
    if not any(int(rec["steps"]) >= 2 and rec["branches"] for rec in meta):
        return None
    if mode == "boundary" and any(rec["stop_reason"] != "boundary" for rec in meta):
        return None
    if mode == "collision" and not any(rec["stop_reason"] == "occupied" for rec in meta):
        return None
    if mode == "midline":
        if not any(bool(rec["on_midline"]) for rec in meta):
            return None
        if not any(bool(rec["on_midline"]) and _surprising_midline(rec["start"], frame_bbox) for rec in meta):
            return None
    if len(meta) >= 3 and len({tuple(rec["direction"]) for rec in meta}) < 2:
        return None
    if np.count_nonzero(out[ir0 : ir1 + 1, ic0 : ic1 + 1] != BLACK) > int(interior_area * 0.65):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
