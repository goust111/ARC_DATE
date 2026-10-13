import importlib.util
import random
from pathlib import Path

import numpy as np


ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("arc_utils_fable_8_030_5", ARC_UTILS_PATH)
_ARC = importlib.util.module_from_spec(_ARC_SPEC)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_SPEC.loader.exec_module(_ARC)

Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg
in_bounds = _ARC.in_bounds
neighbors4 = _ARC.neighbors4


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PATH_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_ATTEMPTS = 1200
SELF_CHECK_ATTEMPTS = 200


def _turn(dir_idx: int, turn: int) -> int:
    return (int(dir_idx) + int(turn)) % 4


def _dir_index(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    for idx, (rr, cc) in enumerate(DIRS):
        if (dr, dc) == (rr, cc):
            return idx
    raise AssertionError(f"non-cardinal step: {a} -> {b}")


def _compress_runs(dir_seq: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for dir_idx in dir_seq:
        if runs and runs[-1][0] == dir_idx:
            runs[-1] = (dir_idx, runs[-1][1] + 1)
        else:
            runs.append((dir_idx, 1))
    return runs


def _trace_path(path_mask: np.ndarray, marker_pos: tuple[int, int]) -> list[tuple[int, int]]:
    start_neighbors = [
        (nr, nc)
        for nr, nc in neighbors4(*marker_pos)
        if in_bounds(path_mask.shape, nr, nc) and bool(path_mask[nr, nc])
    ]
    assert len(start_neighbors) == 1, "marker must touch exactly one path endpoint"
    start = start_neighbors[0]

    degrees = {}
    for r, c in map(tuple, np.argwhere(path_mask)):
        nbrs = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if in_bounds(path_mask.shape, nr, nc) and bool(path_mask[nr, nc])
        ]
        degrees[(r, c)] = len(nbrs)
    assert sorted(degrees.values()).count(1) == 2, "path must be a simple chain"
    assert all(v in (1, 2) for v in degrees.values()), "path must not branch"

    ordered = [marker_pos, start]
    prev = marker_pos
    cur = start
    visited = {start}
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if in_bounds(path_mask.shape, nr, nc) and bool(path_mask[nr, nc]) and (nr, nc) != prev
        ]
        assert len(nxts) <= 1, "path must not branch"
        if not nxts:
            break
        nxt = nxts[0]
        assert nxt not in visited, "path must not loop in the visible residue"
        ordered.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    assert len(visited) == int(path_mask.sum()), "path trace must cover every path cell"
    return ordered


def _decode_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, Color.GREY)]
    assert len(colors) == 2, "expected exactly one path color and one marker color"

    counts = {color: int((g == color).sum()) for color in colors}
    path_color = max(colors, key=lambda color: counts[color])
    marker_color = min(colors, key=lambda color: counts[color])
    assert counts[marker_color] == 1, "marker must be a singleton"
    assert counts[path_color] >= 3, "path residue must contain at least three cells"

    marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))
    path_mask = g == path_color

    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[path_color],
        return_format="dict",
    )
    assert len(comps) == 1, "path must be one 4-connected component"

    ordered = _trace_path(path_mask, marker_pos)
    dir_seq = [_dir_index(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
    runs = _compress_runs(dir_seq)
    assert len(runs) == 3, "visible residue must contain exactly three segments"

    seg_dirs = [run[0] for run in runs]
    seg_lengths = [run[1] for run in runs]
    turn1 = (seg_dirs[1] - seg_dirs[0]) % 4
    turn2 = (seg_dirs[2] - seg_dirs[1]) % 4
    assert turn1 == turn2 and turn1 in (1, 3), "turn sense must be consistent"
    turn = 1 if turn1 == 1 else -1

    step = seg_lengths[1] - seg_lengths[0]
    assert step > 0, "planned lengths must grow"
    assert seg_lengths[2] - seg_lengths[1] == step, "first three segments must form an arithmetic progression"

    return {
        "grid": g,
        "bg": bg,
        "marker_pos": marker_pos,
        "marker_color": marker_color,
        "path_color": path_color,
        "segment_dirs": seg_dirs,
        "segment_lengths": seg_lengths,
        "turn": turn,
        "step": step,
        "endpoint": ordered[-1],
        "ordered_path": ordered[1:],
    }


def _simulate_from_decoded(decoded: dict) -> tuple[np.ndarray, dict]:
    out = np.array(decoded["grid"], copy=True)
    marker_pos = decoded["marker_pos"]
    marker_color = int(decoded["marker_color"])
    path_color = int(decoded["path_color"])
    turn = int(decoded["turn"])
    step = int(decoded["step"])
    dir_idx = int(decoded["segment_dirs"][-1])
    planned = int(decoded["segment_lengths"][-1]) + step
    current = tuple(decoded["endpoint"])

    blocked_turns = 0
    stopped_by_first_block = False
    stopped_by_boundary = False
    added = 0
    max_segments = out.shape[0] * out.shape[1] * 4 + 10

    for _ in range(max_segments):
        dir_idx = _turn(dir_idx, turn)
        drawn = 0
        while drawn < planned:
            nr = current[0] + DIRS[dir_idx][0]
            nc = current[1] + DIRS[dir_idx][1]
            if not in_bounds(out.shape, nr, nc):
                stopped_by_boundary = True
                out[marker_pos] = marker_color
                return out, {
                    "added_cells": added,
                    "blocked_turns": blocked_turns,
                    "stopped_by_first_block": stopped_by_first_block,
                    "stopped_by_boundary": stopped_by_boundary,
                }
            if int(out[nr, nc]) == Color.GREY:
                if drawn == 0:
                    stopped_by_first_block = True
                    out[marker_pos] = marker_color
                    return out, {
                        "added_cells": added,
                        "blocked_turns": blocked_turns,
                        "stopped_by_first_block": stopped_by_first_block,
                        "stopped_by_boundary": stopped_by_boundary,
                    }
                blocked_turns += 1
                break
            out[nr, nc] = path_color
            current = (nr, nc)
            drawn += 1
            added += 1
        planned += step

    out[marker_pos] = marker_color
    raise AssertionError("simulation exceeded conservative segment limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the arithmetic spiral from its three-segment residue while respecting grey blockers and canvas clipping.

    Input structure:
    - A black-background grid contains one singleton start marker, one monochrome path residue, and grey blocker pixels.
    - The residue is exactly the first three orthogonal segments of a polyline; adjacent segments turn by the same 90-degree sense.
    - Those three visible segment lengths are the first three planned lengths of an arithmetic progression, and none of them has been shortened by blockers.

    Rule:
    - Measure the first three segment lengths from the marker onward, infer the common difference, and infer whether turns are clockwise or counterclockwise.
    - Continue drawing from the residue endpoint: before each new segment, turn by the same 90-degree sense, and increase the planned segment length by the same difference.
    - If the next cell of a segment is grey, that segment ends immediately; after a mid-segment grey hit, turn into the next planned segment, but if the first cell after a turn is grey then the drawing stops. If the next step would leave the grid, stop after the in-bounds prefix.

    Key insight:
    - The arithmetic progression governs planned segment lengths, not the actually drawn lengths after blockers start truncating segments.
    """
    decoded = _decode_input(np.asarray(grid, dtype=int))
    out, _meta = _simulate_from_decoded(decoded)
    return out


def _available_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    dir_idx: int,
    occupied: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str]:
    cells: list[tuple[int, int]] = []
    cur = start
    while True:
        nr = cur[0] + DIRS[dir_idx][0]
        nc = cur[1] + DIRS[dir_idx][1]
        if not in_bounds(shape, nr, nc):
            return cells, "boundary"
        nxt = (nr, nc)
        if nxt in occupied or nxt in blockers:
            return cells, "occupied"
        cells.append(nxt)
        cur = nxt


def _draw_cells(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> tuple[int, int]:
    for cell in cells:
        occupied.add(cell)
    return cells[-1]


def _render_case(case: dict, *, residue_only: bool) -> np.ndarray:
    grid = np.full(case["shape"], Color.BLACK, dtype=int)
    for r, c in case["blockers"]:
        grid[r, c] = Color.GREY
    for r, c in case["path_cells"] if not residue_only else case["initial_path_cells"]:
        grid[r, c] = case["path_color"]
    mr, mc = case["marker_pos"]
    grid[mr, mc] = case["marker_color"]
    return grid


def _segment_lengths_from_case(case: dict) -> list[int]:
    return [case["a"], case["a"] + case["d"], case["a"] + 2 * case["d"]]


def _build_case() -> dict | None:
    rows = random.randint(16, 28)
    cols = random.randint(16, 28)
    shape = (rows, cols)

    path_color, marker_color = random.sample(PATH_COLORS, 2)
    turn = random.choice((-1, 1))
    dir0 = random.randrange(4)
    a = random.randint(1, 4)
    d = random.randint(1, 3)

    marker_pos = (
        random.randint(2, rows - 3),
        random.randint(2, cols - 3),
    )

    occupied = {marker_pos}
    blockers: set[tuple[int, int]] = set()
    path_cells: list[tuple[int, int]] = []
    initial_path_cells: list[tuple[int, int]] = []

    current = marker_pos
    dir_idx = dir0
    planned_lengths = [a, a + d, a + 2 * d]
    for planned in planned_lengths:
        cells, reason = _available_cells(shape, current, dir_idx, occupied, blockers)
        if reason != "boundary" and len(cells) < planned:
            return None
        if len(cells) < planned:
            return None
        drawn = cells[:planned]
        current = _draw_cells(drawn, occupied)
        path_cells.extend(drawn)
        initial_path_cells.extend(drawn)
        dir_idx = _turn(dir_idx, turn)

    preterminal_count = random.randint(1, 4)
    preterminal_modes = ["early"]
    for _ in range(preterminal_count - 1):
        preterminal_modes.append(random.choice(["early", "complete"]))
    random.shuffle(preterminal_modes)
    final_mode = random.choice(["boundary", "blocked_first"])

    current_dir = dir_idx
    planned = a + 3 * d
    for mode in preterminal_modes:
        cells, _reason = _available_cells(shape, current, current_dir, occupied, blockers)
        if mode == "complete":
            if len(cells) < planned:
                return None
            drawn = cells[:planned]
            current = _draw_cells(drawn, occupied)
            path_cells.extend(drawn)
        else:
            max_drawn = min(planned - 1, len(cells) - 1)
            if max_drawn < 1:
                return None
            drawn_len = random.randint(1, max_drawn)
            drawn = cells[:drawn_len]
            blocker = cells[drawn_len]
            current = _draw_cells(drawn, occupied)
            blockers.add(blocker)
            path_cells.extend(drawn)
        current_dir = _turn(current_dir, turn)
        planned += d

    cells, reason = _available_cells(shape, current, current_dir, occupied, blockers)
    if final_mode == "boundary":
        if reason != "boundary" or not (1 <= len(cells) < planned):
            return None
        current = _draw_cells(cells, occupied)
        path_cells.extend(cells)
    else:
        if not cells:
            return None
        blockers.add(cells[0])

    extra_blockers = random.randint(0, 4)
    extras_added = 0
    while extras_added < extra_blockers:
        cell = (random.randrange(rows), random.randrange(cols))
        if cell in occupied or cell in blockers:
            continue
        blockers.add(cell)
        extras_added += 1

    if len(path_cells) == len(initial_path_cells):
        return None
    if len(blockers) == 0:
        return None
    if len(occupied) + len(blockers) >= (rows * cols) // 2:
        return None

    case = {
        "shape": shape,
        "path_color": path_color,
        "marker_color": marker_color,
        "marker_pos": marker_pos,
        "blockers": blockers,
        "path_cells": path_cells,
        "initial_path_cells": initial_path_cells,
        "a": a,
        "d": d,
        "turn": turn,
        "dir0": dir0,
        "final_mode": final_mode,
        "preterminal_modes": preterminal_modes,
    }

    input_grid = _render_case(case, residue_only=True)
    output_grid = _render_case(case, residue_only=False)

    decoded = _decode_input(input_grid)
    if decoded["segment_lengths"] != _segment_lengths_from_case(case):
        return None
    if decoded["turn"] != turn:
        return None

    solved = main(input_grid)
    if not np.array_equal(solved, output_grid):
        return None

    sim_out, sim_meta = _simulate_from_decoded(decoded)
    if not np.array_equal(sim_out, output_grid):
        return None
    if sim_meta["added_cells"] <= 0 or sim_meta["blocked_turns"] < 1:
        return None
    if final_mode == "boundary" and not sim_meta["stopped_by_boundary"]:
        return None
    if final_mode == "blocked_first" and not sim_meta["stopped_by_first_block"]:
        return None
    if final_mode == "boundary" and sim_meta["stopped_by_first_block"]:
        return None

    return case


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return _render_case(case, residue_only=True)
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")


def _self_check() -> None:
    sigs = set()
    saw_boundary = False
    saw_first_block = False
    saw_ccw = False
    saw_cw = False
    saw_short_first = False
    saw_longer_step = False
    accepted = 0

    for _ in range(SELF_CHECK_ATTEMPTS * 8):
        case = _build_case()
        if case is None:
            continue
        accepted += 1
        inp = _render_case(case, residue_only=True)
        out = _render_case(case, residue_only=False)
        solved = main(inp)
        assert isinstance(inp, np.ndarray) and inp.ndim == 2 and max(inp.shape) <= 30
        assert isinstance(solved, np.ndarray) and solved.ndim == 2 and max(solved.shape) <= 30
        assert np.array_equal(solved, out)

        decoded = _decode_input(inp)
        assert decoded["segment_lengths"] == _segment_lengths_from_case(case)
        assert decoded["turn"] == case["turn"]
        _, meta = _simulate_from_decoded(decoded)
        assert meta["added_cells"] > 0
        assert meta["blocked_turns"] >= 1

        saw_boundary |= meta["stopped_by_boundary"]
        saw_first_block |= meta["stopped_by_first_block"]
        saw_ccw |= case["turn"] == -1
        saw_cw |= case["turn"] == 1
        saw_short_first |= case["a"] == 1
        saw_longer_step |= case["d"] >= 2
        sigs.add((inp.shape, inp.tobytes()))
        if accepted >= SELF_CHECK_ATTEMPTS:
            break

    assert accepted >= SELF_CHECK_ATTEMPTS
    assert len(sigs) >= 40
    assert saw_boundary and saw_first_block
    assert saw_ccw and saw_cw
    assert saw_short_first and saw_longer_step


if __name__ == "__main__":
    _self_check()
