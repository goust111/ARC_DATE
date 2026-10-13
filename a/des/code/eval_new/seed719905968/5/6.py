from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


AZURE = 8
GREY = 5
GATE_COLORS = [color for color in range(10) if color not in (AZURE, GREY)]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _segment_info(coords: np.ndarray, color: int) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if len(set(map(int, rows))) == 1:
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        length = int(c1 - c0 + 1)
        if length == len(coords) and 2 <= length <= 4:
            return {
                "color": int(color),
                "axis": "h",
                "fixed": row,
                "start": c0,
                "end": c1,
                "length": length,
                "coords": coords,
            }
    if len(set(map(int, cols))) == 1:
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        length = int(r1 - r0 + 1)
        if length == len(coords) and 2 <= length <= 4:
            return {
                "color": int(color),
                "axis": "v",
                "fixed": col,
                "start": r0,
                "end": r1,
                "length": length,
                "coords": coords,
            }
    return None


def _gate_from_segments(a: dict, b: dict, grid: np.ndarray) -> dict | None:
    if a["axis"] != b["axis"]:
        return None

    if a["axis"] == "v":
        if a["fixed"] != b["fixed"]:
            return None
        upper, lower = sorted((a, b), key=lambda seg: int(seg["start"]))
        if int(upper["end"]) + 2 != int(lower["start"]):
            return None
        opening = (int(upper["end"]) + 1, int(upper["fixed"]))
        if int(grid[opening]) != AZURE:
            return None
        return {
            "color": int(a["color"]),
            "channel_axis": "h",
            "opening": opening,
            "segments": (upper, lower),
        }

    if a["fixed"] != b["fixed"]:
        return None
    left, right = sorted((a, b), key=lambda seg: int(seg["start"]))
    if int(left["end"]) + 2 != int(right["start"]):
        return None
    opening = (int(left["fixed"]), int(left["end"]) + 1)
    if int(grid[opening]) != AZURE:
        return None
    return {
        "color": int(a["color"]),
        "channel_axis": "v",
        "opening": opening,
        "segments": (left, right),
    }


def _pair_segments_into_gates(segments: list[dict], grid: np.ndarray) -> list[dict]:
    candidates: list[tuple[int, int, dict]] = []
    for i, seg_a in enumerate(segments):
        for j, seg_b in enumerate(segments[i + 1 :], start=i + 1):
            gate = _gate_from_segments(seg_a, seg_b, grid)
            if gate is not None:
                candidates.append((int(i), int(j), gate))

    solutions: list[list[dict]] = []
    used = [False] * len(segments)

    def search(current: list[dict]) -> None:
        if len(solutions) > 1:
            return
        first_unused = next((idx for idx, flag in enumerate(used) if not flag), None)
        if first_unused is None:
            solutions.append(list(current))
            return
        for i, j, gate in candidates:
            if i != first_unused or used[j]:
                continue
            used[i] = True
            used[j] = True
            current.append(gate)
            search(current)
            current.pop()
            used[i] = False
            used[j] = False

    search([])
    assert len(solutions) == 1, "gate segments must pair into a unique set of gates"
    return solutions[0]


def _channel_cells(gate_a: dict, gate_b: dict, grid: np.ndarray) -> list[tuple[int, int]]:
    assert gate_a["channel_axis"] == gate_b["channel_axis"], "paired gates must face along one axis"
    axis = gate_a["channel_axis"]
    if axis == "h":
        row_a, col_a = gate_a["opening"]
        row_b, col_b = gate_b["opening"]
        assert row_a == row_b, "horizontal channel openings must share a row"
        left = min(int(col_a), int(col_b))
        right = max(int(col_a), int(col_b))
        cells = [(int(row_a), c) for c in range(left, right + 1)]
    else:
        row_a, col_a = gate_a["opening"]
        row_b, col_b = gate_b["opening"]
        assert col_a == col_b, "vertical channel openings must share a column"
        top = min(int(row_a), int(row_b))
        bottom = max(int(row_a), int(row_b))
        cells = [(r, int(col_a)) for r in range(top, bottom + 1)]

    for r, c in cells:
        assert int(grid[r, c]) in (AZURE, GREY), "channel may contain only azure or grey"
    return cells


def _fill_channel(out: np.ndarray, gate_a: dict, gate_b: dict, color: int) -> None:
    axis = gate_a["channel_axis"]
    if axis == "h":
        row = int(gate_a["opening"][0])
        left = min(int(gate_a["opening"][1]), int(gate_b["opening"][1]))
        right = max(int(gate_a["opening"][1]), int(gate_b["opening"][1]))
        for c in range(left, right + 1):
            if int(out[row, c]) == GREY:
                break
            assert int(out[row, c]) in (AZURE, int(color))
            out[row, c] = int(color)
        for c in range(right, left - 1, -1):
            if int(out[row, c]) == GREY:
                break
            assert int(out[row, c]) in (AZURE, int(color))
            out[row, c] = int(color)
        return

    col = int(gate_a["opening"][1])
    top = min(int(gate_a["opening"][0]), int(gate_b["opening"][0]))
    bottom = max(int(gate_a["opening"][0]), int(gate_b["opening"][0]))
    for r in range(top, bottom + 1):
        if int(out[r, col]) == GREY:
            break
        assert int(out[r, col]) in (AZURE, int(color))
        out[r, col] = int(color)
    for r in range(bottom, top - 1, -1):
        if int(out[r, col]) == GREY:
            break
        assert int(out[r, col]) in (AZURE, int(color))
        out[r, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each straight channel between same-color broken gates from both openings until grey blockers stop the spread.

    Input structure:
    - The background is azure.
    - Each gate color appears as exactly two gates; every gate is a broken horizontal
      or vertical line made from two 2-4 cell segments with a single azure opening cell.
    - The two gates of one color face each other across one straight horizontal or
      vertical azure channel, and some channel cells may be grey blockers.
    - Channels for different colors do not cross or overlap.

    Rule:
    - Reconstruct each gate from its two same-color line segments and pair the two
      gates of that color by the straight line between their opening cells.
    - Starting at both opening cells, paint azure channel cells with the gate color
      until a grey blocker is reached from that side.
    - Grey blockers stay grey, and the gate segments themselves stay unchanged.
    - If a channel has no blocker, the whole opening-to-opening path is filled.

    Key insight:
    - The missing single cell inside each broken line segment is the channel endpoint,
      so the task reduces to detecting broken-line gates and running two 1D floods
      along the aligned opening-to-opening corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    colors = [int(color) for color in np.unique(g) if int(color) not in (AZURE, GREY)]
    assert colors, "expected at least one gate color"
    out = g.copy()

    for color in colors:
        segments = []
        for comp in _color_components(g, color):
            info = _segment_info(comp, color)
            assert info is not None, "every gate piece must be a straight 2-4 cell segment"
            segments.append(info)
        assert len(segments) == 4, "each color must contribute exactly two gates"
        gates = _pair_segments_into_gates(segments, g)
        assert len(gates) == 2, "each color must form exactly two gates"
        _channel_cells(gates[0], gates[1], g)
        _fill_channel(out, gates[0], gates[1], color)

    return out


def _expand(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    halo.add((nr, nc))
    return halo


def _horizontal_pair(
    height: int,
    width: int,
    color: int,
    with_obstacle: bool,
) -> dict | None:
    if height < 10 or width < 10:
        return None

    row = random.randint(4, height - 5)
    left_col = random.randint(2, width - 8)
    right_col = random.randint(left_col + 5, width - 3)

    def sample_span(up_cap: int, down_cap: int) -> tuple[int, int]:
        up = random.randint(2, min(4, up_cap))
        down = random.randint(2, min(4, down_cap))
        return up, down

    up_left, down_left = sample_span(row, height - row - 1)
    up_right, down_right = sample_span(row, height - row - 1)

    bars: set[tuple[int, int]] = set()
    for r in range(row - up_left, row):
        bars.add((r, left_col))
    for r in range(row + 1, row + down_left + 1):
        bars.add((r, left_col))
    for r in range(row - up_right, row):
        bars.add((r, right_col))
    for r in range(row + 1, row + down_right + 1):
        bars.add((r, right_col))

    channel = {(row, c) for c in range(left_col, right_col + 1)}
    obstacles: set[tuple[int, int]] = set()
    if with_obstacle:
        obstacle_col = random.randint(left_col + 2, right_col - 2)
        obstacles.add((row, obstacle_col))

    paint = {cell: int(color) for cell in bars}
    for cell in obstacles:
        paint[cell] = GREY
    return {"reserve": bars | channel | obstacles, "paint": paint}


def _vertical_pair(
    height: int,
    width: int,
    color: int,
    with_obstacle: bool,
) -> dict | None:
    if height < 10 or width < 10:
        return None

    col = random.randint(4, width - 5)
    top_row = random.randint(2, height - 8)
    bottom_row = random.randint(top_row + 5, height - 3)

    def sample_span(left_cap: int, right_cap: int) -> tuple[int, int]:
        left = random.randint(2, min(4, left_cap))
        right = random.randint(2, min(4, right_cap))
        return left, right

    left_top, right_top = sample_span(col, width - col - 1)
    left_bottom, right_bottom = sample_span(col, width - col - 1)

    bars: set[tuple[int, int]] = set()
    for c in range(col - left_top, col):
        bars.add((top_row, c))
    for c in range(col + 1, col + right_top + 1):
        bars.add((top_row, c))
    for c in range(col - left_bottom, col):
        bars.add((bottom_row, c))
    for c in range(col + 1, col + right_bottom + 1):
        bars.add((bottom_row, c))

    channel = {(r, col) for r in range(top_row, bottom_row + 1)}
    obstacles: set[tuple[int, int]] = set()
    if with_obstacle:
        obstacle_row = random.randint(top_row + 2, bottom_row - 2)
        obstacles.add((obstacle_row, col))

    paint = {cell: int(color) for cell in bars}
    for cell in obstacles:
        paint[cell] = GREY
    return {"reserve": bars | channel | obstacles, "paint": paint}


def _place_pair(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    color: int,
    channel_axis: str,
    with_obstacle: bool,
) -> bool:
    height, width = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        if channel_axis == "h":
            candidate = _horizontal_pair(height, width, color, with_obstacle)
        else:
            candidate = _vertical_pair(height, width, color, with_obstacle)
        if candidate is None:
            continue

        reserve = candidate["reserve"]
        if any(cell in blocked for cell in reserve):
            continue

        for (r, c), value in candidate["paint"].items():
            assert int(grid[r, c]) == AZURE
            grid[r, c] = int(value)

        blocked.update(_expand(reserve, height, width))
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        grid = np.full((height, width), AZURE, dtype=int)

        if min(height, width) >= 22:
            max_pairs = 3
        elif min(height, width) >= 17:
            max_pairs = 2
        else:
            max_pairs = 1
        pair_count = random.randint(1, max_pairs)

        focus_branch = random.choice(
            [("h", False), ("h", True), ("v", False), ("v", True)]
        )
        branch_specs = [focus_branch]
        for _idx in range(pair_count - 1):
            branch_specs.append(random.choice([("h", False), ("h", True), ("v", False), ("v", True)]))

        colors = random.sample(GATE_COLORS, pair_count)
        blocked: set[tuple[int, int]] = set()
        ok = True
        for color, (channel_axis, with_obstacle) in zip(colors, branch_specs, strict=False):
            if not _place_pair(grid, blocked, color, channel_axis, with_obstacle):
                ok = False
                break
        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("generate_input: failed to build a valid gate-channel puzzle")
