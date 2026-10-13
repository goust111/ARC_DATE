from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS: dict[str, tuple[int, int]] = {
    "L": (0, -1),
    "R": (0, 1),
    "U": (-1, 0),
    "D": (1, 0),
}
AXIS_DIRS: dict[str, tuple[str, str]] = {
    "H": ("L", "R"),
    "V": ("U", "D"),
}
PALETTE = tuple(range(1, 10))
GRID_MIN = 14
GRID_MAX = 28
LANE_GAP = 4
GATE_GAP = 3
GEN_ATTEMPTS = 300
LANE_ATTEMPTS = 200


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    winners = [int(v) for v, n in zip(vals, counts) if int(n) == int(counts.max())]
    assert len(winners) == 1, "expected a unique dominant background color"
    return winners[0]


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _arrow_cells(center: tuple[int, int], axis: str) -> list[tuple[int, int]]:
    r, c = center
    if axis == "H":
        return [(r - 1, c), (r, c), (r + 1, c)]
    if axis == "V":
        return [(r, c - 1), (r, c), (r, c + 1)]
    raise AssertionError(f"unknown axis {axis!r}")


def _extract_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid, bg):
        coords = comp["coords"]
        assert isinstance(coords, list)
        if len(coords) == 1:
            continue
        assert len(coords) == 3, "every multi-cell foreground object must be a three-pixel arrow"
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
        height = int(r1) - int(r0) + 1
        width = int(c1) - int(c0) + 1
        pts = {(int(r), int(c)) for r, c in coords}
        if (height, width) == (3, 1):
            center = (int(r0) + 1, int(c0))
            expected = set(_arrow_cells(center, "H"))
            axis = "H"
        elif (height, width) == (1, 3):
            center = (int(r0), int(c0) + 1)
            expected = set(_arrow_cells(center, "V"))
            axis = "V"
        else:
            raise AssertionError("arrow must be a straight three-pixel line")
        assert pts == expected, "arrow must fill its whole three-pixel line"
        arrows.append(
            {
                "axis": axis,
                "center": center,
                "cells": sorted(pts),
                "color": int(comp["color"]),
            }
        )
    assert arrows, "expected at least one arrow"
    arrows.sort(key=lambda a: (int(a["center"][0]), int(a["center"][1])))
    return arrows


def _add_entry(
    entry_map: dict[tuple[int, int, str], tuple[int, int]],
    key: tuple[int, int, str],
    source: int,
    target: int,
) -> None:
    prev = entry_map.get(key)
    if prev is None:
        entry_map[key] = (int(source), int(target))
        return
    assert prev == (int(source), int(target)), "ambiguous gate entry"


def _extract_gates(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], dict[tuple[int, int, str], tuple[int, int]]]:
    h, w = map(int, grid.shape)
    gates: list[dict[str, object]] = []
    entry_map: dict[tuple[int, int, str], tuple[int, int]] = {}

    for r in range(h - 2):
        for c in range(w - 1):
            lt = int(grid[r, c])
            rt = int(grid[r, c + 1])
            lb = int(grid[r + 2, c])
            rb = int(grid[r + 2, c + 1])
            if min(lt, rt, lb, rb) == bg:
                continue
            if not (lt == lb and rt == rb and lt != rt):
                continue
            if not (int(grid[r + 1, c]) == bg and int(grid[r + 1, c + 1]) == bg):
                continue
            gates.append(
                {
                    "axis": "H",
                    "lane": r + 1,
                    "left": c,
                    "left_color": lt,
                    "right_color": rt,
                }
            )
            _add_entry(entry_map, (r + 1, c, "R"), lt, rt)
            _add_entry(entry_map, (r + 1, c + 1, "L"), rt, lt)

    for r in range(h - 1):
        for c in range(w - 2):
            tl = int(grid[r, c])
            tr = int(grid[r, c + 2])
            bl = int(grid[r + 1, c])
            br = int(grid[r + 1, c + 2])
            if min(tl, tr, bl, br) == bg:
                continue
            if not (tl == tr and bl == br and tl != bl):
                continue
            if not (int(grid[r, c + 1]) == bg and int(grid[r + 1, c + 1]) == bg):
                continue
            gates.append(
                {
                    "axis": "V",
                    "lane": c + 1,
                    "top": r,
                    "top_color": tl,
                    "bottom_color": bl,
                }
            )
            _add_entry(entry_map, (r, c + 1, "D"), tl, bl)
            _add_entry(entry_map, (r + 1, c + 1, "U"), bl, tl)

    assert gates, "expected at least one gate"
    return gates, entry_map


def _first_event(
    center: tuple[int, int],
    color: int,
    direction: str,
    entry_map: dict[tuple[int, int, str], tuple[int, int]],
    shape: tuple[int, int],
) -> dict[str, int | str]:
    r, c = center
    dr, dc = DIRS[direction]
    steps = 0
    while True:
        nr = r + dr
        nc = c + dc
        if not _inside(shape, nr, nc):
            return {"kind": "edge", "steps": steps + 1}
        steps += 1
        pair = entry_map.get((nr, nc, direction))
        if pair is not None:
            return {
                "kind": "match" if int(pair[0]) == color else "mismatch",
                "steps": steps,
            }
        r, c = nr, nc


def _infer_direction(
    arrow: dict[str, object],
    entry_map: dict[tuple[int, int, str], tuple[int, int]],
    shape: tuple[int, int],
) -> str:
    axis = str(arrow["axis"])
    center = tuple(arrow["center"])  # type: ignore[arg-type]
    color = int(arrow["color"])
    matches: list[tuple[str, int]] = []
    for direction in AXIS_DIRS[axis]:
        event = _first_event(center, color, direction, entry_map, shape)
        if str(event["kind"]) == "match":
            matches.append((direction, int(event["steps"])))
    assert len(matches) == 1, "each arrow must have exactly one direction with a matching first gate"
    return matches[0][0]


def _trace_arrow(
    center: tuple[int, int],
    color: int,
    direction: str,
    entry_map: dict[tuple[int, int, str], tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], int]:
    r, c = center
    dr, dc = DIRS[direction]
    path = [(int(r), int(c))]
    current = int(color)
    while True:
        nr = r + dr
        nc = c + dc
        if not _inside(shape, nr, nc):
            return path, current
        pair = entry_map.get((nr, nc, direction))
        if pair is not None:
            source, target = map(int, pair)
            if source != current:
                return path, current
            current = target
        r, c = nr, nc
        path.append((int(r), int(c)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fly each colored arrow along its lane, letting matching two-sided gates recolor it before it stops.

    Input structure:
    - The background is black and the moving objects are monochrome three-pixel arrows.
    - A horizontal-moving arrow is a vertical three-pixel line; a vertical-moving arrow is a horizontal three-pixel line.
    - Gates are four-pixel two-color frames: horizontal gates have two short horizontal bars with a one-row channel between them, while vertical gates have two short vertical bars with a one-column channel between them.

    Rule:
    - Treat the middle pixel of each arrow as its moving center and infer its travel direction from the unique side whose first encountered gate faces the arrow with the same color.
    - Move that center one cell at a time. When it is about to enter a gate channel, compare the arrow color with the source side of that gate; a match allows passage and changes the arrow color to the gate's far-side color, while a mismatch stops the arrow immediately before the gate.
    - If no blocking mismatch occurs, the arrow keeps moving until its center reaches the grid edge.
    - Erase the original arrow, draw the full center path as a one-pixel line in the arrow's final color, keep all gates unchanged, and redraw the final arrow at its stopping position in that same final color.

    Key insight:
    - The gate's near-side color is both a permission check and a state transition, so solving the puzzle requires tracing color changes gate by gate rather than just following a straight path to the edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)

    arrows = _extract_arrows(g, bg)
    _gates, entry_map = _extract_gates(g, bg)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:  # type: ignore[assignment]
            out[int(r), int(c)] = bg

    for arrow in arrows:
        center = tuple(arrow["center"])  # type: ignore[arg-type]
        direction = _infer_direction(arrow, entry_map, g.shape)
        path, final_color = _trace_arrow(center, int(arrow["color"]), direction, entry_map, g.shape)
        for r, c in path:
            prev = int(out[r, c])
            assert prev in (bg, final_color), "arrow paths must not overwrite gates or other arrows"
            out[r, c] = final_color
        for r, c in _arrow_cells(path[-1], str(arrow["axis"])):
            assert _inside(g.shape, r, c), "final arrow must fit inside the grid"
            prev = int(out[r, c])
            assert prev in (bg, final_color), "final arrow must not overwrite gates or other arrows"
            out[r, c] = final_color
    return out


def _sample_spaced_positions(
    low: int, high: int, count: int, gap: int, rng: random.Random
) -> list[int] | None:
    if count <= 0:
        return []
    if high < low:
        return None
    candidates = list(range(low, high + 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if all(abs(value - prev) >= gap for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _pick_color(rng: random.Random, forbidden: set[int]) -> int:
    choices = [color for color in PALETTE if color not in forbidden]
    assert choices, "palette exhausted"
    return int(rng.choice(choices))


def _lane_color_sequence(
    direction: str, gate_count: int, mode: str, rng: random.Random
) -> tuple[int, list[tuple[int, int]]]:
    current = int(rng.choice(PALETTE))
    arrow_color = current
    pairs: list[tuple[int, int]] = []
    for i in range(gate_count):
        is_block = mode == "block" and i == gate_count - 1
        if is_block:
            source = _pick_color(rng, {current})
            target = _pick_color(rng, {source})
        else:
            source = current
            target = _pick_color(rng, {current})
            current = target
        if direction == "R":
            pairs.append((source, target))
        elif direction == "L":
            pairs.append((target, source))
        elif direction == "D":
            pairs.append((source, target))
        elif direction == "U":
            pairs.append((target, source))
        else:
            raise AssertionError(f"unknown direction {direction!r}")
    return arrow_color, pairs


def _lane_modes(count: int, rng: random.Random) -> list[str]:
    if count == 2:
        modes = ["edge", "block"]
    elif count == 3:
        modes = ["edge", "block", rng.choice(["edge", "block"])]
    else:
        modes = ["edge", "edge", "block", "block"][:count]
    rng.shuffle(modes)
    return modes


def _lane_directions(axis: str, count: int, rng: random.Random) -> list[str]:
    base = ["L", "R"] if axis == "H" else ["U", "D"]
    dirs = [base[i % 2] for i in range(count)]
    while len(dirs) < count:
        dirs.append(rng.choice(base))
    rng.shuffle(dirs)
    return dirs[:count]


def _make_horizontal_lane(
    width: int, row: int, direction: str, mode: str, rng: random.Random
) -> dict[str, object] | None:
    min_gates = 1 if mode == "edge" else 2
    if direction == "R":
        low, high = 3, width - 2
    else:
        low, high = 0, width - 5
    if high < low:
        return None
    max_fit = 1 + (high - low) // GATE_GAP
    max_gates = min(3, max_fit)
    if max_gates < min_gates:
        return None
    for _ in range(LANE_ATTEMPTS):
        gate_count = rng.randint(min_gates, max_gates)
        positions = _sample_spaced_positions(low, high, gate_count, GATE_GAP, rng)
        if positions is None:
            continue
        if direction == "R":
            start_low, start_high = 1, positions[0] - 2
            encounter = positions
        else:
            start_low, start_high = positions[-1] + 3, width - 2
            encounter = list(reversed(positions))
        if start_high < start_low:
            continue
        start = rng.randint(start_low, start_high)
        arrow_color, pairs = _lane_color_sequence(direction, gate_count, mode, rng)
        colors_by_pos = {pos: pair for pos, pair in zip(encounter, pairs)}
        gates = [
            {
                "left": pos,
                "left_color": int(colors_by_pos[pos][0]),
                "right_color": int(colors_by_pos[pos][1]),
            }
            for pos in positions
        ]
        return {
            "axis": "H",
            "direction": direction,
            "center": (row, start),
            "color": arrow_color,
            "gates": gates,
        }
    return None


def _make_vertical_lane(
    height: int, col: int, direction: str, mode: str, rng: random.Random
) -> dict[str, object] | None:
    min_gates = 1 if mode == "edge" else 2
    if direction == "D":
        low, high = 3, height - 2
    else:
        low, high = 0, height - 5
    if high < low:
        return None
    max_fit = 1 + (high - low) // GATE_GAP
    max_gates = min(3, max_fit)
    if max_gates < min_gates:
        return None
    for _ in range(LANE_ATTEMPTS):
        gate_count = rng.randint(min_gates, max_gates)
        positions = _sample_spaced_positions(low, high, gate_count, GATE_GAP, rng)
        if positions is None:
            continue
        if direction == "D":
            start_low, start_high = 1, positions[0] - 2
            encounter = positions
        else:
            start_low, start_high = positions[-1] + 3, height - 2
            encounter = list(reversed(positions))
        if start_high < start_low:
            continue
        start = rng.randint(start_low, start_high)
        arrow_color, pairs = _lane_color_sequence(direction, gate_count, mode, rng)
        colors_by_pos = {pos: pair for pos, pair in zip(encounter, pairs)}
        gates = [
            {
                "top": pos,
                "top_color": int(colors_by_pos[pos][0]),
                "bottom_color": int(colors_by_pos[pos][1]),
            }
            for pos in positions
        ]
        return {
            "axis": "V",
            "direction": direction,
            "center": (start, col),
            "color": arrow_color,
            "gates": gates,
        }
    return None


def _draw_horizontal_gate(grid: np.ndarray, row: int, left: int, left_color: int, right_color: int) -> None:
    grid[row - 1, left] = left_color
    grid[row - 1, left + 1] = right_color
    grid[row + 1, left] = left_color
    grid[row + 1, left + 1] = right_color


def _draw_vertical_gate(grid: np.ndarray, top: int, col: int, top_color: int, bottom_color: int) -> None:
    grid[top, col - 1] = top_color
    grid[top, col + 1] = top_color
    grid[top + 1, col - 1] = bottom_color
    grid[top + 1, col + 1] = bottom_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    axis = rng.choice(["H", "V"])

    if axis == "H":
        low, high = 2, height - 3
        max_lanes = 1 + (high - low) // LANE_GAP
    else:
        low, high = 2, width - 3
        max_lanes = 1 + (high - low) // LANE_GAP
    max_lanes = min(4, max_lanes)
    if max_lanes < 2:
        return None
    lane_count = rng.randint(2, max_lanes)
    centers = _sample_spaced_positions(low, high, lane_count, LANE_GAP, rng)
    if centers is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    modes = _lane_modes(lane_count, rng)
    directions = _lane_directions(axis, lane_count, rng)

    lanes: list[dict[str, object]] = []
    for center_value, direction, mode in zip(centers, directions, modes):
        if axis == "H":
            lane = _make_horizontal_lane(width, int(center_value), direction, mode, rng)
        else:
            lane = _make_vertical_lane(height, int(center_value), direction, mode, rng)
        if lane is None:
            return None
        lanes.append(lane)

    for lane in lanes:
        center = tuple(lane["center"])  # type: ignore[arg-type]
        for r, c in _arrow_cells(center, str(lane["axis"])):
            if not _inside(grid.shape, r, c) or int(grid[r, c]) != BG:
                return None
        for r, c in _arrow_cells(center, str(lane["axis"])):
            grid[r, c] = int(lane["color"])
        if str(lane["axis"]) == "H":
            row = int(center[0])
            for gate in lane["gates"]:  # type: ignore[assignment]
                left = int(gate["left"])
                _draw_horizontal_gate(grid, row, left, int(gate["left_color"]), int(gate["right_color"]))
        else:
            col = int(center[1])
            for gate in lane["gates"]:  # type: ignore[assignment]
                top = int(gate["top"])
                _draw_vertical_gate(grid, top, col, int(gate["top_color"]), int(gate["bottom_color"]))

    if _background(grid) != BG:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow-gate inputs")
