from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WALL = 5
GATE = 6
MAROON = 9
TOKEN_COLORS = (BLUE, RED, GREEN, YELLOW)

GLYPH_PRIORITY = {
    ((1, 0), (0, 0)): 1,
    ((1, 1), (0, 0)): 2,
    ((1, 0), (1, 1)): 3,
}


def _neighbors_for_fall(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r + 1, c), (r, c - 1), (r, c + 1))


def _find_gate_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    gates: list[dict] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GATE or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GATE and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            rows = sorted({rr for rr, _ in cells})
            cols = sorted(cc for _, cc in cells)
            assert len(cells) == 3 and len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3))
            gates.append({"row": rows[0], "c0": cols[0], "c1": cols[-1]})
    return gates


def _extract_glyph(grid: np.ndarray, gate: dict) -> np.ndarray:
    r = int(gate["row"])
    c1 = int(gate["c1"])
    h, w = grid.shape
    assert r >= 1, "glyph needs one row above the gate row"
    for c in range(c1 + 1, min(w - 1, c1 + 8)):
        block = grid[r - 1 : r + 1, c : c + 2]
        if block.shape != (2, 2):
            continue
        vals = set(map(int, block.flatten()))
        if vals <= {BLACK, MAROON} and np.any(block == MAROON):
            return block
    raise ValueError("could not find a glyph to the right of a gate")


def _priority_from_glyph(block: np.ndarray) -> int:
    mask = tuple(tuple(1 if int(v) == MAROON else 0 for v in row) for row in block)
    if mask not in GLYPH_PRIORITY:
        raise ValueError(f"unknown glyph pattern: {mask}")
    return GLYPH_PRIORITY[mask]


def _initial_tokens(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    tokens: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in TOKEN_COLORS:
                side = "L" if c < w // 2 else "R"
                tokens.append({"color": color, "pos": (r, c), "side": side})
    assert 2 <= len(tokens) <= 4, "expected 2-4 distinct tokens"
    assert len({t["color"] for t in tokens}) == len(tokens), "token colors should be distinct"
    return tokens


def _landing(base: np.ndarray, start: tuple[int, int], occupied: set[tuple[int, int]], side: str) -> tuple[int, int]:
    h, w = base.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors_for_fall(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or (nr, nc) in occupied:
                continue
            if int(base[nr, nc]) != BLACK:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    best_row = max(r for r, _ in seen)
    best_cells = [(r, c) for r, c in seen if r == best_row]
    if side == "L":
        best_col = min(c for _, c in best_cells)
    else:
        best_col = max(c for _, c in best_cells)
    return best_row, best_col


def _simulate(base: np.ndarray, tokens: list[dict], gate_order: list[dict]) -> np.ndarray:
    work = np.array(base, dtype=int, copy=True)
    state = [{"color": int(t["color"]), "pos": tuple(t["pos"]), "side": t["side"]} for t in tokens]

    for gate in gate_order:
        r = int(gate["row"])
        c0 = int(gate["c0"])
        c1 = int(gate["c1"])
        work[r, c0 : c1 + 1] = BLACK

        indices = sorted(range(len(state)), key=lambda i: (state[i]["pos"][0], state[i]["pos"][1]), reverse=True)
        for idx in indices:
            occupied = {tuple(t["pos"]) for j, t in enumerate(state) if j != idx}
            cur = tuple(state[idx]["pos"])
            landing = _landing(work, cur, occupied, state[idx]["side"])
            if landing[0] > cur[0]:
                state[idx]["pos"] = landing

    out = work.copy()
    for token in state:
        r, c = token["pos"]
        out[r, c] = int(token["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode gate glyphs into an opening order, then drop the colored tokens through the pipe network one gate at a time.

    Input structure:
    - The background is black.
    - A solid grey wall mass contains black pipe corridors, 2-3 magenta horizontal 1x3 gates, and one maroon/black 2x2 glyph to the right of each gate.
    - There are 2-4 distinct single-pixel tokens in blue/red/green/yellow above all gates, entering from the left or right side.

    Rule:
    - Detect every horizontal magenta 1x3 gate and read the 2x2 glyph on its right to decode that gate's priority.
    - Process gates in priority order from 1 upward; when a gate is processed, its three magenta cells become black so the passage opens.
    - Immediately after each opening, any token that can now descend moves through the currently open black corridors, settling at the lowest reachable cell; if several lowest cells are tied, left-entry tokens choose the leftmost one and right-entry tokens choose the rightmost one.
    - Settled tokens remain as obstacles for later drops, while walls and glyphs stay unchanged.

    Key insight:
    - The glyphs do not label colors; they encode the temporal order in which gates open, and that order changes which tokens claim the lower cells first.
    """
    g = np.array(grid, dtype=int, copy=True)
    assert g.ndim == 2

    gates = _find_gate_components(g)
    assert 2 <= len(gates) <= 3
    for gate in gates:
        gate["priority"] = _priority_from_glyph(_extract_glyph(g, gate))

    tokens = _initial_tokens(g)
    base = g.copy()
    for r, c in [t["pos"] for t in tokens]:
        base[r, c] = BLACK

    ordered = sorted(gates, key=lambda rec: int(rec["priority"]))
    return _simulate(base, tokens, ordered)


def _carve_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> None:
    grid[r0 : r1 + 1, col] = BLACK


def _carve_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    grid[row, c0 : c1 + 1] = BLACK


def _place_glyph(grid: np.ndarray, gate_row: int, gate_center: int, priority: int) -> None:
    mask = np.array(list(GLYPH_PRIORITY.keys())[priority - 1], dtype=int)
    block = np.where(mask == 1, MAROON, BLACK)
    c0 = gate_center + 3
    grid[gate_row - 1 : gate_row + 1, c0 : c0 + 2] = block


def _build_candidate() -> np.ndarray:
    h = random.randint(16, 22)
    w = random.randint(16, 20)
    grid = np.zeros((h, w), dtype=int)
    grid[:, 1 : w - 1] = WALL

    xc = w // 2
    xl = 3 + random.randint(0, 1)
    xr = w - 6 - random.randint(0, 1)
    if not (xl + 3 < xc < xr - 2):
        return None

    join_r = random.randint(h - 8, h - 6)
    left_gate_r = random.randint(4, join_r - 3)
    right_gate_r = random.randint(4, join_r - 3)

    include_center = random.random() < 0.6
    center_gate_r = None
    if include_center:
        center_gate_r = random.randint(join_r + 2, h - 3)

    total_tokens = random.randint(2, 4)
    left_count = random.randint(1, total_tokens - 1)
    right_count = total_tokens - left_count
    if max(left_count, right_count) > 2:
        return None

    _carve_vertical(grid, xl, 1, join_r)
    _carve_vertical(grid, xr, 1, join_r)
    _carve_vertical(grid, xc, join_r, h - 2)
    _carve_horizontal(grid, join_r, xl, xc)
    _carve_horizontal(grid, join_r, xc, xr)

    grid[1 : left_count + 1, xl - 1] = BLACK
    grid[1 : left_count + 1, xl] = BLACK
    grid[1 : right_count + 1, xr] = BLACK
    grid[1 : right_count + 1, xr + 1] = BLACK

    grid[left_gate_r, xl - 1 : xl + 2] = GATE
    grid[right_gate_r, xr - 1 : xr + 2] = GATE
    gate_names = ["L", "R"]
    if include_center:
        grid[center_gate_r, xc - 1 : xc + 2] = GATE
        gate_names.append("C")

    random.shuffle(gate_names)
    priorities = {name: idx + 1 for idx, name in enumerate(gate_names)}
    _place_glyph(grid, left_gate_r, xl, priorities["L"])
    _place_glyph(grid, right_gate_r, xr, priorities["R"])
    if include_center and center_gate_r is not None:
        _place_glyph(grid, center_gate_r, xc, priorities["C"])

    colors = random.sample(list(TOKEN_COLORS), total_tokens)
    random.shuffle(colors)
    for i in range(left_count):
        grid[i + 1, xl - 1] = colors[i]
    for j in range(right_count):
        grid[j + 1, xr + 1] = colors[left_count + j]

    return grid


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _interesting(grid: np.ndarray) -> bool:
    if grid is None:
        return False
    out = main(grid)
    if _same_shape(out, grid) and np.array_equal(out, grid):
        return False

    gates = _find_gate_components(grid)
    if len(gates) < 2:
        return False
    tokens = _initial_tokens(grid)
    base = grid.copy()
    for r, c in [t["pos"] for t in tokens]:
        base[r, c] = BLACK
    for gate in gates:
        gate["priority"] = _priority_from_glyph(_extract_glyph(grid, gate))

    ordered = sorted(gates, key=lambda rec: int(rec["priority"]))
    reversed_order = list(reversed(ordered))
    alt = _simulate(base, tokens, reversed_order)
    if np.array_equal(out, alt):
        return False

    start_rows = sorted(r for r, _ in [t["pos"] for t in tokens])
    end_rows = sorted(int(r) for r, _ in np.argwhere(np.isin(out, TOKEN_COLORS)))
    if max(end_rows) - min(start_rows) < 4:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_candidate()
        if _interesting(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
