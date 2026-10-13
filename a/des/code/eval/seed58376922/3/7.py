from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
WIRE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _wire_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BLACK, YELLOW):
            out.append((nr, nc))
    return out


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    sources: list[dict[str, object]] = []
    for comp in _components(g == YELLOW):
        if len(comp) != 4:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0, c1 - c0) != (1, 1):
            continue
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == YELLOW):
            continue
        adj: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    val = int(g[nr, nc])
                    if val not in (BLACK, YELLOW):
                        adj.add((nr, nc))
        assert len(adj) == 1, "each yellow source must touch exactly one wire endpoint"
        endpoint = next(iter(adj))
        er, ec = endpoint
        nxt = _wire_neighbors(g, er, ec)
        assert len(nxt) == 1, "wire endpoint must have a unique forward neighbor"
        nr, nc = nxt[0]
        direction = (nr - er, nc - ec)
        sources.append(
            {
                "bbox": (r0, c0, r1, c1),
                "endpoint": endpoint,
                "color": int(g[er, ec]),
                "direction": direction,
            }
        )
    assert sources, "expected at least one yellow source"
    return sources


def _forward_dirs(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    dirs: dict[int, tuple[int, int]] = {}
    for src in _find_sources(grid):
        color = int(src["color"])
        direction = tuple(src["direction"])
        assert color not in dirs, "wire colors must be unique per source"
        dirs[color] = (int(direction[0]), int(direction[1]))
    return dirs


def _trace_pulse(
    grid: np.ndarray,
    start: tuple[int, int],
    forward_dirs: dict[int, tuple[int, int]],
) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pos = (int(start[0]), int(start[1]))
    seen: set[tuple[int, int]] = set()
    steps = 0
    while True:
        assert pos not in seen, "pulse path must not loop"
        seen.add(pos)
        steps += 1
        color = int(g[pos])
        assert color in forward_dirs, "every wire color must have a source-defined direction"
        dr, dc = forward_dirs[color]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) in (BLACK, YELLOW):
            return steps, pos, (dr, dc), color
        pos = (nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    forward_dirs = _forward_dirs(g)
    for src in _find_sources(g):
        steps, end_pos, direction, color = _trace_pulse(g, tuple(src["endpoint"]), forward_dirs)
        dr, dc = direction
        for k in range(1, steps + 1):
            nr = end_pos[0] + dr * k
            nc = end_pos[1] + dc * k
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color
    return out


def _apply_symmetry(grid: np.ndarray, rot: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, k=rot)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _build_canonical(rng: random.Random) -> np.ndarray:
    nh = rng.randint(1, 3)
    nv = rng.randint(1, 3)
    row_slots = [4, 7, 10]
    col_slots = [4, 7, 10]
    rows = sorted(rng.sample(row_slots, nh))
    cols = sorted(rng.sample(col_slots, nv))
    palette = rng.sample(list(WIRE_COLORS), nh + nv)

    max_row = max(rows)
    max_col = max(cols)
    horiz: list[dict[str, int]] = []
    vert: list[dict[str, int]] = []

    for idx, row in enumerate(rows):
        start = rng.randint(2, 3)
        end = rng.randint(max_col + 1, max_col + 2)
        horiz.append({"row": row, "start": start, "end": end, "color": palette[idx]})
    for idx, col in enumerate(cols):
        start = rng.randint(2, 3)
        end = rng.randint(max_row + 1, max_row + 2)
        vert.append({"col": col, "start": start, "end": end, "color": palette[nh + idx]})

    active_h = max([max_row] + [wire["end"] for wire in vert]) + 1
    active_w = max([max_col] + [wire["end"] for wire in horiz]) + 1
    grid = np.full((active_h, active_w), BLACK, dtype=int)

    for wire in horiz:
        grid[wire["row"], wire["start"] : wire["end"] + 1] = wire["color"]
    for wire in vert:
        grid[wire["start"] : wire["end"] + 1, wire["col"]] = wire["color"]

    intersections: list[tuple[int, int, int, int]] = []
    picks: list[bool] = []
    for hi, hwire in enumerate(horiz):
        for vi, vwire in enumerate(vert):
            intersections.append((hwire["row"], vwire["col"], hi, vi))
            picks.append(rng.random() < 0.5)
    if len(picks) > 1 and (all(picks) or not any(picks)):
        picks[rng.randrange(len(picks))] = not picks[0]
    for (r, c, hi, vi), use_horizontal in zip(intersections, picks):
        grid[r, c] = horiz[hi]["color"] if use_horizontal else vert[vi]["color"]

    for wire in horiz:
        r = wire["row"]
        c = wire["start"]
        assert np.all(grid[r - 1 : r + 1, c - 2 : c] == BLACK)
        grid[r - 1 : r + 1, c - 2 : c] = YELLOW
    for wire in vert:
        r = wire["start"]
        c = wire["col"]
        assert np.all(grid[r - 2 : r, c : c + 2] == BLACK)
        grid[r - 2 : r, c : c + 2] = YELLOW

    return grid


def _trace_all(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int], int]]:
    forward_dirs = _forward_dirs(grid)
    traces = []
    for src in _find_sources(grid):
        traces.append(_trace_pulse(grid, tuple(src["endpoint"]), forward_dirs))
    return traces


def _pad_for_tails(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    traces = _trace_all(grid)
    active_h, active_w = grid.shape
    need_bottom = 0
    need_right = 0
    for steps, end_pos, direction, _color in traces:
        if direction == (1, 0):
            need_bottom = max(need_bottom, end_pos[0] + steps + 1 - active_h)
        elif direction == (0, 1):
            need_right = max(need_right, end_pos[1] + steps + 1 - active_w)
        else:
            raise AssertionError("canonical generator should only move east or south")

    pad_top = rng.randint(0, 2)
    pad_left = rng.randint(0, 2)
    pad_bottom = need_bottom + rng.randint(0, 2)
    pad_right = need_right + rng.randint(0, 2)

    out = np.full(
        (active_h + pad_top + pad_bottom, active_w + pad_left + pad_right),
        BLACK,
        dtype=int,
    )
    out[pad_top : pad_top + active_h, pad_left : pad_left + active_w] = grid
    return out


def _intersection_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    count = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) in (BLACK, YELLOW):
                continue
            deg = 0
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) not in (BLACK, YELLOW):
                    deg += 1
            if deg >= 4:
                count += 1
    return count


def _tail_cells(
    traces: Iterable[tuple[int, tuple[int, int], tuple[int, int], int]]
) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for steps, end_pos, direction, color in traces:
        dr, dc = direction
        for k in range(1, steps + 1):
            cells.append((end_pos[0] + dr * k, end_pos[1] + dc * k, color))
    return cells


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        sources = _find_sources(g)
        forward_dirs = _forward_dirs(g)
        if len(sources) != len(forward_dirs):
            return False
        if _intersection_count(g) == 0:
            return False
        traces = _trace_all(g)
        if len(traces) != len(sources):
            return False
        for r, c, _color in _tail_cells(traces):
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                return False
            if int(g[r, c]) != BLACK:
                return False
        out = main(g)
        diff = out != g
        if not bool(diff.any()):
            return False
        if not bool(np.all(g[diff] == BLACK)):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canonical = _build_canonical(rng)
    padded = _pad_for_tails(canonical, rng)
    transformed = _apply_symmetry(padded, rng.randrange(4), bool(rng.randrange(2)))
    if _validate_generated(transformed):
        return transformed
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
