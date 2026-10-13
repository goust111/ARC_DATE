from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perp_dir(travel: tuple[int, int]) -> tuple[int, int]:
    return (1, 0) if travel[0] == 0 else (0, 1)


def _axis_cell_kind(
    grid: np.ndarray,
    r: int,
    c: int,
    orientation: str,
    base_color: int,
    bg: int,
) -> int | None:
    color = int(grid[r, c])
    if color == base_color:
        return color
    if color == bg:
        return None
    if orientation == "h":
        if not _in_bounds(grid.shape, r - 1, c) or not _in_bounds(grid.shape, r + 1, c):
            return None
        if int(grid[r - 1, c]) == color and int(grid[r + 1, c]) == color:
            return color
    else:
        if not _in_bounds(grid.shape, r, c - 1) or not _in_bounds(grid.shape, r, c + 1):
            return None
        if int(grid[r, c - 1]) == color and int(grid[r, c + 1]) == color:
            return color
    return None


def _build_track(
    grid: np.ndarray,
    top: int,
    left: int,
    color: int,
    side: str,
    offset: int,
    bg: int,
) -> dict | None:
    if side == "left":
        orientation = "h"
        attach_row = top + offset
        near = (attach_row, left - 1)
        travel = (0, 1)
        square_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
        stack_start = (attach_row, left + 2)
    elif side == "right":
        orientation = "h"
        attach_row = top + offset
        near = (attach_row, left + 2)
        travel = (0, -1)
        square_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
        stack_start = (attach_row, left - 1)
    elif side == "up":
        orientation = "v"
        attach_col = left + offset
        near = (top - 1, attach_col)
        travel = (1, 0)
        square_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
        stack_start = (top + 2, attach_col)
    else:
        orientation = "v"
        attach_col = left + offset
        near = (top + 2, attach_col)
        travel = (-1, 0)
        square_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
        stack_start = (top - 1, attach_col)

    away = (-travel[0], -travel[1])
    line_near_to_far: list[tuple[int, int, int | None]] = []
    gate_cells: set[tuple[int, int]] = set()
    r, c = near
    while _in_bounds(grid.shape, r, c):
        kind = _axis_cell_kind(grid, r, c, orientation, color, bg)
        if kind is None:
            break
        gate_color = None if kind == color else kind
        if gate_color is not None:
            if orientation == "h":
                arm_a = (r - 1, c)
                arm_b = (r + 1, c)
            else:
                arm_a = (r, c - 1)
                arm_b = (r, c + 1)
            if not (_in_bounds(grid.shape, *arm_a) and _in_bounds(grid.shape, *arm_b)):
                return None
            if int(grid[arm_a]) != gate_color or int(grid[arm_b]) != gate_color:
                return None
            gate_cells.update((arm_a, arm_b))
        line_near_to_far.append((r, c, gate_color))
        r += away[0]
        c += away[1]

    if not line_near_to_far:
        return None

    line = list(reversed(line_near_to_far))
    line_positions = [(r0, c0) for r0, c0, _ in line]
    gate_colors = [gc for _r0, _c0, gc in line]
    final_color_from_index = [color] * len(line)
    current = color
    for idx in range(len(line) - 1, -1, -1):
        if gate_colors[idx] is not None:
            current = int(gate_colors[idx])
        final_color_from_index[idx] = int(current)

    return {
        "color": int(color),
        "square_top_left": (int(top), int(left)),
        "square_cells": square_cells,
        "orientation": orientation,
        "travel": travel,
        "line_positions": line_positions,
        "line_index": {pos: i for i, pos in enumerate(line_positions)},
        "gate_colors": gate_colors,
        "gate_cells": gate_cells,
        "final_color_from_index": final_color_from_index,
        "stack_start": stack_start,
    }


def _find_tracks(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    tracks: list[dict] = []
    seen_squares: set[tuple[int, int, int]] = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        for r in range(h - 1):
            for c in range(w - 1):
                if not bool(mask[r : r + 2, c : c + 2].all()):
                    continue
                key = (color, r, c)
                if key in seen_squares:
                    continue

                sides: list[tuple[str, int]] = []
                invalid = False

                if c - 1 >= 0:
                    hits = [dr for dr in range(2) if bool(mask[r + dr, c - 1])]
                    if len(hits) == 1:
                        sides.append(("left", hits[0]))
                    elif len(hits) > 1:
                        invalid = True
                if c + 2 < w:
                    hits = [dr for dr in range(2) if bool(mask[r + dr, c + 2])]
                    if len(hits) == 1:
                        sides.append(("right", hits[0]))
                    elif len(hits) > 1:
                        invalid = True
                if r - 1 >= 0:
                    hits = [dc for dc in range(2) if bool(mask[r - 1, c + dc])]
                    if len(hits) == 1:
                        sides.append(("up", hits[0]))
                    elif len(hits) > 1:
                        invalid = True
                if r + 2 < h:
                    hits = [dc for dc in range(2) if bool(mask[r + 2, c + dc])]
                    if len(hits) == 1:
                        sides.append(("down", hits[0]))
                    elif len(hits) > 1:
                        invalid = True

                if invalid or len(sides) != 1:
                    continue

                track = _build_track(grid, r, c, color, sides[0][0], sides[0][1], bg)
                if track is None:
                    continue
                seen_squares.add(key)
                tracks.append(track)
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide scattered colored pixels onto same-color tracks, recolor them at crossed gates,
    and stack them beyond the matching 2x2 endpoint square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    tracks = _find_tracks(g, bg)
    assert tracks, "expected at least one valid track"

    structure_cells: set[tuple[int, int]] = set()
    by_color: dict[int, list[dict]] = defaultdict(list)
    for track in tracks:
        by_color[int(track["color"])].append(track)
        structure_cells |= set(track["square_cells"])
        structure_cells |= set(track["line_positions"])
        structure_cells |= set(track["gate_cells"])

    scattered: list[tuple[int, int]] = []
    for r, c in np.argwhere(g != bg):
        pos = (int(r), int(c))
        if pos not in structure_cells:
            scattered.append(pos)

    assignments: dict[int, list[dict]] = defaultdict(list)
    for r, c in scattered:
        color = int(g[r, c])
        assert color in by_color, "scatter must match at least one track color"
        best: tuple[int, int, int, int] | None = None
        best_track: dict | None = None
        best_idx = -1
        for track in by_color[color]:
            for idx, (tr, tc) in enumerate(track["line_positions"]):
                if tr != r and tc != c:
                    continue
                off = abs(tr - r) + abs(tc - c)
                line_steps = len(track["line_positions"]) - 1 - idx
                cand = (off, line_steps, abs(tr - r), abs(tc - c))
                if best is None or cand < best:
                    best = cand
                    best_track = track
                    best_idx = idx
        assert best is not None and best_track is not None, "unmatched scatter"
        off = best[0]
        line_steps = len(best_track["line_positions"]) - 1 - best_idx
        assignments[id(best_track)].append(
            {
                "src": (r, c),
                "off_steps": int(off),
                "line_steps": int(line_steps),
                "total_steps": int(off + line_steps),
                "final_color": int(best_track["final_color_from_index"][best_idx]),
            }
        )

    out = g.copy()
    for r, c in scattered:
        out[r, c] = bg

    track_lookup = {id(track): track for track in tracks}
    for key, items in assignments.items():
        track = track_lookup[key]
        items.sort(
            key=lambda item: (
                -item["total_steps"],
                -item["line_steps"],
                -item["off_steps"],
                item["src"][0],
                item["src"][1],
            )
        )
        sr, sc = track["stack_start"]
        dr, dc = track["travel"]
        for idx, item in enumerate(items):
            rr = sr + dr * idx
            cc = sc + dc * idx
            assert _in_bounds(out.shape, rr, cc)
            assert int(out[rr, cc]) == bg, "stack cell must be free"
            out[rr, cc] = int(item["final_color"])

    return out


def _sample_gate_positions(line_len: int, gate_count: int) -> list[int]:
    if gate_count == 0:
        return []
    candidates = list(range(1, line_len - 1))
    for _ in range(80):
        random.shuffle(candidates)
        picks = sorted(candidates[:gate_count])
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    raise RuntimeError("failed to sample gate positions")


def _sample_scatters(
    line_len: int,
    gate_positions: list[int],
    scatter_count: int,
) -> list[tuple[int, int]]:
    gate_set = set(gate_positions)
    axis_gaps = list(range(2, 8))
    perp_us = [u for u in range(line_len) if u not in gate_set]

    for _ in range(200):
        coords: list[tuple[int, int]] = []
        totals: set[int] = set()
        ok = True
        for _idx in range(scatter_count):
            for _inner in range(100):
                mode = random.choice(["axis", "perp"])
                if mode == "axis":
                    gap = random.choice(axis_gaps)
                    cand = (-gap, 0)
                    entry = 0
                    off = gap
                else:
                    if not perp_us:
                        continue
                    u = random.choice(perp_us)
                    side = random.choice([-1, 1])
                    dist = random.choice([2, 3, 4])
                    cand = (u, side * dist)
                    entry = u
                    off = dist
                if cand in coords:
                    continue
                if any(abs(cand[0] - u0) + abs(cand[1] - v0) == 1 for u0, v0 in coords):
                    continue
                total = off + (line_len - 1 - entry)
                if total in totals:
                    continue
                coords.append(cand)
                totals.add(total)
                break
            else:
                ok = False
                break
        if not ok:
            continue
        if gate_positions and not any(u <= max(gate_positions) for u, _v in coords if u >= 0):
            continue
        return coords
    raise RuntimeError("failed to sample scatters")


def _make_local_spec(track_color: int, gate_palette: list[int]) -> dict:
    travel = random.choice(DIRS)
    perp = _perp_dir(travel)
    line_len = random.randint(4, 10)
    max_gate = min(3, max(0, (line_len - 1) // 2))
    gate_count = random.randint(0, max_gate)
    gate_positions = _sample_gate_positions(line_len, gate_count)
    gate_map = {u: random.choice(gate_palette) for u in gate_positions}
    scatter_count = random.randint(1, 4)
    attach_variant = random.randint(0, 1)
    scatters = _sample_scatters(line_len, gate_positions, scatter_count)

    paint_local: dict[tuple[int, int], int] = {}
    for u in range(line_len):
        paint_local[(u, 0)] = int(gate_map.get(u, track_color))

    square_vs = (0, 1) if attach_variant == 0 else (-1, 0)
    for du in (line_len, line_len + 1):
        for dv in square_vs:
            paint_local[(du, dv)] = int(track_color)

    for u, gate_color in gate_map.items():
        paint_local[(u, -1)] = int(gate_color)
        paint_local[(u, 1)] = int(gate_color)

    for coord in scatters:
        paint_local[coord] = int(track_color)

    output_slots = [(line_len + 2 + i, 0) for i in range(scatter_count)]
    used_local = set(paint_local) | set(output_slots)

    return {
        "track_color": int(track_color),
        "travel": travel,
        "perp": perp,
        "paint_local": paint_local,
        "used_local": used_local,
    }


def _to_actual(
    origin: tuple[int, int],
    travel: tuple[int, int],
    perp: tuple[int, int],
    local: tuple[int, int],
) -> tuple[int, int]:
    u, v = local
    return (
        origin[0] + u * travel[0] + v * perp[0],
        origin[1] + u * travel[1] + v * perp[1],
    )


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _place_track(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    spec: dict,
) -> dict | None:
    travel = spec["travel"]
    perp = spec["perp"]
    deltas = [_to_actual((0, 0), travel, perp, local) for local in spec["used_local"]]
    min_r = min(r for r, _c in deltas)
    max_r = max(r for r, _c in deltas)
    min_c = min(c for _r, c in deltas)
    max_c = max(c for _r, c in deltas)

    row_lo = -min_r
    row_hi = shape[0] - 1 - max_r
    col_lo = -min_c
    col_hi = shape[1] - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(120):
        origin = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        used_actual = {
            _to_actual(origin, travel, perp, local)
            for local in spec["used_local"]
        }
        if any(cell in blocked for cell in used_actual):
            continue
        paint_actual = {
            _to_actual(origin, travel, perp, local): color
            for local, color in spec["paint_local"].items()
        }
        return {
            "paint_actual": paint_actual,
            "used_actual": used_actual,
            "halo_actual": _halo(used_actual, shape),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    track_count = random.choice([1, 2, 3])
    track_colors = random.sample(list(range(1, 10)), track_count)
    gate_palette = [c for c in range(1, 10) if c not in track_colors]

    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()

    for track_color in track_colors:
        placed = None
        for _ in range(80):
            spec = _make_local_spec(track_color, gate_palette)
            placed = _place_track(grid.shape, blocked, spec)
            if placed is not None:
                for (r, c), color in placed["paint_actual"].items():
                    grid[r, c] = int(color)
                blocked |= placed["halo_actual"]
                break
        if placed is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnetic-track puzzle input")
