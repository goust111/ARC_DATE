from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

WALL = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
BG = 8

TRACE = BLUE
SEP = RED
TOKEN_FAR = GREEN
TOKEN_NEAR = BLUE

DEPTH_PALETTE = [GREEN, BLUE, RED, YELLOW, GREY, PINK, ORANGE]


def _build_opened_grid(
    gates: list[list[tuple[int, int]]],
    holes: list[dict[int, int]],
    width: int,
) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c, color in holes[pipe_idx].items():
            grid[r, c] = color
    for wall_idx, row_gates in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center, post_color in row_gates:
            grid[wr - 1, center] = post_color
            grid[wr, center] = post_color
            grid[wr + 1, center] = post_color
            grid[wr, center - 1] = BG
            grid[wr, center + 1] = BG
    return grid


def _segments_in_row(grid: np.ndarray, r: int, source: dict[str, Any], variant: str) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if not _passable(grid, r, c, source, variant):
            c += 1
            continue
        start = c
        while c + 1 < width and _passable(grid, r, c + 1, source, variant):
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _passable(grid: np.ndarray, r: int, c: int, source: dict[str, Any], variant: str) -> bool:
    value = int(grid[r, c])
    if value == BG:
        return True
    if variant == "D8_2" and value == SEP and source["side"] == "left":
        return True
    return False


def _find_segment(grid: np.ndarray, r: int, c: int, source: dict[str, Any], variant: str) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, source, variant):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(
    grid: np.ndarray,
    r: int,
    seg: tuple[int, int],
    source: dict[str, Any],
    variant: str,
) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    out = []
    for c in range(a, b + 1):
        if _passable(grid, r + 1, c, source, variant) and _passable(grid, r + 2, c, source, variant):
            out.append(c)
    return out


def _entries_above(
    grid: np.ndarray,
    r: int,
    seg: tuple[int, int],
    source: dict[str, Any],
    variant: str,
) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    out = []
    for c in range(a, b + 1):
        if _passable(grid, r - 1, c, source, variant) and _passable(grid, r - 2, c, source, variant):
            out.append(c)
    return out


def _repair_holes(
    gates: list[list[int]],
    holes: list[dict[int, int]],
    width: int,
) -> list[dict[int, int]] | None:
    hole_sets = [dict(row) for row in holes]
    while True:
        colored_gates = [[(c, PINK) for c in row] for row in gates]
        grid = _build_opened_grid(colored_gates, hole_sets, width)
        issue = _find_issue(grid)
        if issue is None:
            return hole_sets
        r, above, below = issue
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None
        cols = sorted(above + below)
        if len(cols) < 2:
            return None
        inserted = False
        for left, right in zip(cols, cols[1:]):
            mid = (left + right) // 2
            if mid in hole_sets[pipe_idx]:
                continue
            if 0 < mid < width - 1 and grid[r, mid] == BG:
                hole_sets[pipe_idx][mid] = WALL
                inserted = True
                break
        if not inserted:
            return None


def _find_issue(grid: np.ndarray) -> tuple[int, list[int], list[int]] | None:
    for r in range(0, grid.shape[0], 2):
        src = {"side": "left"}
        for seg in _segments_in_row(grid, r, src, "BASE"):
            above = _entries_above(grid, r, seg, src, "BASE")
            below = _exits_below(grid, r, seg, src, "BASE")
            if len(above) > 1 or len(below) > 1:
                return r, above, below
    return None


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    out = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue
        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            out.append(c)
    return out


def _build_standard_structure(rng: np.random.Generator) -> tuple[list[list[int]], list[dict[int, int]], int]:
    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[dict[int, int]] = [dict() for _ in range(n_walls + 1)]
    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))
    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 2)]
        if not options:
            raise RuntimeError("failed to seed spine")
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left
    repaired = _repair_holes(gates, holes, width)
    if repaired is None:
        raise RuntimeError("failed to repair holes")
    holes = repaired
    total = sum(len(row) for row in gates)
    target = int(rng.integers(max(6, total), 13))
    while total < target:
        choices: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                choices.append((row, center))
        if not choices:
            break
        row, center = choices[int(rng.integers(len(choices)))]
        new_gates = [list(cols) for cols in gates]
        new_gates[row].append(center)
        new_gates[row].sort()
        new_holes = _repair_holes(new_gates, holes, width)
        if new_holes is None:
            continue
        gates = new_gates
        holes = new_holes
        total += 1
    return gates, holes, width


def _build_multi_exit_structure(rng: np.random.Generator, two_colors: bool = False) -> tuple[list[list[tuple[int, int]]], list[dict[int, int]], int]:
    n_walls = int(rng.integers(5, 8))
    width = int(rng.integers(16, 19))
    centers = [int(rng.integers(4, width - 4))]
    for _ in range(1, n_walls):
        shift = int(rng.choice([-2, -1, 1, 2]))
        centers.append(int(np.clip(centers[-1] + shift, 3, width - 4)))
    gates: list[list[tuple[int, int]]] = []
    for wall_idx, center in enumerate(centers):
        row = [(center, PINK)]
        if wall_idx == n_walls // 2:
            other = min(width - 3, center + 5)
            if other - center >= 5:
                row.append((other, ORANGE if two_colors else PINK))
        gates.append(sorted(row))
    holes = [dict() for _ in range(n_walls + 1)]
    return gates, holes, width


def _assign_gate_colors(gates: list[list[int]], variant: str) -> list[list[tuple[int, int]]]:
    colored: list[list[tuple[int, int]]] = []
    for wall_idx, row in enumerate(gates):
        out_row = []
        for center in row:
            if variant == "D2_1":
                out_row.append((center, PINK if wall_idx % 2 == 0 else ORANGE))
            elif variant == "D7_1":
                out_row.append((center, PINK if center % 2 == 0 else ORANGE))
            else:
                out_row.append((center, PINK))
        colored.append(out_row)
    return colored


def _place_sources(
    grid: np.ndarray,
    gates: list[list[tuple[int, int]]],
    variant: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    height, width = grid.shape
    n_walls = (height - 1) // 2
    sources: list[dict[str, Any]] = []
    order = 0
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        if variant == "D5_0":
            cols = [0] if pipe_idx % 2 == 0 else [width - 1]
            colors = [GREEN]
        elif variant == "D5_1":
            cols = [int(rng.integers(1, width - 1))]
            colors = [GREEN]
        elif variant == "D5_2":
            count = len(gates[pipe_idx]) if pipe_idx < len(gates) else 0
            cols = list(range(count))
            colors = [GREEN] * len(cols)
        elif variant == "D5_3":
            cols = [0, width - 1]
            colors = [GREEN, BLUE]
        elif variant == "D13_0":
            if pipe_idx % 2 == 1:
                continue
            cols = [0, width - 1]
            colors = [GREEN, GREEN]
        else:
            cols = [0, width - 1]
            colors = [GREEN] * len(cols)
        for c, color in zip(cols, colors):
            c = int(np.clip(c, 0, width - 1))
            side = "left" if c <= 1 else "right" if c >= width - 2 else "internal"
            grid[r, c] = color
            sources.append(
                {
                    "r": r,
                    "c": c,
                    "color": color,
                    "side": side,
                    "pipe_idx": pipe_idx,
                    "order": order,
                }
            )
            order += 1
    return sources


def _parse_sources(grid: np.ndarray, variant: str) -> list[dict[str, Any]]:
    sources = []
    token_colors = {GREEN}
    if variant == "D5_3":
        token_colors = {GREEN, BLUE}
    order = 0
    for pipe_idx, r in enumerate(range(0, grid.shape[0], 2)):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color not in token_colors:
                continue
            side = "left" if c <= 1 else "right" if c >= grid.shape[1] - 2 else "internal"
            sources.append(
                {
                    "r": r,
                    "c": c,
                    "color": color,
                    "side": side,
                    "pipe_idx": pipe_idx,
                    "order": order,
                }
            )
            order += 1
    return sources


def _open_from_bar_input(grid: np.ndarray, variant: str) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    g = np.array(grid, dtype=int)
    opened = g.copy()
    n_walls = (g.shape[0] - 1) // 2
    gates: list[list[tuple[int, int]]] = [[] for _ in range(n_walls)]
    token_colors = {GREEN}
    if variant == "D5_3":
        token_colors = {GREEN, BLUE}
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        row = g[wr]
        for c in range(1, g.shape[1] - 1):
            col = int(row[c])
            if col in (WALL, BG, SEP) or col in token_colors:
                continue
            if int(row[c - 1]) == col and int(row[c + 1]) == col:
                gates[wall_idx].append((c, col))
    for wall_idx, row_gates in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center, post_color in row_gates:
            opened[wr, center - 1 : center + 2] = BG
            opened[wr - 1, center] = post_color
            opened[wr, center] = post_color
            opened[wr + 1, center] = post_color
    for src in _parse_sources(g, variant):
        opened[src["r"], src["c"]] = BG
    return opened, gates


def _parse_open_input(grid: np.ndarray, variant: str) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    g = np.array(grid, dtype=int)
    n_walls = (g.shape[0] - 1) // 2
    gates: list[list[tuple[int, int]]] = [[] for _ in range(n_walls)]
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        for c in range(1, g.shape[1] - 1):
            col = int(g[wr, c])
            if col in (WALL, BG, SEP, GREEN, BLUE):
                continue
            if int(g[wr - 1, c]) == col and int(g[wr + 1, c]) == col:
                gates[wall_idx].append((c, col))
    opened = g.copy()
    for src in _parse_sources(g, variant):
        opened[src["r"], src["c"]] = BG
    return opened, gates


def _exit_color(gates: list[list[tuple[int, int]]], wall_idx: int, exit_col: int) -> int:
    if not (0 <= wall_idx < len(gates)):
        return PINK
    for center, color in gates[wall_idx]:
        if abs(center - exit_col) == 1:
            return color
    return PINK


def _pick_exit(
    exits: list[int],
    gates: list[list[tuple[int, int]]],
    source: dict[str, Any],
    wall_idx: int,
    variant: str,
) -> int | None:
    if not exits:
        return None
    if variant == "D7_0":
        return min(exits) if source["side"] == "left" else max(exits)
    if variant == "D7_1":
        preferred = PINK if source["pipe_idx"] % 2 == 0 else ORANGE
        candidates = [c for c in exits if _exit_color(gates, wall_idx, c) == preferred]
        if candidates:
            return min(candidates) if source["side"] != "right" else max(candidates)
        return min(exits)
    if variant == "D7_2":
        return exits[0] if len(exits) == 1 else None
    if variant == "D8_2" and len(exits) > 1:
        return min(exits) if source["side"] == "left" else max(exits)
    return exits[0] if len(exits) == 1 else None


def _pack_tokens(
    out: np.ndarray,
    opened: np.ndarray,
    records: list[dict[str, Any]],
    variant: str,
) -> None:
    buckets: dict[tuple[int, tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["r"], rec["seg"])].append(rec)
    for (r, seg), recs in buckets.items():
        a, b = seg
        sample = recs[0]
        if variant == "D4_1":
            recs.sort(key=lambda x: (x["source"]["pipe_idx"], x["source"]["order"]))
            for idx, rec in enumerate(recs):
                if a + idx <= b:
                    out[r, a + idx] = rec.get("settle_color", rec["source"]["color"])
            continue
        if variant == "D4_2":
            width = b - a + 1
            start = a + max(0, (width - len(recs)) // 2)
            for idx, rec in enumerate(recs):
                if start + idx <= b:
                    out[r, start + idx] = rec.get("settle_color", rec["source"]["color"])
            continue
        if variant == "D4_3":
            used: set[int] = set()
            for rec in sorted(recs, key=lambda x: (x["pref_c"], x["source"]["order"])):
                pref = rec["pref_c"]
                candidates = sorted(range(a, b + 1), key=lambda c: (abs(c - pref), c))
                for c in candidates:
                    if c not in used:
                        out[r, c] = rec.get("settle_color", rec["source"]["color"])
                        used.add(c)
                        break
            continue
        if variant == "D5_3":
            for rec in recs:
                rec["pack_mode"] = "near" if rec["source"]["color"] == BLUE else "far"
        pack_mode = sample.get("pack_mode", "far")
        if variant == "D4_0":
            pack_mode = "near"
        if pack_mode == "center":
            width = b - a + 1
            start = a + max(0, (width - len(recs)) // 2)
            for idx, rec in enumerate(recs):
                out[r, start + idx] = rec.get("settle_color", rec["source"]["color"])
            continue
        if pack_mode == "near":
            entries = _entries_above(opened, r, seg, sample["source"], sample["variant"])
            if len(entries) == 1:
                entry = entries[0]
                pack_end = a if (entry - a) <= (b - entry) else b
            else:
                pack_end = a if sample["final_c"] <= (a + b) // 2 else b
        else:
            entries = _entries_above(opened, r, seg, sample["source"], sample["variant"])
            if len(entries) == 1:
                entry = entries[0]
                pack_end = b if (entry - a) <= (b - entry) else a
            else:
                pack_end = a if sample["final_c"] <= a else b
        ordered = sorted(recs, key=lambda x: x["source"]["order"])
        if pack_end == a:
            for idx, rec in enumerate(ordered):
                if a + idx <= b:
                    out[r, a + idx] = rec.get("settle_color", rec["source"]["color"])
        else:
            for idx, rec in enumerate(ordered):
                if b - idx >= a:
                    out[r, b - idx] = rec.get("settle_color", rec["source"]["color"])


def _advance_token(
    opened: np.ndarray,
    gates: list[list[tuple[int, int]]],
    source: dict[str, Any],
    variant: str,
    visited_segments: set[tuple[int, int, int]],
    out: np.ndarray,
) -> dict[str, Any]:
    r = source["r"]
    c = source["c"]
    depth = 0
    while True:
        seg = _find_segment(opened, r, c, source, variant)
        if seg is None:
            break
        exits = _exits_below(opened, r, seg, source, variant)
        exit_col = _pick_exit(exits, gates, source, r // 2, variant)
        if exit_col is None:
            return {"r": r, "seg": seg, "final_c": c, "pref_c": c, "depth": depth, "source": source, "variant": variant}
        if variant == "D3_1":
            lo, hi = sorted((c, exit_col))
            out[r, lo : hi + 1] = TRACE
        next_r = r + 2
        next_seg = _find_segment(opened, next_r, exit_col, source, variant)
        if variant == "D12_2" and next_seg is not None:
            key = (next_r, next_seg[0], next_seg[1])
            if key in visited_segments:
                return {"r": r, "seg": seg, "final_c": c, "pref_c": c, "depth": depth, "source": source, "variant": variant}
            visited_segments.add(key)
        r = next_r
        c = exit_col
        depth += 1
        if r >= opened.shape[0]:
            break
    seg = _find_segment(opened, r, c, source, variant)
    return {"r": r, "seg": seg or (c, c), "final_c": c, "pref_c": c, "depth": depth, "source": source, "variant": variant}


def _solve_downward(
    opened: np.ndarray,
    gates: list[list[tuple[int, int]]],
    sources: list[dict[str, Any]],
    variant: str,
) -> np.ndarray:
    out = opened.copy()
    visited_segments: set[tuple[int, int, int]] = set()
    records = [_advance_token(opened, gates, source, variant, visited_segments, out) for source in sources]
    for rec in records:
        if variant == "D12_1":
            rec["settle_color"] = DEPTH_PALETTE[min(rec["depth"], len(DEPTH_PALETTE) - 1)]
        elif variant == "D5_3":
            rec["settle_color"] = rec["source"]["color"]
            rec["pack_mode"] = "near" if rec["source"]["color"] == BLUE else "far"
    _pack_tokens(out, opened, records, variant)
    if variant == "D9_2":
        for wall_idx, row_gates in enumerate(gates):
            wr = 2 * wall_idx + 1
            for center, _color in row_gates:
                if out[wr - 1, center] not in (GREEN, BLUE, RED, YELLOW, GREY, PINK, ORANGE) and out[wr + 1, center] not in (
                    GREEN,
                    BLUE,
                    RED,
                    YELLOW,
                    GREY,
                    PINK,
                    ORANGE,
                ):
                    out[wr - 1, center] = BG
                    out[wr, center] = BG
                    out[wr + 1, center] = BG
    return out


def _solve_progressive(opened: np.ndarray, gates: list[list[tuple[int, int]]], sources: list[dict[str, Any]]) -> np.ndarray:
    current = np.full_like(opened, WALL)
    for r in range(0, opened.shape[0], 2):
        current[r] = opened[r]
    positions = [dict(src) for src in sources]
    for wall_idx, row_gates in enumerate(gates):
        wr = 2 * wall_idx + 1
        current[wr] = opened[wr]
        current[wr - 1] = np.where(opened[wr - 1] != WALL, opened[wr - 1], current[wr - 1])
        current[wr + 1] = np.where(opened[wr + 1] != WALL, opened[wr + 1], current[wr + 1])
        moved = True
        while moved:
            moved = False
            new_positions = []
            for src in positions:
                seg = _find_segment(current, src["r"], src["c"], src, "BASE")
                if seg is None:
                    new_positions.append(src)
                    continue
                exits = _exits_below(current, src["r"], seg, src, "BASE")
                if len(exits) != 1:
                    new_positions.append(src)
                    continue
                new_positions.append({**src, "r": src["r"] + 2, "c": exits[0]})
                moved = True
            positions = new_positions
    records = []
    for src in positions:
        seg = _find_segment(current, src["r"], src["c"], src, "BASE")
        records.append({"r": src["r"], "seg": seg or (src["c"], src["c"]), "final_c": src["c"], "pref_c": src["c"], "depth": 0, "source": src, "variant": "BASE"})
    out = current.copy()
    _pack_tokens(out, current, records, "BASE")
    return out


def _solve_sequential(opened: np.ndarray, gates: list[list[tuple[int, int]]], sources: list[dict[str, Any]]) -> np.ndarray:
    out = opened.copy()
    occupied = opened.copy()
    records = []
    for source in sorted(sources, key=lambda x: (x["pipe_idx"], x["c"])):
        src = dict(source)
        r = src["r"]
        c = src["c"]
        while True:
            seg = _find_segment(occupied, r, c, src, "BASE")
            if seg is None:
                break
            exits = _exits_below(occupied, r, seg, src, "BASE")
            if len(exits) != 1:
                break
            r += 2
            c = exits[0]
        seg = _find_segment(occupied, r, c, src, "BASE") or (c, c)
        rec = {"r": r, "seg": seg, "final_c": c, "pref_c": c, "depth": 0, "source": src, "variant": "BASE"}
        tmp = occupied.copy()
        _pack_tokens(tmp, occupied, [rec], "BASE")
        occupied = tmp
        records.append(rec)
    _pack_tokens(out, opened, records, "BASE")
    return out


def _close_posts(opened: np.ndarray, gates: list[list[tuple[int, int]]]) -> np.ndarray:
    out = opened.copy()
    for wall_idx, row_gates in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center, color in row_gates:
            out[wr - 1, center] = BG
            out[wr, center] = color
            out[wr + 1, center] = BG
            out[wr, center - 1] = color
            out[wr, center + 1] = color
    return out


def _build_input_from_opened(
    opened: np.ndarray,
    gates: list[list[tuple[int, int]]],
    variant: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grid = opened.copy()
    if variant != "D9_3":
        for wall_idx, row_gates in enumerate(gates):
            wr = 2 * wall_idx + 1
            for center, color in row_gates:
                grid[wr - 1, center] = BG
                grid[wr, center] = color
                grid[wr + 1, center] = BG
                grid[wr, center - 1] = color
                grid[wr, center + 1] = color
    sources = _place_sources(grid, gates, variant, rng)
    if variant == "D13_0":
        legend = np.full((1, grid.shape[1]), WALL, dtype=int)
        block = max(1, grid.shape[1] // max(1, len(range(0, grid.shape[0], 2))))
        for pipe_idx in range((grid.shape[0] + 1) // 2):
            start = min(grid.shape[1] - 1, pipe_idx * block)
            end = min(grid.shape[1], start + block)
            legend[0, start:end] = GREEN if pipe_idx % 2 == 0 else BLUE
        grid = np.vstack([legend, grid])
    return grid, sources


def _solve_variant_core(grid: np.ndarray, variant: str) -> np.ndarray:
    if variant == "D3_0":
        core = np.flipud(grid)
        return np.flipud(_solve_variant_core(core, "BASE"))
    if variant == "D6_0":
        core = np.rot90(grid, 1)
        return np.rot90(_solve_variant_core(core, "BASE"), -1)
    has_legend = variant == "D13_0"
    core = np.array(grid[1:] if has_legend else grid, dtype=int)
    if variant == "D9_3":
        opened, gates = _parse_open_input(core, variant)
    else:
        opened, gates = _open_from_bar_input(core, variant)
    sources = _parse_sources(core, variant)
    if variant in ("BASE", "D2_1", "D4_0", "D4_1", "D4_2", "D4_3", "D5_0", "D5_1", "D5_2", "D5_3", "D7_0", "D7_1", "D7_2", "D8_2", "D12_1", "D12_2", "D13_0", "D3_1"):
        out = _solve_downward(opened, gates, sources, variant)
    elif variant == "D9_0":
        out = _solve_progressive(opened, gates, sources)
    elif variant == "D9_1":
        out = _solve_sequential(opened, gates, sources)
    elif variant == "D9_2":
        out = _solve_downward(opened, gates, sources, "D9_2")
    elif variant == "D9_3":
        out = _close_posts(_solve_downward(opened, gates, sources, "BASE"), gates)
    else:
        raise ValueError(variant)
    if has_legend:
        return np.vstack([grid[:1], out])
    return out


def _generate_variant(variant: str) -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        try:
            if variant in ("D7_0", "D7_2"):
                gates, holes, width = _build_multi_exit_structure(rng, two_colors=False)
            elif variant == "D7_1":
                gates, holes, width = _build_multi_exit_structure(rng, two_colors=True)
            else:
                raw_gates, holes, width = _build_standard_structure(rng)
                gates = _assign_gate_colors(raw_gates, variant)
                if variant == "D8_2":
                    inserted = False
                    for pipe_idx in range(1, len(holes) - 1):
                        candidates = [c for c in range(2, width - 2) if c not in holes[pipe_idx]]
                        if not candidates:
                            continue
                        holes[pipe_idx][candidates[len(candidates) // 2]] = SEP
                        inserted = True
                        break
                    if not inserted:
                        continue
            opened = _build_opened_grid(gates, holes, width)
            inp, _sources = _build_input_from_opened(opened, gates, variant, rng)
            out = _solve_variant_core(inp, variant)
            if variant in ("D4_0", "D4_1", "D4_2", "D4_3", "D7_0", "D7_1", "D12_1", "D3_1"):
                base_key = "BASE"
                base_inp = inp[1:] if variant == "D13_0" else inp
                base_out = _solve_variant_core(base_inp, base_key)
                cmp_out = out[1:] if variant == "D13_0" else out
                if np.array_equal(base_out, cmp_out):
                    continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed for {variant}")


def _variant_code(dim_id: str, alt_index: int) -> str:
    return f"{dim_id}_{alt_index}"


def build_functions(dim_id: str, alt_index: int):
    variant = _variant_code(dim_id, alt_index)

    def main(grid: np.ndarray) -> np.ndarray:
        return _solve_variant_core(np.array(grid, dtype=int), variant)

    def generate_input() -> np.ndarray:
        return _generate_variant(variant)

    return main, generate_input
