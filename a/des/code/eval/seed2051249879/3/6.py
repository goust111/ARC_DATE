from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400
LENGTH_RANGE = (2, 5)
SCENARIOS = ("simple", "chain", "double")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(counts.argmax())])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _detect_gates(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gate_mask = np.zeros((h, w), dtype=bool)
    gates: list[dict] = []

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        pts = np.argwhere(g == int(color))
        if pts.shape[0] != 2:
            continue
        (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in pts]
        if r0 == r1 and abs(c0 - c1) == 2:
            channel = (r0, (c0 + c1) // 2)
            orient = "horizontal"
        elif c0 == c1 and abs(r0 - r1) == 2:
            channel = ((r0 + r1) // 2, c0)
            orient = "vertical"
        else:
            continue
        if int(g[channel]) != int(bg):
            continue
        gates.append(
            {
                "color": int(color),
                "cells": [(r0, c0), (r1, c1)],
                "channel": (int(channel[0]), int(channel[1])),
                "orient": orient,
                "attachments": [],
                "connected": False,
            }
        )
        gate_mask[r0, c0] = True
        gate_mask[r1, c1] = True

    gates.sort(key=lambda gate: (gate["channel"][0], gate["channel"][1], gate["color"]))
    gate_mask[:] = False
    for gate in gates:
        for r, c in gate["cells"]:
            gate_mask[r, c] = True
    return gates, gate_mask


def _extract_fragments(grid: np.ndarray, bg: int, gate_mask: np.ndarray, gates: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gate_by_channel = {gate["channel"]: idx for idx, gate in enumerate(gates)}

    fragments: list[dict] = []
    for comp in _components((g != int(bg)) & (~gate_mask)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "fragment must be monochrome"
        color = next(iter(colors))
        rows = sorted({int(r) for r, _c in comp})
        cols = sorted({int(c) for _r, c in comp})
        if len(rows) == 1:
            r = rows[0]
            assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal fragment must be contiguous"
            orient = "h"
            coords = [(r, c) for c in cols]
            endpoints = [(r, cols[0]), (r, cols[-1])]
        else:
            assert len(cols) == 1, "fragment must be straight"
            c = cols[0]
            assert rows == list(range(rows[0], rows[-1] + 1)), "vertical fragment must be contiguous"
            orient = "v"
            coords = [(r, c) for r in rows]
            endpoints = [(rows[0], c), (rows[-1], c)]

        gate_ids: list[int | None] = []
        gate_sides: list[str | None] = []
        edge_flags: list[bool] = []
        for end_idx, (r, c) in enumerate(endpoints):
            edge_flags.append(r == 0 or r == h - 1 or c == 0 or c == w - 1)
            gate_id: int | None = None
            gate_side: str | None = None
            if orient == "h":
                dc = -1 if end_idx == 0 else 1
                ch = (r, c + dc)
                if ch in gate_by_channel and gates[gate_by_channel[ch]]["orient"] == "vertical":
                    gate_id = int(gate_by_channel[ch])
                    gate_side = "left" if c < ch[1] else "right"
            else:
                dr = -1 if end_idx == 0 else 1
                ch = (r + dr, c)
                if ch in gate_by_channel and gates[gate_by_channel[ch]]["orient"] == "horizontal":
                    gate_id = int(gate_by_channel[ch])
                    gate_side = "top" if r < ch[0] else "bottom"
            gate_ids.append(gate_id)
            gate_sides.append(gate_side)

        fragments.append(
            {
                "color": int(color),
                "orient": orient,
                "coords": coords,
                "endpoints": endpoints,
                "edge_flags": edge_flags,
                "gate_ids": gate_ids,
                "gate_sides": gate_sides,
                "conn_gate": [None, None],
            }
        )

    fragments.sort(key=lambda frag: (min(r for r, _c in frag["coords"]), min(c for _r, c in frag["coords"])))
    return fragments


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    gates, gate_mask = _detect_gates(g, bg)
    fragments = _extract_fragments(g, bg, gate_mask, gates)

    for frag_id, frag in enumerate(fragments):
        for end_idx, gate_id in enumerate(frag["gate_ids"]):
            if gate_id is None:
                continue
            gates[int(gate_id)]["attachments"].append(
                (int(frag_id), int(end_idx), str(frag["gate_sides"][end_idx]))
            )

    for gate in gates:
        if gate["orient"] == "vertical":
            needed = {"left", "right"}
        else:
            needed = {"top", "bottom"}
        if len(gate["attachments"]) != 2:
            continue
        sides = {side for _frag_id, _end_idx, side in gate["attachments"]}
        frags = {frag_id for frag_id, _end_idx, _side in gate["attachments"]}
        if sides != needed or len(frags) != 2:
            continue
        gate["connected"] = True
        for frag_id, end_idx, _side in gate["attachments"]:
            fragments[int(frag_id)]["conn_gate"][int(end_idx)] = gate

    return g, int(bg), gates, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned line fragments through 1-cell gates and propagate gate colors from the unique edge start.

    Gates are two same-colored pixels with one background cell between them. If opposite fragment
    endpoints meet the same gate channel, the channel is filled and every later fragment along the
    path inherits that gate color. The direction is determined by the only open edge endpoint in
    each connected gate-fragment chain.
    """
    g, bg, gates, fragments = _parse_scene(grid)
    out = g.copy()

    adjacency: list[list[tuple[int, dict]]] = [[] for _ in fragments]
    for gate in gates:
        if not gate["connected"]:
            continue
        (fa, ea, _sa), (fb, eb, _sb) = gate["attachments"]
        adjacency[fa].append((fb, gate))
        adjacency[fb].append((fa, gate))

    seen_frags: set[int] = set()
    for start_frag in range(len(fragments)):
        if start_frag in seen_frags:
            continue
        stack = [start_frag]
        comp_frags: list[int] = []
        comp_gates: set[tuple[int, int]] = set()
        while stack:
            frag_id = stack.pop()
            if frag_id in seen_frags:
                continue
            seen_frags.add(frag_id)
            comp_frags.append(frag_id)
            for nbr, gate in adjacency[frag_id]:
                comp_gates.add(gate["channel"])
                if nbr not in seen_frags:
                    stack.append(nbr)

        if not comp_gates:
            continue

        edge_starts: list[tuple[int, int]] = []
        for frag_id in comp_frags:
            frag = fragments[frag_id]
            for end_idx in range(2):
                if frag["edge_flags"][end_idx] and frag["conn_gate"][end_idx] is None:
                    edge_starts.append((frag_id, end_idx))

        if len(edge_starts) != 1:
            continue

        frag_id, entry_end = edge_starts[0]
        traversed: set[tuple[int, int]] = set()
        while True:
            frag = fragments[frag_id]
            exit_end = 1 - int(entry_end)
            gate = frag["conn_gate"][exit_end]
            if gate is None:
                break
            gate_key = gate["channel"]
            if gate_key in traversed:
                break
            traversed.add(gate_key)
            out[gate["channel"]] = gate["color"]

            (fa, ea, _sa), (fb, eb, _sb) = gate["attachments"]
            if fa == frag_id and ea == exit_end:
                next_frag = fb
                next_entry = eb
            elif fb == frag_id and eb == exit_end:
                next_frag = fa
                next_entry = ea
            else:
                break

            for r, c in fragments[next_frag]["coords"]:
                out[r, c] = gate["color"]
            frag_id = int(next_frag)
            entry_end = int(next_entry)

    return out


def _lane_positions(low: int, high: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    choices = list(range(low, high + 1))
    for _ in range(60):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= 3 for a, b in zip(picked, picked[1:], strict=False)):
            return picked
    return None


def _border_intervals(limit: int, count: int) -> list[tuple[int, int]] | None:
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        for _try in range(80):
            length = random.randint(*LENGTH_RANGE)
            start = random.randint(0, limit - length)
            end = start + length - 1
            if all(end < a - 1 or start > b + 1 for a, b in intervals):
                intervals.append((start, end))
                break
        else:
            return None
    return intervals


def _path_span(frag_lengths: list[int]) -> int:
    return int(sum(frag_lengths) + len(frag_lengths))


def _paint_horizontal_path(
    grid: np.ndarray,
    row: int,
    start_side: str,
    frag_lengths: list[int],
    frag_colors: list[int],
    gate_colors: list[int],
) -> None:
    w = grid.shape[1]
    if start_side == "left":
        cur = 0
        for idx, length in enumerate(frag_lengths):
            grid[row, cur : cur + length] = int(frag_colors[idx])
            cur += length
            gate_col = int(gate_colors[idx])
            grid[row - 1, cur] = gate_col
            grid[row + 1, cur] = gate_col
            if idx < len(frag_lengths) - 1:
                cur += 1
    else:
        cur = w - 1
        for idx, length in enumerate(frag_lengths):
            grid[row, cur - length + 1 : cur + 1] = int(frag_colors[idx])
            cur -= length
            gate_col = int(gate_colors[idx])
            grid[row - 1, cur] = gate_col
            grid[row + 1, cur] = gate_col
            if idx < len(frag_lengths) - 1:
                cur -= 1


def _paint_vertical_path(
    grid: np.ndarray,
    col: int,
    start_side: str,
    frag_lengths: list[int],
    frag_colors: list[int],
    gate_colors: list[int],
) -> None:
    h = grid.shape[0]
    if start_side == "top":
        cur = 0
        for idx, length in enumerate(frag_lengths):
            grid[cur : cur + length, col] = int(frag_colors[idx])
            cur += length
            gate_col = int(gate_colors[idx])
            grid[cur, col - 1] = gate_col
            grid[cur, col + 1] = gate_col
            if idx < len(frag_lengths) - 1:
                cur += 1
    else:
        cur = h - 1
        for idx, length in enumerate(frag_lengths):
            grid[cur - length + 1 : cur + 1, col] = int(frag_colors[idx])
            cur -= length
            gate_col = int(gate_colors[idx])
            grid[cur, col - 1] = gate_col
            grid[cur, col + 1] = gate_col
            if idx < len(frag_lengths) - 1:
                cur -= 1


def _scenario_spec() -> tuple[str, list[int], int]:
    scenario = random.choice(SCENARIOS)
    if scenario == "simple":
        return scenario, [1], random.randint(1, 2)
    if scenario == "chain":
        return scenario, [2], random.randint(0, 1)
    return scenario, [1, 1], random.randint(0, 1)


def _gen_attempt() -> np.ndarray | None:
    scenario, used_gate_counts, isolated_count = _scenario_spec()
    orientation = random.choice(["h", "v"])
    path_count = len(used_gate_counts)

    path_specs: list[dict] = []
    for used_gates in used_gate_counts:
        frag_lengths = [random.randint(*LENGTH_RANGE) for _ in range(used_gates + 1)]
        path_specs.append(
            {
                "used_gates": used_gates,
                "frag_lengths": frag_lengths,
                "span": _path_span(frag_lengths),
            }
        )

    needed_colors = isolated_count + sum(2 * spec["used_gates"] + 2 for spec in path_specs)
    if needed_colors > 9:
        return None

    if orientation == "h":
        min_h = 8 if path_count == 1 else 10
        min_w = max(spec["span"] for spec in path_specs)
        if min_w > 18:
            return None
        h = random.randint(min_h, 18)
        w = random.randint(min_w, min(20, min_w + 6))
        rows = _lane_positions(2, h - 3, path_count)
        if rows is None:
            return None
    else:
        min_w = 8 if path_count == 1 else 10
        min_h = max(spec["span"] for spec in path_specs)
        if min_h > 18:
            return None
        h = random.randint(min_h, min(20, min_h + 6))
        w = random.randint(min_w, 18)
        cols = _lane_positions(2, w - 3, path_count)
        if cols is None:
            return None

    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    color_idx = 0

    grid = np.full((h, w), bg, dtype=int)

    for path_idx, spec in enumerate(path_specs):
        frag_count = spec["used_gates"] + 1
        frag_colors = palette[color_idx : color_idx + frag_count]
        color_idx += frag_count
        gate_colors = palette[color_idx : color_idx + frag_count]
        color_idx += frag_count
        if len(set(frag_colors) | set(gate_colors)) != 2 * frag_count:
            return None
        if orientation == "h":
            _paint_horizontal_path(
                grid,
                rows[path_idx],
                random.choice(["left", "right"]),
                spec["frag_lengths"],
                frag_colors,
                gate_colors,
            )
        else:
            _paint_vertical_path(
                grid,
                cols[path_idx],
                random.choice(["top", "bottom"]),
                spec["frag_lengths"],
                frag_colors,
                gate_colors,
            )

    if isolated_count > 0:
        iso_colors = palette[color_idx : color_idx + isolated_count]
        color_idx += isolated_count
        if len(iso_colors) != isolated_count:
            return None
        if orientation == "h":
            sides = [0, h - 1]
            chosen_rows = random.sample(sides, isolated_count)
            intervals = _border_intervals(w, isolated_count)
            if intervals is None:
                return None
            for row, (c0, c1), color in zip(chosen_rows, intervals, iso_colors, strict=False):
                grid[row, c0 : c1 + 1] = int(color)
        else:
            sides = [0, w - 1]
            chosen_cols = random.sample(sides, isolated_count)
            intervals = _border_intervals(h, isolated_count)
            if intervals is None:
                return None
            for col, (r0, r1), color in zip(chosen_cols, intervals, iso_colors, strict=False):
                grid[r0 : r1 + 1, col] = int(color)

    if _bg_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(main(grid), out):
        return None

    parsed_grid, parsed_bg, gates, fragments = _parse_scene(grid)
    if parsed_bg != bg or not np.array_equal(parsed_grid, grid):
        return None
    if not any(gate["connected"] for gate in gates):
        return None

    connected_gate_count = sum(1 for gate in gates if gate["connected"])
    if scenario == "simple" and connected_gate_count != 1:
        return None
    if scenario == "chain" and connected_gate_count != 2:
        return None
    if scenario == "double" and connected_gate_count != 2:
        return None

    if scenario == "chain":
        if not any(sum(g is not None for g in frag["conn_gate"]) == 2 for frag in fragments):
            return None

    if isolated_count > 0:
        isolated_fragments = [
            frag
            for frag in fragments
            if sum(gate is not None for gate in frag["conn_gate"]) == 0
            and sum(bool(flag) for flag in frag["edge_flags"]) >= 2
        ]
        if len(isolated_fragments) < isolated_count:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
