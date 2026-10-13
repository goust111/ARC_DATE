from __future__ import annotations

from collections import defaultdict
import heapq
import random

import numpy as np


BG = 0
DIR8 = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
DIR_INDEX = {d: i for i, d in enumerate(DIR8)}

COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 29)
GRID_W_RANGE = (15, 29)
LINE_LAYOUTS = ((1, 1), (2, 1), (1, 2), (2, 2))
MIN_LINE_LEN = 6
MIN_PARAM_SEP = 4
GREY_CANDIDATE = 5


def _turn_distance(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    i = DIR_INDEX[d1]
    j = DIR_INDEX[d2]
    diff = abs(i - j)
    return min(diff, 8 - diff)


def _dominant_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _infer_path_color(grid: np.ndarray, bg: int) -> int:
    border_vals = set(int(v) for v in np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1])))
    border_vals.discard(int(bg))
    candidates = set(int(v) for v in np.unique(grid)) - {int(bg)} - border_vals
    assert len(candidates) == 1, "expected exactly one interior-only path color"
    return int(next(iter(candidates)))


def _scan_edge_runs(grid: np.ndarray, bg: int, path_color: int) -> list[dict]:
    h, w = grid.shape
    runs: list[dict] = []

    def scan(cells: list[tuple[int, int]], side: str) -> None:
        i = 0
        while i < len(cells):
            r, c = cells[i]
            color = int(grid[r, c])
            if color in (int(bg), int(path_color)):
                i += 1
                continue
            j = i
            while j + 1 < len(cells) and int(grid[cells[j + 1]]) == color:
                j += 1
            run_cells = cells[i : j + 1]
            width = len(run_cells)
            center_r, center_c = run_cells[width // 2]
            if side == "top":
                anchor = (1, center_c)
            elif side == "bottom":
                anchor = (h - 2, center_c)
            elif side == "left":
                anchor = (center_r, 1)
            else:
                anchor = (center_r, w - 2)
            runs.append(
                {
                    "side": side,
                    "color": color,
                    "width": width,
                    "cells": tuple(run_cells),
                    "center": (center_r, center_c),
                    "anchor": anchor,
                }
            )
            i = j + 1

    scan([(0, c) for c in range(w)], "top")
    scan([(h - 1, c) for c in range(w)], "bottom")
    scan([(r, 0) for r in range(h)], "left")
    scan([(r, w - 1) for r in range(h)], "right")
    return runs


def _pair_endpoints(runs: list[dict]) -> list[tuple[dict, dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(int(run["color"]), int(run["width"]))].append(run)
    pairs = []
    for key, items in grouped.items():
        assert len(items) == 2, f"expected exactly two endpoints for {key}"
        pairs.append((items[0], items[1]))
    return pairs


def _gray_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out = []
    for dr, dc in DIR8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _trace_route(
    gray_mask: np.ndarray,
    start_border: tuple[int, int],
    start_anchor: tuple[int, int],
    target_anchor: tuple[int, int],
) -> list[tuple[int, int]]:
    start_state = (start_border, start_anchor)
    heap = [(0, 0, start_border, start_anchor)]
    best = {start_state: (0, 0)}
    parent: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = {}

    while heap:
        turns, steps, prev, curr = heapq.heappop(heap)
        state = (prev, curr)
        if best.get(state) != (turns, steps):
            continue
        if curr == target_anchor:
            cells = [curr]
            while state != start_state:
                state = parent[state]
                cells.append(state[1])
            cells.reverse()
            assert len(cells) == len(set(cells)), "route must not self-intersect"
            return cells

        incoming = (curr[0] - prev[0], curr[1] - prev[1])
        for nxt in _gray_neighbors(gray_mask, curr):
            if nxt == prev:
                continue
            outgoing = (nxt[0] - curr[0], nxt[1] - curr[1])
            cost = (turns + _turn_distance(incoming, outgoing), steps + 1)
            next_state = (curr, nxt)
            if next_state not in best or cost < best[next_state]:
                best[next_state] = cost
                parent[next_state] = state
                heapq.heappush(heap, (cost[0], cost[1], curr, nxt))

    raise AssertionError("failed to trace a route between paired endpoints")


def _trace_pair(gray_mask: np.ndarray, a: dict, b: dict) -> list[tuple[int, int]]:
    path_ab = _trace_route(gray_mask, a["center"], a["anchor"], b["anchor"])
    path_ba = _trace_route(gray_mask, b["center"], b["anchor"], a["anchor"])
    assert path_ab == list(reversed(path_ba)), "paired tracing should agree from both sides"
    return path_ab


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey edge-to-edge route by the endpoint pair it connects, using direction continuity to resolve woven crossings.

    Input structure:
    - The background is black.
    - Colored endpoint bars lie on the outer border; each bar is a contiguous 1-3 pixel run, and every color/width combination appears exactly twice.
    - Inside the border, grey pixels form one-pixel-wide 8-connected routes that join the paired endpoint bars.
    - Opposite-slope routes may touch inside compact woven intersections, so several routes can merge into one grey cluster without sharing the same endpoint pair.

    Rule:
    - Pair the two border bars that have the same color and the same border-run width.
    - For each pair, start from the grey anchor cell just inside one bar and trace the unique route to the partner bar, preferring the continuation that preserves the current direction.
    - Recolor every traced grey pixel with the pair's color.
    - Keep the colored endpoint bars unchanged.

    Key insight:
    - Plain connected components are not enough because woven crossings fuse several routes into one grey network; the solver must combine endpoint pairing with local direction continuity to separate the routes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    path_color = _infer_path_color(g, bg)
    runs = _scan_edge_runs(g, bg, path_color)
    assert runs, "expected colored edge endpoints"
    pairs = _pair_endpoints(runs)

    gray_mask = g == int(path_color)
    out = g.copy()

    visited = np.zeros(g.shape, dtype=bool)
    for a, b in pairs:
        assert gray_mask[a["anchor"]], "endpoint anchor must touch the grey network"
        assert gray_mask[b["anchor"]], "endpoint anchor must touch the grey network"
        route = _trace_pair(gray_mask, a, b)
        for r, c in route:
            if visited[r, c]:
                assert int(out[r, c]) == int(a["color"]), "shared path cell recolored inconsistently"
            visited[r, c] = True
            out[r, c] = int(a["color"])

    assert np.all(visited[gray_mask]), "every grey path cell must be assigned to a pair"
    return out


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _draw_segment(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = _sign(r1 - r0)
    dc = _sign(c1 - c0)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    assert abs(r1 - r0) in (0, steps)
    assert abs(c1 - c0) in (0, steps)
    return [(r0 + dr * i, c0 + dc * i) for i in range(steps + 1)]


def _build_pos_line(h: int, w: int, param: int) -> dict | None:
    endpoints = []
    candidates = [
        ("top", (1, 1 - param), (0, 1 - param)),
        ("left", (1 + param, 1), (1 + param, 0)),
        ("bottom", (h - 2, h - 2 - param), (h - 1, h - 2 - param)),
        ("right", (w - 2 + param, w - 2), (w - 2 + param, w - 1)),
    ]
    seen = set()
    for side, anchor, center in candidates:
        ar, ac = anchor
        if not (1 <= ar <= h - 2 and 1 <= ac <= w - 2):
            continue
        if anchor in seen:
            return None
        seen.add(anchor)
        endpoints.append({"side": side, "anchor": anchor, "center": center})
    if len(endpoints) != 2:
        return None
    endpoints.sort(key=lambda ep: ep["anchor"])
    cells = _draw_segment(endpoints[0]["anchor"], endpoints[1]["anchor"])
    if len(cells) < MIN_LINE_LEN:
        return None
    return {"kind": "pos", "param": int(param), "cells": tuple(cells), "endpoints": tuple(endpoints)}


def _build_neg_line(h: int, w: int, param: int) -> dict | None:
    endpoints = []
    candidates = [
        ("top", (1, param - 1), (0, param - 1)),
        ("left", (param - 1, 1), (param - 1, 0)),
        ("bottom", (h - 2, param - (h - 2)), (h - 1, param - (h - 2))),
        ("right", (param - (w - 2), w - 2), (param - (w - 2), w - 1)),
    ]
    seen = set()
    for side, anchor, center in candidates:
        ar, ac = anchor
        if not (1 <= ar <= h - 2 and 1 <= ac <= w - 2):
            continue
        if anchor in seen:
            return None
        seen.add(anchor)
        endpoints.append({"side": side, "anchor": anchor, "center": center})
    if len(endpoints) != 2:
        return None
    endpoints.sort(key=lambda ep: ep["anchor"])
    cells = _draw_segment(endpoints[0]["anchor"], endpoints[1]["anchor"])
    if len(cells) < MIN_LINE_LEN:
        return None
    return {"kind": "neg", "param": int(param), "cells": tuple(cells), "endpoints": tuple(endpoints)}


def _candidate_lines(h: int, w: int, parity: int, kind: str) -> list[dict]:
    out = []
    if kind == "pos":
        for param in range(-(w - 3), h - 2):
            if param % 2 != parity:
                continue
            spec = _build_pos_line(h, w, param)
            if spec is not None:
                out.append(spec)
    else:
        for param in range(2, h + w - 4):
            if param % 2 != parity:
                continue
            spec = _build_neg_line(h, w, param)
            if spec is not None:
                out.append(spec)
    random.shuffle(out)
    return out


def _pick_parallel_lines(candidates: list[dict], count: int) -> list[dict] | None:
    chosen: list[dict] = []
    for spec in candidates:
        if all(abs(int(spec["param"]) - int(prev["param"])) >= MIN_PARAM_SEP for prev in chosen):
            chosen.append(spec)
        if len(chosen) == count:
            return chosen
    return None


def _cross_count(pos_lines: list[dict], neg_lines: list[dict], h: int, w: int) -> int:
    count = 0
    for pos_line in pos_lines:
        a = int(pos_line["param"])
        for neg_line in neg_lines:
            b = int(neg_line["param"])
            if (a + b) % 2 == 0:
                continue
            row0 = (a + b - 1) // 2
            col0 = (b - a - 1) // 2
            if 2 <= row0 and row0 + 1 <= h - 3 and 2 <= col0 and col0 + 1 <= w - 3:
                count += 1
    return count


def _edge_interval(endpoint: dict, width: int, h: int, w: int) -> tuple[int, int] | None:
    side = endpoint["side"]
    center_r, center_c = endpoint["center"]
    axis = center_c if side in ("top", "bottom") else center_r
    start = axis - width // 2
    end = start + width - 1
    limit = w - 2 if side in ("top", "bottom") else h - 2
    if start < 1 or end > limit:
        return None
    return int(start), int(end)


def _assign_widths(lines: list[dict], h: int, w: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(1, 3) for _ in lines]
        occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
        ok = True
        for line, width in zip(lines, widths, strict=False):
            for endpoint in line["endpoints"]:
                interval = _edge_interval(endpoint, int(width), h, w)
                if interval is None:
                    ok = False
                    break
                start, end = interval
                for prev_start, prev_end in occupied[endpoint["side"]]:
                    if not (end < prev_start - 1 or prev_end < start - 1):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            for endpoint in line["endpoints"]:
                occupied[endpoint["side"]].append(_edge_interval(endpoint, int(width), h, w))
        if ok:
            return widths
    return None


def _edge_cells(endpoint: dict, width: int, h: int, w: int) -> list[tuple[int, int]]:
    start, _end = _edge_interval(endpoint, width, h, w)
    side = endpoint["side"]
    if side == "top":
        return [(0, start + k) for k in range(width)]
    if side == "bottom":
        return [(h - 1, start + k) for k in range(width)]
    if side == "left":
        return [(start + k, 0) for k in range(width)]
    return [(start + k, w - 1) for k in range(width)]


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_pos, n_neg = random.choice(LINE_LAYOUTS)
    pos_parity = random.randint(0, 1)
    neg_parity = 1 - pos_parity

    pos_candidates = _candidate_lines(h, w, pos_parity, "pos")
    neg_candidates = _candidate_lines(h, w, neg_parity, "neg")
    pos_lines = _pick_parallel_lines(pos_candidates, n_pos)
    neg_lines = _pick_parallel_lines(neg_candidates, n_neg)
    if pos_lines is None or neg_lines is None:
        return None
    if _cross_count(pos_lines, neg_lines, h, w) < 1:
        return None

    lines = pos_lines + neg_lines
    path_cells = [cell for line in lines for cell in line["cells"]]
    if len(path_cells) != len(set(path_cells)):
        return None

    widths = _assign_widths(lines, h, w)
    if widths is None:
        return None

    colors = random.sample(list(COLOR_POOL), len(lines))
    output = np.full((h, w), BG, dtype=int)
    path_mask = np.zeros((h, w), dtype=bool)

    for line, width, color in zip(lines, widths, colors, strict=False):
        for r, c in line["cells"]:
            if int(output[r, c]) != BG:
                return None
            output[r, c] = int(color)
            path_mask[r, c] = True
        for endpoint in line["endpoints"]:
            for r, c in _edge_cells(endpoint, int(width), h, w):
                if int(output[r, c]) not in (BG, int(color)):
                    return None
                output[r, c] = int(color)

    input_grid = output.copy()
    input_grid[path_mask] = GREY_CANDIDATE
    if np.all(input_grid == BG):
        return None
    if np.any(input_grid[0] == GREY_CANDIDATE) or np.any(input_grid[-1] == GREY_CANDIDATE):
        return None
    if np.any(input_grid[:, 0] == GREY_CANDIDATE) or np.any(input_grid[:, -1] == GREY_CANDIDATE):
        return None

    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid woven-path example")
