from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Color = int
Coord = tuple[int, int]
Box = tuple[int, int, int, int]

BG = 0
PALETTE = (1, 2, 3, 4)
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
PAIR_COUNT_OPTIONS = (3, 4)
PAIR_COUNT_WEIGHTS = (3, 5)
RECT_SIZE_OPTIONS = (2, 3)
GAP_RANGE = (2, 5)
MIN_BBOX_GAP = 2
MAX_GENERATION_ATTEMPTS = 800
RAY_DIRS = (
    (-1, 0, "up"),
    (1, 0, "down"),
    (0, -1, "left"),
    (0, 1, "right"),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude:
        colors = [color for color in colors if color not in exclude]
    return colors


def _component_dict(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _assert_solid_rectangle(grid: np.ndarray, color: int, comp: dict) -> Box:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert int(comp["area"]) == h * w, "each component must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be filled"
    return r0, c0, r1, c1


def _facing_corners(upper: Box, lower: Box) -> tuple[Coord, Coord]:
    _, uc0, ur1, uc1 = upper
    lr0, lc0, _, lc1 = lower
    assert ur1 < lr0, "paired rectangles must be vertically separated"

    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("paired rectangles must be diagonally separated")


def _pair_descriptors(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    pairs: list[dict] = []

    for color in _colors_present(grid, exclude={bg}):
        comps = _component_dict(grid, color)
        assert len(comps) == 2, "each foreground color must appear in exactly two rectangles"
        rects = [_assert_solid_rectangle(grid, color, comp) for comp in comps]
        rects.sort(key=lambda box: (box[0], box[1]))
        upper, lower = rects
        start, end = _facing_corners(upper, lower)

        row_delta = end[0] - start[0]
        col_delta = end[1] - start[1]
        assert row_delta > 0 and abs(row_delta) == abs(col_delta), "facing corners must lie on a 45-degree diagonal"

        dr = 1
        dc = 1 if col_delta > 0 else -1
        diag = [(start[0] + step * dr, start[1] + step * dc) for step in range(1, row_delta)]
        pairs.append(
            {
                "color": color,
                "upper": upper,
                "lower": lower,
                "endpoints": (start, end),
                "diag": diag,
            }
        )

    return pairs


def _trace_ray(blocked: np.ndarray, endpoint: Coord, dr: int, dc: int) -> tuple[list[Coord], tuple[str, Coord | None]]:
    h, w = blocked.shape
    r = endpoint[0] + dr
    c = endpoint[1] + dc
    cells: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        if blocked[r, c]:
            return cells, ("blocked", (r, c))
        cells.append((r, c))
        r += dr
        c += dc
    return cells, ("edge", None)


def _simulate(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pairs = sorted(_pair_descriptors(g), key=lambda item: item["color"])
    out = g.copy()
    rectangle_mask = g != bg
    diagonal_mask = np.zeros_like(g, dtype=bool)
    ray_mask = np.zeros_like(g, dtype=bool)
    traces: list[dict] = []

    for pair in pairs:
        for r, c in pair["diag"]:
            assert int(out[r, c]) == bg, "diagonal connector must lie on background"
            out[r, c] = int(pair["color"])
            diagonal_mask[r, c] = True

    for pair in pairs:
        blocked = out != bg
        paint: set[Coord] = set()
        for endpoint in pair["endpoints"]:
            for dr, dc, name in RAY_DIRS:
                cells, stop = _trace_ray(blocked, endpoint, dr, dc)
                paint.update(cells)
                if with_trace:
                    reason = stop[0]
                    blocker = stop[1]
                    if reason == "blocked":
                        br, bc = blocker
                        if ray_mask[br, bc]:
                            reason = "earlier_ray"
                        elif diagonal_mask[br, bc] and not rectangle_mask[br, bc]:
                            reason = "diagonal"
                        else:
                            reason = "rectangle"
                    traces.append(
                        {
                            "color": int(pair["color"]),
                            "endpoint": endpoint,
                            "dir": name,
                            "cells": cells,
                            "stop_reason": reason,
                            "blocker": blocker,
                        }
                    )
        for r, c in paint:
            out[r, c] = int(pair["color"])
            ray_mask[r, c] = True

    if with_trace:
        return out, pairs, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color diagonal rectangle pair, then emit axis-aligned rays from the connector endpoints.

    Input structure:
    - The grid has a dominant black background and 3-4 foreground colors.
    - Each foreground color appears as exactly two separated solid rectangles of the same color.
    - For each color, the upper rectangle and lower rectangle are diagonally offset so their facing corners lie on one 45-degree line.

    Rule:
    - First, for every color, draw the open diagonal segment between the two facing corners, keeping the rectangles unchanged.
    - After all diagonal segments exist, use the two facing-corner pixels themselves as emitters and extend rays up, down, left, and right in that color.
    - Rays stop before the first non-background pixel; diagonals, rectangles, and rays drawn by earlier colors all count as blockers, while the two emitters of one color fire simultaneously and do not block one another.

    Key insight:
    - The diagonal connector is an intermediate geometric object: it creates the ray sources and also becomes part of the obstacle field that truncates later rays.
    """
    return _simulate(grid, with_trace=False)


def _bbox_gap_ok(a: Box, b: Box) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(ar0 - br1 - 1, br0 - ar1 - 1, 0)
    col_gap = max(ac0 - bc1 - 1, bc0 - ac1 - 1, 0)
    return max(row_gap, col_gap) >= MIN_BBOX_GAP


def _rect_cells(box: Box) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _candidate(h: int, w: int, size: int, direction: int, gap: int, upper_r: int, upper_c: int) -> dict | None:
    upper = (upper_r, upper_c, upper_r + size - 1, upper_c + size - 1)
    delta = gap + 1
    if direction == 1:
        start = (upper[2], upper[3])
        end = (start[0] + delta, start[1] + delta)
        lower = (end[0], end[1], end[0] + size - 1, end[1] + size - 1)
    else:
        start = (upper[2], upper[1])
        end = (start[0] + delta, start[1] - delta)
        lower = (end[0], end[1] - size + 1, end[0] + size - 1, end[1])

    if min(*upper, *lower) < 0:
        return None
    if max(upper[2], lower[2]) >= h or max(upper[3], lower[3]) >= w:
        return None

    diag = [(start[0] + step, start[1] + direction * step) for step in range(1, gap + 1)]
    return {
        "upper": upper,
        "lower": lower,
        "diag": diag,
        "endpoints": (start, end),
        "size": size,
        "direction": direction,
        "gap": gap,
    }


def _valid_candidates(
    h: int,
    w: int,
    occupied: np.ndarray,
    boxes: list[Box],
    size: int,
    direction: int,
    gap: int,
) -> list[dict]:
    cands: list[dict] = []
    for upper_r in range(h - size + 1):
        for upper_c in range(w - size + 1):
            cand = _candidate(h, w, size, direction, gap, upper_r, upper_c)
            if cand is None:
                continue
            upper = cand["upper"]
            lower = cand["lower"]
            if any(not _bbox_gap_ok(box, prev) for box in (upper, lower) for prev in boxes):
                continue
            cells = _rect_cells(upper) + _rect_cells(lower) + cand["diag"]
            if any(bool(occupied[r, c]) for r, c in cells):
                continue
            cands.append(cand)
    random.shuffle(cands)
    return cands


def _plan_sizes(count: int) -> list[int]:
    sizes = [random.choice(RECT_SIZE_OPTIONS) for _ in range(count)]
    if count >= 2:
        sizes[random.randrange(count)] = 3
        sizes[random.randrange(count)] = 2
    return sizes


def _plan_directions(count: int) -> list[int]:
    directions = [random.choice((-1, 1)) for _ in range(count)]
    if count >= 2:
        directions[0] = -1
        directions[1] = 1
        random.shuffle(directions)
    return directions


def _trace_summary(traces: list[dict]) -> dict:
    return {
        "diag_blocks": sum(trace["stop_reason"] == "diagonal" for trace in traces),
        "earlier_ray_blocks": sum(trace["stop_reason"] == "earlier_ray" for trace in traces),
        "edge_stops": sum(trace["stop_reason"] == "edge" for trace in traces),
        "rect_blocks": sum(trace["stop_reason"] == "rectangle" for trace in traces),
        "longest_ray": max((len(trace["cells"]) for trace in traces), default=0),
        "nonempty_rays": sum(bool(trace["cells"]) for trace in traces),
    }


def _semantic_validation(inp: np.ndarray, planned_colors: list[int]) -> np.ndarray | None:
    try:
        out, pairs, traces = _simulate(inp, with_trace=True)
    except AssertionError:
        return None

    if sorted(pair["color"] for pair in pairs) != sorted(planned_colors):
        return None

    summary = _trace_summary(traces)
    if summary["diag_blocks"] < 1:
        return None
    if summary["earlier_ray_blocks"] < 1:
        return None
    if summary["longest_ray"] < 3:
        return None
    if summary["nonempty_rays"] < len(planned_colors) * 4:
        return None

    # Every color must contribute both a connector and at least one non-empty ray from each endpoint.
    per_color: dict[int, dict[str, object]] = {}
    for pair in pairs:
        per_color[int(pair["color"])] = {
            "diag_len": len(pair["diag"]),
            "endpoint_hits": {pair["endpoints"][0]: 0, pair["endpoints"][1]: 0},
        }
    for trace in traces:
        if trace["cells"]:
            per_color[trace["color"]]["endpoint_hits"][trace["endpoint"]] += 1
    for color in planned_colors:
        payload = per_color[int(color)]
        if payload["diag_len"] < 2:
            return None
        if min(payload["endpoint_hits"].values()) == 0:
            return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.choices(PAIR_COUNT_OPTIONS, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    colors = sorted(random.sample(PALETTE, count))
    sizes = _plan_sizes(count)
    directions = _plan_directions(count)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count)]
    order = list(range(count))
    order.sort(key=lambda idx: (sizes[idx], gaps[idx]), reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    boxes: list[Box] = []
    chosen: dict[int, dict] = {}

    for idx in order:
        cands = _valid_candidates(h, w, occupied, boxes, sizes[idx], directions[idx], gaps[idx])
        if not cands:
            return None

        pick = None
        for cand in cands[: min(40, len(cands))]:
            endpoints = cand["endpoints"]
            # Prefer placements that expose rows/cols likely to intersect with others later.
            endpoint_rows = {pt[0] for pt in endpoints}
            endpoint_cols = {pt[1] for pt in endpoints}
            spread = (max(endpoint_rows) - min(endpoint_rows)) + (max(endpoint_cols) - min(endpoint_cols))
            if spread >= cand["gap"]:
                pick = cand
                break
        if pick is None:
            pick = cands[0]

        chosen[idx] = pick
        for cell in _rect_cells(pick["upper"]) + _rect_cells(pick["lower"]) + pick["diag"]:
            occupied[cell] = True
        boxes.extend([pick["upper"], pick["lower"]])

    inp = np.zeros((h, w), dtype=int)
    for idx, color in enumerate(colors):
        pick = chosen[idx]
        r0, c0, r1, c1 = pick["upper"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = color
        r0, c0, r1, c1 = pick["lower"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = color

    return inp if _semantic_validation(inp, colors) is not None else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray blocking input")
