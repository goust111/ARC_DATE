from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
COLOR_PRIORITY = (2, 1, 3, 4, 6, 7, 8, 9)
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_PRIORITY)}

# A concrete visible 5-pixel arrow encoding for the four directions.
ARROW_TEMPLATES = {
    "R": {"cells": ((1, 0), (1, 1), (1, 2), (0, 2), (2, 2)), "tip": (1, 2), "dir": (0, 1)},
    "L": {"cells": ((1, 2), (1, 1), (1, 0), (0, 0), (2, 0)), "tip": (1, 0), "dir": (0, -1)},
    "U": {"cells": ((2, 1), (1, 1), (0, 1), (0, 0), (0, 2)), "tip": (0, 1), "dir": (-1, 0)},
    "D": {"cells": ((0, 1), (1, 1), (2, 1), (2, 0), (2, 2)), "tip": (2, 1), "dir": (1, 0)},
}
TEMPLATE_LOOKUP = {
    tuple(sorted(data["cells"])): direction for direction, data in ARROW_TEMPLATES.items()
}

RNG = np.random.default_rng()
GEN_RETRIES = 300
GRID_H_RANGE = (11, 16)
LEFT_W_RANGE = (12, 16)
LANE_STEP = 4
GAP_RANGE = (4, 7)
SCENARIOS = ("pair_tie", "chain_tie", "chain_majority", "double_pair")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for col in range(grid.shape[1]):
        column = grid[:, col]
        if np.all(column == column[0]) and int(column[0]) != bg:
            candidates.append((col, int(column[0])))
    if not candidates:
        raise ValueError("expected a full-height divider column")
    preferred = [item for item in candidates if item[1] == DIVIDER]
    return preferred[0] if preferred else candidates[0]


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _arrow_cells(arrow: dict[str, object]) -> tuple[tuple[int, int], ...]:
    r0, c0 = arrow["anchor"]
    direction = arrow["dir"]
    return tuple((int(r0) + dr, int(c0) + dc) for dr, dc in ARROW_TEMPLATES[direction]["cells"])


def _tip_cell(arrow: dict[str, object]) -> tuple[int, int]:
    r0, c0 = arrow["anchor"]
    dr, dc = ARROW_TEMPLATES[arrow["dir"]]["tip"]
    return (int(r0) + dr, int(c0) + dc)


def _detect_arrows(left: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows = []
    for comp in _components(left, bg):
        cells = comp["cells"]
        if len(cells) != 5:
            raise ValueError("every arrow must occupy exactly 5 pixels")
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        rel = tuple(sorted((r - r0, c - c0) for r, c in cells))
        direction = TEMPLATE_LOOKUP.get(rel)
        if direction is None:
            raise ValueError("unrecognized arrow template")
        arrows.append({"color": int(comp["color"]), "dir": direction, "anchor": [int(r0), int(c0)]})
    if not arrows:
        raise ValueError("expected at least one arrow")
    return arrows


def _can_shift(
    arrow: dict[str, object],
    steps: int,
    h: int,
    left_w: int,
    blocked: set[tuple[int, int]],
) -> bool:
    dr, dc = ARROW_TEMPLATES[arrow["dir"]]["dir"]
    for r, c in _arrow_cells(arrow):
        nr = r + dr * steps
        nc = c + dc * steps
        if not (0 <= nr < h and 0 <= nc < left_w):
            return False
        if (nr, nc) in blocked:
            return False
    return True


def _assert_disjoint(arrows: list[dict[str, object]]) -> None:
    occupied: set[tuple[int, int]] = set()
    for arrow in arrows:
        cells = _arrow_cells(arrow)
        if any(cell in occupied for cell in cells):
            raise ValueError("arrows overlapped during simulation")
        occupied.update(cells)


def _slide_arrows(arrows: list[dict[str, object]], h: int, left_w: int) -> list[dict[str, object]]:
    moving = [
        {"color": int(arrow["color"]), "dir": arrow["dir"], "anchor": [int(arrow["anchor"][0]), int(arrow["anchor"][1])]}
        for arrow in arrows
    ]
    settled: list[dict[str, object]] = []

    while moving:
        blocked: set[tuple[int, int]] = set()
        for arrow in settled:
            blocked.update(_arrow_cells(arrow))

        limits = []
        for arrow in moving:
            steps = 0
            while _can_shift(arrow, steps + 1, h, left_w, blocked):
                steps += 1
            limits.append(steps)

        dt = min(limits)
        for arrow in moving:
            dr, dc = ARROW_TEMPLATES[arrow["dir"]]["dir"]
            arrow["anchor"][0] += dr * dt
            arrow["anchor"][1] += dc * dt

        _assert_disjoint(moving + settled)

        next_moving = []
        for arrow, limit in zip(moving, limits):
            if limit == dt:
                settled.append(arrow)
            else:
                next_moving.append(arrow)
        moving = next_moving

    return settled


def _touches_by_tip(arrow_a: dict[str, object], arrow_b: dict[str, object]) -> bool:
    tip = _tip_cell(arrow_a)
    return any(abs(tip[0] - r) + abs(tip[1] - c) == 1 for r, c in _arrow_cells(arrow_b))


def _collision_components(arrows: list[dict[str, object]]) -> list[list[int]]:
    n = len(arrows)
    graph = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _touches_by_tip(arrows[i], arrows[j]) or _touches_by_tip(arrows[j], arrows[i]):
                graph[i].add(j)
                graph[j].add(i)

    seen = [False] * n
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in graph[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _merged_color(arrows: list[dict[str, object]], comp: list[int]) -> int:
    areas = Counter()
    for idx in comp:
        arrow = arrows[idx]
        areas[int(arrow["color"])] += len(ARROW_TEMPLATES[arrow["dir"]]["cells"])
    return min(
        areas,
        key=lambda color: (-areas[color], COLOR_RANK.get(color, 100 + color), color),
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    divider_col, divider_color = _find_divider(g, bg)
    left = g[:, :divider_col]

    arrows = _detect_arrows(left, bg)
    arrows = _slide_arrows(arrows, h, divider_col)

    out = np.full_like(g, bg)
    out[:, divider_col] = divider_color
    for arrow in arrows:
        for r, c in _arrow_cells(arrow):
            out[r, c] = int(arrow["color"])

    for comp in _collision_components(arrows):
        if len(comp) < 2:
            continue
        color = _merged_color(arrows, comp)
        merged_cells: set[tuple[int, int]] = set()
        for idx in comp:
            merged_cells.update(_arrow_cells(arrows[idx]))
        for r, c in merged_cells:
            out[r, w - 1 - c] = color

    return out


def _sample_unique_colors(count: int) -> list[int]:
    return [int(v) for v in RNG.choice(np.array(COLOR_PRIORITY, dtype=int), size=count, replace=False)]


def _sample_group_colors(size: int, mode: str) -> list[int]:
    if mode == "majority":
        dominant, other = _sample_unique_colors(2)
        colors = [dominant, dominant, other]
        RNG.shuffle(colors)
        return [int(v) for v in colors]
    if mode == "tie":
        return _sample_unique_colors(size)
    return [int(_sample_unique_colors(1)[0])]


def _sample_axis_anchors(size: int, axis_len: int, positive_dir: bool) -> list[int] | None:
    gaps = [int(RNG.integers(GAP_RANGE[0], GAP_RANGE[1])) for _ in range(max(0, size - 1))]
    total_gap = sum(gaps)
    if positive_dir:
        lo = total_gap
        hi = axis_len - 4
        if hi < lo:
            return None
        front = int(RNG.integers(lo, hi + 1))
        anchors = [front]
        pos = front
        for gap in reversed(gaps):
            pos -= gap
            anchors.append(pos)
        anchors.sort()
        return anchors

    lo = 1
    hi = axis_len - 3 - total_gap
    if hi < lo:
        return None
    front = int(RNG.integers(lo, hi + 1))
    anchors = [front]
    pos = front
    for gap in gaps:
        pos += gap
        anchors.append(pos)
    return anchors


def _scenario_specs() -> list[tuple[int, str]]:
    scenario = str(RNG.choice(np.array(SCENARIOS)))
    if scenario == "pair_tie":
        specs = [(2, "tie")]
        specs.extend((1, "solo") for _ in range(int(RNG.integers(1, 3))))
    elif scenario == "chain_tie":
        specs = [(3, "tie")]
        specs.extend((1, "solo") for _ in range(int(RNG.integers(0, 2))))
    elif scenario == "chain_majority":
        specs = [(3, "majority")]
        specs.extend((1, "solo") for _ in range(int(RNG.integers(0, 2))))
    else:
        specs = [(2, "tie"), (2, "tie")]
        specs.extend((1, "solo") for _ in range(int(RNG.integers(0, 2))))
    RNG.shuffle(specs)
    return [(int(size), mode) for size, mode in specs]


def _place_horizontal(h: int, left_w: int, specs: list[tuple[int, str]]) -> np.ndarray | None:
    lanes = list(range(0, h - 2, LANE_STEP))
    if len(lanes) < len(specs):
        return None
    chosen = [int(v) for v in RNG.choice(np.array(lanes, dtype=int), size=len(specs), replace=False)]
    chosen.sort()

    grid = np.full((h, left_w * 2 + 1), BG, dtype=int)
    grid[:, left_w] = DIVIDER

    for row, (size, color_mode) in zip(chosen, specs):
        direction = str(RNG.choice(np.array(["L", "R"])))
        anchors = _sample_axis_anchors(size, left_w, positive_dir=(direction == "R"))
        if anchors is None:
            return None
        colors = _sample_group_colors(size, color_mode)
        for anchor_col, color in zip(anchors, colors):
            for dr, dc in ARROW_TEMPLATES[direction]["cells"]:
                rr = row + dr
                cc = anchor_col + dc
                if grid[rr, cc] != BG:
                    return None
                grid[rr, cc] = color
    return grid


def _place_vertical(h: int, left_w: int, specs: list[tuple[int, str]]) -> np.ndarray | None:
    lanes = list(range(0, left_w - 2, LANE_STEP))
    if len(lanes) < len(specs):
        return None
    chosen = [int(v) for v in RNG.choice(np.array(lanes, dtype=int), size=len(specs), replace=False)]
    chosen.sort()

    grid = np.full((h, left_w * 2 + 1), BG, dtype=int)
    grid[:, left_w] = DIVIDER

    for col, (size, color_mode) in zip(chosen, specs):
        direction = str(RNG.choice(np.array(["U", "D"])))
        anchors = _sample_axis_anchors(size, h, positive_dir=(direction == "D"))
        if anchors is None:
            return None
        colors = _sample_group_colors(size, color_mode)
        for anchor_row, color in zip(anchors, colors):
            for dr, dc in ARROW_TEMPLATES[direction]["cells"]:
                rr = anchor_row + dr
                cc = col + dc
                if grid[rr, cc] != BG:
                    return None
                grid[rr, cc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    left_w = int(RNG.integers(LEFT_W_RANGE[0], LEFT_W_RANGE[1]))
    specs = _scenario_specs()
    mode = str(RNG.choice(np.array(["horizontal", "vertical"])))

    grid = _place_horizontal(h, left_w, specs) if mode == "horizontal" else _place_vertical(h, left_w, specs)
    if grid is None:
        return None

    out = main(grid)
    divider_col, _divider_color = _find_divider(grid, _get_bg(grid))
    if not np.any(out[:, divider_col + 1 :] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
