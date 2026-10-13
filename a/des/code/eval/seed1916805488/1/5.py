from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_dominoes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen: set[tuple[int, int]] = set()
    dominoes: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BG or (r, c) in seen:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

            assert len(comp) == 2, "each non-background component must be a domino"
            rows = sorted({rr for rr, _ in comp})
            cols = sorted({cc for _, cc in comp})
            if len(rows) == 1 and len(cols) == 2 and cols[1] == cols[0] + 1:
                orientation = "h"
                ordered = sorted(comp)
            elif len(cols) == 1 and len(rows) == 2 and rows[1] == rows[0] + 1:
                orientation = "v"
                ordered = sorted(comp)
            else:
                raise AssertionError("component must be a 1x2 or 2x1 domino")

            dominoes.append({"color": color, "cells": ordered, "orientation": orientation})

    return dominoes


def _growth_analysis(grid: np.ndarray, dominoes: list[dict]) -> tuple[dict[int, int], dict[tuple[int, int], set[int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    color_counts = Counter(int(d["color"]) for d in dominoes)
    claims: dict[tuple[int, int], set[int]] = {}
    stats: list[dict] = []

    for domino in dominoes:
        color = int(domino["color"])
        grow_len = int(color_counts[color])
        blocked_by_edge = False
        blocked_by_object = False
        added = 0

        if domino["orientation"] == "h":
            row = int(domino["cells"][0][0])
            left_c = min(c for _, c in domino["cells"])
            right_c = max(c for _, c in domino["cells"])
            rays = [((0, -1), (row, left_c - 1)), ((0, 1), (row, right_c + 1))]
        else:
            col = int(domino["cells"][0][1])
            top_r = min(r for r, _ in domino["cells"])
            bot_r = max(r for r, _ in domino["cells"])
            rays = [((-1, 0), (top_r - 1, col)), ((1, 0), (bot_r + 1, col))]

        for (dr, dc), (start_r, start_c) in rays:
            rr, cc = start_r, start_c
            for _ in range(grow_len):
                if not (0 <= rr < h and 0 <= cc < w):
                    blocked_by_edge = True
                    break
                if int(g[rr, cc]) != BG:
                    blocked_by_object = True
                    break
                claims.setdefault((rr, cc), set()).add(color)
                added += 1
                rr += dr
                cc += dc

        stats.append(
            {
                "color": color,
                "orientation": domino["orientation"],
                "growth_len": grow_len,
                "blocked_by_edge": blocked_by_edge,
                "blocked_by_object": blocked_by_object,
                "added": added,
            }
        )

    return dict(color_counts), claims, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each same-color domino along its axis by the number of dominoes of that color."""
    g = np.asarray(grid, dtype=int)
    dominoes = _find_dominoes(g)
    color_counts, _claims, _stats = _growth_analysis(g, dominoes)
    out = g.copy()
    h, w = out.shape

    for domino in dominoes:
        color = int(domino["color"])
        grow_len = int(color_counts[color])

        if domino["orientation"] == "h":
            row = int(domino["cells"][0][0])
            left_c = min(c for _, c in domino["cells"])
            right_c = max(c for _, c in domino["cells"])
            rays = [((0, -1), (row, left_c - 1)), ((0, 1), (row, right_c + 1))]
        else:
            col = int(domino["cells"][0][1])
            top_r = min(r for r, _ in domino["cells"])
            bot_r = max(r for r, _ in domino["cells"])
            rays = [((-1, 0), (top_r - 1, col)), ((1, 0), (bot_r + 1, col))]

        for (dr, dc), (rr, cc) in rays:
            for _ in range(grow_len):
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(g[rr, cc]) != BG:
                    break
                out[rr, cc] = color
                rr += dr
                cc += dc

    return out


def _valid_placement(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied and (nr, nc) not in cell_set:
                    return False
    return True


def _cells_for_domino(orientation: str, anchor_r: int, anchor_c: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(anchor_r, anchor_c), (anchor_r, anchor_c + 1)]
    return [(anchor_r, anchor_c), (anchor_r + 1, anchor_c)]


def _candidate_boundary_positions(h: int, w: int, orientation: str, grow_len: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    if orientation == "h":
        for r in range(h):
            left_max = min(grow_len - 1, w - 2)
            for c in range(left_max + 1):
                candidates.append((r, c))
            start = max(0, w - grow_len - 1)
            for c in range(start, w - 1):
                if w - (c + 2) < grow_len:
                    candidates.append((r, c))
    else:
        for c in range(w):
            top_max = min(grow_len - 1, h - 2)
            for r in range(top_max + 1):
                candidates.append((r, c))
            start = max(0, h - grow_len - 1)
            for r in range(start, h - 1):
                if h - (r + 2) < grow_len:
                    candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_boundary_domino(spec: dict, occupied: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    for r, c in _candidate_boundary_positions(h, w, spec["orientation"], spec["grow_len"]):
        cells = _cells_for_domino(spec["orientation"], r, c)
        if _valid_placement(cells, occupied, h, w):
            return cells
    return None


def _place_obstacle_pair(src: dict, blocker: dict, occupied: set[tuple[int, int]], h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    grow_len = int(src["grow_len"])
    if grow_len < 2:
        return None

    orientation = src["orientation"]
    sides = ["positive", "negative"]
    random.shuffle(sides)

    if orientation == "h":
        rows = list(range(h))
        cols = list(range(w - 1))
        random.shuffle(rows)
        random.shuffle(cols)
        for r in rows:
            for c in cols:
                src_cells = _cells_for_domino("h", r, c)
                if not _valid_placement(src_cells, occupied, h, w):
                    continue
                right = max(cc for _, cc in src_cells)
                left = min(cc for _, cc in src_cells)
                for side in sides:
                    gaps = list(range(1, grow_len))
                    random.shuffle(gaps)
                    for gap in gaps:
                        if side == "positive":
                            block_c = right + gap + 1
                        else:
                            block_c = left - gap - 1
                        block_cells = _cells_for_domino("h", r, block_c)
                        occupied_with_src = occupied | set(src_cells)
                        if _valid_placement(block_cells, occupied_with_src, h, w):
                            return src_cells, block_cells
    else:
        rows = list(range(h - 1))
        cols = list(range(w))
        random.shuffle(rows)
        random.shuffle(cols)
        for c in cols:
            for r in rows:
                src_cells = _cells_for_domino("v", r, c)
                if not _valid_placement(src_cells, occupied, h, w):
                    continue
                top = min(rr for rr, _ in src_cells)
                bottom = max(rr for rr, _ in src_cells)
                for side in sides:
                    gaps = list(range(1, grow_len))
                    random.shuffle(gaps)
                    for gap in gaps:
                        if side == "positive":
                            block_r = bottom + gap + 1
                        else:
                            block_r = top - gap - 1
                        block_cells = _cells_for_domino("v", block_r, c)
                        occupied_with_src = occupied | set(src_cells)
                        if _valid_placement(block_cells, occupied_with_src, h, w):
                            return src_cells, block_cells

    return None


def _place_random_domino(spec: dict, occupied: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    if spec["orientation"] == "h":
        for r in range(h):
            for c in range(w - 1):
                candidates.append((r, c))
    else:
        for r in range(h - 1):
            for c in range(w):
                candidates.append((r, c))
    random.shuffle(candidates)

    for r, c in candidates:
        cells = _cells_for_domino(spec["orientation"], r, c)
        if _valid_placement(cells, occupied, h, w):
            return cells
    return None


def _gen_attempt(scenario: str) -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    num_colors = random.randint(2, 4)
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], num_colors)

    counts = [random.randint(1, 3) for _ in range(num_colors)]
    if max(counts) == 1:
        counts[random.randrange(num_colors)] = random.randint(2, 3)
    total_dominoes = sum(counts)
    if total_dominoes < 4 or total_dominoes > 8:
        return None

    specs: list[dict] = []
    for color, count in zip(palette, counts):
        for _ in range(count):
            specs.append({"color": int(color), "grow_len": int(count), "orientation": random.choice(["h", "v"])})

    if all(spec["orientation"] == "h" for spec in specs):
        specs[random.randrange(len(specs))]["orientation"] = "v"
    if all(spec["orientation"] == "v" for spec in specs):
        specs[random.randrange(len(specs))]["orientation"] = "h"

    need_boundary = scenario in {"boundary", "both"}
    need_obstacle = scenario in {"obstacle", "both"}

    boundary_idx = None
    obstacle_src_idx = None
    obstacle_blocker_idx = None

    if need_boundary:
        boundary_idx = random.randrange(len(specs))

    if need_obstacle:
        candidates = [i for i, spec in enumerate(specs) if spec["grow_len"] >= 2 and i != boundary_idx]
        if not candidates:
            return None
        obstacle_src_idx = random.choice(candidates)
        blockers = [i for i in range(len(specs)) if i not in {boundary_idx, obstacle_src_idx}]
        if not blockers:
            return None
        obstacle_blocker_idx = random.choice(blockers)
        specs[obstacle_blocker_idx]["orientation"] = specs[obstacle_src_idx]["orientation"]

    occupied: set[tuple[int, int]] = set()
    placements: dict[int, list[tuple[int, int]]] = {}

    if obstacle_src_idx is not None and obstacle_blocker_idx is not None:
        pair = _place_obstacle_pair(specs[obstacle_src_idx], specs[obstacle_blocker_idx], occupied, h, w)
        if pair is None:
            return None
        src_cells, block_cells = pair
        placements[obstacle_src_idx] = src_cells
        placements[obstacle_blocker_idx] = block_cells
        occupied.update(src_cells)
        occupied.update(block_cells)

    if boundary_idx is not None and boundary_idx not in placements:
        cells = _place_boundary_domino(specs[boundary_idx], occupied, h, w)
        if cells is None:
            return None
        placements[boundary_idx] = cells
        occupied.update(cells)

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        if idx in placements:
            continue
        cells = _place_random_domino(specs[idx], occupied, h, w)
        if cells is None:
            return None
        placements[idx] = cells
        occupied.update(cells)

    grid = np.zeros((h, w), dtype=int)
    for idx, spec in enumerate(specs):
        for r, c in placements[idx]:
            grid[r, c] = int(spec["color"])

    dominoes = _find_dominoes(grid)
    if len(dominoes) != len(specs):
        return None

    _, claims, stats = _growth_analysis(grid, dominoes)
    if not claims:
        return None
    if any(len(colors) > 1 for colors in claims.values()):
        return None
    if len({stat["orientation"] for stat in stats}) < 2:
        return None
    if max(Counter(int(d["color"]) for d in dominoes).values()) < 2:
        return None
    has_edge = any(stat["blocked_by_edge"] for stat in stats)
    has_object = any(stat["blocked_by_object"] for stat in stats)
    if need_boundary and not has_edge:
        return None
    if need_obstacle and not has_object:
        return None
    if not need_boundary and has_edge:
        return None
    if not need_obstacle and has_object:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    scenario = random.choice(["boundary", "obstacle", "both"])
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
