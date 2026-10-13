from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SEPARATOR = 5
ORIGIN_MARK = 9
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_FROM_DELTA = {v: k for k, v in DIRS.items()}

GEN_ATTEMPTS = 400
COMPONENT_ATTEMPTS = 250
PATH_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_from(
    grid: np.ndarray, start: tuple[int, int], allowed: callable
) -> tuple[int, set[tuple[int, int]]]:
    q = deque([start])
    color = int(grid[start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in seen:
                if allowed(nr, nc, color):
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return color, seen


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        value = int(col[0])
        if value != bg and bool((col == value).all()):
            candidates.append((c, value))
    assert len(candidates) == 1, "expected one solid separator column"
    return candidates[0]


def _find_origin(grid: np.ndarray, divider: int, bg: int, sep_color: int) -> tuple[int, int, int]:
    pts = []
    h, w = grid.shape
    for r in range(h):
        for c in range(divider + 1, w):
            v = int(grid[r, c])
            if v not in (bg, sep_color):
                pts.append((r, c, v))
    assert len(pts) == 1, "expected one origin marker in the canvas"
    return pts[0]


def _block_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _scan_blocks(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    blocks = []
    for r, c in cells:
        if _block_cells((r, c)).issubset(cells):
            blocks.append((r, c))
    return sorted(blocks)


def _trace_simple_path(path_cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [p for p in _neighbors4(*cur) if p in path_cells and p != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(path_cells), "path cells must be fully connected"
    return path


def _parse_instruction_cells(cells: set[tuple[int, int]]) -> dict:
    cells = set(cells)
    block_tops = _scan_blocks(cells)
    candidates = []
    for i in range(len(block_tops)):
        for j in range(i + 1, len(block_tops)):
            b0 = block_tops[i]
            b1 = block_tops[j]
            bc0 = _block_cells(b0)
            bc1 = _block_cells(b1)
            if bc0 & bc1:
                continue

            path_cells = cells - bc0 - bc1
            if len(path_cells) < 2:
                continue

            deg = {
                cell: sum((nb in path_cells) for nb in _neighbors4(*cell))
                for cell in path_cells
            }
            endpoints = [cell for cell, d in deg.items() if d == 1]
            if len(endpoints) != 2 or any(d not in (1, 2) for d in deg.values()):
                continue

            block_sets = [bc0, bc1]
            endpoint_touch: dict[tuple[int, int], int] = {}
            ok = True
            for cell in path_cells:
                touch = {
                    idx
                    for idx, block in enumerate(block_sets)
                    if any(nb in block for nb in _neighbors4(*cell))
                }
                if cell in endpoints:
                    if len(touch) != 1:
                        ok = False
                        break
                    endpoint_touch[cell] = next(iter(touch))
                elif touch:
                    ok = False
                    break
            if not ok:
                continue
            if set(endpoint_touch.values()) != {0, 1}:
                continue

            start_block = min(b0, b1)
            end_block = max(b0, b1)
            start_idx = 0 if start_block == b0 else 1
            start_endpoint = next(cell for cell, idx in endpoint_touch.items() if idx == start_idx)
            traced = _trace_simple_path(path_cells, start_endpoint)
            end_idx = 1 - start_idx
            if endpoint_touch[traced[-1]] != end_idx:
                continue

            dirs = []
            for a, b in zip(traced, traced[1:]):
                dr = int(b[0] - a[0])
                dc = int(b[1] - a[1])
                dirs.append(DIR_FROM_DELTA[(dr, dc)])
            candidates.append(
                {
                    "start_block": start_block,
                    "end_block": end_block,
                    "dirs": dirs,
                    "total_pixels": len(cells),
                    "path_cells": traced,
                }
            )

    assert len(candidates) == 1, "component must contain one unambiguous instruction"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _get_bg(g)
    divider, sep_color = _find_separator(g, bg)
    origin_r, origin_c, origin_color = _find_origin(g, divider, bg, sep_color)

    seen: set[tuple[int, int]] = set()
    instructions = []
    for r in range(h):
        for c in range(divider):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v in (bg, sep_color):
                continue
            color, comp = _component_from(
                g,
                (r, c),
                lambda nr, nc, color=v: nc < divider and int(g[nr, nc]) == color,
            )
            seen |= comp
            parsed = _parse_instruction_cells(comp)
            instructions.append({"color": color, **parsed})

    assert instructions, "expected at least one instruction path"
    sizes = [item["total_pixels"] for item in instructions]
    assert len(set(sizes)) == len(sizes), "instruction sizes must be unique"
    instructions.sort(key=lambda item: item["total_pixels"])

    out = g.copy()
    out[:, divider + 1 :] = bg
    out[origin_r, origin_c] = origin_color

    r, c = origin_r, origin_c
    for item in instructions:
        for d in item["dirs"]:
            dr, dc = DIRS[d]
            r += dr
            c += dc
            assert 0 <= r < h and divider + 1 <= c < w, "drawing left the canvas"
            out[r, c] = int(item["color"])

    out[origin_r, origin_c] = origin_color
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _has_turn(path_cells: list[tuple[int, int]]) -> bool:
    if len(path_cells) < 3:
        return False
    dirs = []
    for a, b in zip(path_cells, path_cells[1:]):
        dirs.append((b[0] - a[0], b[1] - a[1]))
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _build_random_path(num_cells: int, need_turn: bool) -> list[tuple[int, int]] | None:
    max_span = max(5, num_cells + 1)
    for _ in range(PATH_ATTEMPTS):
        path = [(0, 0)]
        used = {(0, 0)}

        def dfs() -> bool:
            if len(path) == num_cells:
                return True
            r, c = path[-1]
            nbrs = []
            dirs = list(DIRS.values())
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) in used:
                    continue
                if sum((rr, cc) in used for rr, cc in _neighbors4(nr, nc)) != 1:
                    continue
                rows = [p[0] for p in path] + [nr]
                cols = [p[1] for p in path] + [nc]
                if max(rows) - min(rows) + 1 > max_span or max(cols) - min(cols) + 1 > max_span:
                    continue
                nbrs.append((nr, nc))
            random.shuffle(nbrs)
            for nr, nc in nbrs:
                used.add((nr, nc))
                path.append((nr, nc))
                if dfs():
                    return True
                path.pop()
                used.remove((nr, nc))
            return False

        if dfs() and (not need_turn or _has_turn(path)):
            return path
    return None


def _candidate_block_tops(endpoint: tuple[int, int], path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = endpoint
    tops = []
    for r0 in range(r - 2, r + 2):
        for c0 in range(c - 2, c + 2):
            block = _block_cells((r0, c0))
            if block & path_cells:
                continue
            touch = [cell for cell in block if abs(cell[0] - r) + abs(cell[1] - c) == 1]
            if len(touch) != 1:
                continue
            tops.append((r0, c0))
    random.shuffle(tops)
    return tops


def _build_instruction_component(num_path_cells: int, need_turn: bool) -> dict | None:
    for _ in range(COMPONENT_ATTEMPTS):
        path = _build_random_path(num_path_cells, need_turn=need_turn)
        if path is None:
            continue
        path_set = set(path)
        starts = _candidate_block_tops(path[0], path_set)
        ends = _candidate_block_tops(path[-1], path_set)
        for b0 in starts:
            for b1 in ends:
                if _block_cells(b0) & _block_cells(b1):
                    continue
                cells = path_set | _block_cells(b0) | _block_cells(b1)
                try:
                    parsed = _parse_instruction_cells(cells)
                except AssertionError:
                    continue
                norm_cells = _normalize(cells)
                parsed_norm = _parse_instruction_cells(norm_cells)
                return {
                    "cells": norm_cells,
                    "dirs": parsed_norm["dirs"],
                    "total_pixels": parsed_norm["total_pixels"],
                    "bbox": _bbox(norm_cells),
                    "path_turns": _has_turn(parsed_norm["path_cells"]),
                }
    return None


def _simulate_drawing(instructions: list[dict]) -> list[tuple[int, int]] | None:
    pos = (0, 0)
    visited = [pos]
    painted: set[tuple[int, int]] = set()
    for item in sorted(instructions, key=lambda x: x["total_pixels"]):
        for d in item["dirs"]:
            dr, dc = DIRS[d]
            pos = (pos[0] + dr, pos[1] + dc)
            if pos == (0, 0) or pos in painted:
                return None
            painted.add(pos)
            visited.append(pos)
    return visited


def _place_components(instructions: list[dict], height: int, left_width: int) -> dict[int, set[tuple[int, int]]] | None:
    blocked: set[tuple[int, int]] = set()
    placed: dict[int, set[tuple[int, int]]] = {}
    order = list(range(len(instructions)))
    random.shuffle(order)
    for idx in order:
        cells = instructions[idx]["cells"]
        box_h, box_w = instructions[idx]["bbox"]
        ok_here = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if height - box_h - 2 < 1 or left_width - box_w - 2 < 1:
                return None
            r0 = random.randint(1, height - box_h - 2)
            c0 = random.randint(1, left_width - box_w - 2)
            shifted = {(r + r0, c + c0) for r, c in cells}
            if any(c >= left_width for _, c in shifted):
                continue
            if any(cell in blocked for cell in shifted):
                continue
            placed[idx] = shifted
            for r, c in shifted:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < height and 0 <= cc < left_width:
                            blocked.add((rr, cc))
            ok_here = True
            break
        if not ok_here:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    n_instr = random.randint(2, 4)
    num_path_cells = random.sample(range(3, 10), n_instr)
    need_turn_flags = [random.random() < 0.6 for _ in range(n_instr)]
    if not any(need_turn_flags):
        need_turn_flags[random.randrange(n_instr)] = True

    instructions = []
    for n_cells, need_turn in zip(num_path_cells, need_turn_flags):
        comp = _build_instruction_component(n_cells, need_turn=need_turn)
        if comp is None:
            return None
        instructions.append(comp)

    totals = [item["total_pixels"] for item in instructions]
    if len(set(totals)) != len(totals):
        return None

    trace = _simulate_drawing(instructions)
    if trace is None:
        return None

    rows = [r for r, _ in trace]
    cols = [c for _, c in trace]
    pad_r = random.randint(1, 3)
    pad_c = random.randint(1, 3)
    min_canvas_h = 2 * max(-min(rows), max(rows)) + 1 + 2 * pad_r
    min_canvas_w = 2 * max(-min(cols), max(cols)) + 1 + 2 * pad_c

    tallest = max(item["bbox"][0] for item in instructions)
    min_h = max(min_canvas_h, tallest + 4, 13)
    if min_h > 29:
        return None
    height = random.randint(min_h, min(min_h + 8, 29))
    if height % 2 == 0:
        height += 1
    if height > 29:
        return None

    half_h = height // 2
    if any(not (0 <= half_h + r < height) for r in rows):
        return None

    right_w_min = max(min_canvas_w, 11)
    if right_w_min > 29:
        return None
    right_w = random.randint(right_w_min, min(right_w_min + 8, 29))
    if right_w % 2 == 0:
        right_w += 1
    if right_w > 29:
        return None
    half_w = right_w // 2
    if any(not (0 <= half_w + c < right_w) for c in cols):
        return None

    left_min = max(sum(item["bbox"][1] for item in instructions) // 2 + 6, max(item["bbox"][1] for item in instructions) + 4)
    left_width = random.randint(left_min, min(left_min + 10, 24))
    if left_width < 8:
        return None

    placed = _place_components(instructions, height, left_width)
    if placed is None:
        return None

    width = left_width + 1 + right_w
    if width > 30:
        return None

    color_pool = random.sample(PATH_COLORS, n_instr)
    random.shuffle(color_pool)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_width] = SEPARATOR

    for idx, cells in placed.items():
        color = color_pool[idx]
        instructions[idx]["color"] = color
        for r, c in cells:
            grid[r, c] = color

    origin = (height // 2, left_width + 1 + right_w // 2)
    grid[origin] = ORIGIN_MARK

    out = main(grid)
    if int(out[origin]) != ORIGIN_MARK:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
