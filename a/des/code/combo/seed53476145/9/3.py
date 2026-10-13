from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PATH = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
COMPONENT_COUNT_RANGE = (1, 3)
MANDATORY_TURNS_RANGE = (3, 5)
EXTRA_TURNS_RANGE = (1, 4)
SEG_LEN_RANGE = (2, 6)
MARGIN = 1

MAX_COMPONENT_ATTEMPTS = 200
MAX_GENERATE_ATTEMPTS = 300

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


Pos = tuple[int, int]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(pos: Pos, h: int, w: int) -> list[Pos]:
    r, c = pos
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _inside_margin(pos: Pos, h: int, w: int) -> bool:
    r, c = pos
    return MARGIN <= r < h - MARGIN and MARGIN <= c < w - MARGIN


def _is_turn(pos: Pos, neighbors: list[Pos]) -> bool:
    if len(neighbors) != 2:
        return False
    (r1, c1), (r2, c2) = neighbors
    dr1, dc1 = r1 - pos[0], c1 - pos[1]
    dr2, dc2 = r2 - pos[0], c2 - pos[1]
    return not (dr1 == -dr2 and dc1 == -dc2)


def _edge_key(a: Pos, b: Pos) -> tuple[Pos, Pos]:
    return (a, b) if a < b else (b, a)


def _canonical_segment(path: list[Pos]) -> list[Pos]:
    rows = {r for r, _ in path}
    cols = {c for _, c in path}
    if len(rows) == 1:
        return sorted(path, key=lambda p: p[1])
    if len(cols) == 1:
        return sorted(path, key=lambda p: p[0])
    raise AssertionError("segment between nodes must be straight")


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    occupied = {tuple(map(int, pos)) for pos in np.argwhere(g != bg)}
    if not occupied:
        return {
            "bg": bg,
            "path_color": None,
            "neighbors": {},
            "turns": set(),
            "nodes": set(),
            "source_colors": {},
            "segments": [],
        }

    neighbors = {
        pos: [nb for nb in _neighbors4(pos, h, w) if nb in occupied]
        for pos in occupied
    }
    turns = {pos for pos, nbs in neighbors.items() if _is_turn(pos, nbs)}
    nodes = {pos for pos, nbs in neighbors.items() if len(nbs) != 2 or pos in turns}

    non_turn_colors = [int(g[pos]) for pos in occupied if pos not in turns]
    if non_turn_colors:
        path_color = Counter(non_turn_colors).most_common(1)[0][0]
    else:
        path_color = Counter(int(g[pos]) for pos in occupied).most_common(1)[0][0]

    source_colors = {}
    for pos in nodes:
        color = int(g[pos])
        source_colors[pos] = None if color in (bg, path_color) else color

    segments = []
    visited_edges: set[tuple[Pos, Pos]] = set()
    for node in sorted(nodes):
        for nb in sorted(neighbors[node]):
            edge = _edge_key(node, nb)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [node, nb]
            prev, curr = node, nb
            while curr not in nodes:
                nexts = [cand for cand in neighbors[curr] if cand != prev]
                if len(nexts) != 1:
                    break
                nxt = nexts[0]
                edge = _edge_key(curr, nxt)
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(nxt)
                prev, curr = curr, nxt

            ordered = _canonical_segment(path)
            orientation = "h" if len({r for r, _ in ordered}) == 1 else "v"
            segments.append(
                {
                    "cells": ordered,
                    "ends": (ordered[0], ordered[-1]),
                    "orientation": orientation,
                }
            )

    return {
        "bg": bg,
        "path_color": path_color,
        "neighbors": neighbors,
        "turns": turns,
        "nodes": nodes,
        "source_colors": source_colors,
        "segments": segments,
    }


def _segment_case(segment: dict[str, object], source_colors: dict[Pos, int | None]) -> str:
    a, b = segment["ends"]
    ca = source_colors.get(a)
    cb = source_colors.get(b)
    if ca is None and cb is None:
        return "none"
    if ca is None or cb is None:
        return "single"
    if ca == cb:
        return "same"
    return "different"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each straight path segment by propagating the colors of the turn markers
    at its ends. When the two end markers differ, the split point favors the left/top end.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = g.copy()

    for segment in info["segments"]:
        cells = segment["cells"]
        a, b = segment["ends"]
        ca = info["source_colors"].get(a)
        cb = info["source_colors"].get(b)

        if ca is None and cb is None:
            continue
        if ca is None:
            colors = [cb] * len(cells)
        elif cb is None or ca == cb:
            colors = [ca] * len(cells)
        else:
            split = (len(cells) + 1) // 2
            colors = [ca] * split + [cb] * (len(cells) - split)

        for (r, c), color in zip(cells, colors):
            out[r, c] = color

    return out


def _halo(cells: set[Pos], h: int, w: int) -> set[Pos]:
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _can_extend(
    end: Pos,
    direction: Pos,
    length: int,
    occupied: set[Pos],
    global_halo: set[Pos],
    h: int,
    w: int,
) -> list[Pos] | None:
    temp_occupied = set(occupied)
    curr = end
    new_cells = []

    for _ in range(length):
        nxt = (curr[0] + direction[0], curr[1] + direction[1])
        if not _inside_margin(nxt, h, w):
            return None
        if nxt in global_halo or nxt in temp_occupied:
            return None
        adjacent = [nb for nb in _neighbors4(nxt, h, w) if nb in temp_occupied]
        if any(nb != curr for nb in adjacent):
            return None
        new_cells.append(nxt)
        temp_occupied.add(nxt)
        curr = nxt

    return new_cells


def _has_future_turn_exit(
    end: Pos,
    prev_direction: Pos,
    occupied: set[Pos],
    global_halo: set[Pos],
    h: int,
    w: int,
) -> bool:
    for direction in DIRS4:
        if prev_direction is not None and (
            direction[0] == prev_direction[0] or direction[1] == prev_direction[1]
        ):
            continue
        if _can_extend(end, direction, SEG_LEN_RANGE[0], occupied, global_halo, h, w) is not None:
            return True
    return False


def _build_component(
    h: int,
    w: int,
    global_halo: set[Pos],
    turn_count: int,
    rng: random.Random,
) -> tuple[list[Pos], list[Pos]] | None:
    segment_count = turn_count + 1
    lengths = list(range(SEG_LEN_RANGE[0], SEG_LEN_RANGE[1] + 1))

    for _ in range(MAX_COMPONENT_ATTEMPTS):
        start = (rng.randint(MARGIN, h - MARGIN - 1), rng.randint(MARGIN, w - MARGIN - 1))
        if start in global_halo:
            continue

        path = [start]
        occupied = {start}
        vertices = [start]
        prev_direction = None
        ok = True

        for seg_idx in range(segment_count):
            directions = list(DIRS4)
            if prev_direction is not None:
                directions = [
                    direction
                    for direction in directions
                    if direction[0] != prev_direction[0] and direction[1] != prev_direction[1]
                ]
            rng.shuffle(directions)

            placed = False
            for direction in directions:
                sampled_lengths = lengths[:]
                rng.shuffle(sampled_lengths)
                for length in sampled_lengths:
                    new_cells = _can_extend(path[-1], direction, length, occupied, global_halo, h, w)
                    if new_cells is None:
                        continue
                    occupied_after = occupied | set(new_cells)
                    end = new_cells[-1]
                    if seg_idx < segment_count - 1 and not _has_future_turn_exit(
                        end, direction, occupied_after, global_halo, h, w
                    ):
                        continue
                    path.extend(new_cells)
                    occupied = occupied_after
                    vertices.append(end)
                    prev_direction = direction
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                ok = False
                break

        if ok:
            return path, vertices

    return None


def _mandatory_turn_colors(turn_count: int, rng: random.Random) -> list[int]:
    first, second = rng.sample(MARKER_COLORS, 2)
    remaining_palette = [color for color in MARKER_COLORS if color not in (first, second)]
    extra = rng.choice(remaining_palette)
    colors = [first, first, second]
    for _ in range(3, turn_count):
        colors.append(rng.choice((first, second, extra)))
    return colors


def _extra_turn_colors(turn_count: int, rng: random.Random) -> list[int]:
    palette_size = rng.randint(1, min(3, len(MARKER_COLORS)))
    palette = rng.sample(MARKER_COLORS, palette_size)
    return [rng.choice(palette) for _ in range(turn_count)]


def _paint_component(grid: np.ndarray, path: list[Pos], turns: list[Pos], colors: list[int]) -> None:
    for r, c in path:
        grid[r, c] = PATH
    for (r, c), color in zip(turns, colors):
        grid[r, c] = color


def _generated_ok(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    if info["path_color"] != PATH:
        return False
    if not info["turns"]:
        return False
    if any(info["source_colors"].get(pos) is None for pos in info["turns"]):
        return False

    cases = {_segment_case(segment, info["source_colors"]) for segment in info["segments"]}
    if not {"single", "same", "different"} <= cases:
        return False

    out = main(grid)
    if np.array_equal(grid, out):
        return False
    if np.sum(grid == PATH) < 6:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    component_count = rng.randint(*COMPONENT_COUNT_RANGE)
    turn_counts = [rng.randint(*MANDATORY_TURNS_RANGE)]
    for _ in range(component_count - 1):
        turn_counts.append(rng.randint(*EXTRA_TURNS_RANGE))

    global_occupied: set[Pos] = set()
    global_halo: set[Pos] = set()
    components: list[tuple[list[Pos], list[Pos]]] = []
    for turn_count in turn_counts:
        built = _build_component(h, w, global_halo, turn_count, rng)
        if built is None:
            return None
        path, vertices = built
        cells = set(path)
        global_occupied |= cells
        global_halo |= _halo(cells, h, w)
        components.append((path, vertices))

    for idx, (path, vertices) in enumerate(components):
        turns = vertices[1:-1]
        if idx == 0:
            colors = _mandatory_turn_colors(len(turns), rng)
        else:
            colors = _extra_turn_colors(len(turns), rng)
        _paint_component(grid, path, turns, colors)

    if not _generated_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
