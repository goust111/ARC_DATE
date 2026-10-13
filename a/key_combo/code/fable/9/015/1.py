import random
from collections import Counter, deque
from typing import Dict, List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]
DIRS: Sequence[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PALETTE: Sequence[int] = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 600


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(grid: np.ndarray, bg: int) -> List[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: List[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: List[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _touching_edges(shape: Tuple[int, int], cell: Coord) -> List[str]:
    h, w = shape
    r, c = cell
    edges: List[str] = []
    if r == 0:
        edges.append("top")
    if r == h - 1:
        edges.append("bottom")
    if c == 0:
        edges.append("left")
    if c == w - 1:
        edges.append("right")
    return edges


def _initial_dir(shape: Tuple[int, int], start: Coord) -> int:
    edges = _touching_edges(shape, start)
    assert len(edges) == 1, "start marker must touch exactly one edge"
    edge = edges[0]
    if edge == "top":
        return 2
    if edge == "bottom":
        return 0
    if edge == "left":
        return 1
    return 3


def _turn(dir_idx: int, go_left: bool) -> int:
    return (dir_idx - 1) % 4 if go_left else (dir_idx + 1) % 4


def _extract_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    comps = _components(g, bg)
    assert comps, "expected legend bars and start markers"

    legend = []
    starts: Dict[int, Coord] = {}
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(comp["area"]) == 1:
            cell = coords[0]
            edges = _touching_edges((h, w), cell)
            assert len(edges) == 1, "start marker must not sit on a corner"
            assert color not in starts, "generator guarantees one start per legend color"
            starts[color] = cell
            continue

        assert r0 == r1, "legend bar must be horizontal"
        assert c0 >= 1 and r0 >= 1, "legend bars stay inside the top-left region"
        assert int(comp["area"]) == c1 - c0 + 1, "legend bar must be a solid 1xN run"
        legend.append({"row": r0, "color": color, "length": int(comp["area"]), "cells": coords})

    legend.sort(key=lambda item: int(item["row"]))
    assert len(legend) >= 2, "need at least two legend colors"
    assert len(legend) == len(starts), "every legend color must have exactly one start marker"

    legend_colors = [int(item["color"]) for item in legend]
    legend_lengths = [int(item["length"]) for item in legend]
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be unique"
    assert len(set(legend_lengths)) == len(legend_lengths), "legend lengths must be unique"
    assert set(legend_colors) == set(starts), "start colors must match the legend"
    return {"grid": g, "bg": bg, "shape": (h, w), "legend": legend, "starts": starts}


def _run_particle(
    canvas: np.ndarray,
    *,
    start: Coord,
    color: int,
    step_len: int,
    bg: int,
    initial_starts: set[Coord],
) -> dict:
    r, c = map(int, start)
    dir_idx = _initial_dir(canvas.shape, start)
    progress = 1
    go_left = True
    turns = 0
    path = [(r, c)]
    stop_reason = None
    blocker = None
    stop_cell = None

    while True:
        if progress == step_len:
            dir_idx = _turn(dir_idx, go_left)
            go_left = not go_left
            progress = 0
            turns += 1

        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not _in_bounds(canvas.shape, nr, nc):
            stop_reason = "boundary"
            stop_cell = (nr, nc)
            break
        if int(canvas[nr, nc]) != bg:
            stop_reason = "occupied"
            stop_cell = (nr, nc)
            blocker = "start" if (nr, nc) in initial_starts and (nr, nc) != start else "trail"
            break

        canvas[nr, nc] = color
        r, c = nr, nc
        path.append((r, c))
        progress += 1

    return {
        "path": path,
        "path_set": set(path),
        "turns": turns,
        "stop_reason": stop_reason,
        "blocker": blocker,
        "stop_cell": stop_cell,
    }


def _simulate_actual(layout: dict) -> Tuple[np.ndarray, Dict[int, dict]]:
    g = np.array(layout["grid"], dtype=int, copy=True)
    bg = int(layout["bg"])
    for item in layout["legend"]:
        for r, c in item["cells"]:
            g[r, c] = bg

    histories: Dict[int, dict] = {}
    initial_starts = set(layout["starts"].values())
    for item in layout["legend"]:
        color = int(item["color"])
        start = layout["starts"][color]
        g[start] = color
        histories[color] = _run_particle(
            g,
            start=start,
            color=color,
            step_len=int(item["length"]),
            bg=bg,
            initial_starts=initial_starts,
        )
    return g, histories


def _simulate_solo(shape: Tuple[int, int], *, start: Coord, color: int, step_len: int, bg: int) -> dict:
    canvas = np.full(shape, bg, dtype=int)
    canvas[start] = color
    return _run_particle(canvas, start=start, color=color, step_len=step_len, bg=bg, initial_starts={start})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each boundary marker as a staircase particle whose step size comes from its same-colored legend bar.

    Input structure:
    - Black background with a stack of horizontal monochrome legend bars in the top-left.
    - Each legend bar has a unique color and a unique length.
    - The grid boundary also contains isolated single-pixel start markers whose colors match the legend bars.

    Rule:
    - Erase the legend bars, keeping only the black canvas and the boundary start markers.
    - Process colors in legend order from top to bottom.
    - For a legend length N, its particle paints one cell per step, counting the start marker as step 1, turns after every N painted cells, turns left first, then alternates left/right forever.
    - If the next cell would leave the grid or hit any already-colored cell, that particle stops immediately; earlier particles remain as obstacles for later ones.

    Key insight:
    - The legend bar length is the staircase side length, and the legend's vertical order defines which staircase gets to occupy crossings first.
    """
    layout = _extract_layout(np.asarray(grid, dtype=int))
    out, _ = _simulate_actual(layout)
    return out


def _random_primary_pair(
    h: int,
    w: int,
    legend_width: int,
    min_side_row: int,
) -> Tuple[Tuple[str, str], Dict[str, Coord]] | None:
    pair = random.choice(
        [
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
        ]
    )

    for _ in range(80):
        if pair == ("top", "left"):
            pr = random.randint(max(3, min_side_row // 2), max(4, h // 2 - 2))
            pc_low = max(legend_width + 5, 4)
            pc_high = min(w - 4, h - 2 - pr)
            if pc_low > pc_high:
                continue
            pc = random.randint(pc_low, pc_high)
            top_c = pc - pr
            left_r = pc + pr
            if legend_width + 3 <= top_c <= w - 2 and min_side_row <= left_r <= h - 2:
                return pair, {"top": (0, top_c), "left": (left_r, 0)}

        if pair == ("top", "right"):
            pr = random.randint(3, max(4, h // 2 - 1))
            pc_low = max(legend_width + 5, w - pr)
            pc_high = w - 4
            if pc_low > pc_high:
                continue
            pc = random.randint(pc_low, pc_high)
            top_c = pc - pr
            right_r = pc + pr - (w - 1)
            if legend_width + 3 <= top_c <= w - 2 and min_side_row <= right_r <= h - 2:
                return pair, {"top": (0, top_c), "right": (right_r, w - 1)}

        if pair == ("bottom", "left"):
            pr = random.randint(max(min_side_row - 2, h // 2), h - 4)
            pc_low = max(3, pr - (h - 2) + 1)
            pc_high = min(w - 4, h - 2 - pr)
            if pc_low > pc_high:
                continue
            pc = random.randint(pc_low, pc_high)
            bottom_c = pc - pr + (h - 1)
            left_r = pc + pr
            if 1 <= bottom_c <= w - 2 and min_side_row <= left_r <= h - 2:
                return pair, {"bottom": (h - 1, bottom_c), "left": (left_r, 0)}

        if pair == ("bottom", "right"):
            pr = random.randint(max(h // 2, 5), h - 4)
            pc_low = max(w - pr, 3)
            pc_high = w - 4
            if pc_low > pc_high:
                continue
            pc = random.randint(pc_low, pc_high)
            bottom_c = pc - pr + (h - 1)
            right_r = pc + pr - (w - 1)
            if 1 <= bottom_c <= w - 2 and min_side_row <= right_r <= h - 2:
                return pair, {"bottom": (h - 1, bottom_c), "right": (right_r, w - 1)}
    return None


def _random_extra_position(
    edge: str,
    h: int,
    w: int,
    legend_width: int,
    min_side_row: int,
    used: set[Coord],
) -> Coord | None:
    for _ in range(80):
        if edge == "top":
            cell = (0, random.randint(legend_width + 4, w - 2))
        elif edge == "bottom":
            cell = (h - 1, random.randint(1, w - 2))
        elif edge == "left":
            cell = (random.randint(min_side_row, h - 2), 0)
        else:
            cell = (random.randint(min_side_row, h - 2), w - 1)
        if cell not in used:
            return cell
    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    n_colors = random.choices([2, 3, 4], weights=[1, 4, 5])[0]
    lengths = random.sample(range(2, min(8, w - 5)), n_colors)
    colors = random.sample(list(PALETTE), n_colors)
    legend_width = max(lengths) + 1
    min_side_row = 2 * n_colors + 3
    if h - 2 < min_side_row or w <= legend_width + 5:
        return None

    primary = _random_primary_pair(h, w, legend_width, min_side_row)
    if primary is None:
        return None
    _pair, seeded_positions = primary

    edges = list(seeded_positions)
    remaining_edges = [edge for edge in ("top", "left", "bottom", "right") if edge not in edges]
    random.shuffle(remaining_edges)
    while len(edges) < n_colors:
        edges.append(remaining_edges.pop())

    used = set(seeded_positions.values())
    starts: List[Coord] = []
    edge_to_cell = dict(seeded_positions)
    for edge in edges:
        if edge in edge_to_cell:
            starts.append(edge_to_cell[edge])
            continue
        cell = _random_extra_position(edge, h, w, legend_width, min_side_row, used)
        if cell is None:
            return None
        used.add(cell)
        edge_to_cell[edge] = cell
        starts.append(cell)

    g = np.zeros((h, w), dtype=int)
    legend = list(zip(colors, lengths, starts))
    random.shuffle(legend)
    for idx, (color, length, start) in enumerate(legend):
        row = 1 + 2 * idx
        g[row, 1 : 1 + length] = color
        g[start] = color
    return g


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        layout = _extract_layout(grid)
        out, actual = _simulate_actual(layout)
    except Exception:
        return False

    starts = layout["starts"]
    legend = layout["legend"]
    foreign_starts = set(starts.values())
    solo: Dict[int, dict] = {}
    for item in legend:
        color = int(item["color"])
        solo[color] = _simulate_solo(
            layout["shape"],
            start=starts[color],
            color=color,
            step_len=int(item["length"]),
            bg=int(layout["bg"]),
        )

    lengths = [len(actual[int(item["color"])]["path"]) for item in legend]
    if min(lengths) < 4:
        return False
    if sum(meta["turns"] for meta in actual.values()) < 3:
        return False
    if max(lengths) < 8:
        return False

    pair_intersection = False
    ordered_colors = [int(item["color"]) for item in legend]
    for i, c1 in enumerate(ordered_colors):
        for c2 in ordered_colors[i + 1 :]:
            if solo[c1]["path_set"] & solo[c2]["path_set"]:
                pair_intersection = True
                break
        if pair_intersection:
            break
    if not pair_intersection:
        return False

    shortened = False
    trail_collision = False
    for color in ordered_colors:
        actual_meta = actual[color]
        solo_meta = solo[color]
        if any(cell in foreign_starts and cell != starts[color] for cell in actual_meta["path"][1:]):
            return False
        if actual_meta["blocker"] == "start":
            return False
        if len(actual_meta["path"]) < len(solo_meta["path"]):
            shortened = True
        if actual_meta["stop_reason"] == "occupied" and actual_meta["blocker"] == "trail":
            trail_collision = True
    if not shortened or not trail_collision:
        return False

    painted = int(np.count_nonzero(out != layout["bg"]))
    if painted >= (out.shape[0] * out.shape[1]) // 2:
        return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid staircase-particle input")
