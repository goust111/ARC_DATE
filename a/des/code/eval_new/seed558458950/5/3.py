import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray, bg: int) -> Optional[Tuple[int, int]]:
    h, w = grid.shape
    for r in range(h):
        row_vals = np.unique(grid[r])
        if len(row_vals) == 1:
            c = int(row_vals[0])
            if c != bg:
                return r, c
    return None


def _collect_same_color(grid: np.ndarray, start: Tuple[int, int], seen: np.ndarray, color: int) -> List[Tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    if seen[sr, sc] or grid[sr, sc] != color:
        return []

    q = deque([(sr, sc)])
    seen[sr, sc] = True
    comp = [(sr, sc)]
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                seen[nr, nc] = True
                q.append((nr, nc))
                comp.append((nr, nc))
    return comp


def _parse_node(comp: List[Tuple[int, int]]) -> Optional[Dict[str, int]]:
    if len(comp) < 10:
        return None

    cell_set = set(comp)
    min_r = min(r for r, _ in comp)
    max_r = max(r for r, _ in comp)
    min_c = min(c for _, c in comp)
    max_c = max(c for _, c in comp)

    for r0 in range(min_r - 1, max_r + 1):
        for c0 in range(min_c - 1, max_c + 1):
            block = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
            if not all(p in cell_set for p in block):
                continue
            extras = cell_set.difference(block)
            if len(extras) != 1:
                continue
            er, ec = next(iter(extras))
            if (er, ec) == (r0 - 1, c0 + 1):
                direction = "U"
            elif (er, ec) == (r0 + 3, c0 + 1):
                direction = "D"
            elif (er, ec) == (r0 + 1, c0 - 1):
                direction = "L"
            elif (er, ec) == (r0 + 1, c0 + 3):
                direction = "R"
            else:
                continue
            return {
                "r": r0,
                "c": c0,
                "dir": direction,
                "color": None,
                "cells": cell_set,
            }
    return None


def _protrusion_cell(node: Dict[str, int], direction: str) -> Tuple[int, int]:
    r = node["r"]
    c = node["c"]
    if direction == "U":
        return r - 1, c + 1
    if direction == "D":
        return r + 3, c + 1
    if direction == "L":
        return r + 1, c - 1
    return r + 1, c + 3


def _next_from_node(
    node: Dict[str, int],
    node_map: Dict[Tuple[int, int], int],
    grid: np.ndarray,
    sep_row: int,
    bg: int,
    grey: int,
) -> Optional[int]:
    drc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    dr, dc = drc[node["dir"]]

    pr, pc = _protrusion_cell(node, node["dir"])
    h, w = grid.shape
    if not (0 <= pr < sep_row and 0 <= pc < w):
        return None

    r, c = pr + dr, pc + dc
    if not (0 <= r < sep_row and 0 <= c < w):
        return None

    last = (pr, pc)
    while 0 <= r < sep_row and 0 <= c < w:
        v = grid[r, c]
        if v not in (bg, grey):
            return node_map.get((r, c), None)

        neigh = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < sep_row and 0 <= nc < w and (nr, nc) != last and grid[nr, nc] == grey:
                neigh.append((nr, nc))

        if not neigh:
            return None
        # Generated graphs are path-like: one continuation when skipping back edge.
        nxt = neigh[0]
        last = (r, c)
        r, c = nxt

    return None


def _detect_nodes(top: np.ndarray, bg: int, grey: int) -> List[Dict[str, int]]:
    h, w = top.shape
    seen = np.zeros((h, w), dtype=bool)
    nodes: List[Dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(top[r, c])
            if color in (bg, grey):
                continue
            comp = _collect_same_color(top, (r, c), seen, color)
            parsed = _parse_node(comp)
            if parsed is not None:
                parsed["color"] = color
                nodes.append(parsed)

    return nodes


def _detect_templates(bottom: np.ndarray, bg: int, grey: int, row_offset: int) -> List[Tuple[int, int]]:
    h, w = bottom.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: List[Tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or bottom[r, c] != grey:
                continue
            comp = _collect_same_color(bottom, (r, c), seen, grey)
            if len(comp) != 8:
                continue

            rs = [x for x, _ in comp]
            cs = [y for _, y in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 != 2 or c1 - c0 != 2:
                continue

            valid = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr in (r0, r1) or cc in (c0, c1):
                        if bottom[rr, cc] != grey:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
            if bottom[r0 + 1, c0 + 1] != bg:
                continue

            boxes.append((row_offset + r0 + 1, c0 + 1))

    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """Follow every node's protrusion marker to recover the unique visit order and fill templates.

    Input structure:
    - black is background.
    - A full horizontal grey separator row splits upper graph area and lower template area.
    - Upper area has 3×3 solid color nodes with one same-color one-cell protrusion each, indicating outgoing direction.
    - Lower area has an evenly spaced row or column of empty 3×3 template cells drawn with grey borders.

    Rule:
    - Parse node components in the upper area and identify each node’s outgoing direction from its protrusion.
    - Follow grey path from every node along the marker direction to find the directed successor (or no successor at terminal).
    - Choose the indegree-zero node as start and traverse through successors to get an ordered color list.
    - Detect template cells and fill their centers left-to-right (row layout) or top-to-bottom (column layout) in that order.

    Key insight:
    - The tiny same-color protrusion is the hidden pointer key; following it through lines uniquely determines the traversal order.
    """
    if grid.ndim != 2:
        return np.array(grid, copy=True)

    grid = np.array(grid)
    bg = _most_common_color(grid)

    sep_info = _find_separator_row(grid, bg)
    if sep_info is None:
        return np.array(grid, copy=True)

    sep_row, grey = sep_info
    if not (0 < sep_row < grid.shape[0] - 1):
        return np.array(grid, copy=True)

    top = grid[:sep_row, :]
    bottom = grid[sep_row + 1 :]

    nodes = _detect_nodes(top, bg, grey)
    if not nodes:
        return np.array(grid, copy=True)

    node_index: Dict[Tuple[int, int], int] = {}
    for i, node in enumerate(nodes):
        for pos in node["cells"]:
            node_index[pos] = i

    nxt: List[Optional[int]] = [None for _ in nodes]
    for i, node in enumerate(nodes):
        nxt[i] = _next_from_node(node, node_index, grid, sep_row, bg, grey)

    indeg = [0 for _ in nodes]
    for j in nxt:
        if j is not None:
            indeg[j] += 1

    starts = [i for i, d in enumerate(indeg) if d == 0]

    def walk(start: int) -> List[int]:
        cur = start
        seen = set()
        out = []
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.append(cur)
            cur = nxt[cur]
        return out

    if len(starts) == 1:
        order = walk(starts[0])
    else:
        order = max((walk(i) for i in range(len(nodes))), key=len, default=[])

    colors = [nodes[i]["color"] for i in order]
    if not colors:
        return np.array(grid, copy=True)

    templates = _detect_templates(bottom, bg, grey, sep_row + 1)
    if len(templates) != len(colors):
        return np.array(grid, copy=True)

    if len({r for r, _ in templates}) == 1:
        templates.sort(key=lambda p: p[1])
    else:
        templates.sort(key=lambda p: p[0])

    out = np.array(grid, copy=True)
    for color, (r, c) in zip(colors, templates):
        out[r, c] = color
    return out


def _pick_colors(n: int, forbidden: List[int]) -> List[int]:
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    random.shuffle(palette)
    used = set(forbidden)
    out: List[int] = []
    for c in palette:
        if c in used:
            continue
        out.append(c)
        used.add(c)
        if len(out) == n:
            break
    while len(out) < n:
        x = 13 + len(out)
        while x in used:
            x += 1
        out.append(x)
        used.add(x)
    return out


def _draw_node(grid: np.ndarray, r: int, c: int, color: int) -> None:
    grid[r : r + 3, c : c + 3] = color


def _draw_template(grid: np.ndarray, top: int, left: int, grey: int, bg: int) -> None:
    grid[top, left : left + 3] = grey
    grid[top + 2, left : left + 3] = grey
    grid[top : top + 3, left] = grey
    grid[top : top + 3, left + 2] = grey
    grid[top + 1, left + 1] = bg


def _random_path(num_nodes: int, tries: int = 200) -> List[Tuple[int, int]]:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(tries):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < num_nodes:
            x, y = path[-1]
            options = [(x + dx, y + dy) for dx, dy in dirs if (x + dx, y + dy) not in used]
            if not options:
                break
            x2, y2 = random.choice(options)
            path.append((x2, y2))
            used.add((x2, y2))
        if len(path) == num_nodes:
            return path
    return [(i, 0) for i in range(num_nodes)]


def _line_is_clear(grid: np.ndarray, start: Tuple[int, int], end_exclusive: Tuple[int, int], sep_row: int, bg: int) -> bool:
    sr, sc = start
    er, ec = end_exclusive
    h, w = grid.shape
    if not (0 <= sr < sep_row and 0 <= sc < w):
        return False

    if sr == er:
        step = 1 if ec > sc else -1
        c = sc
        while c != ec:
            if not (0 <= c < w and 0 <= sr < sep_row and grid[sr, c] == bg):
                return False
            c += step
        return True

    if sc == ec:
        step = 1 if er > sr else -1
        r = sr
        while r != er:
            if not (0 <= r < sep_row and 0 <= sc < w and grid[r, sc] == bg):
                return False
            r += step
        return True

    return False


def _draw_grey_segment(grid: np.ndarray, start: Tuple[int, int], end_exclusive: Tuple[int, int], grey: int) -> None:
    sr, sc = start
    er, ec = end_exclusive
    if sr == er:
        step = 1 if ec > sc else -1
        c = sc
        while c != ec:
            grid[sr, c] = grey
            c += step
    else:
        step = 1 if er > sr else -1
        r = sr
        while r != er:
            grid[r, sc] = grey
            r += step


def _build_fallback() -> np.ndarray:
    bg = 0
    grey = 1
    h, w = 12, 20
    grid = np.full((h, w), bg, dtype=int)
    sep = 8
    grid[sep, :] = grey

    colors = [2, 3, 4]
    _draw_node(grid, 1, 2, colors[0])
    _draw_node(grid, 1, 8, colors[1])
    _draw_node(grid, 1, 14, colors[2])

    # First link: node[0] -> node[1] (R)
    grid[2, 5] = colors[0]
    grid[2, 6] = grey

    # Second link: node[1] -> node[2] (R)
    grid[2, 11] = colors[1]
    grid[2, 12] = grey

    # End marker for node[2] to top boundary (U)
    grid[0, 17] = colors[2]

    # Fill three templates.
    _draw_template(grid, sep + 1, 2, grey, bg)
    _draw_template(grid, sep + 1, 6, grey, bg)
    _draw_template(grid, sep + 1, 10, grey, bg)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=int)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid


if __name__ == "__main__":
    for _ in range(3):
        inp = generate_input()
        out = main(inp)
        print(inp.shape, out.shape)
