from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
SOURCE_COLORS = (2, 3, 6, 7, 8, 9)

MARGIN = 6
SPACING = 4
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sources: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if seen[r, c] or col in (BLACK, BLUE, GREY):
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if len(cells) != 4:
                continue

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if not np.all(grid[r0 : r0 + 2, c0 : c0 + 2] == col):
                continue

            sources.append({"color": col, "cells": cells, "bbox": (r0, c0, r1, c1)})

    return sources


def _find_terminals(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    terminals: list[tuple[int, int]] = []
    for r in range(h - 3):
        for c in range(w - 3):
            sub = grid[r : r + 4, c : c + 4]
            if not np.all(sub[1:3, 1:3] == BLACK):
                continue
            if not np.all(sub[0, :] == GREY):
                continue
            if not np.all(sub[3, :] == GREY):
                continue
            if not np.all(sub[:, 0] == GREY):
                continue
            if not np.all(sub[:, 3] == GREY):
                continue
            terminals.append((r, c))
    return terminals


def _terminal_border_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dc in range(4):
        cells.append((r0, c0 + dc))
        cells.append((r0 + 3, c0 + dc))
    for dr in range(1, 3):
        cells.append((r0 + dr, c0))
        cells.append((r0 + dr, c0 + 3))
    return cells


def _reachable_pipe_cells(grid: np.ndarray, source_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    start: set[tuple[int, int]] = set()
    for r, c in source_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                start.add((nr, nc))

    visited = set(start)
    q = deque(start)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through grey pipes into every reachable terminal container.

    Input structure:
    - Background is black.
    - Grey pixels form a 1-cell-wide orthogonal pipe network with bends, T-junctions, and crossings.
    - Some pipe cells are replaced by blue valve pixels; a valve breaks connectivity and cannot be crossed.
    - Some pipe endpoints touch solid 2×2 colored source blocks.
    - Other pipe endpoints touch 4×4 grey terminal frames whose 2×2 interiors are black.

    Rule:
    - Detect every 2×2 source block and every 4×4 terminal frame.
    - For each source, flood through grey pipe pixels only; stop at blue valves because they are not passable.
    - Any terminal frame whose grey border is reached by exactly one source has its 2×2 interior filled with that source color.
    - Any terminal reached from multiple sources is filled yellow, while unreachable terminals stay black; pipes, valves, and source blocks remain unchanged.

    Key insight:
    - The task is graph reachability on the grey pipe network, but terminals are passive frame objects and blue valve pixels cut what would otherwise look like continuous paths.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sources = _find_sources(g)
    terminals = _find_terminals(g)
    terminal_hits: list[list[int]] = [[] for _ in terminals]

    for s_idx, src in enumerate(sources):
        reachable = _reachable_pipe_cells(g, src["cells"])  # type: ignore[index]
        for t_idx, (r0, c0) in enumerate(terminals):
            if any(cell in reachable for cell in _terminal_border_cells(r0, c0)):
                terminal_hits[t_idx].append(s_idx)

    for (r0, c0), hits in zip(terminals, terminal_hits):
        if not hits:
            continue
        fill = YELLOW if len(hits) >= 2 else int(sources[hits[0]]["color"])
        out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = fill

    return out


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _random_tree(rows: int, cols: int, rng: np.random.Generator):
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    stack = [start]
    visited = {start}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while stack:
        node = stack[-1]
        fresh = [nbr for nbr in _coarse_neighbors(node, rows, cols) if nbr not in visited]
        if not fresh:
            stack.pop()
            continue
        nxt = fresh[int(rng.integers(len(fresh)))]
        visited.add(nxt)
        stack.append(nxt)
        edges.append(_edge_key(node, nxt))

    return edges


def _coarse_components(
    rows: int,
    cols: int,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    cut_edges: set[tuple[tuple[int, int], tuple[int, int]]],
):
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {(r, c): [] for r in range(rows) for c in range(cols)}
    for a, b in edges:
        if (a, b) in cut_edges:
            continue
        adj[a].append(b)
        adj[b].append(a)

    comp_index: dict[tuple[int, int], int] = {}
    components: list[list[tuple[int, int]]] = []
    for node in adj:
        if node in comp_index:
            continue
        idx = len(components)
        q = deque([node])
        comp_index[node] = idx
        nodes: list[tuple[int, int]] = []
        while q:
            cur = q.popleft()
            nodes.append(cur)
            for nxt in adj[cur]:
                if nxt not in comp_index:
                    comp_index[nxt] = idx
                    q.append(nxt)
        components.append(nodes)
    return components, comp_index


def _component_sites(
    rows: int, cols: int, comp_index: dict[tuple[int, int], int]
) -> dict[int, list[tuple[tuple[int, int], str]]]:
    sites: dict[int, list[tuple[tuple[int, int], str]]] = {}
    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            idx = comp_index[node]
            bucket = sites.setdefault(idx, [])
            if r == 0:
                bucket.append((node, "U"))
            if r == rows - 1:
                bucket.append((node, "D"))
            if c == 0:
                bucket.append((node, "L"))
            if c == cols - 1:
                bucket.append((node, "R"))
    return sites


def _node_to_grid(node: tuple[int, int]) -> tuple[int, int]:
    return MARGIN + SPACING * node[0], MARGIN + SPACING * node[1]


def _site_layout(
    node_rc: tuple[int, int], side: str, kind: str, height: int, width: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r, c = node_rc

    if kind == "source":
        if side == "U":
            footprint = {(rr, cc) for rr in range(0, 2) for cc in range(c - 1, c + 1)}
            footprint |= {(rr, c) for rr in range(2, r)}
            paint = [(rr, cc) for rr in range(0, 2) for cc in range(c - 1, c + 1)]
            stub = [(rr, c) for rr in range(2, r)]
            return footprint, paint, stub
        if side == "D":
            footprint = {(rr, cc) for rr in range(height - 2, height) for cc in range(c - 1, c + 1)}
            footprint |= {(rr, c) for rr in range(r + 1, height - 2)}
            paint = [(rr, cc) for rr in range(height - 2, height) for cc in range(c - 1, c + 1)]
            stub = [(rr, c) for rr in range(r + 1, height - 2)]
            return footprint, paint, stub
        if side == "L":
            footprint = {(rr, cc) for rr in range(r - 1, r + 1) for cc in range(0, 2)}
            footprint |= {(r, cc) for cc in range(2, c)}
            paint = [(rr, cc) for rr in range(r - 1, r + 1) for cc in range(0, 2)]
            stub = [(r, cc) for cc in range(2, c)]
            return footprint, paint, stub
        footprint = {(rr, cc) for rr in range(r - 1, r + 1) for cc in range(width - 2, width)}
        footprint |= {(r, cc) for cc in range(c + 1, width - 2)}
        paint = [(rr, cc) for rr in range(r - 1, r + 1) for cc in range(width - 2, width)]
        stub = [(r, cc) for cc in range(c + 1, width - 2)]
        return footprint, paint, stub

    if side == "U":
        footprint = {(rr, cc) for rr in range(0, 4) for cc in range(c - 1, c + 3)}
        footprint |= {(rr, c) for rr in range(4, r)}
        border = [(rr, cc) for rr in range(0, 4) for cc in range(c - 1, c + 3) if rr in (0, 3) or cc in (c - 1, c + 2)]
        stub = [(rr, c) for rr in range(4, r)]
        return footprint, border, stub
    if side == "D":
        footprint = {(rr, cc) for rr in range(height - 4, height) for cc in range(c - 1, c + 3)}
        footprint |= {(rr, c) for rr in range(r + 1, height - 4)}
        border = [
            (rr, cc)
            for rr in range(height - 4, height)
            for cc in range(c - 1, c + 3)
            if rr in (height - 4, height - 1) or cc in (c - 1, c + 2)
        ]
        stub = [(rr, c) for rr in range(r + 1, height - 4)]
        return footprint, border, stub
    if side == "L":
        footprint = {(rr, cc) for rr in range(r - 1, r + 3) for cc in range(0, 4)}
        footprint |= {(r, cc) for cc in range(4, c)}
        border = [(rr, cc) for rr in range(r - 1, r + 3) for cc in range(0, 4) if rr in (r - 1, r + 2) or cc in (0, 3)]
        stub = [(r, cc) for cc in range(4, c)]
        return footprint, border, stub
    footprint = {(rr, cc) for rr in range(r - 1, r + 3) for cc in range(width - 4, width)}
    footprint |= {(r, cc) for cc in range(c + 1, width - 4)}
    border = [
        (rr, cc)
        for rr in range(r - 1, r + 3)
        for cc in range(width - 4, width)
        if rr in (r - 1, r + 2) or cc in (width - 4, width - 1)
    ]
    stub = [(r, cc) for cc in range(c + 1, width - 4)]
    return footprint, border, stub


def _draw_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], valved: bool):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        cols = range(min(c0, c1), max(c0, c1) + 1)
        cells = [(r0, cc) for cc in cols]
    else:
        rows = range(min(r0, r1), max(r0, r1) + 1)
        cells = [(rr, c0) for rr in rows]
    for rr, cc in cells:
        grid[rr, cc] = GREY
    if valved:
        mid = len(cells) // 2
        rr, cc = cells[mid]
        grid[rr, cc] = BLUE


def _paint_attachment(
    grid: np.ndarray, node: tuple[int, int], side: str, kind: str, color: int | None
) -> bool:
    height, width = grid.shape
    node_rc = _node_to_grid(node)
    footprint, paint_cells, stub_cells = _site_layout(node_rc, side, kind, height, width)

    for rr, cc in footprint:
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if int(grid[rr, cc]) != BLACK:
            return False

    for rr, cc in stub_cells:
        grid[rr, cc] = GREY

    if kind == "source":
        assert color is not None
        for rr, cc in paint_cells:
            grid[rr, cc] = int(color)
    else:
        for rr, cc in paint_cells:
            grid[rr, cc] = GREY
    return True


def _terminal_statuses(inp: np.ndarray, out: np.ndarray) -> list[str]:
    statuses: list[str] = []
    for r0, c0 in _find_terminals(inp):
        vals = set(int(v) for v in out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3].ravel())
        if len(vals) != 1:
            return []
        val = next(iter(vals))
        if val == BLACK:
            statuses.append("unreachable")
        elif val == YELLOW:
            statuses.append("mixed")
        else:
            statuses.append("single")
    return statuses


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(3, 6))
    cols = int(rng.integers(3, 6))
    height = 2 * MARGIN + SPACING * (rows - 1) + 1
    width = 2 * MARGIN + SPACING * (cols - 1) + 1
    if max(height, width) > 30:
        return None

    edges = _random_tree(rows, cols, rng)
    edge_count = len(edges)
    if edge_count < 2:
        return None

    components = None
    comp_index = None
    comp_sites = None
    role_components = None
    cut_edges = None

    for _ in range(80):
        cut_count = 2 if rng.random() < 0.65 else 3
        cut_count = min(cut_count, edge_count)
        chosen = rng.choice(edge_count, size=cut_count, replace=False)
        cut_edges = {edges[int(i)] for i in chosen}
        components, comp_index = _coarse_components(rows, cols, edges, cut_edges)
        if len(components) < 3:
            continue
        comp_sites = _component_sites(rows, cols, comp_index)
        comp_ids = list(range(len(components)))
        rng.shuffle(comp_ids)
        picked = None
        for a in comp_ids:
            if len(comp_sites.get(a, [])) < 3:
                continue
            for b in comp_ids:
                if b == a or len(comp_sites.get(b, [])) < 2:
                    continue
                for c in comp_ids:
                    if c in (a, b) or len(comp_sites.get(c, [])) < 1:
                        continue
                    picked = (a, b, c)
                    break
                if picked is not None:
                    break
            if picked is not None:
                break
        if picked is None:
            continue
        role_components = picked
        break

    if role_components is None or components is None or comp_sites is None or cut_edges is None:
        return None

    comp_multi, comp_single, comp_zero = role_components
    multi_sites = list(comp_sites[comp_multi])
    single_sites = list(comp_sites[comp_single])
    zero_sites = list(comp_sites[comp_zero])
    rng.shuffle(multi_sites)
    rng.shuffle(single_sites)
    rng.shuffle(zero_sites)

    multi_src = 2 + int(len(multi_sites) >= 5 and rng.random() < 0.5)
    multi_term = 1 + int(len(multi_sites) - multi_src >= 2 and rng.random() < 0.5)
    if multi_src + multi_term > len(multi_sites):
        multi_term = max(1, len(multi_sites) - multi_src)
    if multi_src < 2 or multi_src + multi_term > len(multi_sites):
        return None

    single_src = 1
    single_term = 1 + int(len(single_sites) >= 3 and rng.random() < 0.5)
    if single_src + single_term > len(single_sites):
        single_term = 1
    if single_src + single_term > len(single_sites):
        return None

    zero_term = 1 + int(len(zero_sites) >= 2 and rng.random() < 0.5)
    zero_term = min(zero_term, len(zero_sites))
    if zero_term < 1:
        return None

    attachments: list[tuple[str, tuple[int, int], str, int | None]] = []
    for site in multi_sites[:multi_src]:
        attachments.append(("source", site[0], site[1], None))
    for site in multi_sites[multi_src : multi_src + multi_term]:
        attachments.append(("terminal", site[0], site[1], None))

    for site in single_sites[:single_src]:
        attachments.append(("source", site[0], site[1], None))
    for site in single_sites[single_src : single_src + single_term]:
        attachments.append(("terminal", site[0], site[1], None))

    for site in zero_sites[:zero_term]:
        attachments.append(("terminal", site[0], site[1], None))

    source_count = sum(kind == "source" for kind, *_ in attachments)
    if source_count < 3 or source_count > len(SOURCE_COLORS):
        return None
    source_colors = list(rng.choice(np.array(SOURCE_COLORS), size=source_count, replace=False))

    grid = np.full((height, width), BLACK, dtype=int)

    for edge in edges:
        a, b = edge
        _draw_line(grid, _node_to_grid(a), _node_to_grid(b), valved=edge in cut_edges)

    color_idx = 0
    for kind, node, side, _ in attachments:
        color = None
        if kind == "source":
            color = int(source_colors[color_idx])
            color_idx += 1
        if not _paint_attachment(grid, node, side, kind, color):
            return None

    if len(_find_sources(grid)) != source_count:
        return None
    terminals = _find_terminals(grid)
    if len(terminals) < 3:
        return None
    if int(np.count_nonzero(grid == BLUE)) < 2:
        return None

    out = main(grid)
    statuses = _terminal_statuses(grid, out)
    if not statuses:
        return None
    if not {"single", "mixed", "unreachable"}.issubset(set(statuses)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid pipe-and-valve input after {GEN_RETRIES} attempts")
