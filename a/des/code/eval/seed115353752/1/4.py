from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


GEN_ATTEMPTS = 400
CELL_ROW_RANGE = (2, 4)
CELL_COL_RANGE = (2, 4)
CELL_SIZE_RANGE = (5, 7)
CELL_PIXELS_RANGE = (3, 5)
CASE_MODES = ("isolated", "mixed", "connected")


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _unique_winner(counts: Counter[int], message: str) -> int:
    assert counts, message
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, message
    return int(winners[0])


def _line_indices(grid: np.ndarray, bg: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] != bg))]
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] != bg))]


def _spans_from_lines(lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(lines) >= 2, "expected border-delimited table lines"
    assert lines[0] == 0 and lines[-1] == limit - 1, "table should cover the full grid"
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "expected positive-size cell interiors"
        spans.append((int(a + 1), int(b - 1)))
    return spans


def _segment_color(segment: np.ndarray, bg: int) -> int:
    values = {int(v) for v in np.unique(segment)}
    assert bg not in values, "separator segments must stay on the line palette"
    assert len(values) == 1, "each separator segment should be uniform"
    return int(next(iter(values)))


def _cell_id(row: int, col: int, n_cols: int) -> int:
    return int(row * n_cols + col)


def _parse_table(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    row_lines = _line_indices(g, bg, axis=0)
    col_lines = _line_indices(g, bg, axis=1)
    row_spans = _spans_from_lines(row_lines, g.shape[0])
    col_spans = _spans_from_lines(col_lines, g.shape[1])

    line_mask = np.zeros_like(g, dtype=bool)
    line_mask[row_lines, :] = True
    line_mask[:, col_lines] = True
    grey = _unique_winner(Counter(int(v) for v in g[line_mask].ravel() if int(v) != bg), "expected a unique grid-line color")

    n_rows = len(row_spans)
    n_cols = len(col_spans)
    cells: dict[int, tuple[int, int, int, int]] = {}
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            cells[_cell_id(r, c, n_cols)] = (int(r0), int(r1), int(c0), int(c1))

    uf = UnionFind(n_rows * n_cols)
    bridges: dict[tuple[int, int], int] = {}

    for r, (r0, r1) in enumerate(row_spans):
        for c in range(n_cols - 1):
            line_col = col_lines[c + 1]
            color = _segment_color(g[r0 : r1 + 1, line_col], bg)
            if color != grey:
                a = _cell_id(r, c, n_cols)
                b = _cell_id(r, c + 1, n_cols)
                uf.union(a, b)
                bridges[(a, b)] = int(color)

    for r in range(n_rows - 1):
        line_row = row_lines[r + 1]
        for c, (c0, c1) in enumerate(col_spans):
            color = _segment_color(g[line_row, c0 : c1 + 1], bg)
            if color != grey:
                a = _cell_id(r, c, n_cols)
                b = _cell_id(r + 1, c, n_cols)
                uf.union(a, b)
                bridges[(a, b)] = int(color)

    groups: dict[int, list[int]] = defaultdict(list)
    for cell in range(n_rows * n_cols):
        groups[uf.find(cell)].append(cell)

    return {
        "grid": g,
        "bg": int(bg),
        "grey": int(grey),
        "row_lines": row_lines,
        "col_lines": col_lines,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cells": cells,
        "groups": groups,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "bridges": bridges,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group table cells through non-grey separator segments, then repaint each group's
    cell interiors to that group's unique majority non-background color.
    """
    info = _parse_table(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = int(info["bg"])
    grey = int(info["grey"])
    cells: dict[int, tuple[int, int, int, int]] = info["cells"]  # type: ignore[assignment]
    groups: dict[int, list[int]] = info["groups"]  # type: ignore[assignment]

    out = g.copy()
    for group_cells in groups.values():
        votes: Counter[int] = Counter()
        masks: list[tuple[int, int, int, int, np.ndarray]] = []
        for cell in group_cells:
            r0, r1, c0, c1 = cells[cell]
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            paint_mask = patch != bg
            masks.append((r0, r1, c0, c1, paint_mask))
            voters = patch[(patch != bg) & (patch != grey)]
            if voters.size:
                colors, counts = np.unique(voters, return_counts=True)
                for color, count in zip(colors, counts):
                    votes[int(color)] += int(count)

        if not votes:
            continue

        majority = _unique_winner(votes, "expected a unique group majority color")
        for r0, r1, c0, c1, paint_mask in masks:
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[paint_mask] = majority
    return out


def _build_table(
    row_sizes: list[int],
    col_sizes: list[int],
    bg: int,
    grey: int,
) -> tuple[np.ndarray, list[int], list[int], dict[int, tuple[int, int, int, int]]]:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), bg, dtype=int)

    row_lines = [0]
    current = 0
    for size in row_sizes:
        current += size + 1
        row_lines.append(int(current))

    col_lines = [0]
    current = 0
    for size in col_sizes:
        current += size + 1
        col_lines.append(int(current))

    grid[row_lines, :] = grey
    grid[:, col_lines] = grey

    cells: dict[int, tuple[int, int, int, int]] = {}
    n_cols = len(col_sizes)
    r0 = 1
    for r, h in enumerate(row_sizes):
        c0 = 1
        for c, w in enumerate(col_sizes):
            cells[_cell_id(r, c, n_cols)] = (int(r0), int(r0 + h - 1), int(c0), int(c0 + w - 1))
            c0 += w + 1
        r0 += h + 1
    return grid, row_lines, col_lines, cells


def _all_edges(n_rows: int, n_cols: int) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            edges.append((_cell_id(r, c, n_cols), _cell_id(r, c + 1, n_cols), "vertical"))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            edges.append((_cell_id(r, c, n_cols), _cell_id(r + 1, c, n_cols), "horizontal"))
    return edges


def _edge_adjacency(n_rows: int, n_cols: int, edges: list[tuple[int, int, str]]) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (a, b, _kind) in enumerate(edges):
        adjacency[a].append((b, idx))
        adjacency[b].append((a, idx))
    for cell in range(n_rows * n_cols):
        assert adjacency[cell], "each cell should have at least one neighbor in the logical grid"
    return adjacency


def _connected_edge_ids(n_rows: int, n_cols: int, edges: list[tuple[int, int, str]]) -> set[int]:
    adjacency = _edge_adjacency(n_rows, n_cols, edges)
    start = random.randrange(n_rows * n_cols)
    seen = {start}
    stack = [start]
    chosen: set[int] = set()

    while stack:
        cell = stack[-1]
        options = [(nbr, idx) for nbr, idx in adjacency[cell] if nbr not in seen]
        if not options:
            stack.pop()
            continue
        nbr, idx = random.choice(options)
        seen.add(nbr)
        chosen.add(idx)
        stack.append(nbr)

    assert len(seen) == n_rows * n_cols
    return chosen


def _component_count(cell_count: int, edges: list[tuple[int, int, str]], chosen: set[int]) -> int:
    uf = UnionFind(cell_count)
    for idx in chosen:
        a, b, _kind = edges[idx]
        uf.union(a, b)
    return len({uf.find(cell) for cell in range(cell_count)})


def _choose_bridge_edges(n_rows: int, n_cols: int, edges: list[tuple[int, int, str]], mode: str) -> set[int]:
    cell_count = n_rows * n_cols
    if mode == "isolated":
        return set()

    if mode == "connected":
        chosen = _connected_edge_ids(n_rows, n_cols, edges)
        extras = [idx for idx in range(len(edges)) if idx not in chosen and random.random() < 0.25]
        chosen.update(extras)
        return chosen

    for _ in range(120):
        probability = random.uniform(0.25, 0.55)
        chosen = {idx for idx in range(len(edges)) if random.random() < probability}
        if not chosen or len(chosen) == len(edges):
            continue
        components = _component_count(cell_count, edges, chosen)
        if 1 < components < cell_count:
            return chosen
    raise AssertionError("failed to sample a mixed bridge pattern")


def _paint_bridge(
    grid: np.ndarray,
    edge: tuple[int, int, str],
    color: int,
    n_cols: int,
    row_lines: list[int],
    col_lines: list[int],
    cells: dict[int, tuple[int, int, int, int]],
) -> None:
    a, b, kind = edge
    ar = a // n_cols
    ac = a % n_cols
    br = b // n_cols
    bc = b % n_cols

    if kind == "vertical":
        left = min(ac, bc)
        line_col = col_lines[left + 1]
        r0, r1, _c0, _c1 = cells[a]
        grid[r0 : r1 + 1, line_col] = color
    else:
        top = min(ar, br)
        line_row = row_lines[top + 1]
        _r0, _r1, c0, c1 = cells[a]
        grid[line_row, c0 : c1 + 1] = color


def _sample_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(200):
        chosen = random.sample(coords, count)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if np.all(np.any(~mask, axis=1)) and np.all(np.any(~mask, axis=0)):
            return [(int(r), int(c)) for r, c in chosen]
    raise AssertionError("failed to place interior pixels without creating fake full lines")


def _sample_cell_slots(height: int, width: int) -> list[tuple[int, int]]:
    lo, hi = CELL_PIXELS_RANGE
    upper = min(hi, (height * width) // 4)
    lower = min(lo, upper)
    assert lower >= 1 and upper >= 1
    count = random.randint(lower, upper)
    return _sample_positions(height, width, count)


def _color_assignment(total: int, palette: list[int], force_multicolor: bool) -> list[int]:
    if total < 3 or len(palette) < 2:
        return [random.choice(palette)] * total

    use_multicolor = force_multicolor or random.random() < 0.65
    if not use_multicolor:
        return [random.choice(palette)] * total

    max_colors = min(3, len(palette), total - 1)
    n_colors = random.randint(2, max_colors)
    while n_colors >= 2:
        majority_low = max(2, total // 2 + 1)
        majority_high = total - (n_colors - 1)
        if majority_low <= majority_high:
            break
        n_colors -= 1
    if n_colors < 2:
        return [random.choice(palette)] * total

    chosen = random.sample(palette, n_colors)
    majority_count = random.randint(max(2, total // 2 + 1), total - (n_colors - 1))
    remainder = total - majority_count
    counts = [majority_count]
    if n_colors > 1:
        parts = [1] * (n_colors - 1)
        remainder -= n_colors - 1
        for _ in range(remainder):
            parts[random.randrange(n_colors - 1)] += 1
        counts.extend(parts)

    colors: list[int] = []
    for color, count in zip(chosen, counts):
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)
    return colors


def _logical_groups(cell_count: int, edges: list[tuple[int, int, str]], chosen: set[int]) -> dict[int, list[int]]:
    uf = UnionFind(cell_count)
    for idx in chosen:
        a, b, _kind = edges[idx]
        uf.union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for cell in range(cell_count):
        groups[uf.find(cell)].append(cell)
    return groups


def _build_case() -> np.ndarray:
    n_rows = random.randint(*CELL_ROW_RANGE)
    n_cols = random.randint(*CELL_COL_RANGE)
    row_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_rows)]
    col_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_cols)]

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    grey = int(palette[1])
    remaining = palette[2:]
    bridge_palette_size = random.randint(1, min(3, len(remaining) - 2))
    bridge_palette = [int(c) for c in remaining[:bridge_palette_size]]
    interior_palette = [int(c) for c in remaining[bridge_palette_size:]]

    grid, row_lines, col_lines, cells = _build_table(row_sizes, col_sizes, bg, grey)
    edges = _all_edges(n_rows, n_cols)
    mode = random.choice(CASE_MODES)
    chosen_edges = _choose_bridge_edges(n_rows, n_cols, edges, mode)

    for idx in chosen_edges:
        _paint_bridge(grid, edges[idx], random.choice(bridge_palette), n_cols, row_lines, col_lines, cells)

    groups = _logical_groups(n_rows * n_cols, edges, chosen_edges)
    forced_multicolor = random.choice(list(groups.keys()))

    cell_slots: dict[int, list[tuple[int, int]]] = {}
    for cell, (r0, r1, c0, c1) in cells.items():
        cell_slots[cell] = _sample_cell_slots(r1 - r0 + 1, c1 - c0 + 1)

    for root, group_cells in groups.items():
        placements: list[tuple[int, int, int]] = []
        for cell in group_cells:
            r0, _r1, c0, _c1 = cells[cell]
            for dr, dc in cell_slots[cell]:
                placements.append((cell, int(r0 + dr), int(c0 + dc)))

        colors = _color_assignment(len(placements), interior_palette, root == forced_multicolor)
        for (_cell, r, c), color in zip(placements, colors):
            grid[r, c] = int(color)

    info = _parse_table(grid)
    assert int(info["bg"]) == bg
    assert int(info["grey"]) == grey
    assert len(info["groups"]) == len(groups)
    out = main(grid)
    assert np.any(out != grid), "expected at least one group recoloring"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _build_case()
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid puzzle instance")
