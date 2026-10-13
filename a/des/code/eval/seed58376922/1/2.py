from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_VECTORS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
EDGE_ORDER = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400
GEN_COLORS = tuple(range(1, 10))

PIECE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)

ARROW_SHAPES = {
    "top": (
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    "bottom": (
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    "left": (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    "right": (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
    ),
}


def _offset_cells(cells: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + r, c0 + c) for r, c in cells)


def _shape_hw(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _touching_edges(cells: tuple[tuple[int, int], ...], h: int, w: int) -> set[str]:
    edges: set[str] = set()
    for r, c in cells:
        if r == 0:
            edges.add("top")
        if r == h - 1:
            edges.add("bottom")
        if c == 0:
            edges.add("left")
        if c == w - 1:
            edges.add("right")
    return edges


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((int(nr), int(nc)))
            cells.sort()
            components.append({"color": color, "cells": tuple(cells)})

    return components


def _priority(piece: dict[str, object], h: int, w: int) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = _bbox(piece["cells"])
    edge = piece["edge"]
    if edge == "top":
        dist = r0
        axis_a = c0
        axis_b = c1
    elif edge == "bottom":
        dist = h - 1 - r1
        axis_a = c0
        axis_b = c1
    elif edge == "left":
        dist = c0
        axis_a = r0
        axis_b = r1
    else:
        dist = w - 1 - c1
        axis_a = r0
        axis_b = r1
    return dist, EDGE_ORDER.index(edge), axis_a, axis_b, r0 + c0 + r1 + c1, int(piece["color"])


def _parse_scene(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows: dict[int, str] = {}
    fragments: list[dict[str, object]] = []

    for comp in _find_components(g):
        color = int(comp["color"])
        cells = comp["cells"]
        edges = _touching_edges(cells, h, w)
        if edges:
            assert len(cells) == 3, "edge markers must be 3-pixel arrows"
            assert len(edges) == 1, "arrow must belong to exactly one edge"
            edge = next(iter(edges))
            assert color not in arrows, "each color must map to one arrow"
            arrows[color] = edge
        else:
            fragments.append({"color": color, "cells": cells})

    pieces: list[dict[str, object]] = []
    for frag in fragments:
        color = int(frag["color"])
        assert color in arrows, "each fragment color needs a matching arrow"
        piece = dict(frag)
        piece["edge"] = arrows[color]
        pieces.append(piece)

    return h, w, pieces


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w, pieces = _parse_scene(grid)
    out = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    trace: list[dict[str, object]] = []

    for piece in sorted(pieces, key=lambda item: _priority(item, h, w)):
        cells = piece["cells"]
        dr, dc = DIR_VECTORS[piece["edge"]]
        steps = 0
        reason = "boundary"
        while True:
            shifted = tuple((r + (steps + 1) * dr, c + (steps + 1) * dc) for r, c in cells)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
                reason = "boundary"
                break
            if any((r, c) in occupied for r, c in shifted):
                reason = "piece"
                break
            steps += 1
        final_cells = tuple((r + steps * dr, c + steps * dc) for r, c in cells)
        for r, c in final_cells:
            out[r, c] = int(piece["color"])
            occupied.add((int(r), int(c)))
        trace.append(
            {
                "color": int(piece["color"]),
                "edge": piece["edge"],
                "steps": steps,
                "reason": reason,
                "initial": cells,
                "final": final_cells,
            }
        )

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(np.asarray(grid, dtype=int))[0]


def _can_place(cells: tuple[tuple[int, int], ...], forbidden: np.ndarray) -> bool:
    h, w = forbidden.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(forbidden[r, c]):
            return False
    return True


def _stamp(grid: np.ndarray, forbidden: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    h, w = grid.shape
    for r, c in cells:
        grid[r, c] = int(color)
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                forbidden[rr, cc] = True


def _arrow_candidates(h: int, w: int, edge: str, shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    sh, sw = _shape_hw(shape)
    candidates: list[tuple[tuple[int, int], ...]] = []
    if edge == "top":
        starts = [(0, c0) for c0 in range(w - sw + 1)]
    elif edge == "bottom":
        starts = [(h - sh, c0) for c0 in range(w - sw + 1)]
    elif edge == "left":
        starts = [(r0, 0) for r0 in range(h - sh + 1)]
    else:
        starts = [(r0, w - sw) for r0 in range(h - sh + 1)]

    for r0, c0 in starts:
        cells = _offset_cells(shape, r0, c0)
        if _touching_edges(cells, h, w) != {edge}:
            continue
        if any((r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)) for r, c in cells):
            continue
        candidates.append(cells)
    return candidates


def _piece_candidates(h: int, w: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    sh, sw = _shape_hw(shape)
    candidates: list[tuple[tuple[int, int], ...]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            candidates.append(_offset_cells(shape, r0, c0))
    return candidates


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(11, 19))
    w = int(rng.integers(11, 19))
    color_count = int(rng.integers(2, 5))
    colors = [int(v) for v in rng.choice(GEN_COLORS, size=color_count, replace=False)]
    directions = [EDGE_ORDER[int(v)] for v in rng.integers(0, 4, size=color_count)]
    if len(set(directions)) < 2:
        return None

    frag_counts = [int(rng.integers(1, 4)) for _ in range(color_count)]
    if sum(frag_counts) < 4:
        return None

    require_piece_block = bool(rng.random() < 0.45)

    grid = np.zeros((h, w), dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    arrow_order = list(range(color_count))
    rng.shuffle(arrow_order)
    for idx in arrow_order:
        color = colors[idx]
        edge = directions[idx]
        shapes = list(ARROW_SHAPES[edge])
        rng.shuffle(shapes)
        placed = False
        for shape in shapes:
            candidates = _arrow_candidates(h, w, edge, shape)
            rng.shuffle(candidates)
            for cells in candidates:
                if _can_place(cells, forbidden):
                    _stamp(grid, forbidden, cells, color)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    piece_jobs: list[tuple[int, str, tuple[tuple[int, int], ...]]] = []
    for color, edge, count in zip(colors, directions, frag_counts):
        for _ in range(count):
            shape = PIECE_SHAPES[int(rng.integers(0, len(PIECE_SHAPES)))]
            piece_jobs.append((int(color), edge, shape))
    rng.shuffle(piece_jobs)

    for color, edge, shape in piece_jobs:
        candidates = _piece_candidates(h, w, shape)
        rng.shuffle(candidates)
        # Mild bias: keep fragments at least two cells away from their target edge in some samples.
        if rng.random() < 0.5:
            filtered = []
            for cells in candidates:
                r0, c0, r1, c1 = _bbox(cells)
                if edge == "top" and r0 < 2:
                    continue
                if edge == "bottom" and r1 > h - 3:
                    continue
                if edge == "left" and c0 < 2:
                    continue
                if edge == "right" and c1 > w - 3:
                    continue
                filtered.append(cells)
            if filtered:
                candidates = filtered
        placed = False
        for cells in candidates:
            if _can_place(cells, forbidden):
                _stamp(grid, forbidden, cells, color)
                placed = True
                break
        if not placed:
            return None

    out, trace = _simulate(grid)
    if np.array_equal(grid, out):
        return None
    if require_piece_block and not any(item["reason"] == "piece" for item in trace):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
