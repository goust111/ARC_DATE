from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MARKER = 8
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIECE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

RNG = np.random.default_rng()

# Generator configuration.
PIECE_COUNT_OPTIONS = (2, 3, 4)
MIN_PIECE_DIM = 4
RECT_H_RANGE = (9, 14)
RECT_W_RANGE = (9, 14)
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (22, 31)
SCATTER_GAP = 2
LAYOUT_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _normalize_cells(
    cells: set[tuple[int, int]] | list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int, int, int, int]:
    pts = list(cells)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = min(rs), min(cs)
    local = tuple(sorted((r - r0, c - c0) for r, c in pts))
    return local, int(r0), int(c0), max(rs) - r0 + 1, max(cs) - c0 + 1


def _extract_pieces(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid != BG)
    pieces: list[dict[str, object]] = []
    for comp in comps:
        colors = [int(grid[r, c]) for r, c in comp]
        body_colors = sorted({col for col in colors if col not in (BG, MARKER)})
        assert len(body_colors) == 1, "each piece must have exactly one body color"
        marker_count = sum(col == MARKER for col in colors)
        assert marker_count >= 1, "each piece must contain at least one azure marker"

        local_cells, _r0, _c0, height, width = _normalize_cells(comp)
        local_set = set(local_cells)
        boundary = tuple(
            sorted(
                (r, c)
                for r, c in local_cells
                if any((r + dr, c + dc) not in local_set for dr, dc in DIRS)
            )
        )
        pieces.append(
            {
                "base": int(body_colors[0]),
                "markers": int(marker_count),
                "cells": local_cells,
                "boundary": boundary,
                "height": int(height),
                "width": int(width),
            }
        )
    marker_counts = [int(piece["markers"]) for piece in pieces]
    assert len(marker_counts) == len(set(marker_counts)), "marker counts must be unique"
    pieces.sort(key=lambda piece: int(piece["markers"]))
    return pieces


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return sorted(pairs, key=lambda item: (max(item), min(item)))


def _state_key(origins: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in origins)
    min_c = min(c for _, c in origins)
    return tuple((r - min_r, c - min_c) for r, c in origins)


def _search(
    pieces: list[dict[str, object]],
    target_h: int,
    target_w: int,
    idx: int,
    origins: list[tuple[int, int]],
    union: set[tuple[int, int]],
    min_r: int,
    max_r: int,
    min_c: int,
    max_c: int,
    failed: set[tuple[int, tuple[tuple[int, int], ...]]],
) -> list[tuple[int, int]] | None:
    if idx == len(pieces):
        if max_r - min_r + 1 == target_h and max_c - min_c + 1 == target_w:
            return list(origins)
        return None

    key = (idx, _state_key(origins))
    if key in failed:
        return None

    piece = pieces[idx]
    cells = piece["cells"]
    boundary = piece["boundary"]
    height = int(piece["height"])
    width = int(piece["width"])
    union_boundary = [
        (r, c)
        for r, c in union
        if any((r + dr, c + dc) not in union for dr, dc in DIRS)
    ]

    candidate_map: dict[tuple[int, int], tuple[int, int]] = {}
    for ur, uc in union_boundary:
        for dr, dc in DIRS:
            ar, ac = ur + dr, uc + dc
            for pr, pc in boundary:
                origin = (ar - pr, ac - pc)
                if origin in candidate_map:
                    continue

                next_min_r = min(min_r, origin[0])
                next_max_r = max(max_r, origin[0] + height - 1)
                next_min_c = min(min_c, origin[1])
                next_max_c = max(max_c, origin[1] + width - 1)
                if next_max_r - next_min_r + 1 > target_h or next_max_c - next_min_c + 1 > target_w:
                    continue

                overlap = False
                touch_edges = 0
                for rr, cc in cells:
                    gr, gc = origin[0] + rr, origin[1] + cc
                    if (gr, gc) in union:
                        overlap = True
                        break
                    for ddr, ddc in DIRS:
                        if (gr + ddr, gc + ddc) in union:
                            touch_edges += 1
                if overlap or touch_edges == 0:
                    continue

                bbox_area = (next_max_r - next_min_r + 1) * (next_max_c - next_min_c + 1)
                candidate_map[origin] = (-touch_edges, bbox_area)

    ordered = sorted(candidate_map.items(), key=lambda item: (item[1][0], item[1][1], item[0][0], item[0][1]))
    for origin, _score in ordered:
        translated = {(origin[0] + rr, origin[1] + cc) for rr, cc in cells}
        origins.append(origin)
        result = _search(
            pieces,
            target_h,
            target_w,
            idx + 1,
            origins,
            union | translated,
            min(min_r, origin[0]),
            max(max_r, origin[0] + height - 1),
            min(min_c, origin[1]),
            max(max_c, origin[1] + width - 1),
            failed,
        )
        origins.pop()
        if result is not None:
            return result

    failed.add(key)
    return None


def _paint_solution(pieces: list[dict[str, object]], origins: list[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in origins)
    min_c = min(c for _, c in origins)
    max_r = max(r + int(piece["height"]) - 1 for piece, (r, _c) in zip(pieces, origins))
    max_c = max(c + int(piece["width"]) - 1 for piece, (_r, c) in zip(pieces, origins))
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for piece, (origin_r, origin_c) in zip(pieces, origins):
        for rr, cc in piece["cells"]:
            out[origin_r - min_r + rr, origin_c - min_c + cc] = int(piece["base"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a single rectangle using azure counts as the order.

    Each non-black connected component is one piece. Azure marker cells belong to the piece,
    count the placement order, and disappear in the output by turning into the piece color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces = _extract_pieces(g)
    assert pieces, "expected at least one puzzle piece"

    total_area = sum(len(piece["cells"]) for piece in pieces)
    first = pieces[0]
    first_origin = (0, 0)
    first_union = {(r, c) for r, c in first["cells"]}
    first_max_r = int(first["height"]) - 1
    first_max_c = int(first["width"]) - 1

    solutions: list[np.ndarray] = []
    seen = set()
    for target_h, target_w in _factor_pairs(total_area):
        origins = [first_origin]
        failed: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
        solved = _search(
            pieces,
            int(target_h),
            int(target_w),
            1,
            origins,
            first_union,
            0,
            first_max_r,
            0,
            first_max_c,
            failed,
        )
        if solved is None:
            continue
        out = _paint_solution(pieces, solved)
        if out.shape != (int(target_h), int(target_w)):
            continue
        key = (out.shape, out.tobytes())
        if key not in seen:
            seen.add(key)
            solutions.append(out)

    assert solutions, "failed to assemble the pieces into a rectangle"
    return min(solutions, key=lambda arr: (arr.shape[0] * 100 + arr.shape[1], arr.tobytes()))


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _rectangle_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, h, w = rect
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _split_rectangles(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        rects = [(0, 0, height, width)]
        while len(rects) < count:
            candidates: list[tuple[int, list[str]]] = []
            for idx, (_r0, _c0, h, w) in enumerate(rects):
                opts: list[str] = []
                if h >= 2 * MIN_PIECE_DIM:
                    opts.append("h")
                if w >= 2 * MIN_PIECE_DIM:
                    opts.append("v")
                if opts:
                    candidates.append((idx, opts))
            if not candidates:
                break
            idx, opts = candidates[int(RNG.integers(len(candidates)))]
            r0, c0, h, w = rects.pop(idx)
            orient = opts[int(RNG.integers(len(opts)))]
            if orient == "h":
                cut = _randint(MIN_PIECE_DIM, h - MIN_PIECE_DIM)
                rects.extend([(r0, c0, cut, w), (r0 + cut, c0, h - cut, w)])
            else:
                cut = _randint(MIN_PIECE_DIM, w - MIN_PIECE_DIM)
                rects.extend([(r0, c0, h, cut), (r0, c0 + cut, h, w - cut)])
        if len(rects) == count:
            return rects
    return None


def _adjacencies(rects: list[tuple[int, int, int, int]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for i in range(len(rects)):
        r0a, c0a, ha, wa = rects[i]
        r1a, c1a = r0a + ha, c0a + wa
        for j in range(i + 1, len(rects)):
            r0b, c0b, hb, wb = rects[j]
            r1b, c1b = r0b + hb, c0b + wb

            if c1a == c0b:
                start = max(r0a, r0b)
                end = min(r1a, r1b)
                if end - start >= 3:
                    out.append({"kind": 0, "left": i, "right": j, "split": c0b, "start": start, "end": end})
            if c1b == c0a:
                start = max(r0a, r0b)
                end = min(r1a, r1b)
                if end - start >= 3:
                    out.append({"kind": 0, "left": j, "right": i, "split": c0a, "start": start, "end": end})
            if r1a == r0b:
                start = max(c0a, c0b)
                end = min(c1a, c1b)
                if end - start >= 3:
                    out.append({"kind": 1, "top": i, "bottom": j, "split": r0b, "start": start, "end": end})
            if r1b == r0a:
                start = max(c0a, c0b)
                end = min(c1a, c1b)
                if end - start >= 3:
                    out.append({"kind": 1, "top": j, "bottom": i, "split": r0a, "start": start, "end": end})
    return out


def _pick_spread_positions(values: list[int], count: int) -> list[int]:
    vals = values[:]
    RNG.shuffle(vals)
    chosen: list[int] = []
    for value in vals:
        if all(abs(value - prev) > 1 for prev in chosen):
            chosen.append(int(value))
        if len(chosen) == count:
            break
    if len(chosen) < count:
        for value in sorted(vals):
            if value not in chosen:
                chosen.append(int(value))
            if len(chosen) == count:
                break
    return chosen


def _apply_tabs(
    rects: list[tuple[int, int, int, int]], height: int, width: int
) -> list[set[tuple[int, int]]] | None:
    piece_cells = [_rectangle_cells(rect) for rect in rects]
    for adj in _adjacencies(rects):
        if adj["kind"] == 0:
            left = int(adj["left"])
            right = int(adj["right"])
            rows = list(range(int(adj["start"]) + 1, int(adj["end"]) - 1))
            if not rows:
                return None
            tab_count = 1 if len(rows) < 4 else _randint(1, 2)
            for row in _pick_spread_positions(rows, tab_count):
                if bool(RNG.integers(2)):
                    cell = (row, int(adj["split"]))
                    if cell not in piece_cells[right]:
                        return None
                    piece_cells[right].remove(cell)
                    piece_cells[left].add(cell)
                else:
                    cell = (row, int(adj["split"]) - 1)
                    if cell not in piece_cells[left]:
                        return None
                    piece_cells[left].remove(cell)
                    piece_cells[right].add(cell)
        else:
            top = int(adj["top"])
            bottom = int(adj["bottom"])
            cols = list(range(int(adj["start"]) + 1, int(adj["end"]) - 1))
            if not cols:
                return None
            tab_count = 1 if len(cols) < 4 else _randint(1, 2)
            for col in _pick_spread_positions(cols, tab_count):
                if bool(RNG.integers(2)):
                    cell = (int(adj["split"]), col)
                    if cell not in piece_cells[bottom]:
                        return None
                    piece_cells[bottom].remove(cell)
                    piece_cells[top].add(cell)
                else:
                    cell = (int(adj["split"]) - 1, col)
                    if cell not in piece_cells[top]:
                        return None
                    piece_cells[top].remove(cell)
                    piece_cells[bottom].add(cell)

    all_cells = set().union(*piece_cells)
    if len(all_cells) != height * width:
        return None
    if any(not _is_connected(cells) for cells in piece_cells):
        return None
    return piece_cells


def _adjacency_graph(piece_cells: list[set[tuple[int, int]]]) -> list[set[int]]:
    owner = {}
    for pid, cells in enumerate(piece_cells):
        for cell in cells:
            owner[cell] = pid
    graph = [set() for _ in piece_cells]
    for pid, cells in enumerate(piece_cells):
        for r, c in cells:
            for dr, dc in DIRS:
                other = owner.get((r + dr, c + dc))
                if other is not None and other != pid:
                    graph[pid].add(other)
    return graph


def _connected_order(piece_cells: list[set[tuple[int, int]]]) -> list[int]:
    root = next(pid for pid, cells in enumerate(piece_cells) if (0, 0) in cells)
    graph = _adjacency_graph(piece_cells)
    order: list[int] = []
    seen = {root}
    q = deque([root])
    while q:
        node = q.popleft()
        order.append(node)
        nbrs = list(graph[node] - seen)
        RNG.shuffle(nbrs)
        for nbr in nbrs:
            seen.add(nbr)
            q.append(nbr)
    assert len(order) == len(piece_cells), "piece adjacency graph must be connected"
    return order


def _interior_cells(local_cells: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cell_set = set(local_cells)
    return [
        (r, c)
        for r, c in local_cells
        if all((r + dr, c + dc) in cell_set for dr, dc in DIRS)
    ]


def _paint_output(piece_cells: list[set[tuple[int, int]]], colors: list[int], height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for pid, cells in enumerate(piece_cells):
        for r, c in cells:
            out[r, c] = int(colors[pid])
    return out


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    min_h = max(CANVAS_H_RANGE[0], max(patch.shape[0] for patch in patches) + 4)
    min_w = max(CANVAS_W_RANGE[0], max(patch.shape[1] for patch in patches) + 4)
    if min_h >= CANVAS_H_RANGE[1] or min_w >= CANVAS_W_RANGE[1]:
        return None

    canvas_h = _randint(min_h, CANVAS_H_RANGE[1] - 1)
    canvas_w = _randint(min_w, CANVAS_W_RANGE[1] - 1)
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    order = sorted(
        range(len(patches)),
        key=lambda idx: (patches[idx].shape[0] * patches[idx].shape[1], RNG.random()),
        reverse=True,
    )
    placed_boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(RNG.integers(0, canvas_h - ph + 1))
            c0 = int(RNG.integers(0, canvas_w - pw + 1))
            r1, c1 = r0 + ph - 1, c0 + pw - 1
            ok = True
            for br0, bc0, br1, bc1 in placed_boxes:
                if not (r1 + SCATTER_GAP < br0 or br1 + SCATTER_GAP < r0 or c1 + SCATTER_GAP < bc0 or bc1 + SCATTER_GAP < c0):
                    ok = False
                    break
            if not ok:
                continue
            mask = patch != BG
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][mask] != BG):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            placed_boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    piece_count = int(RNG.choice(PIECE_COUNT_OPTIONS))
    height = _randint(*RECT_H_RANGE)
    width = _randint(*RECT_W_RANGE)

    rects = _split_rectangles(height, width, piece_count)
    if rects is None:
        return None

    piece_cells = _apply_tabs(rects, height, width)
    if piece_cells is None:
        return None

    order = _connected_order(piece_cells)
    colors = list(RNG.choice(np.array(PIECE_COLORS, dtype=int), size=piece_count, replace=False))
    output = _paint_output(piece_cells, colors, height, width)

    marker_counts = {pid: rank + 1 for rank, pid in enumerate(order)}
    patches: list[np.ndarray] = []
    for pid, cells in enumerate(piece_cells):
        local_cells, _r0, _c0, ph, pw = _normalize_cells(cells)
        interior = _interior_cells(local_cells)
        if len(interior) < marker_counts[pid]:
            return None
        patch = np.full((ph, pw), BG, dtype=int)
        for r, c in local_cells:
            patch[r, c] = int(colors[pid])
        marker_idx = RNG.choice(len(interior), size=marker_counts[pid], replace=False)
        for idx in np.atleast_1d(marker_idx):
            r, c = interior[int(idx)]
            patch[r, c] = MARKER
        patches.append(patch)

    grid = _scatter_patches(patches)
    if grid is None:
        return None
    if len(_components(grid != BG)) != piece_count:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jigsaw assembly input")
