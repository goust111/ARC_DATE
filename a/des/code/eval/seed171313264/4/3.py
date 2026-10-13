import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
FRAME = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _tight_patch(coords: list[tuple[int, int]], values: dict[tuple[int, int], int]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(values[(r, c)])
    return patch


def _component_cells_from_patch(patch: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != BG)
    return [(int(r), int(c)) for r, c in coords]


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "missing maroon frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame too small"
    for c in range(c0, c1 + 1):
        assert int(grid[r0, c]) == FRAME and int(grid[r1, c]) == FRAME
    for r in range(r0, r1 + 1):
        assert int(grid[r, c0]) == FRAME and int(grid[r, c1]) == FRAME
    return r0, c0, r1, c1


def _piece_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    values = {(r, c): int(grid[r, c]) for r, c in coords}
    order = sum(1 for v in values.values() if int(v) == GREY)
    colors = sorted({int(v) for v in values.values() if int(v) != GREY})
    assert order >= 1, "each piece needs at least one grey marker"
    assert len(colors) == 1, "generator uses monochrome pieces"
    base_color = int(colors[0])
    filled = {pos: (base_color if int(v) == GREY else int(v)) for pos, v in values.items()}
    patch = _tight_patch(coords, filled)
    rel_cells = _component_cells_from_patch(patch)
    return {
        "order": int(order),
        "color": int(base_color),
        "patch": patch,
        "cells": rel_cells,
        "area": int(len(rel_cells)),
        "h": int(patch.shape[0]),
        "w": int(patch.shape[1]),
    }


def _candidate_placements(piece: dict, target_h: int, target_w: int, fixed_origin: bool) -> list[dict]:
    cells = piece["cells"]
    h = int(piece["h"])
    w = int(piece["w"])
    rows = [0] if fixed_origin else range(target_h - h + 1)
    cols = [0] if fixed_origin else range(target_w - w + 1)
    out: list[dict] = []
    for r0 in rows:
        for c0 in cols:
            placed = [(int(r0 + r), int(c0 + c)) for r, c in cells]
            placed_set = set(placed)
            mask = 0
            adj = 0
            for r, c in placed:
                bit = 1 << int(r * target_w + c)
                mask |= bit
            for r, c in placed:
                for nr, nc in _neighbors4(r, c, target_h, target_w):
                    if (nr, nc) not in placed_set:
                        adj |= 1 << int(nr * target_w + nc)
            out.append({"r0": int(r0), "c0": int(c0), "mask": int(mask), "adj": int(adj)})
    return out


def _empty_component_sizes(occupied: int, target_h: int, target_w: int) -> list[int]:
    seen: set[tuple[int, int]] = set()
    sizes: list[int] = []
    for r in range(target_h):
        for c in range(target_w):
            bit = 1 << int(r * target_w + c)
            if occupied & bit or (r, c) in seen:
                continue
            q = deque([(int(r), int(c))])
            seen.add((int(r), int(c)))
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in _neighbors4(cr, cc, target_h, target_w):
                    nbit = 1 << int(nr * target_w + nc)
                    if occupied & nbit or (nr, nc) in seen:
                        continue
                    seen.add((int(nr), int(nc)))
                    q.append((int(nr), int(nc)))
            sizes.append(int(size))
    return sizes


def _subset_sums(values: list[int]) -> set[int]:
    sums = {0}
    for value in values:
        sums |= {int(s + value) for s in sums}
    return sums


def _empty_regions_feasible(occupied: int, target_h: int, target_w: int, remaining_areas: list[int]) -> bool:
    if not remaining_areas:
        return True
    sizes = _empty_component_sizes(occupied, target_h, target_w)
    if len(sizes) > len(remaining_areas):
        return False
    min_area = min(remaining_areas)
    if any(size < min_area for size in sizes):
        return False
    sums = _subset_sums(remaining_areas)
    return all(size in sums for size in sizes)


def _solve_ordered_tiling(pieces: list[dict], target_h: int, target_w: int, count_limit: int = 1):
    full_mask = (1 << int(target_h * target_w)) - 1
    candidates = [
        _candidate_placements(piece, target_h, target_w, fixed_origin=(idx == 0))
        for idx, piece in enumerate(pieces)
    ]
    placements: list[dict | None] = [None] * len(pieces)
    solutions: list[list[dict]] = []
    suffix_areas = [0] * (len(pieces) + 1)
    for idx in range(len(pieces) - 1, -1, -1):
        suffix_areas[idx] = suffix_areas[idx + 1] + int(pieces[idx]["area"])

    def rec(idx: int, occupied: int) -> None:
        if len(solutions) >= count_limit:
            return
        if idx == len(pieces):
            if occupied == full_mask:
                solutions.append([dict(x) for x in placements if x is not None])
            return

        remaining_areas = [int(p["area"]) for p in pieces[idx + 1 :]]
        for cand in candidates[idx]:
            if int(cand["mask"]) & occupied:
                continue
            if idx > 0 and (int(cand["adj"]) & occupied) == 0:
                continue
            new_occupied = int(occupied | int(cand["mask"]))
            covered = new_occupied.bit_count()
            if covered + int(suffix_areas[idx + 1]) != int(target_h * target_w):
                continue
            if not _empty_regions_feasible(new_occupied, target_h, target_w, remaining_areas):
                continue
            placements[idx] = cand
            rec(idx + 1, new_occupied)
            placements[idx] = None
            if len(solutions) >= count_limit:
                return

    rec(0, 0)
    first = solutions[0] if solutions else None
    return first, len(solutions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jagged pieces into the maroon frame using grey-count order.

    The grey pixel count inside each piece gives its placement order. Pieces may
    only translate, never rotate, and together they tile the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fr0, fc0, fr1, fc1 = _frame_bbox(g)
    target_h = int(fr1 - fr0 - 1)
    target_w = int(fc1 - fc0 - 1)

    piece_mask = (g != BG) & (g != FRAME)
    comps = _find_components(piece_mask)
    assert comps, "missing pieces"
    pieces = [_piece_from_component(g, comp) for comp in comps]
    pieces.sort(key=lambda piece: int(piece["order"]))
    expected_orders = list(range(1, len(pieces) + 1))
    assert [int(piece["order"]) for piece in pieces] == expected_orders
    assert sum(int(piece["area"]) for piece in pieces) == int(target_h * target_w)

    placements, count = _solve_ordered_tiling(pieces, target_h, target_w, count_limit=1)
    assert placements is not None and count >= 1, "tiling not found"

    out = np.full((h, w), BG, dtype=int)
    out[g == FRAME] = FRAME
    for piece, placement in zip(pieces, placements):
        patch = piece["patch"]
        pr0 = int(fr0 + 1 + placement["r0"])
        pc0 = int(fc0 + 1 + placement["c0"])
        ph, pw = patch.shape
        sub = out[pr0 : pr0 + ph, pc0 : pc0 + pw]
        paint = patch != BG
        sub[paint] = patch[paint]
    return out


def _is_connected(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells:
        return True
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    return len(seen) == len(cells)


def _piece_frontier(piece: set[tuple[int, int]], remaining: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in piece:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in remaining:
                frontier.add((int(nr), int(nc)))
    return list(frontier)


def _grow_piece(
    remaining: set[tuple[int, int]],
    seed: tuple[int, int],
    target_size: int,
    h: int,
    w: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    piece = {seed}
    residual = set(remaining)
    residual.remove(seed)
    while len(piece) < target_size:
        frontier = _piece_frontier(piece, residual, h, w)
        if not frontier:
            return None
        feasible: list[tuple[float, tuple[int, int]]] = []
        for cell in frontier:
            next_residual = set(residual)
            next_residual.remove(cell)
            if next_residual and not _is_connected(next_residual, h, w):
                continue
            same_neighbors = sum((nr, nc) in piece for nr, nc in _neighbors4(cell[0], cell[1], h, w))
            seed_dist = abs(cell[0] - seed[0]) + abs(cell[1] - seed[1])
            score = 3.0 * same_neighbors - 0.35 * seed_dist + rng.random()
            feasible.append((score, cell))
        if not feasible:
            return None
        feasible.sort(reverse=True)
        top_score = feasible[0][0]
        top_cells = [cell for score, cell in feasible if score >= top_score - 0.8]
        chosen = rng.choice(top_cells)
        piece.add((int(chosen[0]), int(chosen[1])))
        residual.remove(chosen)
    return piece


def _choose_sizes(total_area: int, n_pieces: int, rng: random.Random) -> list[int] | None:
    min_area = int(n_pieces + 3)
    if total_area < min_area * n_pieces:
        return None
    sizes: list[int] = []
    remaining = int(total_area)
    for idx in range(n_pieces - 1):
        left = int(n_pieces - idx)
        avg = remaining // left
        lo = int(min_area)
        hi = int(remaining - min_area * (left - 1))
        hi = min(hi, avg + 4)
        lo = min(lo, hi)
        size = int(rng.randint(lo, hi))
        sizes.append(size)
        remaining -= size
    sizes.append(int(remaining))
    rng.shuffle(sizes)
    largest = max(range(len(sizes)), key=lambda i: sizes[i])
    sizes[0], sizes[largest] = sizes[largest], sizes[0]
    return sizes


def _build_partition(target_h: int, target_w: int, n_pieces: int, rng: random.Random):
    all_cells = {(r, c) for r in range(target_h) for c in range(target_w)}
    sizes = _choose_sizes(target_h * target_w, n_pieces, rng)
    if sizes is None:
        return None
    remaining = set(all_cells)
    placed_union: set[tuple[int, int]] = set()
    pieces: list[set[tuple[int, int]]] = []

    for idx in range(n_pieces - 1):
        if idx == 0:
            seed = (0, 0)
        else:
            frontier = sorted(
                cell
                for cell in remaining
                if any((nr, nc) in placed_union for nr, nc in _neighbors4(cell[0], cell[1], target_h, target_w))
            )
            if not frontier:
                return None
            seed = rng.choice(frontier[: min(len(frontier), 10)])
        piece = _grow_piece(remaining, seed, sizes[idx], target_h, target_w, rng)
        if piece is None:
            return None
        remaining -= piece
        placed_union |= piece
        pieces.append(piece)

    if not remaining or not _is_connected(remaining, target_h, target_w):
        return None
    if not any((nr, nc) in placed_union for cell in remaining for nr, nc in _neighbors4(cell[0], cell[1], target_h, target_w)):
        return None
    pieces.append(set(remaining))
    return pieces


def _piece_is_irregular(cells: set[tuple[int, int]]) -> bool:
    coords = list(cells)
    r0, c0, r1, c1 = _bbox(coords)
    area = len(cells)
    return area != (r1 - r0 + 1) * (c1 - c0 + 1)


def _adjacency_graph(pieces: list[set[tuple[int, int]]], target_h: int, target_w: int) -> list[set[int]]:
    owner = {}
    for idx, piece in enumerate(pieces):
        for cell in piece:
            owner[cell] = idx
    graph = [set() for _ in pieces]
    for idx, piece in enumerate(pieces):
        for r, c in piece:
            for nr, nc in _neighbors4(r, c, target_h, target_w):
                j = owner.get((nr, nc))
                if j is not None and j != idx:
                    graph[idx].add(int(j))
    return graph


def _assign_colors(pieces: list[set[tuple[int, int]]], target_h: int, target_w: int, rng: random.Random) -> list[int] | None:
    graph = _adjacency_graph(pieces, target_h, target_w)
    order = sorted(range(len(pieces)), key=lambda idx: (-len(graph[idx]), -len(pieces[idx])))
    colors = [None] * len(pieces)
    for idx in order:
        banned = {colors[nbr] for nbr in graph[idx] if colors[nbr] is not None}
        choices = [color for color in PIECE_COLORS if color not in banned]
        if not choices:
            return None
        colors[idx] = int(rng.choice(choices))
    return [int(color) for color in colors]


def _marker_cells(piece: set[tuple[int, int]], count: int, target_h: int, target_w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    degree2 = []
    degree1 = []
    all_cells = list(piece)
    for cell in all_cells:
        deg = sum((nr, nc) in piece for nr, nc in _neighbors4(cell[0], cell[1], target_h, target_w))
        if deg >= 3:
            degree2.append(cell)
        elif deg >= 2:
            degree1.append(cell)
    pool = degree2[:]
    extras = [cell for cell in degree1 if cell not in pool]
    pool.extend(extras)
    if len(pool) < count:
        pool = all_cells[:]
    if len(pool) < count:
        return None
    rng.shuffle(pool)
    return [(int(r), int(c)) for r, c in pool[:count]]


def _build_piece_patch(
    piece: set[tuple[int, int]],
    color: int,
    grey_count: int,
    target_h: int,
    target_w: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    markers = _marker_cells(piece, grey_count, target_h, target_w, rng)
    if markers is None:
        return None
    coords = list(piece)
    r0, c0, r1, c1 = _bbox(coords)
    clean = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    marked = np.full_like(clean, BG)
    marker_set = set(markers)
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        clean[rr, cc] = int(color)
        marked[rr, cc] = int(GREY if (r, c) in marker_set else color)
    return clean, marked


def _solver_pieces_from_clean_patches(clean_patches: list[np.ndarray]) -> list[dict]:
    pieces = []
    for idx, patch in enumerate(clean_patches, start=1):
        cells = _component_cells_from_patch(patch)
        pieces.append(
            {
                "order": int(idx),
                "color": int(next(iter(set(int(v) for v in patch.flatten() if int(v) != BG)))),
                "patch": np.array(patch, dtype=int),
                "cells": cells,
                "area": int(len(cells)),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )
    return pieces


def _frame_position(grid_h: int, grid_w: int, frame_h: int, frame_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - frame_w
    if corner == "bl":
        return grid_h - frame_h, 0
    if corner == "br":
        return grid_h - frame_h, grid_w - frame_w
    raise AssertionError(corner)


def _reserve_bbox(mask: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> None:
    rr0 = max(0, int(r0 - margin))
    cc0 = max(0, int(c0 - margin))
    rr1 = min(mask.shape[0], int(r0 + h + margin))
    cc1 = min(mask.shape[1], int(c0 + w + margin))
    mask[rr0:rr1, cc0:cc1] = True


def _place_scattered_pieces(
    grid_h: int,
    grid_w: int,
    frame_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserve = np.zeros((grid_h, grid_w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    for c in range(fc0, fc1 + 1):
        grid[fr0, c] = FRAME
        grid[fr1, c] = FRAME
    for r in range(fr0, fr1 + 1):
        grid[r, fc0] = FRAME
        grid[r, fc1] = FRAME
    _reserve_bbox(reserve, fr0, fc0, fr1 - fr0 + 1, fc1 - fc0 + 1, margin=1)

    order = sorted(range(len(patches)), key=lambda idx: -int(np.count_nonzero(patches[idx] != BG)))
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(500):
            r0 = int(rng.randint(0, grid_h - ph))
            c0 = int(rng.randint(0, grid_w - pw))
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + ph + 1)
            cc1 = min(grid_w, c0 + pw + 1)
            if bool(reserve[rr0:rr1, cc0:cc1].any()):
                continue
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            paint = patch != BG
            if bool(np.any(sub[paint] != BG)):
                continue
            sub[paint] = patch[paint]
            _reserve_bbox(reserve, r0, c0, ph, pw, margin=1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pieces = int(rng.randint(3, 5))
    target_h = int(rng.randint(max(7, n_pieces + 3), 10))
    target_w = int(rng.randint(max(7, n_pieces + 3), 11))
    if target_h * target_w < n_pieces * (n_pieces + 3):
        return None

    pieces = _build_partition(target_h, target_w, n_pieces, rng)
    if pieces is None:
        return None
    if not all(_piece_is_irregular(piece) for piece in pieces):
        return None
    colors = _assign_colors(pieces, target_h, target_w, rng)
    if colors is None:
        return None

    clean_patches: list[np.ndarray] = []
    marked_patches: list[np.ndarray] = []
    for idx, piece in enumerate(pieces, start=1):
        result = _build_piece_patch(piece, colors[idx - 1], idx, target_h, target_w, rng)
        if result is None:
            return None
        clean, marked = result
        clean_patches.append(clean)
        marked_patches.append(marked)

    solver_pieces = _solver_pieces_from_clean_patches(clean_patches)
    solution, count = _solve_ordered_tiling(solver_pieces, target_h, target_w, count_limit=2)
    if solution is None or count != 1:
        return None

    frame_h = int(target_h + 2)
    frame_w = int(target_w + 2)
    weighted_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in marked_patches)
    base = int(np.ceil(np.sqrt(weighted_area + (frame_h + 2) * (frame_w + 2))))
    grid_h = max(frame_h + 4, base + int(rng.randint(5, 9)))
    grid_w = max(frame_w + 4, base + int(rng.randint(6, 11)))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    fr0, fc0 = _frame_position(grid_h, grid_w, frame_h, frame_w, corner)
    frame_bbox = (int(fr0), int(fc0), int(fr0 + frame_h - 1), int(fc0 + frame_w - 1))

    grid = _place_scattered_pieces(grid_h, grid_w, frame_bbox, marked_patches, rng)
    if grid is None:
        return None

    piece_mask = (grid != BG) & (grid != FRAME)
    comps = _find_components(piece_mask)
    if len(comps) != n_pieces:
        return None
    extracted = [_piece_from_component(grid, comp) for comp in comps]
    extracted.sort(key=lambda piece: int(piece["order"]))
    if [int(piece["order"]) for piece in extracted] != list(range(1, n_pieces + 1)):
        return None
    if sum(int(piece["area"]) for piece in extracted) != int(target_h * target_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle after retries")
