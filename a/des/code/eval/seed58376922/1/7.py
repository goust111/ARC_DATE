import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _normalize_cells(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[Tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in cells]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, height, width


def _extract_pieces(grid: np.ndarray) -> List[Dict[str, object]]:
    board = np.asarray(grid[:-1], dtype=int)
    h, w = board.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: List[Dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(board[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(board[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            norm, ph, pw = _normalize_cells(cells)
            pieces.append(
                {
                    "color": color,
                    "cells": norm,
                    "h": ph,
                    "w": pw,
                    "area": len(norm),
                }
            )

    pieces.sort(key=lambda piece: (-int(piece["area"]), int(piece["color"])))
    return pieces


def _factor_pairs(area: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        out.append((h, w))
        if h != w:
            out.append((w, h))
    out.sort(key=lambda dims: (abs(dims[0] - dims[1]), dims[0], dims[1]))
    return out


def _build_placements(piece: Dict[str, object], height: int, width: int) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
    total = height * width
    placements: List[Tuple[int, int, int]] = []
    covers: List[List[int]] = [[] for _ in range(total)]
    cells: Sequence[Tuple[int, int]] = piece["cells"]  # type: ignore[assignment]
    ph = int(piece["h"])
    pw = int(piece["w"])

    for top in range(height - ph + 1):
        for left in range(width - pw + 1):
            bits = 0
            for r, c in cells:
                idx = (r + top) * width + (c + left)
                bits |= 1 << idx
            placement_idx = len(placements)
            placements.append((bits, top, left))
            cur = bits
            while cur:
                lsb = cur & -cur
                cell_idx = lsb.bit_length() - 1
                covers[cell_idx].append(placement_idx)
                cur ^= lsb
    return placements, covers


def _solve_packing(pieces: Sequence[Dict[str, object]], limit: int = 2) -> List[Tuple[int, int, Tuple[Tuple[int, int], ...]]]:
    area = sum(int(piece["area"]) for piece in pieces)
    solutions: List[Tuple[int, int, Tuple[Tuple[int, int], ...]]] = []

    for height, width in _factor_pairs(area):
        if any(int(piece["h"]) > height or int(piece["w"]) > width for piece in pieces):
            continue

        placement_data = [_build_placements(piece, height, width) for piece in pieces]
        if any(not placements for placements, _covers in placement_data):
            continue

        placements_by_piece = [item[0] for item in placement_data]
        covers_by_piece = [item[1] for item in placement_data]
        full_mask = (1 << (height * width)) - 1
        shifts: Dict[int, Tuple[int, int]] = {}

        def search(occupied: int, remaining: Tuple[int, ...]) -> None:
            if len(solutions) >= limit:
                return
            if not remaining:
                solutions.append((height, width, tuple(shifts[i] for i in range(len(pieces)))))
                return

            missing = (~occupied) & full_mask
            if missing == 0:
                return
            target_bit = missing & -missing
            target_idx = target_bit.bit_length() - 1

            choice_piece = None
            choice_options: List[int] = []
            choice_key = None
            for piece_idx in remaining:
                legal: List[int] = []
                for placement_idx in covers_by_piece[piece_idx][target_idx]:
                    bits, _top, _left = placements_by_piece[piece_idx][placement_idx]
                    if bits & occupied:
                        continue
                    legal.append(placement_idx)
                if not legal:
                    continue
                key = (len(legal), -int(pieces[piece_idx]["area"]), piece_idx)
                if choice_key is None or key < choice_key:
                    choice_key = key
                    choice_piece = piece_idx
                    choice_options = legal

            if choice_piece is None:
                return

            next_remaining = tuple(idx for idx in remaining if idx != choice_piece)
            for placement_idx in choice_options:
                bits, top, left = placements_by_piece[choice_piece][placement_idx]
                shifts[choice_piece] = (top, left)
                search(occupied | bits, next_remaining)
                del shifts[choice_piece]
                if len(solutions) >= limit:
                    return

        search(0, tuple(range(len(pieces))))
        if len(solutions) >= limit:
            return solutions

    return solutions


def _assembled_masks(
    pieces: Sequence[Dict[str, object]], solution: Tuple[int, int, Tuple[Tuple[int, int], ...]]
) -> Tuple[int, int, Dict[int, np.ndarray]]:
    height, width, shifts = solution
    masks: Dict[int, np.ndarray] = {}
    for piece, (top, left) in zip(pieces, shifts):
        mask = np.zeros((height, width), dtype=bool)
        for r, c in piece["cells"]:  # type: ignore[index]
            mask[top + r, left + c] = True
        masks[int(piece["color"])] = mask
    return height, width, masks


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _compose_output(height: int, width: int, masks: Dict[int, np.ndarray], layer_order: Sequence[int]) -> np.ndarray:
    out = np.full((height + 2, width + 2), BG, dtype=int)
    for color in layer_order:
        mask = masks[int(color)]
        expanded = _dilate8(np.pad(mask, 1, constant_values=False))
        out[expanded] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome fragments into their original rectangle, then
    expand each fragment by one pixel and resolve overlaps by the bottom legend order.
    """
    g = np.asarray(grid, dtype=int)
    layer_order = [int(v) for v in g[-1].tolist() if int(v) != BG]
    assert layer_order, "legend row must contain the fragment colors"

    pieces = _extract_pieces(g)
    assert len(pieces) == len(layer_order), "legend and fragment counts must match"
    assert {int(piece["color"]) for piece in pieces} == set(layer_order), "legend colors must match fragment colors"

    solutions = _solve_packing(pieces, limit=2)
    assert solutions, "fragments must tile a rectangle"

    height, width, masks = _assembled_masks(pieces, solutions[0])
    return _compose_output(height, width, masks, layer_order)


def _random_targets(area: int, count: int, min_size: int, rng: random.Random) -> List[int]:
    sizes = [min_size] * count
    for _ in range(area - min_size * count):
        sizes[rng.randrange(count)] += 1
    rng.shuffle(sizes)
    return sizes


def _pick_seeds(height: int, width: int, count: int, rng: random.Random) -> List[Tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(80):
        seeds = rng.sample(cells, count)
        if all(abs(r1 - r2) + abs(c1 - c2) >= 2 for i, (r1, c1) in enumerate(seeds) for r2, c2 in seeds[i + 1 :]):
            return seeds
    return rng.sample(cells, count)


def _partition_rectangle(height: int, width: int, count: int, rng: random.Random) -> List[set[Tuple[int, int]]] | None:
    min_size = 4
    if height * width < count * min_size:
        return None

    seeds = _pick_seeds(height, width, count, rng)
    regions = [{seed} for seed in seeds]
    unassigned = {(r, c) for r in range(height) for c in range(width)} - set(seeds)
    targets = _random_targets(height * width, count, min_size, rng)
    frontiers: List[set[Tuple[int, int]]] = []
    for seed in seeds:
        r, c = seed
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) in unassigned:
                frontier.add((nr, nc))
        frontiers.append(frontier)

    while unassigned:
        choices: List[int] = []
        for idx, frontier in enumerate(frontiers):
            if not frontier:
                continue
            deficit = max(targets[idx] - len(regions[idx]), 0)
            choices.extend([idx] * (1 + deficit))
        if not choices:
            return None

        region_idx = rng.choice(choices)
        cell = rng.choice(tuple(frontiers[region_idx]))
        regions[region_idx].add(cell)
        unassigned.remove(cell)

        for frontier in frontiers:
            frontier.discard(cell)

        r, c = cell
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) in unassigned:
                frontiers[region_idx].add((nr, nc))

    return regions


def _piece_from_region(region: set[Tuple[int, int]], color: int) -> Dict[str, object]:
    norm, ph, pw = _normalize_cells(region)
    return {"color": color, "cells": norm, "h": ph, "w": pw, "area": len(norm)}


def _piece_complexity_ok(pieces: Sequence[Dict[str, object]]) -> bool:
    non_rect = sum(int(piece["area"]) < int(piece["h"]) * int(piece["w"]) for piece in pieces)
    if non_rect < 2:
        return False
    return True


def _scatter_pieces(
    pieces: Sequence[Dict[str, object]], upper_h: int, upper_w: int, rng: random.Random
) -> List[Tuple[int, int]] | None:
    occupied = np.zeros((upper_h, upper_w), dtype=bool)
    forbidden = np.zeros((upper_h, upper_w), dtype=bool)
    placements: List[Tuple[int, int] | None] = [None] * len(pieces)
    order = sorted(range(len(pieces)), key=lambda idx: (-int(pieces[idx]["area"]), int(pieces[idx]["color"])))

    for piece_idx in order:
        piece = pieces[piece_idx]
        ph = int(piece["h"])
        pw = int(piece["w"])
        candidates: List[Tuple[int, int]] = []
        attempts = max(60, upper_h * upper_w)
        for _ in range(attempts):
            top = rng.randint(0, upper_h - ph)
            left = rng.randint(0, upper_w - pw)
            if all(not forbidden[top + r, left + c] for r, c in piece["cells"]):  # type: ignore[index]
                candidates.append((top, left))
        if not candidates:
            return None

        top, left = rng.choice(candidates)
        placements[piece_idx] = (top, left)
        for r, c in piece["cells"]:  # type: ignore[index]
            occupied[top + r, left + c] = True
        forbidden = _dilate8(occupied)

    return [(int(top), int(left)) for top, left in placements if top is not None and left is not None]


def _render_input(
    pieces: Sequence[Dict[str, object]],
    scatter: Sequence[Tuple[int, int]],
    layer_order: Sequence[int],
    upper_h: int,
    upper_w: int,
    legend_start: int,
) -> np.ndarray:
    grid = np.full((upper_h + 2, upper_w), BG, dtype=int)
    for piece, (top, left) in zip(pieces, scatter):
        color = int(piece["color"])
        for r, c in piece["cells"]:  # type: ignore[index]
            grid[top + r, left + c] = color
    for idx, color in enumerate(layer_order):
        grid[-1, legend_start + idx] = int(color)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(3, 5)
    rect_h = rng.randint(5, 8)
    rect_w = rng.randint(5, 8)
    if rect_h * rect_w < piece_count * 4:
        return None

    regions = _partition_rectangle(rect_h, rect_w, piece_count, rng)
    if regions is None:
        return None

    colors = rng.sample(list(range(1, 10)), piece_count)
    pieces = [_piece_from_region(region, color) for region, color in zip(regions, colors)]
    pieces.sort(key=lambda piece: (-int(piece["area"]), int(piece["color"])))
    if not _piece_complexity_ok(pieces):
        return None

    solutions = _solve_packing(pieces, limit=2)
    if len(solutions) != 1 or solutions[0][0] != rect_h or solutions[0][1] != rect_w:
        return None

    upper_h = max(rect_h + rng.randint(6, 10), 13)
    upper_w = max(rect_w + rng.randint(7, 12), 14)
    scatter = _scatter_pieces(pieces, upper_h, upper_w, rng)
    if scatter is None:
        return None

    layer_order = colors[:]
    rng.shuffle(layer_order)
    legend_start = rng.randint(0, upper_w - len(layer_order))
    return _render_input(pieces, scatter, layer_order, upper_h, upper_w, legend_start)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
