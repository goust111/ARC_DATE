from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BODY = 1
RED = 2
GREEN = 3
CONNECTOR_COLORS = (RED, GREEN)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if seen[r0, c0] or g[r0, c0] == BG:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            bbox = _bbox_coords(coords)
            comps.append({"coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _crop_component(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _patch_to_cells(patch: np.ndarray) -> dict[tuple[int, int], int]:
    cells: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(patch != BG):
        cells[(int(r), int(c))] = int(patch[r, c])
    return cells


def _connector_positions(cells: dict[tuple[int, int], int]) -> dict[int, tuple[int, int]]:
    connectors: dict[int, tuple[int, int]] = {}
    for coord, color in cells.items():
        if color not in CONNECTOR_COLORS:
            continue
        if color in connectors:
            raise AssertionError("each connector color must appear at most once per piece")
        connectors[color] = coord
    return connectors


def _cells_to_grid(cells: dict[tuple[int, int], int]) -> np.ndarray:
    coords = list(cells)
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in cells.items():
        out[r - r0, c - c0] = color
    return out


def _dock_chain(pieces: list[dict]) -> np.ndarray:
    combo = dict(pieces[0]["cells"])
    for piece in pieces[1:]:
        combo_connectors = _connector_positions(combo)
        piece_connectors = _connector_positions(piece["cells"])
        shared = sorted(set(combo_connectors) & set(piece_connectors))
        assert len(shared) == 1, "expected exactly one shared connector color"
        color = shared[0]
        cr, cc = combo_connectors[color]
        pr, pc = piece_connectors[color]
        dr = cr - pr
        dc = cc + 1 - pc
        shifted = {(r + dr, c + dc): value for (r, c), value in piece["cells"].items()}
        for coord, value in shifted.items():
            if coord in combo and combo[coord] != value:
                raise AssertionError("unexpected overlap during docking")
        combo.update(shifted)
        combo[(cr, cc)] = BODY
        combo[(pr + dr, pc + dc)] = BODY
    return _cells_to_grid(combo)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock three blue sprites in descending area order using matching connector colors, then tight-crop the assembled chain.

    Input structure:
    - The grid has a black background and exactly three separated 4-connected sprites.
    - Every sprite is mostly blue, with one or two embedded connector pixels in non-blue colors.
    - The three sprite areas are all distinct, and the connector colors form a chain: largest shares one connector color with the middle sprite, and the middle shares another connector color with the smallest sprite.

    Rule:
    - Compute each sprite's area including connector pixels, then sort the sprites from largest to smallest.
    - Start from the largest sprite as the current assembly. For each remaining sprite, find the unique connector color shared with the current assembly.
    - Translate the new sprite so its shared-color connector lands immediately to the right of the assembly's shared-color connector on the same row, stamp it in place, and recolor both matched connector pixels to blue.
    - After all sprites are docked, crop the result to the minimal bounding box of non-black pixels.

    Key insight:
    - The assembly order is controlled by relative area, and every docking step changes the absolute location of the next usable connector, so the final layout depends on the whole chain rather than any single pairwise match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert len(components) == 3, "expected exactly three sprites"

    pieces = []
    areas = []
    for comp in components:
        patch = _crop_component(g, comp)
        cells = _patch_to_cells(patch)
        connectors = _connector_positions(cells)
        assert 1 <= len(connectors) <= 2, "each sprite must have one or two connector pixels"
        areas.append(comp["area"])
        pieces.append({"area": comp["area"], "cells": cells})

    assert len(set(areas)) == 3, "sprite areas must be distinct"
    pieces.sort(key=lambda piece: piece["area"], reverse=True)
    return _dock_chain(pieces)


def _normalize_shape(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox_coords(list(coords))
    return {(r - r0, c - c0) for r, c in coords}


def _is_connected_shape(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return len(seen) == len(coords)


def _shape_is_nonrectangular(coords: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox_coords(list(coords))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= 2 and w >= 2 and len(coords) < h * w


def _grow_single_connector_piece(area: int, side: str) -> dict | None:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if area > h * w:
            continue
        conn = (random.randrange(h), 0 if side == "left" else w - 1)
        coords = {conn}
        while len(coords) < area:
            frontier = set()
            for r, c in coords:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(tuple(frontier)))
        if len(coords) != area or not _is_connected_shape(coords):
            continue
        r0, c0, _, _ = _bbox_coords(list(coords))
        connector = (conn[0] - r0, conn[1] - c0)
        coords = {(r - r0, c - c0) for r, c in coords}
        cols = [c for _, c in coords]
        if side == "left" and connector[1] != min(cols):
            continue
        if side == "right" and connector[1] != max(cols):
            continue
        if not _shape_is_nonrectangular(coords):
            continue
        return {"coords": coords, "connectors": {RED if side == "right" else GREEN: connector}}
    return None


def _grow_bridge_piece(area: int) -> dict | None:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(3, 5)
        if area > h * w:
            continue
        left_row = random.randrange(h)
        right_row = random.randrange(h)
        coords = {(left_row, 0)}
        row = left_row
        for col in range(1, w):
            coords.add((row, col))
            remaining = (w - 1) - col
            possible_steps = []
            for step in (-1, 0, 1):
                nr = row + step
                if 0 <= nr < h and abs(right_row - nr) <= remaining:
                    possible_steps.append(step)
            if not possible_steps:
                break
            step = random.choice(possible_steps)
            if step != 0:
                row += step
                coords.add((row, col))
        if row != right_row:
            continue
        if len(coords) > area:
            continue
        while len(coords) < area:
            frontier = set()
            for r, c in coords:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(tuple(frontier)))
        if len(coords) != area or not _is_connected_shape(coords):
            continue
        r0, c0, _, _ = _bbox_coords(list(coords))
        left = (left_row - r0, 0 - c0)
        right = (right_row - r0, (w - 1) - c0)
        coords = {(r - r0, c - c0) for r, c in coords}
        if left == right or not _shape_is_nonrectangular(coords):
            continue
        return {"coords": coords, "connectors": {RED: left, GREEN: right}}
    return None


def _piece_to_patch(piece: dict) -> np.ndarray:
    coords = piece["coords"]
    r0, c0, r1, c1 = _bbox_coords(list(coords))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = BODY
    for color, (r, c) in piece["connectors"].items():
        patch[r - r0, c - c0] = color
    return patch


def _generate_piece_triplet() -> list[dict] | None:
    a_area, b_area, c_area = sorted(random.sample(range(4, 9), 3), reverse=True)
    a_piece = _grow_single_connector_piece(a_area, side="right")
    b_piece = _grow_bridge_piece(b_area)
    c_piece = _grow_single_connector_piece(c_area, side="left")
    if a_piece is None or b_piece is None or c_piece is None:
        return None
    pieces = [a_piece, b_piece, c_piece]
    for piece in pieces:
        patch = _piece_to_patch(piece)
        piece["patch"] = patch
        piece["cells"] = _patch_to_cells(patch)
        piece["area"] = int(np.count_nonzero(patch))
    return pieces


def _min_chebyshev_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            best = min(best, max(abs(r1 - r2), abs(c1 - c2)))
    return best


def _scatter_patches(patches: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    H, W = shape
    canvas = np.zeros((H, W), dtype=int)
    placed_coords: list[list[tuple[int, int]]] = []
    order = sorted(range(len(patches)), key=lambda i: int(np.count_nonzero(patches[i])), reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        cells = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]
        candidates = [(r0, c0) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            abs_cells = [(r0 + r, c0 + c) for r, c in cells]
            if any(canvas[r, c] != BG for r, c in abs_cells):
                continue
            if any(_min_chebyshev_distance(abs_cells, other) < 4 for other in placed_coords):
                continue
            for (r, c), value in zip(abs_cells, (int(patch[r1, c1]) for r1, c1 in cells)):
                canvas[r, c] = value
            placed_coords.append(abs_cells)
            placed = True
            break
        if not placed:
            return None
    return canvas


def _valid_input_structure(grid: np.ndarray) -> bool:
    comps = _find_components(grid)
    if len(comps) != 3:
        return False
    areas = sorted((comp["area"] for comp in comps), reverse=True)
    if len(set(areas)) != 3:
        return False
    for comp in comps:
        patch = _crop_component(grid, comp)
        colors = set(int(v) for v in patch.flatten()) - {BG}
        if BODY not in colors:
            return False
        connector_count = sum(color in CONNECTOR_COLORS for color in colors)
        if connector_count not in (1, 2):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pieces = _generate_piece_triplet()
    if pieces is None:
        return None

    expected = _dock_chain(sorted(pieces, key=lambda piece: piece["area"], reverse=True))
    if max(expected.shape) > 20:
        return None

    patches = [piece["patch"] for piece in pieces]
    total_w = sum(p.shape[1] for p in patches) + 11
    total_h = max(p.shape[0] for p in patches) + 10
    H = random.randint(max(18, total_h), min(24, max(18, total_h + 4)))
    W = random.randint(max(18, total_w), min(26, max(18, total_w + 4)))
    if H > 30 or W > 30:
        return None

    inp = _scatter_patches(patches, (H, W))
    if inp is None or not _valid_input_structure(inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-docking sample")
