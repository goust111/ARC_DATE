from __future__ import annotations

from collections import deque
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


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
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0))


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each board hole with the outside fragment whose shape matches it,
    recoloring the fragment to the hole border's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0 if bool(np.any(g == 0)) else _background_color(g)

    fg_components = _components(g != bg)
    assert fg_components, "grid must contain foreground objects"
    board_comp = max(fg_components, key=len)
    board_cells = set(board_comp)
    other_comps = [comp for comp in fg_components if comp is not board_comp]

    br0, bc0, br1, bc1 = _bbox(board_comp)
    board = g[br0 : br1 + 1, bc0 : bc1 + 1]
    hole_components = []
    for comp in _components(board == bg):
        if any(r == 0 or c == 0 or r == board.shape[0] - 1 or c == board.shape[1] - 1 for r, c in comp):
            continue
        hole_components.append(comp)

    holes_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for comp in hole_components:
        mask, (r0, c0) = _tight_mask(comp)
        border_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, board.shape[0], board.shape[1]):
                if board[nr, nc] != bg:
                    border_colors.add(int(board[nr, nc]))
        assert len(border_colors) == 1, "each hole must have one border color"
        key = _shape_key(mask)
        assert key not in holes_by_key, "hole shapes must be unique"
        holes_by_key[key] = {
            "abs_cells": [(int(br0 + r), int(bc0 + c)) for r, c in comp],
            "fill_color": border_colors.pop(),
        }

    pieces_by_key: dict[tuple[tuple[int, int], bytes], list[tuple[int, int]]] = {}
    for comp in other_comps:
        mask, _ = _tight_mask(comp)
        key = _shape_key(mask)
        assert key not in pieces_by_key, "piece shapes must be unique"
        pieces_by_key[key] = comp

    assert holes_by_key and len(holes_by_key) == len(pieces_by_key)

    out = g.copy()
    for key, piece in pieces_by_key.items():
        assert key in holes_by_key, "every piece must match a hole"
        for r, c in piece:
            out[r, c] = bg

    for key, hole in holes_by_key.items():
        for r, c in hole["abs_cells"]:
            out[r, c] = int(hole["fill_color"])
    return out


# --------------- generator ---------------

BG = 0
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
BOARD_H_RANGE = (15, 21)
BOARD_W_RANGE = (15, 21)
HOLE_COUNT_RANGE = (2, 4)
SHAPE_DIM_RANGE = (1, 5)
SHAPE_AREA_MAX = 8
GEN_ATTEMPTS = 400
HOLE_PLACE_TRIES = 120
PIECE_PLACE_TRIES = 200


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _grow_shape(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))
    while int(mask.sum()) < area:
        frontier = {
            (r, c)
            for r, c in frontier
            if 0 <= r < height and 0 <= c < width and not mask[r, c]
        }
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True
        for nr, nc in _neighbors4(rr, cc, height, width):
            if not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    cropped, _ = _tight_mask(coords)
    return cropped


def _random_shape() -> np.ndarray:
    for _ in range(80):
        h = random.randint(*SHAPE_DIM_RANGE)
        w = random.randint(*SHAPE_DIM_RANGE)
        max_area = min(int(h * w), SHAPE_AREA_MAX)
        min_area = 2 if max_area >= 2 else 1
        area = random.randint(min_area, max_area)
        shape = _grow_shape(h, w, area)

        # Bias toward visually distinct fragments without forbidding bars/blocks.
        full_rect = int(shape.sum()) == int(shape.shape[0] * shape.shape[1])
        if full_rect and area >= 4 and random.random() < 0.7:
            continue

        for _ in range(random.randint(0, 3)):
            shape = np.rot90(shape)
        if random.random() < 0.5:
            shape = np.fliplr(shape)
        return shape.copy()
    raise RuntimeError("failed to sample a fragment shape")


def _sample_shapes(count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen = set()
    irregular_count = 0
    while len(shapes) < count:
        shape = _random_shape()
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        if int(shape.sum()) < int(shape.shape[0] * shape.shape[1]):
            irregular_count += 1
        shapes.append(shape)
    if irregular_count == 0:
        return _sample_shapes(count)
    return shapes


def _can_place_hole(clearance: np.ndarray, top: int, left: int, shape: np.ndarray) -> bool:
    h, w = shape.shape
    r0 = top - 2
    c0 = left - 2
    r1 = top + h + 1
    c1 = left + w + 1
    if r0 < 0 or c0 < 0 or r1 >= clearance.shape[0] or c1 >= clearance.shape[1]:
        return False
    return not bool(clearance[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_hole_clearance(clearance: np.ndarray, top: int, left: int, shape: np.ndarray) -> None:
    h, w = shape.shape
    clearance[top - 2 : top + h + 2, left - 2 : left + w + 2] = True


def _place_holes(board_h: int, board_w: int, shapes: list[np.ndarray]) -> list[tuple[int, int]]:
    clearance = np.zeros((board_h, board_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: -(shapes[i].shape[0] * shapes[i].shape[1]))
    for idx in order:
        shape = shapes[idx]
        h, w = shape.shape
        candidates = []
        for top in range(2, board_h - h - 1):
            for left in range(2, board_w - w - 1):
                if _can_place_hole(clearance, top, left, shape):
                    candidates.append((int(top), int(left)))
        if not candidates:
            raise ValueError("board too small for holes")
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        _mark_hole_clearance(clearance, top, left, shape)
    return [pos for pos in placements if pos is not None]


def _paint_board(
    grid: np.ndarray,
    board_top: int,
    board_left: int,
    board_h: int,
    board_w: int,
    board_color: int,
    hole_shapes: list[np.ndarray],
    hole_pos: list[tuple[int, int]],
    border_colors: list[int],
) -> list[dict]:
    grid[board_top : board_top + board_h, board_left : board_left + board_w] = int(board_color)
    hole_info: list[dict] = []
    for shape, (top, left), border_color in zip(hole_shapes, hole_pos, border_colors):
        h, w = shape.shape
        local = np.zeros((h + 2, w + 2), dtype=bool)
        local[1 : h + 1, 1 : w + 1] = shape
        ring = _dilate8(local) & (~local)
        for r, c in np.argwhere(ring):
            grid[board_top + top - 1 + r, board_left + left - 1 + c] = int(border_color)
        abs_cells = []
        for r, c in np.argwhere(shape):
            rr = int(board_top + top + r)
            cc = int(board_left + left + c)
            grid[rr, cc] = BG
            abs_cells.append((rr, cc))
        hole_info.append({"shape": shape, "cells": abs_cells, "fill": int(border_color)})
    return hole_info


def _can_place_piece(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + height + 1)
    c1 = min(occupied.shape[1], left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_piece(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + height + 1)
    c1 = min(occupied.shape[1], left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _place_pieces(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    shapes: list[np.ndarray],
    piece_colors: list[int],
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    br0, bc0, br1, bc1 = board_bbox
    occupied[max(0, br0 - 1) : min(h, br1 + 2), max(0, bc0 - 1) : min(w, bc1 + 2)] = True

    piece_cells: list[list[tuple[int, int]]] = []
    order = list(range(len(shapes)))
    random.shuffle(order)
    placed_by_index: dict[int, list[tuple[int, int]]] = {}

    for idx in order:
        shape = shapes[idx]
        ph, pw = shape.shape
        candidates = []
        for top in range(0, h - ph + 1):
            for left in range(0, w - pw + 1):
                if _can_place_piece(occupied, top, left, ph, pw):
                    candidates.append((int(top), int(left)))
        if not candidates:
            raise ValueError("not enough room for pieces")
        top, left = random.choice(candidates)
        cells: list[tuple[int, int]] = []
        for r, c in np.argwhere(shape):
            rr = int(top + r)
            cc = int(left + c)
            grid[rr, cc] = int(piece_colors[idx])
            cells.append((rr, cc))
        placed_by_index[idx] = cells
        _mark_piece(occupied, top, left, ph, pw)

    for idx in range(len(shapes)):
        piece_cells.append(placed_by_index[idx])
    return piece_cells


def _expected_output(grid: np.ndarray, holes: list[dict], pieces: list[list[tuple[int, int]]]) -> np.ndarray:
    out = grid.copy()
    for comp in pieces:
        for r, c in comp:
            out[r, c] = BG
    for hole in holes:
        for r, c in hole["cells"]:
            out[r, c] = int(hole["fill"])
    return out


def _generate_attempt() -> np.ndarray | None:
    hole_count = random.randint(*HOLE_COUNT_RANGE)
    shapes = _sample_shapes(hole_count)

    board_h = random.randint(*BOARD_H_RANGE)
    board_w = random.randint(*BOARD_W_RANGE)
    if max(shape.shape[0] for shape in shapes) + 4 >= board_h:
        return None
    if max(shape.shape[1] for shape in shapes) + 4 >= board_w:
        return None

    grid_h = random.randint(max(GRID_H_RANGE[0], board_h + 5), GRID_H_RANGE[1] - 1)
    grid_w = random.randint(max(GRID_W_RANGE[0], board_w + 5), GRID_W_RANGE[1] - 1)

    board_top = random.randint(1, grid_h - board_h - 1)
    board_left = random.randint(1, grid_w - board_w - 1)

    try:
        hole_positions = _place_holes(board_h, board_w, shapes)
    except ValueError:
        return None

    palette = random.sample(range(1, 10), 1 + 2 * hole_count)
    board_color = int(palette[0])
    border_colors = [int(x) for x in palette[1 : 1 + hole_count]]
    piece_colors = [int(x) for x in palette[1 + hole_count :]]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    holes = _paint_board(
        grid,
        board_top,
        board_left,
        board_h,
        board_w,
        board_color,
        shapes,
        hole_positions,
        border_colors,
    )

    board_bbox = (board_top, board_left, board_top + board_h - 1, board_left + board_w - 1)
    try:
        pieces = _place_pieces(grid, board_bbox, shapes, piece_colors)
    except ValueError:
        return None

    expected = _expected_output(grid, holes, pieces)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
