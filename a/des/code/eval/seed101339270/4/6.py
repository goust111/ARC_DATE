from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_RETRY_LIMIT = 400
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
FRAME_H_RANGE = (9, 15)
FRAME_W_RANGE = (9, 15)
PIECE_COUNT_RANGE = (2, 4)
PIECE_GREY_SIZE_RANGE = (2, 8)
PIECE_SOURCE_RETRIES = 200
PIECE_GROW_RETRIES = 40
ANCHOR_SEPARATION = 4


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = int(r) + int(dr)
        nc = int(c) + int(dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bool(passable[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(passable[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(passable[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert scattered mono-grey fragments into a large outlined target by matching
    same-colored interface pixels and translating each fragment so the two
    markers become orthogonally adjacent rather than overlapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"
    frame = max(comps, key=len)
    frame_mask = np.zeros((h, w), dtype=bool)
    for r, c in frame:
        frame_mask[r, c] = True

    frame_values = g[frame_mask]
    grey = _most_common_color(frame_values)

    exterior = _flood_border(~frame_mask)
    interior = (~frame_mask) & (~exterior)
    assert bool(interior.any()), "expected a non-empty outline interior"

    wall_markers: dict[int, tuple[int, int, int, int]] = {}
    for r, c in frame:
        color = int(g[r, c])
        if color == bg or color == grey:
            continue
        inside = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if bool(interior[nr, nc])]
        assert len(inside) == 1, "wall marker must have a unique inward neighbor"
        nr, nc = inside[0]
        assert color not in wall_markers, "wall marker colors must be unique"
        wall_markers[color] = (int(r), int(c), int(nr - r), int(nc - c))

    assert wall_markers, "expected colored wall markers"

    piece_mask = (g != bg) & (~frame_mask)
    piece_components = _components(piece_mask)
    assert piece_components, "expected scattered pieces"

    out = g.copy()
    out[piece_mask] = bg
    occupied = frame_mask.copy()

    for comp in piece_components:
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) != int(grey)]
        assert len(marker_cells) == 1, "each piece must have exactly one colored marker"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        assert marker_color in wall_markers, "piece marker must match a wall marker"

        wall_r, wall_c, dr, dc = wall_markers[marker_color]
        dst_marker_r = int(wall_r + dr)
        dst_marker_c = int(wall_c + dc)
        shift_r = int(dst_marker_r - marker_r)
        shift_c = int(dst_marker_c - marker_c)

        for r, c in comp:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            assert 0 <= rr < h and 0 <= cc < w, "translated piece left the grid"
            assert not bool(frame_mask[rr, cc]), "piece may not overlap the outline"
            assert not bool(occupied[rr, cc]), "translated pieces may not overlap"
            occupied[rr, cc] = True
            out[rr, cc] = int(g[r, c])

    return out


def _compute_boundary(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(filled[r, c]):
                continue
            if any(not bool(filled[nr, nc]) for nr, nc in _neighbors4(r, c, h, w)):
                boundary[r, c] = True
    return boundary


def _mark_with_margin(mask: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in coords:
        for dr, dc in DIRS8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True
        mask[r, c] = True


def _grow_piece(
    allowed: np.ndarray, marker_cell: tuple[int, int], target_size: int
) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    mr, mc = marker_cell
    starters = [(nr, nc) for nr, nc in _neighbors4(mr, mc, h, w) if bool(allowed[nr, nc])]
    if not starters:
        return None

    for _ in range(PIECE_GROW_RETRIES):
        seed = random.choice(starters)
        grey_cells = {seed}
        frontier = set()
        for nr, nc in _neighbors4(seed[0], seed[1], h, w):
            if bool(allowed[nr, nc]) and (nr, nc) not in grey_cells:
                frontier.add((nr, nc))

        while len(grey_cells) < target_size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in grey_cells or not bool(allowed[cell]):
                continue
            grey_cells.add(cell)
            for nr, nc in _neighbors4(cell[0], cell[1], h, w):
                if bool(allowed[nr, nc]) and (nr, nc) not in grey_cells:
                    frontier.add((nr, nc))

        if len(grey_cells) == target_size:
            return [(int(r), int(c)) for r, c in grey_cells]
    return None


def _build_filled_shape(h: int, w: int) -> np.ndarray | None:
    filled = np.zeros((h, w), dtype=bool)

    max_frame_h = min(FRAME_H_RANGE[1], h - 8)
    max_frame_w = min(FRAME_W_RANGE[1], w - 8)
    if max_frame_h < FRAME_H_RANGE[0] or max_frame_w < FRAME_W_RANGE[0]:
        return None

    frame_h = random.randint(FRAME_H_RANGE[0], max_frame_h)
    frame_w = random.randint(FRAME_W_RANGE[0], max_frame_w)
    r0 = random.randint(3, h - frame_h - 4)
    c0 = random.randint(3, w - frame_w - 4)
    filled[r0 : r0 + frame_h, c0 : c0 + frame_w] = True

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    desired_bumps = random.randint(0, 2)
    for side in sides[:desired_bumps]:
        if side == "top":
            max_depth = min(4, r0 - 2)
            if max_depth < 2:
                continue
            depth = random.randint(2, max_depth)
            span = random.randint(3, frame_w - 2)
            start = random.randint(c0 + 1, c0 + frame_w - span - 1)
            filled[r0 - depth : r0, start : start + span] = True
        elif side == "bottom":
            max_depth = min(4, h - (r0 + frame_h) - 2)
            if max_depth < 2:
                continue
            depth = random.randint(2, max_depth)
            span = random.randint(3, frame_w - 2)
            start = random.randint(c0 + 1, c0 + frame_w - span - 1)
            filled[r0 + frame_h : r0 + frame_h + depth, start : start + span] = True
        elif side == "left":
            max_depth = min(4, c0 - 2)
            if max_depth < 2:
                continue
            depth = random.randint(2, max_depth)
            span = random.randint(3, frame_h - 2)
            start = random.randint(r0 + 1, r0 + frame_h - span - 1)
            filled[start : start + span, c0 - depth : c0] = True
        else:
            max_depth = min(4, w - (c0 + frame_w) - 2)
            if max_depth < 2:
                continue
            depth = random.randint(2, max_depth)
            span = random.randint(3, frame_h - 2)
            start = random.randint(r0 + 1, r0 + frame_h - span - 1)
            filled[start : start + span, c0 + frame_w : c0 + frame_w + depth] = True

    boundary = _compute_boundary(filled)
    interior = filled & (~boundary)
    if int(interior.sum()) < 18:
        return None
    return filled


def _choose_anchors(
    boundary: np.ndarray, interior: np.ndarray, piece_count: int
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    h, w = boundary.shape
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {d: [] for d in DIRS4}
    for r in range(h):
        for c in range(w):
            if not bool(boundary[r, c]):
                continue
            inside = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if bool(interior[nr, nc])]
            if len(inside) != 1:
                continue
            nr, nc = inside[0]
            candidates[(int(nr - r), int(nc - c))].append((int(r), int(c)))

    if any(not candidates[d] for d in DIRS4):
        return []

    directions = random.sample(list(DIRS4), piece_count)
    anchors: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    marker_cells: list[tuple[int, int]] = []

    for direction in directions:
        options = candidates[direction][:]
        random.shuffle(options)
        chosen = None
        for wall_r, wall_c in options:
            marker = (int(wall_r + direction[0]), int(wall_c + direction[1]))
            if any(abs(wall_r - ar) + abs(wall_c - ac) < ANCHOR_SEPARATION for (ar, ac), _, _ in anchors):
                continue
            if any(abs(marker[0] - mr) + abs(marker[1] - mc) < ANCHOR_SEPARATION for mr, mc in marker_cells):
                continue
            chosen = ((int(wall_r), int(wall_c)), direction, marker)
            break
        if chosen is None:
            return []
        anchors.append(chosen)
        marker_cells.append(chosen[2])

    return anchors


def _scatter_piece(
    grid: np.ndarray,
    piece_cells: list[tuple[int, int, int]],
    forbidden: np.ndarray,
) -> bool:
    h, w = grid.shape
    coords = [(r, c) for r, c, _ in piece_cells]
    r0, c0, r1, c1 = _bbox(coords)
    patch_h = int(r1 - r0 + 1)
    patch_w = int(c1 - c0 + 1)
    rel_cells = [(int(r - r0), int(c - c0), int(color)) for r, c, color in piece_cells]

    if patch_h + 2 > h or patch_w + 2 > w:
        return False

    for _ in range(PIECE_SOURCE_RETRIES):
        top = random.randint(1, h - patch_h - 1)
        left = random.randint(1, w - patch_w - 1)
        placed = [(int(top + rr), int(left + cc), int(color)) for rr, cc, color in rel_cells]
        if any(bool(forbidden[r, c]) for r, c, _ in placed):
            continue
        for r, c, color in placed:
            grid[r, c] = int(color)
        _mark_with_margin(forbidden, [(r, c) for r, c, _ in placed])
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    filled = _build_filled_shape(h, w)
    if filled is None:
        return None

    boundary = _compute_boundary(filled)
    interior = filled & (~boundary)
    piece_count = random.randint(*PIECE_COUNT_RANGE)
    anchors = _choose_anchors(boundary, interior, piece_count)
    if len(anchors) != piece_count:
        return None

    colors = random.sample(MARKER_COLORS, piece_count)
    reserved_markers = np.zeros((h, w), dtype=bool)
    for _, _, marker in anchors:
        reserved_markers[marker] = True

    placed_pieces: list[dict[str, object]] = []
    used_inside = reserved_markers.copy()

    for color, (wall, direction, marker) in zip(colors, anchors):
        allowed = interior & (~used_inside)
        allowed[marker] = False
        max_grey = min(PIECE_GREY_SIZE_RANGE[1], int(allowed.sum()))
        if max_grey < PIECE_GREY_SIZE_RANGE[0]:
            return None
        grey_target = random.randint(PIECE_GREY_SIZE_RANGE[0], max_grey)
        grey_cells = _grow_piece(allowed, marker, grey_target)
        if grey_cells is None:
            return None
        for r, c in grey_cells:
            used_inside[r, c] = True
        placed_pieces.append(
            {
                "wall": wall,
                "direction": direction,
                "marker": marker,
                "color": int(color),
                "grey_cells": grey_cells,
            }
        )

    out = np.full((h, w), BG, dtype=int)
    out[boundary] = GREY
    for piece in placed_pieces:
        wall_r, wall_c = piece["wall"]
        out[wall_r, wall_c] = int(piece["color"])
        marker_r, marker_c = piece["marker"]
        out[marker_r, marker_c] = int(piece["color"])
        for r, c in piece["grey_cells"]:
            out[r, c] = GREY

    inp = np.full((h, w), BG, dtype=int)
    inp[boundary] = GREY
    for piece in placed_pieces:
        wall_r, wall_c = piece["wall"]
        inp[wall_r, wall_c] = int(piece["color"])

    forbidden = filled.copy()
    _mark_with_margin(forbidden, list(zip(*np.where(boundary))))  # type: ignore[arg-type]
    for piece in placed_pieces:
        marker_r, marker_c = piece["marker"]
        cells = [(int(marker_r), int(marker_c), int(piece["color"]))]
        cells.extend((int(r), int(c), GREY) for r, c in piece["grey_cells"])
        if not _scatter_piece(inp, cells, forbidden):
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
