from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SIDE_TO_COLOR = {"top": RED, "bottom": BLUE, "left": GREEN, "right": YELLOW}
COLOR_TO_SIDE = {color: side for side, color in SIDE_TO_COLOR.items()}
SIDES = ("top", "bottom", "left", "right")
GRID_MIN = 10
GRID_MAX = 16
AREA_MIN = 3
AREA_MAX = 6
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _sorted_coords(coords) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return _sorted_coords((r - r0, c - c0) for r, c in coords)


def _translate(coords, dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return _sorted_coords((r + dr, c + dc) for r, c in coords)


def _reflect_shape(coords, axis_kind: str) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if axis_kind == "horizontal":
        return _sorted_coords((h - 1 - r, c) for r, c in norm)
    if axis_kind == "vertical":
        return _sorted_coords((r, w - 1 - c) for r, c in norm)
    raise ValueError(axis_kind)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _build_frame(h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, 1 : w - 1] = RED
    grid[h - 1, 1 : w - 1] = BLUE
    grid[1 : h - 1, 0] = GREEN
    grid[1 : h - 1, w - 1] = YELLOW
    return grid


def _parse_input(grid: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30
    assert g[0, 0] == g[0, w - 1] == g[h - 1, 0] == g[h - 1, w - 1] == BG
    assert np.all(g[0, 1 : w - 1] == RED)
    assert np.all(g[h - 1, 1 : w - 1] == BLUE)
    assert np.all(g[1 : h - 1, 0] == GREEN)
    assert np.all(g[1 : h - 1, w - 1] == YELLOW)

    interior = g[1 : h - 1, 1 : w - 1]
    non_bg = np.argwhere(interior != BG)
    assert non_bg.size > 0

    shape_mask = g == GREY
    comps = _components(shape_mask)
    assert len(comps) == 1
    shape = _sorted_coords(comps[0])
    assert AREA_MIN <= len(shape) <= AREA_MAX

    shape_set = set(shape)
    marker_cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color not in COLOR_TO_SIDE:
                continue
            if any((nr, nc) in shape_set for nr, nc in _neighbors4(r, c)):
                marker_cells.append((r, c))
    assert len(marker_cells) == 1
    marker = marker_cells[0]
    side = COLOR_TO_SIDE[int(g[marker])]

    assert len(non_bg) == len(shape) + 1
    return shape, marker, side


def _move_to_wall(coords: tuple[tuple[int, int], ...], side: str, h: int, w: int) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(coords)
    if side == "top":
        return _translate(coords, 1 - r0, 0)
    if side == "bottom":
        return _translate(coords, (h - 2) - r1, 0)
    if side == "left":
        return _translate(coords, 0, 1 - c0)
    if side == "right":
        return _translate(coords, 0, (w - 2) - c1)
    raise ValueError(side)


def _mirror_across_middle(coords: tuple[tuple[int, int], ...], side: str, h: int, w: int) -> tuple[tuple[int, int], ...]:
    mirrored = []
    for r, c in coords:
        if side in ("top", "bottom"):
            rr, cc = h - 1 - r, c
        else:
            rr, cc = r, w - 1 - c
        if 1 <= rr < h - 1 and 1 <= cc < w - 1:
            mirrored.append((rr, cc))
    return _sorted_coords(mirrored)


def _opposite_wall_translation(coords: tuple[tuple[int, int], ...], side: str, h: int, w: int) -> tuple[tuple[int, int], ...]:
    if side == "top":
        return _move_to_wall(coords, "bottom", h, w)
    if side == "bottom":
        return _move_to_wall(coords, "top", h, w)
    if side == "left":
        return _move_to_wall(coords, "right", h, w)
    if side == "right":
        return _move_to_wall(coords, "left", h, w)
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the grey shape to the wall selected by the marker color, then mirror the moved shape across the matching interior mid-axis.

    Input structure:
    - A black grid has a one-cell frame whose top, bottom, left, and right sides are red, blue, green, and yellow; the four corners stay black.
    - Inside the frame there is exactly one 4-connected grey shape and exactly one non-grey marker pixel.
    - The marker pixel is 4-neighbor-adjacent to the grey shape, and its color is one of the frame-side colors.

    Rule:
    - Use the marker color to choose the matching frame side: red->top, blue->bottom, green->left, yellow->right.
    - Translate the whole grey shape until it touches that side's inner wall, while keeping the shape unchanged.
    - Remove the marker pixel.
    - Mirror the translated grey shape across the interior middle axis parallel to the chosen side, and recolor the mirrored copy orange.
    - Keep only mirrored pixels that stay strictly inside the frame; preserve the frame and black background.

    Key insight:
    - One visible cue, the marker color, simultaneously chooses the translation direction and the orientation of the mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    shape, _marker, side = _parse_input(g)

    moved = _move_to_wall(shape, side, h, w)
    mirrored = _mirror_across_middle(moved, side, h, w)

    out = _build_frame(h, w)
    for r, c in moved:
        out[r, c] = GREY
    for r, c in mirrored:
        if out[r, c] == BG:
            out[r, c] = ORANGE
        elif out[r, c] == GREY:
            out[r, c] = GREY
    return out


def _random_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        cells.add((nr, nc))
    return _normalize(cells)


def _shape_is_good(shape: tuple[tuple[int, int], ...], side: str) -> bool:
    r0, c0, r1, c1 = _bbox(shape)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h > 4 or w > 4:
        return False
    if side in ("top", "bottom"):
        if h < 2:
            return False
        if _reflect_shape(shape, "horizontal") == _normalize(shape):
            return False
    else:
        if w < 2:
            return False
        if _reflect_shape(shape, "vertical") == _normalize(shape):
            return False
    return True


def _candidate_markers(shape_abs: tuple[tuple[int, int], ...], h: int, w: int):
    shape_set = set(shape_abs)
    out = []
    for r, c in shape_abs:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in shape_set or not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                continue
            dirs = set()
            if (nr + 1, nc) in shape_set:
                dirs.add("top")
            if (nr - 1, nc) in shape_set:
                dirs.add("bottom")
            if (nr, nc + 1) in shape_set:
                dirs.add("left")
            if (nr, nc - 1) in shape_set:
                dirs.add("right")
            if len(dirs) == 1:
                out.append((nr, nc, next(iter(dirs))))
    unique = {}
    for r, c, side in out:
        unique[(r, c)] = side
    return [(r, c, side) for (r, c), side in unique.items()]


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        shape, marker, side = _parse_input(inp)
    except Exception:
        return False

    h, w = inp.shape
    moved = _move_to_wall(shape, side, h, w)
    mirrored = _mirror_across_middle(moved, side, h, w)
    if len(mirrored) != len(shape):
        return False
    if set(moved) & set(mirrored):
        return False

    moved_bbox = _bbox(moved)
    if side == "top" and moved_bbox[0] != 1:
        return False
    if side == "bottom" and moved_bbox[2] != h - 2:
        return False
    if side == "left" and moved_bbox[1] != 1:
        return False
    if side == "right" and moved_bbox[3] != w - 2:
        return False

    opposite = _opposite_wall_translation(moved, side, h, w)
    if mirrored == opposite:
        return False

    shape_set = set(shape)
    marker_side = None
    mr, mc = marker
    for nr, nc in _neighbors4(mr, mc):
        if (nr, nc) in shape_set:
            if nr == mr - 1:
                marker_side = "bottom"
            elif nr == mr + 1:
                marker_side = "top"
            elif nc == mc - 1:
                marker_side = "right"
            elif nc == mc + 1:
                marker_side = "left"
            break
    if marker_side is None:
        return False

    grey_cells = _sorted_coords(np.argwhere(out == GREY))
    orange_cells = _sorted_coords(np.argwhere(out == ORANGE))
    if grey_cells != moved or orange_cells != mirrored:
        return False

    if any(int(out[r, c]) in COLOR_TO_SIDE for r in range(1, h - 1) for c in range(1, w - 1)):
        return False

    if not np.array_equal(out[0, 1 : w - 1], np.full(w - 2, RED)):
        return False
    if not np.array_equal(out[h - 1, 1 : w - 1], np.full(w - 2, BLUE)):
        return False
    if not np.array_equal(out[1 : h - 1, 0], np.full(h - 2, GREEN)):
        return False
    if not np.array_equal(out[1 : h - 1, w - 1], np.full(h - 2, YELLOW)):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(SIDES)

    for _ in range(80):
        area = random.randint(AREA_MIN, AREA_MAX)
        shape = _random_polyomino(area)
        if not _shape_is_good(shape, side):
            continue

        _, _, r1, c1 = _bbox(shape)
        shape_h = r1 + 1
        shape_w = c1 + 1
        max_r = h - 2 - shape_h
        max_c = w - 2 - shape_w
        if max_r < 2 or max_c < 2:
            continue

        base_r = random.randint(2, max_r)
        base_c = random.randint(2, max_c)
        shape_abs = _translate(shape, base_r, base_c)
        moved = _move_to_wall(shape_abs, side, h, w)
        mirrored = _mirror_across_middle(moved, side, h, w)
        if len(mirrored) != len(shape_abs) or set(moved) & set(mirrored):
            continue
        if mirrored == _opposite_wall_translation(moved, side, h, w):
            continue

        candidates = _candidate_markers(shape_abs, h, w)
        if not candidates:
            continue
        wrong_side = [cand for cand in candidates if cand[2] != side]
        prefer_wrong = bool(wrong_side) and random.random() < 0.8
        marker_r, marker_c, _ = random.choice(wrong_side if prefer_wrong else candidates)

        grid = _build_frame(h, w)
        for r, c in shape_abs:
            grid[r, c] = GREY
        grid[marker_r, marker_c] = SIDE_TO_COLOR[side]

        out = main(grid)
        if _semantic_checks(grid, out):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
