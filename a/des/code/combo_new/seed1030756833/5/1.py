from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-colored seeds inside each gray rectangle and shoot that many pixels outward from its colored border gap.

    Input structure:
    - Black is the background.
    - The grid contains one or more one-pixel-thick gray rectangular frames.
    - Each frame border has exactly one non-black, non-gray colored gap pixel, and its interior contains scattered seed pixels of that same color.

    Rule:
    - For each gray rectangular frame, locate the single colored gap pixel on its border and take its color C.
    - Count the number N of interior pixels whose color is C.
    - Determine whether the gap is on the top, bottom, left, or right side of the frame.
    - Starting just outside the gap, draw a straight ray of N pixels in the outward normal direction using color C.
    - Preserve all original frame, gap, and seed pixels.

    Key insight:
    - The gap color is both the key for counting the matching interior seeds and the color of the outward ray, while the side of the gap gives the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    gray_components = _components4(g == GREY)
    assert gray_components, "expected at least one gray rectangular frame"

    for comp in gray_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frames must enclose an interior"

        border = set(_border_cells(r0, c0, r1, c1))
        comp_set = set(comp)
        assert comp_set <= border, "gray component must be a rectangular border"

        gaps: list[tuple[int, int]] = []
        for pos in border:
            value = int(g[pos])
            if value == GREY:
                continue
            assert value != BLACK, "the missing border cell must be colored, not black"
            gaps.append(pos)
        assert len(gaps) == 1, "each frame must have exactly one colored border gap"

        gap = gaps[0]
        color = int(g[gap])
        assert comp_set == border - {gap}, "all non-gap border cells must be gray"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_colors = set(map(int, np.unique(interior)))
        assert interior_colors <= {BLACK, color}, "frame interiors may only contain black or matching seed pixels"
        seed_count = int(np.sum(interior == color))
        assert seed_count > 0, "each frame must contain at least one matching seed"

        gr, gc = gap
        sides: list[str] = []
        if gr == r0:
            sides.append("top")
        if gr == r1:
            sides.append("bottom")
        if gc == c0:
            sides.append("left")
        if gc == c1:
            sides.append("right")
        assert len(sides) == 1, "gap must lie on exactly one side, not on a corner"

        dr, dc = DIRECTIONS[sides[0]]
        for step in range(1, seed_count + 1):
            rr = gr + dr * step
            cc = gc + dc * step
            assert 0 <= rr < h and 0 <= cc < w, "ray must fit inside the grid"
            assert int(g[rr, cc]) == BLACK, "ray path must be black in the input"
            out[rr, cc] = color

    return out


def _ray_cells(gap: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRECTIONS[side]
    gr, gc = gap
    return [(gr + dr * step, gc + dc * step) for step in range(1, length + 1)]


def _mark_rect_halo(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _mark_ray_halo(blocked: np.ndarray, ray: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in ray:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                blocked[rr, cc] = True


def _try_place_frame(grid: np.ndarray, blocked: np.ndarray, side: str, color: int) -> bool:
    h, w = grid.shape
    for _ in range(260):
        rect_h = random.randint(4, 8)
        rect_w = random.randint(4, 8)
        if rect_h + 2 > h or rect_w + 2 > w:
            continue

        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(1, w - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

        if side == "top":
            ray_room = r0
            gap = (r0, random.randint(c0 + 1, c1 - 1))
            inward = (r0 + 1, gap[1])
        elif side == "bottom":
            ray_room = h - 1 - r1
            gap = (r1, random.randint(c0 + 1, c1 - 1))
            inward = (r1 - 1, gap[1])
        elif side == "left":
            ray_room = c0
            gap = (random.randint(r0 + 1, r1 - 1), c0)
            inward = (gap[0], c0 + 1)
        else:
            ray_room = w - 1 - c1
            gap = (random.randint(r0 + 1, r1 - 1), c1)
            inward = (gap[0], c1 - 1)

        seed_cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if (r, c) != inward
        ]
        max_seeds = min(6, ray_room, len(seed_cells))
        if max_seeds < 1:
            continue
        seed_count = random.randint(1, max_seeds)
        seeds = random.sample(seed_cells, seed_count)
        ray = _ray_cells(gap, side, seed_count)

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        if any(blocked[r, c] or int(grid[r, c]) != BLACK for r, c in ray):
            continue

        for r, c in _border_cells(r0, c0, r1, c1):
            grid[r, c] = GREY
        grid[gap] = color
        for r, c in seeds:
            grid[r, c] = color

        _mark_rect_halo(blocked, r0, c0, r1, c1)
        _mark_ray_halo(blocked, ray)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 30)
    w = random.randint(12, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    frame_count = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
    colors = random.sample(PALETTE, frame_count)
    sides = list(DIRECTIONS)
    random.shuffle(sides)
    while len(sides) < frame_count:
        extra = list(DIRECTIONS)
        random.shuffle(extra)
        sides.extend(extra)

    for color, side in zip(colors, sides[:frame_count]):
        if not _try_place_frame(grid, blocked, side, color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(700):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
