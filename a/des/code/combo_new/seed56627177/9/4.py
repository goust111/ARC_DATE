from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RIGHT_TURN = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
LEFT_TURN = {v: k for k, v in RIGHT_TURN.items()}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_coords(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _parse_arrow(comp_id: int, color: int, coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    assert int(patch.sum()) == 11, "each arrow must have a 3x3 head and a length-2 tail"

    square_r = None
    square_c = None
    direction = None
    if h == 3 and w == 5:
        if bool(patch[:, :3].all()) and bool(patch[1, 3]) and bool(patch[1, 4]):
            square_r, square_c = r0, c0
            direction = (0, -1)
        elif bool(patch[:, 2:].all()) and bool(patch[1, 0]) and bool(patch[1, 1]):
            square_r, square_c = r0, c0 + 2
            direction = (0, 1)
    elif h == 5 and w == 3:
        if bool(patch[:3, :].all()) and bool(patch[3, 1]) and bool(patch[4, 1]):
            square_r, square_c = r0, c0
            direction = (-1, 0)
        elif bool(patch[2:, :].all()) and bool(patch[0, 1]) and bool(patch[1, 1]):
            square_r, square_c = r0 + 2, c0
            direction = (1, 0)

    assert direction is not None, "foreground component does not match the required arrow shape"
    square_cells = {
        (rr, cc)
        for rr in range(square_r, square_r + 3)
        for cc in range(square_c, square_c + 3)
    }
    return {
        "id": comp_id,
        "color": color,
        "coords": coords,
        "dir": direction,
        "square_r": square_r,
        "square_c": square_c,
        "square_cells": square_cells,
    }


def _ray_start(arrow: dict) -> tuple[int, int]:
    sr = int(arrow["square_r"])
    sc = int(arrow["square_c"])
    dr, dc = arrow["dir"]
    if (dr, dc) == (0, 1):
        return sr + 1, sc + 3
    if (dr, dc) == (0, -1):
        return sr + 1, sc - 1
    if (dr, dc) == (1, 0):
        return sr + 3, sc + 1
    return sr - 1, sc + 1


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict], dict[int, int | None]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    comps = _component_coords(g, bg)
    arrows: list[dict] = []
    labels = np.zeros(g.shape, dtype=int)

    for comp_id, comp in enumerate(comps, start=1):
        arrow = _parse_arrow(comp_id, int(comp["color"]), comp["coords"])
        arrows.append(arrow)
        labels[comp["coords"][:, 0], comp["coords"][:, 1]] = comp_id

    by_id = {arrow["id"]: arrow for arrow in arrows}
    succ: dict[int, int | None] = {}
    h, w = g.shape
    for arrow in arrows:
        dr, dc = arrow["dir"]
        rr, cc = _ray_start(arrow)
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            hit = int(labels[rr, cc])
            if hit != 0:
                if (rr, cc) in by_id[hit]["square_cells"]:
                    target = hit
                break
            rr += dr
            cc += dc
        succ[int(arrow["id"])] = target

    return bg, arrows, succ


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift arrow colors one step downstream along the directed arrow chain.

    Input structure:
    - The background is the dominant color.
    - Each foreground object is a monochrome arrow made from a solid 3x3 square head plus a length-2 single-cell tail.
    - The tail sits on one side of the head's center cell, so the arrow points in the opposite cardinal direction.
    - Arrows are separated from each other, and an arrow may face another arrow's head side along a clear row or column.

    Rule:
    - Detect every arrow object and infer its pointing direction from which side carries the tail.
    - For each arrow, cast a ray from the center of the pointed side and record the first arrow head it reaches, if any.
    - An arrow with no incoming arrow keeps its original color.
    - An arrow with exactly one incoming arrow is recolored to that upstream arrow's original color, while its shape and position stay unchanged.

    Key insight:
    - The asymmetric tail reveals direction, and the recoloring is a one-step color transfer based on the directed predecessor relation rather than on geometric proximity alone.
    """
    g = np.asarray(grid, dtype=int)
    _bg, arrows, succ = _analyze_grid(g)
    out = g.copy()

    incoming: dict[int, list[int]] = {int(arrow["id"]): [] for arrow in arrows}
    for src, dst in succ.items():
        if dst is not None:
            incoming[int(dst)].append(int(src))

    original_colors = {int(arrow["id"]): int(arrow["color"]) for arrow in arrows}
    for arrow in arrows:
        arrow_id = int(arrow["id"])
        preds = incoming[arrow_id]
        assert len(preds) <= 1, "each arrow must have at most one upstream arrow"
        new_color = original_colors[preds[0]] if preds else original_colors[arrow_id]
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    return out


GEN_ATTEMPTS = 400
BASE_SIDE = 27
LATTICE_SIDE = 5
STEP = 5
OFFSET = 2
MIN_ARROWS = 3
MAX_ARROWS = 6
MARGIN_VALUES = (0, 1, 2)
COLOR_POOL = list(range(10))


def _lattice_to_square(cell: tuple[int, int]) -> tuple[int, int]:
    return OFFSET + cell[0] * STEP, OFFSET + cell[1] * STEP


def _tail_cells(square_r: int, square_c: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return [(square_r + 1, square_c - 1), (square_r + 1, square_c - 2)]
    if direction == (0, -1):
        return [(square_r + 1, square_c + 3), (square_r + 1, square_c + 4)]
    if direction == (1, 0):
        return [(square_r - 1, square_c + 1), (square_r - 2, square_c + 1)]
    return [(square_r + 3, square_c + 1), (square_r + 4, square_c + 1)]


def _paint_arrow(grid: np.ndarray, square_r: int, square_c: int, direction: tuple[int, int], color: int) -> None:
    grid[square_r : square_r + 3, square_c : square_c + 3] = color
    for rr, cc in _tail_cells(square_r, square_c, direction):
        grid[rr, cc] = color


def _path_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return dirs


def _build_path(length: int) -> list[tuple[int, int]] | None:
    def search(path: list[tuple[int, int]], used: set[tuple[int, int]], prev_dir: tuple[int, int] | None):
        if len(path) == length:
            return list(path)

        r, c = path[-1]
        choices = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < LATTICE_SIDE and 0 <= nc < LATTICE_SIDE):
                continue
            if (nr, nc) in used:
                continue
            if prev_dir is not None and (dr, dc) == prev_dir:
                continue
            choices.append((dr, dc, nr, nc))

        random.shuffle(choices)
        for dr, dc, nr, nc in choices:
            path.append((nr, nc))
            used.add((nr, nc))
            result = search(path, used, (dr, dc))
            if result is not None:
                return result
            used.remove((nr, nc))
            path.pop()
        return None

    for _ in range(120):
        start = (random.randrange(LATTICE_SIDE), random.randrange(LATTICE_SIDE))
        result = search([start], {start}, None)
        if result is not None:
            return result
    return None


def _candidate_last_dirs(incoming: tuple[int, int]) -> list[tuple[int, int]]:
    return [LEFT_TURN[incoming], RIGHT_TURN[incoming]]


def _crop_to_content(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != bg)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    top = random.choice(MARGIN_VALUES)
    bottom = random.choice(MARGIN_VALUES)
    left = random.choice(MARGIN_VALUES)
    right = random.choice(MARGIN_VALUES)
    rr0 = max(0, r0 - top)
    rr1 = min(grid.shape[0] - 1, r1 + bottom)
    cc0 = max(0, c0 - left)
    cc1 = min(grid.shape[1] - 1, c1 + right)
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _edge_colors(arrows: list[dict], succ: dict[int, int | None]) -> dict[int, int | None]:
    color_by_id = {int(arrow["id"]): int(arrow["color"]) for arrow in arrows}
    return {
        int(color_by_id[src]): (None if dst is None else int(color_by_id[dst]))
        for src, dst in succ.items()
    }


def _expected_output(grid: np.ndarray) -> np.ndarray:
    bg, arrows, succ = _analyze_grid(grid)
    out = np.full_like(grid, bg)
    out[:, :] = grid
    incoming: dict[int, list[int]] = {int(arrow["id"]): [] for arrow in arrows}
    for src, dst in succ.items():
        if dst is not None:
            incoming[int(dst)].append(int(src))
    color_by_id = {int(arrow["id"]): int(arrow["color"]) for arrow in arrows}
    for arrow in arrows:
        preds = incoming[int(arrow["id"])]
        assert len(preds) <= 1
        color = color_by_id[preds[0]] if preds else color_by_id[int(arrow["id"])]
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_arrows = random.randint(MIN_ARROWS, MAX_ARROWS)
    path = _build_path(n_arrows)
    if path is None:
        return None

    dirs = _path_dirs(path)
    last_options = _candidate_last_dirs(dirs[-1])
    random.shuffle(last_options)

    bg = random.choice(COLOR_POOL)
    colors = random.sample([c for c in COLOR_POOL if c != bg], n_arrows)
    expected_succ = {
        colors[i]: (colors[i + 1] if i + 1 < n_arrows else None)
        for i in range(n_arrows)
    }

    for last_dir in last_options:
        full_dirs = dirs + [last_dir]
        grid = np.full((BASE_SIDE, BASE_SIDE), bg, dtype=int)
        for idx, cell in enumerate(path):
            square_r, square_c = _lattice_to_square(cell)
            _paint_arrow(grid, square_r, square_c, full_dirs[idx], colors[idx])

        cropped = _crop_to_content(grid, bg)
        if max(cropped.shape) > 30:
            continue

        try:
            _, arrows, succ = _analyze_grid(cropped)
        except AssertionError:
            continue

        if len(arrows) != n_arrows:
            continue
        observed = _edge_colors(arrows, succ)
        if observed != expected_succ:
            continue

        indegree = Counter(dst for dst in succ.values() if dst is not None)
        if sum(1 for arrow in arrows if indegree[int(arrow["id"])] == 0) != 1:
            continue

        expected = _expected_output(cropped)
        actual = main(cropped)
        if not np.array_equal(actual, expected):
            continue

        return cropped
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain input")
