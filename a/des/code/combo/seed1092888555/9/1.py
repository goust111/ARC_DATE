import random
from collections import deque
from typing import Iterable

import numpy as np


BACKGROUND = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400
PLACEMENT_TRIES = 120


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _cavity_info(grid: np.ndarray, comp: dict, bg: int) -> tuple[str, np.ndarray]:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole = patch == bg
    assert np.any(hole), "object must contain a cavity in its bounding box"

    side_hits = {
        "up": bool(np.any(hole[0, :])),
        "down": bool(np.any(hole[-1, :])),
        "left": bool(np.any(hole[:, 0])),
        "right": bool(np.any(hole[:, -1])),
    }
    open_sides = [side for side, hit in side_hits.items() if hit]
    assert len(open_sides) == 1, "expected exactly one opening side"

    visited = np.zeros_like(hole, dtype=bool)
    cavity_cells = []
    starters: list[tuple[int, int]]
    side = open_sides[0]
    if side == "up":
        starters = [(0, c) for c in np.where(hole[0])[0]]
    elif side == "down":
        starters = [(hole.shape[0] - 1, c) for c in np.where(hole[-1])[0]]
    elif side == "left":
        starters = [(r, 0) for r in np.where(hole[:, 0])[0]]
    else:
        starters = [(r, hole.shape[1] - 1) for r in np.where(hole[:, -1])[0]]

    q = deque(starters)
    for sr, sc in starters:
        visited[sr, sc] = True
    while q:
        rr, cc = q.popleft()
        cavity_cells.append((rr, cc))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < hole.shape[0] and 0 <= nc < hole.shape[1] and hole[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    cavity = np.zeros_like(hole, dtype=bool)
    for rr, cc in cavity_cells:
        cavity[rr, cc] = True
    assert np.array_equal(cavity, hole), "bbox background must be a single opening cavity"
    return side, cavity


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = get_bg(g)
    comps = find_components(g, bg)
    assert comps, "expected foreground objects"

    out = g.copy()
    h, w = g.shape
    for comp in comps:
        side, cavity = _cavity_info(g, comp, bg)
        dr, dc = DIR_VECS[side]
        r0, c0, _r1, _c1 = comp["bbox"]
        color = comp["color"]
        for rr, cc in np.argwhere(cavity):
            ar = r0 + int(rr)
            ac = c0 + int(cc)
            while 0 <= ar < h and 0 <= ac < w and int(g[ar, ac]) == bg:
                out[ar, ac] = color
                ar += dr
                ac += dc
    return out


def _build_patch(height: int, width: int, side: str, color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((height, width), color, dtype=int)
    if side in ("up", "down"):
        notch_depth = random.randint(1, height - 2)
        notch_width = random.randint(1, width - 2)
        start_c = random.randint(1, width - notch_width - 1)
        rows = slice(0, notch_depth) if side == "up" else slice(height - notch_depth, height)
        cols = slice(start_c, start_c + notch_width)
    else:
        notch_depth = random.randint(1, width - 2)
        notch_height = random.randint(1, height - 2)
        start_r = random.randint(1, height - notch_height - 1)
        rows = slice(start_r, start_r + notch_height)
        cols = slice(0, notch_depth) if side == "left" else slice(width - notch_depth, width)
    patch[rows, cols] = BACKGROUND
    cavity = patch == BACKGROUND
    return patch, cavity


def _dilate(coords: Iterable[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _ray_cells(
    cavity_abs: list[tuple[int, int]],
    side: str,
    occupancy: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    h, w = shape
    dr, dc = DIR_VECS[side]
    painted = set()
    hit_object = False
    for sr, sc in cavity_abs:
        r, c = sr, sc
        while 0 <= r < h and 0 <= c < w:
            if (r, c) in occupancy:
                hit_object = True
                break
            painted.add((r, c))
            r += dr
            c += dc
    return sorted(painted), hit_object


def _valid_opening_patch(patch: np.ndarray, side: str) -> bool:
    comps = find_components(patch, BACKGROUND)
    if len(comps) != 1:
        return False
    try:
        found_side, _ = _cavity_info(patch, comps[0], BACKGROUND)
    except AssertionError:
        return False
    return found_side == side


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    grid = np.zeros((h, w), dtype=int)
    occupancy: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    colors = random.sample(COLORS, random.randint(3, min(7, len(COLORS))))
    n_objects = random.randint(2, min(5, len(colors)))
    direction_pool = [random.choice(list(DIR_VECS)) for _ in range(n_objects)]
    if n_objects >= 4:
        direction_pool[:4] = ["up", "down", "left", "right"]
        random.shuffle(direction_pool)

    for idx in range(n_objects):
        color = colors[idx]
        side = direction_pool[idx]
        want_hit = idx > 0 and random.random() < 0.25
        placed = False
        for _ in range(PLACEMENT_TRIES):
            ph = random.randint(4, 7)
            pw = random.randint(4, 7)
            patch, cavity = _build_patch(ph, pw, side, color)
            if not _valid_opening_patch(patch, side):
                continue

            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            obj_cells = [(r0 + r, c0 + c) for r, c in np.argwhere(patch == color)]
            cavity_cells = [(r0 + r, c0 + c) for r, c in np.argwhere(cavity)]

            if any(cell in forbidden for cell in obj_cells):
                continue
            bbox_pad = _dilate([(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)], (h, w), 1)
            if any(cell in occupancy for cell in bbox_pad):
                continue

            ray_cells, hit_object = _ray_cells(cavity_cells, side, occupancy, (h, w))
            if want_hit and not hit_object:
                continue
            if not want_hit and hit_object and random.random() < 0.75:
                continue

            for r, c in obj_cells:
                grid[r, c] = color
            occupancy.update(obj_cells)
            forbidden.update(_dilate(obj_cells, (h, w), 1))
            placed = True
            break
        if not placed:
            return None

    comps = find_components(grid, BACKGROUND)
    if len(comps) != n_objects:
        return None
    for comp in comps:
        try:
            _cavity_info(grid, comp, BACKGROUND)
        except AssertionError:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
