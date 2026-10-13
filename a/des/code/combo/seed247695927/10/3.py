from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
COLOR_VALUES = tuple(range(10))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append((color, coords))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 4-connected monochrome foreground object downward by its own area.

    Input structure:
    - The dominant color is the background.
    - Foreground consists of separated 4-connected single-color objects.

    Rule:
    - Clear the canvas to background.
    - For each foreground object, compute its pixel count.
    - Redraw the object at the same columns, shifted downward by that count.
    - Any pixels moved below the bottom edge are cropped away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full((h, w), bg, dtype=int)

    for color, coords in _components_by_color(g, bg):
        shift = len(coords)
        for r, c in coords:
            nr = r + shift
            if 0 <= nr < h:
                out[nr, c] = color
    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_connected_shape() -> np.ndarray:
    if random.random() < 0.5:
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        mask = np.ones((h, w), dtype=bool)
        if h * w >= 4 and random.random() < 0.5:
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, cc] = False
            if not mask.any():
                mask[rr, cc] = True
            elif not _is_connected(mask):
                mask[rr, cc] = True
        return _trim_mask(mask)

    target = random.randint(2, 6)
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < target:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)
    rel = _normalize(cells)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in rel:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _placeable(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    clearance: bool,
) -> bool:
    cell_set = set(cells)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if occupied & cell_set:
        return False
    if not clearance:
        return True
    for r, c in cells:
        for dr, dc in DIR8:
            nbr = (r + dr, c + dc)
            if nbr in occupied:
                return False
    return True


def _translated_cells(cells: list[tuple[int, int]], shift: int, h: int) -> list[tuple[int, int]]:
    return [(r + shift, c) for r, c in cells if r + shift < h]


def _clip_status(cells: list[tuple[int, int]], shift: int, h: int) -> str:
    shifted_rows = [r + shift for r, _ in cells]
    any_in = any(r < h for r in shifted_rows)
    any_out = any(r >= h for r in shifted_rows)
    if any_in and any_out:
        return "clipped"
    if any_in:
        return "inside"
    return "gone"


def _status_plan(n_objects: int) -> list[str]:
    if n_objects == 1:
        return [random.choice(["inside", "clipped"])]
    mode = random.choice(["all_inside", "all_clipped", "mixed"])
    if mode == "all_inside":
        return ["inside"] * n_objects
    if mode == "all_clipped":
        return ["clipped"] * n_objects
    plan = ["inside", "clipped"]
    while len(plan) < n_objects:
        plan.append(random.choice(["inside", "clipped"]))
    random.shuffle(plan)
    return plan


def _candidate_positions(
    mask: np.ndarray,
    status: str,
    in_occ: set[tuple[int, int]],
    out_occ: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]]:
    mh, mw = mask.shape
    rel = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    area = len(rel)
    choices = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            cells = [(r0 + r, c0 + c) for r, c in rel]
            if not _placeable(cells, in_occ, h, w, clearance=True):
                continue
            actual = _clip_status(cells, area, h)
            if actual != status:
                continue
            moved = _translated_cells(cells, area, h)
            if not moved:
                continue
            if not _placeable(moved, out_occ, h, w, clearance=False):
                continue
            choices.append((r0, c0, cells, moved))
    random.shuffle(choices)
    return choices


def _sample_grid() -> np.ndarray | None:
    h = random.randint(9, 18)
    w = random.randint(9, 18)
    bg = random.choice(COLOR_VALUES)
    n_objects = random.randint(2, 5)
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_objects)
    plan = _status_plan(n_objects)

    masks = [_random_connected_shape() for _ in range(n_objects)]
    order = list(range(n_objects))
    order.sort(key=lambda i: int(masks[i].sum()), reverse=True)

    placements: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
    in_occ: set[tuple[int, int]] = set()
    out_occ: set[tuple[int, int]] = set()

    for idx in order:
        mask = masks[idx]
        status = plan[idx]
        candidates = _candidate_positions(mask, status, in_occ, out_occ, h, w)
        if not candidates:
            return None
        _r0, _c0, cells, moved = random.choice(candidates)
        placements[idx] = (cells, moved)
        in_occ |= set(cells)
        out_occ |= set(moved)

    grid = np.full((h, w), bg, dtype=int)
    for idx, color in enumerate(colors):
        cells, _moved = placements[idx]
        for r, c in cells:
            grid[r, c] = color

    comps = _components_by_color(grid, bg)
    if len(comps) != n_objects:
        return None
    if Counter(grid.flatten())[bg] <= (h * w) // 2:
        return None

    out = main(grid)
    if np.array_equal(out, np.full((h, w), bg, dtype=int)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _sample_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
