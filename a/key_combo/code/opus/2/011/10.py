import random
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _components_for_value(grid: np.ndarray, value: int) -> List[List[Tuple[int, int]]]:
    mask = grid == value
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Separate the black template object, the small colored indicator pixels,
        and the large colored falling objects on the gray background.
    Analyze:
        Drop every large object under gravity, then derive one cardinal tiling
        direction per indicator from the template-centroid-to-indicator vector.
    Transform:
        Repeatedly place colored template copies at fixed step intervals until a
        copy would leave the grid or collide with any non-gray cell.
    Return:
        Return the final grid containing the settled large objects, unchanged
        template and indicators, and every successfully tiled template copy.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    template = _components_for_value(grid, 0)[0]
    template_rows = [r for r, _ in template]
    template_cols = [c for _, c in template]
    centroid_r = sum(template_rows) // len(template_rows)
    centroid_c = sum(template_cols) // len(template_cols)
    top, bottom = min(template_rows), max(template_rows)
    left, right = min(template_cols), max(template_cols)
    step_map = {
        "left": right - left + 2,
        "right": right - left + 2,
        "up": bottom - top + 2,
        "down": bottom - top + 2,
    }

    indicators = [(r, c, int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, [1, 2, 3]))]
    indicators.sort(key=lambda item: item[0])

    big_groups = []
    for color in (4, 7, 9):
        for comp in _components_for_value(grid, color):
            if len(comp) > 6:
                big_groups.append({"color": color, "coords": comp})

    big_groups.sort(key=lambda item: max(r for r, _ in item["coords"]), reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    for r, c in template:
        occupied[r, c] = True
    for r, c, _ in indicators:
        occupied[r, c] = True

    for group in big_groups:
        for r, c in group["coords"]:
            out[r, c] = 5

    for group in big_groups:
        coords = group["coords"]
        shift = 0
        while True:
            blocked = False
            for r, c in coords:
                nr = r + shift + 1
                if nr >= h or occupied[nr, c]:
                    blocked = True
                    break
            if blocked:
                break
            shift += 1
        final_coords = [(r + shift, c) for r, c in coords]
        group["coords"] = final_coords
        for r, c in final_coords:
            occupied[r, c] = True
            out[r, c] = group["color"]

    def direction(ind_r: int, ind_c: int) -> Tuple[str, int, int]:
        dr = ind_r - centroid_r
        dc = ind_c - centroid_c
        if abs(dc) >= abs(dr):
            return ("right", 0, 1) if dc > 0 else ("left", 0, -1)
        return ("down", 1, 0) if dr > 0 else ("up", -1, 0)

    for ind_r, ind_c, color in indicators:
        name, dr, dc = direction(ind_r, ind_c)
        step = step_map[name]
        offset = step
        while True:
            shifted = [(r + dr * offset, c + dc * offset) for r, c in template]
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in shifted):
                break
            if any(out[r, c] != 5 for r, c in shifted):
                break
            for r, c in shifted:
                out[r, c] = color
            offset += step
    return out


_BIG_SHAPES: Sequence[Sequence[Tuple[int, int]]] = (
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
)


def generate_input() -> np.ndarray:
    h = 14
    w = 14
    for _ in range(5000):
        grid = np.full((h, w), 5, dtype=int)
        template = [(4, 5), (4, 6), (5, 5), (5, 6)]
        for r, c in template:
            grid[r, c] = 0

        indicators = [(4, 7, 1), (6, 5, 2)]
        if random.random() < 0.5:
            indicators.append((4, 4, 3))
        for r, c, color in indicators:
            grid[r, c] = color

        big_specs = [
            (random.choice(_BIG_SHAPES), 0, 4, 4),
            (random.choice(_BIG_SHAPES), 1, 0, 7),
        ]
        if random.random() < 0.7:
            big_specs.append((random.choice(_BIG_SHAPES), 0, 10, 9))

        ok = True
        for shape, top, left, color in big_specs:
            for r, c in shape:
                rr, cc = top + r, left + c
                if rr >= h or cc >= w or grid[rr, cc] != 5:
                    ok = False
                    break
                grid[rr, cc] = color
            if not ok:
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid task-10 input")
