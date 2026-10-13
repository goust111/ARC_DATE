import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or comp_ids[r, c] != -1:
                continue
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            comp_id = len(components)
            comp_ids[r, c] = comp_id
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and comp_ids[nr, nc] == -1:
                        comp_ids[nr, nc] = comp_id
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return comp_ids, components


def _door_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    doors = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            if int(grid[r, c - 1]) == RED and int(grid[r, c + 1]) == RED:
                if int(grid[r - 1, c]) != RED and int(grid[r + 1, c]) != RED:
                    doors[r, c] = True
    return doors


def _trace_path(
    grid: np.ndarray,
    comp_ids: np.ndarray,
    door_mask: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    out = grid.copy()
    starts = np.argwhere(grid == BLUE)
    assert starts.shape[0] == 1, "expected exactly one blue start"
    r, c = map(int, starts[0])

    current_region = int(comp_ids[r, c])
    assert current_region >= 0, "start must be inside a sealed region"
    order = [current_region]

    h, w = out.shape
    while r > 0:
        above = int(out[r - 1, c])
        if above != RED:
            assert above != GREY, "gray obstacles should only block door fronts"
            out[r - 1, c] = BLUE
            r -= 1
            continue

        reach = c
        while reach + 1 < w and int(out[r, reach + 1]) not in (RED, GREY):
            reach += 1

        gap_c = None
        for cc in range(c + 1, reach + 1):
            if not door_mask[r - 1, cc]:
                continue
            if r - 2 >= 0 and int(out[r - 2, cc]) == GREY:
                continue
            gap_c = cc
            break

        if gap_c is None:
            break

        out[r, c : gap_c + 1] = BLUE
        out[r - 1, gap_c] = BLUE

        if r - 2 >= 0:
            next_region = int(comp_ids[r - 2, gap_c])
            assert next_region >= 0, "door must lead into another sealed region"
            if next_region != current_region:
                assert next_region not in order, "path should visit each region at most once"
                order.append(next_region)
                current_region = next_region

        r, c = r - 1, gap_c

    return out, order


def _fill_regions(
    original: np.ndarray,
    traced: np.ndarray,
    components: list[np.ndarray],
    order: list[int],
) -> np.ndarray:
    out = traced.copy()
    rank_by_region = {region: idx for idx, region in enumerate(order)}
    for region, coords in enumerate(components):
        idx = rank_by_region.get(region)
        if idx is None:
            continue
        fill = None
        if idx % 3 == 0:
            fill = YELLOW
        elif idx % 3 == 1:
            fill = GREEN
        if fill is None:
            continue
        for r, c in coords:
            if int(original[r, c]) == BLACK and int(out[r, c]) == BLACK:
                out[r, c] = fill
    return out


def _solve(grid: np.ndarray, *, return_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    doors = _door_mask(g)
    sealed = g.copy()
    sealed[doors] = RED
    comp_ids, components = _components(sealed != RED)

    traced, order = _trace_path(g, comp_ids, doors)
    out = _fill_regions(g, traced, components, order)

    if return_debug:
        return out, {
            "door_mask": doors,
            "sealed": sealed,
            "comp_ids": comp_ids,
            "components": components,
            "order": order,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow a blue path through walled regions, then recolor regions by the path's visit order modulo three.

    Input structure:
    - The grid is mostly black background with thin red walls that partition it into enclosed 4-connected regions.
    - One blue start cell lies in the lowest region, and some wall gaps act as upward channels between consecutive regions.
    - Gray cells may sit immediately above some gaps, marking blocked channels that the path must skip.

    Rule:
    - Start from the blue cell and keep extending the blue path upward one cell at a time.
    - Whenever the cell above is red, scan right along the current row for the nearest wall gap; ignore any gap whose next cell above contains a gray obstacle.
    - Move horizontally to the first unblocked gap, pass through it, and continue upward in the next region.
    - Record the regions in the order the path enters them, with the start region as index 0.
    - Fill black cells of regions with indices 0, 3, 6, ... in yellow, fill indices 1, 4, 7, ... in green, and leave indices 2, 5, 8, ... black.
    - Keep all red walls, gray obstacles, and blue path cells unchanged.

    Key insight:
    - The fill color is not determined by geometric position alone; you must first reconstruct the room order by simulating the upward path and its rightward detours around blocked doorways.
    """
    return _solve(grid)


GEN_RETRIES = 400
ROOM_COUNTS = (6, 7, 8)
HEIGHT_CHOICES = (2, 2, 3)
STEP_CHOICES = (1, 2, 2, 3)
RIGHT_MARGIN_CHOICES = (1, 2, 3)
LEFT_MARGIN_CHOICES = (1, 2)


def _draw_room(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    grid[top - 1, left - 1 : right + 2] = RED
    grid[bottom + 1, left - 1 : right + 2] = RED
    grid[top - 1 : bottom + 2, left - 1] = RED
    grid[top - 1 : bottom + 2, right + 1] = RED


def _sample_layout():
    n_rooms = random.choice(ROOM_COUNTS)
    heights = [random.choice(HEIGHT_CHOICES) for _ in range(n_rooms)]
    h = sum(heights) + n_rooms + 1
    if h > 29:
        return None

    steps = [random.choice(STEP_CHOICES) for _ in range(n_rooms - 1)]
    if max(steps) < 3:
        steps[random.randrange(len(steps))] = 3

    path_cols = [random.randint(2, 4)]
    for step in steps:
        path_cols.append(path_cols[-1] + step)
    if path_cols[-1] > 18:
        return None

    door_cols: list[list[int]] = []
    blocked_count = 0
    for i, step in enumerate(steps):
        blocked = []
        if step >= 3 and random.random() < 0.75:
            blocked = [path_cols[i] + 1]
        blocked_count += len(blocked)
        door_cols.append(blocked + [path_cols[i + 1]])

    if blocked_count == 0:
        for i, step in enumerate(steps):
            if step >= 3:
                door_cols[i] = [path_cols[i] + 1, path_cols[i + 1]]
                blocked_count = 1
                break
    if blocked_count == 0:
        return None

    lefts: list[int] = []
    rights: list[int] = []
    for i in range(n_rooms):
        if i == 0:
            left_anchor = path_cols[0] - random.choice(LEFT_MARGIN_CHOICES)
        else:
            left_anchor = min(door_cols[i - 1]) - random.randint(0, 2)
        left = max(1, left_anchor)
        need_right = path_cols[i + 1] if i < n_rooms - 1 else path_cols[i]
        right = need_right + random.choice(RIGHT_MARGIN_CHOICES)
        if right - left + 1 < 5:
            left = max(1, right - 4)
        lefts.append(left)
        rights.append(right)

    w = max(rights) + 2
    if w > 29:
        return None

    tops: list[int] = []
    bottoms: list[int] = []
    bottom = h - 2
    for height in heights:
        top = bottom - height + 1
        tops.append(top)
        bottoms.append(bottom)
        bottom = top - 2
    if tops[-1] != 1:
        return None

    return {
        "n_rooms": n_rooms,
        "h": h,
        "w": w,
        "tops": tops,
        "bottoms": bottoms,
        "lefts": lefts,
        "rights": rights,
        "path_cols": path_cols,
        "door_cols": door_cols,
    }


def _build_grid(layout: dict) -> np.ndarray:
    grid = np.full((layout["h"], layout["w"]), BLACK, dtype=int)

    for top, bottom, left, right in zip(
        layout["tops"],
        layout["bottoms"],
        layout["lefts"],
        layout["rights"],
    ):
        _draw_room(grid, top, bottom, left, right)

    for i, cols in enumerate(layout["door_cols"]):
        wall_r = layout["tops"][i] - 1
        for door_c in cols:
            grid[wall_r, door_c] = BLACK
        for blocked_c in cols[:-1]:
            grid[wall_r - 1, blocked_c] = GREY

    grid[layout["bottoms"][0], layout["path_cols"][0]] = BLUE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _sample_layout()
        if layout is None:
            continue
        grid = _build_grid(layout)
        try:
            out, debug = _solve(grid, return_debug=True)
        except AssertionError:
            continue

        order = debug["order"]
        if len(order) != layout["n_rooms"]:
            continue
        if len(set(order)) != len(order):
            continue
        if not np.any(out == YELLOW) or not np.any(out == GREEN):
            continue
        return grid
    raise RuntimeError("failed to generate a valid periodic flood-fill puzzle")
