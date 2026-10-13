from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SIDE_COLORS = (RED, BLUE, GREEN, YELLOW)
DIR_BY_COLOR = {
    RED: (-1, 0),
    BLUE: (0, 1),
    GREEN: (1, 0),
    YELLOW: (0, -1),
}
AXIS_BY_COLOR = {
    RED: "horizontal",
    GREEN: "horizontal",
    BLUE: "vertical",
    YELLOW: "vertical",
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    )


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}, (r0, c0)


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    norm, _ = _normalize(coords)
    r1 = max(r for r, _ in norm)
    c1 = max(c for _, c in norm)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _extract_components(grid: np.ndarray, colors: tuple[int, ...] = SIDE_COLORS) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (1 <= nr < H - 1 and 1 <= nc < W - 1):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coord_set = set(coords)
            r0, c0, r1, c1 = _bbox(coord_set)
            local = {(rr - r0, cc - c0) for rr, cc in coord_set}
            comps.append(
                {
                    "color": color,
                    "coords": coord_set,
                    "bbox": (r0, c0, r1, c1),
                    "mask": _coords_to_mask(local),
                }
            )
    return comps


def _copy_offset(mask: np.ndarray, color: int) -> tuple[int, int]:
    h, w = mask.shape
    if color == RED:
        return -h, 1
    if color == BLUE:
        return 1, w
    if color == GREEN:
        return h, -1
    if color == YELLOW:
        return -1, -w
    raise ValueError(f"unexpected color {color}")


def _reflected_cells(mask: np.ndarray, color: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    base = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if AXIS_BY_COLOR[color] == "horizontal":
        reflected = {(h - 1 - r, c) for r, c in base}
    else:
        reflected = {(r, w - 1 - c) for r, c in base}
    dr, dc = _copy_offset(mask, color)
    return {(r + dr, c + dc) for r, c in reflected}


def _component_result(
    mask: np.ndarray, color: int, top_left: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], bool]:
    r0, c0 = top_left
    original_local = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    copy_local = _reflected_cells(mask, color)
    original = {(r0 + r, c0 + c) for r, c in original_local}
    copy = {(r0 + r, c0 + c) for r, c in copy_local}
    merged = original | copy
    return copy, merged, _connected(merged)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each frame-colored interior sprite toward its matching frame side, then recolor only the connected mirror-merges.

    Input structure:
    - The grid uses black background and a one-cell-thick rectangular frame whose top/right/bottom/left sides are red, blue, green, and yellow, with black corners.
    - Inside the frame are several separated 4-connected monochrome sprites, each colored with exactly one of those four frame-side colors.
    - Each sprite color uniquely identifies which frame side it matches.

    Rule:
    - Red and green sprites use a horizontal mirror; blue and yellow sprites use a vertical mirror.
    - Place the mirrored copy flush toward the matched side, keeping the copy on that side of the sprite's bounding box.
    - If the original sprite together with its mirrored copy forms one 4-connected component, recolor that whole merged set orange.
    - Otherwise keep both the original and the mirrored copy in the sprite's original color. The frame stays unchanged.

    Key insight:
    - The frame color does not say which pixels to recolor directly; it first chooses the mirror axis and placement side, and only the resulting connectivity decides whether orange appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H >= 6 and W >= 6

    out = g.copy()
    for comp in _extract_components(g):
        r0, c0, _r1, _c1 = comp["bbox"]
        copy, merged, is_connected = _component_result(comp["mask"], comp["color"], (r0, c0))
        paint = ORANGE if is_connected else comp["color"]
        for rr, cc in (merged if is_connected else copy):
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = paint
        if is_connected:
            for rr, cc in comp["coords"]:
                out[rr, cc] = ORANGE
    return out


def _random_polyomino(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    norm, _ = _normalize(cells)
    mode = random.randrange(8)
    transformed = set(norm)
    if mode & 1:
        max_r = max(r for r, _ in transformed)
        transformed = {(max_r - r, c) for r, c in transformed}
    if mode & 2:
        max_c = max(c for _, c in transformed)
        transformed = {(r, max_c - c) for r, c in transformed}
    if mode & 4:
        transformed = {(c, r) for r, c in transformed}
    transformed, _ = _normalize(transformed)
    return transformed


def _shape_with_branch(color: int, want_connected: bool) -> np.ndarray:
    for _ in range(400):
        area = random.randint(3, 8)
        coords = _random_polyomino(area)
        mask = _coords_to_mask(coords)
        h, w = mask.shape
        if h > 5 or w > 5:
            continue
        if AXIS_BY_COLOR[color] == "horizontal" and h < 2:
            continue
        if AXIS_BY_COLOR[color] == "vertical" and w < 2:
            continue
        copy_local = _reflected_cells(mask, color)
        merged_local = {(int(r), int(c)) for r, c in np.argwhere(mask)} | copy_local
        if _connected(merged_local) != want_connected:
            continue
        return mask
    raise RuntimeError(f"failed to build branch shape for color={color}, connected={want_connected}")


def _frame_grid(H: int, W: int) -> np.ndarray:
    g = np.zeros((H, W), dtype=int)
    g[0, 1:-1] = RED
    g[1:-1, -1] = BLUE
    g[-1, 1:-1] = GREEN
    g[1:-1, 0] = YELLOW
    return g


def _expanded(cells: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W:
            out.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < H and 0 <= nc < W:
                out.add((nr, nc))
    return out


def _origin_bounds(H: int, W: int, mask: np.ndarray, color: int) -> tuple[range, range]:
    h, w = mask.shape
    min_r, max_r = 1, H - h - 1
    min_c, max_c = 1, W - w - 1
    if color == RED:
        min_r = max(min_r, h + 1)
        max_c = min(max_c, W - w - 2)
    elif color == GREEN:
        max_r = min(max_r, H - 2 * h - 1)
        min_c = max(min_c, 2)
    elif color == BLUE:
        max_r = min(max_r, H - h - 2)
        max_c = min(max_c, W - 2 * w - 1)
    elif color == YELLOW:
        min_r = max(min_r, 2)
        min_c = max(min_c, w + 1)
    return range(min_r, max_r + 1), range(min_c, max_c + 1)


def _placeable(
    H: int,
    W: int,
    mask: np.ndarray,
    color: int,
    top_left: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]], set[tuple[int, int]], bool]:
    r0, c0 = top_left
    copy, merged, is_connected = _component_result(mask, color, (r0, c0))
    original = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
    for cells in (original, copy):
        for rr, cc in cells:
            if not (1 <= rr < H - 1 and 1 <= cc < W - 1):
                return False, set(), set(), False
    if merged & blocked:
        return False, set(), set(), False
    return True, original, merged, is_connected


def _build_attempt() -> np.ndarray | None:
    H = random.randint(15, 18)
    W = random.randint(15, 18)
    grid = _frame_grid(H, W)

    horiz_targets = [True, False]
    vert_targets = [True, False]
    random.shuffle(horiz_targets)
    random.shuffle(vert_targets)

    specs = [
        (RED, horiz_targets[0]),
        (GREEN, horiz_targets[1]),
        (BLUE, vert_targets[0]),
        (YELLOW, vert_targets[1]),
    ]
    if random.random() < 0.45:
        extra_color = random.choice(SIDE_COLORS)
        extra_branch = random.choice((True, False))
        specs.append((extra_color, extra_branch))
    random.shuffle(specs)

    placed = []
    blocked: set[tuple[int, int]] = set()
    branch_hits = {True: 0, False: 0}
    axis_hits = {"horizontal": 0, "vertical": 0}

    for color, target_connected in specs:
        mask = _shape_with_branch(color, target_connected)
        row_range, col_range = _origin_bounds(H, W, mask, color)
        origins = [(r, c) for r in row_range for c in col_range]
        random.shuffle(origins)
        chosen = None
        for origin in origins:
            ok, original, merged, actual_connected = _placeable(H, W, mask, color, origin, blocked)
            if not ok or actual_connected != target_connected:
                continue
            chosen = (origin, original, merged, actual_connected)
            break
        if chosen is None:
            return None

        origin, original, merged, actual_connected = chosen
        for rr, cc in original:
            grid[rr, cc] = color
        blocked |= _expanded(merged, H, W)
        placed.append((color, mask, origin, merged, actual_connected))
        branch_hits[actual_connected] += 1
        axis_hits[AXIS_BY_COLOR[color]] += 1

    if branch_hits[True] == 0 or branch_hits[False] == 0:
        return None
    if axis_hits["horizontal"] == 0 or axis_hits["vertical"] == 0:
        return None

    comps = _extract_components(grid)
    if len(comps) != len(specs):
        return None
    if {comp["color"] for comp in comps} != {RED, BLUE, GREEN, YELLOW}:
        return None
    if not all(3 <= len(comp["coords"]) <= 8 for comp in comps):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == ORANGE)) == 0:
        return None

    orange_components = len(_extract_components(out, colors=(ORANGE,)))
    if orange_components == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
