from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
COLOR_POOL = list(range(1, 10))
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_dict(coords: list[tuple[int, int]]) -> dict:
    pts = np.array(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return {
        "coords": pts,
        "area": int(len(coords)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def find_components(mask: np.ndarray, connectivity: int = 8) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = NEIGHBORS8 if int(connectivity) == 8 else NEIGHBORS4
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(_component_dict(coords))
    return comps


def rotate_mask(mask: np.ndarray, steps_cw: int) -> np.ndarray:
    k = int(steps_cw) % 4
    if k == 0:
        return mask.copy()
    h, w = mask.shape
    if h == w:
        return np.rot90(mask, -k)

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        y = float(r) - center_r
        x = float(c) - center_c
        if k == 1:
            ny, nx = -x, y
        elif k == 2:
            ny, nx = -y, -x
        else:
            ny, nx = x, -y
        nr = center_r + ny
        nc = center_c + nx
        ir, ic = int(round(nr)), int(round(nc))
        if abs(nr - ir) < 1e-6 and abs(nc - ic) < 1e-6 and 0 <= ir < h and 0 <= ic < w:
            out[ir, ic] = True
    return out


def _largest_foreground_component(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid != bg, connectivity=8)
    assert comps, "expected foreground"
    comps.sort(key=lambda comp: comp["area"], reverse=True)
    return comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    board = _largest_foreground_component(g, bg)
    r0, c0, r1, c1 = board["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    out = crop.copy()

    for color in sorted(int(v) for v in np.unique(crop) if int(v) != int(bg)):
        comps = find_components(outside == color, connectivity=8)
        cue_size = max((comp["area"] for comp in comps), default=0)
        steps = int(cue_size) % 4
        if steps == 0:
            continue
        layer = crop == color
        rotated = rotate_mask(layer, steps)
        out[layer] = bg
        out[rotated] = color

    return out


def _random_connected_shape_in_mask(allowed: np.ndarray, area: int) -> np.ndarray | None:
    cells = [tuple(map(int, p)) for p in np.argwhere(allowed)]
    if len(cells) < area or area <= 0:
        return None
    start = random.choice(cells)
    chosen = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc] and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(chosen) < area:
        if not frontier:
            return None
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        chosen.add(nxt)
        add_frontier(nxt)

    mask = np.zeros_like(allowed, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _random_overlay_mask(n: int, free: np.ndarray, steps: int | None) -> np.ndarray | None:
    free_count = int(free.sum())
    if free_count < 4:
        return None
    min_area = 4
    max_area = min(free_count, max(6, n * n // 4))
    for _ in range(80):
        area = random.randint(min_area, max_area)
        mask = _random_connected_shape_in_mask(free, area)
        if mask is None:
            continue
        if not np.any(mask[1:-1, 1:-1]):
            continue
        if steps in (1, 2, 3) and np.array_equal(mask, rotate_mask(mask, steps)):
            continue
        return mask
    return None


def _cue_area_for_steps(steps: int) -> int:
    choices = {
        0: [4, 8],
        1: [1, 5],
        2: [2, 6],
        3: [3, 7],
    }
    return random.choice(choices[int(steps)])


def _random_cue_shape(area: int) -> np.ndarray | None:
    side = 3 if area <= 5 else 4
    allowed = np.ones((side, side), dtype=bool)
    return _random_connected_shape_in_mask(allowed, area)


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _can_place_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not np.any(occupied[r0:r1, c0:c1])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _random_branch_assignment(colors: list[int]) -> dict[int, int | None]:
    branches = {}
    base = colors[0]
    branches[base] = random.choice([None, 0])
    nonbase = colors[1:]
    weighted = [None, 0, 1, 2, 3]
    for color in nonbase:
        branches[color] = random.choice(weighted)
    if nonbase and not any(branches[c] in (1, 2, 3) for c in nonbase):
        chosen = random.choice(nonbase)
        branches[chosen] = random.choice([1, 2, 3])
    return branches


def _make_board_and_branches() -> tuple[np.ndarray, dict[int, int | None]] | None:
    n = random.randint(7, 11)
    num_colors = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, num_colors)
    branches = _random_branch_assignment(colors)

    board = np.full((n, n), colors[0], dtype=int)
    free = np.ones((n, n), dtype=bool)
    visible_rotation = False
    for color in colors[1:]:
        mask = _random_overlay_mask(n, free, branches[color])
        if mask is None:
            return None
        board[mask] = color
        free[mask] = False
        if branches[color] in (1, 2, 3) and not np.array_equal(mask, rotate_mask(mask, int(branches[color]))):
            visible_rotation = True

    if not visible_rotation:
        return None
    return board, branches


def _try_place_cue(
    grid: np.ndarray,
    occupied: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    color: int,
    steps: int,
) -> bool:
    cue = _random_cue_shape(_cue_area_for_steps(steps))
    if cue is None:
        return False
    cr0, cc0, cr1, cc1 = _bbox_from_mask(cue)
    cue = cue[cr0 : cr1 + 1, cc0 : cc1 + 1]
    h, w = cue.shape
    gh, gw = grid.shape
    br0, bc0, br1, bc1 = board_bbox

    zones = []
    zones.extend((r, c) for r in range(0, br0 - h - 1) for c in range(0, gw - w + 1))
    zones.extend((r, c) for r in range(br1 + 2, gh - h + 1) for c in range(0, gw - w + 1))
    zones.extend((r, c) for r in range(0, gh - h + 1) for c in range(0, bc0 - w - 1))
    zones.extend((r, c) for r in range(0, gh - h + 1) for c in range(bc1 + 2, gw - w + 1))
    random.shuffle(zones)

    for top, left in zones[:500]:
        if not _can_place_bbox(occupied, top, left, h, w, pad=1):
            continue
        _place_mask(grid, cue, color, top, left)
        occupied[top : top + h, left : left + w] |= cue
        return True
    return False


def _valid_instance(grid: np.ndarray, branches: dict[int, int | None]) -> bool:
    bg = get_bg(grid)
    comps = find_components(grid != bg, connectivity=8)
    if not comps:
        return False
    areas = sorted((comp["area"] for comp in comps), reverse=True)
    if len(areas) > 1 and areas[0] <= areas[1]:
        return False

    board = _largest_foreground_component(grid, bg)
    r0, c0, r1, c1 = board["bbox"]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(v) for v in np.unique(crop) if int(v) != int(bg)}
    if colors != set(branches):
        return False

    outside = grid.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    for color, branch in branches.items():
        comps = find_components(outside == int(color), connectivity=8)
        if branch is None and comps:
            return False
        if branch is not None:
            if len(comps) != 1:
                return False
            if comps[0]["area"] % 4 != int(branch):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    made = _make_board_and_branches()
    if made is None:
        return None
    board, branches = made
    n = board.shape[0]

    gh = n + random.randint(8, 14)
    gw = n + random.randint(8, 14)
    top = random.randint(2, gh - n - 2)
    left = random.randint(2, gw - n - 2)

    grid = np.zeros((gh, gw), dtype=int)
    grid[top : top + n, left : left + n] = board
    occupied = np.zeros((gh, gw), dtype=bool)
    occupied[top : top + n, left : left + n] = True
    board_bbox = (top, left, top + n - 1, left + n - 1)

    cue_colors = [color for color, branch in branches.items() if branch is not None]
    random.shuffle(cue_colors)
    for color in cue_colors:
        if not _try_place_cue(grid, occupied, board_bbox, color, int(branches[color])):
            return None

    if not _valid_instance(grid, branches):
        return None

    out = main(grid)
    if out.shape != board.shape:
        return None
    if np.array_equal(out, board):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
