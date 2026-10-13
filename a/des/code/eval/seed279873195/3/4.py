import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
ORANGE = 7
BLOCK_COLORS = (1, 2, 3, 4, 6, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (22, 34)
GRID_W_RANGE = (22, 34)
WALL_COUNT_RANGE = (2, 4)
OPEN_WALL_H_RANGE = (4, 6)
OPEN_WALL_W_RANGE = (4, 7)
CLOSED_WALL_H_RANGE = (4, 7)
CLOSED_WALL_W_RANGE = (4, 8)
BLOCK_H_RANGE = (4, 6)
BLOCK_W_RANGE = (4, 8)
CORRIDOR_RANGE = (1, 4)
EXTRA_BLOCK_GOAL = (2, 4)
PLACEMENT_TRIES = 160
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _gap_side(
    gap: tuple[int, int] | None,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> str | None:
    if gap is None:
        return None
    gr, gc = gap
    is_top = gr == r0 and c0 < gc < c1
    is_bottom = gr == r1 and c0 < gc < c1
    is_left = gc == c0 and r0 < gr < r1
    is_right = gc == c1 and r0 < gr < r1
    matches = [side for side, ok in zip(SIDES, (is_top, is_bottom, is_left, is_right)) if ok]
    if len(matches) != 1:
        return None
    return matches[0]


def _detect_walls(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    walls = []
    for comp in _component_coords(g == WALL):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = _border_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set <= border:
            continue
        missing = sorted(border - comp_set)
        if len(missing) not in (0, 1):
            continue
        gap = missing[0] if missing else None
        side = _gap_side(gap, r0, c0, r1, c1)
        if gap is not None and side is None:
            continue
        walls.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
            }
        )
    return walls


def _ray_color(grid: np.ndarray, gap: tuple[int, int], side: str) -> int | None:
    h, w = grid.shape
    dr, dc = SIDE_TO_DIR[side]
    r, c = gap
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        value = int(grid[r, c])
        if value == BG:
            continue
        if value == WALL:
            return None
        return value


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for wall in _detect_walls(g):
        gap = wall["gap"]
        side = wall["side"]
        if gap is None or side is None:
            continue
        fill_color = _ray_color(g, gap, side)
        if fill_color is None:
            continue
        r0, c0, r1, c1 = wall["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color
        out[gap] = ORANGE
    return out


def _bbox_free(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = reserved.shape
    return (
        0 <= r0 <= r1 < h
        and 0 <= c0 <= c1 < w
        and not bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())
    )


def _mark_bbox(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _mark_corridor(
    reserved: np.ndarray,
    gap: tuple[int, int],
    side: str,
    block_bbox: tuple[int, int, int, int],
) -> None:
    gr, gc = gap
    br0, bc0, br1, bc1 = block_bbox
    if side == "top":
        reserved[br1 + 1 : gr, gc] = True
    elif side == "bottom":
        reserved[gr + 1 : br0, gc] = True
    elif side == "left":
        reserved[gr, bc1 + 1 : gc] = True
    else:
        reserved[gr, gc + 1 : bc0] = True


def _draw_wall(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap: tuple[int, int] | None,
) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for r, c in _border_cells(r0, c0, r1, c1):
        if gap is not None and (r, c) == gap:
            continue
        grid[r, c] = WALL


def _sample_gap(
    rng: random.Random,
    side: str,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "top":
        return (r0, rng.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, rng.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (rng.randint(r0 + 1, r1 - 1), c0)
    return (rng.randint(r0 + 1, r1 - 1), c1)


def _place_open_wall(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    color: int,
) -> dict | None:
    h_grid, w_grid = grid.shape
    for _ in range(PLACEMENT_TRIES):
        side = rng.choice(SIDES)
        wh = rng.randint(*OPEN_WALL_H_RANGE)
        ww = rng.randint(*OPEN_WALL_W_RANGE)
        bh = rng.randint(*BLOCK_H_RANGE)
        bw = rng.randint(*BLOCK_W_RANGE)
        corridor = rng.randint(*CORRIDOR_RANGE)

        min_r0 = 1
        max_r0 = h_grid - wh - 1
        min_c0 = 1
        max_c0 = w_grid - ww - 1
        if side == "top":
            min_r0 = bh + corridor + 1
        elif side == "bottom":
            max_r0 = h_grid - wh - bh - corridor - 2
        elif side == "left":
            min_c0 = bw + corridor + 1
        else:
            max_c0 = w_grid - ww - bw - corridor - 2
        if min_r0 > max_r0 or min_c0 > max_c0:
            continue

        r0 = rng.randint(min_r0, max_r0)
        c0 = rng.randint(min_c0, max_c0)
        r1 = r0 + wh - 1
        c1 = c0 + ww - 1
        if not _bbox_free(reserved, r0, c0, r1, c1):
            continue

        gap = _sample_gap(rng, side, r0, c0, wh, ww)
        gr, gc = gap

        if side == "top":
            br1 = r0 - corridor - 1
            br0 = br1 - bh + 1
            bc0_lo = max(0, gc - bw + 1)
            bc0_hi = min(gc, w_grid - bw)
            if bc0_lo > bc0_hi:
                continue
            bc0 = rng.randint(bc0_lo, bc0_hi)
            bc1 = bc0 + bw - 1
        elif side == "bottom":
            br0 = r1 + corridor + 1
            br1 = br0 + bh - 1
            bc0_lo = max(0, gc - bw + 1)
            bc0_hi = min(gc, w_grid - bw)
            if bc0_lo > bc0_hi:
                continue
            bc0 = rng.randint(bc0_lo, bc0_hi)
            bc1 = bc0 + bw - 1
        elif side == "left":
            bc1 = c0 - corridor - 1
            bc0 = bc1 - bw + 1
            br0_lo = max(0, gr - bh + 1)
            br0_hi = min(gr, h_grid - bh)
            if br0_lo > br0_hi:
                continue
            br0 = rng.randint(br0_lo, br0_hi)
            br1 = br0 + bh - 1
        else:
            bc0 = c1 + corridor + 1
            bc1 = bc0 + bw - 1
            br0_lo = max(0, gr - bh + 1)
            br0_hi = min(gr, h_grid - bh)
            if br0_lo > br0_hi:
                continue
            br0 = rng.randint(br0_lo, br0_hi)
            br1 = br0 + bh - 1

        if not _bbox_free(reserved, br0, bc0, br1, bc1):
            continue

        grid[br0 : br1 + 1, bc0 : bc1 + 1] = color
        _draw_wall(grid, r0, c0, wh, ww, gap)

        _mark_bbox(reserved, r0, c0, r1, c1)
        _mark_bbox(reserved, br0, bc0, br1, bc1)
        _mark_corridor(reserved, gap, side, (br0, bc0, br1, bc1))

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "side": side,
            "fill_color": color,
        }
    return None


def _place_closed_wall(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
) -> dict | None:
    h_grid, w_grid = grid.shape
    for _ in range(PLACEMENT_TRIES):
        wh = rng.randint(*CLOSED_WALL_H_RANGE)
        ww = rng.randint(*CLOSED_WALL_W_RANGE)
        if h_grid - wh - 2 < 1 or w_grid - ww - 2 < 1:
            continue
        r0 = rng.randint(1, h_grid - wh - 2)
        c0 = rng.randint(1, w_grid - ww - 2)
        r1 = r0 + wh - 1
        c1 = c0 + ww - 1
        if not _bbox_free(reserved, r0, c0, r1, c1):
            continue
        _draw_wall(grid, r0, c0, wh, ww, None)
        _mark_bbox(reserved, r0, c0, r1, c1)
        return {"bbox": (r0, c0, r1, c1), "gap": None}
    return None


def _place_extra_block(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    color: int,
) -> bool:
    h_grid, w_grid = grid.shape
    for _ in range(PLACEMENT_TRIES):
        bh = rng.randint(*BLOCK_H_RANGE)
        bw = rng.randint(*BLOCK_W_RANGE)
        if h_grid - bh - 1 < 1 or w_grid - bw - 1 < 1:
            continue
        r0 = rng.randint(1, h_grid - bh - 1)
        c0 = rng.randint(1, w_grid - bw - 1)
        r1 = r0 + bh - 1
        c1 = c0 + bw - 1
        if not _bbox_free(reserved, r0, c0, r1, c1):
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        _mark_bbox(reserved, r0, c0, r1, c1)
        return True
    return False


def _validate_candidate(
    grid: np.ndarray,
    open_specs: list[dict],
    closed_specs: list[dict],
) -> bool:
    walls = _detect_walls(grid)
    if len(walls) != len(open_specs) + len(closed_specs):
        return False

    detected_open = [wall for wall in walls if wall["gap"] is not None]
    if len(detected_open) != len(open_specs):
        return False

    expected = {spec["gap"]: spec["fill_color"] for spec in open_specs}
    for wall in detected_open:
        ray_color = _ray_color(grid, wall["gap"], wall["side"])
        if expected.get(wall["gap"]) != ray_color:
            return False

    out = main(grid)
    for spec in open_specs:
        r0, c0, r1, c1 = spec["bbox"]
        gr, gc = spec["gap"]
        fill_color = spec["fill_color"]
        if int(out[gr, gc]) != ORANGE:
            return False
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not bool(np.all(interior == fill_color)):
            return False

    for spec in closed_specs:
        r0, c0, r1, c1 = spec["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not bool(np.all(interior == BG)):
            return False

    return True


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    wall_count = rng.randint(*WALL_COUNT_RANGE)
    open_count = rng.randint(1, wall_count - 1)
    total_blocks = rng.randint(
        max(EXTRA_BLOCK_GOAL[0], open_count),
        min(EXTRA_BLOCK_GOAL[1], len(BLOCK_COLORS)),
    )
    colors = rng.sample(list(BLOCK_COLORS), total_blocks)

    open_specs: list[dict] = []
    closed_specs: list[dict] = []

    for color in colors[:open_count]:
        spec = _place_open_wall(grid, reserved, rng, color)
        if spec is None:
            return None
        open_specs.append(spec)

    for _ in range(wall_count - open_count):
        spec = _place_closed_wall(grid, reserved, rng)
        if spec is None:
            return None
        closed_specs.append(spec)

    for color in colors[open_count:]:
        if not _place_extra_block(grid, reserved, rng, color):
            return None

    if not _validate_candidate(grid, open_specs, closed_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
