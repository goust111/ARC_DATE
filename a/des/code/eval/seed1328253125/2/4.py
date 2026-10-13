import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
MAROON = 9
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _region_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict] = []
    for comp in _components(g != GREEN):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        region = set(comp)
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREEN, MAROON)]
        arrows = [(r, c) for r, c in comp if int(g[r, c]) == MAROON]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        assert len(arrows) == 1, "each enclosed region must contain exactly one arrow"
        seed = seeds[0]
        arrow = arrows[0]
        delta = (int(arrow[0] - seed[0]), int(arrow[1] - seed[1]))
        assert delta in DIR_TO_INDEX, "arrow must be orthogonally adjacent to the seed"
        regions.append(
            {
                "region": region,
                "seed": seed,
                "arrow": arrow,
                "color": int(g[seed]),
                "steps": sum(1 for r, c in comp if int(g[r, c]) == BLACK),
                "dir_idx": DIR_TO_INDEX[delta],
            }
        )
    regions.sort(key=lambda item: (item["seed"][0], item["seed"][1]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one vine per enclosed region.

    - Green pixels are walls.
    - A single colored seed and a single maroon arrow sit inside each walled region.
    - The arrow gives the initial direction from seed to arrow.
    - The vine paints exactly as many black cells as originally existed in that region,
      turning clockwise whenever its current heading is blocked.
    - Arrow pixels are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[g == MAROON] = BLACK

    for spec in _region_specs(g):
        current = spec["arrow"]
        dir_idx = int(spec["dir_idx"])
        color = int(spec["color"])
        region = spec["region"]

        for _ in range(int(spec["steps"])):
            advanced = False
            for turn in range(4):
                next_dir = (dir_idx + turn) % 4
                dr, dc = DIRS[next_dir]
                nr = current[0] + dr
                nc = current[1] + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) not in region:
                    continue
                if int(g[nr, nc]) != BLACK:
                    continue
                if int(out[nr, nc]) != BLACK:
                    continue
                out[nr, nc] = color
                current = (int(nr), int(nc))
                dir_idx = int(next_dir)
                advanced = True
                break
            if not advanced:
                break

    return out


GEN_RETRIES = 500
PLACE_TRIES = 80
MIN_OBJECTS = 1
MAX_OBJECTS = 4
MAX_SIDE = 36
MIN_INNER = 3
PAD = 1
MAX_BULGE_DEPTH = 2
MAX_BULGE_SPAN = 3
MAX_BULGES = 4
RESERVE_MARGIN = PAD + MAX_BULGE_DEPTH

PLACEMENTS = [
    ("tl", 1),
    ("tl", 2),
    ("tr", 2),
    ("tr", 3),
    ("br", 3),
    ("br", 0),
    ("bl", 0),
    ("bl", 1),
]


def _inner_bounds(n_objects: int) -> tuple[int, int]:
    if n_objects >= 4:
        return MIN_INNER, 5
    if n_objects == 3:
        return MIN_INNER, 6
    return MIN_INNER, 8


def _room_seed_cell(r0: int, c0: int, inner_h: int, inner_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return r0 + 1, c0 + 1
    if corner == "tr":
        return r0 + 1, c0 + inner_w
    if corner == "br":
        return r0 + inner_h, c0 + inner_w
    return r0 + inner_h, c0 + 1


def _reserve_box(r0: int, c0: int, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    top = r0 - RESERVE_MARGIN
    left = c0 - RESERVE_MARGIN
    bottom = r0 + outer_h + RESERVE_MARGIN
    right = c0 + outer_w + RESERVE_MARGIN
    return top, left, bottom, right


def _draw_ring(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> None:
    grid[r0, c0 : c0 + outer_w] = GREEN
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = GREEN
    grid[r0 : r0 + outer_h, c0] = GREEN
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = GREEN


def _add_bulges(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, rng: random.Random) -> None:
    bulge_count = rng.randint(1, MAX_BULGES)
    for _ in range(bulge_count):
        side = rng.choice(("top", "bottom", "left", "right"))
        depth = rng.randint(1, MAX_BULGE_DEPTH)
        if side in ("top", "bottom"):
            max_span = min(MAX_BULGE_SPAN, outer_w - 2)
            span = rng.randint(1, max_span)
            start = rng.randint(1, outer_w - span - 1)
            cols = slice(c0 + start, c0 + start + span)
            if side == "top":
                for d in range(depth):
                    grid[r0 - 1 - d, cols] = GREEN
            else:
                for d in range(depth):
                    grid[r0 + outer_h + d, cols] = GREEN
        else:
            max_span = min(MAX_BULGE_SPAN, outer_h - 2)
            span = rng.randint(1, max_span)
            start = rng.randint(1, outer_h - span - 1)
            rows = slice(r0 + start, r0 + start + span)
            if side == "left":
                for d in range(depth):
                    grid[rows, c0 - 1 - d] = GREEN
            else:
                for d in range(depth):
                    grid[rows, c0 + outer_w + d] = GREEN


def _place_room(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    n_objects: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    min_inner, max_inner = _inner_bounds(n_objects)
    inner_h = rng.randint(min_inner, max_inner)
    inner_w = rng.randint(min_inner, max_inner)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    corner, dir_idx = rng.choice(PLACEMENTS)

    positions = []
    row_lo = RESERVE_MARGIN
    row_hi = h - outer_h - RESERVE_MARGIN
    col_lo = RESERVE_MARGIN
    col_hi = w - outer_w - RESERVE_MARGIN
    if row_lo > row_hi or col_lo > col_hi:
        return False
    for _ in range(PLACE_TRIES):
        positions.append((rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)))

    for r0, c0 in positions:
        top, left, bottom, right = _reserve_box(r0, c0, outer_h, outer_w)
        if bool(reserved[top:bottom, left:right].any()):
            continue

        _draw_ring(grid, r0, c0, outer_h, outer_w)
        _add_bulges(grid, r0, c0, outer_h, outer_w, rng)

        seed_r, seed_c = _room_seed_cell(r0, c0, inner_h, inner_w, corner)
        dr, dc = DIRS[dir_idx]
        arrow_r = seed_r + dr
        arrow_c = seed_c + dc
        grid[seed_r, seed_c] = color
        grid[arrow_r, arrow_c] = MAROON

        reserved[top:bottom, left:right] = True
        return True

    return False


def _validate_generated(grid: np.ndarray, expected_objects: int) -> bool:
    try:
        regions = _region_specs(grid)
        if len(regions) != expected_objects:
            return False
        colors = [spec["color"] for spec in regions]
        if len(colors) != len(set(colors)):
            return False
        if any(spec["steps"] < 4 for spec in regions):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if bool(np.any(out == MAROON)):
            return False

        for spec in regions:
            region = spec["region"]
            arrow = spec["arrow"]
            color = spec["color"]
            black_cells = [(r, c) for r, c in region if int(out[r, c]) == BLACK]
            if black_cells != [arrow]:
                return False
            if int(out[spec["seed"]]) != color:
                return False
            if sum(1 for r, c in region if int(out[r, c]) == color) != len(region) - 1:
                return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    min_side = 12 + 4 * n_objects
    h = rng.randint(min_side, MAX_SIDE)
    w = rng.randint(min_side, MAX_SIDE)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    colors = rng.sample(SEED_COLORS, n_objects)
    for color in colors:
        if not _place_room(grid, reserved, color, n_objects, rng):
            return None

    if not _validate_generated(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
