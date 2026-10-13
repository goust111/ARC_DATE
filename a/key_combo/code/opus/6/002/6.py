import random
from collections import deque

import numpy as np


BG = 0
START = 5
HIT = 4
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
ODD_TURN = [1, 0, 3, 2]   # N->E, E->N, S->W, W->S
EVEN_TURN = [3, 2, 1, 0]  # N->W, E->S, S->E, W->N
PALETTE = [1, 2, 3, 6, 7, 8, 9]
MAX_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_cycle_blocks(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != START)
    blocks = []
    for comp in _components(mask):
        vals = {int(grid[r, c]) for r, c in comp}
        if len(vals) != 1 or len(comp) != 4:
            continue
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) != 2 or len(cols) != 2:
            continue
        if rows[1] != rows[0] + 1 or cols[1] != cols[0] + 1:
            continue
        blocks.append(
            {
                "color": int(next(iter(vals))),
                "coords": set(comp),
                "bbox": (rows[0], cols[0], rows[1], cols[1]),
            }
        )
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks


def _find_start(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    pts = np.argwhere(grid == START)
    assert len(pts) == 1, "expected exactly one gray start marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0 and 0 < c < w - 1:
        return (r, c), 2
    if r == h - 1 and 0 < c < w - 1:
        return (r, c), 0
    if c == 0 and 0 < r < h - 1:
        return (r, c), 1
    if c == w - 1 and 0 < r < h - 1:
        return (r, c), 3
    raise AssertionError("start marker must be on a non-corner border cell")


def _extract_cycle(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    blocks = _find_cycle_blocks(grid)
    assert 2 <= len(blocks) <= 3, "expected 2 or 3 top-left 2x2 cycle blocks"
    cycle = [block["color"] for block in blocks]
    coords = set()
    for block in blocks:
        coords |= block["coords"]
    return cycle, coords


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.array(grid, dtype=int, copy=True)
    cycle, block_cells = _extract_cycle(out)
    start, dir_idx = _find_start(out)
    h, w = out.shape

    draw_step: dict[tuple[int, int], int] = {}
    collision_kinds: list[str] = []
    path_cells: set[tuple[int, int]] = set()

    current = start
    cycle_idx = 0
    step_idx = 0
    max_iters = h * w * 8

    for _ in range(max_iters):
        dr, dc = DIRS[dir_idx]
        nr, nc = current[0] + dr, current[1] + dc

        if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
            step_idx += 1
            color = cycle[cycle_idx % len(cycle)]
            cycle_idx += 1
            out[nr, nc] = color
            draw_step[(nr, nc)] = step_idx
            path_cells.add((nr, nc))
            current = (nr, nc)
            continue

        if not _in_bounds(out.shape, nr, nc):
            collision_kinds.append("edge")
            if current != start:
                step_idx += 1
                out[current] = HIT
                draw_step[current] = step_idx
                path_cells.add(current)
        elif (nr, nc) in block_cells:
            collision_kinds.append("block")
        else:
            collision_kinds.append("trail")

        turn = ODD_TURN if len(collision_kinds) % 2 == 1 else EVEN_TURN
        next_dir = turn[dir_idx]
        tr = current[0] + DIRS[next_dir][0]
        tc = current[1] + DIRS[next_dir][1]
        if not _in_bounds(out.shape, tr, tc):
            stop_reason = "turn_oob"
            break
        if int(out[tr, tc]) != BG:
            stop_reason = "turn_occupied"
            break
        dir_idx = next_dir
    else:
        stop_reason = "iteration_cap"

    meta = {
        "cycle": cycle,
        "block_cells": block_cells,
        "draw_step": draw_step,
        "path_cells": path_cells,
        "collision_kinds": collision_kinds,
        "boundary_collisions": collision_kinds.count("edge"),
        "stop_reason": stop_reason,
    }
    return out, meta


def _fill_regions(grid: np.ndarray, draw_step: dict[tuple[int, int], int]) -> tuple[np.ndarray, dict]:
    out = np.array(grid, dtype=int, copy=True)
    comps = _components(out == BG)
    if not comps:
        return out, {"filled_regions": [], "largest_unique": True, "total_filled": 0}

    areas = sorted((len(comp), idx) for idx, comp in enumerate(comps))
    largest_area = areas[-1][0]
    largest_count = sum(area == largest_area for area, _ in areas)
    exterior_idx = max(range(len(comps)), key=lambda idx: len(comps[idx]))

    filled_regions = []
    total_filled = 0
    for idx, comp in enumerate(comps):
        if idx == exterior_idx:
            continue
        touching = []
        touching_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in draw_step:
                    touching.append((draw_step[(nr, nc)], nr, nc))
                    touching_colors.add(int(out[nr, nc]))
        if not touching:
            continue
        last_step, rr, cc = max(touching)
        fill_color = int(out[rr, cc])
        for r, c in comp:
            out[r, c] = fill_color
        total_filled += len(comp)
        filled_regions.append(
            {
                "area": len(comp),
                "fill_color": fill_color,
                "adjacent_colors": sorted(touching_colors),
                "chosen_step": last_step,
            }
        )

    meta = {
        "filled_regions": filled_regions,
        "largest_unique": largest_count == 1,
        "total_filled": total_filled,
    }
    return out, meta


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    traced, trace_meta = _simulate(grid)
    filled, fill_meta = _fill_regions(traced, trace_meta["draw_step"])
    meta = dict(trace_meta)
    meta.update(fill_meta)
    meta["path_len"] = len(trace_meta["path_cells"])
    return filled, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate an alternately-turning cycle-colored trail, then fill the black pockets it encloses.

    Input structure:
    - The background is black.
    - The upper-left contains 2 or 3 monochrome 2x2 blocks; their left-to-right order defines the color cycle.
    - One gray border pixel is the unique start marker; every other non-black input cell belongs to the cycle blocks.

    Rule:
    - Start from the gray marker and move one cell inward at a time through black cells. Each successful step paints the next cycle color.
    - Whenever the next straight cell is blocked, count one collision. If that blocked step would leave the grid, repaint the current trail cell yellow.
    - Odd collisions turn by the table N->E, E->N, S->W, W->S; even collisions turn by N->W, E->S, S->E, W->N. If the turned next cell is also blocked, stop.
    - Treat all non-black cells as barriers, split the remaining black cells into 4-connected regions, keep the largest region unchanged, and fill every smaller region with the color of the latest-painted adjacent trail cell.

    Key insight:
    - The output depends on both the trail geometry and its time order: a pocket inherits the color of the last neighboring trail pixel, not a global contour color.
    """
    out, _meta = _solve_with_meta(np.asarray(grid, dtype=int))
    return out


def _edge_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    pts = []
    for c in range(1, w - 1):
        if int(grid[0, c]) == BG:
            pts.append((0, c))
        if int(grid[h - 1, c]) == BG:
            pts.append((h - 1, c))
    for r in range(1, h - 1):
        if int(grid[r, 0]) == BG:
            pts.append((r, 0))
        if int(grid[r, w - 1]) == BG:
            pts.append((r, w - 1))
    return pts


def _place_blocks(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, list[int]]:
    grid = np.zeros((h, w), dtype=int)
    n_blocks = rng.choice([2, 3])
    row0 = rng.randint(0, min(3, h - 3))
    c = rng.randint(0, 1)
    cols = []
    for i in range(n_blocks):
        if i > 0:
            c += rng.randint(1, 3)
        cols.append(c)
        c += 1
    if cols[-1] + 1 >= min(w - 2, 12):
        raise ValueError("cycle blocks do not fit the top-left zone")
    colors = rng.sample(PALETTE, n_blocks)
    for color, c0 in zip(colors, cols):
        grid[row0 : row0 + 2, c0 : c0 + 2] = color
    return grid, colors


def _region_is_informative(meta: dict) -> bool:
    for region in meta["filled_regions"]:
        if len(region["adjacent_colors"]) >= 2:
            return True
    return False


def _valid_instance(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    cycle, block_cells = _extract_cycle(inp)
    start, _ = _find_start(inp)

    if meta["stop_reason"] != "turn_occupied":
        return False
    if meta["path_len"] < 24 or meta["path_len"] > 72:
        return False
    if len(meta["collision_kinds"]) < 6:
        return False
    if meta["boundary_collisions"] < 2:
        return False
    if meta["collision_kinds"].count("block") > 1:
        return False
    if not meta["filled_regions"] or meta["total_filled"] < 2:
        return False
    if meta["total_filled"] > 30:
        return False
    if len(meta["filled_regions"]) > 2:
        return False
    if not meta["largest_unique"]:
        return False
    if not _region_is_informative(meta):
        return False
    if int(out[start]) != START:
        return False
    for r, c in block_cells:
        if int(out[r, c]) != int(inp[r, c]):
            return False
    if len(cycle) != len(set(cycle)):
        return False
    if np.array_equal(inp, out):
        return False
    changed = int(np.sum(inp != out))
    return 12 <= changed <= 85


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 22)
    w = rng.randint(15, 22)
    try:
        base, _colors = _place_blocks(rng, h, w)
    except ValueError:
        return None

    starts = _edge_candidates(base)
    rng.shuffle(starts)
    for start in starts[: min(24, len(starts))]:
        grid = np.array(base, copy=True)
        grid[start] = START
        try:
            out, meta = _solve_with_meta(grid)
        except AssertionError:
            continue
        if _valid_instance(grid, out, meta):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
