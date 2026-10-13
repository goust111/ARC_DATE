import random
from collections import Counter

import numpy as np


BG = 7  # orange
PALETTE = [0, 1, 2, 3, 4, 5, 6, 8, 9]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(int(v) for v in grid.ravel()).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_pools(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    pools: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if set(comp) != border:
                continue
            if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            pools.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": color})
    return pools


def _source_direction(h: int, w: int, r: int, c: int) -> tuple[int, int] | None:
    if r == 0 and 0 < c < w - 1:
        return 1, 0
    if r == h - 1 and 0 < c < w - 1:
        return -1, 0
    if c == 0 and 0 < r < h - 1:
        return 0, 1
    if c == w - 1 and 0 < r < h - 1:
        return 0, -1
    return None


def _ray_hit(
    grid: np.ndarray,
    pools: list[dict[str, int]],
    pool_by_border: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    bg: int,
) -> tuple[list[tuple[int, int]], int | None, int]:
    h, w = grid.shape
    path: list[tuple[int, int]] = []
    cr, cc = r + dr, c + dc
    while 0 <= cr < h and 0 <= cc < w:
        if (cr, cc) in pool_by_border:
            return path, pool_by_border[(cr, cc)], len(path) + 1
        if int(grid[cr, cc]) != bg:
            return path, None, len(path)
        path.append((cr, cc))
        if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
            return path, None, len(path)
        cr += dr
        cc += dc
    return path, None, len(path)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rays are fired inward from single border markers. A ray paints a one-cell
    line across background cells until it reaches a pool frame or the far edge.
    Only same-colored rays can fill a pool's interior; different-colored rays
    stop at the frame without filling it. If multiple matching rays target the
    same pool, the one with the shortest travel distance fills first.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    pools = _find_pools(g, bg)

    pool_by_border: dict[tuple[int, int], int] = {}
    for idx, pool in enumerate(pools):
        for rr in range(pool["r0"], pool["r1"] + 1):
            for cc in range(pool["c0"], pool["c1"] + 1):
                if rr in (pool["r0"], pool["r1"]) or cc in (pool["c0"], pool["c1"]):
                    pool_by_border[(rr, cc)] = idx

    out = g.copy()
    candidates: list[tuple[int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if int(g[r, c]) == bg:
                continue
            if (r, c) in pool_by_border:
                continue
            direction = _source_direction(h, w, r, c)
            if direction is None:
                continue
            dr, dc = direction
            path, hit_pool, dist = _ray_hit(g, pools, pool_by_border, r, c, dr, dc, bg)
            src_color = int(g[r, c])
            for pr, pc in path:
                out[pr, pc] = src_color
            if hit_pool is not None and src_color == pools[hit_pool]["color"]:
                candidates.append((dist, hit_pool, src_color))

    claimed: set[int] = set()
    for _dist, pool_idx, color in sorted(candidates):
        if pool_idx in claimed:
            continue
        pool = pools[pool_idx]
        out[pool["r0"] + 1 : pool["r1"], pool["c0"] + 1 : pool["c1"]] = color
        claimed.add(pool_idx)

    return out


def _draw_pool(grid: np.ndarray, pool: dict[str, int]) -> None:
    r0, c0, r1, c1, color = pool["r0"], pool["c0"], pool["r1"], pool["c1"], pool["color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_pools(rng: random.Random, h: int, w: int) -> list[dict[str, int]] | None:
    count = rng.randint(1, 4)
    colors = rng.sample(PALETTE, count)
    pools: list[dict[str, int]] = []
    for color in colors:
        placed = False
        for _ in range(80):
            ph = rng.randint(4, min(7, h - 4))
            pw = rng.randint(4, min(7, w - 4))
            r0 = rng.randint(2, h - ph - 2)
            c0 = rng.randint(2, w - pw - 2)
            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            if any(
                not (r1 + 2 < p["r0"] or p["r1"] + 2 < r0 or c1 + 2 < p["c0"] or p["c1"] + 2 < c0)
                for p in pools
            ):
                continue
            pools.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": color})
            placed = True
            break
        if not placed:
            return None
    return pools


def _pool_candidates(
    pools: list[dict[str, int]],
    h: int,
    w: int,
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    hit_candidates: list[dict[str, int | str]] = []
    miss_candidates: list[dict[str, int | str]] = []
    for c in range(1, w - 1):
        blockers = [p for p in pools if p["c0"] <= c <= p["c1"]]
        if blockers:
            top_pool = min(blockers, key=lambda p: p["r0"])
            bot_pool = max(blockers, key=lambda p: p["r1"])
            hit_candidates.append({"edge": "top", "pos": c, "pool": pools.index(top_pool), "dist": top_pool["r0"]})
            hit_candidates.append({"edge": "bottom", "pos": c, "pool": pools.index(bot_pool), "dist": h - 1 - bot_pool["r1"]})
        else:
            miss_candidates.append({"edge": "top", "pos": c})
            miss_candidates.append({"edge": "bottom", "pos": c})
    for r in range(1, h - 1):
        blockers = [p for p in pools if p["r0"] <= r <= p["r1"]]
        if blockers:
            left_pool = min(blockers, key=lambda p: p["c0"])
            right_pool = max(blockers, key=lambda p: p["c1"])
            hit_candidates.append({"edge": "left", "pos": r, "pool": pools.index(left_pool), "dist": left_pool["c0"]})
            hit_candidates.append({"edge": "right", "pos": r, "pool": pools.index(right_pool), "dist": w - 1 - right_pool["c1"]})
        else:
            miss_candidates.append({"edge": "left", "pos": r})
            miss_candidates.append({"edge": "right", "pos": r})
    return hit_candidates, miss_candidates


def _source_cell(edge: str, pos: int, h: int, w: int) -> tuple[int, int]:
    if edge == "top":
        return 0, pos
    if edge == "bottom":
        return h - 1, pos
    if edge == "left":
        return pos, 0
    return pos, w - 1


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 21)
    w = rng.randint(12, 21)
    pools = _place_pools(rng, h, w)
    if pools is None:
        return None

    hit_candidates, miss_candidates = _pool_candidates(pools, h, w)
    by_pool: dict[int, list[dict[str, int | str]]] = {i: [] for i in range(len(pools))}
    for cand in hit_candidates:
        by_pool[int(cand["pool"])].append(cand)
    if not any(by_pool.values()):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for pool in pools:
        _draw_pool(grid, pool)

    used_cells: set[tuple[int, int]] = set()
    sources: list[tuple[int, int, int]] = []

    scenario = rng.randint(0, 3)

    match_pool = rng.choice([idx for idx, vals in by_pool.items() if vals])
    match_cand = rng.choice(by_pool[match_pool])
    mr, mc = _source_cell(str(match_cand["edge"]), int(match_cand["pos"]), h, w)
    used_cells.add((mr, mc))
    sources.append((mr, mc, pools[match_pool]["color"]))

    mismatch_pool_choices = [idx for idx, vals in by_pool.items() if vals and idx != match_pool]
    if not mismatch_pool_choices:
        mismatch_pool_choices = [match_pool]
    mismatch_pool = rng.choice(mismatch_pool_choices)
    mismatch_options = [c for c in by_pool[mismatch_pool] if _source_cell(str(c["edge"]), int(c["pos"]), h, w) not in used_cells]
    if not mismatch_options:
        return None
    mismatch_cand = rng.choice(mismatch_options)
    rr, cc = _source_cell(str(mismatch_cand["edge"]), int(mismatch_cand["pos"]), h, w)
    used_cells.add((rr, cc))
    wrong_colors = [c for c in PALETTE if c != pools[mismatch_pool]["color"]]
    sources.append((rr, cc, rng.choice(wrong_colors)))

    if scenario in (1, 2) and miss_candidates:
        miss_options = [c for c in miss_candidates if _source_cell(str(c["edge"]), int(c["pos"]), h, w) not in used_cells]
        if miss_options:
            miss = rng.choice(miss_options)
            rr, cc = _source_cell(str(miss["edge"]), int(miss["pos"]), h, w)
            used_cells.add((rr, cc))
            sources.append((rr, cc, rng.choice(PALETTE)))

    if scenario in (2, 3):
        contest_pools = []
        for idx, vals in by_pool.items():
            usable = [c for c in vals if _source_cell(str(c["edge"]), int(c["pos"]), h, w) not in used_cells]
            dists = {int(c["dist"]) for c in usable}
            if len(usable) >= 1 and (idx != match_pool or len({int(c["dist"]) for c in by_pool[idx]}) >= 2):
                if idx == match_pool:
                    contest_pools.append(idx)
                elif len(dists) >= 1:
                    contest_pools.append(idx)
        if contest_pools:
            cp = rng.choice(contest_pools)
            usable = [c for c in by_pool[cp] if _source_cell(str(c["edge"]), int(c["pos"]), h, w) not in used_cells]
            if cp == match_pool:
                usable = [c for c in usable if int(c["dist"]) != int(match_cand["dist"])]
            if usable:
                extra = rng.choice(usable)
                rr, cc = _source_cell(str(extra["edge"]), int(extra["pos"]), h, w)
                used_cells.add((rr, cc))
                sources.append((rr, cc, pools[cp]["color"]))

    if len(sources) < 2:
        return None

    for r, c, color in sources:
        grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
