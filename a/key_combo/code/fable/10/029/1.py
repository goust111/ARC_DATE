from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIZE_PAIRS = ((2, 3), (2, 4), (3, 4))
BRANCHES = ("free", "small", "large")
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _detect_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    comps = _find_mono_components(g, bg)

    squares = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        side = r1 - r0 + 1
        if side != c1 - c0 + 1:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == int(comp["color"])):
            squares.append(
                {
                    "color": int(comp["color"]),
                    "size": int(side),
                    "pos": (int(r0), int(c0)),
                }
            )

    assert len(squares) >= 2
    squares.sort(key=lambda s: (s["size"], s["color"]))
    small, large = squares[0], squares[-1]
    assert int(small["size"]) < int(large["size"])

    counts = Counter(map(int, g.ravel()))
    river = None
    for color, _ in counts.most_common():
        if color in (bg, small["color"], large["color"]):
            continue
        river = int(color)
        break
    assert river is not None

    return {"grid": g, "bg": bg, "river": river, "small": small, "large": large}


def _feasible_mask(traversable: np.ndarray, side: int) -> np.ndarray:
    h, w = traversable.shape
    s = int(side)
    assert 1 <= s <= h and 1 <= s <= w
    arr = traversable.astype(np.int16)
    ps = np.pad(arr, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)
    window = ps[s:, s:] - ps[:-s, s:] - ps[s:, :-s] + ps[:-s, :-s]
    return window == s * s


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = -np.ones(mask.shape, dtype=int)
    sr, sc = map(int, start)
    assert bool(mask[sr, sc])
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and int(dist[nr, nc]) < 0:
                dist[nr, nc] = int(dist[r, c] + 1)
                q.append((nr, nc))
    return dist


def _canonical_path(mask: np.ndarray, start: tuple[int, int], hint: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    hr, hc = map(int, hint)
    dist0 = _bfs_dist(mask, (sr, sc))
    reachable = np.argwhere(dist0 >= 0)
    assert len(reachable) > 0

    best = min(
        (
            abs(int(r) - hr) + abs(int(c) - hc),
            -int(dist0[r, c]),
            int(r),
            int(c),
        )
        for r, c in reachable
    )
    target = (best[2], best[3])
    dist1 = _bfs_dist(mask, target)

    cur = (sr, sc)
    out = [cur]
    while cur != target:
        r, c = cur
        candidates = []
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if int(dist0[nr, nc]) != int(dist0[r, c]) + 1:
                continue
            if int(dist1[nr, nc]) < 0:
                continue
            if int(dist0[nr, nc]) + int(dist1[nr, nc]) != int(dist0[target]):
                continue
            candidates.append((nr, nc))
        assert candidates
        candidates.sort(key=lambda p: (abs(p[0] - hr) + abs(p[1] - hc), p[0], p[1]))
        cur = candidates[0]
        out.append(cur)
    return out


def _rect(pos: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = map(int, pos)
    s = int(side)
    return r, c, r + s - 1, c + s - 1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _simulate_duel(
    small_path: list[tuple[int, int]],
    small_side: int,
    large_path: list[tuple[int, int]],
    large_side: int,
) -> dict:
    i_small = 0
    i_large = 0
    active_small = True
    active_large = True
    first_blocked = None
    first_reason = None

    while active_small or active_large:
        cand_small = small_path[i_small + 1] if active_small and i_small + 1 < len(small_path) else None
        cand_large = large_path[i_large + 1] if active_large and i_large + 1 < len(large_path) else None

        move_small = False
        move_large = False

        if active_small and cand_small is not None:
            other_rect = _rect(cand_large, large_side) if active_large and cand_large is not None else _rect(large_path[i_large], large_side)
            move_small = not _overlap(_rect(cand_small, small_side), other_rect)

        if active_large and cand_large is not None:
            other_rect = _rect(cand_small, small_side) if active_small and cand_small is not None else _rect(small_path[i_small], small_side)
            move_large = not _overlap(_rect(cand_large, large_side), other_rect)

        blocked_now = []
        reasons = {}
        if active_small and not move_small:
            active_small = False
            blocked_now.append("small")
            reasons["small"] = "width" if cand_small is None else "overlap"
        if active_large and not move_large:
            active_large = False
            blocked_now.append("large")
            reasons["large"] = "width" if cand_large is None else "overlap"

        if first_blocked is None and blocked_now:
            first_blocked = "both" if len(blocked_now) == 2 else blocked_now[0]
            if first_blocked == "both":
                first_reason = "overlap" if reasons.get("small") == reasons.get("large") == "overlap" else "mixed"
            else:
                first_reason = reasons[first_blocked]

        if move_small:
            i_small += 1
        if move_large:
            i_large += 1
        if not move_small and not move_large:
            break

    return {
        "small_final": small_path[i_small],
        "large_final": large_path[i_large],
        "small_steps": i_small,
        "large_steps": i_large,
        "first_blocked": first_blocked or "none",
        "first_reason": first_reason or "none",
    }


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    scene = _detect_scene(grid)
    g = scene["grid"]
    river = int(scene["river"])
    small = scene["small"]
    large = scene["large"]

    traversable = (g == river) | (g == int(small["color"])) | (g == int(large["color"]))
    feasible_small = _feasible_mask(traversable, int(small["size"]))
    feasible_large = _feasible_mask(traversable, int(large["size"]))
    assert bool(feasible_small[small["pos"]])
    assert bool(feasible_large[large["pos"]])

    small_path = _canonical_path(feasible_small, small["pos"], large["pos"])
    large_path = _canonical_path(feasible_large, large["pos"], small["pos"])
    duel = _simulate_duel(small_path, int(small["size"]), large_path, int(large["size"]))

    out = g.copy()
    sr, sc = map(int, small["pos"])
    ss = int(small["size"])
    lr, lc = map(int, large["pos"])
    ls = int(large["size"])
    out[sr : sr + ss, sc : sc + ss] = river
    out[lr : lr + ls, lc : lc + ls] = river

    fr, fc = duel["small_final"]
    out[fr : fr + ss, fc : fc + ss] = int(small["color"])
    fr, fc = duel["large_final"]
    out[fr : fr + ls, fc : fc + ls] = int(large["color"])

    meta = {
        "bg": int(scene["bg"]),
        "river": river,
        "small_size": ss,
        "large_size": ls,
        "small_start": tuple(map(int, small["pos"])),
        "large_start": tuple(map(int, large["pos"])),
        "small_path_len": len(small_path),
        "large_path_len": len(large_path),
        **duel,
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide two different-sized solid squares toward each other through a winding corridor.

    Input structure:
    - The most common color is the background, and one river color forms a single winding corridor.
    - The two corridor ends contain solid square sliders of different sizes and different colors.
    - Some corridor stretches are roomy while others narrow into chokepoints.

    Rule:
    - Treat the two square colors as temporary river so both sliders can start moving at the same time.
    - Each slider advances one cell per step along the corridor toward the other slider, but only into placements where its full square still fits inside the river.
    - If a slider's next step would fail because the corridor is too narrow or because the two moved squares would overlap, that slider stops forever; the other keeps advancing until it also gets blocked.
    - Refill both original slider locations with the river color, then paint the two sliders at their final resting placements.

    Key insight:
    - The final meeting point is not the corridor midpoint: the decisive event is which square gets blocked first by a size-sensitive chokepoint, after which the other square keeps consuming the remaining path.
    """
    return _solve(np.asarray(grid, dtype=int), return_meta=False)


def _snake_coarse(rows: int, cols: int) -> list[tuple[int, int]]:
    coarse = []
    for r in range(rows):
        cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cols_iter:
            coarse.append((r, c))
    if random.random() < 0.5:
        coarse.reverse()
    return coarse


def _expand_backbone(coarse: list[tuple[int, int]], step: int, margin: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    anchors = [(margin + r * step, margin + c * step) for r, c in coarse]
    path = [anchors[0]]
    runs = []
    run_start = 0
    for a, b in zip(anchors, anchors[1:]):
        r, c = a
        rr, cc = b
        dr = (rr > r) - (rr < r)
        dc = (cc > c) - (cc < c)
        while (r, c) != (rr, cc):
            r += dr
            c += dc
            path.append((r, c))
        runs.append((run_start, len(path) - 1, "h" if dr == 0 else "v"))
        run_start = len(path) - 1
    return path, runs


def _paint_backbone(grid: np.ndarray, path: list[tuple[int, int]], side: int, color: int) -> None:
    s = int(side)
    for r, c in path:
        grid[r : r + s, c : c + s] = int(color)


def _carve_neck(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    run: tuple[int, int, str],
    large_side: int,
    keep_width: int,
    bg: int,
) -> bool:
    a, b, orient = run
    run_len = b - a
    if run_len < max(large_side + 2, 5):
        return False

    neck_len = random.randint(max(large_side, 3), min(run_len - 1, large_side + 2))
    start = random.randint(a + 1, b - neck_len)
    seg = path[start : start + neck_len]

    if orient == "h":
        r = seg[0][0]
        cols = [c for _, c in seg]
        c0 = min(cols)
        c1 = max(cols) + large_side - 1
        grid[r + keep_width : r + large_side, c0 : c1 + 1] = int(bg)
    else:
        c = seg[0][1]
        rows = [r for r, _ in seg]
        r0 = min(rows)
        r1 = max(rows) + large_side - 1
        grid[r0 : r1 + 1, c + keep_width : c + large_side] = int(bg)
    return True


def _choose_colors() -> tuple[int, int, int, int]:
    palette = list(range(10))
    bg = random.choice((0, 3, 5, 8))
    palette.remove(bg)
    river = random.choice(palette)
    palette.remove(river)
    c1, c2 = random.sample(palette, 2)
    return bg, river, c1, c2


def _gen_attempt(desired_branch: str) -> np.ndarray | None:
    small_side, large_side = random.choice(SIZE_PAIRS)
    step = large_side + random.choice((2, 3))
    coarse_rows = random.randint(3, 4)
    coarse_cols = random.randint(3, 4)
    if 1 + (coarse_rows - 1) * step + large_side + 2 > 30:
        return None
    if 1 + (coarse_cols - 1) * step + large_side + 2 > 30:
        return None

    coarse = _snake_coarse(coarse_rows, coarse_cols)
    path, runs = _expand_backbone(coarse, step=step, margin=1)
    h = max(r for r, _ in path) + large_side + 3
    w = max(c for _, c in path) + large_side + 3
    if max(h, w) > 30:
        return None

    bg, river, color_a, color_b = _choose_colors()
    grid = np.full((h, w), bg, dtype=int)
    _paint_backbone(grid, path, small_side, river)
    _paint_backbone(grid, path, large_side, river)

    if desired_branch != "free":
        keep_width = random.randint(1, small_side - 1) if desired_branch == "small" else random.randint(small_side, large_side - 1)
        candidates = [run for run in runs if run[1] - run[0] >= large_side + 2]
        if not candidates:
            return None
        if desired_branch == "small":
            early = [run for run in candidates if run[0] < len(path) // 2]
            candidates = early or candidates
        else:
            late = [run for run in candidates if run[1] > len(path) // 2]
            candidates = late or candidates
        if not _carve_neck(grid, path, random.choice(candidates), large_side, keep_width, bg):
            return None

    if random.random() < 0.5:
        small_pos = path[0]
        large_pos = path[-1]
    else:
        small_pos = path[-1]
        large_pos = path[0]

    sr, sc = small_pos
    lr, lc = large_pos
    grid[sr : sr + small_side, sc : sc + small_side] = color_a
    grid[lr : lr + large_side, lc : lc + large_side] = color_b

    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return None

    if desired_branch == "free" and meta["first_blocked"] != "both":
        return None
    if desired_branch == "small" and meta["first_blocked"] != "small":
        return None
    if desired_branch == "large" and meta["first_blocked"] != "large":
        return None

    river_mask = (grid == river) | (grid == color_a) | (grid == color_b)
    if np.any(river_mask, axis=1).sum() < max(6, int(0.6 * h)):
        return None
    if np.any(river_mask, axis=0).sum() < max(6, int(0.6 * w)):
        return None
    if meta["small_path_len"] < 5 or meta["large_path_len"] < 3:
        return None
    if meta["small_steps"] == 0 or meta["large_steps"] == 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        desired_branch = random.choices(BRANCHES, weights=(3, 3, 4), k=1)[0]
        grid = _gen_attempt(desired_branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-slider corridor")
