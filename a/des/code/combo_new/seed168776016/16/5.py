import random

import numpy as np


BG = 0
MARKER = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
MAX_N = 6
GEN_RETRIES = 400


def _is_corner(h: int, w: int, r: int, c: int) -> bool:
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _clockwise_key(h: int, w: int, r: int, c: int) -> int:
    if r == 0:
        return c
    if c == w - 1:
        return (w - 1) + r
    if r == h - 1:
        return (w - 1) + (h - 1) + (w - 1 - c)
    return (w - 1) + (h - 1) + (w - 1) + (h - 1 - r)


def _seed_side_and_count(grid: np.ndarray, r: int, c: int) -> tuple[str, int]:
    h, w = grid.shape
    if r == 0:
        count = 0 if c in (0, w - 1) else int(np.count_nonzero(grid[1 : h - 1, c] == MARKER))
        return "top", count
    if c == w - 1:
        count = 0 if r in (0, h - 1) else int(np.count_nonzero(grid[r, 1 : w - 1] == MARKER))
        return "right", count
    if r == h - 1:
        count = 0 if c in (0, w - 1) else int(np.count_nonzero(grid[1 : h - 1, c] == MARKER))
        return "bottom", count
    count = 0 if r in (0, h - 1) else int(np.count_nonzero(grid[r, 1 : w - 1] == MARKER))
    return "left", count


def _side_directions(side: str) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _extend_on_boundary(
    out: np.ndarray, r: int, c: int, dr: int, dc: int, steps: int, color: int
) -> tuple[int, int] | None:
    h, w = out.shape
    cur_r, cur_c = r, c
    reached = None
    for _ in range(steps):
        nr = cur_r + dr
        nc = cur_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if not (nr in (0, h - 1) or nc in (0, w - 1)):
            break
        out[nr, nc] = color
        cur_r, cur_c = nr, nc
        if _is_corner(h, w, cur_r, cur_c):
            reached = (cur_r, cur_c)
            break
    return reached


def _fill_corner(out: np.ndarray, corner: tuple[int, int], n: int, color: int) -> None:
    if n <= 0:
        return
    h, w = out.shape
    n = min(n, h - 2, w - 2)
    if n <= 0:
        return
    r, c = corner
    if (r, c) == (0, 0):
        out[1 : 1 + n, 1 : 1 + n] = color
    elif (r, c) == (0, w - 1):
        out[1 : 1 + n, w - 1 - n : w - 1] = color
    elif (r, c) == (h - 1, 0):
        out[h - 1 - n : h - 1, 1 : 1 + n] = color
    else:
        out[h - 1 - n : h - 1, w - 1 - n : w - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored border seeds along the boundary and fill corner blocks encoded by grey markers.

    Input structure:
    - The background is black.
    - Single-pixel colored seeds lie on the outer border of the grid.
    - Grey single-pixel markers lie in the interior; each seed matches either the markers in its column (top/bottom seeds) or the markers in its row (left/right seeds).

    Rule:
    - Process the seeds in clockwise border order starting from the top edge.
    - For each seed, count its aligned grey markers to obtain N.
    - Recolor up to N boundary cells in both boundary directions from that seed, stopping at corners without wrapping.
    - Whenever one of those boundary runs reaches a corner, fill the interior N x N corner rectangle with the seed color.
    - Clear grey markers to black; later seeds overwrite earlier boundary paint where they overlap.

    Key insight:
    - The grey markers encode a hidden distance: their count on the seed's own row or column tells both how far the border color propagates and whether a corner block must appear.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True

    seed_cells = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(border & (g != BG) & (g != MARKER))
    ]
    seed_cells.sort(key=lambda item: _clockwise_key(h, w, item[0], item[1]))

    out = np.where(g == MARKER, BG, g).astype(int, copy=True)
    for r, c, color in seed_cells:
        side, n = _seed_side_and_count(g, r, c)
        out[r, c] = color
        reached_corners = []
        for dr, dc in _side_directions(side):
            corner = _extend_on_boundary(out, r, c, dr, dc, n, color)
            if corner is not None:
                reached_corners.append(corner)
        for corner in reached_corners:
            _fill_corner(out, corner, n, color)

    return out


def _safe_slots(length: int) -> list[int]:
    if length >= 8:
        return list(range(2, length - 2))
    return list(range(1, length - 1))


def _candidate_slots(length: int, max_n: int) -> list[int]:
    return [p for p in _safe_slots(length) if min(p, length - 1 - p) <= max_n]


def _choose_close_pair(pool: list[int], rng: random.Random) -> list[int] | None:
    if len(pool) < 2:
        return None
    for _ in range(80):
        p1 = rng.choice(pool)
        options = [p for p in pool if p != p1 and 2 <= abs(p - p1) <= 4]
        if options:
            return sorted([p1, rng.choice(options)])
    return None


def _assign_axis_positions(
    rng: random.Random,
    length: int,
    count_a: int,
    count_b: int,
    side_a: str,
    side_b: str,
    overlap_side: str | None,
    max_n: int,
) -> dict[str, list[int]] | None:
    assigned = {side_a: [], side_b: []}
    slots = _candidate_slots(length, max_n)
    if len(slots) < count_a + count_b:
        return None

    used = set()
    if overlap_side in assigned and (count_a if overlap_side == side_a else count_b) >= 2:
        pair = _choose_close_pair(slots, rng)
        if pair is None:
            return None
        assigned[overlap_side].extend(pair)
        used.update(pair)

    remaining = [side_a] * (count_a - len(assigned[side_a])) + [side_b] * (count_b - len(assigned[side_b]))
    free_slots = [p for p in slots if p not in used]
    if len(free_slots) < len(remaining):
        return None
    rng.shuffle(free_slots)
    for side, pos in zip(remaining, free_slots):
        assigned[side].append(pos)
    return assigned


def _choose_side_counts(rng: random.Random) -> tuple[dict[str, int], str | None]:
    overlap_side = rng.choice(SIDES) if rng.random() < 0.5 else None
    total = rng.randint(3, 5) if overlap_side is not None else rng.randint(3, 4)
    counts = {side: 0 for side in SIDES}

    if overlap_side is not None:
        counts[overlap_side] = 2

    if counts["top"] + counts["bottom"] == 0:
        counts[rng.choice(["top", "bottom"])] += 1
    if counts["left"] + counts["right"] == 0:
        counts[rng.choice(["left", "right"])] += 1

    while sum(counts.values()) < total:
        side = rng.choice(SIDES)
        limit = 2 if side == overlap_side else 1
        if counts[side] >= limit:
            continue
        counts[side] += 1

    return counts, overlap_side


def _corner_dist(seed: dict[str, int], h: int, w: int) -> int:
    if seed["side"] in ("top", "bottom"):
        return min(seed["c"], w - 1 - seed["c"])
    return min(seed["r"], h - 1 - seed["r"])


def _line_capacity(seed: dict[str, int], h: int, w: int, n_h: int, n_v: int) -> int:
    if seed["side"] in ("top", "bottom"):
        return min(MAX_N, h - 2 - n_v)
    return min(MAX_N, w - 2 - n_h)


def _boundary_interval(seed: dict[str, int], h: int, w: int) -> tuple[int, int]:
    n = seed["n"]
    if seed["side"] in ("top", "bottom"):
        c = seed["c"]
        return max(0, c - n), min(w - 1, c + n)
    r = seed["r"]
    return max(0, r - n), min(h - 1, r + n)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 16)
    w = rng.randint(10, 16)

    counts, overlap_side = _choose_side_counts(rng)
    n_h = counts["top"] + counts["bottom"]
    n_v = counts["left"] + counts["right"]
    if n_h == 0 or n_v == 0:
        return None

    cap_h = min(MAX_N, h - 2 - n_v)
    cap_v = min(MAX_N, w - 2 - n_h)
    if cap_h < 2 or cap_v < 2:
        return None

    horizontal = _assign_axis_positions(
        rng, w, counts["top"], counts["bottom"], "top", "bottom", overlap_side, cap_h
    )
    vertical = _assign_axis_positions(
        rng, h, counts["left"], counts["right"], "left", "right", overlap_side, cap_v
    )
    if horizontal is None or vertical is None:
        return None

    seeds = []
    for c in horizontal["top"]:
        seeds.append({"side": "top", "r": 0, "c": c})
    for c in horizontal["bottom"]:
        seeds.append({"side": "bottom", "r": h - 1, "c": c})
    for r in vertical["left"]:
        seeds.append({"side": "left", "r": r, "c": 0})
    for r in vertical["right"]:
        seeds.append({"side": "right", "r": r, "c": w - 1})

    if len(seeds) != sum(counts.values()):
        return None

    colors = rng.sample(SEED_COLORS, len(seeds))
    rng.shuffle(colors)
    for seed, color in zip(seeds, colors):
        seed["color"] = color
        seed["corner_dist"] = _corner_dist(seed, h, w)
        seed["cap"] = _line_capacity(seed, h, w, n_h, n_v)
        seed["coord"] = seed["c"] if seed["side"] in ("top", "bottom") else seed["r"]
        if seed["corner_dist"] < 2 or seed["cap"] < seed["corner_dist"]:
            return None

    overlap_indices = []
    if overlap_side is not None:
        overlap_indices = [i for i, seed in enumerate(seeds) if seed["side"] == overlap_side]
        if len(overlap_indices) != 2:
            return None

    required_touch = set(overlap_indices)
    max_touch = len(seeds) - 1
    min_touch = max(1, len(required_touch))
    if min_touch > max_touch:
        return None
    touch_total = rng.randint(min_touch, max_touch)
    touch_set = set(required_touch)
    remaining = [i for i in range(len(seeds)) if i not in touch_set]
    rng.shuffle(remaining)
    touch_set.update(remaining[: touch_total - len(touch_set)])

    for i, seed in enumerate(seeds):
        if i in touch_set:
            continue
        max_n = seed["corner_dist"] - 1
        if max_n < 1:
            return None
        seed["n"] = rng.randint(1, min(max_n, seed["cap"]))

    for i, seed in enumerate(seeds):
        if i in touch_set and "n" not in seed:
            seed["n"] = rng.randint(seed["corner_dist"], seed["cap"])

    if overlap_indices:
        i1, i2 = overlap_indices
        s1 = seeds[i1]
        s2 = seeds[i2]
        gap = abs(s1["coord"] - s2["coord"])
        for _ in range(80):
            n1 = rng.randint(s1["corner_dist"], s1["cap"])
            need_n2 = max(s2["corner_dist"], gap - n1)
            if need_n2 <= s2["cap"]:
                s1["n"] = n1
                s2["n"] = rng.randint(need_n2, s2["cap"])
                break
        else:
            return None
        lo1, hi1 = _boundary_interval(s1, h, w)
        lo2, hi2 = _boundary_interval(s2, h, w)
        if hi1 < lo2 or hi2 < lo1:
            return None

    if not any(seed["n"] >= seed["corner_dist"] for seed in seeds):
        return None
    if not any(seed["n"] < seed["corner_dist"] for seed in seeds):
        return None

    g = np.full((h, w), BG, dtype=int)
    horizontal_cols = {seed["c"] for seed in seeds if seed["side"] in ("top", "bottom")}
    vertical_rows = {seed["r"] for seed in seeds if seed["side"] in ("left", "right")}

    for seed in seeds:
        g[seed["r"], seed["c"]] = seed["color"]

    used_markers = set()
    for seed in seeds:
        if seed["side"] in ("top", "bottom"):
            candidates = [(r, seed["c"]) for r in range(1, h - 1) if r not in vertical_rows]
        else:
            candidates = [(seed["r"], c) for c in range(1, w - 1) if c not in horizontal_cols]
        if len(candidates) < seed["n"]:
            return None
        for cell in rng.sample(candidates, seed["n"]):
            if cell in used_markers:
                return None
            used_markers.add(cell)
            g[cell] = MARKER

    out = main(g)
    if out.shape != g.shape:
        return None
    if np.count_nonzero(out == MARKER) != 0:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
