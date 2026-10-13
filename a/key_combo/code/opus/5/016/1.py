from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MOVE_DIRS = ((0, 1), (1, 0))  # generator uses a monotone east/south winding spine
SQUARE_SIZES = (1, 2, 3)
CHANNEL_WIDTHS = (1, 2, 3)
CHANNEL_COLORS = (5, 8)
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 9)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _largest_square_map(mask: np.ndarray, max_size: int = 5) -> np.ndarray:
    h, w = mask.shape
    sat = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    sat = sat.cumsum(axis=0).cumsum(axis=1)

    def square_sum(r: int, c: int, k: int) -> int:
        r1 = r + k
        c1 = c + k
        return int(sat[r1, c1] - sat[r, c1] - sat[r1, c] + sat[r, c])

    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            lim = min(max_size, h - r, w - c)
            best = 0
            for k in range(1, lim + 1):
                if square_sum(r, c, k) == k * k:
                    best = k
                else:
                    break
            out[r, c] = best
    return out


def _paint_square(arr: np.ndarray, r: int, c: int, s: int, value: int) -> None:
    arr[r : r + s, c : c + s] = int(value)


def _square_new_count(covered: np.ndarray, r: int, c: int, s: int) -> int:
    return int(s * s - covered[r : r + s, c : c + s].sum())


def _recover_anchor_path(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    cap_map = _largest_square_map(mask, max_size=5)
    starts = np.argwhere(cap_map > 0)
    if starts.size == 0:
        raise ValueError("empty channel")
    start = tuple(map(int, min(map(tuple, starts.tolist()))))
    path = [start]
    caps = [int(cap_map[start])]
    covered = np.zeros_like(mask, dtype=bool)
    _paint_square(covered, start[0], start[1], caps[0], True)
    visited = {start}

    while True:
        cand: list[tuple[int, int, int, int, int]] = []
        r, c = path[-1]
        for dr, dc in MOVE_DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(mask.shape, nr, nc) or (nr, nc) in visited:
                continue
            s = int(cap_map[nr, nc])
            if s <= 0:
                continue
            new_count = _square_new_count(covered, nr, nc, s)
            if new_count <= 0:
                continue
            cand.append((new_count, s, -nr, -nc, 0 if dc == 1 else 1))
        if not cand:
            break
        cand.sort(reverse=True)
        _new_count, s, _neg_r, _neg_c, _dir_rank = cand[0]
        nr = -cand[0][2]
        nc = -cand[0][3]
        visited.add((nr, nc))
        path.append((nr, nc))
        caps.append(int(s))
        _paint_square(covered, nr, nc, int(s), True)

    if not np.array_equal(covered, mask):
        raise ValueError("failed to recover channel spine")
    return path, caps


def _detect_objects(grid: np.ndarray, channel_color: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {BG, int(channel_color)}):
        mask = grid == int(color)
        for coords in _find_components(mask):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h != w:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(coords.shape[0]) != h * w:
                continue
            if not bool((patch == int(color)).all()):
                continue
            objects.append({"color": int(color), "r": int(r0), "c": int(c0), "size": int(h)})
    if not objects:
        raise ValueError("no blocks found")
    return objects


def _feasible_map(channel_mask: np.ndarray, size: int) -> np.ndarray:
    h, w = channel_mask.shape
    out = np.zeros((h, w), dtype=bool)
    lim_r = h - size + 1
    lim_c = w - size + 1
    for r in range(max(0, lim_r)):
        for c in range(max(0, lim_c)):
            out[r, c] = bool(channel_mask[r : r + size, c : c + size].all())
    return out


def _downstream_score(feasible: np.ndarray) -> np.ndarray:
    h, w = feasible.shape
    score = np.full((h, w), -1, dtype=int)
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if not feasible[r, c]:
                continue
            best = 1
            if c + 1 < w and score[r, c + 1] >= 0:
                best = max(best, 1 + int(score[r, c + 1]))
            if r + 1 < h and score[r + 1, c] >= 0:
                best = max(best, 1 + int(score[r + 1, c]))
            score[r, c] = best
    return score


def _choose_next(r: int, c: int, feasible: np.ndarray, score: np.ndarray) -> tuple[int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for dr, dc in MOVE_DIRS:
        nr, nc = r + dr, c + dc
        if not _in_bounds(feasible.shape, nr, nc) or not feasible[nr, nc]:
            continue
        candidates.append((int(score[nr, nc]), nr + nc, -nc, -nr))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0]
    nr = -best[3]
    nc = -best[2]
    return nr, nc


def _terminal_reason(r: int, c: int, size: int, channel_mask: np.ndarray, feasible: np.ndarray) -> str:
    if _choose_next(r, c, feasible, _downstream_score(feasible)) is not None:
        return "budget"
    h, w = channel_mask.shape
    rr0 = min(h, r + size)
    cc0 = min(w, c + size)
    if rr0 >= h and cc0 >= w:
        return "end"
    downstream = channel_mask[r:, c:]
    if downstream.size > 0 and bool(downstream.any()):
        return "width"
    return "end"


def _simulate(
    grid_shape: tuple[int, int],
    channel_mask: np.ndarray,
    channel_color: int,
    objects: list[dict],
) -> tuple[np.ndarray, list[dict], dict[int, np.ndarray], dict[int, np.ndarray]]:
    feasible_by_size = {s: _feasible_map(channel_mask, s) for s in SQUARE_SIZES}
    score_by_size = {s: _downstream_score(feasible_by_size[s]) for s in SQUARE_SIZES}

    enriched = []
    for obj in objects:
        s = int(obj["size"])
        if not bool(feasible_by_size[s][int(obj["r"]), int(obj["c"])]):
            raise ValueError("object is not on a valid channel placement")
        enriched.append({**obj, "order_key": (int(obj["r"]) + int(obj["c"]), int(obj["r"]), int(obj["c"]))})
    enriched.sort(key=lambda obj: obj["order_key"])

    occupied = np.zeros(grid_shape, dtype=bool)
    final_objects: list[dict] = []
    for obj in enriched:
        s = int(obj["size"])
        budget = int(s * s)
        r = int(obj["r"])
        c = int(obj["c"])
        feasible = feasible_by_size[s]
        score = score_by_size[s]
        stop_reason = "budget"
        while budget > 0:
            nxt = _choose_next(r, c, feasible, score)
            if nxt is None:
                stop_reason = _terminal_reason(r, c, s, channel_mask, feasible)
                break
            nr, nc = nxt
            if bool(occupied[nr : nr + s, nc : nc + s].any()):
                stop_reason = "collision"
                break
            r, c = nr, nc
            budget -= 1
        occupied[r : r + s, c : c + s] = True
        final_objects.append(
            {
                **obj,
                "start_key": (int(obj["r"]) + int(obj["c"]), int(obj["r"]), int(obj["c"])),
                "final_key": (int(r) + int(c), int(r), int(c)),
                "final_r": int(r),
                "final_c": int(c),
                "stop_reason": stop_reason if budget > 0 else "budget",
            }
        )

    out = np.zeros(grid_shape, dtype=int)
    out[channel_mask] = int(channel_color)
    for obj in final_objects:
        _paint_square(out, obj["final_r"], obj["final_c"], obj["size"], obj["color"])
    return out, final_objects, feasible_by_size, score_by_size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multiple solid squares forward through a winding channel, with area-limited travel and bottleneck blocking.

    Input structure:
    - Black (0) is background.
    - One dominant non-black color forms a single non-branching channel with two ends.
    - Three to five distinct-colored solid squares lie fully on that channel and do not touch each other.

    Rule:
    - Infer the channel direction from the end nearest the upper-left toward the opposite end.
    - Process the squares from upstream to downstream along that channel.
    - A square of side s may advance one channel step at a time for at most s^2 steps.
    - It stops immediately when the next channel position is too narrow for side s, when an already-settled earlier square blocks that next position, when it reaches the channel end, or when it has used all s^2 steps.
    - After all squares are processed, keep each square at its stopping place and restore every vacated source cell to the channel color.

    Key insight:
    - The same square area controls both motion budget and passability: larger blocks can travel farther in principle, but they are also the first to fail at narrow bottlenecks, and earlier settled blocks become new obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    non_bg = [int(v) for v in g.flatten() if int(v) != BG]
    if not non_bg:
        return g.copy()
    channel_color = int(Counter(non_bg).most_common(1)[0][0])
    objects = _detect_objects(g, channel_color)
    channel_mask = g != BG
    out, _finals, _feasible, _score = _simulate(g.shape, channel_mask, channel_color, objects)
    return out


def _build_monotone_path() -> list[tuple[int, int]]:
    while True:
        seg_count = random.randint(4, 6)
        dirs: list[tuple[int, int]] = []
        first_is_east = random.random() < 0.5
        for i in range(seg_count):
            use_east = first_is_east if i % 2 == 0 else not first_is_east
            dirs.append((0, 1) if use_east else (1, 0))

        lengths = [random.randint(2, 5) for _ in range(seg_count)]
        total_south = sum(length for length, d in zip(lengths, dirs) if d == (1, 0))
        total_east = sum(length for length, d in zip(lengths, dirs) if d == (0, 1))
        start_r = random.randint(1, 3)
        start_c = random.randint(1, 3)
        end_r = start_r + total_south
        end_c = start_c + total_east
        h = end_r + 5
        w = end_c + 5
        if h > 30 or w > 30:
            continue

        path = [(start_r, start_c)]
        r, c = start_r, start_c
        for (dr, dc), length in zip(dirs, lengths):
            for _ in range(length):
                r += dr
                c += dc
                path.append((r, c))
        return path


def _turn_indices(path: list[tuple[int, int]]) -> set[int]:
    turns: set[int] = set()
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns.add(i)
    return turns


def _segment_ranges(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 1:
        return [(0, 0)]
    out: list[tuple[int, int]] = []
    seg_start = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(1, len(path) - 1):
        cur_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if cur_dir != prev_dir:
            out.append((seg_start, i))
            seg_start = i
            prev_dir = cur_dir
    out.append((seg_start, len(path) - 1))
    return out


def _build_caps(path: list[tuple[int, int]]) -> list[int]:
    segs = _segment_ranges(path)
    caps = [0] * len(path)
    if len(segs) < 3:
        return [1] * len(path)

    internal_widths = [random.choice((2, 3)) for _ in segs[1:-1]]
    if 2 not in internal_widths:
        internal_widths[random.randrange(len(internal_widths))] = 2
    if 3 not in internal_widths:
        internal_widths[random.randrange(len(internal_widths))] = 3

    seg_widths = [1] + internal_widths + [1]
    for seg_i, ((a, b), width) in enumerate(zip(segs, seg_widths)):
        start = a if seg_i == 0 else a + 1
        for i in range(start, b + 1):
            caps[i] = width
    if caps[0] == 0:
        caps[0] = seg_widths[0]
    return caps


def _render_channel(path: list[tuple[int, int]], caps: list[int]) -> np.ndarray:
    h = max(r + s for (r, _c), s in zip(path, caps)) + 1
    w = max(c + s for (_r, c), s in zip(path, caps)) + 1
    mask = np.zeros((h, w), dtype=bool)
    for (r, c), s in zip(path, caps):
        mask[r : r + s, c : c + s] = True
    return mask


def _eligible_object_sites(path: list[tuple[int, int]], caps: list[int], size: int) -> list[int]:
    turns = _turn_indices(path)
    out: list[int] = []
    for i, ((r, c), cap) in enumerate(zip(path, caps)):
        if cap < size:
            continue
        if i < 2 or i > len(path) - 3:
            continue
        if i in turns or i - 1 in turns or i + 1 in turns:
            continue
        out.append(i)
    return out


def _squares_are_separate(placed: list[dict], cand_idx: int, path: list[tuple[int, int]], size: int) -> bool:
    cr, cc = path[cand_idx]
    r0 = cr - 1
    c0 = cc - 1
    r1 = cr + size
    c1 = cc + size
    for obj in placed:
        or0 = obj["r"] - 1
        oc0 = obj["c"] - 1
        or1 = obj["r"] + obj["size"]
        oc1 = obj["c"] + obj["size"]
        if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
            return False
    return True


def _plan_objects(path: list[tuple[int, int]], caps: list[int]) -> list[dict] | None:
    sizes: list[int] = [2, 3]
    if random.random() < 0.75:
        sizes.append(1)
    count = random.randint(3, 5)
    while len(sizes) < count:
        sizes.append(random.choice(SQUARE_SIZES))
    random.shuffle(sizes)

    colors = random.sample(OBJECT_COLORS, count)
    placed: list[dict] = []
    for size, color in zip(sizes, colors):
        candidates = _eligible_object_sites(path, caps, size)
        random.shuffle(candidates)
        chosen = None
        for idx in candidates:
            if _squares_are_separate(placed, idx, path, size):
                chosen = idx
                break
        if chosen is None:
            return None
        r, c = path[chosen]
        placed.append({"color": int(color), "size": int(size), "r": int(r), "c": int(c)})
    return placed


def _build_grid(channel_mask: np.ndarray, channel_color: int, objects: list[dict]) -> np.ndarray:
    grid = np.zeros(channel_mask.shape, dtype=int)
    grid[channel_mask] = int(channel_color)
    for obj in objects:
        _paint_square(grid, obj["r"], obj["c"], obj["size"], obj["color"])
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray, finals: list[dict], path: list[tuple[int, int]], caps: list[int]) -> bool:
    if np.array_equal(inp, out):
        return False
    if len({obj["size"] for obj in finals}) < 2:
        return False
    if len({obj["stop_reason"] for obj in finals}) < 2:
        return False
    moved = sum((obj["start_key"] != obj["final_key"]) for obj in finals)
    if moved < 2:
        return False
    if max(obj["size"] for obj in finals) < 3:
        return False
    if 2 not in caps:
        return False
    if len(path) < 12:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    for _ in range(200):
        path = _build_monotone_path()
        caps = _build_caps(path)
        channel_mask = _render_channel(path, caps)
        if max(channel_mask.shape) > 30:
            continue

        channel_color = random.choice(CHANNEL_COLORS)
        objects = _plan_objects(path, caps)
        if objects is None:
            continue

        inp = _build_grid(channel_mask, channel_color, objects)
        try:
            out, finals, _feasible, _score = _simulate(inp.shape, channel_mask, channel_color, objects)
        except Exception:
            continue
        try:
            solver_out = main(inp)
        except Exception:
            continue
        if not np.array_equal(out, solver_out):
            continue
        if not _semantic_checks(inp, out, finals, path, caps):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
