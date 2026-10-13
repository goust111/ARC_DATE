from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


SIDE_ORDER = ("top", "bottom", "left", "right")
OBSTACLE_COLOR = 9
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class Seed:
    side: str
    color: int
    start: int
    end: int
    index: int


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rectangles_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        if np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
            rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _find_obstacle_color(grid: np.ndarray, bg: int) -> int | None:
    h, w = grid.shape
    edge_colors = set(map(int, np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])))
    counts = Counter(map(int, grid.reshape(-1)))
    candidates = [color for color in counts if color != bg and color not in edge_colors]
    if not candidates:
        return None

    best_color = None
    best_score = -1
    for color in candidates:
        mask = grid == color
        rects = _rectangles_from_mask(mask)
        if not rects:
            continue
        if sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects) != int(mask.sum()):
            continue
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_color = int(color)
    return best_color


def _scan_line_for_seeds(
    values: list[int], side: str, offset: int, banned: set[int], start_index: int
) -> tuple[list[Seed], int]:
    seeds: list[Seed] = []
    idx = start_index
    i = 0
    while i < len(values):
        color = int(values[i])
        if color in banned:
            i += 1
            continue
        j = i
        while j + 1 < len(values) and int(values[j + 1]) == color:
            j += 1
        seeds.append(Seed(side=side, color=color, start=offset + i, end=offset + j, index=idx))
        idx += 1
        i = j + 1
    return seeds, idx


def _find_seeds(grid: np.ndarray, bg: int, obstacle_color: int | None) -> list[Seed]:
    h, w = grid.shape
    banned = {bg}
    if obstacle_color is not None:
        banned.add(obstacle_color)

    seeds: list[Seed] = []
    next_index = 0

    top, next_index = _scan_line_for_seeds(grid[0, :].tolist(), "top", 0, banned, next_index)
    seeds.extend(top)
    bottom, next_index = _scan_line_for_seeds(grid[h - 1, :].tolist(), "bottom", 0, banned, next_index)
    seeds.extend(bottom)
    if h > 2:
        left, next_index = _scan_line_for_seeds(grid[1 : h - 1, 0].tolist(), "left", 1, banned, next_index)
        seeds.extend(left)
        right, next_index = _scan_line_for_seeds(
            grid[1 : h - 1, w - 1].tolist(), "right", 1, banned, next_index
        )
        seeds.extend(right)
    return seeds


def _rotate_coord(r: int, c: int, h: int, w: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _inverse_rotate_coord(r: int, c: int, h: int, w: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return c, w - 1 - r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return h - 1 - c, r


def _seed_cells(seed: Seed, h: int, w: int) -> list[tuple[int, int]]:
    if seed.side == "top":
        return [(0, c) for c in range(seed.start, seed.end + 1)]
    if seed.side == "bottom":
        return [(h - 1, c) for c in range(seed.start, seed.end + 1)]
    if seed.side == "left":
        return [(r, 0) for r in range(seed.start, seed.end + 1)]
    return [(r, w - 1) for r in range(seed.start, seed.end + 1)]


def _grow_down(
    obstacle_mask: np.ndarray, obstacle_rects: list[tuple[int, int, int, int]], start_c0: int, start_c1: int
) -> tuple[dict[tuple[int, int], int], bool]:
    h, w = obstacle_mask.shape
    painted: dict[tuple[int, int], int] = {(0, c): 0 for c in range(start_c0, start_c1 + 1)}
    queue = deque([(0, start_c0, start_c1)])
    seen_states = {(0, start_c0, start_c1)}
    branched = False

    while queue:
        row, c0, c1 = queue.popleft()
        if row >= h - 1:
            continue

        next_row = row + 1
        blockers = [
            rect
            for rect in obstacle_rects
            if rect[0] <= next_row <= rect[2] and not (rect[3] < c0 or rect[1] > c1)
        ]

        if not blockers:
            for c in range(c0, c1 + 1):
                if not obstacle_mask[next_row, c]:
                    painted.setdefault((next_row, c), next_row)
            state = (next_row, c0, c1)
            if state not in seen_states:
                seen_states.add(state)
                queue.append(state)
            continue

        branched = True
        left_target = min(rect[1] for rect in blockers) - 1
        right_target = max(rect[3] for rect in blockers) + 1

        for target, anchor in ((left_target, c0), (right_target, c1)):
            if not (0 <= target < w):
                continue
            lo, hi = sorted((target, anchor))
            for c in range(lo, hi + 1):
                if not obstacle_mask[row, c]:
                    painted.setdefault((row, c), row)
            state = (row, target, target)
            if state not in seen_states and not obstacle_mask[row, target]:
                seen_states.add(state)
                queue.append(state)

    return painted, branched


def _simulate_seed(grid: np.ndarray, obstacle_color: int | None, seed: Seed) -> tuple[dict[tuple[int, int], int], bool]:
    h, w = grid.shape
    rotation = {"top": 0, "right": 1, "bottom": 2, "left": 3}[seed.side]
    rotated = np.rot90(grid, rotation)
    if obstacle_color is None:
        obstacle_mask = np.zeros_like(rotated, dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
    else:
        obstacle_mask = rotated == obstacle_color
        rects = _rectangles_from_mask(obstacle_mask)

    rotated_cells = [_rotate_coord(r, c, h, w, rotation) for r, c in _seed_cells(seed, h, w)]
    cols = sorted(c for r, c in rotated_cells if r == 0)
    if not cols:
        return {}, False

    painted_rot, branched = _grow_down(obstacle_mask, rects, cols[0], cols[-1])
    painted_orig: dict[tuple[int, int], int] = {}
    for (rr, cc), depth in painted_rot.items():
        r, c = _inverse_rotate_coord(rr, cc, h, w, rotation)
        painted_orig[(r, c)] = min(depth, painted_orig.get((r, c), depth))
    return painted_orig, branched


def _solve_grid(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    obstacle_color = _find_obstacle_color(g, bg)
    seeds = _find_seeds(g, bg, obstacle_color)

    claims: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    branch_count = 0
    for seed in seeds:
        painted, branched = _simulate_seed(g, obstacle_color, seed)
        branch_count += int(branched)
        for cell, depth in painted.items():
            claims.setdefault(cell, []).append((depth, seed.index, seed.color))

    out = g.copy()
    has_overlap = False
    for cell, options in claims.items():
        if len(options) > 1:
            has_overlap = True
        _, _, color = min(options, key=lambda item: (item[0], item[1]))
        if obstacle_color is not None and int(g[cell]) == obstacle_color:
            continue
        out[cell] = color

    if not return_meta:
        return out

    obstacle_count = 0 if obstacle_color is None else len(_rectangles_from_mask(g == obstacle_color))
    meta = {
        "bg": bg,
        "obstacle_color": obstacle_color,
        "seed_count": len(seeds),
        "obstacle_count": obstacle_count,
        "branch_count": branch_count,
        "has_overlap": has_overlap,
        "new_pixels": int(np.count_nonzero(out != g)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid)


def _interval_fits(existing: list[tuple[int, int]], start: int, end: int, pad: int = 1) -> bool:
    for s0, s1 in existing:
        if not (end + pad < s0 or start - pad > s1):
            return False
    return True


def _add_seed(
    seeds: list[Seed],
    used: dict[str, list[tuple[int, int]]],
    side: str,
    start: int,
    end: int,
    color: int,
    shared_intervals: list[tuple[int, int]] | None = None,
    pad: int = 1,
) -> bool:
    if not _interval_fits(used[side], start, end, pad=pad):
        return False
    if shared_intervals is not None and not _interval_fits(shared_intervals, start, end, pad=pad):
        return False
    seeds.append(Seed(side=side, color=color, start=start, end=end, index=len(seeds)))
    used[side].append((start, end))
    if shared_intervals is not None:
        shared_intervals.append((start, end))
    return True


def _max_seed_len(side: str, h: int, w: int) -> int:
    span = w - 2 if side in ("top", "bottom") else h - 2
    return max(1, min(4, span))


def _random_seed_interval(rng: random.Random, side: str, h: int, w: int) -> tuple[int, int] | None:
    max_len = _max_seed_len(side, h, w)
    if max_len <= 0:
        return None
    length = rng.randint(1, max_len)
    span = w if side in ("top", "bottom") else h
    start_min = 1
    start_max = span - 1 - length
    if start_min > start_max:
        return None
    start = rng.randint(start_min, start_max)
    return start, start + length - 1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _place_blocking_obstacle(
    rng: random.Random,
    seed: Seed,
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        if seed.side in ("top", "bottom"):
            length = seed.end - seed.start + 1
            oh = rng.randint(2, min(4, h - 4))
            ow = rng.randint(length, min(length + 2, w - 2))
            left_lo = max(1, seed.end - ow + 1)
            left_hi = min(seed.start, w - 1 - ow)
            if left_lo > left_hi:
                continue
            c0 = rng.randint(left_lo, left_hi)
            r0_lo, r0_hi = 2, h - oh - 3
            if r0_lo > r0_hi:
                continue
            r0 = rng.randint(r0_lo, r0_hi)
            rect = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        else:
            length = seed.end - seed.start + 1
            ow = rng.randint(2, min(4, w - 4))
            oh = rng.randint(length, min(length + 2, h - 2))
            top_lo = max(1, seed.end - oh + 1)
            top_hi = min(seed.start, h - 1 - oh)
            if top_lo > top_hi:
                continue
            r0 = rng.randint(top_lo, top_hi)
            c0_lo, c0_hi = 2, w - ow - 3
            if c0_lo > c0_hi:
                continue
            c0 = rng.randint(c0_lo, c0_hi)
            rect = (r0, c0, r0 + oh - 1, c0 + ow - 1)

        if any(_rects_overlap(rect, other, pad=1) for other in rects):
            continue
        return rect
    return None


def _direct_path_mask(seeds: list[Seed], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for seed in seeds:
        if seed.side in ("top", "bottom"):
            mask[:, seed.start : seed.end + 1] = True
        else:
            mask[seed.start : seed.end + 1, :] = True
    return mask


def _place_offpath_obstacle(
    rng: random.Random,
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    path_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    for _ in range(120):
        oh = rng.randint(2, min(4, h - 4))
        ow = rng.randint(2, min(4, w - 4))
        r0_lo, r0_hi = 2, h - oh - 3
        c0_lo, c0_hi = 2, w - ow - 3
        if r0_lo > r0_hi or c0_lo > c0_hi:
            return None
        r0 = rng.randint(r0_lo, r0_hi)
        c0 = rng.randint(c0_lo, c0_hi)
        rect = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        if any(_rects_overlap(rect, other, pad=1) for other in rects):
            continue
        if path_mask[r0 : r0 + oh, c0 : c0 + ow].any():
            continue
        return rect
    return None


def _paint_seed(grid: np.ndarray, seed: Seed) -> None:
    if seed.side == "top":
        grid[0, seed.start : seed.end + 1] = seed.color
    elif seed.side == "bottom":
        grid[-1, seed.start : seed.end + 1] = seed.color
    elif seed.side == "left":
        grid[seed.start : seed.end + 1, 0] = seed.color
    else:
        grid[seed.start : seed.end + 1, -1] = seed.color


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _build_seed_plan(
    rng: random.Random, h: int, w: int, bg: int, want_overlap: bool, want_branch: bool
) -> list[Seed] | None:
    target_seeds = 2 + rng.randint(0, 2)
    if want_branch:
        target_seeds = max(target_seeds, 3)

    palette = [c for c in range(10) if c not in (bg, OBSTACLE_COLOR)]
    if len(palette) < target_seeds:
        return None
    colors = rng.sample(palette, target_seeds)

    seeds: list[Seed] = []
    used = {side: [] for side in SIDE_ORDER}
    shared_axis_intervals: list[tuple[int, int]] | None = None
    allowed_sides = list(SIDE_ORDER)
    seed_pad = 1

    if not want_overlap:
        seed_pad = 2
        shared_axis_intervals = []
        allowed_sides = [rng.choice(list(SIDE_ORDER))]

    color_idx = 0

    if want_overlap:
        axis = rng.choice(["vertical", "horizontal"])
        if axis == "vertical":
            max_len = min(3, _max_seed_len("top", h, w))
            if max_len < 1:
                return None
            length = rng.randint(1, max_len)
            start = rng.randint(1, w - 1 - length)
            if not _add_seed(seeds, used, "top", start, start + length - 1, colors[color_idx], pad=seed_pad):
                return None
            color_idx += 1
            if not _add_seed(seeds, used, "bottom", start, start + length - 1, colors[color_idx], pad=seed_pad):
                return None
            color_idx += 1
        else:
            max_len = min(3, _max_seed_len("left", h, w))
            if max_len < 1:
                return None
            length = rng.randint(1, max_len)
            start = rng.randint(1, h - 1 - length)
            if not _add_seed(seeds, used, "left", start, start + length - 1, colors[color_idx], pad=seed_pad):
                return None
            color_idx += 1
            if not _add_seed(seeds, used, "right", start, start + length - 1, colors[color_idx], pad=seed_pad):
                return None
            color_idx += 1

    for _ in range(120):
        if len(seeds) >= target_seeds:
            break
        side = rng.choice(allowed_sides)
        interval = _random_seed_interval(rng, side, h, w)
        if interval is None:
            continue
        start, end = interval
        if _add_seed(
            seeds,
            used,
            side,
            start,
            end,
            colors[color_idx],
            shared_intervals=shared_axis_intervals,
            pad=seed_pad,
        ):
            color_idx += 1
    if len(seeds) != target_seeds:
        return None
    return [Seed(seed.side, seed.color, seed.start, seed.end, i) for i, seed in enumerate(seeds)]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 19)
    w = rng.randint(12, 19)
    bg = rng.choice([c for c in range(9) if c != OBSTACLE_COLOR])
    want_branch = rng.random() < 0.5
    want_overlap = rng.random() < 0.4

    seeds = _build_seed_plan(rng, h, w, bg, want_overlap, want_branch)
    if seeds is None:
        return None

    rects: list[tuple[int, int, int, int]] = []
    if want_branch:
        candidates = seeds[2:] if want_overlap and len(seeds) > 2 else seeds[:]
        rng.shuffle(candidates)
        blocked_goal = 1 if len(candidates) == 1 else rng.choice([1, min(2, len(candidates))])
        for seed in candidates[:blocked_goal]:
            rect = _place_blocking_obstacle(rng, seed, h, w, rects)
            if rect is None:
                return None
            rects.append(rect)
    else:
        path_mask = _direct_path_mask(seeds, h, w)
        obstacle_goal = rng.randint(1, 2)
        for _ in range(obstacle_goal):
            rect = _place_offpath_obstacle(rng, h, w, rects, path_mask)
            if rect is None:
                return None
            rects.append(rect)

    if want_branch and rng.random() < 0.35:
        path_mask = _direct_path_mask(seeds, h, w)
        rect = _place_offpath_obstacle(rng, h, w, rects, path_mask)
        if rect is not None:
            rects.append(rect)

    grid = np.full((h, w), bg, dtype=int)
    for seed in seeds:
        _paint_seed(grid, seed)
    for rect in rects:
        _paint_rect(grid, rect, OBSTACLE_COLOR)

    out, meta = _solve_grid(grid, return_meta=True)
    if meta["seed_count"] != len(seeds) or meta["obstacle_count"] != len(rects):
        return None
    if meta["new_pixels"] < max(h, w) // 2:
        return None
    if meta["new_pixels"] > (h * w) // 2:
        return None
    if bool(meta["has_overlap"]) != want_overlap:
        return None
    if (meta["branch_count"] > 0) != want_branch:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid vine scene")
