from collections import Counter, defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
RECEIVER = 5
TOWER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_towers(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    towers = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BACKGROUND, RECEIVER) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [x for x, _ in cells]
            cols = [y for _, y in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if (
                len(cells) == 9
                and r1 - r0 == 2
                and c1 - c0 == 2
                and np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == color)
            ):
                towers.append((r0, c0, color))
    towers.sort()
    return towers


def _make_tower_mask(shape, towers):
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, _ in towers:
        mask[r0 : r0 + 3, c0 : c0 + 3] = True
    return mask


def _collect_rays(shape, towers, tower_mask):
    h, w = shape
    paths = []
    hits = defaultdict(list)
    for r0, c0, color in towers:
        starts = (
            (r0, c0 + 1, -1, 0),
            (r0 + 2, c0 + 1, 1, 0),
            (r0 + 1, c0, 0, -1),
            (r0 + 1, c0 + 2, 0, 1),
        )
        for er, ec, dr, dc in starts:
            r, c = er + dr, ec + dc
            dist = 1
            while 0 <= r < h and 0 <= c < w:
                if tower_mask[r, c]:
                    break
                paths.append((r, c, color, dist))
                hits[(r, c)].append((color, dist))
                r += dr
                c += dc
                dist += 1
    return paths, hits


def _choose_receiver_color(ray_hits):
    if not ray_hits:
        return RECEIVER
    counts = Counter(color for color, _ in ray_hits)
    least = min(counts.values())
    candidates = [color for color in counts if counts[color] == least]
    if len(candidates) == 1:
        return candidates[0]
    nearest = {
        color: min(dist for hit_color, dist in ray_hits if hit_color == color)
        for color in candidates
    }
    return min(candidates, key=lambda color: nearest[color])


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast same-color rays from 3x3 towers and recolor receivers by the rarest passing ray color.

    Input structure:
    - Black background with solid 3x3 signal towers, each one a single non-gray color.
    - Gray single-pixel receivers lie away from towers and the border.
    - Towers do not touch one another and all objects stay inside the border.

    Rule:
    - Each tower emits four straight rays from the middle cell of each side.
    - A ray advances up, down, left, or right until it would leave the grid or hit any tower cell; tower cells are never overwritten.
    - Receivers do not block rays. For each receiver, count the colors of all rays that pass through it. Replace gray with the color having the fewest passing rays. If several colors tie for fewest, choose the tied color whose nearest source tower is closest along its ray.
    - Draw all rays on the output while preserving tower cells.

    Key insight:
    The visible colored lines reveal the four-way tower broadcasts, while receivers use the rarest passing color rather than a majority color; ties are resolved by source proximity.
    """
    arr = np.array(grid, copy=True)
    towers = _find_towers(arr)
    tower_mask = _make_tower_mask(arr.shape, towers)
    paths, hits = _collect_rays(arr.shape, towers, tower_mask)

    out = arr.copy()
    for r, c, color, _ in paths:
        if not tower_mask[r, c]:
            out[r, c] = color

    for r, c in zip(*np.where(arr == RECEIVER)):
        out[r, c] = _choose_receiver_color(hits.get((int(r), int(c)), []))

    out[tower_mask] = arr[tower_mask]
    return out


def _can_place(towers, r0, c0, h, w):
    if r0 < 1 or c0 < 1 or r0 + 2 >= h - 1 or c0 + 2 >= w - 1:
        return False
    for tr, tc, _ in towers:
        separate = (
            r0 + 2 < tr - 1
            or r0 > tr + 3
            or c0 + 2 < tc - 1
            or c0 > tc + 3
        )
        if not separate:
            return False
    return True


def _place_tower(grid, towers, r0, c0, color):
    grid[r0 : r0 + 3, c0 : c0 + 3] = color
    towers.append((r0, c0, color))


def _horizontal_tower_col(target_col, width, side, rng):
    if side == "left":
        lo, hi = 1, target_col - 4
    else:
        lo, hi = target_col + 2, width - 4
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _receiver_allowed(shape, tower_mask, used, r, c):
    h, w = shape
    if r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1:
        return False
    if tower_mask[r, c]:
        return False
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if tower_mask[nr, nc]:
                return False
    return all(abs(r - ur) > 1 or abs(c - uc) > 1 for ur, uc in used)


def _cell_branch(ray_hits):
    counts = Counter(color for color, _ in ray_hits)
    if not counts:
        return None
    if len(counts) == 1:
        return "single"
    least = min(counts.values())
    tied = [color for color in counts if counts[color] == least]
    if len(tied) == 1:
        return "minority"
    dists = [
        min(dist for hit_color, dist in ray_hits if hit_color == color)
        for color in tied
    ]
    return "tie" if dists.count(min(dists)) == 1 else None


def _extra_tower_ok(r0, c0, fixed_rows, fixed_col, fixed_receivers):
    if c0 <= fixed_col <= c0 + 2:
        return False
    if any(r0 <= row <= r0 + 2 for row in fixed_rows):
        return False
    for rr, cc in fixed_receivers:
        if r0 - 1 <= rr <= r0 + 3 and c0 - 1 <= cc <= c0 + 3:
            return False
    return True


def _build_candidate_pools(shape, towers):
    tower_mask = _make_tower_mask(shape, towers)
    _, hits = _collect_rays(shape, towers, tower_mask)
    pools = {"single": [], "minority": [], "tie": []}
    for (r, c), ray_hits in hits.items():
        branch = _cell_branch(ray_hits)
        if branch in pools and _receiver_allowed(shape, tower_mask, [], r, c):
            pools[branch].append((r, c))
    return pools, hits, tower_mask


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(250):
        h = rng.randint(20, 30)
        w = rng.randint(18, 30)
        grid = np.zeros((h, w), dtype=int)
        towers = []
        color_a, color_b, color_c = rng.sample(TOWER_COLORS, 3)

        mid_col = rng.randint(6, w - 7)
        top_r = rng.randint(5, h - 12)
        tie_r = rng.randint(2, top_r - 3)
        minor_r = rng.randint(top_r + 5, h - 7)
        bottom_r = rng.randint(minor_r + 3, h - 4)

        fixed = [
            (top_r, mid_col - 1, color_a),
            (bottom_r, mid_col - 1, color_a),
        ]
        minor_side = rng.choice(("left", "right"))
        tie_side = rng.choice(("left", "right"))
        minor_c = _horizontal_tower_col(mid_col, w, minor_side, rng)
        tie_c = _horizontal_tower_col(mid_col, w, tie_side, rng)
        if minor_c is None or tie_c is None:
            continue
        fixed.extend(
            [
                (minor_r - 1, minor_c, color_b),
                (tie_r - 1, tie_c, color_c),
            ]
        )
        if any(not _can_place(towers, r, c, h, w) for r, c, _ in fixed):
            continue
        for r, c, color in fixed:
            _place_tower(grid, towers, r, c, color)

        fixed_receivers = [(minor_r, mid_col), (tie_r, mid_col)]
        fixed_rows = {minor_r, tie_r}
        for _extra in range(rng.randint(0, 3)):
            for _try in range(80):
                r0 = rng.randint(1, h - 4)
                c0 = rng.randint(1, w - 4)
                if not _can_place(towers, r0, c0, h, w):
                    continue
                if not _extra_tower_ok(r0, c0, fixed_rows, mid_col, fixed_receivers):
                    continue
                _place_tower(grid, towers, r0, c0, rng.choice(TOWER_COLORS))
                break

        pools, hits, tower_mask = _build_candidate_pools(grid.shape, towers)
        if _cell_branch(hits.get((minor_r, mid_col), [])) != "minority":
            continue
        if _cell_branch(hits.get((tie_r, mid_col), [])) != "tie":
            continue
        if not all(
            _receiver_allowed(grid.shape, tower_mask, [], r, c)
            for r, c in fixed_receivers
        ):
            continue

        used = []
        for r, c in fixed_receivers:
            grid[r, c] = RECEIVER
            used.append((r, c))

        extra_count = rng.randint(1, 5)
        branch_order = ["single", "minority", "tie"]
        rng.shuffle(branch_order)
        while len(branch_order) < extra_count:
            branch_order.append(rng.choice(("single", "minority", "tie")))
        for branch in branch_order[:extra_count]:
            choices = pools[branch][:]
            rng.shuffle(choices)
            for r, c in choices:
                if grid[r, c] == BACKGROUND and _receiver_allowed(
                    grid.shape, tower_mask, used, r, c
                ):
                    grid[r, c] = RECEIVER
                    used.append((r, c))
                    break

        return grid

    grid = np.zeros((24, 24), dtype=int)
    towers = []
    for r, c, color in (
        (6, 11, 2),
        (18, 11, 2),
        (13, 3, 7),
        (2, 17, 4),
    ):
        _place_tower(grid, towers, r, c, color)
    grid[14, 12] = RECEIVER
    grid[3, 12] = RECEIVER
    grid[14, 20] = RECEIVER
    return grid
