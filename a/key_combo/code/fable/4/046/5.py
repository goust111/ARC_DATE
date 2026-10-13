from __future__ import annotations

import itertools
import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _corners(rect: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = rect
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _extract_points(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = set(map(int, g.ravel()))
    assert colors <= {bg, YELLOW}, "input must contain only background and yellow corner markers"
    points = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    assert points, "expected at least one yellow corner marker"
    return sorted(points)


def _candidate_rectangles(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, ...]], list[list[int]]]:
    point_to_idx = {pt: idx for idx, pt in enumerate(points)}
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})

    rects: list[tuple[int, int, int, int]] = []
    rect_points: list[tuple[int, ...]] = []
    point_rects = [[] for _ in points]

    for r0_i, r0 in enumerate(rows):
        for r1 in rows[r0_i + 1 :]:
            for c0_i, c0 in enumerate(cols):
                for c1 in cols[c0_i + 1 :]:
                    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                    if any(pt not in point_to_idx for pt in corners):
                        continue
                    rect = (r0, c0, r1, c1)
                    idxs = tuple(point_to_idx[pt] for pt in corners)
                    rid = len(rects)
                    rects.append(rect)
                    rect_points.append(idxs)
                    for pid in idxs:
                        point_rects[pid].append(rid)

    return rects, rect_points, point_rects


def _search_unique_rectangles(points: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rects, rect_points, point_rects = _candidate_rectangles(points)
    assert rects, "no rectangles can be formed from the markers"
    assert all(point_rects), "every marker must belong to at least one candidate rectangle"

    solution_masks: list[tuple[int, ...]] = []
    n_points = len(points)
    n_rects = len(rects)

    def dfs(state: np.ndarray, counts: np.ndarray) -> None:
        if len(solution_masks) >= 2:
            return

        best_point = None
        best_undecided: list[int] | None = None
        best_branch_count = None

        for pid in range(n_points):
            count = int(counts[pid])
            if count > 2:
                return
            undecided = [rid for rid in point_rects[pid] if state[rid] == 0]
            if count + len(undecided) < 1:
                return

            feasible_sizes = []
            for target in (1, 2):
                need = target - count
                if 0 <= need <= len(undecided):
                    feasible_sizes.append(need)
            if not feasible_sizes:
                return

            if undecided:
                branch_count = sum(math.comb(len(undecided), need) for need in feasible_sizes)
                if best_branch_count is None or branch_count < best_branch_count:
                    best_point = pid
                    best_undecided = undecided
                    best_branch_count = branch_count

        if best_point is None:
            if np.all((1 <= counts) & (counts <= 2)):
                picked = tuple(sorted(int(rid) for rid in np.flatnonzero(state == 1)))
                solution_masks.append(picked)
            return

        count = int(counts[best_point])
        undecided = best_undecided if best_undecided is not None else []
        option_sizes = []
        for target in (1, 2):
            need = target - count
            if 0 <= need <= len(undecided):
                option_sizes.append(need)

        for need in option_sizes:
            for chosen in itertools.combinations(undecided, need):
                chosen_set = set(chosen)
                next_state = state.copy()
                next_counts = counts.copy()

                for rid in undecided:
                    if rid in chosen_set:
                        next_state[rid] = 1
                        for pid in rect_points[rid]:
                            next_counts[pid] += 1
                    else:
                        next_state[rid] = -1

                dfs(next_state, next_counts)
                if len(solution_masks) >= 2:
                    return

    dfs(np.zeros(n_rects, dtype=np.int8), np.zeros(n_points, dtype=np.int8))
    assert len(solution_masks) == 1, "expected a unique rectangle decomposition"
    return [rects[rid] for rid in solution_masks[0]]


def _render(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    cover = np.zeros(shape, dtype=np.int16)
    for r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1

    out = np.zeros(shape, dtype=int)
    out[cover > 0] = RED
    out[(cover % 2) == 1] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden axis-aligned rectangles from yellow corner markers and color cells by coverage parity.

    Input structure:
    - The background is black and the only foreground pixels are yellow single-cell corner markers.
    - Each hidden rectangle is axis-aligned and all four of its corners are marked in yellow.
    - A yellow marker may be shared by up to two rectangles, so the total marker count need not be a multiple of four.

    Rule:
    - Find the unique globally consistent set of rectangles whose corners are exactly the yellow markers.
    - Count, for every cell including rectangle borders, how many recovered rectangle boxes cover that cell.
    - Recolor odd positive coverage blue, even positive coverage red, and uncovered cells black; original yellow markers are recolored by the same rule.

    Key insight:
    - The corner markers must be solved as a global rectangle-selection constraint problem, then the output color comes from full coverage-depth parity rather than from a simple overlap threshold.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    points = _extract_points(g)
    rects = _search_unique_rectangles(points)
    return _render(rects, g.shape)


def _point_use_counts(rects: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for rect in rects:
        for pt in _corners(rect):
            counts[pt] = counts.get(pt, 0) + 1
    return counts


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strict_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[0] and a[1] < b[1] and a[2] > b[2] and a[3] > b[3]


def _free_values(low: int, high: int, used: set[int]) -> list[int]:
    return [v for v in range(low, high + 1) if v not in used]


def _sample_fresh_rect(height: int, width: int, used_rows: set[int], used_cols: set[int], rng: random.Random) -> tuple[int, int, int, int] | None:
    row_pairs = []
    for r0 in range(0, height - 2):
        for r1 in range(r0 + 2, height):
            if r0 in used_rows or r1 in used_rows:
                continue
            if r1 - r0 > 10:
                continue
            row_pairs.append((r0, r1))

    col_pairs = []
    for c0 in range(0, width - 2):
        for c1 in range(c0 + 2, width):
            if c0 in used_cols or c1 in used_cols:
                continue
            if c1 - c0 > 10:
                continue
            col_pairs.append((c0, c1))

    if not row_pairs or not col_pairs:
        return None
    r0, r1 = rng.choice(row_pairs)
    c0, c1 = rng.choice(col_pairs)
    return (r0, c0, r1, c1)


def _sample_shared_corner(rects: list[tuple[int, int, int, int]], height: int, width: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    counts = _point_use_counts(rects)
    used_rows = {r for rect in rects for r in (rect[0], rect[2])}
    used_cols = {c for rect in rects for c in (rect[1], rect[3])}

    corners = []
    for rect in rects:
        for name, point in zip(("tl", "tr", "bl", "br"), _corners(rect)):
            if counts[point] == 1:
                corners.append((name, point))
    rng.shuffle(corners)

    for name, (r, c) in corners:
        if name == "tl":
            rows = _free_values(0, r - 2, used_rows)
            cols = _free_values(0, c - 2, used_cols)
            if rows and cols:
                return (rng.choice(rows), rng.choice(cols), r, c)
        elif name == "tr":
            rows = _free_values(0, r - 2, used_rows)
            cols = _free_values(c + 2, width - 1, used_cols)
            if rows and cols:
                return (rng.choice(rows), c, r, rng.choice(cols))
        elif name == "bl":
            rows = _free_values(r + 2, height - 1, used_rows)
            cols = _free_values(0, c - 2, used_cols)
            if rows and cols:
                return (r, rng.choice(cols), rng.choice(rows), c)
        else:
            rows = _free_values(r + 2, height - 1, used_rows)
            cols = _free_values(c + 2, width - 1, used_cols)
            if rows and cols:
                return (r, c, rng.choice(rows), rng.choice(cols))
    return None


def _sample_inner_or_outer(anchor: tuple[int, int, int, int], height: int, width: int, used_rows: set[int], used_cols: set[int], rng: random.Random) -> tuple[int, int, int, int] | None:
    if rng.random() < 0.5:
        inner_rows = [r for r in range(anchor[0] + 1, anchor[2]) if r not in used_rows]
        inner_cols = [c for c in range(anchor[1] + 1, anchor[3]) if c not in used_cols]
        pairs_r = [(r0, r1) for r0 in inner_rows for r1 in inner_rows if r1 - r0 >= 2]
        pairs_c = [(c0, c1) for c0 in inner_cols for c1 in inner_cols if c1 - c0 >= 2]
        if pairs_r and pairs_c:
            r0, r1 = rng.choice(pairs_r)
            c0, c1 = rng.choice(pairs_c)
            return (r0, c0, r1, c1)
    outer_top = [r for r in range(0, anchor[0]) if r not in used_rows]
    outer_bottom = [r for r in range(anchor[2] + 1, height) if r not in used_rows]
    outer_left = [c for c in range(0, anchor[1]) if c not in used_cols]
    outer_right = [c for c in range(anchor[3] + 1, width) if c not in used_cols]
    if outer_top and outer_bottom and outer_left and outer_right:
        return (rng.choice(outer_top), rng.choice(outer_left), rng.choice(outer_bottom), rng.choice(outer_right))
    return None


def _sample_overlap(anchor: tuple[int, int, int, int], height: int, width: int, used_rows: set[int], used_cols: set[int], rng: random.Random) -> tuple[int, int, int, int] | None:
    r0a, c0a, r1a, c1a = anchor
    row_inside = [r for r in range(r0a + 1, r1a) if r not in used_rows]
    col_inside = [c for c in range(c0a + 1, c1a) if c not in used_cols]
    row_above = [r for r in range(0, r0a) if r not in used_rows]
    row_below = [r for r in range(r1a + 1, height) if r not in used_rows]
    col_left = [c for c in range(0, c0a) if c not in used_cols]
    col_right = [c for c in range(c1a + 1, width) if c not in used_cols]

    patterns = []
    if row_inside and row_below and col_inside and col_right:
        patterns.append((rng.choice(row_inside), rng.choice(col_inside), rng.choice(row_below), rng.choice(col_right)))
    if row_above and row_inside and col_inside and col_right:
        patterns.append((rng.choice(row_above), rng.choice(col_inside), rng.choice(row_inside), rng.choice(col_right)))
    if row_inside and row_below and col_left and col_inside:
        patterns.append((rng.choice(row_inside), rng.choice(col_left), rng.choice(row_below), rng.choice(col_inside)))
    if row_above and row_inside and col_left and col_inside:
        patterns.append((rng.choice(row_above), rng.choice(col_left), rng.choice(row_inside), rng.choice(col_inside)))
    if not patterns:
        return None
    return rng.choice(patterns)


def _valid_rect(rect: tuple[int, int, int, int], height: int, width: int) -> bool:
    r0, c0, r1, c1 = rect
    return 0 <= r0 < r1 < height and 0 <= c0 < c1 < width and (r1 - r0) >= 2 and (c1 - c0) >= 2


def _pairwise_clear(rects: list[tuple[int, int, int, int]]) -> bool:
    counts = _point_use_counts(rects)
    if any(v > 2 for v in counts.values()):
        return False

    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            shared = len(set(_corners(a)) & set(_corners(b)))
            if shared > 1:
                return False
    return True


def _coverage_is_clear(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    cover = np.zeros(shape, dtype=np.int16)
    for r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    overlap = int(np.count_nonzero(cover >= 2))
    colored = int(np.count_nonzero(cover))
    return colored > 0 and overlap * 4 <= colored * 3 and int(cover.max()) <= 4


def _relation_counts(rects: list[tuple[int, int, int, int]]) -> dict[str, int]:
    out = {"shared": 0, "contain": 0, "overlap": 0, "disjoint": 0}
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            shared_corner = bool(set(_corners(a)) & set(_corners(b)))
            if shared_corner:
                out["shared"] += 1
            elif _strict_contains(a, b) or _strict_contains(b, a):
                out["contain"] += 1
            elif _intersection_area(a, b) > 0:
                out["overlap"] += 1
            else:
                out["disjoint"] += 1
    return out


def _build_attempt(rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    height = rng.randint(16, 29)
    width = rng.randint(16, 29)
    target = rng.choices([2, 3, 4], weights=[3, 4, 2], k=1)[0]

    rects: list[tuple[int, int, int, int]] = []
    first = _sample_fresh_rect(height, width, set(), set(), rng)
    if first is None:
        return None
    rects.append(first)

    modes = ["disjoint", "overlap", "contain", "share"]
    mode_weights = [2, 3, 2, 3]

    while len(rects) < target:
        used_rows = {r for rect in rects for r in (rect[0], rect[2])}
        used_cols = {c for rect in rects for c in (rect[1], rect[3])}
        anchor = rng.choice(rects)
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]

        candidate = None
        if mode == "disjoint":
            candidate = _sample_fresh_rect(height, width, used_rows, used_cols, rng)
        elif mode == "share":
            candidate = _sample_shared_corner(rects, height, width, rng)
        elif mode == "contain":
            candidate = _sample_inner_or_outer(anchor, height, width, used_rows, used_cols, rng)
        else:
            candidate = _sample_overlap(anchor, height, width, used_rows, used_cols, rng)

        if candidate is None or not _valid_rect(candidate, height, width) or candidate in rects:
            continue

        trial = rects + [candidate]
        if not _pairwise_clear(trial):
            continue
        if not _coverage_is_clear(trial, (height, width)):
            continue
        rects = trial

    relations = _relation_counts(rects)
    if sum(relations.values()) > 0 and sum(relations[k] for k in ("shared", "contain", "overlap")) == 0 and rng.random() < 0.7:
        return None
    return rects


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        rects = _build_attempt(rng)
        if rects is None:
            continue

        height = max(r[2] for r in rects) + 1
        width = max(r[3] for r in rects) + 1
        grid = np.zeros((height, width), dtype=int)
        for rect in rects:
            for r, c in _corners(rect):
                grid[r, c] = YELLOW

        try:
            solved = sorted(_search_unique_rectangles(_extract_points(grid)))
        except AssertionError:
            continue

        if solved != sorted(rects):
            continue
        return grid

    raise RuntimeError("failed to generate a valid invisible-rectangle input")
