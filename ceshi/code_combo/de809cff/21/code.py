# Combo variant of de809cff (index 21)
# combo_spec: D1/0+D5/0+D9/0
# Name: 十字形补丁+边界裁剪+矩形区域

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    Base task:
#    - Clean isolated wrong-color specks and one-cell boundary contamination.
#    - Detect each original black dent by the cleaned 3x3 strict majority.
#    - Paint the full 3x3 box in the opposite color, then mark the center teal.
#    Combo task:
#    - Keep the same cleanup and dent detection.
#    - Reconstruct the dent host region after cleanup because dent centers remain black there.
#    - Propose an orthogonal cross instead of a 3x3 square, then keep only the cross cells that
#      still lie inside the dent's host region.
#    - Example around a boundary dent inside a rectangular left block:
#          A A | B          A A | B
#          A . | B    ->    A 8 | B
#          A A | B          X X | B
#      Here the old task would flood a 3x3 square across the boundary, while the combo paints
#      only the cross and clips away the right arm, yielding a T-shaped fragment.
# b) Complexity calibration:
#    - D1 is visible because interior dents suppress diagonal recolors, so many marks are clearly
#      plus signs instead of solid 3x3 blocks.
#    - D9 is visible because a quota of dents is placed adjacent to the region split, so some
#      plus signs lose one arm at the boundary.
#    - D5 simplifies the scene into two large rectangles, which lowers shape ambiguity while still
#      leaving enough interior cells and enough straight boundary cells to demonstrate both full
#      and clipped crosses repeatedly.
# c) Hidden assumptions audited:
#    - `_main_colors` must not silently depend on color id when the two rectangles have equal area;
#      this version uses first appearance to break ties.
#    - `_clean_foreground` still assumes the only corruption is isolated one-cell noise; the
#      generator preserves that contract.
#    - `_dent_centers` still assumes every real dent is an original black cell with a strict local
#      host majority in the cleaned 3x3. The generator explicitly checks that before placement.
#    - Writing directly to `out[r-1:r+2, c-1:c+2]` is invalid twice over in the combo: D1 removes
#      diagonals, and D9 forbids recoloring cells outside the host region. The solver therefore
#      routes every patch write through a bounded shape writer plus a reconstructed host mask.
#    - Using `clean` itself as the clipping mask is unsafe because paired dents remain black in
#      `clean`. The solver fills dent centers back with their host colors before clipping.
# d) Placement legality:
#    - The generator builds exactly two axis-aligned rectangles inside the black canvas, split by a
#      single straight line. This guarantees clean rectangular ownership for every candidate dent.
#    - Dent seeds are globally Chebyshev-spaced across both colors, so opposite-color crosses do
#      not collapse into ambiguous overlaps.
#    - Optional 2-cell pairs are added only when the partner is still a valid dent center and does
#      not violate spacing against all already reserved centers except its own anchor.
# e) Visual observability:
#    - Every accepted sample must contain several interior dents and several split-boundary dents.
#      That gives the solver two repeated motifs to compare: full crosses inside rectangles and
#      clipped T-fragments on the straight inter-rectangle boundary.
#    - The generator also rejects samples where clipped-cross output equals unclipped-cross output
#      or where clipped-cross output is too close to clipped-square output, preventing invisible
#      degeneration back to D9-only or D1-only behavior.
# f) Visual cleanliness:
#    - The scene has only three semantic layers: black background, two rectangular foreground
#      blocks, and the teal-centered opposite-color dent marks. Straight boundaries and spacing
#      constraints keep those layers legible.
# g) Color-rule inferability and new solver assumptions:
#    - Colors are still structurally interpretable: two main foreground colors exchange roles at
#      each dent, and teal marks the exact dent centers.
#    - `main()` now assumes clipped dents belong to a recoverable host region after cleanup; the
#      generator guarantees this by keeping regions rectangular and by validating every dent center
#      under the same local majority test used by the solver.

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    seen: list[int] = []
    present: set[int] = set()
    for value in g.flat:
        color = int(value)
        if color == int(bg) or color in present:
            continue
        present.add(color)
        seen.append(color)
    counts = {color: int(np.sum(g == color)) for color in seen}
    seen.sort(key=lambda color: counts[color], reverse=True)
    assert len(seen) >= 2, "expected two foreground colors"
    return int(seen[0]), int(seen[1])


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    Two sweeps are enough for the intended corruption pattern: the first removes
    stray protrusions, the second stabilizes the corrected local majority.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                sub = cur[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == bg))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = bg
                elif own <= 2:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _reconstruct_regions(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    """Fill detected dent centers back with their host colors before clipping patches."""
    region_mask = clean.copy()
    for r, c, host in centers:
        region_mask[r, c] = int(host)
    return region_mask


def _cross_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
    return cells


def _paint_clipped_cross(
    out: np.ndarray, region_mask: np.ndarray, r: int, c: int, host: int, stamp_color: int
) -> None:
    for rr, cc in _cross_cells(out.shape, r, c):
        if int(region_mask[rr, cc]) == int(host):
            out[rr, cc] = int(stamp_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents inside two rectangular foreground regions into clipped opposite-color crosses.

    Input structure:
    - Black background with exactly two large foreground rectangles in different colors.
    - Each rectangle may contain black dents plus a few isolated one-cell specks in the background
      or one-cell cross-boundary contamination that should be removed first.

    Rule:
    - First clean the foreground locally so isolated specks revert to black and one-cell boundary
      contamination flips back to its host rectangle color.
    - For each original black dent whose cleaned 3x3 neighborhood is mostly one foreground color,
      infer that host rectangle, reconstruct the host region mask, and paint only the orthogonal
      cross centered at the dent in the opposite foreground color.
    - Clip that cross to the dent's host region boundary, then recolor the dent center teal (8).

    Key insight: The dent host is still inferred from the cleaned 3x3 majority, but the visible
    marker is no longer a full 3x3 block; it is an opposite-color plus sign whose arms are cut
    off wherever the proposed cross would leave the dent's own rectangular region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    centers = _dent_centers(g, clean, colors, bg)
    region_mask = _reconstruct_regions(clean, centers)

    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        _paint_clipped_cross(out, region_mask, r, c, int(host), int(other))

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
OUTER_SIZE_FRAC_MIN, OUTER_SIZE_FRAC_MAX = 0.72, 0.90
OUTER_MARGIN_MIN = 1
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84
MIN_RECT_SIDE = 8

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13
CLIPPED_DENT_FRAC = 0.35
MIN_CLIPPED_DENTS_PER_COLOR = 2
MIN_INTERIOR_DENTS_PER_COLOR = 2
MIN_SHARED_CLIPPED_PER_COLOR = 1

# --- CFG: visual distinctness ---
MIN_TOTAL_CLIP_DELTA = 8
MIN_TOTAL_DIAGONAL_DELTA = 18

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400
BUILD_RETRY_ATTEMPTS = 120
SELECT_RETRY_ATTEMPTS = 160


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Build two large adjacent rectangles separated by one straight split."""
    min_outer = max(14, int(np.ceil(N * OUTER_SIZE_FRAC_MIN)))
    max_outer = min(N - 2 * OUTER_MARGIN_MIN, int(np.floor(N * OUTER_SIZE_FRAC_MAX)))
    if min_outer > max_outer:
        return None

    for _ in range(BUILD_RETRY_ATTEMPTS):
        outer_h = rng.randint(min_outer, max_outer)
        outer_w = rng.randint(min_outer, max_outer)
        coverage = (outer_h * outer_w) / float(N * N)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue

        max_r0 = N - outer_h - OUTER_MARGIN_MIN
        max_c0 = N - outer_w - OUTER_MARGIN_MIN
        if max_r0 < OUTER_MARGIN_MIN or max_c0 < OUTER_MARGIN_MIN:
            continue
        r0 = rng.randint(OUTER_MARGIN_MIN, max_r0)
        c0 = rng.randint(OUTER_MARGIN_MIN, max_c0)
        r1 = r0 + outer_h
        c1 = c0 + outer_w

        orientation = rng.choice(["vertical", "horizontal"])
        grid = np.zeros((N, N), dtype=int)

        if orientation == "vertical":
            min_split = max(MIN_RECT_SIDE, int(np.ceil(MIN_REGION_CELLS / outer_h)))
            if outer_w < 2 * min_split:
                continue
            left_w = rng.randint(min_split, outer_w - min_split)
            mid = c0 + left_w
            grid[r0:r1, c0:mid] = int(color_a)
            grid[r0:r1, mid:c1] = int(color_b)
        else:
            min_split = max(MIN_RECT_SIDE, int(np.ceil(MIN_REGION_CELLS / outer_w)))
            if outer_h < 2 * min_split:
                continue
            top_h = rng.randint(min_split, outer_h - min_split)
            mid = r0 + top_h
            grid[r0:mid, c0:c1] = int(color_a)
            grid[mid:r1, c0:c1] = int(color_b)

        cnt_a = int(np.sum(grid == int(color_a)))
        cnt_b = int(np.sum(grid == int(color_b)))
        if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
            continue
        return grid

    return None


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _cross_neighbor_stats(
    grid: np.ndarray, r: int, c: int, host: int, other: int
) -> tuple[int, bool, bool]:
    host_neighbors = 0
    touches_other = False
    touches_bg = False
    h, w = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        value = int(grid[rr, cc])
        if value == int(host):
            host_neighbors += 1
        elif value == int(other):
            touches_other = True
        else:
            touches_bg = True
    return host_neighbors, touches_other, touches_bg


def _far_enough(candidate: tuple[int, int], reserved: list[tuple[int, int]]) -> bool:
    r, c = candidate
    return all(
        max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in reserved
    )


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    reserved: list[tuple[int, int]],
    rng: random.Random,
    shared_candidates: list[tuple[int, int]],
    clipped_candidates: list[tuple[int, int]],
    interior_candidates: list[tuple[int, int]],
    shared_target: int,
    clipped_target: int,
    interior_target: int,
) -> list[tuple[int, int]] | None:
    candidate_set = set(candidates)
    shared_pool = [p for p in shared_candidates if p in candidate_set]
    clipped_extra_pool = [p for p in clipped_candidates if p in candidate_set and p not in set(shared_pool)]
    interior_pool = [p for p in interior_candidates if p in candidate_set]

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        blocked = list(reserved)

        for pool, target in (
            (shared_pool, shared_target),
            (clipped_extra_pool, max(0, clipped_target - shared_target)),
            (interior_pool, interior_target),
        ):
            if target <= 0:
                continue
            order = pool.copy()
            rng.shuffle(order)
            added = 0
            for r, c in order:
                if (r, c) in chosen:
                    continue
                if _far_enough((r, c), blocked):
                    chosen.append((r, c))
                    blocked.append((r, c))
                    added += 1
                    if added >= target:
                        break
            if added < target:
                break
        else:
            remaining = [p for p in candidates if p not in chosen]
            rng.shuffle(remaining)
            for r, c in remaining:
                if _far_enough((r, c), blocked):
                    chosen.append((r, c))
                    blocked.append((r, c))
                    if len(chosen) >= n:
                        return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(r, c)]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(clean[nr, nc]) != int(host) or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any(
                    (rr, cc) != (r, c)
                    and max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION
                    for rr, cc in reserved
                ):
                    continue
                cells.append((nr, nc))
                reserved.add((nr, nc))
                break
        result.append(cells)
    return result


def _square_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [
        (rr, cc)
        for rr in range(max(0, r - 1), min(h, r + 2))
        for cc in range(max(0, c - 1), min(w, c + 2))
    ]


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
    *,
    patch_shape: str,
    clipped: bool,
) -> np.ndarray:
    output = clean.copy()
    for dent_groups, host, stamp_color in (
        (dent_groups_a, color_a, color_b),
        (dent_groups_b, color_b, color_a),
    ):
        for cells in dent_groups:
            for r, c in cells:
                if patch_shape == "cross":
                    paint_cells = _cross_cells(clean.shape, r, c)
                else:
                    paint_cells = _square_cells(clean.shape, r, c)
                for rr, cc in paint_cells:
                    if clipped and int(clean[rr, cc]) != int(host):
                        continue
                    output[rr, cc] = int(stamp_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _group_center_stats(
    clean: np.ndarray, groups: list[list[tuple[int, int]]], host: int, other: int
) -> tuple[int, int, int]:
    clipped = 0
    interior = 0
    shared = 0
    for cells in groups:
        for r, c in cells:
            host_neighbors, touches_other, touches_bg = _cross_neighbor_stats(clean, r, c, host, other)
            if host_neighbors < 4:
                clipped += 1
                if touches_other:
                    shared += 1
            if host_neighbors == 4 and not touches_other and not touches_bg:
                interior += 1
    return clipped, interior, shared


def _add_bg_noise(
    inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r = int(r_)
        c = int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = (
            _Counter(nearby_fg).most_common(1)[0][0]
            if nearby_fg
            else rng.choice([color_a, color_b])
        )
        inp[r, c] = speck_color
        placed += 1


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == color)) == 4:
                    return True
    return False


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = other
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == int(host)))
        other_count = int(np.sum(sub_now == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    valid: dict[int, list[tuple[int, int]]] = {color_a: [], color_b: []}
    clipped: dict[int, list[tuple[int, int]]] = {color_a: [], color_b: []}
    shared: dict[int, list[tuple[int, int]]] = {color_a: [], color_b: []}
    interior: dict[int, list[tuple[int, int]]] = {color_a: [], color_b: []}

    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if not _is_safe_dent_center(clean, r, c, v, other):
                continue
            valid[v].append((r, c))
            host_neighbors, touches_other, touches_bg = _cross_neighbor_stats(clean, r, c, v, other)
            if host_neighbors < 4:
                clipped[v].append((r, c))
                if touches_other:
                    shared[v].append((r, c))
            if host_neighbors == 4 and not touches_other and not touches_bg:
                interior[v].append((r, c))

    target_clip_a = max(MIN_CLIPPED_DENTS_PER_COLOR, int(round(n_a * CLIPPED_DENT_FRAC)))
    target_clip_b = max(MIN_CLIPPED_DENTS_PER_COLOR, int(round(n_b * CLIPPED_DENT_FRAC)))
    if (
        len(shared[color_a]) < MIN_SHARED_CLIPPED_PER_COLOR
        or len(shared[color_b]) < MIN_SHARED_CLIPPED_PER_COLOR
        or len(clipped[color_a]) < target_clip_a
        or len(clipped[color_b]) < target_clip_b
        or len(interior[color_a]) < MIN_INTERIOR_DENTS_PER_COLOR
        or len(interior[color_b]) < MIN_INTERIOR_DENTS_PER_COLOR
    ):
        return None

    reserved: list[tuple[int, int]] = []
    dents: dict[int, list[tuple[int, int]]] = {}
    order = [color_a, color_b]
    rng.shuffle(order)
    targets = {color_a: n_a, color_b: n_b}
    clip_targets = {color_a: target_clip_a, color_b: target_clip_b}
    for color in order:
        chosen = _pick_dent_centers(
            valid[color],
            targets[color],
            reserved,
            rng,
            shared[color],
            clipped[color],
            interior[color],
            MIN_SHARED_CLIPPED_PER_COLOR,
            clip_targets[color],
            MIN_INTERIOR_DENTS_PER_COLOR,
        )
        if chosen is None:
            return None
        dents[color] = chosen
        reserved.extend(chosen)

    all_reserved = set(dents[color_a] + dents[color_b])
    expand_order = [color_a, color_b]
    rng.shuffle(expand_order)
    groups: dict[int, list[list[tuple[int, int]]]] = {}
    for color in expand_order:
        other = color_b if color == color_a else color_a
        groups[color] = _expand_to_pairs(dents[color], color, other, clean, all_reserved, rng)
    groups_a = groups[color_a]
    groups_b = groups[color_b]

    clip_a, interior_a, shared_a = _group_center_stats(clean, groups_a, color_a, color_b)
    clip_b, interior_b, shared_b = _group_center_stats(clean, groups_b, color_b, color_a)
    if (
        clip_a < target_clip_a
        or clip_b < target_clip_b
        or interior_a < MIN_INTERIOR_DENTS_PER_COLOR
        or interior_b < MIN_INTERIOR_DENTS_PER_COLOR
        or shared_a < MIN_SHARED_CLIPPED_PER_COLOR
        or shared_b < MIN_SHARED_CLIPPED_PER_COLOR
    ):
        return None

    output = _stamp_dents(
        clean, groups_a, groups_b, color_a, color_b, patch_shape="cross", clipped=True
    )
    unclipped_cross = _stamp_dents(
        clean, groups_a, groups_b, color_a, color_b, patch_shape="cross", clipped=False
    )
    clipped_square = _stamp_dents(
        clean, groups_a, groups_b, color_a, color_b, patch_shape="square", clipped=True
    )

    clip_delta = int(np.sum(output != unclipped_cross))
    diagonal_delta = int(np.sum(output != clipped_square))
    if clip_delta < MIN_TOTAL_CLIP_DELTA or diagonal_delta < MIN_TOTAL_DIAGONAL_DELTA:
        return None

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
