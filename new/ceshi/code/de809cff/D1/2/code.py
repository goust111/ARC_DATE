# Variant of de809cff
# Dimension: D1 — 核心变换规则：凹痕到3x3补丁 (transform)
# Change: 每个黑色凹痕的3x3邻域多数前景色为A时，凹痕变为3x3的前景色B补丁，中心为teal(8)... → 凹痕大小决定补丁颜色：单像素用对侧色，多像素用同侧色
#
# Design analysis
# spec:
#   core_change:
#     original: each detected black dent always stamps the opposite foreground color.
#     variant: a 1-cell dent still stamps the opposite color, but a 4-connected dent
#       cluster of size >= 2 stamps its own host color instead.
#   concrete_pixel_behavior:
#     objects: two large foreground regions on black background, plus small black dents
#       and removable 1-cell noise.
#     dent_types:
#       single: one black cell inside/along a region; output shows a 3x3 opposite-color
#         patch with a teal center.
#       multi: an orthogonally adjacent 2-cell dent pair; output shows the union of the
#         two 3x3 patches in the host color, with both original black cells marked teal.
#     trace_example:
#       clean local area:
#         3 3 3 3
#         3 3 3 3
#         3 3 3 3
#       input single dent:
#         3 3 3 3
#         3 0 3 3
#         3 3 3 3
#       original output: surrounding 3x3 becomes color B, center becomes 8.
#       variant output: same as original because size == 1.
#       input pair dent:
#         3 3 3 3
#         3 0 0 3
#         3 3 3 3
#       original output: both 3x3 neighborhoods become color B.
#       variant output: both 3x3 neighborhoods stay/turn to color 3 instead, and only
#         the two dent cells become 8. The branching point is therefore "group dent
#         cells first, then choose patch color from group size".
#   complexity_calibration:
#     keep the same global scene grammar, colors, and noise model.
#     keep only two dent size buckets visible to the solver: size 1 and size 2.
#     generator guarantees both buckets appear repeatedly so the rule is inferable.
#   implicit_old_assumptions_that_break:
#     _dent_centers() in the seed solver treats every black cell independently.
#       That is invalid now because two adjacent black cells must be classified as one
#       dent cluster before deciding patch color.
#     main() unconditionally assigns the opposite color to every detected dent.
#       That would overwrite correct same-host patches for multi-cell dents.
#     generator's pair expansion only had to avoid patch overlap; it did not have to
#       prevent two expanded dents from becoming one larger black connected component.
#       The new generator must separate dent groups globally.
#   placement_legality:
#     every dent cell must still satisfy the local host-majority test after cleaning.
#     multi-cell dents are restricted to 2-cell orthogonal pairs inside one host region.
#     different dent groups are globally separated so the solver's 4-connected grouping
#       matches the generator's intended grouping.
#   visual_observability:
#     each generated grid contains several singles and several pairs.
#     singles produce obvious opposite-color 3x3 patches; pairs produce teal centers on
#       a same-color local patch. This creates a visible categorical contrast without
#       adding extra colors or hidden ordering tricks.
#   cleanliness:
#     only two large regions, black background, and teal markers remain.
#     no extra symbolic overlay is introduced; the output stays legible.
#   solver_generator_contract:
#     solver assumes dent groups are 4-connected black candidate cells whose local host
#       color is consistent within the group.
#     generator enforces that by placing only same-region pairs and rejecting any sample
#       where the oracle or visibility checks fail.

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


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

    Two sweeps are enough for the observed corruption pattern: the first removes
    protruding wrong-color cells, the second lets their neighboring true cells
    keep the corrected local majority.
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


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _is_dent_candidate(clean: np.ndarray, colors: tuple[int, int], r: int, c: int, host: int) -> bool:
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    other = colors[1] if int(host) == int(colors[0]) else colors[0]
    return counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]


def _dent_candidate_map(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = original.shape
    mask = np.zeros((h, w), dtype=bool)
    host_map = np.full((h, w), -1, dtype=int)
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        if _is_dent_candidate(clean, colors, rr, cc, int(host)):
            mask[rr, cc] = True
            host_map[rr, cc] = int(host)
    return mask, host_map


def _dent_groups(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[list[tuple[int, int]], int]]:
    mask, host_map = _dent_candidate_map(original, clean, colors, bg)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: list[tuple[list[tuple[int, int]], int]] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        host = int(host_map[rr, cc])
        stack = [(rr, cc)]
        seen[rr, cc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            cells.append((cr, cc2))
            for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                if int(host_map[nr, nc]) != host:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        cells.sort()
        groups.append((cells, host))
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into teal-marked 3x3 patches whose
    patch color depends on dent size.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain small black dents plus a few isolated
      one-cell color specks or boundary protrusions that should be cleaned away.
    - Some dents are single black cells, while others are 4-connected multi-cell
      clusters made of orthogonally adjacent black cells.

    Rule:
    - First denoise the two foreground regions exactly as in the seed task:
      isolated specks revert to black, and one-cell cross-color protrusions are
      pulled back into the locally dominant region.
    - From the original black cells, keep only those whose cleaned 3x3
      neighborhood is clearly dominated by one foreground color, then merge
      orthogonally adjacent such cells into dent groups.
    - Stamp a 3x3 patch around every dent cell: groups of size 1 use the
      opposite foreground color, groups of size at least 2 use their own host
      foreground color. Finally recolor every original dent cell teal (8).

    Key insight: The black defects must be grouped before recoloring. A lone
    defect still behaves like the original opposite-color patch, but a connected
    black cluster switches to a same-side patch, so dent area becomes the hidden
    classifier that controls the output color.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    h, w = g.shape
    groups = _dent_groups(g, clean, colors, bg)

    for cells, host in groups:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        stamp_color = int(other) if len(cells) == 1 else int(host)
        for r, c in cells:
            out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color

    for cells, _host in groups:
        for r, c in cells:
            out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
MULTI_DENT_FRAC_MIN, MULTI_DENT_FRAC_MAX = 0.18, 0.30
MIN_MULTI_PER_COLOR = 1

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles (simple occlusion, painter's algorithm)."""
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen_corners = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        mid = color_b if first == color_a else color_a
        rect_colors = [first, mid, first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    """Check whether placing `color` at (r,c) would complete a 2x2 block of that color."""
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == color)) == 4:
                    return True
    return False


def _is_boundaryish_cell(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return True
    if r == 0 or c == 0 or r == h - 1 or c == w - 1:
        return True
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if int(grid[nr, nc]) != int(host):
            return True
    return False


def _pick_dent_centers_by_color(
    valid: dict[int, list[tuple[int, int]]], counts: dict[int, int], rng: random.Random
) -> dict[int, list[tuple[int, int]]] | None:
    for _ in range(12):
        pools = {color: cells[:] for color, cells in valid.items()}
        for cells in pools.values():
            rng.shuffle(cells)

        order = [color for color, n in counts.items() for _ in range(n)]
        rng.shuffle(order)

        chosen = {color: [] for color in valid}
        all_centers: list[tuple[int, int]] = []
        ok = True
        for color in order:
            pick = None
            for idx, (r, c) in enumerate(pools[color]):
                if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in all_centers):
                    pick = idx
                    break
            if pick is None:
                ok = False
                break
            center = pools[color].pop(pick)
            chosen[color].append(center)
            all_centers.append(center)
        if ok:
            return chosen
    return None


def _multi_target(n_groups: int, rng: random.Random) -> int:
    lo = min(n_groups, max(MIN_MULTI_PER_COLOR, int(round(n_groups * MULTI_DENT_FRAC_MIN))))
    hi = min(n_groups, max(lo, int(round(n_groups * MULTI_DENT_FRAC_MAX))))
    return rng.randint(lo, hi)


def _far_from_other_dents(
    cell: tuple[int, int], occupied: set[tuple[int, int]], own_group: set[tuple[int, int]]
) -> bool:
    r, c = cell
    for rr, cc in occupied:
        if (rr, cc) in own_group:
            continue
        if max(abs(r - rr), abs(c - cc)) < DENT_PATCH_SEPARATION:
            return False
    return True


def _near_any_dent(r: int, c: int, dent_cells: set[tuple[int, int]]) -> bool:
    return any(max(abs(r - dr), abs(c - dc)) <= 1 for dr, dc in dent_cells)


def _expand_to_pairs(
    centers: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    occupied: set[tuple[int, int]],
    target_multi: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    groups = {center: [center] for center in centers}
    boundaryish = [center for center in centers if _is_boundaryish_cell(clean, center[0], center[1], host)]
    interior = [center for center in centers if center not in boundaryish]
    rng.shuffle(boundaryish)
    rng.shuffle(interior)
    order = boundaryish + interior

    expanded = 0
    for center in order:
        if expanded >= target_multi:
            break
        r, c = center
        own_group = set(groups[center])
        neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(neighbors)
        neighbors.sort(key=lambda pos: 0 if _is_boundaryish_cell(clean, pos[0], pos[1], host) else 1)
        for nr, nc in neighbors:
            if not (0 <= nr < clean.shape[0] and 0 <= nc < clean.shape[1]):
                continue
            if (nr, nc) in occupied:
                continue
            if int(clean[nr, nc]) != int(host):
                continue
            if not _is_safe_dent_center(clean, nr, nc, host, other):
                continue
            if not _far_from_other_dents((nr, nc), occupied, own_group):
                continue
            trial = clean.copy()
            for tr, tc in ((r, c), (nr, nc)):
                trial[tr, tc] = Color.BLACK
            if all(_is_dent_candidate(trial, (host, other), tr, tc, host) for tr, tc in ((r, c), (nr, nc))):
                groups[center].append((nr, nc))
                occupied.add((nr, nc))
                expanded += 1
                break

    if expanded < target_multi:
        return None

    result = list(groups.values())
    for cells in result:
        cells.sort()
    return result


def _stamp_dents_variant(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, host, other in (
        (dent_groups_a, color_a, color_b),
        (dent_groups_b, color_b, color_a),
    ):
        for cells in dent_groups:
            stamp_color = int(other) if len(cells) == 1 else int(host)
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _stamp_dents_original_style(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    dent_cells: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    """Place isolated fg-color specks in background area."""
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if _near_any_dent(r, c, dent_cells):
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` won't cause the solver to misclassify any neighbor."""
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
            own_n = int(np.sum(nsub == host))
            other_n = int(np.sum(nsub == other))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    dent_cells: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    """Place cross-color contamination at region-region boundaries."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (color_a, color_b):
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if _near_any_dent(r, c, dent_cells):
            continue
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == host))
        other_count = int(np.sum(sub_now == other))
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

    h, w = clean.shape
    valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    picked = _pick_dent_centers_by_color(valid, {color_a: n_a, color_b: n_b}, rng)
    if picked is None:
        return None

    occupied = set(picked[color_a] + picked[color_b])
    target_multi_a = _multi_target(len(picked[color_a]), rng)
    target_multi_b = _multi_target(len(picked[color_b]), rng)

    groups_a = _expand_to_pairs(picked[color_a], color_a, color_b, clean, occupied, target_multi_a, rng)
    if groups_a is None:
        return None
    groups_b = _expand_to_pairs(picked[color_b], color_b, color_a, clean, occupied, target_multi_b, rng)
    if groups_b is None:
        return None

    multi_groups = [cells for cells in groups_a + groups_b if len(cells) > 1]
    if len(multi_groups) < 2:
        return None
    output = _stamp_dents_variant(clean, groups_a, groups_b, color_a, color_b)
    original_style_output = _stamp_dents_original_style(clean, groups_a, groups_b, color_a, color_b)
    if np.array_equal(output, original_style_output):
        return None

    inp = clean.copy()
    dent_cells = {cell for cells in groups_a + groups_b for cell in cells}
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, dent_cells, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, dent_cells, rng)

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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
