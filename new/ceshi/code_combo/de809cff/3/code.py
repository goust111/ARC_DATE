# Combo variant of de809cff (index 3)
# combo_spec: D1/1+D13/2
# Name: 同侧色补丁+单向映射

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
# - The cleaned scene still has two large foreground regions on black background,
#   with black dents plus one-cell noise.
# - Original task:
#     larger blue dent or smaller red dent -> stamp a 3x3 patch in the OTHER
#     foreground color -> mark the center teal.
# - Fused variant:
#     1. Determine the active host color as the larger cleaned region.
#     2. If a dent belongs to the active region, stamp a 3x3 patch in the SAME
#        host color and then mark its center teal.
#     3. If a dent belongs to the inactive region, leave it black.
# - Example trace:
#     clean 3x3 around a blue dent has 6 blue, 2 black-edge cells, 1 red.
#     Original output: whole 3x3 turns red, center teal.
#     Combo output when blue is the larger region: whole 3x3 turns blue,
#     center teal.
#     A red dent in the smaller region differs even more: original would create
#     a blue patch with teal center, but the combo leaves the red-region dent as
#     a black hole.
#
# b) Complexity calibration
# - The active/inactive decision is kept binary and repeated many times by using
#   multiple dents in both colors.
# - To make the D1 inversion visible, the generator requires at least a few
#   active dents whose same-color 3x3 repair changes neighbor cells, not only
#   the center cell.
# - To make the D13 asymmetry visible, the inactive region is forced to contain
#   several dents that remain black in the final output.
#
# c) Main() assumption review
# - `_main_colors(...)` still assumes exactly two dominant non-black colors.
#   The generator preserves that palette.
# - `_clean_foreground(...)` still assumes every non-main-color cell is noise or
#   local contamination. The combo does not alter the corruption model, so this
#   denoising step remains valid.
# - `_dent_centers(...)` still assumes a true dent can be assigned to the local
#   majority color when that color occupies at least 5 cells in the cleaned 3x3
#   and strictly beats the other color.
# - The old solver stamped every detected center and always used the opposite
#   color. Both assumptions are now invalid: `main(...)` must first choose an
#   active host color, then stamp only active dents, and must stamp them with
#   their own host color instead of the other one.
# - Starting from `clean.copy()` would incorrectly heal inactive dents unless
#   they are explicitly written back to black at the end.
#
# d) Placement legality
# - Dent centers are chosen only from host cells that remain detectable after
#   blackening, so `_dent_centers(...)` can recover them.
# - Centers from both colors obey a global Chebyshev spacing constraint, so an
#   active 3x3 repair cannot overwrite an inactive dent center.
# - Optional 2-cell dents are expanded only when the second cell also remains a
#   legal dent center and does not violate the global spacing constraints.
#
# e) Visual observability
# - The larger-region rule is human-visible because the generator enforces a
#   noticeable area gap between the two cleaned colors.
# - Each sample contains multiple active repairs and multiple inactive black
#   dents, so the one-way mapping has repeated support.
# - At least a few active dents must be edge-adjacent or boundary-adjacent, so
#   the same-color repair visibly extends into nearby non-host cells.
#
# f) Visual cleanliness
# - Noise remains one-cell only and is cleaned before any dent transformation.
# - Dent spacing avoids heavy overlap between local 3x3 effects, keeping teal
#   markers, black holes, and repaired patches easy to distinguish.
#
# g) Color-mapping inference
# - Only two foreground colors participate, and neither rule uses hidden color
#   ids. The active color is inferred from visible area, and same-color repair
#   is inferred from repeated active dents whose surrounding cells become the
#   host color instead of the other foreground color.
#
# h) Anti-degeneration checks
# - The generator rejects samples whose combo output would match the D1-only
#   variant (same-color repair on both regions) or the D13-only variant
#   (one-way activation but opposite-color repair).
# - This forces both fused dimensions to manifest in every accepted sample.
#
# i) New implicit assumptions introduced by the combo
# - `main(...)` now assumes the larger cleaned region is still unambiguous after
#   dents are punched out and after noise is removed. The generator enforces an
#   area gap before and after cleanup.
# - `main(...)` also assumes active dents include some visually revealing cases
#   where a same-color 3x3 affects neighbors, not just the center. The
#   generator measures that count on the solver-resolved centers and rejects
#   samples that fail it.


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


def _active_host_color(clean: np.ndarray, colors: tuple[int, int]) -> int:
    counts = {int(color): int(np.sum(clean == int(color))) for color in colors}
    return int(max(colors, key=lambda color: counts[int(color)]))


def _render_combo(
    clean: np.ndarray, centers: list[tuple[int, int, int]], active: int
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    active_centers: list[tuple[int, int]] = []
    inactive_centers: list[tuple[int, int]] = []
    for r, c, host in centers:
        if int(host) == int(active):
            output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(host)
            active_centers.append((r, c))
        else:
            inactive_centers.append((r, c))
    for r, c in active_centers:
        output[r, c] = Color.TEAL
    for r, c in inactive_centers:
        output[r, c] = Color.BLACK
    return output


def _render_same_side_bidir(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host in centers:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(host)
    for r, c, _host in centers:
        output[r, c] = Color.TEAL
    return output


def _render_one_way_opposite(
    clean: np.ndarray,
    centers: list[tuple[int, int, int]],
    active: int,
    colors: tuple[int, int],
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host in centers:
        if int(host) == int(active):
            other = int(colors[1] if int(host) == int(colors[0]) else colors[0])
            output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = other
            output[r, c] = Color.TEAL
        else:
            output[r, c] = Color.BLACK
    return output


def _center_has_visible_extension(clean: np.ndarray, r: int, c: int, host: int) -> bool:
    sub = _clean_sub_with_host(clean, r, c, host)
    return bool(np.any(sub != int(host)))


def _clean_sub_with_host(clean: np.ndarray, r: int, c: int, host: int) -> np.ndarray:
    sub = _sub3x3(clean, r, c).copy()
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    sub[r - r0, c - c0] = int(host)
    return sub


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair dents only in the larger colored region, using same-color 3x3 patches.

    Input structure:
    - Black background with two large non-black regions, typically with one
      region visibly larger than the other after cleanup.
    - Both regions may contain black dents, and the input also includes a small
      amount of one-cell foreground noise in the background or at the border
      between the two regions.

    Rule:
    - First clean the two foreground regions locally: isolated color specks are
      removed to black, and one-cell cross-boundary contamination flips back to
      the locally dominant region color.
    - Assign each original black cell to the foreground color that dominates its
      cleaned 3x3 neighborhood. Only dents in the larger cleaned region are
      transformed, and they are repaired with a 3x3 patch in that SAME host
      color rather than the opposite color.
    - Mark transformed dent centers with teal (8), but leave dents in the
      smaller region black.

    Key insight: the solver must infer two things from the examples at once:
    which color region is active, and that active dents are repaired in-place
    with their own color instead of receiving a contrasting patch.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    active = _active_host_color(clean, colors)
    centers = _dent_centers(g, clean, colors, bg)
    return _render_combo(clean, centers, active)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85
ACTIVE_AREA_GAP_MIN = 28
SOLVER_ACTIVE_AREA_GAP_MIN = 14

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
MIN_ACTIVE_DENTS = 6
MIN_INACTIVE_DENTS = 4
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13
VISIBLE_ACTIVE_DENTS_MIN = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 700
CENTER_PICK_ATTEMPTS = 20


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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
    if abs(cnt_a - cnt_b) < ACTIVE_AREA_GAP_MIN:
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


def _is_visible_patch_center(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    """True when same-color 3x3 stamping changes more than the center cell."""
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    return host_count >= 6 and bool(np.any(sub != int(host))) and _has_same_2x2(grid, r, c, host)


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
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


def _far_enough(r: int, c: int, others: list[tuple[int, int]]) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in others)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked = [] if blocked is None else list(blocked)
    for r, c in pool:
        if _far_enough(r, c, chosen) and _far_enough(r, c, blocked):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _pick_dent_centers_with_visible(
    visible: list[tuple[int, int]],
    hidden: list[tuple[int, int]],
    n: int,
    min_visible: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    blocked = [] if blocked is None else list(blocked)
    visible_pool = list(visible)
    hidden_pool = list(hidden)
    rng.shuffle(visible_pool)
    rng.shuffle(hidden_pool)

    for r, c in visible_pool:
        if len(chosen) >= min_visible:
            break
        if _far_enough(r, c, chosen) and _far_enough(r, c, blocked):
            chosen.append((r, c))
    if len(chosen) < min_visible:
        return None

    remaining = [pos for pos in visible_pool if pos not in chosen] + hidden_pool
    rng.shuffle(remaining)
    for r, c in remaining:
        if len(chosen) >= n:
            return chosen
        if _far_enough(r, c, chosen) and _far_enough(r, c, blocked):
            chosen.append((r, c))
    return chosen if len(chosen) == n else None


def _dent_remains_detectable(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != Color.BLACK:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked_centers: list[tuple[int, int]],
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
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                other_blocked = [center for center in blocked_centers if center != (r, c)]
                if any((nr, nc) == center for center in other_blocked):
                    continue
                if not _far_enough(nr, nc, other_blocked):
                    continue
                temp = clean.copy()
                temp[r, c] = Color.BLACK
                temp[nr, nc] = Color.BLACK
                if not (
                    _dent_remains_detectable(temp, r, c, host, other)
                    and _dent_remains_detectable(temp, nr, nc, host, other)
                ):
                    continue
                cells.append((nr, nc))
                break
        result.append(cells)
    return result


def _count_active_visible_repairs(
    clean: np.ndarray, centers: list[tuple[int, int, int]], active: int
) -> int:
    return sum(
        int(int(host) == int(active) and _center_has_visible_extension(clean, r, c, host))
        for r, c, host in centers
    )


def _add_bg_noise(
    inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


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
            nsub[r - nr0, c - nc0] = int(other)
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
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b if v == int(color_a) else color_a)
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
        inp[r, c] = int(other)
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    counts = {int(color_a): int(np.sum(clean == int(color_a))), int(color_b): int(np.sum(clean == int(color_b)))}
    active_color = int(max((color_a, color_b), key=lambda color: counts[int(color)]))
    inactive_color = int(color_b if active_color == int(color_a) else color_a)

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_active_min = max(MIN_ACTIVE_DENTS, n_total // 2)
    n_active_max = n_total - MIN_INACTIVE_DENTS
    if n_active_min > n_active_max:
        return None
    n_active = rng.randint(n_active_min, n_active_max)
    n_inactive = n_total - n_active

    valid_visible = {int(color_a): [], int(color_b): []}
    valid_hidden = {int(color_a): [], int(color_b): []}
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a) and _is_safe_dent_center(clean, r, c, color_a, color_b):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_a) else valid_hidden
                target[int(color_a)].append((r, c))
            elif v == int(color_b) and _is_safe_dent_center(clean, r, c, color_b, color_a):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_b) else valid_hidden
                target[int(color_b)].append((r, c))

    if len(valid_visible[active_color]) + len(valid_hidden[active_color]) < n_active:
        return None
    if len(valid_visible[active_color]) < VISIBLE_ACTIVE_DENTS_MIN:
        return None
    if len(valid_visible[inactive_color]) + len(valid_hidden[inactive_color]) < n_inactive:
        return None

    dents_active = None
    dents_inactive = None
    for _ in range(CENTER_PICK_ATTEMPTS):
        dents_active = _pick_dent_centers_with_visible(
            valid_visible[active_color],
            valid_hidden[active_color],
            n_active,
            VISIBLE_ACTIVE_DENTS_MIN,
            rng,
        )
        if dents_active is None:
            continue
        dents_inactive = _pick_dent_centers(
            valid_visible[inactive_color] + valid_hidden[inactive_color],
            n_inactive,
            rng,
            blocked=dents_active,
        )
        if dents_inactive is not None:
            break
    if dents_active is None or dents_inactive is None:
        return None

    blocked_centers = dents_active + dents_inactive
    active_groups = _expand_to_pairs(dents_active, active_color, inactive_color, clean, blocked_centers, rng)
    inactive_groups = _expand_to_pairs(
        dents_inactive, inactive_color, active_color, clean, blocked_centers, rng
    )

    inp = clean.copy()
    for cells in active_groups + inactive_groups:
        for r, c in cells:
            inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    solver_clean = _clean_foreground(inp, (int(color_a), int(color_b)), Color.BLACK)
    resolved_active = _active_host_color(solver_clean, (int(color_a), int(color_b)))
    resolved_inactive = int(color_b if resolved_active == int(color_a) else color_a)
    if resolved_active != active_color:
        return None
    active_area = int(np.sum(solver_clean == resolved_active))
    inactive_area = int(np.sum(solver_clean == resolved_inactive))
    if active_area - inactive_area < SOLVER_ACTIVE_AREA_GAP_MIN:
        return None

    solver_centers = _dent_centers(inp, solver_clean, (int(color_a), int(color_b)), Color.BLACK)
    resolved_active_count = sum(int(int(host) == resolved_active) for _r, _c, host in solver_centers)
    resolved_inactive_count = sum(int(int(host) == resolved_inactive) for _r, _c, host in solver_centers)
    if resolved_active_count < MIN_ACTIVE_DENTS or resolved_inactive_count < MIN_INACTIVE_DENTS:
        return None
    if _count_active_visible_repairs(solver_clean, solver_centers, resolved_active) < VISIBLE_ACTIVE_DENTS_MIN:
        return None

    combo_output = _render_combo(solver_clean, solver_centers, resolved_active)
    same_side_bidir_output = _render_same_side_bidir(solver_clean, solver_centers)
    one_way_opposite_output = _render_one_way_opposite(
        solver_clean, solver_centers, resolved_active, (int(color_a), int(color_b))
    )
    if np.array_equal(combo_output, same_side_bidir_output):
        return None
    if np.array_equal(combo_output, one_way_opposite_output):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, combo_output):
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
