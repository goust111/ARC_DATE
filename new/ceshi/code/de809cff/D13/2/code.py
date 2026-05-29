# Variant of de809cff
# Dimension: D13 — 两种前景色之间的对调映射关系 (meta)
# Change: 固定双色对调：区域A的凹痕用色B填充补丁，区域B的凹痕用色A填充补丁（简单二元交换）... → 单向映射：只有一种区域的凹痕触发变换

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the D13 variant
# -----------------------------------
# a) concretization_and_trace:
#    - The visible scene is still two large colored regions on black background,
#      with black dents and a small amount of one-cell noise.
#    - The original task treated dents in both colors symmetrically:
#      green dent -> 3x3 yellow patch, yellow dent -> 3x3 green patch.
#    - This variant keeps the opposite-color patch, but only for the active
#      region. The inactive region's dents remain black one-cell (or paired)
#      holes in the output.
#    - To make "region A" observable without using color ids, region A is
#      concretized as the larger cleaned foreground region by visible area.
#      Example: if blue occupies clearly more cells than red, then a black hole
#      inside blue becomes a red 3x3 patch with teal center, while a black hole
#      inside red stays black after cleanup.
# b) complexity_calibration:
#    - The rule change is small but visible: every sample contains dents in both
#      colors, so the output simultaneously shows transformed dents and
#      untouched black dents.
#    - The active/inactive choice is binary and repeated multiple times inside
#      each example, avoiding sparse one-off evidence.
# c) implicit_assumption_review:
#    - `_main_colors(...)` may still identify the two foreground colors, but it
#      can no longer imply that both should be transformed.
#    - `_dent_centers(...)` still detects dents for both hosts; the changed
#      assumption is only in `main(...)`, which must now split dents by active
#      host instead of stamping every detected center.
#    - Starting from `clean.copy()` would incorrectly heal inactive dents unless
#      they are explicitly restored to black after stamping active patches.
# d) placement_legality:
#    - Generator keeps dents globally separated across both colors so an active
#      3x3 patch cannot cover an inactive dent center.
#    - Active and inactive regions both receive several dents, and pair-expansion
#      is allowed only when it preserves the same global spacing constraints.
# e) visual_observability:
#    - The larger region is forced to exceed the smaller by a clear cell-count
#      margin, so humans can see which color is active from the examples.
#    - Both colors appear with multiple dents, making the one-way behavior easy
#      to compare.
# f) visual_cleanliness:
#    - Noise remains one-cell and is cleaned away first, so the decisive visual
#      difference is the presence of teal-marked opposite-color patches versus
#      untouched black holes.
# g) color_mapping_inference:
#    - Only two foreground colors participate, and the mapping is still
#      structured: active color dents map to the other color; inactive color
#      dents do not map at all. No hidden color-id ordering is used.


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
    return max(colors, key=lambda color: counts[int(color)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert dents from only the larger colored region into marked opposite-color patches.

    Input structure:
    - Black background with two large foreground colors and a noticeable size imbalance.
    - Both colored regions can contain black dents, plus a few one-cell specks
      or boundary protrusions that should be cleaned first.

    Rule:
    - First denoise the two foreground regions in the same local way as the
      original task: isolated specks become black and wrong-color protrusions
      near the boundary revert to the surrounding host region.
    - Determine which foreground color owns the larger cleaned area. Only dents
      attributed to that larger region produce a 3x3 patch in the other color.
    - Mark active dent centers with teal (8), but keep dents in the smaller
      region as black holes.

    Key insight: The asymmetry is no longer "every dent swaps to the other
    color"; instead, size decides which color is active, so only the dominant
    region's dents get the opposite-color teal-marked treatment.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    active = int(_active_host_color(clean, colors))
    inactive = int(colors[1] if active == int(colors[0]) else colors[0])

    out = clean.copy()
    active_centers: list[tuple[int, int]] = []
    inactive_centers: list[tuple[int, int]] = []
    for r, c, host in _dent_centers(g, clean, colors, bg):
        if int(host) == active:
            active_centers.append((r, c))
        else:
            inactive_centers.append((r, c))

    h, w = g.shape
    for r, c in active_centers:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = inactive

    for r, c in active_centers:
        out[r, c] = Color.TEAL

    for r, c in inactive_centers:
        out[r, c] = bg

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85
ACTIVE_AREA_GAP_MIN = 35
SOLVER_ACTIVE_AREA_GAP_MIN = 20

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
MIN_ACTIVE_DENTS = 6
MIN_INACTIVE_DENTS = 4
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300
CENTER_PICK_ATTEMPTS = 12


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
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


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
    """Select n dent centers with global Chebyshev separation."""
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


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked_centers: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand single-cell dents into 2-cell adjacent pairs."""
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
                if clean[nr, nc] != host or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any((nr, nc) == center for center in blocked_centers if center != (r, c)):
                    continue
                if not _far_enough(nr, nc, [center for center in blocked_centers if center != (r, c)]):
                    continue
                cells.append((nr, nc))
                break
        result.append(cells)
    return result


def _build_output(
    clean: np.ndarray,
    active_groups: list[list[tuple[int, int]]],
    inactive_groups: list[list[tuple[int, int]]],
    inactive_color: int,
) -> np.ndarray:
    """Build expected output for the one-way mapping variant."""
    h, w = clean.shape
    output = clean.copy()
    for cells in active_groups:
        for r, c in cells:
            output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = inactive_color
    for cells in active_groups:
        for r, c in cells:
            output[r, c] = Color.TEAL
    for cells in inactive_groups:
        for r, c in cells:
            output[r, c] = Color.BLACK
    return output


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    """Place isolated fg-color specks in background area."""
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
    """Check that changing (r,c) to `other` won't cause the solver to misclassify any neighbor."""
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or inp[nr, nc] != host:
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


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random) -> None:
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
        if inp[r, c] != host:
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

    counts = {color_a: int(np.sum(clean == color_a)), color_b: int(np.sum(clean == color_b))}
    if abs(counts[color_a] - counts[color_b]) < ACTIVE_AREA_GAP_MIN:
        return None
    active_color = max((color_a, color_b), key=lambda color: counts[color])
    inactive_color = color_b if active_color == color_a else color_a

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_active_min = max(MIN_ACTIVE_DENTS, n_total // 2)
    n_active_max = n_total - MIN_INACTIVE_DENTS
    if n_active_min > n_active_max:
        return None
    n_active = rng.randint(n_active_min, n_active_max)
    n_inactive = n_total - n_active

    h, w = clean.shape
    valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))

    if len(valid[active_color]) < n_active + 2 or len(valid[inactive_color]) < n_inactive + 2:
        return None

    dents_active = None
    dents_inactive = None
    for _ in range(CENTER_PICK_ATTEMPTS):
        dents_active = _pick_dent_centers(valid[active_color], n_active, rng)
        if dents_active is None:
            continue
        dents_inactive = _pick_dent_centers(valid[inactive_color], n_inactive, rng, blocked=dents_active)
        if dents_inactive is not None:
            break
    if dents_active is None or dents_inactive is None:
        return None

    blocked_centers = dents_active + dents_inactive
    active_groups = _expand_to_pairs(dents_active, active_color, inactive_color, clean, blocked_centers, rng)
    inactive_groups = _expand_to_pairs(dents_inactive, inactive_color, active_color, clean, blocked_centers, rng)

    output = _build_output(clean, active_groups, inactive_groups, inactive_color)

    inp = clean.copy()
    for cells in active_groups + inactive_groups:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    # The solver chooses the active region after dents are blackened and noise
    # is cleaned. Keep that visible ordering explicit instead of relying only on
    # the final oracle comparison.
    solver_clean = _clean_foreground(inp, (color_a, color_b), Color.BLACK)
    resolved_active = int(_active_host_color(solver_clean, (color_a, color_b)))
    resolved_inactive = int(color_b if resolved_active == color_a else color_a)
    if resolved_active != int(active_color):
        return None
    if abs(int(np.sum(solver_clean == resolved_active)) - int(np.sum(solver_clean == resolved_inactive))) < SOLVER_ACTIVE_AREA_GAP_MIN:
        return None
    solver_centers = _dent_centers(inp, solver_clean, (color_a, color_b), Color.BLACK)
    resolved_active_count = sum(int(host == resolved_active) for _r, _c, host in solver_centers)
    resolved_inactive_count = sum(int(host == resolved_inactive) for _r, _c, host in solver_centers)
    if resolved_active_count < MIN_ACTIVE_DENTS or resolved_inactive_count < MIN_INACTIVE_DENTS:
        return None

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
