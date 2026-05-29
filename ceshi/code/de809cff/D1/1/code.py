# Variant of de809cff
# Dimension: D1 — 核心变换规则：凹痕到3x3补丁 (transform)
# Change: 每个黑色凹痕的3x3邻域多数前景色为A时，凹痕变为3x3的前景色B补丁，中心为teal(8)... → 凹痕变为同侧前景色补丁（非对侧）

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the D1 inversion variant
# --------------------------------------------
# a) Concrete behavior trace
# - The input still contains two large non-black regions on a black background,
#   with black dents and one-cell noise. The solver still has to clean the
#   regions first, then identify every original black cell whose cleaned 3x3
#   neighborhood clearly belongs to one host color.
# - Original behavior at one dent:
#     clean 3x3 majority = red -> paint the whole 3x3 yellow -> recolor center teal.
# - Variant behavior at the same dent:
#     clean 3x3 majority = red -> paint the whole 3x3 red -> recolor center teal.
# - The difference matters most when the 3x3 touches background or the other
#   region. In that case the original task "plants" a foreign-color block,
#   while the variant "repairs" the host region by expanding its own color into
#   nearby non-host cells. Interior dents remain mostly teal-center markers.
#
# b) Complexity calibration
# - If dents were mostly deep inside solid rectangles, the new 3x3 repaint would
#   often be visually indistinguishable from simply restoring the center before
#   marking it teal. That would weaken the D1 signal.
# - To keep the variant observable without changing the task family, the
#   generator requires multiple visible edge-adjacent dents overall, while still
#   keeping many interior dents. This preserves repeated evidence for
#   "same-side repair" in 2-4 ARC examples without over-constraining layout
#   sampling.
#
# c) Main() line-by-line assumption review
# - _main_colors assumes exactly two dominant foreground colors after excluding
#   black; that still holds because the input palette is unchanged.
# - _clean_foreground assumes all non-main colors are noise and should collapse
#   to black or the local majority. That remains valid because the variant only
#   changes the final dent repaint, not the corruption model.
# - _dent_centers assumes a dent can be assigned to the local majority color if
#   that color has at least 5 cells in the cleaned 3x3 and strictly beats the
#   other color. This still defines the host region correctly for the variant.
# - The original main() relied on the extra step "host -> other" when stamping
#   the 3x3 patch. That is the only old-property dependency that must change.
#   In the variant, stamping must use host directly; otherwise the solver would
#   silently revert to the old rule.
#
# d) Placement legality
# - Dent centers are only chosen from host-colored cells whose 3x3 neighborhood
#   still has at least 5 host cells after removing the center. This guarantees
#   the solver can recover the host color from the input.
# - Centers keep Chebyshev distance >= 3 from other chosen centers of the same
#   color, preventing accidental overlap between unrelated 3x3 patches.
# - Optional 2-cell dents are added only when the second cell is also a valid
#   dent center and does not crowd other same-color dents.
#
# e) Visual observability
# - The generator classifies dent candidates as "visible-patch" when their
#   clean 3x3 already contains at least one non-host cell. These are the cases
#   where same-color stamping visibly extends the region.
# - Each generated grid must contain multiple such dents after final cleanup.
#   This avoids a degenerate dataset where the variant is technically correct
#   but hard to infer by eye.
#
# f) Visual cleanliness
# - Noise remains one-cell only, and dents are spaced apart. The output shows
#   large clean regions plus teal markers, with only local 3x3 repairs around
#   dents. That keeps the image readable even when some repairs clip over region
#   boundaries or into background.
#
# g) Color inference
# - No hidden color ordering is introduced. The host color comes purely from the
#   visible majority in the cleaned 3x3. The only special output color remains
#   teal at dent centers, matching the original task family.
#
# h) New implicit assumptions after the change
# - main() now assumes that at least some dents have non-host cells in their
#   surrounding 3x3 if the dataset is meant to advertise the "same-side patch"
#   rule visually. generate_input() enforces that by validating the final count
#   of visible same-side repairs after noise and cleanup.
# - main() also assumes the cleaned grid is the intended pre-dent canvas; the
#   generator therefore continues using output-first construction plus an oracle
#   equality check so every produced input is solvable by the implemented logic.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair black dents in two colored regions and mark them with teal centers.

    Input structure:
    - Black background with two broad foreground regions in two non-black colors.
    - The regions contain black dents plus a small amount of one-cell foreground
      noise in background space or along the border between the two regions.

    Rule:
    - First denoise the foreground locally: isolated color specks disappear into
      black, and one-cell cross-boundary contamination flips back to the region
      that dominates its local neighborhood.
    - Then inspect each original black cell. If the cleaned 3x3 neighborhood is
      mostly one foreground color, repaint that whole 3x3 with the SAME host
      color rather than the opposite one.
    - Finally recolor the dent center itself to teal (8).

    Key insight: teal still marks where the original black defects were, but the
    surrounding 3x3 now repairs the damaged region with its own color instead of
    inserting a contrasting block from the other region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    centers = _dent_centers(g, clean, colors, bg)
    return _render_centers(clean, centers)


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
DENT_PAIR_PROB = 0.13
VISIBLE_DENTS_TOTAL_MIN = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


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
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _is_visible_patch_center(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    """True when same-color 3x3 stamping changes more than the center cell."""
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    non_host = sub.size - host_count
    return 1 <= non_host <= 2 and host_count >= 6 and _has_same_2x2(grid, r, c, host)


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


def _pick_dent_centers(candidates, n, rng):
    rng.shuffle(candidates)
    chosen = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _pick_dent_centers_with_visible(visible, hidden, n: int, min_visible: int, rng: random.Random):
    """Pick n same-color dent centers while reserving enough visually revealing ones."""
    visible = list(visible)
    hidden = list(hidden)
    rng.shuffle(visible)
    rng.shuffle(hidden)

    chosen: list[tuple[int, int]] = []

    def _try_add(pool) -> None:
        for r, c in pool:
            if len(chosen) >= n:
                return
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
                chosen.append((r, c))

    _try_add(visible)
    if len(chosen) < min_visible:
        return None

    remaining = [pos for pos in visible if pos not in chosen] + hidden
    rng.shuffle(remaining)
    _try_add(remaining)
    if len(chosen) != n:
        return None
    return chosen


def _dent_remains_detectable(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check the main() dent predicate on an already-blackened working grid."""
    if int(grid[r, c]) != Color.BLACK:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _expand_to_pairs(dents, host, other, clean, all_dents, rng):
    h, w = clean.shape
    result = []
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
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in all_dents
                    if (r2, c2) != (r, c)
                ):
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


def _stamp_dents(clean, dent_groups_a, dent_groups_b, color_a, color_b):
    """Build output by stamping 3x3 host-color patches and marking teal centers."""
    dent_cells = []
    for dent_groups, stamp_color in ((dent_groups_a, color_a), (dent_groups_b, color_b)):
        for cells in dent_groups:
            for r, c in cells:
                dent_cells.append((r, c, stamp_color))
    return _render_centers(clean, sorted(dent_cells))


def _render_centers(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host in centers:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(host)
    for r, c, _host in centers:
        output[r, c] = Color.TEAL
    return output


def _center_has_visible_extension(clean: np.ndarray, r: int, c: int, host: int) -> bool:
    """After cleanup, would same-color stamping change any neighbor besides the center?"""
    sub = _sub3x3(clean, r, c).copy()
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    sub[r - r0, c - c0] = host
    return bool(np.any(sub != int(host)))


def _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng):
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


def _neighbor_cleanup_safe(inp, r, c, host, other):
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


def _add_boundary_noise(inp, clean, n_at_boundary, color_a, color_b, rng):
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

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    valid_visible = {color_a: [], color_b: []}
    valid_hidden = {color_a: [], color_b: []}
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_a) else valid_hidden
                target[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_b) else valid_hidden
                target[color_b].append((r, c))

    if len(valid_visible[color_a]) + len(valid_hidden[color_a]) < n_a:
        return None
    if len(valid_visible[color_b]) + len(valid_hidden[color_b]) < n_b:
        return None

    dents_a = _pick_dent_centers_with_visible(
        valid_visible[color_a], valid_hidden[color_a], n_a, 0, rng
    )
    dents_b = _pick_dent_centers_with_visible(
        valid_visible[color_b], valid_hidden[color_b], n_b, 0, rng
    )
    if dents_a is None or dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_b, rng)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        cleaned_input = _clean_foreground(inp, (color_a, color_b), Color.BLACK)
        centers = _dent_centers(inp, cleaned_input, (color_a, color_b), Color.BLACK)
        output = _render_centers(cleaned_input, centers)
        produced = main(inp)
    except Exception:
        return None
    if len(centers) < DENT_COUNT_MIN:
        return None
    if sum(_center_has_visible_extension(cleaned_input, r, c, host) for r, c, host in centers) < VISIBLE_DENTS_TOTAL_MIN:
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
