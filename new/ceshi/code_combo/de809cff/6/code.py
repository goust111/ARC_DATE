# Combo variant of de809cff (index 6)
# combo_spec: D5/2+D9/0
# Name: 同心嵌套区域+补丁裁剪到区域边界

from __future__ import annotations

# Design analysis for the D5/2 + D9/0 combo
# -------------------------------------------
# a) Concrete behavior:
#    The original task has two large foreground regions side by side, and every
#    detected black dent writes an unclipped 3x3 opposite-color square. In this
#    combo, the clean scene is a nested layout: an outer foreground ring fully
#    surrounds a smaller inner foreground island. Example around an inner-region
#    boundary dent (`i` = inner, `o` = outer, `.` = black dent):
#      o o o o o         o o o o o
#      o i i i o         o o o o o
#      o i . i o   ->    o o 8 o o
#      o i i i o         o o o o o
#      o o o o o         o o o o o
#    Under the base rule the whole 3x3 box would flip to the outer color anyway;
#    under the new rule only cells that belong to the dent's host region are
#    recolored. For inner dents this creates strongly truncated patches; for outer
#    dents near the inner boundary the patch is clipped from the other side.
# b) Complexity calibration:
#    The nested geometry adds one new relation only: "inside the outer region"
#    versus "at the interface". The generator keeps both interior and boundary
#    dents in each region, so solvers repeatedly see full 3x3 patches and clipped
#    irregular patches in the same sample. That is enough evidence for the combo
#    without introducing many unrelated layout families.
# c) Hidden assumptions reviewed:
#    - The original write-back `out[r-1:r+2, c-1:c+2] = other` assumes every
#      in-bounds cell of the 3x3 box should flip; D9/0 breaks that.
#    - Using the cleaned grid directly as a clipping mask would leave dent centers
#      black and would incorrectly punch holes in paired dents. The solver must
#      reconstruct the host regions by filling every detected dent center back
#      with its host color before clipping.
#    - The original generator assumes side-by-side rectangles. D5/2 breaks that:
#      there must now be an explicit containment relation, and outer-region dents
#      near the inner interface must remain valid under the same local majority
#      test used by the solver.
# d) Placement legality:
#    The generator builds one large outer rectangle and one smaller inner
#    rectangle strictly inside it, with margins on all sides so the outer region
#    stays a connected ring and the inner region stays fully enclosed. Dent
#    centers are selected with Chebyshev spacing, and the mandatory clipped-dent
#    quotas are checked before noise is added.
# e) Visual observability:
#    Samples always contain several inner-boundary dents and several outer dents
#    touching the inner island, so the clipped-patch rule is visible repeatedly.
#    Interior dents remain present as controls: humans can compare square patches
#    against clipped wedges/L-shapes in the same color system.
# f) Visual cleanliness:
#    The two regions use only two foreground colors plus teal dent centers, and
#    dent centers are spaced apart globally so clipped patches do not collapse
#    into unreadable blobs along the interface.
# g) Color-rule inferability:
#    Color semantics stay simple: each dent generates the opposite foreground
#    color and the center becomes teal. The only new inference burden is spatial:
#    the patch is limited to the dent's own enclosing region.

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
    """Fill dent centers back with their host colors to recover the clean region mask."""
    region_mask = clean.copy()
    for r, c, host in centers:
        region_mask[r, c] = int(host)
    return region_mask


def _paint_clipped_patch(
    out: np.ndarray, region_mask: np.ndarray, r: int, c: int, host: int, stamp_color: int
) -> None:
    h, w = out.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(region_mask[rr, cc]) == int(host):
                out[rr, cc] = int(stamp_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in nested colored regions into clipped opposite-color patches.

    Input structure:
    - Black background with two large foreground regions in a containment layout:
      one outer region fully encloses a smaller inner region.
    - Both regions may contain black dents plus a few isolated color specks or
      one-cell boundary contaminations that should be cleaned first.

    Rule:
    - Locally clean the two foreground colors exactly as in the base task, so
      isolated specks disappear and one-cell cross-boundary contamination reverts
      to the surrounding region.
    - Each original black dent whose cleaned 3x3 neighborhood is dominated by one
      foreground color belongs to that host region and proposes a 3x3 patch in the
      opposite foreground color.
    - Keep only the proposed patch cells that lie inside the dent's own host
      region; any part crossing into background or the other nested region stays
      unchanged, and the dent center itself becomes teal (8).

    Key insight: After cleaning, reconstruct which cells belong to each nested
    region, then clip every opposite-color 3x3 dent patch to that recovered host
    mask before marking the dent center.
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
        _paint_clipped_patch(out, region_mask, r, c, int(host), int(other))

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & nested region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
OUTER_FRAC_MIN, OUTER_FRAC_MAX = 0.72, 0.88
INNER_FRAC_MIN, INNER_FRAC_MAX = 0.40, 0.56
OUTER_BORDER_MIN, OUTER_BORDER_MAX = 1, 3
INNER_MARGIN_MIN = 3
INNER_CENTER_JITTER = 1
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.10
MIN_INNER_CLIPPED_DENTS = 3
MIN_OUTER_INTERFACE_DENTS = 3
MIN_INTERIOR_DENTS_PER_REGION = 2
MIN_TOTAL_CLIP_LOSS = 14
SELECT_RETRY_ATTEMPTS = 160

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """Return the 3x3 neighborhood of (r,c), clipped to grid bounds."""
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _balanced_split(total: int, min_each: int, rng: random.Random) -> tuple[int, int] | None:
    if total < 2 * min_each:
        return None
    lo = max(min_each, total // 2 - INNER_CENTER_JITTER)
    hi = min(total - min_each, total // 2 + INNER_CENTER_JITTER)
    if lo > hi:
        return None
    first = rng.randint(lo, hi)
    return int(first), int(total - first)


def _build_nested_regions(N: int, outer_color: int, inner_color: int, rng: random.Random) -> np.ndarray | None:
    """Build a large outer rectangle containing a smaller inner rectangle."""
    outer_h_lo = max(16, int(round(N * OUTER_FRAC_MIN)))
    outer_h_hi = min(N - 2 * OUTER_BORDER_MIN, int(round(N * OUTER_FRAC_MAX)))
    outer_w_lo = max(16, int(round(N * OUTER_FRAC_MIN)))
    outer_w_hi = min(N - 2 * OUTER_BORDER_MIN, int(round(N * OUTER_FRAC_MAX)))
    if outer_h_lo > outer_h_hi or outer_w_lo > outer_w_hi:
        return None

    outer_h = rng.randint(outer_h_lo, outer_h_hi)
    outer_w = rng.randint(outer_w_lo, outer_w_hi)
    r0_lo = OUTER_BORDER_MIN
    r0_hi = min(OUTER_BORDER_MAX, N - outer_h - OUTER_BORDER_MIN)
    c0_lo = OUTER_BORDER_MIN
    c0_hi = min(OUTER_BORDER_MAX, N - outer_w - OUTER_BORDER_MIN)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    outer_r0 = rng.randint(r0_lo, r0_hi)
    outer_c0 = rng.randint(c0_lo, c0_hi)
    outer_r1 = outer_r0 + outer_h
    outer_c1 = outer_c0 + outer_w

    inner_h_lo = max(10, int(round(N * INNER_FRAC_MIN)))
    inner_h_hi = min(outer_h - 2 * INNER_MARGIN_MIN, int(round(N * INNER_FRAC_MAX)))
    inner_w_lo = max(10, int(round(N * INNER_FRAC_MIN)))
    inner_w_hi = min(outer_w - 2 * INNER_MARGIN_MIN, int(round(N * INNER_FRAC_MAX)))
    if inner_h_lo > inner_h_hi or inner_w_lo > inner_w_hi:
        return None

    inner_h = rng.randint(inner_h_lo, inner_h_hi)
    inner_w = rng.randint(inner_w_lo, inner_w_hi)
    vertical = _balanced_split(outer_h - inner_h, INNER_MARGIN_MIN, rng)
    horizontal = _balanced_split(outer_w - inner_w, INNER_MARGIN_MIN, rng)
    if vertical is None or horizontal is None:
        return None
    margin_top, margin_bottom = vertical
    margin_left, margin_right = horizontal

    inner_r0 = outer_r0 + margin_top
    inner_c0 = outer_c0 + margin_left
    inner_r1 = outer_r1 - margin_bottom
    inner_c1 = outer_c1 - margin_right

    grid = np.zeros((N, N), dtype=int)
    grid[outer_r0:outer_r1, outer_c0:outer_c1] = int(outer_color)
    grid[inner_r0:inner_r1, inner_c0:inner_c1] = int(inner_color)

    outer_cells = int(np.sum(grid == int(outer_color)))
    inner_cells = int(np.sum(grid == int(inner_color)))
    coverage = (outer_cells + inner_cells) / float(N * N)
    if outer_cells < MIN_REGION_CELLS or inner_cells < MIN_REGION_CELLS:
        return None
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _clip_loss(clean: np.ndarray, r: int, c: int, host: int) -> int:
    sub = _sub3x3(clean, r, c)
    return int(sub.size - np.sum(sub == int(host)))


def _touches_color(clean: np.ndarray, r: int, c: int, color: int) -> bool:
    return bool(np.any(_sub3x3(clean, r, c) == int(color)))


def _spaced_enough(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = candidate
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    *,
    blocked: list[tuple[int, int]] | None = None,
    preferred_groups: list[tuple[list[tuple[int, int]], int]] | None = None,
) -> list[tuple[int, int]] | None:
    blocked = [] if blocked is None else list(blocked)
    preferred_groups = [] if preferred_groups is None else list(preferred_groups)
    candidate_set = set(candidates)
    filtered_groups = [([p for p in group if p in candidate_set], int(target)) for group, target in preferred_groups]
    if sum(target for _group, target in filtered_groups) > n:
        return None

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        ok = True

        for group, target in filtered_groups:
            order = list(group)
            rng.shuffle(order)
            picked = 0
            for candidate in order:
                if candidate in used:
                    continue
                if _spaced_enough(candidate, blocked + chosen):
                    chosen.append(candidate)
                    used.add(candidate)
                    picked += 1
                    if picked >= target:
                        break
            if picked < target:
                ok = False
                break
        if not ok:
            continue

        remaining = [candidate for candidate in candidates if candidate not in used]
        rng.shuffle(remaining)
        for candidate in remaining:
            if _spaced_enough(candidate, blocked + chosen):
                chosen.append(candidate)
                if len(chosen) >= n:
                    return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_primary_dents: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand some single-cell dents to 2-cell orthogonal pairs."""
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
                if all(
                    max(abs(nr - rr), abs(nc - cc)) >= DENT_PATCH_SEPARATION
                    for rr, cc in all_primary_dents
                    if (rr, cc) != (r, c)
                ):
                    cells.append((nr, nc))
                    break
        result.append(cells)
    return result


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_outer: list[list[tuple[int, int]]],
    dent_groups_inner: list[list[tuple[int, int]]],
    outer_color: int,
    inner_color: int,
    *,
    clipped: bool,
) -> np.ndarray:
    """Build output with either clipped or legacy unclipped opposite-color patches."""
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, host, stamp_color in (
        (dent_groups_outer, outer_color, inner_color),
        (dent_groups_inner, inner_color, outer_color),
    ):
        for cells in dent_groups:
            for r, c in cells:
                if clipped:
                    for rr in range(max(0, r - 1), min(h, r + 2)):
                        for cc in range(max(0, c - 1), min(w, c + 2)):
                            if int(clean[rr, cc]) == int(host):
                                output[rr, cc] = int(stamp_color)
                else:
                    output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(
                        stamp_color
                    )
    for dent_groups in (dent_groups_outer, dent_groups_inner):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _clip_metrics(clean: np.ndarray, dent_groups: list[list[tuple[int, int]]], host: int) -> tuple[int, int]:
    """Return (number of clipped dent cells, total patch cells prevented by clipping)."""
    clipped_count = 0
    total_loss = 0
    for cells in dent_groups:
        for r, c in cells:
            loss = _clip_loss(clean, r, c, host)
            if loss > 0:
                clipped_count += 1
                total_loss += loss
    return clipped_count, total_loss


def _interface_clipped_count(
    clean: np.ndarray, dent_groups: list[list[tuple[int, int]]], host: int, other: int
) -> int:
    count = 0
    for cells in dent_groups:
        for r, c in cells:
            if _clip_loss(clean, r, c, host) > 0 and _touches_color(clean, r, c, other):
                count += 1
    return count


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, outer_color: int, inner_color: int, rng: random.Random) -> None:
    """Place isolated foreground specks in background."""
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if int(np.sum(_sub3x3(inp, r, c) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = (
            _Counter(nearby_fg).most_common(1)[0][0]
            if nearby_fg
            else rng.choice([outer_color, inner_color])
        )
        inp[r, c] = int(speck_color)
        placed += 1


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


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` won't break local cleanup assumptions nearby."""
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


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, outer_color: int, inner_color: int, rng: random.Random) -> None:
    """Place cross-color contamination along the outer/inner interface."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (outer_color, inner_color):
                continue
            other = inner_color if v == outer_color else outer_color
            if _touches_color(clean, r, c, other):
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
    fg_pool = [color for color in Color.NOT_BLACK if color != Color.TEAL]
    outer_color, inner_color = rng.sample(fg_pool, 2)

    clean = _build_nested_regions(N, outer_color, inner_color, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_inner = n_total // 2
    n_outer = n_total - n_inner
    if rng.random() < 0.5:
        n_inner, n_outer = n_outer, n_inner
    n_inner = max(n_inner, MIN_INNER_CLIPPED_DENTS + MIN_INTERIOR_DENTS_PER_REGION)
    n_outer = max(n_outer, MIN_OUTER_INTERFACE_DENTS + MIN_INTERIOR_DENTS_PER_REGION)
    if n_inner + n_outer > n_total:
        return None
    if n_inner + n_outer < n_total:
        n_outer += n_total - (n_inner + n_outer)

    h, w = clean.shape
    valid_outer: list[tuple[int, int]] = []
    outer_interface: list[tuple[int, int]] = []
    outer_interior: list[tuple[int, int]] = []
    valid_inner: list[tuple[int, int]] = []
    inner_clipped: list[tuple[int, int]] = []
    inner_interior: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(outer_color) and _is_safe_dent_center(clean, r, c, outer_color, inner_color):
                valid_outer.append((r, c))
                loss = _clip_loss(clean, r, c, outer_color)
                if loss == 0:
                    outer_interior.append((r, c))
                elif _touches_color(clean, r, c, inner_color):
                    outer_interface.append((r, c))
            elif v == int(inner_color) and _is_safe_dent_center(clean, r, c, inner_color, outer_color):
                valid_inner.append((r, c))
                if _clip_loss(clean, r, c, inner_color) == 0:
                    inner_interior.append((r, c))
                else:
                    inner_clipped.append((r, c))

    if len(valid_outer) < n_outer + 2 or len(valid_inner) < n_inner + 2:
        return None
    if len(outer_interface) < MIN_OUTER_INTERFACE_DENTS or len(inner_clipped) < MIN_INNER_CLIPPED_DENTS:
        return None
    if len(outer_interior) < MIN_INTERIOR_DENTS_PER_REGION or len(inner_interior) < MIN_INTERIOR_DENTS_PER_REGION:
        return None

    dents_inner = _pick_dent_centers(
        valid_inner,
        n_inner,
        rng,
        preferred_groups=[
            (inner_clipped, MIN_INNER_CLIPPED_DENTS),
            (inner_interior, MIN_INTERIOR_DENTS_PER_REGION),
        ],
    )
    if dents_inner is None:
        return None

    dents_outer = _pick_dent_centers(
        valid_outer,
        n_outer,
        rng,
        blocked=dents_inner,
        preferred_groups=[
            (outer_interface, MIN_OUTER_INTERFACE_DENTS),
            (outer_interior, MIN_INTERIOR_DENTS_PER_REGION),
        ],
    )
    if dents_outer is None:
        return None

    all_primary_dents = dents_outer + dents_inner
    groups_outer = _expand_to_pairs(dents_outer, outer_color, inner_color, clean, all_primary_dents, rng)
    groups_inner = _expand_to_pairs(dents_inner, inner_color, outer_color, clean, all_primary_dents, rng)

    outer_interface_count = _interface_clipped_count(clean, groups_outer, outer_color, inner_color)
    inner_clipped_count, inner_clip_loss = _clip_metrics(clean, groups_inner, inner_color)
    outer_clipped_count, outer_clip_loss = _clip_metrics(clean, groups_outer, outer_color)
    if outer_interface_count < MIN_OUTER_INTERFACE_DENTS or inner_clipped_count < MIN_INNER_CLIPPED_DENTS:
        return None
    if outer_clipped_count < MIN_OUTER_INTERFACE_DENTS:
        return None
    if inner_clip_loss + outer_clip_loss < MIN_TOTAL_CLIP_LOSS:
        return None

    output = _stamp_dents(clean, groups_outer, groups_inner, outer_color, inner_color, clipped=True)
    legacy_output = _stamp_dents(clean, groups_outer, groups_inner, outer_color, inner_color, clipped=False)
    if np.array_equal(output, legacy_output):
        return None

    inp = clean.copy()
    for dent_groups in (groups_outer, groups_inner):
        for cells in dent_groups:
            for r, c in cells:
                inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, outer_color, inner_color, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, outer_color, inner_color, rng)

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
