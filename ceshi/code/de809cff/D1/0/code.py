# Variant of de809cff
# Dimension: D1 — 核心变换规则：凹痕到3x3补丁 (transform)
# Change: 每个黑色凹痕的3x3邻域多数前景色为A时，凹痕变为3x3的前景色B补丁，中心为teal(8)... → 凹痕变为对侧色十字形而非3x3方块

from __future__ import annotations

# Design analysis for the D1 variant
# ----------------------------------
# source_task:
#   none
# variant_spec:
#   preserved:
#     - still exactly two large foreground colors on black
#     - still clean isolated foreground specks and single-cell boundary contamination first
#     - still identify each dent by asking whether the cleaned 3x3 around an original black
#       cell has a strict majority of one foreground color with at least 5 host cells
#     - still recolor the dent center teal
#   changed:
#     - the opposite-color mark is no longer the whole 3x3 box
#     - only the orthogonal cross is painted: center + up/down/left/right
#     - after painting that cross, the center is overwritten to teal, so the visible mark is
#       a 5-cell cross with teal center and four opposite-color arms
# concrete_behavior:
#   - Original local behavior at one dent:
#       clean 3x3 majority says the hole belongs to region A
#       => output paints all 9 cells in the clipped 3x3 to color B, then center becomes teal
#   - Variant local behavior at one dent:
#       same host decision
#       => output paints only the 4-neighbors of the hole to color B, and the hole itself teal
#   - Difference:
#       diagonals that were recolored in the original task now stay whatever the cleaned grid had
#       there (host color, opposite color, or background at outer edges)
# complexity_calibration:
#   - The rule change is visually obvious because every interior dent becomes a plus sign rather
#     than a solid 3x3 square.
#   - The generator keeps many dents and keeps most of them away from the outer border so several
#     full crosses are visible in each sample; some clipped edge cases remain for realism.
#   - Crosses are still sparse and separable because seed centers use the same conservative
#     Chebyshev spacing as the original 3x3 patches, and spacing is enforced across both colors.
# implicit_assumption_audit:
#   - `_main_colors` must still recover exactly two foreground colors, but tie handling should not
#     depend on color id. This variant keeps first-appearance order for equal counts.
#   - `_clean_foreground` still assumes noise is only 1-cell specks/protrusions; generator keeps
#     that contract unchanged.
#   - `_dent_centers` still assumes dents are original black cells and host is read from the
#     cleaned 3x3 majority; unchanged and still valid.
#   - The original `out[r-1:r+2, c-1:c+2] = other` relied on "the patch is the whole square".
#     That assumption is invalid now, so the write is replaced by bounded writes only on the
#     orthogonal cross cells.
#   - The final teal overwrite remains necessary and safe: even if another cross arm touches a
#     dent center, the second pass restores the teal marker.
# placement_legality:
#   - Seed dent centers are globally spaced so crosses from opposite host colors do not collapse
#     into ambiguous overlaps.
#   - Optional 2-cell dent pairs are still allowed, but the added partner is checked against all
#     already-reserved dent centers except its own anchor.
#   - The oracle remains output-first: build intended output, derive input by blackening dent
#     cells, add bounded noise, then require `main(input) == output`.
# observability_and_tidiness:
#   - The decisive metric is the number of cells where square-stamping and cross-stamping differ.
#     For a genuine D1 sample this should be positive, and usually large because every interior
#     dent suppresses four diagonal recolors.
#   - Requiring several non-edge dents avoids a degenerate sample set where every cross is clipped
#     and the shape cue becomes weak.

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


def _cross_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into marked opposite-color crosses.

    Input structure:
    - Black background with two broad foreground regions in different colors.
    - The regions contain black dents plus a small amount of one-cell foreground
      noise in the background or along the boundary between the two regions.

    Rule:
    - First locally clean the two foreground regions so isolated specks revert
      to black and one-cell boundary contamination flips back to its host color.
    - For each original black dent whose cleaned 3x3 neighborhood is mostly one
      foreground color, paint the opposite color only on the orthogonal cross
      centered at that dent, not on the diagonal cells.
    - Recolor the dent center itself to teal (8).

    Key insight: The black defects are still found from the cleaned 3x3 majority,
    but the output signature is now a sparse opposite-color plus sign with a teal
    center rather than a filled 3x3 opposite-color block.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for rr, cc in _cross_cells(g.shape, r, c):
            out[rr, cc] = int(other)

    for r, c, _host in centers:
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
DENT_PAIR_PROB = 0.13
MIN_FULL_CROSS_CENTERS = 5

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
) -> list[tuple[int, int]] | None:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    blocked = list(reserved)
    for r, c in shuffled:
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
                if clean[nr, nc] != host or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any(
                    (r0, c0) != (r, c)
                    and max(abs(nr - r0), abs(nc - c0)) < DENT_PATCH_SEPARATION
                    for r0, c0 in reserved
                ):
                    continue
                cells.append((nr, nc))
                reserved.add((nr, nc))
                break
        result.append(cells)
    return result


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                for rr, cc in _cross_cells(clean.shape, r, c):
                    output[rr, cc] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _stamp_square_baseline(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    output = clean.copy()
    h, w = clean.shape
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
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
        speck_color = (
            _Counter(nearby_fg).most_common(1)[0][0]
            if nearby_fg
            else rng.choice([color_a, color_b])
        )
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
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
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
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


def _count_full_cross_centers(shape: tuple[int, int], groups: list[list[tuple[int, int]]]) -> int:
    h, w = map(int, shape)
    total = 0
    for cells in groups:
        for r, c in cells:
            if 0 < r < h - 1 and 0 < c < w - 1:
                total += 1
    return total


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

    dents: dict[int, list[tuple[int, int]]] = {}
    reserved: list[tuple[int, int]] = []
    order = [color_a, color_b]
    rng.shuffle(order)
    targets = {color_a: n_a, color_b: n_b}
    for color in order:
        chosen = _pick_dent_centers(valid[color], targets[color], reserved, rng)
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

    if _count_full_cross_centers(clean.shape, groups_a + groups_b) < min(MIN_FULL_CROSS_CENTERS, n_total):
        return None

    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b)
    square_baseline = _stamp_square_baseline(clean, groups_a, groups_b, color_a, color_b)
    if int(np.sum(output != square_baseline)) <= 0:
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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
