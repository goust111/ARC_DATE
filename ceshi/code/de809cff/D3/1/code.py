# Variant of de809cff
# Dimension: D3 — 凹痕识别条件：3x3邻域多数色判定 (transform)
# Change: 凹痕的（清洁后）3x3邻域中多数前景色为某一色时，该凹痕被认定为属于该色区域并触发变换... → 要求邻域全部为同一前景色（非多数）

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the D3 variant:
# - Concrete behavior:
#   The original task transformed a black dent when one foreground color merely
#   dominated its cleaned 3x3 neighborhood. Here the trigger is stricter:
#   a black cell is transformed only when all non-black cells in that cleaned
#   3x3 belong to one foreground color, and there are still at least 5 such
#   host cells. Mixed neighborhoods with both foreground colors stay black.
#   Small trace:
#     old: [a a a; a 0 b; a b b] -> host=a by 5>3, so it transformed.
#     new: same neighborhood contains both a and b, so it does not transform.
# - Complexity calibration:
#   To make the difference visible, the generator creates both:
#   1) deep interior dents whose 3x3 is uniformly one color and do transform;
#   2) boundary-adjacent dents whose 3x3 contains both foreground colors and
#      therefore remain black in the output.
#   This avoids a degenerate dataset where every dent is still transformable.
# - Implicit assumptions audit:
#   `_main_colors` still assumes exactly two dominant foreground colors; the
#   generator keeps that true. `_clean_foreground` still assumes noise is local
#   one-cell corruption; D3 does not alter that stage, so it stays unchanged.
#   The old `_dent_centers` relied on `max(counts)` and majority thresholds;
#   that would mis-handle mixed dents in this variant, so it is replaced with a
#   uniform-neighborhood check. The stamping order remains safe because dent
#   centers are selected with Chebyshev separation >= 3, except optional strict
#   pairs which are constructed so each member still satisfies the new rule.
# - Placement legality:
#   Strict dent centers are sampled only from full 3x3 host interiors, so after
#   blackening they still have at least 7 host neighbors of one color. Mixed
#   dents are sampled only where both foreground colors occur in the clean 3x3
#   and no background is present, which guarantees they fail the new rule while
#   still satisfying the old majority-style condition.
# - Visual observability:
#   The visible contrast is: some dents become opposite-color 3x3 patches with
#   teal centers, while some equally plausible black dents near the color-color
#   interface remain untouched. The condition is structural and local; it does
#   not depend on color ids or hidden ordering.
# - Visual cleanliness / color inference:
#   Only two foreground colors plus teal are used. Mixed dents are spaced away
#   from strict dents, so the untouched black holes do not get overwritten by
#   nearby 3x3 stamps. That keeps the before/after relation readable.

MIN_UNIFORM_HOST_3X3 = 5


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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

                sub = _sub3x3(cur, r, c)
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


def _uniform_host(clean: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    present = [int(color) for color in colors if counts[int(color)] > 0]
    if len(present) != 1:
        return None
    host = int(present[0])
    if counts[host] < MIN_UNIFORM_HOST_3X3:
        return None
    return host


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _uniform_host(clean, rr, cc, colors)
        if host is not None:
            centers.append((rr, cc, host))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn only fully enclosed black dents into marked opposite-color patches.

    Input structure:
    - Black background with two large foreground-color regions.
    - The regions contain black dents plus a small amount of one-cell noise in
      the background or near the color-color boundary.

    Rule:
    - First locally clean the two foreground regions, removing isolated specks
      and reverting one-cell boundary contamination.
    - Then inspect each original black cell on the cleaned grid: transform it
      only when every non-black cell in its clipped 3x3 neighborhood belongs to
      one foreground color and that color occupies at least 5 cells there.
    - For each such dent, paint its clipped 3x3 neighborhood with the opposite
      foreground color and recolor the dent center itself to teal (8).

    Key insight: black holes are no longer triggered by a mere local majority;
    only dents whose cleaned neighborhood is uniformly embedded in one region
    get the opposite-color 3x3 stamp, while mixed-neighborhood holes stay black.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

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
MIXED_DENT_COUNT_MIN, MIXED_DENT_COUNT_MAX = 3, 6
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3X3 = 6
BOUNDARY_MIN_HOST_3X3 = 6

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


def _is_relaxed_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    return host_count >= MIN_UNIFORM_HOST_3X3 and host_count > other_count


def _is_deep_strict_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if not _is_relaxed_dent_center(grid, r, c, host, other):
        return False
    sub = _sub3x3(grid, r, c)
    return bool(np.all(sub == host))


def _is_mixed_majority_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if not _is_relaxed_dent_center(grid, r, c, host, other):
        return False
    sub = _sub3x3(grid, r, c)
    other_count = int(np.sum(sub == other))
    bg_count = int(np.sum(sub == Color.BLACK))
    return other_count >= 1 and bg_count == 0


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


def _pick_with_blockers(
    candidates: list[tuple[int, int]],
    n: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in blocked + chosen):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_strict_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    occupied: list[tuple[int, int]],
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
                if not _is_deep_strict_center(clean, nr, nc, host, other):
                    continue
                if any((nr, nc) == (rr, cc) for rr, cc in occupied):
                    continue
                if all(
                    max(abs(nr - rr), abs(nc - cc)) >= DENT_PATCH_SEPARATION
                    for rr, cc in occupied
                    if (rr, cc) != (r, c)
                ):
                    cells.append((nr, nc))
                    occupied.append((nr, nc))
                    break
        result.append(cells)
    return result


def _stamp_output(
    clean: np.ndarray,
    strict_groups_a: list[list[tuple[int, int]]],
    strict_groups_b: list[list[tuple[int, int]]],
    mixed_a: list[tuple[int, int]],
    mixed_b: list[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c in mixed_a + mixed_b:
        output[r, c] = Color.BLACK
    for dent_groups, stamp_color in [(strict_groups_a, color_b), (strict_groups_b, color_a)]:
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (strict_groups_a, strict_groups_b):
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
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3X3:
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
        if host_count < BOUNDARY_MIN_HOST_3X3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _split_two(total: int, rng: random.Random) -> tuple[int, int]:
    a, b = total // 2, total - total // 2
    if rng.random() < 0.5:
        a, b = b, a
    return a, b


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    mixed_hi = min(MIXED_DENT_COUNT_MAX, n_total - 6)
    if mixed_hi < MIXED_DENT_COUNT_MIN:
        return None
    n_mixed = rng.randint(MIXED_DENT_COUNT_MIN, mixed_hi)
    n_strict = n_total - n_mixed
    n_strict_a, n_strict_b = _split_two(n_strict, rng)
    n_mixed_a, n_mixed_b = _split_two(n_mixed, rng)

    h, w = clean.shape
    strict_valid = {color_a: [], color_b: []}
    mixed_valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a:
                if _is_deep_strict_center(clean, r, c, color_a, color_b):
                    strict_valid[color_a].append((r, c))
                elif _is_mixed_majority_center(clean, r, c, color_a, color_b):
                    mixed_valid[color_a].append((r, c))
            elif v == color_b:
                if _is_deep_strict_center(clean, r, c, color_b, color_a):
                    strict_valid[color_b].append((r, c))
                elif _is_mixed_majority_center(clean, r, c, color_b, color_a):
                    mixed_valid[color_b].append((r, c))

    if len(strict_valid[color_a]) < n_strict_a or len(strict_valid[color_b]) < n_strict_b:
        return None
    if len(mixed_valid[color_a]) < n_mixed_a or len(mixed_valid[color_b]) < n_mixed_b:
        return None

    mixed_a = _pick_with_blockers(mixed_valid[color_a], n_mixed_a, [], rng)
    mixed_b = _pick_with_blockers(mixed_valid[color_b], n_mixed_b, mixed_a or [], rng)
    if mixed_a is None or mixed_b is None:
        return None

    blocked = mixed_a + mixed_b
    strict_a = _pick_with_blockers(strict_valid[color_a], n_strict_a, blocked, rng)
    strict_b = _pick_with_blockers(strict_valid[color_b], n_strict_b, blocked + (strict_a or []), rng)
    if strict_a is None or strict_b is None:
        return None

    occupied = mixed_a + mixed_b + strict_a + strict_b
    strict_groups_a = _expand_strict_pairs(strict_a, color_a, color_b, clean, occupied, rng)
    strict_groups_b = _expand_strict_pairs(strict_b, color_b, color_a, clean, occupied, rng)

    output = _stamp_output(clean, strict_groups_a, strict_groups_b, mixed_a, mixed_b, color_a, color_b)

    inp = clean.copy()
    for r, c in mixed_a + mixed_b:
        inp[r, c] = Color.BLACK
    for cells in strict_groups_a + strict_groups_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK

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
