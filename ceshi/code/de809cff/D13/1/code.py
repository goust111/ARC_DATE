# Variant of de809cff
# Dimension: D13 — 两种前景色之间的对调映射关系 (meta)
# Change: 固定双色对调：区域A的凹痕用色B填充补丁，区域B的凹痕用色A填充补丁（简单二元交换）... → 凹痕补丁颜色为两种前景色的混合标志

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the D13 variant
# ----------------------------------
# concretization:
# - The input still contains two large foreground regions on black, plus black dents
#   and single-cell noise. The cleaning and dent-detection stages remain the same.
# - The output change is local to each detected dent. Instead of replacing the whole
#   3x3 neighborhood with the opposite solid color, the dent becomes a mixed badge:
#       host other host
#       other teal other
#       host other host
#   Here "host" means the region color that owns the dent after cleanup, and
#   "other" is the second foreground color.
# - Small trace example:
#   original rule for a dent inside color A:
#       A A A      B B B
#       A 0 A  ->  B 8 B
#       A A A      B B B
#   new variant:
#       A A A      A B A
#       A 0 A  ->  B 8 B
#       A A A      A B A
#   The change happens only at the patch-stamping step; dent detection itself is
#   unchanged.
#
# complexity calibration:
# - Keeping both foreground colors inside every patch makes the variant visibly
#   distinct from the original opposite-color swap.
# - The mixed patch has only one orientation class per dent: corners keep the host
#   color, orthogonal neighbors take the other color. That is strong enough to infer
#   from a few examples without adding extra hidden cases.
#
# implicit-assumption review of main():
# - _main_colors() assumes exactly two non-black input colors dominate the scene.
#   This still holds because the generator uses exactly two foreground colors and
#   teal appears only in outputs, never in inputs.
# - _clean_foreground() assumes non-foreground cells should become background during
#   cleanup. This remains valid because the variant changes only the output marking,
#   not the input corruption pattern.
# - _dent_centers() assumes a dent is an original black cell whose cleaned 3x3 is
#   majority-owned by one foreground color with >=5 host cells. Still valid.
# - The old `out[...] = other` patch fill relied on every patch cell sharing the
#   same color. That assumption is now false, so the stamping step must become
#   parity-aware and write each patch cell individually.
#
# placement legality:
# - Mixed patches are no longer commutative under overlap. The original generator's
#   2-cell dent pairs and per-color independent center selection could create
#   overlapping 3x3 patches whose result would depend on write order.
# - To keep the local rule deterministic and visually clean, the generator now:
#   1. uses only single-cell dents;
#   2. enforces the patch-separation constraint across both colors together.
#   This guarantees all 3x3 mixed patches are disjoint, so solver and generator
#   agree without hidden precedence rules.
#
# visual observability / color inferability:
# - No color-id arithmetic is used for the transformation rule. Humans only need to
#   see that dents become checkerboards made from the two visible region colors, with
#   teal still marking the center.
# - Because corners match the surrounding host region, the patch remains easy to
#   associate with its source region instead of looking like arbitrary noise.
#
# main() new assumptions audit:
# - The solver assumes any generated dent patches are effectively disjoint. The
#   generator guarantees this by global center spacing and by disabling adjacent-pair
#   dents.
# - The solver also assumes the dent host can still be inferred from the cleaned 3x3
#   majority. The generator continues to place dents only where that majority check
#   succeeds.


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


def _paint_mixed_patch(
    out: np.ndarray, r: int, c: int, host: int, colors: tuple[int, int], teal: int
) -> None:
    h, w = out.shape
    other = colors[1] if int(host) == int(colors[0]) else colors[0]
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if rr == r and cc == c:
                continue
            out[rr, cc] = int(host if (abs(rr - r) + abs(cc - c)) % 2 == 0 else other)
    out[r, c] = int(teal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into mixed checkerboard badges.

    Input structure:
    - Black background with two large foreground colors.
    - The large regions contain small black dents and a few one-cell color specks
      or protrusions along region boundaries.

    Rule:
    - First clean the two foreground regions locally: isolated specks become black,
      and one-cell boundary contamination reverts to the surrounding region color.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly owned by
      one foreground color becomes the center of a 3x3 mixed patch.
    - In that patch, the four corners keep the dent's host color, the four
      orthogonal neighbors use the other foreground color, and the center becomes
      teal (8).

    Key insight: The black defects are still located by the cleanup-majority test,
    but the repair marker is no longer a solid opposite-color square; it is a
    two-color checkerboard badge that preserves the host color on the corners.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    for r, c, host in centers:
        _paint_mixed_patch(out, r, c, host, colors, Color.TEAL)

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

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400
MAX_DENT_PICK_ATTEMPTS = 80


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


def _far_enough(r: int, c: int, chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen)


def _pick_dent_centers_by_color(
    valid: dict[int, list[tuple[int, int]]], counts: dict[int, int], colors: tuple[int, int], rng: random.Random
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(MAX_DENT_PICK_ATTEMPTS):
        pending: list[tuple[int, tuple[int, int]]] = []
        for color in colors:
            pending.extend((int(color), cell) for cell in valid[int(color)])
        rng.shuffle(pending)

        chosen: dict[int, list[tuple[int, int]]] = {int(colors[0]): [], int(colors[1]): []}
        occupied: list[tuple[int, int]] = []
        for color, (r, c) in pending:
            if len(chosen[color]) >= counts[color]:
                continue
            if not _far_enough(r, c, occupied):
                continue
            chosen[color].append((r, c))
            occupied.append((r, c))
            if all(len(chosen[int(color_)]) >= counts[int(color_)] for color_ in colors):
                return chosen[int(colors[0])], chosen[int(colors[1])]
    return None


def _stamp_dents(clean: np.ndarray, dent_groups: list[tuple[int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    """Build output by stamping checkerboard patches and marking teal at dent centers."""
    output = clean.copy()
    for r, c, host in dent_groups:
        _paint_mixed_patch(output, r, c, host, colors, Color.TEAL)
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


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random
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
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a
    counts = {int(color_a): n_a, int(color_b): n_b}

    h, w = clean.shape
    valid = {int(color_a): [], int(color_b): []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[int(color_a)].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[int(color_b)].append((r, c))

    if len(valid[int(color_a)]) < n_a + 2 or len(valid[int(color_b)]) < n_b + 2:
        return None

    chosen = _pick_dent_centers_by_color(valid, counts, colors, rng)
    if chosen is None:
        return None
    dents_a, dents_b = chosen

    dent_groups = [(r, c, int(color_a)) for r, c in dents_a] + [(r, c, int(color_b)) for r, c in dents_b]
    output = _stamp_dents(clean, dent_groups, colors)

    inp = clean.copy()
    for r, c, _host in dent_groups:
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
