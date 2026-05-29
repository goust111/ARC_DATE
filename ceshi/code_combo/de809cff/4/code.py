# Combo variant of de809cff (index 4)
# combo_spec: D2/3+D7/2
# Name: 中心颜色交替+凹痕沿直线排列

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the fused variant:
# - Concrete behavior: the base task still cleans 1-cell noise, detects black dents
#   by local 3x3 host-color majority, and stamps 3x3 opposite-color patches. The
#   fusion changes two linked stages. First, generator-side dents are no longer
#   scattered: each foreground region gets a same-direction straight run of dent
#   centers spaced by 2 cells, so adjacent 3x3 patches overlap into a visible band.
#   Second, solver-side center markers are no longer uniform: after dents are found,
#   they are enumerated in row-major scan order and colored teal, maroon, teal,
#   maroon, ...
# - Complexity calibration: using exactly two long lines keeps the linear pattern
#   obvious in 2-4 examples, while repeated overlap along each line gives enough
#   evidence for the alternating stripe rule. Allowing only horizontal, vertical,
#   or NW-SE diagonal runs avoids ambiguous anti-diagonal ordering.
# - Hidden assumptions audit: the original solver assumes every true dent remains a
#   black cell whose cleaned 3x3 neighborhood still has >=5 host cells and beats the
#   other color. Spacing dents by step 2 preserves that assumption because centers do
#   not enter each other's 3x3 windows, even though their output patches overlap.
#   The new alternation assumes scan order is visually recoverable, so generator keeps
#   one line in the upper-left region and one later line in the lower-right region.
# - Placement legality: candidate line cells are prefiltered through the same dent
#   safety test used by the solver, clipped to bounds, required to remain separated
#   across the two regions, and rechecked by an oracle after noise is added.
# - Observability and cleanliness: only two marker colors are used, with a single
#   global alternation rule. The line layout makes odd/even indexing visible without
#   relying on numeric color ids or hidden metadata, and avoiding pair-dents prevents
#   clutter that would blur the stripe structure.


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

    Two sweeps are enough for the intended corruption pattern: the first removes
    wrong-color outliers, the second stabilizes their neighbors.
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
    centers.sort(key=lambda item: (int(item[0]), int(item[1])))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn cleaned black dents into opposite-color stripe patches with alternating center markers.

    Input structure:
    - Black background with two large foreground-color regions.
    - Each region contains a straight horizontal, vertical, or NW-SE diagonal run of
      black dents, plus a few isolated color specks or one-cell boundary intrusions.

    Rule:
    - First clean the two foreground regions locally: isolated foreground specks revert
      to black, and one-cell boundary contamination snaps back to the surrounding region.
    - Every original black dent whose cleaned 3x3 neighborhood is dominated by one
      foreground color becomes the center of a 3x3 patch in the opposite foreground color.
    - Order all detected dents by row-major scan order; color odd-indexed centers teal (8)
      and even-indexed centers maroon (9).

    Key insight: The black cells that survive the local majority test are the meaningful
    dents, and the straight-run layout makes their scan order legible enough to explain
    why the center markers alternate teal-maroon-teal-maroon across the output.
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

    for idx, (r, c, _host) in enumerate(centers):
        out[r, c] = Color.TEAL if idx % 2 == 0 else Color.MAROON

    return out


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.55, 0.82
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 110
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84
OVERLAP_FRAC_MIN, OVERLAP_FRAC_MAX = 0.08, 0.30

# --- CFG: dent lines ---
LINE_DIRECTIONS = ((0, 2), (2, 0), (2, 2))
LINE_LEN_MIN, LINE_LEN_MAX = 6, 10
INTER_LINE_SEPARATION = 5
DENT_MIN_HOST_3x3 = 5

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((N, N), dtype=int)

    ha = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    wa = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    hb = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    wb = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))

    ra0 = rng.randint(0, RECT_MARGIN_MAX)
    ca0 = rng.randint(0, RECT_MARGIN_MAX)
    rb0 = max(0, N - hb - rng.randint(0, RECT_MARGIN_MAX))
    cb0 = max(0, N - wb - rng.randint(0, RECT_MARGIN_MAX))
    ra1, ca1 = min(N, ra0 + ha), min(N, ca0 + wa)
    rb1, cb1 = min(N, rb0 + hb), min(N, cb0 + wb)

    overlap_h = max(0, min(ra1, rb1) - max(ra0, rb0))
    overlap_w = max(0, min(ca1, cb1) - max(ca0, cb0))
    overlap = overlap_h * overlap_w
    if overlap == 0:
        return None

    grid[ra0:ra1, ca0:ca1] = color_a
    grid[rb0:rb1, cb0:cb1] = color_b

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    overlap_frac = overlap / (N * N)
    if not (OVERLAP_FRAC_MIN <= overlap_frac <= OVERLAP_FRAC_MAX):
        return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


def _scan_key(coord: tuple[int, int]) -> tuple[int, int]:
    return int(coord[0]), int(coord[1])


def _min_chebyshev(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a for r2, c2 in b)


def _collect_valid_positions(
    clean: np.ndarray, host: int, other: int
) -> list[tuple[int, int]]:
    h, w = clean.shape
    valid: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if clean[r, c] == host and _is_safe_dent_center(clean, r, c, host, other):
                valid.append((int(r), int(c)))
    return valid


def _line_candidates(
    valid: list[tuple[int, int]], step: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    pos_set = set(valid)
    dr, dc = map(int, step)
    candidates: list[list[tuple[int, int]]] = []
    for start in sorted(valid):
        prev = (start[0] - dr, start[1] - dc)
        if prev in pos_set:
            continue
        run: list[tuple[int, int]] = []
        cur = start
        while cur in pos_set:
            run.append(cur)
            cur = (cur[0] + dr, cur[1] + dc)
        if len(run) < LINE_LEN_MIN:
            continue
        max_len = min(LINE_LEN_MAX, len(run))
        for length in range(LINE_LEN_MIN, max_len + 1):
            for offset in range(0, len(run) - length + 1):
                candidates.append(run[offset : offset + length])
    return candidates


def _choose_line_pair(
    cand_a: list[list[tuple[int, int]]],
    cand_b: list[list[tuple[int, int]]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    feasible: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    early_a = sorted(cand_a, key=lambda line: (_scan_key(line[0]), _scan_key(line[-1])))[:80]
    late_b = sorted(cand_b, key=lambda line: (_scan_key(line[0]), _scan_key(line[-1])), reverse=True)[:80]
    for line_a in early_a:
        end_a = _scan_key(line_a[-1])
        for line_b in late_b:
            start_b = _scan_key(line_b[0])
            if start_b <= end_a:
                continue
            if _min_chebyshev(line_a, line_b) < INTER_LINE_SEPARATION:
                continue
            feasible.append((line_a, line_b))
    if not feasible:
        return None
    return rng.choice(feasible)


def _stamp_output(
    clean: np.ndarray,
    dents: list[tuple[int, int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host in dents:
        other = color_b if int(host) == int(color_a) else color_a
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for idx, (r, c, _host) in enumerate(sorted(dents, key=lambda item: (item[0], item[1]))):
        output[r, c] = Color.TEAL if idx % 2 == 0 else Color.MAROON
    return output


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


def _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng):
    h, w = inp.shape
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
    fg_pool = [c for c in Color.NOT_BLACK if c not in (Color.TEAL, Color.MAROON)]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    valid_a = _collect_valid_positions(clean, color_a, color_b)
    valid_b = _collect_valid_positions(clean, color_b, color_a)
    if len(valid_a) < LINE_LEN_MIN or len(valid_b) < LINE_LEN_MIN:
        return None

    directions = list(LINE_DIRECTIONS)
    rng.shuffle(directions)
    chosen = None
    for step in directions:
        cand_a = _line_candidates(valid_a, step)
        cand_b = _line_candidates(valid_b, step)
        if not cand_a or not cand_b:
            continue
        chosen = _choose_line_pair(cand_a, cand_b, rng)
        if chosen is not None:
            break
    if chosen is None:
        return None

    line_a, line_b = chosen
    dents = [(r, c, color_a) for r, c in line_a] + [(r, c, color_b) for r, c in line_b]
    output = _stamp_output(clean, dents, color_a, color_b)

    inp = clean.copy()
    for r, c, _host in dents:
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
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
