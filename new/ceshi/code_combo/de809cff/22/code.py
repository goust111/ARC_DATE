# Combo variant of de809cff (index 22)
# combo_spec: D3/2+D13/1
# Name: 少数色决定补丁色+棋盘格混合补丁

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the D3/2 + D13/1 combo
# -------------------------------------------
# a) concretization and behavior trace:
# - The input scene is still the same family as the original task: black background,
#   two large foreground regions, black dents, plus one-cell noise that must be
#   cleaned away first.
# - The fusion changes the dent-stamping stage. After cleanup, every valid dent is
#   examined through its cleaned 3x3 neighborhood. We count both foreground colors:
#   the larger count is the majority color, the smaller non-zero count is the
#   minority color. The 3x3 patch is then a checkerboard badge whose corner slots
#   use the minority color, whose orthogonal slots use the majority color, and
#   whose center is teal.
# - Small trace:
#       cleaned 3x3 around a dent:         original output:          combo output:
#           A A A                               B B B                  B A B
#           A 0 B            counts A=6,B=2     B 8 B                  A 8 A
#           A A B             majority=A        B B B                  B A B
#                             minority=B
#   The D13 component makes the patch mixed instead of solid. The D3 component says
#   the minority color chooses the primary checkerboard parity; because the center
#   is reserved for teal, those primary slots are the four corners.
#
# b) complexity calibration:
# - A solid opposite-color square would be too close to the source task. A mixed
#   checkerboard is visibly different on every dent.
# - A fixed checkerboard orientation would make the D3 part vacuous. Requiring each
#   generated dent to have both foreground colors in its clean 3x3 makes the notion
#   of a visible minority color real, not implicit.
# - The rule still has only one checkerboard orientation class: corners are the
#   primary slots. This keeps the pattern easy to infer from 2-4 ARC examples.
#
# c) implicit-assumption review of main():
# - _main_colors() still assumes exactly two dominant non-black colors in the input.
#   The generator preserves that.
# - _clean_foreground() still models the corruption as isolated background specks and
#   one-cell boundary contamination. The combo does not alter the cleanup logic.
# - Dent detection can no longer stop at "host region found". It must also preserve
#   the minority color count, because that now controls which checkerboard parity
#   gets which color.
# - A bulk assignment like `out[r-1:r+2, c-1:c+2] = color` is no longer valid. Patch
#   painting must write cells individually by parity.
#
# d) placement legality:
# - Mixed checkerboard patches are not commutative under overlap, so the generator
#   disables 2-cell dents and enforces a global Chebyshev gap of at least 3 between
#   all dent centers across both colors.
# - Valid dent candidates must stay inside the clean foreground, survive the majority
#   test after blackening the center, and have a non-zero minority count in the
#   cleaned 3x3.
#
# e) visual observability:
# - Humans can see the rule without color-id arithmetic: each black hole becomes a
#   teal-centered checkerboard made from the same two foreground colors already
#   present in the scene.
# - Because the minority color always lands on the corners, boundary dents with
#   different local ratios still share a repeated visual motif.
#
# f) visual neatness:
# - Disjoint patches prevent ambiguous overwrite order.
# - Restricting dents to mixed neighborhoods keeps the corners meaningfully tied to
#   nearby opposite-color evidence instead of looking like arbitrary recoloring.
#
# g) change inferability:
# - Only two foreground colors exist, so the solver never has to infer a many-way
#   color mapping. The extra burden is just to notice that the less common local
#   color takes the corner slots of the checkerboard.
#
# main() new assumptions audit:
# - Every transformed dent must come from an original black cell. The generator
#   explicitly blackens only chosen dent centers and adds no other black noise in
#   foreground regions.
# - Every transformed dent must have both foreground colors visible in its cleaned
#   3x3. The generator filters candidates with minority_count >= 1.
# - Patch overlap would make later writes matter. The generator guarantees global
#   center spacing so the parity pattern stays deterministic.


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


def _dent_specs(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        majority = max(colors, key=lambda color: counts[int(color)])
        minority = min(colors, key=lambda color: counts[int(color)])
        if (
            counts[int(majority)] >= 5
            and counts[int(majority)] > counts[int(minority)]
            and counts[int(minority)] >= 1
        ):
            centers.append((rr, cc, int(majority), int(minority)))
    return centers


def _paint_checker_patch(out: np.ndarray, r: int, c: int, primary: int, secondary: int, teal: int) -> None:
    h, w = out.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if rr == r and cc == c:
                continue
            out[rr, cc] = int(primary if (abs(rr - r) + abs(cc - c)) % 2 == 0 else secondary)
    out[r, c] = int(teal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into minority-led checkerboard markers.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain black dents plus a small amount of isolated
      one-cell noise in the background or across the region boundary.

    Rule:
    - First clean the foreground locally so isolated stray cells vanish and
      one-cell cross-boundary contamination reverts to its surrounding region.
    - For each original black cell whose cleaned 3x3 neighborhood has one
      foreground color in the majority and the other still present as a minority,
      build a 3x3 checkerboard patch centered there.
    - The minority foreground color fills the four corner slots, the majority
      foreground color fills the four orthogonal slots, and the center becomes
      teal (8).

    Key insight: The black defects are still found by cleaning and inspecting the
    local 3x3 neighborhood, but the repair is now a mixed badge whose corner color
    is chosen by the less frequent neighboring foreground color rather than by a
    fixed region-to-region swap.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    for r, c, majority, minority in _dent_specs(g, clean, colors, bg):
        _paint_checker_patch(out, r, c, minority, majority, Color.TEAL)
    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 10, 18
DENT_MIN_HOST_3X3 = 5
DENT_MIN_OTHER_3X3 = 1
DENT_PATCH_SEPARATION = 3
MAX_DENT_PICK_ATTEMPTS = 120

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3X3 = 6
BOUNDARY_MIN_HOST_3X3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 800


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles (simple occlusion, painter's algorithm)."""
    n_rects = rng.choice([2, 3, 3])
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


def _dent_role_from_clean(
    grid: np.ndarray, r: int, c: int, host: int, other: int
) -> tuple[int, int] | None:
    if grid[r, c] != host:
        return None
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    if host_count < DENT_MIN_HOST_3X3 or other_count < DENT_MIN_OTHER_3X3 or host_count <= other_count:
        return None
    return host_count, other_count


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


def _stamp_dents(clean: np.ndarray, dent_specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    output = clean.copy()
    for r, c, majority, minority in dent_specs:
        _paint_checker_patch(output, r, c, minority, majority, Color.TEAL)
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


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random
) -> None:
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
            if v == color_a and _dent_role_from_clean(clean, r, c, int(color_a), int(color_b)) is not None:
                valid[int(color_a)].append((r, c))
            elif v == color_b and _dent_role_from_clean(clean, r, c, int(color_b), int(color_a)) is not None:
                valid[int(color_b)].append((r, c))

    if len(valid[int(color_a)]) < n_a + 2 or len(valid[int(color_b)]) < n_b + 2:
        return None

    chosen = _pick_dent_centers_by_color(valid, counts, colors, rng)
    if chosen is None:
        return None
    dents_a, dents_b = chosen

    dent_specs: list[tuple[int, int, int, int]] = []
    for host, other, dents in (
        (int(color_a), int(color_b), dents_a),
        (int(color_b), int(color_a), dents_b),
    ):
        for r, c in dents:
            role = _dent_role_from_clean(clean, r, c, host, other)
            if role is None:
                return None
            dent_specs.append((r, c, host, other))

    output = _stamp_dents(clean, dent_specs)

    inp = clean.copy()
    for r, c, _majority, _minority in dent_specs:
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
