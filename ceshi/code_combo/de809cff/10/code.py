# Combo variant of de809cff (index 10)
# combo_spec: D3/1+D14/2
# Name: 严格全一邻域判定+突出视为凹痕

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for combo D3/1 + D14/2
# a) Concretization and behavior trace
#    - The base canvas still has two large foreground regions on black background.
#    - There are now three visually distinct local situations:
#      1) strict dents: black holes whose cleaned 3x3 neighborhood contains only one
#         foreground color, so they still trigger a 3x3 opposite-color patch;
#      2) mixed dents: black holes near the color-color boundary whose cleaned 3x3 has
#         a majority host but not a uniform host, so they would transform in the base
#         task but remain black here;
#      3) protrusions: one-cell cross-boundary bulges that cleanup rewrites back to the
#         surrounding host color, but they are then recovered as trigger centers and
#         stamped just like dents.
#    - Small trace:
#      old dent rule: [a a a; a 0 b; a b b] -> host a by local majority -> transform.
#      combo dent rule: same neighborhood is mixed, so this hole stays black.
#      old protrusion rule: [a a a; a b a; a a a] centered on a b-cell inside region a
#      gets cleaned back to a and disappears. Combo rule: cleanup still restores a, but
#      the center is recorded as a trigger and becomes a teal-centered b patch.
#
# b) Complexity calibration
#    - The generator deliberately includes both mixed dents and protrusions in every
#      sample. If either count were allowed to collapse to zero, the combo could look
#      indistinguishable from a single-variant or even the base task.
#    - Mixed dents are kept sparse and separated from true triggers so the viewer can
#      notice that some black holes transform while others do not.
#    - Protrusions are limited to a small count on the region-region interface; this is
#      enough to expose the new branch without turning the boundary into a solid band.
#
# c) Implicit assumptions audit
#    - `_main_colors` still assumes exactly two salient foreground colors. The generator
#      only uses two non-black, non-teal colors and keeps all noise within those colors.
#    - The original `_dent_centers` relied on majority counts; that assumption breaks
#      under D3/1, so it is replaced with a uniform-foreground check.
#    - The original cleanup erased protrusions permanently. Under D14/2, cleanup can no
#      longer be the final state; it is only a baseline reconstruction before trigger
#      extraction.
#    - Starting the output from `clean.copy()` is still valid because mixed dents are
#      explicitly written back to black in the generator's oracle output, while true
#      triggers are reintroduced by stamping.
#
# d) Placement legality
#    - All trigger centers use Chebyshev separation >= 3, except optional strict dent
#      pairs, so independent 3x3 stamps do not overlap.
#    - Strict dent pairs are only expanded into neighboring cells that also satisfy the
#      strict-uniform criterion and do not collide with any other reserved center.
#    - Protrusion candidates are accepted only if inserting the source-colored bulge and
#      running cleanup reconstructs the original clean host region locally.
#
# e) Visual observability
#    - Across a few examples, a solver can see two coupled changes:
#      1) some boundary-adjacent black holes no longer transform;
#      2) some non-black boundary cells do transform.
#    - Both effects are local and visible directly in the input/output grids; no hidden
#      dependence on color ids or object ordering is needed.
#
# f) Visual cleanliness
#    - Only two foreground colors plus teal are used, and bg specks remain removable.
#    - Mixed dents are kept away from trigger centers so untouched black holes are not
#      overwritten by nearby stamps.
#    - Protrusions are sparse and separated from dents, keeping teal markers legible.
#
# g) Change inferability
#    - The color logic remains structural: a trigger centered in one region produces a
#      3x3 patch of the opposite region color.
#    - For a protrusion, the "opposite region color" equals the protrusion's source
#      color because cleanup reveals the host region underneath. Repeated examples make
#      this correspondence inferable without any numeric-color convention.

MIN_UNIFORM_HOST_3X3 = 5

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

# --- CFG: protrusions ---
PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX = 3, 6

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 16
BG_SPECK_MIN_BG_3X3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 800


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


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
    Reconstruct the clean two-region baseline before checking trigger centers.

    Isolated foreground specks collapse to black, and one-cell cross-boundary
    bulges collapse back to the local host region. The combo rule then examines
    the original grid together with this reconstructed baseline.
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


def _protrusion_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int]
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            source = int(original[r, c])
            host = int(clean[r, c])
            if source not in colors or host not in colors or source == host:
                continue
            if _has_same_2x2(original, r, c, source):
                continue

            same_touch = 0
            host_touch = 0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                nv = int(original[nr, nc])
                if nv == source:
                    same_touch += 1
                elif nv == host:
                    host_touch += 1

            if same_touch >= 1 and host_touch > same_touch and host_touch >= 2:
                centers.append((r, c, host))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert strict black dents and one-cell protrusions into teal-centered patches.

    Input structure:
    - Black background with two large foreground-color regions built from overlapping rectangles.
    - Inside and near the boundary of those regions are black dents, a few isolated background
      specks, and some one-cell protrusions where one region pokes into the other.

    Rule:
    - First reconstruct the clean two-region baseline by removing isolated foreground specks and
      collapsing one-cell boundary bulges back to the surrounding host color.
    - Then scan the original black cells: only dents whose cleaned clipped 3x3 neighborhood has
      foreground cells from exactly one color, with at least 5 such cells, are treated as triggers.
      Mixed-neighborhood black holes stay black.
    - Also scan original non-black cells that cleanup rewrote into the opposite foreground region:
      each such one-cell protrusion is treated as a trigger too. Every trigger stamps its clipped
      3x3 neighborhood with the opposite color of the cleaned host region, and the trigger center
      itself becomes teal (8).

    Key insight:
    - The task now has two trigger sources with different visibility signatures: deep uniform dents
      still flip, mixed dents stop flipping, and cross-boundary protrusions start flipping. The
      same teal-centered 3x3 patch rule applies after those sources are identified.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    dents = _dent_centers(g, clean, colors, bg)
    protrusions = _protrusion_centers(g, clean, colors)

    out = clean.copy()
    h, w = g.shape
    for r, c, host in dents + protrusions:
        stamp_color = _other_color(colors, host)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)

    for r, c, _host in dents + protrusions:
        out[r, c] = Color.TEAL

    return out


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
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    return host_count >= MIN_UNIFORM_HOST_3X3 and host_count > other_count


def _is_deep_strict_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if not _is_relaxed_dent_center(grid, r, c, host, other):
        return False
    return bool(np.all(_sub3x3(grid, r, c) == int(host)))


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
    protrusions: list[tuple[int, int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c in mixed_a + mixed_b:
        output[r, c] = Color.BLACK
    for dent_groups, stamp_color in ((strict_groups_a, color_b), (strict_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for r, c, source in protrusions:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(source)
    for dent_groups in (strict_groups_a, strict_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    for r, c, _ in protrusions:
        output[r, c] = Color.TEAL
    return output


def _add_bg_noise(
    inp: np.ndarray, clean: np.ndarray, n_noise: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    h, w = inp.shape
    for r_, c_ in bg_candidates:
        if placed >= n_noise:
            break
        r, c = int(r_), int(c_)
        if any(
            not (0 <= nr < h and 0 <= nc < w) or int(clean[nr, nc]) != 0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            continue
        if int(np.sum(_sub3x3(inp, r, c) == 0)) < BG_SPECK_MIN_BG_3X3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


def _neighbor_protrusion_safe(
    clean: np.ndarray,
    r: int,
    c: int,
    source: int,
    host: int,
    bg: int,
) -> bool:
    h, w = clean.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(clean[nr, nc]) != int(source):
                continue
            if _has_same_2x2(clean, nr, nc, source):
                continue

            nsub = _sub3x3(clean, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = source
            own_n = int(np.sum(nsub == source))
            other_n = int(np.sum(nsub == host))
            bg_n = int(np.sum(nsub == bg))
            if (other_n > own_n and other_n >= 5) or (bg_n >= 5 and own_n <= 4) or own_n <= 2:
                return False
    return True


def _safe_protrusion_candidate(
    clean: np.ndarray,
    r: int,
    c: int,
    source: int,
    host: int,
    bg: int,
) -> bool:
    if int(clean[r, c]) != int(host):
        return False
    if _would_form_2x2(clean, r, c, source):
        return False

    h, w = clean.shape
    same_touch = 0
    host_touch = 0
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        nv = int(clean[nr, nc])
        if nv == source:
            same_touch += 1
        elif nv == host:
            host_touch += 1
    if same_touch < 1 or host_touch <= same_touch or host_touch < 2:
        return False

    sub = _sub3x3(clean, r, c).copy()
    rr0, cc0 = max(0, r - 1), max(0, c - 1)
    sub[r - rr0, c - cc0] = source
    own = int(np.sum(sub == source))
    other_count = int(np.sum(sub == host))
    if other_count <= own or other_count < 5:
        return False
    if not _neighbor_protrusion_safe(clean, r, c, source, host, bg):
        return False
    return True


def _find_protrusion_candidates(
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
) -> list[tuple[int, int, int]]:
    color_a, color_b = colors
    h, w = clean.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host == color_a and _safe_protrusion_candidate(clean, r, c, color_b, color_a, bg):
                candidates.append((r, c, color_b))
            elif host == color_b and _safe_protrusion_candidate(clean, r, c, color_a, color_b, bg):
                candidates.append((r, c, color_a))
    return candidates


def _split_two(total: int, rng: random.Random) -> tuple[int, int]:
    a, b = total // 2, total - total // 2
    if rng.random() < 0.5:
        a, b = b, a
    return a, b


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (color_a, color_b)
    bg = Color.BLACK

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
    occupied = mixed_a + mixed_b + [cell for group in strict_groups_a + strict_groups_b for cell in group]

    protrusion_candidates = _find_protrusion_candidates(clean, colors, bg)
    desired_hi = min(PROTRUSION_COUNT_MAX, len(protrusion_candidates))
    if desired_hi < PROTRUSION_COUNT_MIN:
        return None
    target = rng.randint(PROTRUSION_COUNT_MIN, desired_hi)
    protrusion_positions = None
    for n_try in range(target, PROTRUSION_COUNT_MIN - 1, -1):
        protrusion_positions = _pick_with_blockers(
            [(r, c) for r, c, _ in protrusion_candidates],
            n_try,
            occupied,
            rng,
        )
        if protrusion_positions is not None:
            break
    if protrusion_positions is None:
        return None

    chosen_pos = set(protrusion_positions)
    protrusions = [(r, c, source) for r, c, source in protrusion_candidates if (r, c) in chosen_pos]
    if len(protrusions) != len(chosen_pos):
        return None

    output = _stamp_output(clean, strict_groups_a, strict_groups_b, mixed_a, mixed_b, protrusions, color_a, color_b)

    inp = clean.copy()
    for r, c in mixed_a + mixed_b:
        inp[r, c] = bg
    for cells in strict_groups_a + strict_groups_b:
        for r, c in cells:
            inp[r, c] = bg
    for r, c, source in protrusions:
        inp[r, c] = source

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    _add_bg_noise(inp, clean, n_noise, color_a, color_b, rng)

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
