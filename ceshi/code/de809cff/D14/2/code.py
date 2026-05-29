# Variant of de809cff
# Dimension: D14 — 边界突出（protrusion）的定义与处理 (multi_step)
# Change: 单像素边界突出被识别为：一个前景像素的4-邻居中多数为另一区域色或背景色，该像素被重着色为周围多数色... → 突出被视为凹痕触发变换

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the D14 variant
# a) Concretization and behavior trace
#    - The canvas still contains two large rectangular foreground regions on black.
#    - Black dents remain true holes: a host-colored cell is replaced by black in the input.
#    - The new object type is a one-cell attached protrusion at the boundary between the
#      two foreground regions: a single source-colored cell is painted one step into the
#      opposite region while staying 4-connected to its source region.
#    - Original flow on such a cell:
#      clean_foreground() rewrites it back to the surrounding host color, so it vanishes.
#    - Variant flow:
#      the same cleanup still reconstructs the clean baseline, but the rewritten cell is
#      recorded as a trigger center. Its 3x3 neighborhood is then stamped with the source
#      color and the center becomes teal, just like a dent produces a 3x3 patch + teal.
#    - Small example: if a blue cell protrudes one step into a red block, the old solver
#      would restore that cell to red. The new solver still restores the baseline to red,
#      but then stamps a blue 3x3 patch centered there and colors the center teal.
#
# b) Complexity calibration
#    - Dents are kept in their original count range so the task is still recognizably the
#      same family.
#    - 3-6 protrusions are added per sample, enough to make boundary-triggered patches
#      obvious but not enough to flood the boundary with overlapping marks.
#    - We restrict protrusions to region-region boundaries, not arbitrary background specks.
#      This keeps the new branch visible and avoids ambiguous patches floating in empty bg.
#
# c) Implicit assumptions audit
#    - _clean_foreground() in the base task assumes any one-cell boundary anomaly should be
#      erased. That remains useful only as a baseline reconstruction step; we therefore keep
#      the cleanup but no longer let its edits discard information.
#    - _dent_centers() only scans black cells, so protrusions would be invisible unless a new
#      detector scans foreground cells that cleanup changes to the opposite foreground color.
#    - Starting output from the cleaned grid is still valid because protrusions should not
#      survive as raw noise; they should survive only through the stamped patch.
#
# d) Placement legality
#    - Every dent/protrusion center is chosen with Chebyshev separation >= 3, so 3x3 stamps
#      do not overlap.
#    - A protrusion candidate is accepted only if inserting that one cell and then applying
#      cleanup reconstructs the exact clean grid. This guarantees the solver's baseline.
#    - Protrusions are rejected if they would form a 2x2 block of their source color, which
#      would stop them from behaving like single-cell bulges.
#
# e) Visual observability
#    - Human-solvable cue: besides interior black dents, some teal-centered patches now sit
#      exactly on the boundary between the two colored regions even though the input center is
#      not black there. Across 2-4 examples this is the visible signature of D14.
#    - No color-id arithmetic or hidden ordering is used; only local majority structure and
#      the two visible foreground regions matter.
#
# f) Visual cleanliness
#    - Boundary protrusions are sparse and separated, so teal marks do not stack into messy
#      bands.
#    - Background specks remain removable noise and do not turn into extra patch types.
#
# g) Color-change inferability
#    - Dent rule is unchanged: host-colored hole -> 3x3 patch of the other foreground color.
#    - Protrusion rule is structurally parallel: a source-colored one-cell bulge inside the
#      opposite region -> 3x3 patch of that same source color, center teal.
#    - Because there are only two foreground colors, the "opposite side" relation is easy to
#      infer from repeated boundary examples.


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
    Reconstruct the clean two-region baseline before applying dent/protrusion stamps.

    The cleanup logic is intentionally close to the original task: isolated foreground
    noise is removed, and one-cell boundary bulges collapse back to the local majority.
    In the variant, those collapsed bulges are later recovered as trigger centers.
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


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _protrusion_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    def _is_center(rr: int, cc: int) -> bool:
        v = int(original[rr, cc])
        host = int(clean[rr, cc])
        if v not in colors or host not in colors or host == v:
            return False
        if _has_same_2x2(original, rr, cc, v):
            return False

        same_touch = 0
        host_touch = 0
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nv = int(original[nr, nc])
            if nv == v:
                same_touch += 1
            elif nv == host:
                host_touch += 1

        return same_touch >= 1 and host_touch > same_touch and host_touch >= 2

    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if _is_center(r, c):
                centers.append((r, c, int(original[r, c])))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert dents and boundary protrusions into teal-centered 3x3 patches.

    Input structure:
    - Black background with two large foreground regions made from overlapping rectangles.
    - The regions contain black dents, a few removable background specks, and some
      one-cell protrusions that cross from one foreground region into the other.

    Rule:
    - First reconstruct the clean two-region baseline by removing isolated specks and
      collapsing one-cell bulges back to the local majority color.
    - Every original black dent whose cleaned 3x3 neighborhood is dominated by one
      foreground color becomes a 3x3 patch in the opposite foreground color.
    - Every original one-cell boundary protrusion that cleanup rewrites into the other
      foreground region is treated the same way: its 3x3 neighborhood is filled, using
      the protrusion's source color as the patch color, and the trigger center is teal.

    Key insight:
    - Cleanup is no longer the end of the story for boundary anomalies. Black holes and
      single-cell cross-boundary bulges are both trigger points, and teal marks the exact
      cells that caused a 3x3 opposite-side patch to appear.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    dents = _dent_centers(g, clean, colors, bg)
    protrusions = _protrusion_centers(g, clean, colors, bg)
    h, w = g.shape

    for r, c, host in dents:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c, stamp_color in protrusions:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)

    for r, c, _ in dents + protrusions:
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

# --- CFG: protrusions ---
PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX = 3, 6

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 16
BG_SPECK_MIN_BG_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 600


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


def _pick_separated_centers(
    candidates: list[tuple[int, int]],
    n: int,
    occupied: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    blocked = list(occupied)
    for r, c in candidates:
        if all(
            max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION
            for rr, cc in blocked + chosen
        ):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_dents: list[tuple[int, int]],
    occupied: list[tuple[int, int]],
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
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any(
                    max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION
                    for rr, cc in occupied
                    if (rr, cc) != (r, c)
                ):
                    continue
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in all_dents
                    if (r2, c2) != (r, c)
                ):
                    cells.append((nr, nc))
                    occupied.append((nr, nc))
                    break
        result.append(cells)
    return result


def _stamp_triggers(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    protrusions: list[tuple[int, int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for r, c, stamp_color in protrusions:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    for r, c, _ in protrusions:
        output[r, c] = Color.TEAL
    return output


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_noise: int, color_a: int, color_b: int, rng: random.Random) -> None:
    """Place isolated foreground-color specks in background area."""
    h, w = inp.shape
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_noise:
            break
        r, c = int(r_), int(c_)
        if any(
            not (0 <= nr < h and 0 <= nc < w) or int(clean[nr, nc]) != 0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
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
    """Check that adding a source-colored protrusion does not destabilize nearby source cells."""
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
    colors: tuple[int, int],
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
            if host == color_a and _safe_protrusion_candidate(clean, r, c, color_b, color_a, colors, bg):
                candidates.append((r, c, color_b))
            elif host == color_b and _safe_protrusion_candidate(clean, r, c, color_a, color_b, colors, bg):
                candidates.append((r, c, color_a))
    return candidates


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

    dents_a = _pick_separated_centers(valid[color_a][:], n_a, [], rng)
    dents_b = _pick_separated_centers(valid[color_b][:], n_b, dents_a or [], rng)
    if dents_a is None or dents_b is None:
        return None

    occupied = list(dents_a) + list(dents_b)
    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a, occupied, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_b, occupied, rng)
    occupied = [cell for group in groups_a + groups_b for cell in group]

    protrusion_candidates = _find_protrusion_candidates(clean, colors, bg)
    desired_hi = min(PROTRUSION_COUNT_MAX, len(protrusion_candidates))
    if desired_hi < PROTRUSION_COUNT_MIN:
        return None
    target = rng.randint(PROTRUSION_COUNT_MIN, desired_hi)
    protrusion_positions = None
    for n_try in range(target, PROTRUSION_COUNT_MIN - 1, -1):
        protrusion_positions = _pick_separated_centers(
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

    output = _stamp_triggers(clean, groups_a, groups_b, protrusions, color_a, color_b)

    inp = clean.copy()
    for cells in groups_a + groups_b:
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
