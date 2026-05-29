# Combo variant of de809cff (index 12)
# combo_spec: D2/0+D14/0
# Name: 空心补丁（中心黑色）+突出标记为teal

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for combo D2/0 + D14/0
# a) Concretization and behavior trace
#    - Base task:
#      1) cleanup removes isolated bg specks and rewrites one-cell cross-boundary
#         protrusions back into the surrounding host region;
#      2) each black dent stamps a 3x3 patch of the opposite foreground color;
#      3) the dent center itself becomes teal.
#    - Combo task:
#      1) cleanup still reconstructs the clean two-region baseline;
#      2) each black dent still stamps the opposite-color 3x3 patch, but its center
#         is restored to black instead of teal, so the result is a hollow 3x3 patch;
#      3) each one-cell region-region protrusion is no longer erased semantically:
#         it is shown at its original location as a teal marker, with no 3x3 stamp.
#    - Small local trace, with `a`/`b` the two foreground colors:
#      old dent:         [a a a]      old output:      [b b b]
#                        [a 0 a]                       [b 8 b]
#                        [a a a]                       [b b b]
#      combo dent: same input, but output becomes     [b b b]
#                                                     [b 0 b]
#                                                     [b b b]
#      old protrusion:   [a a a]      old cleanup/output center -> a
#                        [a b a]
#                        [a a a]
#      combo protrusion: cleanup still says host is `a`, but the final output center
#      is teal at that same location, so protrusions remain visible instead of vanishing.
#
# b) Complexity calibration
#    - The combo would collapse visually if samples had only dents or only protrusions.
#      The generator therefore enforces both: many dents and 3-6 protrusions per sample.
#    - Dent centers stay well separated, so hollow centers are easy to see as black dots
#      inside opposite-color 3x3 patches rather than merging into larger black blobs.
#    - Protrusions are sparse and live only on the fg-fg boundary, which makes teal's
#      new meaning repeated and legible without flooding the canvas.
#
# c) Implicit assumptions audit
#    - `_main_colors` still assumes exactly two salient foreground colors. The generator
#      only uses two non-black, non-teal colors and keeps noise within those colors.
#    - The original `_clean_foreground` treated protrusions as disposable corruption.
#      That assumption no longer holds for the final output, but it remains valid as a
#      baseline reconstruction step before we re-mark protrusion cells.
#    - The original `out = clean.copy()` is no longer sufficient, because it would lose
#      both combo signals: dent centers would not be forced back to black, and protrusion
#      sites would stay host-colored. The new writeback explicitly restores both.
#    - The dent detector still scans original black cells only. This remains correct
#      because protrusions are handled by a separate detector on foreground cells whose
#      cleaned color differs from their input color.
#
# d) Placement legality
#    - Dents and protrusions use global Chebyshev separation >= 3, except for optional
#      two-cell dent pairs. This prevents unrelated 3x3 dent patches from overlapping
#      and prevents protrusion markers from landing inside a dent patch.
#    - Protrusion candidates are accepted only when inserting the source-colored cell
#      would be cleaned back to the host region under the solver's own cleanup logic.
#    - Background specks are placed only on isolated background cells so they remain
#      removable noise rather than creating accidental new protrusions or dents.
#
# e) Visual observability
#    - Across 2-4 examples, a human can see two stable cues:
#      1) transformed dents are surrounded by the opposite color but keep a black center;
#      2) teal no longer appears at black input holes and instead appears only on
#         non-black cells sitting at the boundary between the two regions.
#    - No reasoning depends on numeric color ids; only visible local structure matters.
#
# f) Visual cleanliness
#    - Only four visible roles exist: black background, two foreground colors, teal
#      protrusion markers. That is still a compact palette.
#    - Because protrusions are not expanded into patches in this combo, teal remains a
#      clean point marker rather than mixing with opposite-color 3x3 stamps.
#    - Hollow dents remain readable because their black centers are enclosed by colored
#      3x3 neighborhoods, so they do not visually blend with large background regions.
#
# g) Change inferability
#    - The color mapping for dents is still structural and repeated many times:
#      a dent inside region A stamps color B, and vice versa.
#    - The new distinction is role-based, not color-id-based:
#      black input hole -> hollow opposite-color patch;
#      non-black one-cell protrusion -> teal point.
#    - Repeated examples support the inference that teal means "protrusion" while black
#      can mean either background or the preserved center of a transformed dent.
#
# 9) New solver assumptions audited against the generator
#    - Every protrusion marker in the output must correspond to an input foreground cell
#      whose cleaned color is the opposite foreground color; the generator constructs only
#      such candidates.
#    - Every hollow dent center must be an original black cell whose cleaned 3x3 has a
#      unique foreground majority >= 5; the generator places dents only at those sites.
#    - No protrusion center may fall inside any dent patch, or the teal marker would
#      obscure the hollow-center cue. The generator enforces separation from all dent cells.


# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3X3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: protrusions ---
PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX = 3, 6

# --- CFG: noise ---
BG_NOISE_COUNT_MIN, BG_NOISE_COUNT_MAX = 6, 16
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
    Reconstruct the clean two-region baseline before applying combo-specific marks.

    Isolated foreground noise collapses into background, and one-cell region-region
    protrusions collapse back into the local host region. The combo then reuses this
    cleaned baseline to decide which black cells are dents and which foreground cells
    were protrusions.
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
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = _other_color(colors, int(host))
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _protrusion_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int]] = []
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
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents into hollow opposite-color patches and keep protrusions as teal markers.

    Input structure:
    - Black background with two large foreground-color regions.
    - The regions contain black dents, isolated foreground specks in the background,
      and a few one-cell protrusions where one foreground color pokes into the other.

    Rule:
    - First reconstruct the clean two-region baseline by removing isolated background
      specks and collapsing one-cell protrusions back into the surrounding host region.
    - Every original black dent whose cleaned 3x3 neighborhood is mostly one foreground
      color becomes a 3x3 patch in the opposite foreground color, but its center is then
      restored to black, producing a hollow patch.
    - Every original one-cell protrusion that cleanup rewrites into the other foreground
      region is kept only as a teal (8) point marker at its original location; it does
      not generate a 3x3 patch.

    Key insight:
    - Cleanup is only a baseline reconstruction step. Black input cells that belong to a
      foreground region become hollow opposite-color patches, while non-black input cells
      that cleanup erases as protrusions are reintroduced as teal markers. Teal therefore
      marks protrusions only, and black can mean either true background or a dent center.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    dents = _dent_centers(g, clean, colors, bg)
    protrusions = _protrusion_centers(g, clean, colors)
    h, w = g.shape

    for r, c, host in dents:
        other = _other_color(colors, int(host))
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c, _host in dents:
        out[r, c] = bg

    for r, c in protrusions:
        out[r, c] = Color.TEAL

    return out


# --------------- generator (combo cx) ---------------


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles using a simple painter's algorithm."""
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
    """After setting (r,c) to black, the solver must still detect a host-colored dent."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3X3 and host_count > int(np.sum(sub == other))


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
    """Optionally expand some dents into 2-cell orthogonal pairs."""
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


def _neighbor_protrusion_safe(
    clean: np.ndarray,
    r: int,
    c: int,
    source: int,
    host: int,
    bg: int,
) -> bool:
    """Reject candidates that would destabilize nearby source cells under cleanup."""
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
    host_count = int(np.sum(sub == host))
    if host_count <= own or host_count < 5:
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


def _build_output(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    protrusions: list[tuple[int, int, int]],
    color_a: int,
    color_b: int,
    bg: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = bg
    for r, c, _source in protrusions:
        output[r, c] = Color.TEAL
    return output


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_noise: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
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
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3X3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


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

    protrusion_positions = None
    for n_try in range(desired_hi, PROTRUSION_COUNT_MIN - 1, -1):
        protrusion_positions = _pick_separated_centers(
            [(r, c) for r, c, _source in protrusion_candidates],
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

    output = _build_output(clean, groups_a, groups_b, protrusions, color_a, color_b, bg)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = bg
    for r, c, source in protrusions:
        inp[r, c] = source

    n_noise = rng.randint(BG_NOISE_COUNT_MIN, BG_NOISE_COUNT_MAX)
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
