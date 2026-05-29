# Combo variant of de809cff (index 40)
# combo_spec: D3/0+D6/0+D14/2
# Name: 无预清洁+突出视为凹痕+4连通判定

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - The clean latent scene is still two large rectangular foreground regions on black,
#      but the solver no longer reconstructs that latent scene before acting.
#    - A black dent is now judged directly on the noisy input. Example: if a black cell sits
#      on a red/blue boundary and its orthogonal neighbors are red, red, blue, black, the
#      old task could ignore it because the cleaned 3x3 might not have a red 5-cell majority.
#      The combo variant still triggers because the 4-neighbor vote is red 2 vs blue 1.
#    - A one-cell cross-boundary protrusion is no longer erased. Example: a red cell poking
#      one step into a blue block, while still touching at least one red neighbor, is treated
#      as a trigger centered on that red cell; its 3x3 patch becomes red and the center teal.
#    - Unrelated background specks are not cleaned away. They simply stay in the output unless
#      a nearby trigger patch overwrites them.
#
# b) Complexity calibration
#    - The generator keeps many ordinary dents so the family remains recognizable, but it also
#      forces at least one orth-only dent whose 4-neighbor vote succeeds while the old 3x3 test
#      would fail. This makes the D3 change visible rather than theoretical.
#    - 3-6 protrusions are added per grid, enough to show that boundary bulges now generate
#      teal-centered patches, without letting the boundary dissolve into overlapping marks.
#    - Background specks are sparse and placed away from trigger patches so "no cleanup" is
#      visually legible as preserved noise rather than incidental patch overwrite.
#
# c) Implicit assumptions audit
#    - Base solver assumption "output starts from cleaned foreground" is invalid here; output
#      must start from the raw input because D6/0 preserves non-trigger noise.
#    - Base dent detector assumes black cells are judged after cleanup and by a 3x3 majority.
#      Both parts change: we use the raw grid and only the 4 orthogonal neighbors.
#    - D14/2 needs a protrusion detector that does not rely on cleanup rewriting the center.
#      We instead require a foreground center whose orthogonal majority belongs to the other
#      foreground color, plus at least one same-color orthogonal attachment and no same-color 2x2.
#    - To avoid accidental triggers from arbitrary edge-contact noise, the 4-neighbor winner
#      must appear at least twice. The generator explicitly guarantees this for all intended
#      triggers.
#
# d) Placement legality
#    - Trigger centers are chosen with Chebyshev separation >= 3, so distinct 3x3 patches do
#      not overlap except for deliberate 2-cell dent pairs inside one group.
#    - Background specks are placed only on clean background cells with all four orthogonal
#      neighbors also background, keeping them isolated and non-triggering.
#    - Protrusions are inserted only where the clean host color would still win the orthogonal
#      vote after insertion, so the solver's host inference is well-defined.
#
# e) Visual observability
#    - Across examples, the human-visible signature is: some teal-centered patches originate at
#      non-black boundary cells, and some black dents on messy boundaries still fire even though
#      diagonal evidence points the other way. Both cues are visible from local geometry alone.
#    - No hidden ordering or numeric color-id trick is used; every decision is traceable from
#      immediate orthogonal context and from whether the center is a hole or a one-cell bulge.
#
# f) Visual cleanliness
#    - Trigger neighborhoods are separated and background specks are kept off those 3x3 windows,
#      so the resulting grids remain readable despite the added noise.
#    - Protrusions are limited to single-cell bulges rather than thicker incursions, preventing
#      ambiguous boundary slabs.
#
# g) Color-change inferability
#    - There are still only two foreground colors. After identifying the host side via the
#      orthogonal vote, the 3x3 patch always uses the other foreground color and the center
#      becomes teal.
#    - For a protrusion, that "other foreground color" is exactly the protrusion's own color,
#      which gives repeated boundary examples of the same rule from a different trigger type.


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


def _orth_color_counts(grid: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> dict[int, int]:
    h, w = grid.shape
    counts = {int(colors[0]): 0, int(colors[1]): 0}
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(grid[nr, nc])
            if v in counts:
                counts[v] += 1
    return counts


def _orth_majority_host(grid: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    counts = _orth_color_counts(grid, r, c, colors)
    a, b = map(int, colors)
    if counts[a] >= 2 and counts[a] > counts[b]:
        return a
    if counts[b] >= 2 and counts[b] > counts[a]:
        return b
    return None


def _other_color(color: int, colors: tuple[int, int]) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


def _dent_centers(original: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _orth_majority_host(original, rr, cc, colors)
        if host is not None:
            centers.append((rr, cc, int(host)))
    return centers


def _protrusion_centers(original: np.ndarray, colors: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(original[r, c])
            if v not in colors or _has_same_2x2(original, r, c, v):
                continue
            host = _orth_majority_host(original, r, c, colors)
            if host is None or int(host) == v:
                continue
            same_touch = _orth_color_counts(original, r, c, colors)[v]
            if same_touch >= 1:
                centers.append((r, c, int(host)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform raw dents and one-cell protrusions directly on the noisy grid.

    Input structure:
    - Black background with two large foreground regions formed by overlapping rectangles.
    - The raw grid may contain black dents, isolated foreground specks in background, and
      one-cell cross-boundary protrusions that remain visible instead of being pre-cleaned.

    Rule:
    - Skip all pre-cleaning and start from the original input.
    - Any black cell whose four orthogonal neighbors give a strict foreground-color majority
      is treated as a dent in that host region; stamp its 3x3 neighborhood with the other
      foreground color.
    - Any foreground cell whose orthogonal majority belongs to the other foreground color,
      while still touching at least one same-color orthogonal neighbor, is treated as a
      protrusion trigger and stamped by the same opposite-side rule. Every trigger center
      is finally recolored teal (8).

    Key insight:
    - The task is solved in the noisy image itself. Diagonals no longer decide ownership,
      preserved specks stay unless overwritten, and both black holes and single-cell bulges
      become teal-centered 3x3 patches through the same 4-neighbor host vote.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)

    out = g.copy()
    dents = _dent_centers(g, colors, bg)
    protrusions = _protrusion_centers(g, colors)
    h, w = g.shape

    for r, c, host in dents:
        stamp_color = _other_color(host, colors)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)

    for r, c, host in protrusions:
        stamp_color = _other_color(host, colors)
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

# --- CFG: dents & protrusions ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 20
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13
MIN_SPECIAL_DENTS = 1
PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX = 3, 6

# --- CFG: preserved noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 14
BG_SPECK_MIN_BG_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 900


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


def _legacy_black_host_3x3(clean: np.ndarray, r: int, c: int, host: int, other: int) -> int | None:
    sub = _sub3x3(clean, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    if host_count >= 5 and host_count > other_count:
        return int(host)
    if other_count >= 5 and other_count > host_count:
        return int(other)
    return None


def _is_safe_dent_center(clean: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(clean[r, c]) != int(host):
        return False
    counts = _orth_color_counts(clean, r, c, (host, other))
    return counts[int(host)] >= 2 and counts[int(host)] > counts[int(other)]


def _is_special_dent_center(clean: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    return _is_safe_dent_center(clean, r, c, host, other) and _legacy_black_host_3x3(clean, r, c, host, other) != int(host)


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
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked = list(occupied)
    for r, c in pool:
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
                cells.append((nr, nc))
                occupied.append((nr, nc))
                break
        result.append(cells)
    return result


def _safe_protrusion_candidate(clean: np.ndarray, r: int, c: int, source: int, host: int) -> bool:
    if int(clean[r, c]) != int(host):
        return False
    if _would_form_2x2(clean, r, c, source):
        return False
    counts = _orth_color_counts(clean, r, c, (source, host))
    same_touch = counts[int(source)]
    host_touch = counts[int(host)]
    return same_touch >= 1 and host_touch >= 2 and host_touch > same_touch


def _find_protrusion_candidates(
    clean: np.ndarray,
    colors: tuple[int, int],
) -> list[tuple[int, int, int]]:
    color_a, color_b = map(int, colors)
    h, w = clean.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host == color_a and _safe_protrusion_candidate(clean, r, c, color_b, color_a):
                candidates.append((r, c, color_b))
            elif host == color_b and _safe_protrusion_candidate(clean, r, c, color_a, color_b):
                candidates.append((r, c, color_a))
    return candidates


def _stamp_triggers(
    inp: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    protrusions: list[tuple[int, int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = inp.shape
    output = inp.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for r, c, stamp_color in protrusions:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    for r, c, _ in protrusions:
        output[r, c] = Color.TEAL
    return output


def _forbidden_patch_cells(
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    protrusions: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    forbidden: set[tuple[int, int]] = set()
    for cells in dent_groups_a + dent_groups_b:
        for r, c in cells:
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    forbidden.add((rr, cc))
    for r, c, _ in protrusions:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                forbidden.add((rr, cc))
    return forbidden


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_noise: int,
    forbidden: set[tuple[int, int]],
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> int:
    h, w = inp.shape
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    used: list[tuple[int, int]] = []
    for r_, c_ in bg_candidates:
        if placed >= n_noise:
            break
        r, c = int(r_), int(c_)
        if (r, c) in forbidden:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
            continue
        if any(
            not (0 <= nr < h and 0 <= nc < w) or int(clean[nr, nc]) != 0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            continue
        if int(np.sum(_sub3x3(clean, r, c) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue

        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = int(speck_color)
        used.append((r, c))
        placed += 1

    return placed


def _choose_special_seed(
    specials_a: list[tuple[int, int]],
    specials_b: list[tuple[int, int]],
    n_a: int,
    n_b: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> tuple[int, int, int] | None:
    pool: list[tuple[int, int, int]] = []
    if n_a > 0:
        pool.extend((r, c, color_a) for r, c in specials_a)
    if n_b > 0:
        pool.extend((r, c, color_b) for r, c in specials_b)
    if not pool:
        return None
    rng.shuffle(pool)
    return pool[0]


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
    valid_a: list[tuple[int, int]] = []
    valid_b: list[tuple[int, int]] = []
    specials_a: list[tuple[int, int]] = []
    specials_b: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid_a.append((r, c))
                if _is_special_dent_center(clean, r, c, color_a, color_b):
                    specials_a.append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid_b.append((r, c))
                if _is_special_dent_center(clean, r, c, color_b, color_a):
                    specials_b.append((r, c))

    if len(valid_a) < n_a + 1 or len(valid_b) < n_b + 1:
        return None
    if len(specials_a) + len(specials_b) < MIN_SPECIAL_DENTS:
        return None

    special_seed = _choose_special_seed(
        specials_a, specials_b, n_a, n_b, color_a, color_b, rng
    )
    if special_seed is None:
        return None

    seed_r, seed_c, seed_color = special_seed
    occupied = [(seed_r, seed_c)]
    dents_a: list[tuple[int, int]] = []
    dents_b: list[tuple[int, int]] = []
    if seed_color == color_a:
        dents_a.append((seed_r, seed_c))
        n_a -= 1
    else:
        dents_b.append((seed_r, seed_c))
        n_b -= 1

    pool_a = [pos for pos in valid_a if pos != (seed_r, seed_c)]
    pool_b = [pos for pos in valid_b if pos != (seed_r, seed_c)]
    extra_a = _pick_separated_centers(pool_a, n_a, occupied, rng) if n_a > 0 else []
    occupied.extend(extra_a or [])
    extra_b = _pick_separated_centers(pool_b, n_b, occupied, rng) if n_b > 0 else []
    if (n_a > 0 and extra_a is None) or (n_b > 0 and extra_b is None):
        return None
    dents_a.extend(extra_a or [])
    dents_b.extend(extra_b or [])

    occupied = list(dents_a) + list(dents_b)
    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, occupied, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, occupied, rng)
    occupied = [cell for group in groups_a + groups_b for cell in group]

    protrusion_candidates = _find_protrusion_candidates(clean, (color_a, color_b))
    desired_hi = min(PROTRUSION_COUNT_MAX, len(protrusion_candidates))
    if desired_hi < PROTRUSION_COUNT_MIN:
        return None
    target = rng.randint(PROTRUSION_COUNT_MIN, desired_hi)
    chosen_positions = None
    for n_try in range(target, PROTRUSION_COUNT_MIN - 1, -1):
        chosen_positions = _pick_separated_centers(
            [(r, c) for r, c, _ in protrusion_candidates],
            n_try,
            occupied,
            rng,
        )
        if chosen_positions is not None:
            break
    if chosen_positions is None:
        return None
    chosen_set = set(chosen_positions)
    protrusions = [(r, c, source) for r, c, source in protrusion_candidates if (r, c) in chosen_set]
    if len(protrusions) != len(chosen_set):
        return None

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK
    for r, c, source in protrusions:
        inp[r, c] = int(source)

    forbidden = _forbidden_patch_cells(groups_a, groups_b, protrusions, inp.shape)
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    placed_noise = _add_bg_noise(inp, clean, n_noise, forbidden, color_a, color_b, rng)
    if placed_noise < n_noise:
        return None

    output = _stamp_triggers(inp, groups_a, groups_b, protrusions, color_a, color_b)

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
