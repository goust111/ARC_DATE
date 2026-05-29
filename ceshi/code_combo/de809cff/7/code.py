# Combo variant of de809cff (index 7)
# combo_spec: D6/0+D15/0
# Name: 无预清洁+色斑保留为干扰项

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant:
# a) Concrete behavior and trace:
#    - Base task path:
#      1. Clean the raw grid: remove isolated foreground specks in black background
#         and snap one-cell boundary intrusions back to their host region.
#      2. Detect black dents on that cleaned grid by a raw-black / cleaned-neighbor
#         split: the center must be black in the input, but its cleaned 3x3 window
#         must have one foreground color appearing at least 5 times and beating the
#         other foreground color.
#      3. Start output from the cleaned grid, stamp a 3x3 opposite-color patch at
#         each dent center, then recolor the center teal.
#    - Fused variant path:
#      1. Skip cleaning entirely.
#      2. Detect dents directly on the raw 3x3 neighborhoods.
#      3. Start output from the raw input itself, so background specks and boundary
#         intrusions remain visible unless a dent patch later overwrites them.
#    - Small concrete example:
#      Suppose color A = 2, color B = 3, and a raw dent neighborhood is
#      [[2, 2, 2],
#       [2, 0, 3],
#       [2, 2, 2]].
#      The original solver would first flip the lone 3 back to 2, then detect the
#      center from the cleaned window. The fused solver does not repair that 3; it
#      still detects the dent because A appears 6 times and B appears once. The
#      output patch is therefore stamped on top of a still-noisy base image.
#      Separately, a background speck like
#      [[0, 0, 0],
#       [0, 2, 0],
#       [0, 0, 0]]
#      vanishes in the original task but stays as a distractor here unless a dent
#      patch covers it.
# b) Complexity calibration:
#    - The fusion only removes denoising; it does not add new color vocabularies,
#      new object types, or hidden indexing rules. That keeps the rule inferable.
#    - To make the change visible, the generator always includes both raw background
#      specks and raw boundary intrusions, and keeps several of them outside every
#      dent patch so the output visibly differs from the cleaned-base behavior.
# c) Hidden assumptions audit:
#    - `_main_colors` still assumes the two large regions dominate all noise cells.
#      Generator guarantees that by keeping region areas large and noise counts small.
#    - Raw dent detection now assumes true dents remain black and still have a unique
#      local majority despite nearby noise. Generator therefore places noise at least
#      two cells away from dent centers and rejects any sample where raw detection
#      gains or loses a dent.
#    - Output stamping now assumes the correct output background is the raw input,
#      not a normalized clean grid. The solver therefore copies the original grid.
# d) Placement legality:
#    - Rectangles are clipped to the grid bounds.
#    - Dent centers are chosen only on valid host cells and kept globally separated,
#      except for explicit 2-cell dent pairs whose two black cells are jointly
#      rechecked for raw detectability.
#    - Noise cells are placed on distinct cells, away from dent patches, and only on
#      background or boundary-adjacent foreground cells that match their intended role.
# e) Visual observability:
#    - Across 2-4 examples a human can see that some colored singletons in black
#      background and some one-cell cross-color intrusions survive into the output,
#      while black dents still trigger opposite-color 3x3 patches with teal centers.
#      No hidden numeric color ordering is used.
# f) Visual cleanliness:
#    - Noise counts stay moderate, and noise is spaced out from dents and from other
#      noise cells to avoid unreadable clutter.
#    - Primary dent centers are globally separated so most 3x3 stamps remain distinct.
# g) Change inferability:
#    - The only color transformation is still "dent host color -> opposite patch color
#      with teal center". The new information is structural: unlike the original task,
#      small off-structure color cells are not erased beforehand. The generator
#      explicitly checks that each sample differs from the legacy cleaned solver so the
#      combo does not collapse back to the base behavior.


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _sub5x5(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 2) : min(h, r + 3), max(0, c - 2) : min(w, c + 3)]


def _raw_dent_centers(g: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    color_a, color_b = map(int, colors)
    for r, c in zip(*np.where(g == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(g, rr, cc)
        count_a = int(np.sum(sub == color_a))
        count_b = int(np.sum(sub == color_b))
        if count_a >= 5 and count_a > count_b:
            centers.append((rr, cc, color_a))
        elif count_b >= 5 and count_b > count_a:
            centers.append((rr, cc, color_b))
    centers.sort(key=lambda item: (int(item[0]), int(item[1])))
    return centers


def _stamp_from_raw(g: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    color_a, color_b = map(int, colors)
    for r, c, host in centers:
        other = color_b if int(host) == color_a else color_a
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for r, c, _host in centers:
        out[r, c] = int(Color.TEAL)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform raw black dents directly, without first cleaning foreground noise.

    Input structure:
    - Black background with two dominant foreground-color regions created by large
      overlapping rectangles.
    - The regions contain true black dents, but the raw grid also keeps isolated
      foreground specks in background and one-cell cross-color intrusions near the
      boundary between the two regions.

    Rule:
    - Do not repair or erase any noise before reasoning.
    - Scan every black cell in the raw input. If one foreground color already forms a
      strict 3x3 majority of at least 5 cells around that black center, treat that
      black cell as a dent in that host region.
    - Copy the raw input to the output, stamp a 3x3 patch of the opposite foreground
      color around each detected dent, and recolor the dent center itself teal (8).

    Key insight: The meaningful black cells are the ones that already sit inside a
    strong local foreground majority on the uncleaned grid; all stray color specks and
    boundary blemishes are distractions that stay in place unless a dent patch covers them.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    centers = _raw_dent_centers(g, colors, bg)
    return _stamp_from_raw(g, centers, colors)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.48, 0.78
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84

# --- CFG: dents ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 20
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.10

# --- CFG: noise that must remain visible in the raw-rule output ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 8, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.55, 0.75
BG_SPECK_MIN_BG_3x3 = 7
NOISE_SEPARATION = 2
MIN_BOUNDARY_NOISE = 2
MIN_VISIBLE_BG_SPECKS = 2
MIN_VISIBLE_BOUNDARY_NOISE = 2

# --- CFG: rejection / metrics ---
LEGACY_DIFF_MIN = 4
MAX_GENERATION_ATTEMPTS = 500


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
    if not _boundary_candidates(grid, color_a, color_b):
        return None
    return grid


def _raw_safe_host_cell(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if all(max(abs(int(r) - rr), abs(int(c) - cc)) >= DENT_PATCH_SEPARATION for rr, cc in blocked + chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(
    roots: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    work = clean.copy()
    groups: list[list[tuple[int, int]]] = []
    reserved = list(blocked)
    for r, c in roots:
        work[r, c] = 0
        cells = [(int(r), int(c))]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < work.shape[0] and 0 <= nc < work.shape[1]):
                    continue
                if clean[nr, nc] != int(host):
                    continue
                if any(
                    max(abs(int(nr) - rr), abs(int(nc) - cc)) < DENT_PATCH_SEPARATION
                    for rr, cc in reserved
                    if (rr, cc) != (int(r), int(c))
                ):
                    continue
                probe = work.copy()
                probe[nr, nc] = 0
                if _raw_dent_centers(probe, (int(host), int(other)), 0):
                    if _raw_safe_black_center(probe, r, c, host, other) and _raw_safe_black_center(probe, nr, nc, host, other):
                        work = probe
                        cells.append((int(nr), int(nc)))
                        reserved.append((int(nr), int(nc)))
                        break
        groups.append(cells)
        reserved.append((int(r), int(c)))
    return groups


def _raw_safe_black_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != 0:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _boundary_candidates(clean: np.ndarray, color_a: int, color_b: int) -> list[tuple[int, int, int, int]]:
    h, w = clean.shape
    out: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            sub = _sub3x3(clean, r, c)
            if int(np.sum(sub == other)) >= 1:
                out.append((int(r), int(c), int(v), int(other)))
    return out


def _patch_mask(shape: tuple[int, int], centers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c, _host in centers:
        mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return mask


def _far_from_all(r: int, c: int, points: list[tuple[int, int]], min_cheb: int) -> bool:
    return all(max(abs(int(r) - rr), abs(int(c) - cc)) >= min_cheb for rr, cc in points)


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    dent_points: list[tuple[int, int]],
    occupied: list[tuple[int, int]],
    colors: tuple[int, int],
    n_target: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in zip(*np.where(clean == 0), strict=False):
        rr = int(r)
        cc = int(c)
        if not _far_from_all(rr, cc, dent_points, 2):
            continue
        if not _far_from_all(rr, cc, occupied, NOISE_SEPARATION):
            continue
        if int(np.sum(_sub3x3(clean, rr, cc) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        if not bool(np.any(_sub5x5(clean, rr, cc) != 0)):
            continue
        candidates.append((rr, cc))

    rng.shuffle(candidates)
    placed: list[tuple[int, int, int]] = []
    for r, c in candidates:
        if len(placed) >= n_target:
            break
        nearby = _sub5x5(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) in colors]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice(list(colors))
        inp[r, c] = int(speck_color)
        placed.append((int(r), int(c), int(speck_color)))
        occupied.append((int(r), int(c)))
    return placed


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    dent_points: list[tuple[int, int]],
    occupied: list[tuple[int, int]],
    color_a: int,
    color_b: int,
    n_target: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    candidates = _boundary_candidates(clean, color_a, color_b)
    rng.shuffle(candidates)
    placed: list[tuple[int, int, int, int]] = []
    for r, c, host, other in candidates:
        if len(placed) >= n_target:
            break
        if not _far_from_all(r, c, dent_points, 2):
            continue
        if not _far_from_all(r, c, occupied, NOISE_SEPARATION):
            continue
        if int(inp[r, c]) != int(host):
            continue
        inp[r, c] = int(other)
        placed.append((int(r), int(c), int(host), int(other)))
        occupied.append((int(r), int(c)))
    return placed


def _intended_centers(groups_a: list[list[tuple[int, int]]], groups_b: list[list[tuple[int, int]]], color_a: int, color_b: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for cells in groups_a:
        out.extend((int(r), int(c), int(color_a)) for r, c in cells)
    for cells in groups_b:
        out.extend((int(r), int(c), int(color_b)) for r, c in cells)
    out.sort(key=lambda item: (int(item[0]), int(item[1])))
    return out


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


def _legacy_clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()
    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = int(bg)
                    continue
                sub = _sub3x3(cur, r, c)
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == int(bg)))
                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = int(bg)
                elif own <= 2:
                    nxt[r, c] = int(bg)
                else:
                    nxt[r, c] = v
        cur = nxt
    return cur


def _legacy_dent_centers(original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    color_a, color_b = map(int, colors)
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc)
        count_a = int(np.sum(sub == color_a))
        count_b = int(np.sum(sub == color_b))
        if count_a >= 5 and count_a > count_b:
            centers.append((rr, cc, color_a))
        elif count_b >= 5 and count_b > count_a:
            centers.append((rr, cc, color_b))
    centers.sort(key=lambda item: (int(item[0]), int(item[1])))
    return centers


def _legacy_output(g: np.ndarray) -> np.ndarray:
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    clean = _legacy_clean_foreground(g, colors, bg)
    centers = _legacy_dent_centers(g, clean, colors, bg)
    return _stamp_from_raw(clean, centers, colors)


def _gen_attempt(rng: random.Random, *, return_meta: bool = False):
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    valid = {color_a: [], color_b: []}
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a) and _raw_safe_host_cell(clean, r, c, color_a, color_b):
                valid[color_a].append((int(r), int(c)))
            elif v == int(color_b) and _raw_safe_host_cell(clean, r, c, color_b, color_a):
                valid[color_b].append((int(r), int(c)))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    dents_a = _pick_dent_centers(valid[color_a], n_a, [], rng)
    if dents_a is None:
        return None
    dents_b = _pick_dent_centers(valid[color_b], n_b, dents_a, rng)
    if dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a + dents_b, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_a + dents_b, rng)
    centers = _intended_centers(groups_a, groups_b, color_a, color_b)
    dent_points = [(int(r), int(c)) for r, c, _host in centers]

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    n_boundary = max(MIN_BOUNDARY_NOISE, n_noise - n_bg)
    occupied: list[tuple[int, int]] = []
    bg_noise = _add_bg_noise(inp, clean, dent_points, occupied, (color_a, color_b), n_bg, rng)
    boundary_noise = _add_boundary_noise(inp, clean, dent_points, occupied, color_a, color_b, n_boundary, rng)
    if len(bg_noise) < MIN_VISIBLE_BG_SPECKS or len(boundary_noise) < MIN_VISIBLE_BOUNDARY_NOISE:
        return None

    detected = _raw_dent_centers(inp, (color_a, color_b), 0)
    if detected != centers:
        return None

    produced = main(inp)
    legacy = _legacy_output(inp)
    diff_vs_legacy = int(np.sum(produced != legacy))
    if diff_vs_legacy < LEGACY_DIFF_MIN:
        return None

    patch_mask = _patch_mask(inp.shape, centers)
    visible_bg = sum(
        1
        for r, c, _color in bg_noise
        if not bool(patch_mask[r, c]) and int(produced[r, c]) == int(inp[r, c]) and int(legacy[r, c]) != int(produced[r, c])
    )
    visible_boundary = sum(
        1
        for r, c, _host, _other in boundary_noise
        if not bool(patch_mask[r, c]) and int(produced[r, c]) == int(inp[r, c]) and int(legacy[r, c]) != int(produced[r, c])
    )
    if visible_bg < MIN_VISIBLE_BG_SPECKS or visible_boundary < MIN_VISIBLE_BOUNDARY_NOISE:
        return None

    meta = {
        "centers": centers,
        "bg_noise": bg_noise,
        "boundary_noise": boundary_noise,
        "diff_vs_legacy": diff_vs_legacy,
        "visible_bg": visible_bg,
        "visible_boundary": visible_boundary,
    }
    return (inp, meta) if return_meta else inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng, return_meta=False)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
