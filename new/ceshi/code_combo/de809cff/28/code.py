# Combo variant of de809cff (index 28)
# combo_spec: D6/1+D14/0
# Name: 突出标记teal+清洁后色斑也标teal
#
# Design analysis for the fused variant
# a) Concrete behavior and trace example
#    - The base task has two large foreground regions on black, black dents inside them, isolated foreground specks in black
#      background, and one-cell cross-boundary contaminants near the interface between the two regions.
#    - Original behavior:
#        lone foreground speck in black -> cleanup deletes it to black -> no visible trace in output.
#        one wrong-color boundary contaminant inside the other region -> cleanup recolors it into the surrounding region.
#        black dent -> stamp a 3x3 patch of the opposite foreground color and mark the center teal.
#    - Fused behavior:
#        the speck is still deleted for reasoning, but its output cell becomes teal.
#        the boundary contaminant is still conceptually repaired for dent reasoning, but its output cell becomes teal instead
#        of blending back into the surrounding region.
#        the dent center remains teal inside an opposite-color 3x3 patch.
#    - So teal now has three observable roles:
#        1. dent center inside an opposite-color 3x3 patch,
#        2. standalone teal trace in black background where a speck was removed,
#        3. teal point sitting on a region boundary where a one-cell protrusion was repaired.
# b) Complexity calibration
#    - The variant stays close to the source task: the large rectangles, dent logic, and local cleanup dynamics are unchanged.
#    - Complexity is added only by making cleanup visible in two different ways. The generator forces several background specks
#      and several boundary protrusions per sample so both new teal roles appear repeatedly instead of as one-off accidents.
#    - The branch count is still low: every teal pixel belongs to one of three spatially distinct contexts.
# c) Hidden assumptions review
#    - The original solver assumed cleanup side-effects were disposable. That is no longer true, so the new solver records
#      cleanup-marker coordinates from the original input before running the same two cleanup sweeps for dent reasoning.
#    - Dent detection still depends only on original black cells. Therefore cleanup markers must be collected from original
#      foreground cells, not from the cleaned grid.
#    - The original boundary-noise branch recolored a wrong-color cell into the opposite foreground color. In the fused rule,
#      those cells still need that repaired color internally so dent neighborhoods are judged on the cleaned grid, but the
#      final rendering overwrites their coordinates with teal.
#    - Delete-to-black cells are reserved for the D6 trace role; recolor-to-other cells are reserved for the D14 protrusion
#      role. This keeps the two cleanup-derived teal semantics disjoint.
# d) Placement legality
#    - Dent centers are sampled from safe host-dominant cells and spaced apart so their 3x3 patches only overlap in the
#      intended controlled way.
#    - Trace specks are placed on black background cells with black-dominant 3x3 neighborhoods and outside every dent patch.
#    - Protrusions are placed on genuine region-boundary cells, also outside dent patches and away from trace specks, and only
#      where the cleanup rule would definitely classify them as recolor-to-other repairs.
# e) Visual observability
#    - From a few examples, a human can see that some teal cells sit at patch centers, some are isolated in black background,
#      and some lie on the seam between the two large colored regions. That geometric context reveals the three roles without
#      relying on color ids or any invisible ordering.
# f) Visual cleanliness
#    - The generator separates cleanup markers from dent patches and lightly spaces cleanup markers from one another, which
#      prevents dense teal clusters that would hide the role distinctions.
#    - Counts are kept moderate so the two large foreground regions remain visually dominant and recognizable.
# g) Color inference
#    - Only two foreground colors plus black and teal are used. Teal is always a marker color and never a region color.
#      Its meaning is inferred from local structure: patch-centered, background-isolated, or boundary-embedded.
#
# New solver assumptions checked against generator
# - The input contains exactly two non-black, non-teal foreground colors: guaranteed by color sampling.
# - Every intended protrusion is a one-cell wrong-color boundary contaminant that the local cleanup would recolor into the
#   opposite foreground color: guaranteed by placement checks using the same cleanup classifier as the solver.
# - Every intended trace speck is a delete-to-black case on the original input and is kept away from dent patches and from
#   protrusion sites: guaranteed by candidate filtering and post-placement verification.

from __future__ import annotations

from collections import Counter as _Counter
import random

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) not in (int(bg), int(Color.TEAL))]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


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


def _cleanup_decision(
    g: np.ndarray, r: int, c: int, colors: tuple[int, int], bg: int
) -> tuple[int, str]:
    """
    Return the internally cleaned color and the visible marker role, if any.

    The fused rule keeps the original local cleanup thresholds. Cells that would
    be recolored into the opposite foreground region become D14 protrusion marks.
    Cells that would be deleted to background become D6 speck-trace marks.
    """
    v = int(g[r, c])
    if v not in colors:
        return int(bg), ""

    other = _other_color(colors, v)
    sub = _sub3x3(g, r, c)
    own = int(np.sum(sub == v))
    other_count = int(np.sum(sub == other))
    bg_count = int(np.sum(sub == bg))

    if _has_same_2x2(g, r, c, v):
        return v, ""
    if other_count > own and other_count >= 5:
        return int(other), "protrusion"
    if bg_count >= 5 and own <= 4:
        return int(bg), "speck"
    if own <= 2:
        return int(bg), "speck"
    return v, ""


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Apply the original two-pass cleanup before evaluating dent neighborhoods.

    The fused variant changes only the final rendering of repaired cells, not the
    internal cleanup dynamics used to infer host regions.
    """
    h, w = g.shape
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                new_v, _marker = _cleanup_decision(cur, r, c, colors, bg)
                nxt[r, c] = int(new_v)
        cur = nxt

    return cur


def _cleanup_markers(
    g: np.ndarray, colors: tuple[int, int], bg: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    specks: list[tuple[int, int]] = []
    protrusions: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            _new_v, marker = _cleanup_decision(g, r, c, colors, bg)
            if marker == "speck":
                specks.append((r, c))
            elif marker == "protrusion":
                protrusions.append((r, c))
    return specks, protrusions


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
        other = _other_color(colors, host)
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents into opposite-color patches while keeping both kinds of cleanup marks visible.

    Input structure:
    - Black background with two large foreground regions in distinct non-teal colors.
    - Inside and around those regions are three kinds of defects: black dents inside the regions,
      isolated one-cell foreground specks in background areas, and one-cell wrong-color protrusions
      along the boundary where the two foreground regions meet.

    Rule:
    - First interpret the picture through the same two-pass local cleanup as the source task:
      removable background specks are treated as black, and wrong-color boundary protrusions are
      internally repaired into the surrounding foreground region.
    - Every original black cell whose cleaned 3x3 neighborhood is dominated by one foreground color
      becomes the center of a 3x3 patch in the opposite foreground color.
    - Teal (8) is then written in three situations: dent centers, deleted speck locations, and
      repaired protrusion locations.

    Key insight:
    - Teal is now polysemous but still locally readable. A teal cell inside an opposite-color 3x3 patch
      is a dent center; a lone teal point in black background is a removed speck trace; a teal point
      on the seam between the two large regions marks a repaired protrusion.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    specks, protrusions = _cleanup_markers(g, colors, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host in centers:
        other = _other_color(colors, host)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c in specks:
        out[r, c] = int(Color.TEAL)
    for r, c in protrusions:
        out[r, c] = int(Color.TEAL)
    for r, c, _host in centers:
        out[r, c] = int(Color.TEAL)

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

# --- CFG: cleanup-marker placement ---
TRACE_SPECK_COUNT_MIN, TRACE_SPECK_COUNT_MAX = 4, 9
PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX = 3, 7
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6
MARKER_SEPARATION = 2

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


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
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _pick_dent_centers(
    candidates: list[tuple[int, int]], n: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
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
                if int(clean[nr, nc]) != int(host) or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in all_dents
                    if (r2, c2) != (r, c)
                ):
                    cells.append((nr, nc))
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
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = int(Color.TEAL)
    return output


def _dent_patch_mask(
    shape: tuple[int, int],
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return mask


def _far_from_existing(r: int, c: int, existing: list[tuple[int, int]], min_sep: int) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in existing)


def _add_trace_specks(
    inp: np.ndarray,
    clean: np.ndarray,
    forbidden: np.ndarray,
    n_trace: int,
    colors: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    bg_candidates = list(zip(*np.where((clean == 0) & (~forbidden)), strict=False))
    rng.shuffle(bg_candidates)

    traces: list[tuple[int, int]] = []
    for r_, c_ in bg_candidates:
        if len(traces) >= n_trace:
            break
        r = int(r_)
        c = int(c_)
        if int(inp[r, c]) != 0 or not _far_from_existing(r, c, traces, MARKER_SEPARATION):
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby_fg = [int(x) for x in _sub3x3(clean, r, c).flat if int(x) != 0]
        if not nearby_fg:
            continue
        speck_color = _Counter(nearby_fg).most_common(1)[0][0]
        inp[r, c] = int(speck_color)
        new_v, marker = _cleanup_decision(inp, r, c, colors, Color.BLACK)
        if int(new_v) != int(Color.BLACK) or marker != "speck":
            inp[r, c] = 0
            continue
        traces.append((r, c))

    if len(traces) < n_trace:
        return None
    return traces


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0 = max(0, nr - 1)
            nc0 = max(0, nc - 1)
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or (bg_n >= 5 and own_n <= 4) or own_n <= 2:
                return False
    return True


def _add_protrusion_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    forbidden: np.ndarray,
    n_protrusions: int,
    colors: tuple[int, int],
    existing_markers: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host not in colors or bool(forbidden[r, c]):
                continue
            other = _other_color(colors, host)
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((r, c, host, other))
    rng.shuffle(boundary_candidates)

    protrusions: list[tuple[int, int]] = []
    occupied = list(existing_markers)
    for r, c, host, other in boundary_candidates:
        if len(protrusions) >= n_protrusions:
            break
        if int(inp[r, c]) != int(host) or not _far_from_existing(r, c, occupied, MARKER_SEPARATION):
            continue

        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == int(host)))
        other_count = int(np.sum(sub_now == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue

        inp[r, c] = int(other)
        new_v, marker = _cleanup_decision(inp, r, c, colors, Color.BLACK)
        if int(new_v) != int(host) or marker != "protrusion":
            inp[r, c] = int(host)
            continue
        protrusions.append((r, c))
        occupied.append((r, c))

    if len(protrusions) < n_protrusions:
        return None
    return protrusions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a = n_total // 2
    n_b = n_total - n_a
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    h, w = clean.shape
    valid = {int(color_a): [], int(color_b): []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a) and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[int(color_a)].append((r, c))
            elif v == int(color_b) and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[int(color_b)].append((r, c))

    if len(valid[int(color_a)]) < n_a + 2 or len(valid[int(color_b)]) < n_b + 2:
        return None

    dents_a = _pick_dent_centers(valid[int(color_a)], n_a, rng)
    dents_b = _pick_dent_centers(valid[int(color_b)], n_b, rng)
    if dents_a is None or dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_b, rng)

    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b)
    dent_mask = _dent_patch_mask(clean.shape, groups_a, groups_b)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_trace = rng.randint(TRACE_SPECK_COUNT_MIN, TRACE_SPECK_COUNT_MAX)
    traces = _add_trace_specks(inp, clean, dent_mask, n_trace, colors, rng)
    if traces is None:
        return None

    n_protrusions = rng.randint(PROTRUSION_COUNT_MIN, PROTRUSION_COUNT_MAX)
    protrusions = _add_protrusion_noise(inp, clean, dent_mask, n_protrusions, colors, traces, rng)
    if protrusions is None:
        return None

    for r, c in traces:
        output[r, c] = int(Color.TEAL)
    for r, c in protrusions:
        output[r, c] = int(Color.TEAL)

    speck_marks, protrusion_marks = _cleanup_markers(inp, colors, Color.BLACK)
    if set(speck_marks) != set(traces):
        return None
    if set(protrusion_marks) != set(protrusions):
        return None

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
