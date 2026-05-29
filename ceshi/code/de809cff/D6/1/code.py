# Variant of de809cff
# Dimension: D6 — 预清洁步骤：孤立色斑与边界突出的处理 (multi_step)
# Change: 变换前先清洁前景区域：移除背景/边缘区域中的孤立前景色斑点（single-cell specks），将单像素边界突出（protrusion）重着色为周围区域色
#         → 清洁后保留色斑痕迹（teal标记）
#
# Design analysis for the D6 variant
# a) Concrete behavior and trace example
#    - Input still shows two large foreground regions on black, plus black dents and a mix of one-cell corruptions.
#    - The new part is only about cells that cleanup would delete to black: those single-cell specks do not vanish in the output.
#      They leave a teal point behind at the same coordinate.
#    - Cross-color boundary protrusions still behave like before: they are recolored into the surrounding region, not marked teal.
#    - Small example:
#        original logic: a lone red pixel in black background -> cleanup turns it black -> output has no mark there.
#        new logic: the same lone red pixel -> cleanup still removes it for dent reasoning -> final output writes teal at that cell.
#        difference: teal is now polysemous, because some teal cells are dent centers inside opposite-color 3x3 patches,
#        while some are standalone speck traces outside those patches.
# b) Complexity calibration
#    - The task remains visually close to the source task: same rectangles, same dents, same boundary noise.
#    - To make D6 observable, the generator forces several removable background specks per sample and keeps them away from dent
#      patches so a human can notice the second teal role from 2-4 examples.
#    - The rule stays low-branching: teal has only two structural roles, distinguishable by context.
# c) Hidden assumptions review
#    - _main_colors assumed the input contains exactly the two non-black foreground colors. This still holds because the
#      generator never places teal in the input.
#    - The original cleanup directly overwrote removable specks with black. That was harmless before because deleted cells left
#      no visible output state. In this variant that would lose information, so the solver first records which original cells
#      qualify as delete-to-black specks, then performs the same cleanup.
#    - Dent detection relies on original black cells only. Speck traces therefore must not be inferred from black cells; they
#      are recorded from original foreground cells that the cleanup rule would delete to black.
# d) Placement legality
#    - Dent centers are chosen exactly as in the original task, with spacing so 3x3 opposite-color patches can overlap only in
#      the intended controlled way.
#    - Trace specks are placed only on black background cells, with bg-dominant 3x3 neighborhoods and a distance from any dent
#      patch so they remain isolated and are guaranteed to be cleanup-to-black cases.
#    - Boundary contamination is still placed only where the solver's local majority rule restores the host color.
# e) Visual observability
#    - The new behavior is readable because standalone teal points appear in black background / outer-edge areas, whereas dent
#      teal points sit at the centers of opposite-color 3x3 patches inside or touching the big regions.
#    - No invisible ordering or color-id arithmetic is used; all distinctions are local geometric cues.
# f) Visual cleanliness
#    - The generator avoids placing trace specks inside dent patches and keeps trace specks separated from each other, so the
#      output does not become a dense teal haze.
# g) Color inference
#    - Only two foreground colors plus black/teal are used. Teal is always a marker color, never a region color.
#      Its two meanings are disambiguated by whether it is embedded in a 3x3 opposite-color patch or stands alone.
#
# New solver assumptions checked against generator
# - The input has exactly two foreground colors and no teal: guaranteed by color sampling.
# - Every removable speck that should leave a trace already satisfies the delete-to-black rule on the original grid:
#   guaranteed by trace-speck placement in black background.
# - Dent centers stay black in the input and never overlap trace specks: guaranteed by constructing dents before specks and
#   only adding trace specks on background cells outside all dent patches.

from __future__ import annotations

from collections import Counter as _Counter
import random

import numpy as np

from arc_utils import Color


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


def _cleanup_decision(g: np.ndarray, r: int, c: int, colors: tuple[int, int], bg: int) -> tuple[int, bool]:
    v = int(g[r, c])
    if v not in colors:
        return int(bg), False

    a, b = map(int, colors)
    sub = g[max(0, r - 1) : min(g.shape[0], r + 2), max(0, c - 1) : min(g.shape[1], c + 2)]
    own = int(np.sum(sub == v))
    other = b if v == a else a
    other_count = int(np.sum(sub == other))
    bg_count = int(np.sum(sub == bg))

    if _has_same_2x2(g, r, c, v):
        return v, False
    if other_count > own and other_count >= 5:
        return other, False
    if bg_count >= 5 and own <= 4:
        return int(bg), True
    if own <= 2:
        return int(bg), True
    return v, False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Apply the original two-pass local cleanup before dent detection.

    The D6 variant keeps the cleanup dynamics unchanged; only the final rendering
    of cells deleted to background is different.
    """
    h, w = g.shape
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                new_v, _mark_trace = _cleanup_decision(cur, r, c, colors, bg)
                nxt[r, c] = int(new_v)
        cur = nxt

    return cur


def _speck_traces(g: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int]]:
    """
    Record original one-cell foreground specks that cleanup would delete to black.

    These positions are the new D6-only teal trace markers. They are measured on
    the original input so boundary contaminants that should recolor into a region
    are not mistaken for traces.
    """
    traces: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            new_v, mark_trace = _cleanup_decision(g, r, c, colors, bg)
            if mark_trace and int(new_v) == int(bg):
                traces.append((r, c))
    return traces


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents into opposite-color patches while preserving deleted speck traces.

    Input structure:
    - Black background with two large foreground regions in distinct non-teal colors.
    - The regions contain black dents plus one-cell corruption: some cells are isolated
      foreground specks in background/outer-edge areas, others are cross-color boundary protrusions.

    Rule:
    - First interpret the foreground with the same local cleanup as the source task:
      removable isolated specks are treated as background, while boundary protrusions are
      recolored into the locally dominant neighboring region.
    - Every original black dent whose cleaned 3x3 neighborhood is mostly one foreground
      color receives a 3x3 patch of the opposite foreground color.
    - Teal (8) is written in two situations: at each dent center, and at each original
      single-cell speck that the cleanup deleted to black.

    Key insight:
    - Teal no longer means only "dent center". A teal cell inside an opposite-color 3x3 patch
      marks a dent, while a lone teal point in background/outer-edge space is the footprint of
      a removed pre-cleaning speck.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    traces = _speck_traces(g, colors, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c in traces:
        out[r, c] = Color.TEAL

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
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: cleanup-noise placement ---
TRACE_SPECK_COUNT_MIN, TRACE_SPECK_COUNT_MAX = 4, 9
BOUNDARY_NOISE_COUNT_MIN, BOUNDARY_NOISE_COUNT_MAX = 2, 8
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6
TRACE_SPECK_SEPARATION = 2

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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
    if grid[r, c] != host:
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
                if clean[nr, nc] != host or not _is_safe_dent_center(clean, nr, nc, host, other):
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
    for dent_groups, stamp_color in [(dent_groups_a, color_b), (dent_groups_b, color_a)]:
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
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


def _add_trace_specks(
    inp: np.ndarray,
    clean: np.ndarray,
    forbidden: np.ndarray,
    n_trace: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    bg_candidates = list(zip(*np.where((clean == 0) & (~forbidden)), strict=False))
    rng.shuffle(bg_candidates)

    traces: list[tuple[int, int]] = []
    for r_, c_ in bg_candidates:
        if len(traces) >= n_trace:
            break
        r, c = int(r_), int(c_)
        if inp[r, c] != 0:
            continue
        if any(max(abs(r - rr), abs(c - cc)) < TRACE_SPECK_SEPARATION for rr, cc in traces):
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        if not nearby_fg:
            continue
        speck_color = _Counter(nearby_fg).most_common(1)[0][0]
        inp[r, c] = speck_color
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
) -> bool:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
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
    return placed >= n_at_boundary


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

    dents_a = _pick_dent_centers(valid[color_a], n_a, rng)
    dents_b = _pick_dent_centers(valid[color_b], n_b, rng)
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
    traces = _add_trace_specks(inp, clean, dent_mask, n_trace, color_a, color_b, rng)
    if traces is None:
        return None

    n_boundary = rng.randint(BOUNDARY_NOISE_COUNT_MIN, BOUNDARY_NOISE_COUNT_MAX)
    if not _add_boundary_noise(inp, clean, n_boundary, color_a, color_b, rng):
        return None

    for r, c in traces:
        output[r, c] = Color.TEAL

    if len(_speck_traces(inp, (color_a, color_b), Color.BLACK)) < n_trace:
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
