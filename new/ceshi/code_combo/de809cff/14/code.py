# Combo variant of de809cff (index 14)
# combo_spec: D6/1+D15/2
# Name: 清洁后色斑标teal+色斑颜色指示补丁中心色
#
# Design analysis for the combo variant
# a) Concrete behavior and trace example
#    - The source task has two large foreground regions, black dents inside them, removable one-cell background specks,
#      and removable one-cell boundary contamination.
#    - This combo changes two stages together. Each removable background speck is now an auxiliary-color memory token:
#      after cleanup it leaves a teal trace at its own coordinate, but its original input color is reused later as the
#      center color for every dent patch in the nearby host region.
#    - Small trace:
#        source logic: a black-hole dent inside region A -> opposite-color 3x3 patch with teal center.
#        D6 part: a lone marker speck in background -> deleted by cleanup but visible as a teal point in output.
#        D15 part in the combo: if that speck originally had color 7 and sat next to region A, then every dent hosted by
#        region A gets center color 7 instead of teal.
#        difference: the solver must remember a disappearing input color, bind it to a region, then use that memory after
#        cleanup when stamping dent centers.
# b) Complexity calibration
#    - The mapping stays low-branching: exactly two large regions and one center-color code per region.
#    - Each sample repeats the same region->center-color relation across several dents and several marker specks, so the
#      rule is visible from a few examples rather than hinging on a single cell.
#    - The auxiliary colors are limited to two sparse colors, distinct from the two region colors, black, and teal.
# c) Hidden assumptions review
#    - The original solver assumed every non-black color belonged to one of the two main regions. That no longer holds.
#      `_main_colors` is therefore relied on only as an area-based detector of the two dominant region colors; the generator
#      guarantees the auxiliary marker colors stay sparse and cannot outrank the main regions.
#    - The original cleanup could safely discard removable specks because no later step needed their colors. That is false
#      here, so the solver records every removable non-region speck before cleanup.
#    - The original teal-center rule is removed. Dent centers now depend on a region->marker-color map built from the trace
#      specks, so the solver must compute traces first, cleanup second, mapping third, and dent stamping last.
#    - Region binding for a trace assumes the cleaned 5x5 neighborhood around that trace has a unique dominant main-region
#      color. The generator explicitly places marker specks only where this is true.
# d) Placement legality
#    - Dent centers use the same safety checks and spacing as the source task, so opposite-color 3x3 patches remain valid.
#    - Marker specks are placed only on black cells outside every dent patch, with a minimum separation from one another.
#    - Boundary contamination is still placed only where the two-pass cleanup deterministically restores the host region.
# e) Visual observability
#    - The combo is readable because the input shows two sparse auxiliary colors near different regions; in the output those
#      same positions become teal traces, while dent centers inside one region consistently adopt one auxiliary color and dent
#      centers in the other region adopt the other auxiliary color.
#    - The rule uses only visible geometry and repeated color correspondences. No ordering by raw color id is required.
# f) Visual cleanliness
#    - Marker specks are kept away from dent patches and from each other, so teal traces stay isolated rather than merging
#      with the dent patches.
#    - Only one extra center color is active per region, which avoids noisy many-to-many color changes inside the same area.
# g) Change inferability
#    - The color change is structurally supported: auxiliary color in input background -> teal trace at that same coordinate
#      in output, and the same auxiliary color reappears as dent centers in the adjacent region.
#    - Because each region has multiple dents and multiple traces, the region->center-color mapping can be inferred from
#      repeated evidence rather than guessed from a singleton.
#
# New solver assumptions checked against generator
# - The two main region colors remain the two most frequent non-black colors.
# - Every auxiliary-color cell is an isolated removable background speck; no extra colors appear inside the main regions.
# - Each main region gets at least one nearby marker speck, and all specks bound to a given region share one color.
# - Every marker speck sits where the cleaned 5x5 neighborhood has a unique dominant host region, making the mapping stable.

from __future__ import annotations

from collections import Counter as _Counter
import random

import numpy as np

from arc_utils import Color


TRACE_MAP_RADIUS = 2
TRACE_MAP_MIN_HOST = 3


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two dominant foreground colors"
    return int(colors[0]), int(colors[1])


def _subsquare(grid: np.ndarray, r: int, c: int, radius: int = 1) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)]


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
    if v == int(bg):
        return int(bg), False

    if v not in colors:
        return int(bg), True

    a, b = map(int, colors)
    sub = _subsquare(g, r, c, radius=1)
    own = int(np.sum(sub == v))
    other = b if v == a else a
    other_count = int(np.sum(sub == other))
    bg_count = int(np.sum(sub == int(bg)))

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
    Apply the source task's two-pass local cleanup while deleting auxiliary marker specks.

    The cleanup dynamics for the two dominant regions stay the same; the new part is that
    sparse non-region colors are treated as removable background specks.
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


def _trace_specks(g: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    traces: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            new_v, mark_trace = _cleanup_decision(g, r, c, colors, bg)
            if mark_trace and int(new_v) == int(bg):
                traces.append((r, c, int(g[r, c])))
    return traces


def _trace_host_from_clean(clean: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    sub = _subsquare(clean, r, c, radius=TRACE_MAP_RADIUS)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = colors[1] if int(host) == int(colors[0]) else colors[0]
    if counts[int(host)] >= TRACE_MAP_MIN_HOST and counts[int(host)] > counts[int(other)]:
        return int(host)
    return None


def _region_center_colors(
    traces: list[tuple[int, int, int]], clean: np.ndarray, colors: tuple[int, int]
) -> dict[int, int]:
    votes: dict[int, _Counter[int]] = {int(colors[0]): _Counter(), int(colors[1]): _Counter()}
    for r, c, marker_color in traces:
        host = _trace_host_from_clean(clean, int(r), int(c), colors)
        if host is not None:
            votes[int(host)][int(marker_color)] += 1

    mapping: dict[int, int] = {}
    for host, counter in votes.items():
        if counter:
            mapping[int(host)] = int(counter.most_common(1)[0][0])
    return mapping


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _subsquare(clean, rr, cc, radius=1)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents into opposite-color patches whose center colors are remembered from removable specks.

    Input structure:
    - Black background with two large dominant foreground regions plus a few sparse one-cell corruptions.
    - The corruptions come in two kinds: cross-color boundary protrusions between the main regions, and isolated
      background specks in small auxiliary colors that do not belong to either large region.

    Rule:
    - First record every removable speck before cleanup. Then run the same two-pass local cleanup on the dominant
      regions: boundary protrusions are recolored into the locally dominant region, while isolated background specks
      disappear to black.
    - Each recorded background speck leaves a teal trace in the output at its own coordinate, but its original input
      color is not lost: after cleanup, that color is assigned to the nearby host region that dominates the speck's
      cleaned 5x5 neighborhood.
    - Every original black dent whose cleaned 3x3 neighborhood is mostly one dominant region receives a 3x3 patch of
      the opposite dominant region color, and the dent center is recolored with that region's remembered speck color.

    Key insight:
    - The disappearing background specks do two jobs at once. Their output footprint is teal, but their input colors
      act as instructions for the dent-center colors of the adjacent large regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    traces = _trace_specks(g, colors, bg)
    clean = _clean_foreground(g, colors, bg)
    center_colors = _region_center_colors(traces, clean, colors)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c, marker_color in traces:
        _ = marker_color
        out[int(r), int(c)] = Color.TEAL

    for r, c, host in centers:
        out[int(r), int(c)] = int(center_colors.get(int(host), Color.TEAL))

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

# --- CFG: marker-speck placement ---
TRACE_PER_REGION_MIN, TRACE_PER_REGION_MAX = 2, 4
TRACE_SPECK_SEPARATION = 3
BOUNDARY_NOISE_COUNT_MIN, BOUNDARY_NOISE_COUNT_MAX = 2, 8
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


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
    sub = _subsquare(grid, r, c, radius=1)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _pick_dent_centers(
    candidates: list[tuple[int, int]], n: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for r, c in shuffled:
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
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, nr, nc, host, other):
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
    center_a: int,
    center_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for cells in dent_groups_a:
        for r, c in cells:
            output[int(r), int(c)] = int(center_a)
    for cells in dent_groups_b:
        for r, c in cells:
            output[int(r), int(c)] = int(center_b)
    return output


def _dent_patch_mask(
    shape: tuple[int, int],
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return mask


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(inp[nr, nc]) != int(host):
                continue
            nsub = _subsquare(inp, nr, nc, radius=1).copy()
            nr0 = max(0, nr - 1)
            nc0 = max(0, nc - 1)
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
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
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if int(np.sum(_subsquare(clean, r, c, radius=1) == int(other))) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _subsquare(inp, r, c, radius=1)
        host_count = int(np.sum(sub_now == int(host)))
        other_count = int(np.sum(sub_now == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = int(other)
        placed += 1
    return placed >= n_at_boundary


def _trace_candidates_for_host(
    clean: np.ndarray, forbidden: np.ndarray, colors: tuple[int, int], host: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where((clean == 0) & (~forbidden))
    for r_, c_ in zip(rows, cols, strict=False):
        r = int(r_)
        c = int(c_)
        if _trace_host_from_clean(clean, r, c, colors) == int(host):
            candidates.append((r, c))
    return candidates


def _place_host_traces(
    inp: np.ndarray,
    output: np.ndarray,
    clean: np.ndarray,
    forbidden: np.ndarray,
    colors: tuple[int, int],
    host: int,
    marker_color: int,
    count: int,
    used: list[tuple[int, int]],
    rng: random.Random,
) -> bool:
    candidates = _trace_candidates_for_host(clean, forbidden, colors, host)
    rng.shuffle(candidates)

    placed = 0
    for r, c in candidates:
        if placed >= count:
            break
        if int(inp[r, c]) != 0:
            continue
        if any(max(abs(r - rr), abs(c - cc)) < TRACE_SPECK_SEPARATION for rr, cc in used):
            continue
        inp[r, c] = int(marker_color)
        output[r, c] = Color.TEAL
        used.append((r, c))
        placed += 1
    return placed >= count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    color_a, color_b = rng.sample(fg_pool, 2)

    marker_pool = [c for c in fg_pool if c not in (int(color_a), int(color_b))]
    center_a, center_b = rng.sample(marker_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
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

    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b, center_a, center_b)
    dent_mask = _dent_patch_mask(clean.shape, groups_a, groups_b)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    used_traces: list[tuple[int, int]] = []
    colors = (int(color_a), int(color_b))
    n_trace_a = rng.randint(TRACE_PER_REGION_MIN, TRACE_PER_REGION_MAX)
    n_trace_b = rng.randint(TRACE_PER_REGION_MIN, TRACE_PER_REGION_MAX)
    if not _place_host_traces(
        inp, output, clean, dent_mask, colors, color_a, center_a, n_trace_a, used_traces, rng
    ):
        return None
    if not _place_host_traces(
        inp, output, clean, dent_mask, colors, color_b, center_b, n_trace_b, used_traces, rng
    ):
        return None

    n_boundary = rng.randint(BOUNDARY_NOISE_COUNT_MIN, BOUNDARY_NOISE_COUNT_MAX)
    if not _add_boundary_noise(inp, clean, n_boundary, color_a, color_b, rng):
        return None

    traces = _trace_specks(inp, colors, Color.BLACK)
    if len([1 for _r, _c, marker in traces if int(marker) == int(center_a)]) < n_trace_a:
        return None
    if len([1 for _r, _c, marker in traces if int(marker) == int(center_b)]) < n_trace_b:
        return None

    center_map = _region_center_colors(traces, _clean_foreground(inp, colors, Color.BLACK), colors)
    if center_map.get(int(color_a)) != int(center_a):
        return None
    if center_map.get(int(color_b)) != int(center_b):
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
