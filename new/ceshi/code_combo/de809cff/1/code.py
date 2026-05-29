# Combo variant of de809cff (index 1)
# combo_spec: D1/0+D3/0
# Name: 十字形补丁+4连通邻域判定

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Keep the original scene template: two large foreground regions on black,
#     small black dents inside them, and one-cell foreground noise that must be
#     cleaned before dent reasoning.
#   - Fuse two coordinated changes:
#       1. the opposite-color patch is now an orthogonal cross instead of a full
#          3x3 square;
#       2. dent ownership is no longer inferred from the whole 3x3 window, but
#          from the orthogonal 4-neighborhood only.
#   - The geometry is intentionally aligned: the same orthogonal adjacency that
#     decides whether a hole belongs to a region also decides which surrounding
#     cells are recolored.
# concrete_behavior_trace:
#   - Regular interior dent:
#         A A A
#         A x A
#         A A A
#     Original task:
#       * cleaned 3x3 says host=A
#       * output writes a 3x3 block of B, center teal
#     Combo task:
#       * cleaned 4-neighbors also say host=A
#       * output writes only the orthogonal cross of B, center teal
#     Difference:
#       * diagonals stay as in the cleaned region instead of being recolored.
#   - Boundary diagnostic hole:
#         A A A
#         A x B
#         B B A
#     Original 3x3 host test:
#       * A count = 5, B count = 3, so the hole is transformed
#     Combo 4-neighbor host test:
#       * orthogonal counts are A=2, B=2, so there is no strict orthogonal
#         majority and the hole is not transformed
#     Difference:
#       * this sample exposes the D3 change directly; the black cell remains a
#         black hole in the combo output instead of turning into a marked patch.
# complexity_calibration:
#   - D1 needs several non-edge dents so the cross-vs-square cue is obvious.
#   - D3 needs at least one boundary-pattern hole where 3x3 and 4-neighbor
#     ownership disagree; otherwise the combo could visually collapse into the
#     D1-only variant.
#   - The generator therefore enforces both:
#       * multiple full visible crosses, and
#       * at least one diagnostic hole that changes behavior under 4-neighbor
#         ownership compared with the old 3x3 ownership rule.
# implicit_assumption_audit:
#   - `_clean_foreground` still assumes corruption is only one-cell specks or
#     one-cell protrusions; that contract is preserved.
#   - The original dent detector implicitly relied on 3x3 support suppressing
#     accidental detections on ordinary background cells. Under the combo rule,
#     that must be replaced with a stricter orthogonal-majority test over all
#     available orthogonal neighbors, not just foreground counts; otherwise a
#     plain background corner outside a rectangle could be mistaken for a dent.
#   - The original square write `out[r-1:r+2, c-1:c+2] = other` is invalid once
#     the patch geometry changes; stamping must become bounded cell-wise writes
#     on `(center, up, down, left, right)`.
#   - The final teal overwrite remains necessary because multiple crosses or
#     paired dents may touch; the center marker must dominate regardless of
#     stamp order.
# placement_legality:
#   - Dent anchors are chosen with global Chebyshev spacing across both colors
#     so visible crosses stay separable.
#   - Optional 2-cell dents remain orthogonally adjacent and are only expanded
#     from already-valid 4-neighbor dents.
#   - Noise placement avoids the immediate vicinity of reserved dent cells, and
#     every generated sample is oracle-checked with the final solver.
# observability_and_tidiness:
#   - The generator rejects samples if the combo output equals the square-patch
#     baseline, because that would hide D1.
#   - It also rejects samples if the combo output equals the cross+3x3 baseline,
#     because that would hide D3.
#   - Diagnostic holes are kept sparse so the grid still reads as "mostly marked
#     dents plus a few exceptional boundary holes", rather than chaotic overlap.

import random
import sys
from collections import Counter as _Counter
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import Color


ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    seen: list[int] = []
    present: set[int] = set()
    for value in g.flat:
        color = int(value)
        if color == int(bg) or color in present:
            continue
        present.add(color)
        seen.append(color)
    counts = {color: int(np.sum(g == color)) for color in seen}
    seen.sort(key=lambda color: counts[color], reverse=True)
    assert len(seen) >= 2, "expected two foreground colors"
    return int(seen[0]), int(seen[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1] if int(color) == int(colors[0]) else colors[0])


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
    Remove one-cell specks/protrusions before detecting black dents.

    Two sweeps are enough for the intended corruption pattern: the first removes
    protruding wrong-color cells, and the second stabilizes the corrected local
    majority around them.
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


def _host_by_3x3(clean: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    counts = {int(color): int(np.sum(_sub3x3(clean, r, c) == int(color))) for color in colors}
    center = int(clean[r, c])
    if center in counts:
        counts[center] -= 1
    host = max(colors, key=lambda color: counts[int(color)])
    other = _other_color(colors, int(host))
    if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
        return int(host)
    return None


def _host_by_4(clean: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    h, w = clean.shape
    counts = {int(color): 0 for color in colors}
    total = 0
    for dr, dc in ORTH_DIRS:
        rr = int(r) + dr
        cc = int(c) + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        total += 1
        value = int(clean[rr, cc])
        if value in counts:
            counts[value] += 1
    host = max(colors, key=lambda color: counts[int(color)])
    other = _other_color(colors, int(host))
    if counts[int(host)] > counts[int(other)] and counts[int(host)] * 2 > total:
        return int(host)
    return None


def _dent_centers(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    mode: str,
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _host_by_4(clean, rr, cc, colors) if mode == "4" else _host_by_3x3(clean, rr, cc, colors)
        if host is not None:
            centers.append((rr, cc, int(host)))
    return centers


def _cross_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), *ORTH_DIRS):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
    return cells


def _solve_with_mode(grid: np.ndarray, mode: str, patch_shape: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg, mode)
    h, w = g.shape
    for r, c, host in centers:
        other = _other_color(colors, host)
        if patch_shape == "square":
            out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
        else:
            for rr, cc in _cross_cells(g.shape, r, c):
                out[rr, cc] = int(other)

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn qualifying black dents in two colored regions into orthogonal opposite-color marks.

    Input structure:
    - Black background with two large foreground regions in different colors.
    - The regions contain black dents plus a small amount of one-cell foreground
      noise in the background or along the contact boundary of the two regions.

    Rule:
    - First locally clean the foreground so isolated color specks disappear and
      one-cell boundary contamination flips back to the surrounding region.
    - For each original black cell, decide dent ownership from the cleaned
      orthogonal 4-neighborhood only; diagonal support no longer counts.
    - If one foreground color has a strict orthogonal majority, paint only the
      opposite-color cross centered there, then recolor the center itself teal.

    Key insight: The combo aligns detection and rendering around the same
    orthogonal geometry: 4-neighbor support identifies a dent, and only the
    orthogonal cross is drawn, so diagonal evidence and diagonal painting both
    disappear compared with the original task.
    """
    return _solve_with_mode(grid, mode="4", patch_shape="cross")


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13
MIN_FULL_CROSS_CENTERS = 5
DIAGNOSTIC_DENTS_MIN = 1
DIAGNOSTIC_DENTS_MAX = 3

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6
NOISE_FORBID_RADIUS = 1

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


def _far_enough(candidate: tuple[int, int], reserved: list[tuple[int, int]]) -> bool:
    r, c = candidate
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in reserved)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    reserved: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    blocked = list(reserved)
    for r, c in shuffled:
        if _far_enough((r, c), blocked):
            chosen.append((r, c))
            blocked.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None if n > 0 else []


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    clean: np.ndarray,
    colors: tuple[int, int],
    reserved: set[tuple[int, int]],
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
                if _host_by_4(clean, nr, nc, colors) != int(host):
                    continue
                if any(
                    (r0, c0) != (r, c)
                    and max(abs(nr - r0), abs(nc - c0)) < DENT_PATCH_SEPARATION
                    for r0, c0 in reserved
                ):
                    continue
                cells.append((nr, nc))
                reserved.add((nr, nc))
                break
        result.append(cells)
    return result


def _count_full_cross_centers(shape: tuple[int, int], centers: list[tuple[int, int]]) -> int:
    h, w = map(int, shape)
    return sum(1 for r, c in centers if 0 < r < h - 1 and 0 < c < w - 1)


def _too_close(r: int, c: int, forbidden: set[tuple[int, int]], radius: int) -> bool:
    return any(max(abs(r - rr), abs(c - cc)) <= radius for rr, cc in forbidden)


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if _too_close(r, c, forbidden, NOISE_FORBID_RADIUS):
            continue
        if int(np.sum(_sub3x3(inp, r, c) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


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


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
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
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if _too_close(r, c, forbidden, NOISE_FORBID_RADIUS):
                continue
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
        if int(inp[r, c]) != int(host):
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


def _candidate_lists(
    clean: np.ndarray, colors: tuple[int, int]
) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    regular: dict[int, list[tuple[int, int]]] = {int(colors[0]): [], int(colors[1]): []}
    diagnostic: list[tuple[int, int]] = []
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            value = int(clean[r, c])
            if value not in regular:
                continue
            host4 = _host_by_4(clean, r, c, colors)
            if host4 == value:
                regular[value].append((r, c))
            host3 = _host_by_3x3(clean, r, c, colors)
            if host3 is not None and host4 != host3:
                diagnostic.append((r, c))
    return regular, diagnostic


def _choose_diag_centers(
    candidates: list[tuple[int, int]], n_total: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if len(candidates) < DIAGNOSTIC_DENTS_MIN:
        return None
    upper = min(DIAGNOSTIC_DENTS_MAX, len(candidates), max(DIAGNOSTIC_DENTS_MIN, n_total - 6))
    if upper < DIAGNOSTIC_DENTS_MIN:
        return None
    target = rng.randint(DIAGNOSTIC_DENTS_MIN, upper)
    chosen = _pick_dent_centers(candidates, target, [], rng)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    regular, diagnostic = _candidate_lists(clean, colors)
    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    diag_centers = _choose_diag_centers(diagnostic, n_total, rng)
    if diag_centers is None:
        return None

    n_regular_total = n_total - len(diag_centers)
    desired_a = n_regular_total // 2
    desired_b = n_regular_total - desired_a
    if rng.random() < 0.5:
        desired_a, desired_b = desired_b, desired_a

    reserved = list(diag_centers)
    targets = {
        color_a: max(0, desired_a),
        color_b: max(0, desired_b),
    }

    chosen_regular: dict[int, list[tuple[int, int]]] = {}
    order = [color_a, color_b]
    rng.shuffle(order)
    for color in order:
        chosen = _pick_dent_centers(regular[color], targets[color], reserved, rng)
        if chosen is None:
            return None
        chosen_regular[color] = chosen
        reserved.extend(chosen)

    all_reserved = set(diag_centers + chosen_regular[color_a] + chosen_regular[color_b])
    groups: dict[int, list[list[tuple[int, int]]]] = {}
    expand_order = [color_a, color_b]
    rng.shuffle(expand_order)
    for color in expand_order:
        groups[color] = _expand_to_pairs(chosen_regular[color], color, clean, colors, all_reserved, rng)

    holes: set[tuple[int, int]] = set(diag_centers)
    visible_regular_centers: list[tuple[int, int]] = []
    for color in colors:
        for cells in groups[color]:
            visible_regular_centers.append(cells[0])
            holes.update(cells)

    if _count_full_cross_centers(clean.shape, visible_regular_centers) < min(MIN_FULL_CROSS_CENTERS, n_regular_total):
        return None

    base_input = clean.copy()
    for r, c in holes:
        base_input[r, c] = Color.BLACK

    combo_output = _solve_with_mode(base_input, mode="4", patch_shape="cross")
    square_baseline = _solve_with_mode(base_input, mode="4", patch_shape="square")
    old_detection_baseline = _solve_with_mode(base_input, mode="3x3", patch_shape="cross")
    if int(np.sum(combo_output != square_baseline)) <= 0:
        return None
    if int(np.sum(combo_output != old_detection_baseline)) <= 0:
        return None

    inp = base_input.copy()
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, holes, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, holes, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, combo_output):
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
