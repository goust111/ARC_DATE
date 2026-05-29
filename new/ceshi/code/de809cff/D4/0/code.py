# Variant of de809cff
# Dimension: D4 — 前景色区域的数量与类型 (src)
# Change: 恰好两个大面积前景色区域（如绿色3和黄色4），形成不规则形状占据网格大部分面积... → 三个前景色区域

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the D4 variant
# ----------------------------------
# concrete_behavior:
#   clean scene:
#     - black background plus exactly three large connected foreground regions
#     - the three regions play stable spatial roles: left, right, bottom
#     - each region has black dents and a few one-cell wrong-color specks
#   output scene:
#     - after cleanup, every dent becomes a 3x3 patch whose color follows a
#       visible role cycle: left -> right, right -> bottom, bottom -> left
#     - the dent center itself is teal
#   small execution trace:
#     - original task: a dent inside the green region always stamps the only
#       other foreground color, say yellow
#     - new task: a dent inside the left region does not use "the only other"
#       color because there are two alternatives; instead it uses the color of
#       the right region, while right dents use bottom and bottom dents use left
#     - difference: the solver must infer three roles first, then map host role
#       through a 3-cycle before stamping the 3x3 patch
# complexity_calibration:
#   - using three large regions keeps the main visual motif from the source task
#   - role-based color mapping has only one branch pattern (a fixed cycle), so
#     samples can repeat it enough for humans to infer it
#   - the generator guarantees dents in all three regions, preventing collapse
#     back to an effectively two-region training signal
# implicit_assumption_audit:
#   - old _main_colors assumed two colors; now the solver asserts three
#   - old cleanup compared a cell only against one "other" color; now it checks
#     for any unique dominant rival among three colors
#   - old dent stamping used host -> unique other-color; now it uses host-role
#     -> mapped role color, so role inference must happen before stamping
# placement_legality:
#   - generator paints three large rectangles in fixed spatial roles so the
#     centroid-based role inference is stable
#   - dent centers are chosen only where the host color is the unique 3x3
#     majority even after the center is removed
#   - noise is added with local legality checks and every sample is still
#     validated once by the full solver at the end
# visual_observability:
#   - the rule never depends on color ids or area ranking
#   - role identity is visible from position, and each sample contains all three
#     host colors producing dents, so the cycle is observable from examples
# neatness:
#   - only one overlay layer is added per dent (3x3 patch + teal center)
#   - patches are separated by Chebyshev distance >= 3, limiting clutter


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 3, "expected three foreground colors"
    return int(colors[0]), int(colors[1]), int(colors[2])


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


def _dominant_color(sub: np.ndarray, colors: tuple[int, int, int], threshold: int = 5) -> int | None:
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if best < threshold or len(winners) != 1:
        return None
    return int(winners[0])


def _clean_cell(cur: np.ndarray, r: int, c: int, colors: tuple[int, int, int], bg: int) -> int:
    v = int(cur[r, c])
    if v not in colors:
        return int(bg)

    if _has_same_2x2(cur, r, c, v):
        return v

    sub = _sub3x3(cur, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    own = counts[v]
    bg_count = int(np.sum(sub == int(bg)))
    dominant = _dominant_color(sub, colors, threshold=5)

    if dominant is not None and int(dominant) != v and counts[int(dominant)] > own:
        return int(dominant)
    if bg_count >= 5 and own <= 4:
        return int(bg)
    if own <= 2:
        return int(bg)
    return v


def _clean_foreground(g: np.ndarray, colors: tuple[int, int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    The same two-sweep local majority cleanup still works in the three-color
    setting as long as rival recoloring requires a unique dominant foreground
    color in the 3x3 neighborhood.
    """
    h, w = g.shape
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                nxt[r, c] = _clean_cell(cur, r, c, colors, bg)
        cur = nxt

    return cur


def _role_colors(clean: np.ndarray, colors: tuple[int, int, int]) -> dict[str, int]:
    stats: dict[int, tuple[float, float]] = {}
    for color in colors:
        coords = np.argwhere(clean == int(color))
        assert len(coords) > 0, "missing region color"
        stats[int(color)] = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    bottom = max(colors, key=lambda color: stats[int(color)][0])
    top_two = [int(color) for color in colors if int(color) != int(bottom)]
    left = min(top_two, key=lambda color: stats[int(color)][1])
    right = max(top_two, key=lambda color: stats[int(color)][1])
    return {"left": int(left), "right": int(right), "bottom": int(bottom)}


def _stamp_map(clean: np.ndarray, colors: tuple[int, int, int]) -> dict[int, int]:
    roles = _role_colors(clean, colors)
    return {
        int(roles["left"]): int(roles["right"]),
        int(roles["right"]): int(roles["bottom"]),
        int(roles["bottom"]): int(roles["left"]),
    }


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int, int], bg: int
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(original == int(bg)):
        rr = int(r)
        cc = int(c)
        dominant = _dominant_color(_sub3x3(clean, rr, cc), colors, threshold=5)
        if dominant is not None:
            centers.append((rr, cc, int(dominant)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in three colored regions into marked cyclic-color patches.

    Input structure:
    - Black background with three large foreground regions, visually arranged as
      left, right, and bottom masses.
    - The regions contain small black dents plus one-cell foreground specks in
      the background or along region boundaries.

    Rule:
    - First clean local one-cell corruption: isolated stray foreground cells
      revert to black, while a lone boundary intrusion flips back to the unique
      neighboring foreground majority.
    - For each original black dent whose cleaned 3x3 neighborhood is dominated
      by one region color, place a 3x3 patch in the cyclic target color:
      left dents use right, right dents use bottom, bottom dents use left.
    - Recolor every dent center teal (8) after stamping the surrounding patch.

    Key insight: The puzzle is no longer "use the only other color"; instead,
    the three large regions define spatial roles, and the dent patch color is
    determined by the role cycle left -> right -> bottom -> left.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    stamp_map = _stamp_map(clean, colors)

    out = clean.copy()
    h, w = g.shape
    centers = _dent_centers(g, clean, colors, bg)
    for r, c, host in centers:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_map[int(host)])

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
TOP_RECT_H_FRAC = (0.58, 0.80)
TOP_RECT_W_FRAC = (0.46, 0.62)
BOTTOM_RECT_H_FRAC = (0.36, 0.52)
BOTTOM_RECT_W_FRAC = (0.58, 0.82)
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 85
COVERAGE_MIN, COVERAGE_MAX = 0.60, 0.84
LEFT_RIGHT_OVERLAP_MIN = 2
BOTTOM_OVERLAP_MIN = 10

# --- CFG: dent placement ---
DENTS_PER_ROLE_MIN, DENTS_PER_ROLE_MAX = 4, 6
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.14

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 8, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.80
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _rand_span(N: int, frac_range: tuple[float, float], rng: random.Random) -> int:
    lo = max(1, int(round(N * frac_range[0])))
    hi = max(lo, int(round(N * frac_range[1])))
    return rng.randint(lo, hi)


def _build_regions(N: int, role_colors: dict[str, int], rng: random.Random) -> np.ndarray | None:
    left = int(role_colors["left"])
    right = int(role_colors["right"])
    bottom = int(role_colors["bottom"])

    for _ in range(60):
        l_h = _rand_span(N, TOP_RECT_H_FRAC, rng)
        l_w = _rand_span(N, TOP_RECT_W_FRAC, rng)
        r_h = _rand_span(N, TOP_RECT_H_FRAC, rng)
        r_w = _rand_span(N, TOP_RECT_W_FRAC, rng)
        b_h = _rand_span(N, BOTTOM_RECT_H_FRAC, rng)
        b_w = _rand_span(N, BOTTOM_RECT_W_FRAC, rng)

        l_r0 = rng.randint(0, RECT_MARGIN_MAX)
        l_c0 = rng.randint(0, RECT_MARGIN_MAX)
        r_r0 = rng.randint(0, RECT_MARGIN_MAX)
        r_c0 = max(0, N - r_w - rng.randint(0, RECT_MARGIN_MAX))
        b_r0 = max(0, N - b_h - rng.randint(0, RECT_MARGIN_MAX))
        b_center_jitter = rng.randint(-max(1, N // 10), max(1, N // 10))
        b_c0 = (N - b_w) // 2 + b_center_jitter
        b_c0 = min(max(0, b_c0), max(0, N - b_w))

        l_r1, l_c1 = min(N, l_r0 + l_h), min(N, l_c0 + l_w)
        r_r1, r_c1 = min(N, r_r0 + r_h), min(N, r_c0 + r_w)
        b_r1, b_c1 = min(N, b_r0 + b_h), min(N, b_c0 + b_w)

        lr_overlap_h = max(0, min(l_r1, r_r1) - max(l_r0, r_r0))
        lr_overlap_w = max(0, min(l_c1, r_c1) - max(l_c0, r_c0))
        left_bottom_overlap = max(0, min(l_r1, b_r1) - max(l_r0, b_r0)) * max(0, min(l_c1, b_c1) - max(l_c0, b_c0))
        right_bottom_overlap = max(0, min(r_r1, b_r1) - max(r_r0, b_r0)) * max(0, min(r_c1, b_c1) - max(r_c0, b_c0))
        if lr_overlap_h <= 0 or lr_overlap_w < LEFT_RIGHT_OVERLAP_MIN:
            continue
        if left_bottom_overlap < BOTTOM_OVERLAP_MIN or right_bottom_overlap < BOTTOM_OVERLAP_MIN:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[l_r0:l_r1, l_c0:l_c1] = left
        grid[r_r0:r_r1, r_c0:r_c1] = right
        grid[b_r0:b_r1, b_c0:b_c1] = bottom

        counts = {color: int(np.sum(grid == color)) for color in role_colors.values()}
        if min(counts.values()) < MIN_REGION_CELLS:
            continue
        coverage = sum(counts.values()) / float(N * N)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        if _role_colors(grid, _main_colors(grid, Color.BLACK)) != role_colors:
            continue
        return grid
    return None


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, colors: tuple[int, int, int]) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    counts[int(host)] -= 1
    host_count = counts[int(host)]
    other_max = max(counts[int(color)] for color in colors if int(color) != int(host))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_max


def _pick_dent_centers(
    valid: dict[int, list[tuple[int, int]]], wanted: dict[int, int], rng: random.Random
) -> dict[int, list[tuple[int, int]]] | None:
    chosen: dict[int, list[tuple[int, int]]] = {int(color): [] for color in valid}
    occupied: list[tuple[int, int]] = []
    order = sorted(valid, key=lambda color: len(valid[int(color)]))

    for color in order:
        candidates = valid[int(color)].copy()
        rng.shuffle(candidates)
        for r, c in candidates:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied):
                chosen[int(color)].append((int(r), int(c)))
                occupied.append((int(r), int(c)))
                if len(chosen[int(color)]) >= int(wanted[int(color)]):
                    break
        if len(chosen[int(color)]) < int(wanted[int(color)]):
            return None
    return chosen


def _expand_to_pairs(
    dents: dict[int, list[tuple[int, int]]],
    clean: np.ndarray,
    colors: tuple[int, int, int],
    rng: random.Random,
) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = clean.shape
    occupied = {(r, c) for centers in dents.values() for r, c in centers}
    groups: dict[int, list[list[tuple[int, int]]]] = {int(color): [] for color in dents}

    for color, centers in dents.items():
        for r, c in centers:
            cells = [(int(r), int(c))]
            if rng.random() < DENT_PAIR_PROB:
                neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                rng.shuffle(neighbors)
                for nr, nc in neighbors:
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    if int(clean[nr, nc]) != int(color):
                        continue
                    if not _is_safe_dent_center(clean, nr, nc, int(color), colors):
                        continue
                    if any(max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION for rr, cc in occupied if (rr, cc) != (r, c)):
                        continue
                    cells.append((int(nr), int(nc)))
                    occupied.add((int(nr), int(nc)))
                    break
            groups[int(color)].append(cells)
    return groups


def _stamp_dents(clean: np.ndarray, dent_groups: dict[int, list[list[tuple[int, int]]]], stamp_map: dict[int, int]) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for host, groups in dent_groups.items():
        stamp_color = int(stamp_map[int(host)])
        for cells in groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for groups in dent_groups.values():
        for cells in groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _nearby_speck_color(clean: np.ndarray, r: int, c: int, colors: tuple[int, int, int], rng: random.Random) -> int:
    nearby = [int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors]
    if nearby:
        return int(_Counter(nearby).most_common(1)[0][0])
    return int(rng.choice(list(colors)))


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int, int], rng: random.Random) -> int:
    bg_candidates = list(np.argwhere(clean == Color.BLACK))
    rng.shuffle(bg_candidates)

    placed = 0
    for r, c in bg_candidates:
        if placed >= n_in_bg:
            break
        rr = int(r)
        cc = int(c)
        if int(inp[rr, cc]) != int(Color.BLACK):
            continue
        if int(np.sum(_sub3x3(inp, rr, cc) == Color.BLACK)) < BG_SPECK_MIN_BG_3x3:
            continue
        inp[rr, cc] = _nearby_speck_color(clean, rr, cc, colors, rng)
        placed += 1
    return placed


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if bool(np.all(block == int(color))):
                    return True
    return False


def _boundary_noise_safe(
    inp: np.ndarray,
    r: int,
    c: int,
    host: int,
    other: int,
    colors: tuple[int, int, int],
    bg: int,
) -> bool:
    if int(inp[r, c]) != int(host):
        return False

    sub_now = _sub3x3(inp, r, c)
    host_count = int(np.sum(sub_now == int(host)))
    other_count = int(np.sum(sub_now == int(other)))
    if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
        return False
    if _would_form_2x2(inp, r, c, other):
        return False

    test = inp.copy()
    test[r, c] = int(other)
    if _clean_cell(test, r, c, colors, bg) != int(host):
        return False

    r0 = max(0, r - 1)
    r1 = min(inp.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(inp.shape[1], c + 2)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            before = _clean_cell(inp, rr, cc, colors, bg)
            after = _clean_cell(test, rr, cc, colors, bg)
            if (rr, cc) == (r, c):
                if after != int(host):
                    return False
            elif after != before:
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    colors: tuple[int, int, int],
    rng: random.Random,
) -> int:
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            host = int(clean[r, c])
            if host not in colors:
                continue
            nearby = sorted({int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors and int(x) != host})
            for other in nearby:
                candidates.append((int(r), int(c), int(host), int(other)))
    rng.shuffle(candidates)

    placed = 0
    for r, c, host, other in candidates:
        if placed >= n_at_boundary:
            break
        if _boundary_noise_safe(inp, r, c, host, other, colors, Color.BLACK):
            inp[r, c] = int(other)
            placed += 1
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    left_color, right_color, bottom_color = rng.sample(fg_pool, 3)
    role_colors = {"left": left_color, "right": right_color, "bottom": bottom_color}

    clean = _build_regions(N, role_colors, rng)
    if clean is None:
        return None

    colors = _main_colors(clean, Color.BLACK)
    stamp_map = _stamp_map(clean, colors)

    valid: dict[int, list[tuple[int, int]]] = {int(color): [] for color in colors}
    for r in range(N):
        for c in range(N):
            host = int(clean[r, c])
            if host in valid and _is_safe_dent_center(clean, r, c, host, colors):
                valid[host].append((int(r), int(c)))

    wanted: dict[int, int] = {}
    for color in colors:
        extras = rng.randint(0, DENTS_PER_ROLE_MAX - DENTS_PER_ROLE_MIN)
        wanted[int(color)] = DENTS_PER_ROLE_MIN + extras
        if len(valid[int(color)]) < wanted[int(color)] + 2:
            return None

    dents = _pick_dent_centers(valid, wanted, rng)
    if dents is None:
        return None

    groups = _expand_to_pairs(dents, clean, colors, rng)
    output = _stamp_dents(clean, groups, stamp_map)

    inp = clean.copy()
    for host_groups in groups.values():
        for cells in host_groups:
            for r, c in cells:
                inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    placed_bg = _add_bg_noise(inp, clean, n_bg, colors, rng)
    placed_bd = _add_boundary_noise(inp, clean, n_noise - placed_bg, colors, rng)
    if placed_bg + placed_bd < max(6, n_noise - 3):
        return None

    produced = main(inp)
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
