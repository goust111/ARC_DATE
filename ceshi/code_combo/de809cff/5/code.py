# Combo variant of de809cff (index 5)
# combo_spec: D4/0+D13/0
# Name: 三色区域+图例定义映射

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   clean scene:
#     - black background plus exactly three large foreground regions in stable
#       spatial roles: left, right, bottom
#     - one isolated edge legend uses two parallel strips of length 3: the
#       edge-facing strip lists the three host colors, the inward strip lists
#       the patch colors assigned to those hosts
#     - each large region contains black dents and some one-cell noise
#   behavior trace:
#     - source task: for two colors, a dent always stamps the only other color
#     - D4 alone: with three regions, a dent could instead follow a fixed
#       positional cycle such as left -> right -> bottom -> left
#     - fused rule here: first decode the legend, e.g. top legend keys
#       [red, blue, green] with fills [blue, green, blue] means
#       red -> blue, blue -> green, green -> blue; then every dent in a host
#       region stamps that mapped color and keeps a teal center
#     - difference: the solver must combine spatial grouping (to find the host
#       region around a black dent) with symbolic decoding (to know which of the
#       two non-host colors to stamp for that host)
# complexity_calibration:
#   - the region layout remains close to the original task, so the new visual
#     novelty is concentrated in two places: a third region and the edge legend
#   - each sample contains dents in all three host colors, so every legend entry
#     is exercised multiple times within the same grid
#   - mapping freedom is limited to "each host chooses one of the other two
#     region colors"; this keeps the rule space human-readable while still
#     making the legend necessary
# implicit_assumption_audit:
#   - old `_main_colors` assumed exactly two meaningful foreground colors; now
#     the solver expects three dominant region colors
#   - old cleanup logic could collapse every non-main cell to black; that would
#     destroy the legend, so legend cells must be preserved and excluded from
#     local majority counts
#   - old stamping logic used a hard-coded opposite color; that no longer
#     exists, because each host now needs a legend lookup
#   - old patch painting could freely overwrite any 3x3 target; with a legend
#     present, both solver and generator must keep legend cells immutable
# placement_legality:
#   - generator reserves the legend block plus a one-cell black moat before
#     painting the large rectangles
#   - dent centers are only allowed where their 3x3 patch does not touch the
#     reserved legend zone
#   - noise placement is also blocked from the legend moat and from already-used
#     dent neighborhoods to avoid accidental overlaps
# visual_observability:
#   - host identity is visible from the large spatial regions, not from numeric
#     color ids
#   - the legend is isolated on an edge, so a human can notice that its outer
#     strip lists region colors and its inner strip predicts patch colors
#   - because mappings vary across generated samples and are not tied to a fixed
#     positional cycle, the edge legend is the only stable explanation
# neatness_and_inference:
#   - only two overlay layers remain in the output: a mapped 3x3 patch and a
#     teal center mark
#   - patches are separated by Chebyshev distance >= 3, keeping the scene legible
#   - allowing repeated fill colors is fine because each legend entry is paired
#     with a visible host key and repeated by several dents in that host region


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 3, "expected three foreground colors"
    return int(colors[0]), int(colors[1]), int(colors[2])


def _legend_masks(shape: tuple[int, int], side: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    legend = np.zeros((h, w), dtype=bool)
    if side == "top":
        r0, r1, c0, c1 = 0, 2, start, start + 3
    elif side == "bottom":
        r0, r1, c0, c1 = h - 2, h, start, start + 3
    elif side == "left":
        r0, r1, c0, c1 = start, start + 3, 0, 2
    else:
        r0, r1, c0, c1 = start, start + 3, w - 2, w

    legend[r0:r1, c0:c1] = True
    reserved = np.zeros((h, w), dtype=bool)
    reserved[max(0, r0 - 1) : min(h, r1 + 1), max(0, c0 - 1) : min(w, c1 + 1)] = True
    return legend, reserved


def _legend_array(side: str, keys: tuple[int, int, int], fills: tuple[int, int, int]) -> np.ndarray:
    if side == "top":
        return np.array([keys, fills], dtype=int)
    if side == "bottom":
        return np.array([fills, keys], dtype=int)
    if side == "left":
        return np.array([[keys[0], fills[0]], [keys[1], fills[1]], [keys[2], fills[2]]], dtype=int)
    return np.array([[fills[0], keys[0]], [fills[1], keys[1]], [fills[2], keys[2]]], dtype=int)


def _paint_legend(canvas: np.ndarray, side: str, start: int, legend_block: np.ndarray) -> None:
    if side == "top":
        canvas[0:2, start : start + 3] = legend_block
    elif side == "bottom":
        canvas[-2:, start : start + 3] = legend_block
    elif side == "left":
        canvas[start : start + 3, 0:2] = legend_block
    else:
        canvas[start : start + 3, -2:] = legend_block


def _surrounding_ring_is_bg(g: np.ndarray, legend_mask: np.ndarray, bg: int) -> bool:
    rows, cols = np.where(legend_mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    rr0, rr1 = max(0, r0 - 1), min(g.shape[0], r1 + 1)
    cc0, cc1 = max(0, c0 - 1), min(g.shape[1], c1 + 1)
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            if legend_mask[r, c]:
                continue
            if int(g[r, c]) != int(bg):
                return False
    return True


def _palette_mapping(g: np.ndarray, colors: tuple[int, int, int], bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    color_set = {int(color) for color in colors}
    candidates: list[tuple[dict[int, int], np.ndarray]] = []

    def _try_candidate(side: str, start: int) -> None:
        legend_mask, _reserved = _legend_masks((h, w), side, start)
        if side == "top":
            block = g[0:2, start : start + 3]
            outer = [int(x) for x in block[0].tolist()]
            inner = [int(x) for x in block[1].tolist()]
        elif side == "bottom":
            block = g[h - 2 : h, start : start + 3]
            outer = [int(x) for x in block[1].tolist()]
            inner = [int(x) for x in block[0].tolist()]
        elif side == "left":
            block = g[start : start + 3, 0:2]
            outer = [int(x) for x in block[:, 0].tolist()]
            inner = [int(x) for x in block[:, 1].tolist()]
        else:
            block = g[start : start + 3, w - 2 : w]
            outer = [int(x) for x in block[:, 1].tolist()]
            inner = [int(x) for x in block[:, 0].tolist()]

        if set(outer) != color_set or len(set(outer)) != 3:
            return
        if any(v == int(bg) or v not in color_set for v in inner):
            return
        if any(inner[i] == outer[i] for i in range(3)):
            return
        if not np.all(block != int(bg)):
            return
        if not _surrounding_ring_is_bg(g, legend_mask, bg):
            return
        candidates.append(({outer[i]: inner[i] for i in range(3)}, legend_mask))

    if w >= 3:
        for c0 in range(w - 2):
            _try_candidate("top", c0)
            _try_candidate("bottom", c0)
    if h >= 3:
        for r0 in range(h - 2):
            _try_candidate("left", r0)
            _try_candidate("right", r0)

    assert len(candidates) == 1, "expected exactly one edge legend"
    mapping, legend_mask = candidates[0]
    assert set(mapping) == color_set, "legend keys must match the three main colors"
    return mapping, legend_mask


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


def _dominant_color(
    sub: np.ndarray,
    colors: tuple[int, int, int],
    valid_mask: np.ndarray,
    threshold: int = 5,
) -> int | None:
    counts = {int(color): int(np.sum((sub == int(color)) & valid_mask)) for color in colors}
    best = max(counts.values())
    winners = [int(color) for color, count in counts.items() if count == best]
    if best < threshold or len(winners) != 1:
        return None
    return int(winners[0])


def _clean_cell(
    cur: np.ndarray,
    r: int,
    c: int,
    colors: tuple[int, int, int],
    bg: int,
    ignore_mask: np.ndarray,
) -> int:
    if bool(ignore_mask[r, c]):
        return int(cur[r, c])

    v = int(cur[r, c])
    if v not in colors:
        return int(bg)

    if _has_same_2x2(cur, r, c, v):
        return v

    rs = slice(max(0, r - 1), min(cur.shape[0], r + 2))
    cs = slice(max(0, c - 1), min(cur.shape[1], c + 2))
    sub = cur[rs, cs]
    valid = ~ignore_mask[rs, cs]
    counts = {int(color): int(np.sum((sub == int(color)) & valid)) for color in colors}
    own = counts[int(v)]
    bg_count = int(np.sum((sub == int(bg)) & valid))
    dominant = _dominant_color(sub, colors, valid, threshold=5)

    if dominant is not None and int(dominant) != int(v) and counts[int(dominant)] > own:
        return int(dominant)
    if bg_count >= 5 and own <= 4:
        return int(bg)
    if own <= 2:
        return int(bg)
    return int(v)


def _clean_foreground(
    g: np.ndarray,
    colors: tuple[int, int, int],
    bg: int,
    legend_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    The same two-sweep local repair works in the fused setting as long as the
    legend is treated as immutable structure and excluded from neighborhood tallies.
    """
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(cur.shape[0]):
            for c in range(cur.shape[1]):
                nxt[r, c] = _clean_cell(cur, r, c, colors, bg, legend_mask)
        cur = nxt

    return cur


def _role_colors(
    clean: np.ndarray,
    colors: tuple[int, int, int],
    ignore_mask: np.ndarray | None = None,
) -> dict[str, int]:
    if ignore_mask is None:
        ignore_mask = np.zeros(clean.shape, dtype=bool)

    stats: dict[int, tuple[float, float]] = {}
    for color in colors:
        coords = np.argwhere((clean == int(color)) & (~ignore_mask))
        assert len(coords) > 0, "missing region color"
        stats[int(color)] = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    bottom = max(colors, key=lambda color: stats[int(color)][0])
    top_two = [int(color) for color in colors if int(color) != int(bottom)]
    left = min(top_two, key=lambda color: stats[int(color)][1])
    right = max(top_two, key=lambda color: stats[int(color)][1])
    return {"left": int(left), "right": int(right), "bottom": int(bottom)}


def _dent_centers(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int, int],
    bg: int,
    legend_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((original == int(bg)) & (~legend_mask)):
        rr = int(r)
        cc = int(c)
        rs = slice(max(0, rr - 1), min(clean.shape[0], rr + 2))
        cs = slice(max(0, cc - 1), min(clean.shape[1], cc + 2))
        sub = clean[rs, cs]
        valid = ~legend_mask[rs, cs]
        dominant = _dominant_color(sub, colors, valid, threshold=5)
        if dominant is not None:
            centers.append((rr, cc, int(dominant)))
    return centers


def _paint_patch(out: np.ndarray, legend_mask: np.ndarray, r: int, c: int, color: int) -> None:
    rs = slice(max(0, r - 1), min(out.shape[0], r + 2))
    cs = slice(max(0, c - 1), min(out.shape[1], c + 2))
    patch = out[rs, cs].copy()
    writable = ~legend_mask[rs, cs]
    patch[writable] = int(color)
    out[rs, cs] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in three colored regions into legend-directed patches.

    Input structure:
    - Black background with three large foreground regions arranged as left,
      right, and bottom masses, plus one isolated edge legend made from two
      parallel strips of length 3.
    - The legend's edge-facing strip lists the three host-region colors, and
      the inward strip lists which foreground color each host should stamp.
    - The large regions still contain black dents and a few one-cell noise
      specks or protrusions, while the legend is surrounded by a black moat.

    Rule:
    - First decode the edge legend as a host-color -> patch-color mapping and
      preserve those legend cells unchanged.
    - Then locally clean the three large regions: isolated stray foreground
      cells revert to black, and lone boundary intrusions flip back to the
      unique neighboring foreground majority.
    - Each original black dent whose cleaned 3x3 neighborhood is dominated by
      one region color becomes the center of a 3x3 patch in that host's mapped
      legend color, and the dent center itself becomes teal (8).

    Key insight: The solver still identifies dents by local majority after
    cleanup, but with three region colors the patch color cannot come from a
    fixed cycle or from "the only other color"; it must be read from the edge
    legend for that specific host color.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    mapping, legend_mask = _palette_mapping(g, colors, bg)
    clean = _clean_foreground(g, colors, bg, legend_mask)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg, legend_mask)
    for r, c, host in centers:
        _paint_patch(out, legend_mask, r, c, mapping[int(host)])

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
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84
LEFT_RIGHT_OVERLAP_MIN = 2
BOTTOM_OVERLAP_MIN = 10

# --- CFG: dent placement ---
DENTS_PER_ROLE_MIN, DENTS_PER_ROLE_MAX = 4, 5
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.14

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 8, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.80
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: legend ---
LEGEND_SIDES = ("top", "bottom", "left", "right")

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _rand_span(N: int, frac_range: tuple[float, float], rng: random.Random) -> int:
    lo = max(1, int(round(N * frac_range[0])))
    hi = max(lo, int(round(N * frac_range[1])))
    return rng.randint(lo, hi)


def _build_regions(
    N: int,
    role_colors: dict[str, int],
    reserved_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    left = int(role_colors["left"])
    right = int(role_colors["right"])
    bottom = int(role_colors["bottom"])

    for _ in range(80):
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
        grid[reserved_mask] = Color.BLACK

        counts = {color: int(np.sum(grid == color)) for color in role_colors.values()}
        if min(counts.values()) < MIN_REGION_CELLS:
            continue
        coverage = sum(counts.values()) / float(N * N)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        colors = _main_colors(grid, Color.BLACK)
        if _role_colors(grid, colors) != role_colors:
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


def _patch_clear_of_reserved(reserved_mask: np.ndarray, r: int, c: int) -> bool:
    return not bool(np.any(reserved_mask[max(0, r - 1) : min(reserved_mask.shape[0], r + 2), max(0, c - 1) : min(reserved_mask.shape[1], c + 2)]))


def _pick_dent_centers(
    valid: dict[int, list[tuple[int, int]]],
    wanted: dict[int, int],
    rng: random.Random,
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
    reserved_mask: np.ndarray,
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
                    if reserved_mask[nr, nc] or not _patch_clear_of_reserved(reserved_mask, nr, nc):
                        continue
                    if int(clean[nr, nc]) != int(color):
                        continue
                    if not _is_safe_dent_center(clean, nr, nc, int(color), colors):
                        continue
                    if any(
                        max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION
                        for rr, cc in occupied
                        if (rr, cc) != (r, c)
                    ):
                        continue
                    cells.append((int(nr), int(nc)))
                    occupied.add((int(nr), int(nc)))
                    break
            groups[int(color)].append(cells)
    return groups


def _stamp_dents(
    region_clean: np.ndarray,
    legend_mask: np.ndarray,
    legend_block: np.ndarray,
    dent_groups: dict[int, list[list[tuple[int, int]]]],
    mapping: dict[int, int],
) -> np.ndarray:
    output = region_clean.copy()
    output[legend_mask] = legend_block[legend_mask]
    for host, groups in dent_groups.items():
        stamp_color = int(mapping[int(host)])
        for cells in groups:
            for r, c in cells:
                _paint_patch(output, legend_mask, r, c, stamp_color)
    for groups in dent_groups.values():
        for cells in groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _nearby_speck_color(
    clean: np.ndarray,
    r: int,
    c: int,
    colors: tuple[int, int, int],
    rng: random.Random,
) -> int:
    nearby = [int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors]
    if nearby:
        return int(_Counter(nearby).most_common(1)[0][0])
    return int(rng.choice(list(colors)))


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    colors: tuple[int, int, int],
    blocked_mask: np.ndarray,
    rng: random.Random,
) -> int:
    bg_candidates = list(np.argwhere((clean == int(Color.BLACK)) & (~blocked_mask)))
    rng.shuffle(bg_candidates)

    placed = 0
    for r, c in bg_candidates:
        if placed >= n_in_bg:
            break
        rr = int(r)
        cc = int(c)
        if int(inp[rr, cc]) != int(Color.BLACK):
            continue
        if int(np.sum(_sub3x3(inp, rr, cc) == int(Color.BLACK))) < BG_SPECK_MIN_BG_3x3:
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
    ignore_mask: np.ndarray,
) -> bool:
    if int(inp[r, c]) != int(host):
        return False

    rs = slice(max(0, r - 1), min(inp.shape[0], r + 2))
    cs = slice(max(0, c - 1), min(inp.shape[1], c + 2))
    sub_now = inp[rs, cs]
    valid = ~ignore_mask[rs, cs]
    host_count = int(np.sum((sub_now == int(host)) & valid))
    other_count = int(np.sum((sub_now == int(other)) & valid))
    if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
        return False
    if _would_form_2x2(inp, r, c, other):
        return False

    test = inp.copy()
    test[r, c] = int(other)
    if _clean_cell(test, r, c, colors, bg, ignore_mask) != int(host):
        return False

    r0 = max(0, r - 1)
    r1 = min(inp.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(inp.shape[1], c + 2)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if bool(ignore_mask[rr, cc]):
                continue
            before = _clean_cell(inp, rr, cc, colors, bg, ignore_mask)
            after = _clean_cell(test, rr, cc, colors, bg, ignore_mask)
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
    blocked_mask: np.ndarray,
    rng: random.Random,
) -> int:
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            if blocked_mask[r, c]:
                continue
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
        if _boundary_noise_safe(inp, r, c, host, other, colors, Color.BLACK, blocked_mask):
            inp[r, c] = int(other)
            placed += 1
    return placed


def _dent_guard_mask(shape: tuple[int, int], dent_groups: dict[int, list[list[tuple[int, int]]]]) -> np.ndarray:
    h, w = shape
    guard = np.zeros((h, w), dtype=bool)
    for groups in dent_groups.values():
        for cells in groups:
            for r, c in cells:
                guard[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return guard


def _choose_mapping(role_colors: dict[str, int], rng: random.Random) -> dict[int, int]:
    colors = [int(role_colors["left"]), int(role_colors["right"]), int(role_colors["bottom"])]
    fixed_cycle = {
        int(role_colors["left"]): int(role_colors["right"]),
        int(role_colors["right"]): int(role_colors["bottom"]),
        int(role_colors["bottom"]): int(role_colors["left"]),
    }
    while True:
        mapping = {
            int(host): int(rng.choice([color for color in colors if int(color) != int(host)]))
            for host in colors
        }
        if len(set(mapping.values())) < 2:
            continue
        if mapping == fixed_cycle:
            continue
        return mapping


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    left_color, right_color, bottom_color = rng.sample(fg_pool, 3)
    role_colors = {"left": left_color, "right": right_color, "bottom": bottom_color}

    side = rng.choice(LEGEND_SIDES)
    start = rng.randint(0, N - 3)
    legend_mask, reserved_mask = _legend_masks((N, N), side, start)
    mapping = _choose_mapping(role_colors, rng)

    key_order = [left_color, right_color, bottom_color]
    rng.shuffle(key_order)
    fill_order = [int(mapping[int(key)]) for key in key_order]
    legend_canvas = np.zeros((N, N), dtype=int)
    _paint_legend(
        legend_canvas,
        side,
        start,
        _legend_array(side, (key_order[0], key_order[1], key_order[2]), (fill_order[0], fill_order[1], fill_order[2])),
    )

    clean = _build_regions(N, role_colors, reserved_mask, rng)
    if clean is None:
        return None

    colors = _main_colors(clean, Color.BLACK)
    valid: dict[int, list[tuple[int, int]]] = {int(color): [] for color in colors}
    for r in range(N):
        for c in range(N):
            if reserved_mask[r, c] or not _patch_clear_of_reserved(reserved_mask, r, c):
                continue
            host = int(clean[r, c])
            if host in valid and _is_safe_dent_center(clean, r, c, host, colors):
                valid[int(host)].append((int(r), int(c)))

    wanted: dict[int, int] = {}
    for color in colors:
        wanted[int(color)] = DENTS_PER_ROLE_MIN + rng.randint(0, DENTS_PER_ROLE_MAX - DENTS_PER_ROLE_MIN)
        if len(valid[int(color)]) < wanted[int(color)] + 2:
            return None

    dents = _pick_dent_centers(valid, wanted, rng)
    if dents is None:
        return None

    groups = _expand_to_pairs(dents, clean, colors, reserved_mask, rng)
    noise_blocked = reserved_mask | _dent_guard_mask(clean.shape, groups)

    output = _stamp_dents(clean, legend_mask, legend_canvas, groups, mapping)

    inp = clean.copy()
    inp[legend_mask] = legend_canvas[legend_mask]
    for host_groups in groups.values():
        for cells in host_groups:
            for r, c in cells:
                inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    placed_bg = _add_bg_noise(inp, clean, n_bg, colors, noise_blocked, rng)
    placed_bd = _add_boundary_noise(inp, clean, n_noise - placed_bg, colors, noise_blocked, rng)
    if placed_bg + placed_bd < max(6, n_noise - 4):
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
