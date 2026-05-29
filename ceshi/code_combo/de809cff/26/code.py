# Combo variant of de809cff (index 26)
# combo_spec: D1/1+D2/2
# Name: 中心色=补丁同色+同侧色补丁

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
# - The input family is unchanged: two large non-black regions sit on black
#   background, with black dents plus one-cell foreground noise.
# - Original behavior at one dent:
#     cleaned 3x3 majority = red -> paint 3x3 yellow -> center becomes teal.
# - D1/1 alone would do:
#     cleaned 3x3 majority = red -> paint 3x3 red -> center still teal.
# - This fused D1/1+D2/2 variant must do:
#     cleaned 3x3 majority = red -> paint 3x3 red -> center also red.
# - Therefore each valid dent becomes a solid host-colored 3x3 block. The
#   largest visible difference appears when that 3x3 reaches background or the
#   other region; deep interior dents may disappear completely after repair.
#
# b) Complexity calibration
# - Removing the teal marker makes the task much easier to degenerate: if most
#   dents are interior, the output can look like "just clean the noise and fill
#   the holes". That weakens the intended fusion signal.
# - To keep the rule inferable from 2-4 ARC examples, the generator forces
#   multiple visually revealing dents whose same-side 3x3 repair expands beyond
#   the center cell, and it requires such evidence for both foreground colors.
#
# c) Main() line-by-line assumption review
# - _main_colors still assumes exactly two dominant foreground colors after
#   excluding black. The fusion does not alter the input palette, so this holds.
# - _clean_foreground still treats any non-main-color local corruption as noise.
#   That remains valid because the corruption model is unchanged.
# - _dent_centers still uses the cleaned 3x3 majority with threshold >= 5 host
#   cells and strict dominance over the other color. The host-inference rule is
#   unchanged by the fusion.
# - The old main() depended on two output-only assumptions that no longer hold:
#   "stamp opposite color" and "recolor the center teal". Both must be removed.
# - The new renderer assumes that a valid dent should become a solid host-color
#   3x3 repair. generate_input() guarantees that enough of these repairs are
#   visibly larger than center-only restoration.
#
# d) Placement legality
# - Dent centers are sampled only from host cells whose 3x3 remains majority
#   host after blackening the center, so the solver can recover the host color.
# - Dent centers of either color keep Chebyshev spacing >= 3 to avoid confusing
#   overlaps between unrelated 3x3 repairs.
# - Optional pair dents add a second orthogonally adjacent black cell only when
#   both cells remain individually detectable and the extra cell stays clear of
#   all other dents.
#
# e) Visual observability
# - The generator measures whether a repaired dent changes any non-center cell
#   after cleanup. Samples are accepted only when several dents satisfy this and
#   the total number of visible extra repaired cells is substantial.
# - This makes the intended rule readable by eye: dents are not marked with a
#   third color anymore, so the repeated clue must come from same-color local
#   expansion into nearby background / foreign-color cells.
#
# f) Visual cleanliness
# - Noise remains one-cell only, dents are spaced out, and cross-color repair
#   overlaps are blocked. The output stays visually sparse: two large regions
#   plus a handful of local solid-color repairs.
#
# g) Color inference
# - No hidden ordering of colors is used. The output color for a dent is always
#   the locally dominant host color seen around that dent after cleanup. Because
#   there is no teal marker, the generator keeps the palette to exactly the two
#   foreground colors plus black input background.
#
# h) New implicit assumptions after the fusion
# - main() now assumes the dataset supplies enough edge-adjacent or boundary-
#   adjacent dents for the same-side repair to be visible at all.
# - main() also assumes that each original black dent can be judged from the
#   cleaned neighborhood alone; generate_input() preserves that with output-
#   first validation and an oracle check on every accepted sample.


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


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    Two sweeps are enough for the observed corruption pattern: the first removes
    protruding wrong-color cells, the second lets neighboring true cells keep
    the corrected local majority.
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

                sub = cur[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
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
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _render_centers(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host in centers:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(host)
    return output


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fully repair black dents in two colored regions using solid same-color patches.

    Input structure:
    - Black background with two large foreground regions in two non-black colors.
    - The regions contain black dents plus a small number of one-cell foreground
      specks in the background or one-cell contamination along region borders.

    Rule:
    - First clean the foreground locally: isolated color specks disappear into
      black, and one-cell cross-boundary contamination flips back to the locally
      dominant region color.
    - Then inspect each original black cell. If its cleaned 3x3 neighborhood is
      mostly one foreground color, repaint that clipped 3x3 with the SAME host
      color rather than the opposite color.
    - Do not place a special center marker; the dent center keeps the same color
      as the rest of the repaired 3x3 patch.

    Key insight: valid dents are healed into solid 3x3 blocks of their own host
    color, so the informative cases are the ones where that repair visibly grows
    the region into nearby background or the other foreground region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _render_centers(clean, centers)


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
VISIBLE_PICK_MIN_PER_COLOR = 2
VISIBLE_DENTS_PER_COLOR_MIN = 1
VISIBLE_DENTS_TOTAL_MIN = 5
VISIBLE_EXTRA_CELLS_MIN = 8

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles with simple occlusion."""
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
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _visible_repair_gain(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """How many non-center cells would solid host-color repair overwrite here?"""
    sub = _sub3x3(clean, r, c).copy()
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    sub[r - r0, c - c0] = int(host)
    return int(np.sum(sub != int(host)))


def _is_visible_patch_center(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    gain = _visible_repair_gain(grid, r, c, host)
    return gain >= 1 and _has_same_2x2(grid, r, c, host)


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


def _far_enough(pos: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen)


def _pick_dent_centers_with_visible(
    visible: list[tuple[int, int]],
    hidden: list[tuple[int, int]],
    n: int,
    min_visible: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    """Pick n centers while reserving enough visibly revealing same-side repairs."""
    visible = list(visible)
    hidden = list(hidden)
    rng.shuffle(visible)
    rng.shuffle(hidden)

    chosen: list[tuple[int, int]] = []
    visible_chosen = 0

    for pos in visible:
        if visible_chosen >= min_visible:
            break
        if _far_enough(pos, blocked) and _far_enough(pos, chosen):
            chosen.append(pos)
            visible_chosen += 1
    if visible_chosen < min_visible:
        return None

    remaining = [pos for pos in visible if pos not in chosen] + hidden
    rng.shuffle(remaining)
    for pos in remaining:
        if len(chosen) >= n:
            break
        if _far_enough(pos, blocked) and _far_enough(pos, chosen):
            chosen.append(pos)

    if len(chosen) != n:
        return None
    return chosen


def _dent_remains_detectable(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check the main() dent predicate on an already-blackened working grid."""
    if int(grid[r, c]) != Color.BLACK:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host)))
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


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
    taken = list(occupied)
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
                    max(abs(nr - r2), abs(nc - c2)) < DENT_PATCH_SEPARATION
                    for r2, c2 in taken
                    if (r2, c2) != (r, c)
                ):
                    continue
                temp = clean.copy()
                temp[r, c] = Color.BLACK
                temp[nr, nc] = Color.BLACK
                if not (
                    _dent_remains_detectable(temp, r, c, host, other)
                    and _dent_remains_detectable(temp, nr, nc, host, other)
                ):
                    continue
                cells.append((nr, nc))
                taken.append((nr, nc))
                break
        result.append(cells)
    return result


def _add_bg_noise(
    inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    """Place isolated fg-color specks in background area."""
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` will not create solver side-effects nearby."""
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
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    """Place cross-color contamination at region-region boundaries."""
    h, w = clean.shape
    boundary_candidates = []
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
        if int(inp[r, c]) != int(host):
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
        placed += 1


def _visible_metrics(
    cleaned_input: np.ndarray, centers: list[tuple[int, int, int]]
) -> tuple[int, int, dict[int, int]]:
    visible_centers = 0
    per_color = {host: 0 for _, _, host in centers}
    for r, c, host in centers:
        if _visible_repair_gain(cleaned_input, r, c, host) > 0:
            visible_centers += 1
            per_color[host] = per_color.get(host, 0) + 1
    center_only = cleaned_input.copy()
    for r, c, host in centers:
        center_only[r, c] = int(host)
    rendered = _render_centers(cleaned_input, centers)
    extra_visible_cells = int(np.sum(rendered != center_only))
    return visible_centers, extra_visible_cells, per_color


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify the fused behavior on one generated sample.

    Metrics:
    - total_centers: dents actually detected by the solver.
    - visible_centers: dents whose solid host-color repair changes at least one
      non-center cell after cleanup.
    - extra_visible_cells: changed output cells beyond simple center restoration.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    color_a, color_b = _main_colors(g, bg)
    cleaned = _clean_foreground(g, (color_a, color_b), bg)
    centers = _dent_centers(g, cleaned, (color_a, color_b), bg)
    visible_centers, extra_visible_cells, per_color = _visible_metrics(cleaned, centers)
    return {
        "total_centers": len(centers),
        "visible_centers": visible_centers,
        "extra_visible_cells": extra_visible_cells,
        "visible_color_a": per_color.get(color_a, 0),
        "visible_color_b": per_color.get(color_b, 0),
    }


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

    valid_visible = {color_a: [], color_b: []}
    valid_hidden = {color_a: [], color_b: []}
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_a) else valid_hidden
                target[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                target = valid_visible if _is_visible_patch_center(clean, r, c, color_b) else valid_hidden
                target[color_b].append((r, c))

    if len(valid_visible[color_a]) + len(valid_hidden[color_a]) < n_a:
        return None
    if len(valid_visible[color_b]) + len(valid_hidden[color_b]) < n_b:
        return None

    picked: dict[int, list[tuple[int, int]]] = {}
    blocked: list[tuple[int, int]] = []
    color_order = [color_a, color_b]
    rng.shuffle(color_order)
    targets = {color_a: n_a, color_b: n_b}
    others = {color_a: color_b, color_b: color_a}

    for color in color_order:
        min_visible = max(VISIBLE_PICK_MIN_PER_COLOR, targets[color] // 3)
        centers = _pick_dent_centers_with_visible(
            valid_visible[color],
            valid_hidden[color],
            targets[color],
            min_visible,
            blocked,
            rng,
        )
        if centers is None:
            return None
        picked[color] = centers
        blocked.extend(centers)

    occupied = picked[color_a] + picked[color_b]
    groups: dict[int, list[list[tuple[int, int]]]] = {}
    expand_order = [color_a, color_b]
    rng.shuffle(expand_order)
    for color in expand_order:
        groups[color] = _expand_to_pairs(picked[color], color, others[color], clean, occupied, rng)
        for cells in groups[color]:
            if len(cells) == 2:
                occupied.append(cells[1])

    inp = clean.copy()
    for dent_groups in (groups[color_a], groups[color_b]):
        for cells in dent_groups:
            for r, c in cells:
                inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        cleaned_input = _clean_foreground(inp, (color_a, color_b), Color.BLACK)
        centers = _dent_centers(inp, cleaned_input, (color_a, color_b), Color.BLACK)
        output = _render_centers(cleaned_input, centers)
        produced = main(inp)
    except Exception:
        return None

    if len(centers) < n_total:
        return None

    visible_centers, extra_visible_cells, per_color = _visible_metrics(cleaned_input, centers)
    if visible_centers < VISIBLE_DENTS_TOTAL_MIN:
        return None
    if extra_visible_cells < VISIBLE_EXTRA_CELLS_MIN:
        return None
    if per_color.get(color_a, 0) < VISIBLE_DENTS_PER_COLOR_MIN:
        return None
    if per_color.get(color_b, 0) < VISIBLE_DENTS_PER_COLOR_MIN:
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
